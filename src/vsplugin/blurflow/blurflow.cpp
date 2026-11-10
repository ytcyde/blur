// blurflow: renders blur's interpolated + blended frames on the gpu.
//
// blur interpolates a clip to a very high framerate and then averages runs of those frames into
// each output frame. doing that with svp's SmoothFps means every interpolated frame is read back
// from the gpu just to be averaged on the cpu, and that readback is what the time goes on.
// this filter takes the motion vectors from svp1.Analyse, renders the interpolated frames itself
// and sums them on the gpu, so only the finished output frames are read back.

#include "VapourSynth4.h"
#include "chroma.h"
#include "kernel.h"
#include "opencl_min.h"

#include <algorithm>
#include <cstring>
#include <emmintrin.h>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef _WIN32
#	define NOMINMAX
#	define WIN32_LEAN_AND_MEAN
#	include <windows.h>
#else
#	include <dlfcn.h>
#endif

bool OpenCL::load() {
#ifdef _WIN32
	HMODULE lib = LoadLibraryA("OpenCL.dll");
	if (!lib)
		return false;
#	define X(ret, name, args)                                                                                         \
		name = reinterpret_cast<decltype(name)>(GetProcAddress(lib, #name));                                           \
		if (!name)                                                                                                     \
			return false;
#else
	void* lib = dlopen("libOpenCL.so.1", RTLD_NOW);
	if (!lib)
		return false;
#	define X(ret, name, args)                                                                                         \
		name = reinterpret_cast<decltype(name)>(dlsym(lib, #name));                                                    \
		if (!name)                                                                                                     \
			return false;
#endif
	BLURFLOW_CL_FUNCTIONS(X)
#undef X
	return true;
}

OpenCL& opencl() {
	static OpenCL cl;
	return cl;
}

namespace {
	void check(cl_int err, const char* what) {
		if (err != CL_SUCCESS)
			throw std::runtime_error(std::string(what) + " failed (opencl error " + std::to_string(err) + ")");
	}

	// svp1.Analyse stores the vectors of one frame pair in a one-row gray8 frame: a header of
	// 16 ints, then for each direction a count followed by one 8-byte record per block
	struct VectorLayout {
		int block_w = 0;
		int block_h = 0;
		int pel = 0;
		int blocks_x = 0;
		int blocks_y = 0;

		static constexpr int HEADER_SIZE = 64;
		static constexpr int RECORD_SIZE = 8;

		[[nodiscard]] size_t direction_size() const {
			return sizeof(int32_t) + (size_t)RECORD_SIZE * blocks_x * blocks_y;
		}

		bool operator==(const VectorLayout&) const = default;
	};

	VectorLayout parse_vector_header(const uint8_t* data, int size, int width, int height) {
		if (size < VectorLayout::HEADER_SIZE)
			throw std::runtime_error("vector frame is too small");

		int32_t header[16];
		memcpy(header, data, sizeof(header));

		VectorLayout layout{
			.block_w = header[3],
			.block_h = header[4],
			.pel = header[5],
			.blocks_x = header[12],
			.blocks_y = header[13],
		};

		bool valid = header[8] == width && header[9] == height && layout.block_w > 0 && layout.block_h > 0 &&
		             layout.pel > 0 && layout.blocks_x > 0 && layout.blocks_y > 0 &&
		             layout.blocks_x <= width && layout.blocks_y <= height &&
		             (size_t)size >= VectorLayout::HEADER_SIZE + 2 * layout.direction_size();
		if (!valid)
			throw std::runtime_error("unrecognised vector frame layout");

		return layout;
	}

	struct FrameSlot {
		int n = -1;
		uint64_t used = 0;
		cl_mem planes[3] = {};
		cl_mem linear_luma = nullptr;
	};

	struct VectorSlot {
		int n = -1;
		uint64_t used = 0;
		cl_mem texture = nullptr;
	};

	// the weights an output frame puts on the interpolated frames of one source interval:
	// weights[0] is the source frame itself, weights[k] the k-th frame interpolated towards the next one
	struct Interval {
		int n = 0;
		std::vector<double> weights;
	};

	// one run of the kernel: frames interpolated between two source frames, at `weights.size()` evenly
	// spaced positions. weights[0] is the first frame itself, tail the second
	struct Pass {
		int first = 0;
		int second = 0;
		int vectors = 0;
		std::vector<double> weights;
		double tail = 0;

		[[nodiscard]] bool interpolated() const {
			return std::any_of(weights.begin() + 1, weights.end(), [](double w) {
				return w != 0;
			});
		}

		[[nodiscard]] bool uses_second() const {
			return tail != 0 || interpolated();
		}
	};

	struct Plane {
		int width;
		int height;
		int base; // offset into the accumulation and output buffers
		int ratio; // subsampling relative to luma
	};

	struct Renderer {
		VSNode* clip = nullptr;
		VSNode* vectors = nullptr;
		VSVideoInfo vi = {};

		int source_frames = 0;
		int factor = 0; // interpolated frames per source frame
		int step = 0; // interpolated frames per output frame
		int64_t interpolated_frames = 0;
		std::vector<double> weights;
		double weight_sum = 0;

		// source frames that differ by less than this count as duplicates. negative = off
		double duplicate_threshold = -1;
		// don't interpolate between duplicates, hold the nearest one instead
		bool hold_static = false;
		// a duplicate followed by a new frame is a dropped frame: it's replaced by the frame halfway
		// to the new one, by stretching the interpolation towards the new frame over both intervals
		bool fill_drops = false;
		std::vector<int8_t> duplicate_cache; // whether frame n + 1 duplicates frame n. -1 = not yet known

		Plane planes[3] = {};
		size_t buffer_size = 0;

		cl_device_id device = nullptr;
		cl_context context = nullptr;
		cl_command_queue queue = nullptr;
		cl_program program = nullptr;
		cl_kernel to_linear = nullptr;
		cl_kernel interval = nullptr;
		cl_mem accumulator = nullptr;
		cl_mem output = nullptr;
		cl_mem weights_buffer = nullptr;
		size_t weights_capacity = 0;
		cl_mem no_vectors = nullptr;

		std::vector<FrameSlot> frame_slots;
		std::vector<VectorSlot> vector_slots;
		VectorLayout layout;
		bool have_layout = false;
		uint64_t tick = 0;

		std::vector<uint8_t> scratch;

		~Renderer() {
			auto& cl = opencl();
			if (!cl.clReleaseMemObject)
				return;

			if (queue)
				cl.clFinish(queue);

			for (auto& slot : frame_slots) {
				for (cl_mem plane : slot.planes)
					release(plane);
				release(slot.linear_luma);
			}
			for (auto& slot : vector_slots)
				release(slot.texture);
			release(accumulator);
			release(output);
			release(weights_buffer);
			release(no_vectors);

			if (to_linear)
				cl.clReleaseKernel(to_linear);
			if (interval)
				cl.clReleaseKernel(interval);
			if (program)
				cl.clReleaseProgram(program);
			if (queue)
				cl.clReleaseCommandQueue(queue);
			if (context)
				cl.clReleaseContext(context);
		}

		static void release(cl_mem mem) {
			if (mem)
				opencl().clReleaseMemObject(mem);
		}

		[[nodiscard]] int window_frames() const {
			return ((int)weights.size() - 1) / factor + 2;
		}

		void init_gpu(int device_index) {
			auto& cl = opencl();
			if (!cl.load())
				throw std::runtime_error("opencl is not available");

			cl_platform_id platforms[16];
			cl_uint num_platforms = 0;
			cl.clGetPlatformIDs(16, platforms, &num_platforms);

			std::vector<cl_device_id> devices;
			for (cl_uint i = 0; i < num_platforms; i++) {
				cl_device_id found[16];
				cl_uint num_found = 0;
				if (cl.clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_GPU, 16, found, &num_found) == CL_SUCCESS)
					devices.insert(devices.end(), found, found + num_found);
			}

			if (device_index < 0 || device_index >= (int)devices.size())
				throw std::runtime_error("no opencl gpu found");
			device = devices[device_index];

			cl_int err;
			context = cl.clCreateContext(nullptr, 1, &device, nullptr, nullptr, &err);
			check(err, "clCreateContext");
			queue = cl.clCreateCommandQueue(context, device, 0, &err);
			check(err, "clCreateCommandQueue");

			const char* source = BLURFLOW_KERNEL_SOURCE;
			program = cl.clCreateProgramWithSource(context, 1, &source, nullptr, &err);
			check(err, "clCreateProgramWithSource");
			if (cl.clBuildProgram(program, 1, &device, "", nullptr, nullptr) != CL_SUCCESS) {
				std::string log(16384, '\0');
				size_t log_size = 0;
				cl.clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log.size(), log.data(), &log_size);
				log.resize(std::min(log_size, log.size()));
				throw std::runtime_error("failed to build the opencl kernel: " + log);
			}

			to_linear = cl.clCreateKernel(program, "to_linear", &err);
			check(err, "clCreateKernel");
			interval = cl.clCreateKernel(program, "interval", &err);
			check(err, "clCreateKernel");

			accumulator = cl.clCreateBuffer(context, CL_MEM_READ_WRITE, buffer_size * sizeof(float), nullptr, &err);
			check(err, "clCreateBuffer");
			output = cl.clCreateBuffer(context, CL_MEM_READ_WRITE, buffer_size, nullptr, &err);
			check(err, "clCreateBuffer");

			// room for the frames of the output frame being rendered plus a few of its neighbours,
			// as consecutive output frames share source frames
			const cl_image_format plane_format = { CL_R, CL_UNORM_INT8 };
			const cl_image_format linear_format = { CL_R, CL_FLOAT };
			frame_slots.resize(window_frames() + 4);
			for (auto& slot : frame_slots) {
				for (int p = 0; p < 3; p++) {
					slot.planes[p] = cl.clCreateImage2D(
						context, CL_MEM_READ_ONLY, &plane_format, planes[p].width, planes[p].height, 0, nullptr, &err
					);
					check(err, "clCreateImage2D");
				}
				slot.linear_luma = cl.clCreateImage2D(
					context, CL_MEM_READ_WRITE, &linear_format, planes[0].width, planes[0].height, 0, nullptr, &err
				);
				check(err, "clCreateImage2D");
			}

			no_vectors = create_vector_texture(1, 1);
		}

		cl_mem create_vector_texture(int w, int h) const {
			const cl_image_format format = { CL_RGBA, CL_UNORM_INT16 };
			cl_int err;
			cl_mem texture = opencl().clCreateImage2D(context, CL_MEM_READ_ONLY, &format, w, h, 0, nullptr, &err);
			check(err, "clCreateImage2D");
			return texture;
		}

		// which source intervals output frame n draws on, and how heavily
		[[nodiscard]] std::vector<Interval> window(int n) const {
			std::vector<Interval> intervals;
			int radius = (int)weights.size() / 2;

			for (int i = 0; i < (int)weights.size(); i++) {
				if (weights[i] == 0)
					continue;

				int64_t frame = std::clamp<int64_t>((int64_t)n * step + i - radius, 0, interpolated_frames - 1);
				int source = (int)(frame / factor);

				if (intervals.empty() || intervals.back().n != source)
					intervals.push_back({ .n = source, .weights = std::vector<double>(factor) });

				intervals.back().weights[frame % factor] += weights[i];
			}

			return intervals;
		}

		// the source frames needed to work out how an interval is rendered
		[[nodiscard]] std::pair<int, int> frames_to_inspect(const Interval& interval) const {
			int last = source_frames - 1;
			if (fill_drops)
				return { std::max(interval.n - 1, 0), std::min(interval.n + 2, last) };
			return { interval.n, std::min(interval.n + 1, last) };
		}

		// whether frame n + 1 is a duplicate of frame n
		bool is_duplicate(int n, VSFrameContext* frame_ctx, const VSAPI* vsapi) {
			if (n < 0 || n >= source_frames - 1)
				return true;

			if (duplicate_cache[n] < 0) {
				const VSFrame* a = vsapi->getFrameFilter(n, clip, frame_ctx);
				const VSFrame* b = vsapi->getFrameFilter(n + 1, clip, frame_ctx);

				// same measure as PlaneStatsDiff from std.PlaneStats on the luma plane
				const uint8_t* pa = vsapi->getReadPtr(a, 0);
				const uint8_t* pb = vsapi->getReadPtr(b, 0);
				ptrdiff_t stride_a = vsapi->getStride(a, 0);
				ptrdiff_t stride_b = vsapi->getStride(b, 0);
				int w = planes[0].width;
				int h = planes[0].height;

				uint64_t total = 0;
				for (int y = 0; y < h; y++) {
					__m128i sum = _mm_setzero_si128();
					int x = 0;
					for (; x + 16 <= w; x += 16) {
						__m128i va = _mm_loadu_si128((const __m128i*)(pa + x));
						__m128i vb = _mm_loadu_si128((const __m128i*)(pb + x));
						sum = _mm_add_epi64(sum, _mm_sad_epu8(va, vb));
					}
					total += (uint64_t)_mm_cvtsi128_si64(sum) + (uint64_t)_mm_cvtsi128_si64(_mm_srli_si128(sum, 8));
					for (; x < w; x++)
						total += pa[x] > pb[x] ? pa[x] - pb[x] : pb[x] - pa[x];
					pa += stride_a;
					pb += stride_b;
				}

				vsapi->freeFrame(a);
				vsapi->freeFrame(b);

				double diff = (double)total / ((double)w * h) / 255.0;
				duplicate_cache[n] = diff < duplicate_threshold ? 1 : 0;
			}

			return duplicate_cache[n] == 1;
		}

		// a dropped frame: it repeats the frame before it while the frame after it is new
		bool is_dropped(int n, VSFrameContext* frame_ctx, const VSAPI* vsapi) {
			return fill_drops && n < source_frames - 1 && is_duplicate(n - 1, frame_ctx, vsapi) &&
				!is_duplicate(n, frame_ctx, vsapi);
		}

		// how an interval's frames are rendered. needs the frames from frames_to_inspect
		Pass resolve(const Interval& interval, VSFrameContext* frame_ctx, const VSAPI* vsapi) {
			int n = interval.n;
			bool has_next = n < source_frames - 1;

			if (is_dropped(n, frame_ctx, vsapi)) {
				// starts at the frame halfway to the next one: the second half of the stretched interpolation
				Pass pass{ .first = n, .second = n + 1, .vectors = n, .weights = std::vector<double>(factor * 2) };
				std::copy(interval.weights.begin(), interval.weights.end(), pass.weights.begin() + factor);
				return pass;
			}

			if (has_next && is_dropped(n + 1, frame_ctx, vsapi)) {
				// ends at the frame halfway between the next two. the next frame repeats this one, so this
				// is the first half of the interpolation from the next frame to the one after it
				Pass pass{
					.first = n + 1, .second = n + 2, .vectors = n + 1, .weights = std::vector<double>(factor * 2)
				};
				std::copy(interval.weights.begin(), interval.weights.end(), pass.weights.begin());
				return pass;
			}

			Pass pass{ .first = n, .second = n + 1, .vectors = n, .weights = interval.weights };

			if (has_next && hold_static && pass.interpolated() && is_duplicate(n, frame_ctx, vsapi)) {
				// nothing moves between the two frames, so hold the nearest one instead of interpolating
				int first_half = (factor + 1) / 2;
				double head = std::accumulate(pass.weights.begin(), pass.weights.begin() + first_half, 0.0);
				pass.tail = std::accumulate(pass.weights.begin() + first_half, pass.weights.end(), 0.0);
				pass.weights = { head };
			}

			return pass;
		}

		// turns the window into gpu passes
		std::vector<Pass> plan(int n, VSFrameContext* frame_ctx, const VSAPI* vsapi) {
			std::vector<Pass> passes;

			for (const auto& interval : window(n)) {
				Pass pass = resolve(interval, frame_ctx, vsapi);

				if (!passes.empty()) {
					Pass& last = passes.back();

					if (last.first == pass.first && last.weights.size() == pass.weights.size()) {
						// two halves of the same interpolation
						for (size_t k = 0; k < pass.weights.size(); k++)
							last.weights[k] += pass.weights[k];
						last.tail += pass.tail;
						continue;
					}

					// the previous pass already has this pass' first frame bound as its second
					if (last.second == pass.first && last.uses_second()) {
						last.tail += pass.weights[0];
						pass.weights[0] = 0;
					}
				}

				if (pass.uses_second() || pass.weights[0] != 0)
					passes.push_back(std::move(pass));
			}

			return passes;
		}

		template<typename Slot>
		static Slot& claim(std::vector<Slot>& slots, int n, uint64_t tick, bool& hit) {
			Slot* oldest = nullptr;
			for (auto& slot : slots) {
				if (slot.n == n) {
					slot.used = tick;
					hit = true;
					return slot;
				}
				if (slot.used != tick && (!oldest || slot.used < oldest->used))
					oldest = &slot;
			}

			if (!oldest)
				throw std::runtime_error("ran out of gpu slots");

			hit = false;
			oldest->n = n;
			oldest->used = tick;
			return *oldest;
		}

		FrameSlot& frame_slot(
			int n, std::vector<const VSFrame*>& held, VSFrameContext* frame_ctx, const VSAPI* vsapi
		) {
			bool hit;
			FrameSlot& slot = claim(frame_slots, n, tick, hit);
			if (hit)
				return slot;

			auto& cl = opencl();
			const VSFrame* frame = vsapi->getFrameFilter(n, clip, frame_ctx);
			held.push_back(frame); // the uploads read from it until the queue has run

			for (int p = 0; p < 3; p++) {
				const size_t origin[3] = { 0, 0, 0 };
				const size_t region[3] = { (size_t)planes[p].width, (size_t)planes[p].height, 1 };
				check(
					cl.clEnqueueWriteImage(
						queue,
						slot.planes[p],
						CL_FALSE,
						origin,
						region,
						(size_t)vsapi->getStride(frame, p),
						0,
						vsapi->getReadPtr(frame, p),
						0,
						nullptr,
						nullptr
					),
					"clEnqueueWriteImage"
				);
			}

			const size_t size[2] = { (size_t)planes[0].width, (size_t)planes[0].height };
			check(cl.clSetKernelArg(to_linear, 0, sizeof(cl_mem), &slot.planes[0]), "clSetKernelArg");
			check(cl.clSetKernelArg(to_linear, 1, sizeof(cl_mem), &slot.linear_luma), "clSetKernelArg");
			check(
				cl.clEnqueueNDRangeKernel(queue, to_linear, 2, nullptr, size, nullptr, 0, nullptr, nullptr),
				"clEnqueueNDRangeKernel"
			);

			return slot;
		}

		VectorSlot& vector_slot(
			int n, std::vector<std::vector<uint16_t>>& uploads, VSFrameContext* frame_ctx, const VSAPI* vsapi
		) {
			const VSFrame* frame = vsapi->getFrameFilter(n, vectors, frame_ctx);
			std::unique_ptr<const VSFrame, void (*)(const VSFrame*)> frame_guard(frame, vsapi->freeFrame);

			const uint8_t* data = vsapi->getReadPtr(frame, 0);
			VectorLayout frame_layout = parse_vector_header(
				data, vsapi->getFrameWidth(frame, 0), planes[0].width, planes[0].height
			);

			if (!have_layout) {
				layout = frame_layout;
				have_layout = true;

				vector_slots.resize(window_frames() + 3);
				for (auto& slot : vector_slots)
					slot.texture = create_vector_texture(layout.blocks_x, layout.blocks_y);
			}
			else if (frame_layout != layout) {
				throw std::runtime_error("vector frame layout changed");
			}

			bool hit;
			VectorSlot& slot = claim(vector_slots, n, tick, hit);
			if (hit)
				return slot;

			// a texel holds both vectors of a block: (x, y) of the first in (r, a), of the second in (b, g).
			// vectors are stored offset by 1024 so that they fit an unsigned texture. past the last block
			// the texture clamps, which is how svp extends the block grid to the frame's edge
			constexpr uint16_t ZERO = 1024;
			auto& texels = uploads.emplace_back((size_t)layout.blocks_x * layout.blocks_y * 4);

			const uint8_t* first = data + VectorLayout::HEADER_SIZE + sizeof(int32_t);
			const uint8_t* second = first + layout.direction_size();

			for (int y = 0; y < layout.blocks_y; y++) {
				uint16_t* out = &texels[(size_t)y * layout.blocks_x * 4];
				for (int x = 0; x < layout.blocks_x; x++) {
					size_t at = ((size_t)y * layout.blocks_x + x) * VectorLayout::RECORD_SIZE;
					int16_t a[2];
					int16_t b[2];
					memcpy(a, first + at, sizeof(a));
					memcpy(b, second + at, sizeof(b));

					out[x * 4 + 0] = (uint16_t)(a[1] + ZERO);
					out[x * 4 + 1] = (uint16_t)(b[0] + ZERO);
					out[x * 4 + 2] = (uint16_t)(b[1] + ZERO);
					out[x * 4 + 3] = (uint16_t)(a[0] + ZERO);
				}
			}

			const size_t origin[3] = { 0, 0, 0 };
			const size_t region[3] = { (size_t)layout.blocks_x, (size_t)layout.blocks_y, 1 };
			check(
				opencl().clEnqueueWriteImage(
					queue, slot.texture, CL_FALSE, origin, region, 0, 0, texels.data(), 0, nullptr, nullptr
				),
				"clEnqueueWriteImage"
			);

			return slot;
		}

		const VSFrame* render(int n, VSFrameContext* frame_ctx, VSCore* core, const VSAPI* vsapi) {
			auto& cl = opencl();

			std::vector<Pass> passes = plan(n, frame_ctx, vsapi);
			if (passes.empty())
				throw std::runtime_error("nothing to render");

			// everything the queued commands read from has to stay alive until the final read returns
			std::vector<const VSFrame*> held;
			std::vector<std::vector<uint16_t>> vector_uploads;
			std::vector<float> pass_weights;
			std::vector<size_t> weights_at;

			struct Guard {
				std::vector<const VSFrame*>& held;
				cl_command_queue queue;
				const VSAPI* vsapi;
				bool finished = false;

				~Guard() {
					if (!finished)
						opencl().clFinish(queue);
					for (const VSFrame* frame : held)
						vsapi->freeFrame(frame);
				}
			} guard{ held, queue, vsapi };

			tick++;

			for (const Pass& pass : passes) {
				weights_at.push_back(pass_weights.size());
				for (double weight : pass.weights)
					pass_weights.push_back((float)weight);
			}

			size_t weights_size = pass_weights.size() * sizeof(float);
			if (weights_size > weights_capacity) {
				release(weights_buffer);
				cl_int err;
				weights_buffer = cl.clCreateBuffer(context, CL_MEM_READ_ONLY, weights_size, nullptr, &err);
				check(err, "clCreateBuffer");
				weights_capacity = weights_size;
			}
			check(
				cl.clEnqueueWriteBuffer(
					queue, weights_buffer, CL_FALSE, 0, weights_size, pass_weights.data(), 0, nullptr, nullptr
				),
				"clEnqueueWriteBuffer"
			);

			for (size_t i = 0; i < passes.size(); i++) {
				const Pass& pass = passes[i];

				bool interpolated = pass.interpolated();

				FrameSlot& first = frame_slot(pass.first, held, frame_ctx, vsapi);
				FrameSlot& second = pass.uses_second() ? frame_slot(pass.second, held, frame_ctx, vsapi) : first;
				cl_mem vector_texture =
					interpolated ? vector_slot(pass.vectors, vector_uploads, frame_ctx, vsapi).texture : no_vectors;

				for (int p = 0; p < 3; p++) {
					const Plane& plane = planes[p];
					bool luma = p == 0;

					cl_int base = plane.base;
					cl_int weights_offset = (cl_int)weights_at[i];
					cl_int steps = (cl_int)pass.weights.size();
					float tail = (float)pass.tail;
					float divisor = (float)weight_sum;
					cl_int ratio = plane.ratio;
					float block_w = have_layout ? (float)layout.block_w : 1.0f;
					float block_h = have_layout ? (float)layout.block_h : 1.0f;
					float vector_div = (float)(plane.ratio * (have_layout ? layout.pel : 1)) * 256.0f;
					cl_int flags = (i == 0 ? 1 : 0) | (i == passes.size() - 1 ? 2 : 0) | (luma ? 4 : 0) |
					               (interpolated ? 8 : 0);

					cl_uint arg = 0;
					auto set = [&](size_t size, const void* value) {
						check(cl.clSetKernelArg(interval, arg++, size, value), "clSetKernelArg");
					};

					set(sizeof(cl_mem), &accumulator);
					set(sizeof(cl_mem), &output);
					set(sizeof(base), &base);
					set(sizeof(cl_mem), &first.planes[p]);
					set(sizeof(cl_mem), &second.planes[p]);
					set(sizeof(cl_mem), luma ? &first.linear_luma : &first.planes[p]);
					set(sizeof(cl_mem), luma ? &second.linear_luma : &second.planes[p]);
					set(sizeof(cl_mem), &vector_texture);
					set(sizeof(cl_mem), &weights_buffer);
					set(sizeof(weights_offset), &weights_offset);
					set(sizeof(steps), &steps);
					set(sizeof(tail), &tail);
					set(sizeof(divisor), &divisor);
					set(sizeof(ratio), &ratio);
					set(sizeof(block_w), &block_w);
					set(sizeof(block_h), &block_h);
					set(sizeof(vector_div), &vector_div);
					set(sizeof(flags), &flags);

					const size_t size[2] = { (size_t)plane.width, (size_t)plane.height };
					check(
						cl.clEnqueueNDRangeKernel(queue, interval, 2, nullptr, size, nullptr, 0, nullptr, nullptr),
						"clEnqueueNDRangeKernel"
					);
				}
			}

			const VSFrame* prop_src = vsapi->getFrameFilter(passes[0].first, clip, frame_ctx);
			VSFrame* dst = vsapi->newVideoFrame(&vi.format, vi.width, vi.height, prop_src, core);
			vsapi->freeFrame(prop_src);

			VSMap* props = vsapi->getFramePropertiesRW(dst);
			vsapi->mapSetInt(props, "_DurationNum", vi.fpsDen, maReplace);
			vsapi->mapSetInt(props, "_DurationDen", vi.fpsNum, maReplace);

			// read straight into the frame where its rows are packed, which they normally are
			size_t scratch_needed = 0;
			for (int p = 0; p < 3; p++) {
				if (vsapi->getStride(dst, p) != planes[p].width)
					scratch_needed = buffer_size;
			}
			scratch.resize(scratch_needed);

			try {
				for (int p = 0; p < 3; p++) {
					const Plane& plane = planes[p];
					bool packed = vsapi->getStride(dst, p) == plane.width;
					uint8_t* target = packed ? vsapi->getWritePtr(dst, p) : scratch.data() + plane.base;

					check(
						cl.clEnqueueReadBuffer(
							queue,
							output,
							p == 2 ? CL_TRUE : CL_FALSE,
							(size_t)plane.base,
							(size_t)plane.width * plane.height,
							target,
							0,
							nullptr,
							nullptr
						),
						"clEnqueueReadBuffer"
					);
				}
				guard.finished = true;
			}
			catch (...) {
				cl.clFinish(queue);
				guard.finished = true;
				vsapi->freeFrame(dst);
				throw;
			}

			for (int p = 0; p < 3; p++) {
				const Plane& plane = planes[p];
				ptrdiff_t stride = vsapi->getStride(dst, p);
				if (stride == plane.width)
					continue;

				uint8_t* out = vsapi->getWritePtr(dst, p);
				for (int y = 0; y < plane.height; y++)
					memcpy(out + y * stride, scratch.data() + plane.base + (size_t)y * plane.width, plane.width);
			}

			return dst;
		}
	};

	// frameData[0] holds which stage of the request we're at
	const intptr_t STAGE_INSPECT = 1;
	const intptr_t STAGE_RENDER = 2;

	const VSFrame* VS_CC get_frame(
		int n,
		int activation_reason,
		void* instance_data,
		void** frame_data,
		VSFrameContext* frame_ctx,
		VSCore* core,
		const VSAPI* vsapi
	) {
		auto* r = static_cast<Renderer*>(instance_data);

		try {
			if (activation_reason == arInitial) {
				// analysing motion is expensive and how an interval is rendered can depend on its frames,
				// so in that case the vectors are only requested once the frames have been looked at
				bool inspect = r->hold_static || r->fill_drops;

				for (const auto& interval : r->window(n)) {
					if (inspect) {
						auto [from, to] = r->frames_to_inspect(interval);
						for (int i = from; i <= to; i++)
							vsapi->requestFrameFilter(i, r->clip, frame_ctx);
						continue;
					}

					Pass pass = r->resolve(interval, frame_ctx, vsapi);
					vsapi->requestFrameFilter(pass.first, r->clip, frame_ctx);
					if (pass.interpolated()) {
						vsapi->requestFrameFilter(pass.second, r->clip, frame_ctx);
						vsapi->requestFrameFilter(pass.vectors, r->vectors, frame_ctx);
					}
				}

				frame_data[0] = (void*)(inspect ? STAGE_INSPECT : STAGE_RENDER);
				return nullptr;
			}

			if (activation_reason != arAllFramesReady)
				return nullptr;

			if ((intptr_t)frame_data[0] == STAGE_INSPECT) {
				frame_data[0] = (void*)STAGE_RENDER;

				bool requested = false;
				for (const Pass& pass : r->plan(n, frame_ctx, vsapi)) {
					if (pass.interpolated()) {
						vsapi->requestFrameFilter(pass.vectors, r->vectors, frame_ctx);
						requested = true;
					}
				}

				if (requested)
					return nullptr;
			}

			return r->render(n, frame_ctx, core, vsapi);
		}
		catch (const std::exception& e) {
			vsapi->setFilterError(("blurflow: " + std::string(e.what())).c_str(), frame_ctx);
			return nullptr;
		}
	}

	void VS_CC free_renderer(void* instance_data, VSCore* core, const VSAPI* vsapi) {
		auto* r = static_cast<Renderer*>(instance_data);
		vsapi->freeNode(r->clip);
		vsapi->freeNode(r->vectors);
		delete r;
	}

	void VS_CC create(const VSMap* in, VSMap* out, void* user_data, VSCore* core, const VSAPI* vsapi) {
		auto r = std::make_unique<Renderer>();
		r->clip = vsapi->mapGetNode(in, "clip", 0, nullptr);
		r->vectors = vsapi->mapGetNode(in, "vectors", 0, nullptr);

		auto fail = [&](const std::string& message) {
			vsapi->mapSetError(out, ("blurflow: " + message).c_str());
			vsapi->freeNode(r->clip);
			vsapi->freeNode(r->vectors);
		};

		const VSVideoInfo* clip_vi = vsapi->getVideoInfo(r->clip);
		const VSVideoInfo* vectors_vi = vsapi->getVideoInfo(r->vectors);

		bool yuv420p8 = clip_vi->format.colorFamily == cfYUV && clip_vi->format.sampleType == stInteger &&
		                clip_vi->format.bitsPerSample == 8 && clip_vi->format.subSamplingW == 1 &&
		                clip_vi->format.subSamplingH == 1;
		if (!yuv420p8 || clip_vi->width <= 0 || clip_vi->height <= 0)
			return fail("clip must be constant format yuv420p8");

		if (vectors_vi->format.colorFamily != cfGray || vectors_vi->format.bitsPerSample != 8 ||
		    vectors_vi->height != 1 || vectors_vi->numFrames != clip_vi->numFrames)
			return fail("vectors must come from svp1.Analyse on the same clip");

		int err = 0;
		r->factor = (int)vsapi->mapGetInt(in, "factor", 0, nullptr);
		r->step = (int)vsapi->mapGetInt(in, "step", 0, nullptr);
		if (r->factor < 1 || r->step < 1)
			return fail("factor and step must be at least 1");

		int num_weights = vsapi->mapNumElements(in, "weights");
		if (num_weights % 2 == 0)
			return fail("an odd number of weights is required");
		for (int i = 0; i < num_weights; i++)
			r->weights.push_back(vsapi->mapGetFloat(in, "weights", i, nullptr));
		r->weight_sum = std::accumulate(r->weights.begin(), r->weights.end(), 0.0);
		if (r->weight_sum == 0)
			return fail("weights must not sum to zero");

		double duplicate_threshold = vsapi->mapGetFloat(in, "threshold", 0, &err);
		if (!err)
			r->duplicate_threshold = duplicate_threshold;
		r->hold_static = vsapi->mapGetInt(in, "hold", 0, &err) != 0 && r->duplicate_threshold >= 0;
		r->fill_drops = vsapi->mapGetInt(in, "fill", 0, &err) != 0 && r->duplicate_threshold >= 0;

		int device_index = (int)vsapi->mapGetInt(in, "device", 0, &err);
		if (err)
			device_index = 0;

		// the interpolated clip has `factor` frames per source frame and ends on the last source frame
		r->source_frames = clip_vi->numFrames;
		r->interpolated_frames = (int64_t)(r->source_frames - 1) * r->factor + 1;
		r->duplicate_cache.assign(r->source_frames, -1);

		r->vi = *clip_vi;
		r->vi.numFrames = (int)(r->interpolated_frames / r->step);
		if (r->vi.numFrames < 1)
			return fail("clip is too short");

		int64_t fps_num = clip_vi->fpsNum * r->factor;
		int64_t fps_den = clip_vi->fpsDen * r->step;
		int64_t divisor = std::gcd(fps_num, fps_den);
		r->vi.fpsNum = divisor ? fps_num / divisor : fps_num;
		r->vi.fpsDen = divisor ? fps_den / divisor : fps_den;

		int w = clip_vi->width;
		int h = clip_vi->height;
		r->planes[0] = { .width = w, .height = h, .base = 0, .ratio = 1 };
		r->planes[1] = { .width = w / 2, .height = h / 2, .base = w * h, .ratio = 2 };
		r->planes[2] = { .width = w / 2, .height = h / 2, .base = w * h + (w / 2) * (h / 2), .ratio = 2 };
		r->buffer_size = (size_t)w * h + 2 * (size_t)(w / 2) * (h / 2);

		try {
			r->init_gpu(device_index);
		}
		catch (const std::exception& e) {
			return fail(e.what());
		}

		const VSFilterDependency deps[] = { { r->clip, rpGeneral }, { r->vectors, rpGeneral } };
		const VSVideoInfo vi = r->vi;
		// rendering is serial on the gpu, but the frames it needs can be prepared in parallel
		vsapi->createVideoFilter(
			out, "Render", &vi, get_frame, free_renderer, fmParallelRequests, deps, 2, r.release(), core
		);
	}
}

VS_EXTERNAL_API(void) VapourSynthPluginInit2(VSPlugin* plugin, const VSPLUGINAPI* vspapi) {
	vspapi->configPlugin(
		"com.blur.flow", "blurflow", "blur's gpu frame renderer", VS_MAKE_VERSION(1, 0), VAPOURSYNTH_API_VERSION, 0, plugin
	);
	vspapi->registerFunction(
		"Render",
		"clip:vnode;vectors:vnode;factor:int;step:int;weights:float[];threshold:float:opt;hold:int:opt;fill:int:opt;device:int:opt;",
		"clip:vnode;",
		create,
		nullptr,
		plugin
	);
	register_chroma_filters(plugin, vspapi);
}
