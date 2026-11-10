// a source filter that decodes on the gpu, with nvidia's decoder.
//
// decoding is what holds the rest of the pipeline up once rendering is fast: a regular source
// filter decodes on the cpu, which the motion analysis is already using all of, and hands over one
// frame at a time. here the video is decoded by the gpu's decoder on a thread of its own, which
// runs a little ahead of the frames being asked for.
//
// the compressed video comes from a process (ffmpeg, copying the stream out of its container) and
// the caller says which of its frames each output frame is, so that the clip is the same frame for
// frame as the fallback clip: a regular source filter on the same file. every frame that can't be
// had from the decoder comes from the fallback instead, and so does everything once anything
// about the decoding isn't as expected.

#include "source.h"

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <emmintrin.h>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#ifdef _WIN32
#	define NOMINMAX
#	define WIN32_LEAN_AND_MEAN
#	include <windows.h>

#	include "ffnvcodec/dynlink_cuda.h"
#	include "ffnvcodec/dynlink_nvcuvid.h"

namespace {
	// how far the decoder runs ahead of the newest frame asked for
	constexpr int AHEAD_FRAMES = 8;
	// how many decoded frames are kept. frames are asked for out of order when several are being
	// worked on at once, so this has to cover the ones still to be asked for as well
	constexpr int HELD_FRAMES = 48;
	// lets the decoder work on a few pictures at once
	constexpr int DISPLAY_DELAY = 4;
	constexpr DWORD PIPE_BUFFER_SIZE = 4 << 20;

	// both libraries come with nvidia's driver, so there is nothing to link against or ship
#	define BLURFLOW_CUDA_FUNCTIONS(X)                                                                                 \
		X(cuInit)                                                                                                      \
		X(cuDeviceGet)                                                                                                 \
		X(cuCtxCreate_v2)                                                                                              \
		X(cuCtxDestroy_v2)                                                                                             \
		X(cuMemHostAlloc)                                                                                              \
		X(cuMemFreeHost)                                                                                               \
		X(cuMemcpy2D_v2)

#	define BLURFLOW_CUVID_FUNCTIONS(X)                                                                                \
		X(cuvidGetDecoderCaps)                                                                                         \
		X(cuvidCreateDecoder)                                                                                          \
		X(cuvidDestroyDecoder)                                                                                         \
		X(cuvidDecodePicture)                                                                                          \
		X(cuvidMapVideoFrame64)                                                                                        \
		X(cuvidUnmapVideoFrame64)                                                                                      \
		X(cuvidCreateVideoParser)                                                                                      \
		X(cuvidParseVideoData)                                                                                         \
		X(cuvidDestroyVideoParser)

	struct Nvidia {
#	define X(name) t##name* name = nullptr;
		BLURFLOW_CUDA_FUNCTIONS(X)
		BLURFLOW_CUVID_FUNCTIONS(X)
#	undef X

		bool loaded = false;

		Nvidia() {
			HMODULE cuda = LoadLibraryA("nvcuda.dll");
			HMODULE cuvid = LoadLibraryA("nvcuvid.dll");
			if (!cuda || !cuvid)
				return;

#	define X(name)                                                                                                    \
		name = reinterpret_cast<decltype(name)>(GetProcAddress(cuda, #name));                                          \
		if (!name)                                                                                                     \
			return;
			BLURFLOW_CUDA_FUNCTIONS(X)
#	undef X
#	define X(name)                                                                                                    \
		name = reinterpret_cast<decltype(name)>(GetProcAddress(cuvid, #name));                                         \
		if (!name)                                                                                                     \
			return;
			BLURFLOW_CUVID_FUNCTIONS(X)
#	undef X

			loaded = true;
		}
	};

	Nvidia& nvidia() {
		static Nvidia nv;
		return nv;
	}

	std::wstring widen(const char* text, int length) {
		int size = MultiByteToWideChar(CP_UTF8, 0, text, length, nullptr, 0);
		std::wstring wide(size, L'\0');
		MultiByteToWideChar(CP_UTF8, 0, text, length, wide.data(), size);
		return wide;
	}

	// quotes an argument the way CommandLineToArgvW (and the c runtime) will undo
	void append_argument(std::wstring& command_line, const std::wstring& argument) {
		if (!command_line.empty())
			command_line += L' ';

		command_line += L'"';
		size_t backslashes = 0;
		for (wchar_t c : argument) {
			if (c == L'\\') {
				backslashes++;
			}
			else {
				if (c == L'"')
					command_line.append(backslashes + 1, L'\\');
				backslashes = 0;
			}
			command_line += c;
		}
		command_line.append(backslashes, L'\\');
		command_line += L'"';
	}

	// a process whose output is read through a pipe
	struct Process {
		HANDLE process = nullptr;
		HANDLE job = nullptr;
		HANDLE output = nullptr;

		~Process() {
			terminate();
			if (output)
				CloseHandle(output);
			if (process)
				CloseHandle(process);
			if (job)
				CloseHandle(job); // takes the process down with it, should it have outlived us
		}

		bool start(std::wstring command_line) {
			SECURITY_ATTRIBUTES inheritable = { sizeof(SECURITY_ATTRIBUTES), nullptr, TRUE };

			HANDLE read_end = nullptr;
			HANDLE write_end = nullptr;
			if (!CreatePipe(&read_end, &write_end, &inheritable, PIPE_BUFFER_SIZE))
				return false;
			SetHandleInformation(read_end, HANDLE_FLAG_INHERIT, 0);
			output = read_end;

			HANDLE nul = CreateFileW(
				L"NUL",
				GENERIC_READ | GENERIC_WRITE,
				FILE_SHARE_READ | FILE_SHARE_WRITE,
				&inheritable,
				OPEN_EXISTING,
				0,
				nullptr
			);

			STARTUPINFOW startup = {};
			startup.cb = sizeof(startup);
			startup.dwFlags = STARTF_USESTDHANDLES;
			startup.hStdInput = nul;
			startup.hStdOutput = write_end;
			startup.hStdError = nul;

			PROCESS_INFORMATION info = {};
			BOOL created = CreateProcessW(
				nullptr,
				command_line.data(),
				nullptr,
				nullptr,
				TRUE,
				CREATE_NO_WINDOW | CREATE_SUSPENDED,
				nullptr,
				nullptr,
				&startup,
				&info
			);

			CloseHandle(write_end);
			if (nul != INVALID_HANDLE_VALUE)
				CloseHandle(nul);

			if (!created)
				return false;

			process = info.hProcess;

			job = CreateJobObjectW(nullptr, nullptr);
			if (job) {
				JOBOBJECT_EXTENDED_LIMIT_INFORMATION limits = {};
				limits.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
				SetInformationJobObject(job, JobObjectExtendedLimitInformation, &limits, sizeof(limits));
				AssignProcessToJobObject(job, process);
			}

			ResumeThread(info.hThread);
			CloseHandle(info.hThread);
			return true;
		}

		// ends anything waiting in read() as well
		void terminate() const {
			if (process)
				TerminateProcess(process, 0);
		}

		// 0 once there is nothing more
		size_t read(uint8_t* data, size_t size) const {
			DWORD got = 0;
			if (!ReadFile(output, data, (DWORD)size, &got, nullptr))
				return 0;
			return got;
		}
	};

	// cuts an annex b stream into access units: the data of one picture each, as the container
	// had them in its packets
	struct AccessUnits {
		bool hevc = false;
		std::vector<uint8_t> data;
		size_t start = 0; // of the unit being put together
		size_t scan = 0; // how far the data has been looked through
		bool has_picture = false; // the unit being put together has (part of) its picture already

		static constexpr size_t NOT_FOUND = ~(size_t)0;

		void push(const uint8_t* bytes, size_t size) {
			data.erase(data.begin(), data.begin() + start);
			scan -= start;
			start = 0;
			data.insert(data.end(), bytes, bytes + size);
		}

		// the next 00 00 01
		[[nodiscard]] size_t find_start_code(size_t from) const {
			size_t i = from + 2;
			while (i < data.size()) {
				const void* found = memchr(&data[i], 1, data.size() - i);
				if (!found)
					break;
				i = static_cast<const uint8_t*>(found) - data.data();
				if (data[i - 1] == 0 && data[i - 2] == 0)
					return i - 2;
				i++;
			}
			return NOT_FOUND;
		}

		// whether the nal unit with this header is the first of an access unit
		bool starts_unit(const uint8_t* nal) {
			bool slice;
			bool first_slice;
			bool leads_unit; // a kind that can only come before a unit's picture

			if (hevc) {
				int type = (nal[0] >> 1) & 0x3f;
				slice = type < 32;
				first_slice = (nal[2] & 0x80) != 0;
				leads_unit = (type >= 32 && type <= 35) || type == 39 || (type >= 41 && type <= 44) ||
				             (type >= 48 && type <= 55);
			}
			else {
				int type = nal[0] & 0x1f;
				slice = type >= 1 && type <= 5;
				first_slice = (nal[1] & 0x80) != 0;
				leads_unit = (type >= 6 && type <= 9) || (type >= 14 && type <= 18);
			}

			if (slice) {
				bool starts = has_picture && first_slice;
				has_picture = true;
				return starts;
			}

			if (has_picture && leads_unit) {
				has_picture = false;
				return true;
			}

			return false;
		}

		// a complete unit, if there is one yet. it stays valid until more data is pushed
		bool next(const uint8_t*& unit, size_t& size) {
			for (;;) {
				size_t at = find_start_code(scan);
				if (at == NOT_FOUND) {
					if (data.size() >= 2)
						scan = std::max(scan, data.size() - 2);
					return false;
				}

				// the header and the start of what follows it are needed to tell
				if (at + 6 > data.size()) {
					scan = at;
					return false;
				}

				scan = at + 3;
				if (!starts_unit(&data[at + 3]))
					continue;

				size_t end = at;
				if (end > start && data[end - 1] == 0)
					end--; // start codes can have a leading zero
				if (end == start)
					continue;

				unit = &data[start];
				size = end - start;
				start = end;
				return true;
			}
		}

		// what's left when the stream has ended
		bool rest(const uint8_t*& unit, size_t& size) {
			if (start >= data.size() || !has_picture)
				return false;

			unit = &data[start];
			size = data.size() - start;
			start = data.size();
			return true;
		}
	};

	// what the decoder's pictures are, next to the clip's frames
	enum class Chroma {
		SAME, // 4:4:4, like the clip
		INTERLEAVED, // 4:2:0 like the clip, which the decoder puts out with its two chroma planes interleaved
		HALVED, // 4:4:4, of which the clip has every other chroma sample
	};

	struct Buffer {
		uint8_t* data = nullptr; // the frame's planes, one after another
		int frame = -1;
		int users = 0;
		bool served = false;
	};

	struct Source {
		VSNode* fallback = nullptr;
		VSVideoInfo vi = {};
		std::wstring command_line;
		bool hevc = false;
		Chroma chroma = Chroma::SAME;
		std::vector<int> frame_map; // which decoded frame, counting in display order, each output frame is
		std::vector<int> display_order; // where the frame of each packet comes in display order
		std::vector<bool> needed; // the decoded frames that are an output frame
		size_t frame_size = 0;

		Process process;
		std::thread thread;
		std::once_flag start_once;

		std::mutex mutex;
		std::condition_variable frame_ready;
		std::condition_variable decoder_wake;
		std::vector<Buffer> buffers;
		int decoded = 0; // frames before this one have been through the decoder
		int wanted = -1; // the newest frame asked for
		bool finished = false; // nothing more is coming from the decoder
		bool stopping = false;
		bool closing = false;

		std::once_flag check_once;
		std::atomic<bool> checked = false;
		std::atomic<bool> rejected = false;
		const VSFrame* prop_src = nullptr;

		std::atomic<int> frames_decoded = 0;
		std::atomic<int> frames_fallback = 0;

		// the decoding thread's
		CUcontext context = nullptr;
		CUvideoparser parser = nullptr;
		CUvideodecoder decoder = nullptr;
		unsigned int coded_width = 0;
		unsigned int coded_height = 0;
		int packets = 0;
		int displayed = 0;
		const char* problem = nullptr;

		~Source() {
			{
				std::lock_guard lock(mutex);
				stopping = true;
				closing = true;
			}
			decoder_wake.notify_all();
			process.terminate();
			if (thread.joinable())
				thread.join();
		}

		[[nodiscard]] int plane_width(int plane) const {
			return plane ? vi.width >> vi.format.subSamplingW : vi.width;
		}

		[[nodiscard]] int plane_height(int plane) const {
			return plane ? vi.height >> vi.format.subSamplingH : vi.height;
		}

		// stops decoding. frames come from the fallback from here on
		void abandon() {
			{
				std::lock_guard lock(mutex);
				stopping = true;
			}
			decoder_wake.notify_all();
			frame_ready.notify_all();
			process.terminate();
		}

		void run() {
			if (!decode()) {
				process.terminate();
				if (std::getenv("BLURFLOW_DEBUG"))
					fprintf(stderr, "blurflow: gpu decoding stopped: %s\n", problem ? problem : "stopped");
			}

			std::unique_lock lock(mutex);
			finished = true;
			frame_ready.notify_all();

			// frames that were decoded are still handed out, so their memory stays until the end
			decoder_wake.wait(lock, [&] {
				return closing;
			});

			auto& nv = nvidia();
			if (parser)
				nv.cuvidDestroyVideoParser(parser);
			if (decoder)
				nv.cuvidDestroyDecoder(decoder);
			for (auto& buffer : buffers) {
				if (buffer.data)
					nv.cuMemFreeHost(buffer.data);
			}
			if (context)
				nv.cuCtxDestroy_v2(context);
		}

		bool fail(const char* what) {
			if (!problem)
				problem = what;
			return false;
		}

		bool decode() {
			auto& nv = nvidia();
			if (!nv.loaded)
				return fail("nvidia's decoder isn't available");

			CUdevice device;
			if (nv.cuInit(0) != CUDA_SUCCESS || nv.cuDeviceGet(&device, 0) != CUDA_SUCCESS ||
			    nv.cuCtxCreate_v2(&context, CU_CTX_SCHED_BLOCKING_SYNC, device) != CUDA_SUCCESS)
				return fail("no cuda device");

			// memory the driver can copy frames from the gpu into directly
			for (auto& buffer : buffers) {
				void* memory = nullptr;
				if (nv.cuMemHostAlloc(&memory, frame_size, 0) != CUDA_SUCCESS)
					return fail("out of memory");
				buffer.data = static_cast<uint8_t*>(memory);
			}

			CUVIDPARSERPARAMS parser_params = {};
			parser_params.CodecType = hevc ? cudaVideoCodec_HEVC : cudaVideoCodec_H264;
			parser_params.ulMaxNumDecodeSurfaces = 1; // on_sequence says how many
			parser_params.ulMaxDisplayDelay = DISPLAY_DELAY;
			parser_params.pUserData = this;
			parser_params.pfnSequenceCallback = [](void* self, CUVIDEOFORMAT* format) {
				return static_cast<Source*>(self)->on_sequence(format);
			};
			parser_params.pfnDecodePicture = [](void* self, CUVIDPICPARAMS* picture) {
				return static_cast<Source*>(self)->on_decode(picture);
			};
			parser_params.pfnDisplayPicture = [](void* self, CUVIDPARSERDISPINFO* info) {
				return static_cast<Source*>(self)->on_display(info);
			};
			if (nv.cuvidCreateVideoParser(&parser, &parser_params) != CUDA_SUCCESS)
				return fail("couldn't create a parser");

			if (!process.start(command_line))
				return fail("couldn't start the process");

			AccessUnits units;
			units.hevc = hevc;
			std::vector<uint8_t> chunk(1 << 20);
			const uint8_t* unit;
			size_t size;

			while (size_t got = process.read(chunk.data(), chunk.size())) {
				units.push(chunk.data(), got);
				while (units.next(unit, size)) {
					if (!parse(unit, size))
						return false;
				}
			}

			if (units.rest(unit, size) && !parse(unit, size))
				return false;

			CUVIDSOURCEDATAPACKET end = {};
			end.flags = CUVID_PKT_ENDOFSTREAM;
			if (nv.cuvidParseVideoData(parser, &end) != CUDA_SUCCESS || problem)
				return fail("decoding failed");

			return true;
		}

		bool parse(const uint8_t* unit, size_t size) {
			if (packets >= (int)display_order.size())
				return fail("more pictures than the container has packets");

			// each picture is given the time it's shown at, which here is just its place among the
			// frames. the decoder hands that back with the frame, which is how the frames are known
			// to be the ones they're taken for
			CUVIDSOURCEDATAPACKET packet = {};
			packet.flags = CUVID_PKT_TIMESTAMP | CUVID_PKT_ENDOFPICTURE;
			packet.payload = unit;
			packet.payload_size = (tcu_ulong)size;
			packet.timestamp = display_order[packets++];

			if (nvidia().cuvidParseVideoData(parser, &packet) != CUDA_SUCCESS || problem)
				return fail("decoding failed");
			return true;
		}

		int on_sequence(CUVIDEOFORMAT* format) {
			auto& nv = nvidia();
			int surfaces = format->min_num_decode_surfaces + DISPLAY_DELAY + 4;

			if (decoder) {
				if (format->coded_width != coded_width || format->coded_height != coded_height)
					return fail("the video changes format");
				return surfaces;
			}

			bool interleaved = chroma == Chroma::INTERLEAVED;
			cudaVideoChromaFormat chroma_format = interleaved ? cudaVideoChromaFormat_420 : cudaVideoChromaFormat_444;
			cudaVideoSurfaceFormat output = interleaved ? cudaVideoSurfaceFormat_NV12 : cudaVideoSurfaceFormat_YUV444;

			bool expected = format->codec == (hevc ? cudaVideoCodec_HEVC : cudaVideoCodec_H264) &&
			                format->bit_depth_luma_minus8 == 0 && format->bit_depth_chroma_minus8 == 0 &&
			                format->progressive_sequence && format->chroma_format == chroma_format &&
			                format->display_area.right - format->display_area.left == vi.width &&
			                format->display_area.bottom - format->display_area.top == vi.height;
			if (!expected)
				return fail("not a video this decodes");

			CUVIDDECODECAPS caps = {};
			caps.eCodecType = format->codec;
			caps.eChromaFormat = chroma_format;
			caps.nBitDepthMinus8 = 0;
			if (nv.cuvidGetDecoderCaps(&caps) != CUDA_SUCCESS || !caps.bIsSupported ||
			    !(caps.nOutputFormatMask & (1 << output)) || format->coded_width > caps.nMaxWidth ||
			    format->coded_height > caps.nMaxHeight || format->coded_width < caps.nMinWidth ||
			    format->coded_height < caps.nMinHeight ||
			    (format->coded_width / 16) * (format->coded_height / 16) > caps.nMaxMBCount)
				return fail("the gpu can't decode this video");

			CUVIDDECODECREATEINFO info = {};
			info.ulWidth = format->coded_width;
			info.ulHeight = format->coded_height;
			info.ulMaxWidth = format->coded_width;
			info.ulMaxHeight = format->coded_height;
			info.ulNumDecodeSurfaces = surfaces;
			info.CodecType = format->codec;
			info.ChromaFormat = chroma_format;
			info.ulCreationFlags = cudaVideoCreate_PreferCUVID;
			info.bitDepthMinus8 = 0;
			info.display_area.left = (short)format->display_area.left;
			info.display_area.top = (short)format->display_area.top;
			info.display_area.right = (short)format->display_area.right;
			info.display_area.bottom = (short)format->display_area.bottom;
			info.OutputFormat = output;
			info.DeinterlaceMode = cudaVideoDeinterlaceMode_Weave;
			info.ulTargetWidth = vi.width;
			info.ulTargetHeight = vi.height;
			info.ulNumOutputSurfaces = 2;
			if (nv.cuvidCreateDecoder(&decoder, &info) != CUDA_SUCCESS)
				return fail("couldn't create a decoder");

			coded_width = format->coded_width;
			coded_height = format->coded_height;
			return surfaces;
		}

		int on_decode(CUVIDPICPARAMS* picture) {
			if (!decoder)
				return fail("picture before the video's format");

			{
				std::lock_guard lock(mutex);
				if (stopping)
					return fail("stopped");
			}

			if (nvidia().cuvidDecodePicture(decoder, picture) != CUDA_SUCCESS)
				return fail("decoding a picture failed");
			return 1;
		}

		// a buffer for a newly decoded frame, once there's reason to have the frame
		Buffer* claim(int frame) {
			std::unique_lock lock(mutex);

			for (;;) {
				if (stopping)
					return nullptr;

				Buffer* unused = nullptr;
				Buffer* oldest_served = nullptr;
				Buffer* oldest = nullptr;
				for (auto& buffer : buffers) {
					if (buffer.users)
						continue;
					if (buffer.frame < 0)
						unused = &buffer;
					if (buffer.served && (!oldest_served || buffer.frame < oldest_served->frame))
						oldest_served = &buffer;
					if (!oldest || buffer.frame < oldest->frame)
						oldest = &buffer;
				}

				// something is waiting on this frame or one after it, so it has to go somewhere,
				// even in place of a frame that's yet to be asked for
				bool awaited = wanted >= frame;

				Buffer* buffer = unused ? unused : oldest_served;
				if (!buffer && awaited)
					buffer = oldest;

				if (buffer && (awaited || frame <= wanted + AHEAD_FRAMES)) {
					buffer->frame = -1;
					buffer->served = false;
					buffer->users = 1;
					return buffer;
				}

				decoder_wake.wait(lock);
			}
		}

		int on_display(CUVIDPARSERDISPINFO* info) {
			auto& nv = nvidia();
			int frame = displayed++;

			// frames come out in the order they're shown in. if one doesn't come out where it was
			// expected to, it can't be said which frames these are
			if (info->timestamp != frame)
				return fail("frames aren't in the expected order");

			if (!needed[frame]) {
				std::lock_guard lock(mutex);
				decoded = frame + 1;
				return 1;
			}

			Buffer* buffer = claim(frame);
			if (!buffer)
				return fail("stopped");

			CUVIDPROCPARAMS process_params = {};
			process_params.progressive_frame = info->progressive_frame;
			process_params.top_field_first = info->top_field_first;
			process_params.unpaired_field = info->repeat_first_field < 0;

			unsigned long long mapped = 0;
			unsigned int pitch = 0;
			bool copied = false;
			if (nv.cuvidMapVideoFrame64(decoder, info->picture_index, &mapped, &pitch, &process_params) == CUDA_SUCCESS) {
				// the planes are one under the other, each as wide as the frame
				auto read_rows = [&](size_t first, size_t step, size_t rows, uint8_t* to) {
					CUDA_MEMCPY2D copy = {};
					copy.srcMemoryType = CU_MEMORYTYPE_DEVICE;
					copy.srcDevice = mapped + first * pitch;
					copy.srcPitch = step * pitch;
					copy.dstMemoryType = CU_MEMORYTYPE_HOST;
					copy.dstHost = to;
					copy.dstPitch = vi.width;
					copy.WidthInBytes = vi.width;
					copy.Height = rows;
					return nv.cuMemcpy2D_v2(&copy) == CUDA_SUCCESS;
				};

				size_t height = vi.height;
				if (chroma == Chroma::HALVED) {
					// only the chroma rows the clip has samples from: the lower of each two, in
					// both planes
					copied = read_rows(0, 1, height, buffer->data) &&
					         read_rows(height + 1, 2, height, buffer->data + height * vi.width);
				}
				else {
					copied = read_rows(0, 1, frame_size / vi.width, buffer->data);
				}

				nv.cuvidUnmapVideoFrame64(decoder, mapped);
			}

			{
				std::lock_guard lock(mutex);
				buffer->users = 0;
				if (copied) {
					buffer->frame = frame;
					decoded = frame + 1;
				}
			}
			frame_ready.notify_all();

			if (!copied)
				return fail("couldn't read a frame back");
			return 1;
		}

		void copy_planes(const uint8_t* in, VSFrame* frame, const VSAPI* vsapi) const {
			auto copy_plane = [&](int plane, const uint8_t* from) {
				int w = plane_width(plane);
				int h = plane_height(plane);
				uint8_t* out = vsapi->getWritePtr(frame, plane);
				ptrdiff_t stride = vsapi->getStride(frame, plane);

				if (stride == w) {
					memcpy(out, from, (size_t)w * h);
				}
				else {
					for (int y = 0; y < h; y++)
						memcpy(out + y * stride, from + (size_t)y * w, w);
				}
			};

			size_t luma_size = (size_t)vi.width * vi.height;
			copy_plane(0, in);

			if (chroma == Chroma::SAME) {
				copy_plane(1, in + luma_size);
				copy_plane(2, in + 2 * luma_size);
				return;
			}

			// what's there has two bytes for each chroma sample of the clip. interleaved, they are
			// the sample of either plane. otherwise it's the rows of one plane and then the other,
			// of which the clip has the left sample of each two, like resize.Point would take
			const uint8_t* rows = in + luma_size;
			int w = plane_width(1);
			int h = plane_height(1);
			uint8_t* out_u = vsapi->getWritePtr(frame, 1);
			uint8_t* out_v = vsapi->getWritePtr(frame, 2);
			ptrdiff_t stride = vsapi->getStride(frame, 1);
			const __m128i low_bytes = _mm_set1_epi16(0x00ff);

			auto even_bytes = [&](const uint8_t* row, uint8_t* out) {
				int x = 0;
				for (; x + 16 <= w; x += 16) {
					__m128i a = _mm_and_si128(_mm_loadu_si128((const __m128i*)(row + 2 * x)), low_bytes);
					__m128i b = _mm_and_si128(_mm_loadu_si128((const __m128i*)(row + 2 * x + 16)), low_bytes);
					_mm_storeu_si128((__m128i*)(out + x), _mm_packus_epi16(a, b));
				}
				for (; x < w; x++)
					out[x] = row[2 * x];
			};

			auto odd_bytes = [&](const uint8_t* row, uint8_t* out) {
				int x = 0;
				for (; x + 16 <= w; x += 16) {
					__m128i a = _mm_srli_epi16(_mm_loadu_si128((const __m128i*)(row + 2 * x)), 8);
					__m128i b = _mm_srli_epi16(_mm_loadu_si128((const __m128i*)(row + 2 * x + 16)), 8);
					_mm_storeu_si128((__m128i*)(out + x), _mm_packus_epi16(a, b));
				}
				for (; x < w; x++)
					out[x] = row[2 * x + 1];
			};

			for (int y = 0; y < h; y++) {
				if (chroma == Chroma::INTERLEAVED) {
					const uint8_t* row = rows + (size_t)y * vi.width;
					even_bytes(row, out_u + y * stride);
					odd_bytes(row, out_v + y * stride);
				}
				else {
					even_bytes(rows + (size_t)y * vi.width, out_u + y * stride);
					even_bytes(rows + (size_t)(h + y) * vi.width, out_v + y * stride);
				}
			}
		}

		void start() {
			std::call_once(start_once, [&] {
				thread = std::thread([this] {
					run();
				});
			});
		}

		// output frame n, or nullptr when it has to come from the fallback instead
		const VSFrame* get(int n, VSCore* core, const VSAPI* vsapi) {
			if (rejected)
				return nullptr;

			// unless it was told to get going, a clip that's never read from doesn't decode anything
			start();

			int frame = frame_map[n];
			Buffer* buffer = nullptr;

			{
				std::unique_lock lock(mutex);

				if (frame > wanted) {
					wanted = frame;
					decoder_wake.notify_all();
				}

				for (;;) {
					for (auto& candidate : buffers) {
						if (candidate.frame == frame)
							buffer = &candidate;
					}
					if (buffer)
						break;

					// gone already, or never coming
					if (frame < decoded || finished || stopping)
						return nullptr;

					frame_ready.wait(lock);
				}

				buffer->users++;
				buffer->served = true;
			}

			VSFrame* out = vsapi->newVideoFrame(&vi.format, vi.width, vi.height, prop_src, core);
			copy_planes(buffer->data, out, vsapi);

			{
				std::lock_guard lock(mutex);
				buffer->users--;
			}
			decoder_wake.notify_all();

			return out;
		}

		[[nodiscard]] bool matches(const VSFrame* a, const VSFrame* b, const VSAPI* vsapi) const {
			for (int p = 0; p < vi.format.numPlanes; p++) {
				int w = plane_width(p);
				for (int y = 0; y < plane_height(p); y++) {
					const uint8_t* row_a = vsapi->getReadPtr(a, p) + y * vsapi->getStride(a, p);
					const uint8_t* row_b = vsapi->getReadPtr(b, p) + y * vsapi->getStride(b, p);
					if (memcmp(row_a, row_b, w) != 0)
						return false;
				}
			}
			return true;
		}

		// has to be done, with the fallback's first frame, before any frame is taken from the decoder
		void check(const VSFrame* first, VSCore* core, const VSAPI* vsapi) {
			std::call_once(check_once, [&] {
				// frame properties (colour description and so on) are taken from the fallback's frames,
				// less the ones that are about one frame in particular
				VSFrame* props = vsapi->copyFrame(first, core);
				vsapi->mapDeleteKey(vsapi->getFramePropertiesRW(props), "_PictType");
				vsapi->mapDeleteKey(vsapi->getFramePropertiesRW(props), "_AbsoluteTime");
				prop_src = props;

				// the frames can't be trusted to be the fallback's if even the first one isn't
				const VSFrame* mine = get(0, core, vsapi);
				if (!mine || !matches(mine, first, vsapi)) {
					rejected = true;
					abandon();
				}
				if (mine)
					vsapi->freeFrame(mine);

				checked = true;
			});
		}
	};

	const intptr_t STAGE_CHECK = 1;
	const intptr_t STAGE_FALLBACK = 2;

	const VSFrame* VS_CC get_frame(
		int n,
		int activation_reason,
		void* instance_data,
		void** frame_data,
		VSFrameContext* frame_ctx,
		VSCore* core,
		const VSAPI* vsapi
	) {
		auto* s = static_cast<Source*>(instance_data);

		if (activation_reason == arInitial) {
			if (!s->checked) {
				frame_data[0] = (void*)STAGE_CHECK;
				vsapi->requestFrameFilter(0, s->fallback, frame_ctx);
				return nullptr;
			}
		}
		else if (activation_reason == arAllFramesReady) {
			if ((intptr_t)frame_data[0] == STAGE_FALLBACK)
				return vsapi->getFrameFilter(n, s->fallback, frame_ctx);

			const VSFrame* first = vsapi->getFrameFilter(0, s->fallback, frame_ctx);
			s->check(first, core, vsapi);
			vsapi->freeFrame(first);
		}
		else {
			return nullptr;
		}

		if (const VSFrame* frame = s->get(n, core, vsapi)) {
			s->frames_decoded++;
			return frame;
		}

		s->frames_fallback++;
		frame_data[0] = (void*)STAGE_FALLBACK;
		vsapi->requestFrameFilter(n, s->fallback, frame_ctx);
		return nullptr;
	}

	void VS_CC free_source(void* instance_data, VSCore* core, const VSAPI* vsapi) {
		auto* s = static_cast<Source*>(instance_data);

		if (std::getenv("BLURFLOW_DEBUG")) {
			fprintf(
				stderr,
				"blurflow: %d frames from the gpu decoder, %d from the fallback\n",
				s->frames_decoded.load(),
				s->frames_fallback.load()
			);
		}

		vsapi->freeFrame(s->prop_src);
		vsapi->freeNode(s->fallback);
		delete s;
	}

	void VS_CC create(const VSMap* in, VSMap* out, void* user_data, VSCore* core, const VSAPI* vsapi) {
		auto s = std::make_unique<Source>();
		s->fallback = vsapi->mapGetNode(in, "fallback", 0, nullptr);
		s->vi = *vsapi->getVideoInfo(s->fallback);

		auto fail = [&](const char* message) {
			vsapi->mapSetError(out, message);
			vsapi->freeNode(s->fallback);
		};

		const VSVideoFormat& format = s->vi.format;
		bool yuv8 = format.colorFamily == cfYUV && format.sampleType == stInteger && format.bitsPerSample == 8 &&
		            s->vi.width > 0 && s->vi.height > 0;
		bool yuv444 = format.subSamplingW == 0 && format.subSamplingH == 0;
		bool yuv420 = format.subSamplingW == 1 && format.subSamplingH == 1 && s->vi.width % 2 == 0 &&
		              s->vi.height % 2 == 0;
		if (!yuv8 || !(yuv444 || yuv420))
			return fail("blurflow: the source must be constant format yuv420p8 or yuv444p8");

		// the clip's frames can be 4:2:0 when the video is 4:4:4, to save converting them
		int err = 0;
		bool halved = vsapi->mapGetInt(in, "subsample", 0, &err) != 0;
		if (halved && !yuv420)
			return fail("blurflow: the source must be yuv420p8 to subsample to");

		size_t luma_size = (size_t)s->vi.width * s->vi.height;
		if (halved) {
			s->chroma = Chroma::HALVED;
			s->frame_size = luma_size * 2;
		}
		else if (yuv420) {
			s->chroma = Chroma::INTERLEAVED;
			s->frame_size = luma_size * 3 / 2;
		}
		else {
			s->chroma = Chroma::SAME;
			s->frame_size = luma_size * 3;
		}

		std::string codec(vsapi->mapGetData(in, "codec", 0, nullptr), vsapi->mapGetDataSize(in, "codec", 0, nullptr));
		if (codec != "hevc" && codec != "h264")
			return fail("blurflow: codec must be hevc or h264");
		s->hevc = codec == "hevc";

		if (!nvidia().loaded)
			return fail("blurflow: nvidia's decoder isn't available");

		int num_arguments = vsapi->mapNumElements(in, "command");
		for (int i = 0; i < num_arguments; i++) {
			const char* argument = vsapi->mapGetData(in, "command", i, nullptr);
			append_argument(s->command_line, widen(argument, vsapi->mapGetDataSize(in, "command", i, nullptr)));
		}

		int num_packets = vsapi->mapNumElements(in, "order");
		const int64_t* order = vsapi->mapGetIntArray(in, "order", nullptr);
		for (int i = 0; i < num_packets; i++) {
			if (order[i] < 0 || order[i] >= num_packets)
				return fail("blurflow: order must say where each packet's frame comes in display order");
			s->display_order.push_back((int)order[i]);
		}

		int num_frames = vsapi->mapNumElements(in, "frames");
		if (num_frames != s->vi.numFrames)
			return fail("blurflow: frames must say which decoded frame each of the clip's frames is");

		s->needed.resize(num_packets);
		const int64_t* frames = vsapi->mapGetIntArray(in, "frames", nullptr);
		for (int i = 0; i < num_frames; i++) {
			if (frames[i] < 0 || frames[i] >= num_packets || (i > 0 && frames[i] < frames[i - 1]))
				return fail("blurflow: frames must be decoded frames, in order");
			s->frame_map.push_back((int)frames[i]);
			s->needed[frames[i]] = true;
		}

		s->buffers.resize(HELD_FRAMES);

		// setting the decoder up takes a moment, which can pass while the rest of the script is run
		if (vsapi->mapGetInt(in, "start", 0, &err) != 0)
			s->start();

		const VSFilterDependency deps[] = { { s->fallback, rpGeneral } };
		const VSVideoInfo vi = s->vi;
		vsapi->createVideoFilter(out, "Source", &vi, get_frame, free_source, fmParallel, deps, 1, s.release(), core);
	}
}

void register_source_filter(VSPlugin* plugin, const VSPLUGINAPI* vspapi) {
	vspapi->registerFunction(
		"Source",
		"fallback:vnode;command:data[];frames:int[];order:int[];codec:data;subsample:int:opt;start:int:opt;",
		"clip:vnode;",
		create,
		nullptr,
		plugin
	);
}

#else

// only implemented for windows so far
void register_source_filter(VSPlugin* plugin, const VSPLUGINAPI* vspapi) {}

#endif
