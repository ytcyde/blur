// point-resamples chroma between yuv444p8 and yuv420p8.
//
// svp works in yuv420p8, so most renders convert every source frame to it and every output frame
// back. these give the same frames as resize.Point does for that, but share the luma plane with
// the source frame instead of copying it and only do the one thing, which makes them several times
// cheaper.

#include "chroma.h"

#include <cstring>
#include <emmintrin.h>
#include <memory>
#include <vector>

namespace {
	struct Resampler {
		VSNode* clip = nullptr;
		VSVideoInfo vi = {};
		bool subsample = false;
	};

	// takes the left sample of each pair on the lower of each two rows, like resize.Point
	void subsample_plane(const uint8_t* src, ptrdiff_t src_stride, uint8_t* dst, ptrdiff_t dst_stride, int w, int h) {
		const __m128i low_bytes = _mm_set1_epi16(0x00ff);

		for (int y = 0; y < h; y++) {
			const uint8_t* row = src + (2 * y + 1) * src_stride;
			uint8_t* out = dst + y * dst_stride;

			int x = 0;
			for (; x + 16 <= w; x += 16) {
				__m128i a = _mm_and_si128(_mm_loadu_si128((const __m128i*)(row + 2 * x)), low_bytes);
				__m128i b = _mm_and_si128(_mm_loadu_si128((const __m128i*)(row + 2 * x + 16)), low_bytes);
				_mm_storeu_si128((__m128i*)(out + x), _mm_packus_epi16(a, b));
			}
			for (; x < w; x++)
				out[x] = row[2 * x];
		}
	}

	// repeats each sample over two rows and two columns. resize.Point treats the chroma samples as
	// sitting on the left luma sample of each pair, which puts the columns one to the left of a
	// plain doubling
	void upsample_plane(
		const uint8_t* src, ptrdiff_t src_stride, uint8_t* dst, ptrdiff_t dst_stride, int w, int h, uint8_t* doubled
	) {
		for (int y = 0; y < h; y++) {
			const uint8_t* row = src + y * src_stride;

			int x = 0;
			for (; x + 16 <= w; x += 16) {
				__m128i v = _mm_loadu_si128((const __m128i*)(row + x));
				_mm_storeu_si128((__m128i*)(doubled + 2 * x), _mm_unpacklo_epi8(v, v));
				_mm_storeu_si128((__m128i*)(doubled + 2 * x + 16), _mm_unpackhi_epi8(v, v));
			}
			for (; x < w; x++)
				doubled[2 * x] = doubled[2 * x + 1] = row[x];
			doubled[2 * w] = row[w - 1];

			memcpy(dst + (2 * y) * dst_stride, doubled + 1, (size_t)2 * w);
			memcpy(dst + (2 * y + 1) * dst_stride, doubled + 1, (size_t)2 * w);
		}
	}

	const VSFrame* VS_CC get_frame(
		int n,
		int activation_reason,
		void* instance_data,
		void** frame_data,
		VSFrameContext* frame_ctx,
		VSCore* core,
		const VSAPI* vsapi
	) {
		auto* r = static_cast<Resampler*>(instance_data);

		if (activation_reason == arInitial) {
			vsapi->requestFrameFilter(n, r->clip, frame_ctx);
			return nullptr;
		}

		if (activation_reason != arAllFramesReady)
			return nullptr;

		const VSFrame* src = vsapi->getFrameFilter(n, r->clip, frame_ctx);

		// the luma plane is the source's own
		const VSFrame* plane_src[3] = { src, nullptr, nullptr };
		const int planes[3] = { 0, 0, 0 };
		VSFrame* dst = vsapi->newVideoFrame2(&r->vi.format, r->vi.width, r->vi.height, plane_src, planes, src, core);

		int chroma_w = r->vi.width / 2;
		int chroma_h = r->vi.height / 2;
		std::vector<uint8_t> doubled;
		if (!r->subsample)
			doubled.resize((size_t)r->vi.width + 16);

		for (int p = 1; p < 3; p++) {
			const uint8_t* in = vsapi->getReadPtr(src, p);
			ptrdiff_t in_stride = vsapi->getStride(src, p);
			uint8_t* out = vsapi->getWritePtr(dst, p);
			ptrdiff_t out_stride = vsapi->getStride(dst, p);

			if (r->subsample)
				subsample_plane(in, in_stride, out, out_stride, chroma_w, chroma_h);
			else
				upsample_plane(in, in_stride, out, out_stride, chroma_w, chroma_h, doubled.data());
		}

		// chroma only has a location when it's subsampled
		VSMap* props = vsapi->getFramePropertiesRW(dst);
		if (r->subsample)
			vsapi->mapSetInt(props, "_ChromaLocation", 0, maReplace);
		else
			vsapi->mapDeleteKey(props, "_ChromaLocation");

		vsapi->freeFrame(src);
		return dst;
	}

	void VS_CC free_resampler(void* instance_data, VSCore* core, const VSAPI* vsapi) {
		auto* r = static_cast<Resampler*>(instance_data);
		vsapi->freeNode(r->clip);
		delete r;
	}

	void VS_CC create(const VSMap* in, VSMap* out, void* user_data, VSCore* core, const VSAPI* vsapi) {
		auto r = std::make_unique<Resampler>();
		r->subsample = user_data != nullptr;
		r->clip = vsapi->mapGetNode(in, "clip", 0, nullptr);
		r->vi = *vsapi->getVideoInfo(r->clip);

		int subsampling = r->subsample ? 0 : 1;
		const VSVideoFormat& format = r->vi.format;
		bool supported = format.colorFamily == cfYUV && format.sampleType == stInteger && format.bitsPerSample == 8 &&
		                 format.subSamplingW == subsampling && format.subSamplingH == subsampling && r->vi.width > 0 &&
		                 r->vi.width % 2 == 0 && r->vi.height % 2 == 0;
		if (!supported) {
			vsapi->mapSetError(
				out,
				r->subsample ? "blurflow: clip must be yuv444p8 with even dimensions"
							 : "blurflow: clip must be yuv420p8 with even dimensions"
			);
			vsapi->freeNode(r->clip);
			return;
		}

		int target = r->subsample ? 1 : 0;
		vsapi->queryVideoFormat(&r->vi.format, cfYUV, stInteger, 8, target, target, core);

		const VSFilterDependency deps[] = { { r->clip, rpStrictSpatial } };
		const VSVideoInfo vi = r->vi;
		const char* name = r->subsample ? "Subsample" : "Upsample";
		vsapi->createVideoFilter(out, name, &vi, get_frame, free_resampler, fmParallel, deps, 1, r.release(), core);
	}
}

void register_chroma_filters(VSPlugin* plugin, const VSPLUGINAPI* vspapi) {
	static int subsample;
	vspapi->registerFunction("Subsample", "clip:vnode;", "clip:vnode;", create, &subsample, plugin);
	vspapi->registerFunction("Upsample", "clip:vnode;", "clip:vnode;", create, nullptr, plugin);
}
