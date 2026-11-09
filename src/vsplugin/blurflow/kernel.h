#pragma once

// renders every interpolated frame between two source frames and accumulates their weighted sum,
// so that only finished blended frames ever leave the gpu.
//
// the interpolation reproduces what svp's SmoothFps does for algorithm 13 on the gpu: block vectors
// are upsampled with a cubic b-spline, both source frames are warped along them, and the result is
// the median of the two warps and a plain cross-fade. luma is interpolated in linear light.
inline const char* const BLURFLOW_KERNEL_SOURCE = R"CLC(
const sampler_t s_linear = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_LINEAR;
const sampler_t s_nearest = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_NONE | CLK_FILTER_NEAREST;

__kernel void to_linear(__read_only image2d_t src, __write_only image2d_t dst)
{
	int2 p = (int2)(get_global_id(0), get_global_id(1));
	float v = read_imagef(src, s_nearest, p).x;
	v = v < 0.081f ? native_divide(v, 4.5f) : native_powr(native_divide(v + 0.099f, 1.099f), 1.0f / 0.45f);
	write_imagef(dst, p, (float4)(v, 0.0f, 0.0f, 0.0f));
}

float to_gamma(float v)
{
	v = native_divide(v, 255.0f);
	v = v < 0.018f ? v * 4.5f : 1.099f * native_powr(v, 0.45f) - 0.099f;
	return v * 255.0f;
}

float quantise(float v)
{
	return clamp(round(v), 0.0f, 255.0f);
}

// cubic b-spline through four bilinear taps
float4 vector_at(__read_only image2d_t vectors, float2 pos)
{
	float2 g = pos - 0.5f;
	float2 i = floor(g);
	float2 f = g - i;
	float2 o = 1.0f - f;
	float2 w0 = 1.0f / 6.0f * o * o * o;
	float2 w1 = 2.0f / 3.0f - 0.5f * f * f * (2.0f - f);
	float2 w2 = 2.0f / 3.0f - 0.5f * o * o * (2.0f - o);
	float2 w3 = 1.0f / 6.0f * f * f * f;
	float2 g0 = w0 + w1;
	float2 g1 = w2 + w3;
	float2 h0 = (w1 / g0) - 0.5f + i;
	float2 h1 = (w3 / g1) + 1.5f + i;
	float4 a = read_imagef(vectors, s_linear, h0);
	float4 b = read_imagef(vectors, s_linear, (float2)(h1.x, h0.y));
	float4 c = read_imagef(vectors, s_linear, (float2)(h0.x, h1.y));
	float4 d = read_imagef(vectors, s_linear, h1);
	a = mix(c, a, g0.y);
	b = mix(d, b, g0.y);
	return mix(b, a, g0.x) * 65535.0f - 1024.0f;
}

// one plane of one source interval.
//   src_f/src_b: the two source frames as stored. ref_f/ref_b: what gets warped (linear light for luma)
//   weights[0]: weight of the first source frame, weights[k]: weight of the k-th interpolated frame
//   tail: weight of the second source frame
//   flags: 1 = first pass of the output frame, 2 = last pass, 4 = luma, 8 = has interpolated frames
__kernel void interval(__global float* acc, __global uchar* dst, int base,
                       __read_only image2d_t src_f, __read_only image2d_t src_b,
                       __read_only image2d_t ref_f, __read_only image2d_t ref_b,
                       __read_only image2d_t vectors,
                       __global const float* weights, int weights_at, int steps, float tail, float divisor,
                       int ratio, float block_w, float block_h, float vector_div, int flags)
{
	int x = get_global_id(0), y = get_global_id(1);
	int w = get_global_size(0), h = get_global_size(1);
	int idx = base + y * w + x;
	float2 here = (float2)(x, y);
	float2 centre = here + 0.5f;
	weights += weights_at;

	float sum = (flags & 1) ? 0.0f : acc[idx];

	if (weights[0] != 0.0f)
		sum += weights[0] * quantise(255.0f * read_imagef(src_f, s_nearest, (int2)(x, y)).x);
	if (tail != 0.0f)
		sum += tail * quantise(255.0f * read_imagef(src_b, s_nearest, (int2)(x, y)).x);

	if (flags & 8) {
		float2 size = (float2)(w - 1, h - 1);
		float4 vec = vector_at(vectors, (float2)(x * ratio, y * ratio) / (float2)(block_w, block_h));
		float f0 = 255.0f * read_imagef(ref_f, s_linear, centre).x;
		float b0 = 255.0f * read_imagef(ref_b, s_linear, centre).x;

		for (int k = 1; k < steps; k++) {
			float wk = weights[k];
			if (wk == 0.0f)
				continue;

			int time = (k * 512 + steps) / (2 * steps); // position between the frames in 1/256ths
			float2 pos_f = clamp(here + native_divide(vec.xw * (float)time, vector_div), (float2)(0.0f), size) + 0.5f;
			float2 pos_b = clamp(here + native_divide(vec.zy * (float)(256 - time), vector_div), (float2)(0.0f), size) + 0.5f;
			float rf = 255.0f * read_imagef(ref_f, s_linear, pos_f).x;
			float rb = 255.0f * read_imagef(ref_b, s_linear, pos_b).x;

			float lo = fmin(rf, rb);
			float hi = rf + rb - lo;
			float res = fmax(lo, fmin(hi, mix(f0, b0, native_divide((float)time, 256.0f))));

			if (flags & 4)
				res = to_gamma(res);

			sum += wk * quantise(res);
		}
	}

	if (flags & 2)
		dst[idx] = (uchar)quantise(native_divide(sum, divisor));
	else
		acc[idx] = sum;
}
)CLC";
