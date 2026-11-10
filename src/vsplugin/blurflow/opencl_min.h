// the small part of opencl that blurflow uses, resolved from the system's OpenCL library at runtime
// so that building the plugin needs no opencl sdk

#pragma once

#include <cstddef>
#include <cstdint>

#ifdef _WIN32
#	define CL_API __stdcall
#else
#	define CL_API
#endif

using cl_int = int32_t;
using cl_uint = uint32_t;
using cl_ulong = uint64_t;
using cl_bool = cl_uint;
using cl_bitfield = cl_ulong;

using cl_platform_id = struct _cl_platform_id*;
using cl_device_id = struct _cl_device_id*;
using cl_context = struct _cl_context*;
using cl_command_queue = struct _cl_command_queue*;
using cl_mem = struct _cl_mem*;
using cl_program = struct _cl_program*;
using cl_kernel = struct _cl_kernel*;
using cl_event = struct _cl_event*;

struct cl_image_format {
	cl_uint image_channel_order;
	cl_uint image_channel_data_type;
};

constexpr cl_int CL_SUCCESS = 0;
constexpr cl_bool CL_FALSE = 0;
constexpr cl_bool CL_TRUE = 1;
constexpr cl_bitfield CL_DEVICE_TYPE_GPU = 1 << 2;
constexpr cl_uint CL_DEVICE_NAME = 0x102B;
constexpr cl_bitfield CL_MEM_READ_WRITE = 1 << 0;
constexpr cl_bitfield CL_MEM_READ_ONLY = 1 << 2;
constexpr cl_bitfield CL_MEM_ALLOC_HOST_PTR = 1 << 4;
constexpr cl_bitfield CL_MAP_READ = 1 << 0;
constexpr cl_bitfield CL_MAP_WRITE = 1 << 1;
constexpr cl_uint CL_R = 0x10B0;
constexpr cl_uint CL_RGBA = 0x10B5;
constexpr cl_uint CL_UNORM_INT8 = 0x10D2;
constexpr cl_uint CL_UNORM_INT16 = 0x10D3;
constexpr cl_uint CL_FLOAT = 0x10DE;
constexpr cl_uint CL_PROGRAM_BUILD_LOG = 0x1183;

// X(return type, name, arguments)
#define BLURFLOW_CL_FUNCTIONS(X)                                                                                       \
	X(cl_int, clGetPlatformIDs, (cl_uint, cl_platform_id*, cl_uint*))                                                  \
	X(cl_int, clGetDeviceIDs, (cl_platform_id, cl_bitfield, cl_uint, cl_device_id*, cl_uint*))                         \
	X(cl_int, clGetDeviceInfo, (cl_device_id, cl_uint, size_t, void*, size_t*))                                        \
	X(cl_context, clCreateContext, (const intptr_t*, cl_uint, const cl_device_id*, void*, void*, cl_int*))             \
	X(cl_command_queue, clCreateCommandQueue, (cl_context, cl_device_id, cl_bitfield, cl_int*))                        \
	X(cl_program, clCreateProgramWithSource, (cl_context, cl_uint, const char**, const size_t*, cl_int*))              \
	X(cl_int, clBuildProgram, (cl_program, cl_uint, const cl_device_id*, const char*, void*, void*))                   \
	X(cl_int, clGetProgramBuildInfo, (cl_program, cl_device_id, cl_uint, size_t, void*, size_t*))                      \
	X(cl_kernel, clCreateKernel, (cl_program, const char*, cl_int*))                                                   \
	X(cl_mem, clCreateImage2D, (cl_context, cl_bitfield, const cl_image_format*, size_t, size_t, size_t, void*, cl_int*)) \
	X(cl_mem, clCreateBuffer, (cl_context, cl_bitfield, size_t, void*, cl_int*))                                       \
	X(cl_int, clSetKernelArg, (cl_kernel, cl_uint, size_t, const void*))                                               \
	X(cl_int, clEnqueueNDRangeKernel,                                                                                  \
	  (cl_command_queue, cl_kernel, cl_uint, const size_t*, const size_t*, const size_t*, cl_uint, const cl_event*, cl_event*)) \
	X(cl_int, clEnqueueWriteImage,                                                                                     \
	  (cl_command_queue, cl_mem, cl_bool, const size_t*, const size_t*, size_t, size_t, const void*, cl_uint, const cl_event*, cl_event*)) \
	X(cl_int, clEnqueueWriteBuffer,                                                                                    \
	  (cl_command_queue, cl_mem, cl_bool, size_t, size_t, const void*, cl_uint, const cl_event*, cl_event*))           \
	X(cl_int, clEnqueueReadBuffer,                                                                                     \
	  (cl_command_queue, cl_mem, cl_bool, size_t, size_t, void*, cl_uint, const cl_event*, cl_event*))                 \
	X(void*, clEnqueueMapBuffer,                                                                                       \
	  (cl_command_queue, cl_mem, cl_bool, cl_bitfield, size_t, size_t, cl_uint, const cl_event*, cl_event*, cl_int*))  \
	X(cl_int, clEnqueueUnmapMemObject, (cl_command_queue, cl_mem, void*, cl_uint, const cl_event*, cl_event*))         \
	X(cl_int, clFlush, (cl_command_queue))                                                                             \
	X(cl_int, clFinish, (cl_command_queue))                                                                            \
	X(cl_int, clReleaseMemObject, (cl_mem))                                                                            \
	X(cl_int, clReleaseKernel, (cl_kernel))                                                                            \
	X(cl_int, clReleaseProgram, (cl_program))                                                                          \
	X(cl_int, clReleaseCommandQueue, (cl_command_queue))                                                               \
	X(cl_int, clReleaseContext, (cl_context))

struct OpenCL {
#define X(ret, name, args) ret(CL_API* name) args = nullptr;
	BLURFLOW_CL_FUNCTIONS(X)
#undef X

	// returns false when the opencl library or one of the functions is missing
	bool load();
};

OpenCL& opencl();
