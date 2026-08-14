#include <c10/core/AllocatorConfig.h>
#include <c10/cuda/CUDAAllocatorConfig.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAFunctions.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/util/Exception.h>

#include <gtest/gtest.h>

#include <string>

// Device test: a single allocation larger than the stream's downsized
// per-segment reserve must fail with an actionable OutOfMemoryError rather than
// crashing. Lives in its own binary because it mutates the process-global
// allocator config and tags the current stream.
TEST(ExpandableSegmentReserveDeviceTest, AllocLargerThanReserveThrows) {
  if (c10::cuda::device_count() == 0) {
    GTEST_SKIP() << "no CUDA device";
  }
  c10::cuda::set_device(0);
  c10::cuda::CUDACachingAllocator::init(c10::cuda::device_count());

  cudaDeviceProp prop{};
  C10_CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
  const size_t device_total = prop.totalGlobalMem;

  // Reserve ~5% of the device for the tagged class, then ask for 4x that. The
  // request is far below total device memory, so it only fails because a single
  // allocation cannot span expandable segments.
  c10::CachingAllocator::setAllocatorSettings(
      "expandable_segments:True,"
      "expandable_segments_reserve_by_class:[tiny:0.05]");
  ASSERT_TRUE(c10::cuda::CUDACachingAllocator::CUDAAllocatorConfig::
                  expandable_segments())
      << "expandable_segments must be enabled for this test to be meaningful";

  auto stream = c10::cuda::getCurrentCUDAStream(0);
  c10::cuda::CUDACachingAllocator::setExpandableSegmentReserveClassForStream(
      stream.stream(), "tiny");

  const size_t reserve = device_total / 20; // 0.05 * total
  const size_t request = reserve * 4;
  ASSERT_LT(request, device_total)
      << "request must be satisfiable if segments could span";

  try {
    auto blk = c10::cuda::CUDACachingAllocator::get()->allocate(request);
    ADD_FAILURE() << "expected an OutOfMemoryError for a " << request
                  << "-byte allocation against a " << reserve
                  << "-byte per-segment reserve";
  } catch (const c10::OutOfMemoryError& e) {
    const std::string msg = e.what();
    EXPECT_NE(msg.find("expandable-segment reserve"), std::string::npos) << msg;
    EXPECT_NE(
        msg.find("expandable_segments_reserve_by_class"), std::string::npos)
        << msg;
  }

  c10::cuda::CUDACachingAllocator::setExpandableSegmentReserveClassForStream(
      stream.stream(), "");
}

// Same failure via the global knob on an untagged stream -- i.e. what a user
// hits with only PYTORCH_CUDA_ALLOC_CONF set and no code change.
TEST(ExpandableSegmentReserveDeviceTest, GlobalReserveUntaggedStreamThrows) {
  if (c10::cuda::device_count() == 0) {
    GTEST_SKIP() << "no CUDA device";
  }
  c10::cuda::set_device(0);
  c10::cuda::CUDACachingAllocator::init(c10::cuda::device_count());

  cudaDeviceProp prop{};
  C10_CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
  const size_t device_total = prop.totalGlobalMem;

  c10::CachingAllocator::setAllocatorSettings(
      "expandable_segments:True,expandable_segments_reserve:0.05");

  // A stream with no reserve class resolves to the global reserve. Clear the
  // tag explicitly so this does not depend on another test having cleaned up.
  auto stream = c10::cuda::getCurrentCUDAStream(0);
  c10::cuda::CUDACachingAllocator::setExpandableSegmentReserveClassForStream(
      stream.stream(), "");
  ASSERT_EQ(
      c10::cuda::CUDACachingAllocator::
          getExpandableSegmentReserveClassForStream(stream.stream()),
      "");

  const size_t request = (device_total / 20) * 4;
  try {
    auto blk = c10::cuda::CUDACachingAllocator::get()->allocate(request);
    ADD_FAILURE() << "expected an OutOfMemoryError for a " << request
                  << "-byte allocation against a 5% global reserve";
  } catch (const c10::OutOfMemoryError& e) {
    const std::string msg = e.what();
    EXPECT_NE(msg.find("expandable-segment reserve"), std::string::npos) << msg;
    EXPECT_NE(msg.find("expandable_segments_reserve"), std::string::npos)
        << msg;
  }
}
