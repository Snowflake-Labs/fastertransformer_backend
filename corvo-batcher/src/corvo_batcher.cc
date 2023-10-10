// Custom batcher for corvo, which:
// - Prevents merging of batches.
// - Limits the number of concurrent requests.

#include <memory>
#include <mutex>

#include "triton/core/tritonbackend.h"

namespace {

constexpr uint64_t kNumConcurrentRequest = 16;

// Semaphore implements a basic semaphore in C++, similar to
// std::counting_semaphore in C++20.
class Semaphore {
 public:
  Semaphore(uint64_t n) : n_(n) {}

  bool tryAcquire()
  {
    std::unique_lock lock(mu_);
    if (n_ > 0) {
      n_--;
      return true;
    }
    return false;
  }

  void release()
  {
    std::unique_lock lock(mu_);
    n_++;
  }

 private:
  std::mutex mu_;
  uint64_t n_;  // protected by mu_
};

class CorvoBatcher {
 public:
  CorvoBatcher(uint64_t n) : semaphore_(std::make_unique<Semaphore>(n)) {}

  Semaphore* semaphore() const { return semaphore_.get(); }

 private:
  // Heap allocate the Semaphore to avoid a const_cast in
  // TRITONBACKEND_ModelBatchInitialize.
  std::unique_ptr<Semaphore> semaphore_;
};

extern "C" {
/// TRITONBACKEND Batching
///
/// API to add custom batching strategy
///
/// The following functions can be implemented by a backend to add custom
/// batching conditionals on top of the existing Triton batching strategy. The
/// functions are optional but all or none must be implemented.
///

/// Create a new batcher for use with custom batching. This is called during
/// model loading. The batcher will point to a user-defined data structure
/// that holds read-only data used for custom batching.
///
/// \param batcher User-defined placeholder for backend to store and
/// retrieve information about the batching strategy for this
/// model.RITONBACKEND_ISPEC return a TRITONSERVER_Error indicating success or
/// failure. \param model The backend model for which Triton is forming a
/// batch. \return a TRITONSERVER_Error indicating success or failure.
TRITONBACKEND_ISPEC TRITONSERVER_Error*
TRITONBACKEND_ModelBatcherInitialize(
    TRITONBACKEND_Batcher** batcher, TRITONBACKEND_Model* model)
{
  *batcher = reinterpret_cast<TRITONBACKEND_Batcher*>(
      new CorvoBatcher(kNumConcurrentRequest));
  return nullptr;
}

/// Free memory associated with batcher. This is called during model
/// unloading.
///
/// \param batcher User-defined placeholder for backend to store and
/// retrieve information about the batching strategy for this model.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONBACKEND_ISPEC TRITONSERVER_Error*
TRITONBACKEND_ModelBatcherFinalize(TRITONBACKEND_Batcher* batcher)
{
  delete reinterpret_cast<CorvoBatcher*>(batcher);
  return nullptr;
}

/// Check whether a request should be added to the pending model batch.
///
/// \param request The request to be added to the pending batch.
/// \param userp The placeholder for backend to store and retrieve information
/// about this pending batch. When the callback returns, this should reflect
/// the latest batch information.
/// \param should_include The pointer to be updated on whether the request
/// should be included in the batch.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONBACKEND_ISPEC TRITONSERVER_Error*
TRITONBACKEND_ModelBatchIncludeRequest(
    TRITONBACKEND_Request* request, void* userp, bool* should_include)
{
  *should_include = false;  // Do not batch together multiple requests.
  return nullptr;
}

/// Callback to be invoked when Triton has begun forming a batch.
///
/// \param batcher The read-only placeholder for backend to retrieve
// information about the batching strategy for this model.
/// \param userp The placeholder for backend to store and retrieve information
/// about this pending batch.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONBACKEND_ISPEC TRITONSERVER_Error*
TRITONBACKEND_ModelBatchInitialize(
    const TRITONBACKEND_Batcher* batcher, void** userp)
{
  auto* semaphore = reinterpret_cast<const CorvoBatcher*>(batcher)->semaphore();
  if (semaphore->tryAcquire()) {
    *userp = reinterpret_cast<void*>(semaphore);
    return nullptr;
  }
  return TRITONSERVER_ErrorNew(
      TRITONSERVER_ERROR_UNAVAILABLE, "too many concurrent requests");
}

/// Callback to be invoked when Triton has finishing forming a batch.
///
/// \param userp The placeholder for backend to store and retrieve information
/// about this pending batch.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONBACKEND_ISPEC TRITONSERVER_Error*
TRITONBACKEND_ModelBatchFinalize(void* userp)
{
  reinterpret_cast<Semaphore*>(userp)->release();
  return nullptr;
}
}

}  // namespace
