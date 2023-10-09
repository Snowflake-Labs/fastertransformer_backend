#include "corvo_batcher.h"

#include <semaphore>

#include "triton/core/tritonbackend.h"

namespace snowflake::corvo_batcher { namespace {

struct CorvoBatcher {
  std::counting_semaphore semaphore;
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
  *batcher = static_cast<TRITONBACKEND_Batcher*>(
      new CorvoBatcher(max_concurrent_requests));
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
  delete static_cast<CorvoBatcher*>(batcher);
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
  if (static_cast<CorvoBatcher*>(batcher)->semaphore.try_acquire()) {
    *userp = batcher;
    return nullptr;
  }
  return TRITONSERVER_ErrorNew(
      TRITONSERVER_ERROR_UNAVAILABLE, "too many concurrent requests")
}

/// Callback to be invoked when Triton has finishing forming a batch.
///
/// \param userp The placeholder for backend to store and retrieve information
/// about this pending batch.
/// \return a TRITONSERVER_Error indicating success or failure.
TRITONBACKEND_ISPEC TRITONSERVER_Error*
TRITONBACKEND_ModelBatchFinalize(void* userp)
{
  static_cast<CorvoBatcher*>(batcher)->semaphore.release();
  return nullptr;
}
}

}}  // namespace snowflake::corvo_batcher
