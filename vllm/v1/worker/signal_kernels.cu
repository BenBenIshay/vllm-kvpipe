#include <cuda_runtime.h>

extern "C" {
    // Writes the finished layer_id to a memory address mapped to Host Pinned Memory.
    __global__ void signal_layer_completion(int* status_ptr, int layer_id) {
        // Ensure all KV-cache writes from the previous Attention kernel 
        // are physically committed to HBM and visible to the system bus.
        __threadfence_system(); 

        if (threadIdx.x == 0 && blockIdx.x == 0) {
            // Volatile store ensures we bypass GPU L1/L2 caches to hit the PCIe bus.
            *((volatile int*)status_ptr) = layer_id;
        }
    }

    // Host-callable wrapper that launches the kernel on the default stream.
    void launch_signal_layer_completion(int* status_ptr, int layer_id) {
        signal_layer_completion<<<1, 1>>>(status_ptr, layer_id);
    }
}
