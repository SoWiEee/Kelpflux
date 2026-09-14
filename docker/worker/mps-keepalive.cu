#include <chrono>
#include <thread>

#include <cuda_runtime.h>

int main() {
    void* allocation = nullptr;
    if (cudaMalloc(&allocation, 1) != cudaSuccess) {
        return 1;
    }

    while (true) {
        std::this_thread::sleep_for(std::chrono::seconds(10));
    }
}
