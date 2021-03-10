
#include "collision.h"
__managed__ Body **d_bodies;

// Helper variables for rendering and checksum computation.
__device__ int r_draw_counter = 0;
__device__ float r_Body_pos_x[kNumBodies];
__device__ float r_Body_pos_y[kNumBodies];
__device__ float r_Body_vel_x[kNumBodies];
__device__ float r_Body_vel_y[kNumBodies];
__device__ float r_Body_mass[kNumBodies];
int host_draw_counter;
float host_Body_pos_x[kNumBodies];
float host_Body_pos_y[kNumBodies];
float host_Body_vel_x[kNumBodies];
float host_Body_vel_y[kNumBodies];
float host_Body_mass[kNumBodies];
float host_Body_is_active[kNumBodies];

__device__ void Body_apply_force(IndexT id, IndexT other) {
    // Update `other`.
    if (other != id) {
        float dist;
        float F;
    dist = d_bodies[other]->computeDistance(d_bodies[id]);
    F     = d_bodies[other]->computeForce(d_bodies[id], dist);
        d_bodies[other]->updateForceX(d_bodies[id], F);
        d_bodies[other]->updateForceY(d_bodies[id], F);
    }
}

__device__ void Body_compute_force(IndexT id) {
    d_bodies[id]->initForce();

    // device_do
    for (IndexT i = 0; i < kNumBodies; ++i) {
        if (d_bodies[i]->is_active) {
            Body_apply_force(i, id);
        }
    }
}

__device__ void Body_update(IndexT id) {
    d_bodies[id]->updateVelX();
    d_bodies[id]->updateVelY();
    d_bodies[id]->updatePosX();
    d_bodies[id]->updatePosY();
    float idposx = d_bodies[id]->PosX();
    float idposy = d_bodies[id]->PosY();
    float idvelx = d_bodies[id]->VelX();
    float idvely = d_bodies[id]->VelY();

    if (idposx < -1 || idposx > 1) {
        d_bodies[id]->set_VelX(-idvelx);
    }

    if (idposy < -1 || idposy > 1) {
        d_bodies[id]->set_VelY(-idvely);
    }
}
// __device__ void Body_update(IndexT id) {
//   d_bodies[id]->vel_x += d_bodies[id]->force_x*kTimeInterval /
//   d_bodies[id]->mass; d_bodies[id]->vel_y +=
//   d_bodies[id]->force_y*kTimeInterval / d_bodies[id]->mass;
//   d_bodies[id]->pos_x += d_bodies[id]->vel_x*kTimeInterval;
//   d_bodies[id]->pos_y += d_bodies[id]->vel_y*kTimeInterval;

//   if (d_bodies[id]->pos_x < -1 || d_bodies[id]->pos_x > 1) {
//     d_bodies[id]->vel_x = -d_bodies[id]->vel_x;
//   }

//   if (d_bodies[id]->pos_y < -1 || d_bodies[id]->pos_y > 1) {
//     d_bodies[id]->vel_y = -d_bodies[id]->vel_y;
//   }
// }
__device__ void Body_check_merge_into_this(IndexT id, IndexT other) {
    // Only merge into larger body.
    bool cond1 = d_bodies[other]->get_incoming_merge();
    float othermass = d_bodies[other]->get_mass();
    float idmass = d_bodies[id]->get_mass();

    if (!cond1 && idmass > othermass) {
        float dist_square = d_bodies[id]->computeDistance(d_bodies[other]);
        dist_square *= dist_square;
        if (dist_square < kMergeThreshold * kMergeThreshold) {
            // Try to merge this one.
            // There is a race condition here: Multiple threads may try to merge
            // this body.
            d_bodies[id]->set_merge_target(other);
            d_bodies[other]->set_incoming_merge(true);
        }
    }
}

__device__ void Body_initialize_merge(IndexT id) {
    d_bodies[id]->set_merge_target(kNullptr);
    d_bodies[id]->set_incoming_merge(false);
    d_bodies[id]->set_is_successful_merge(false);
}

__device__ void Body_prepare_merge(IndexT id) {
    // device_do
    for (IndexT i = 0; i < kNumBodies; ++i) {
        if (d_bodies[i]->active()) {
            Body_check_merge_into_this(i, id);
        }
    }
}

// __device__ void Body_update_merge(IndexT id) {
//   IndexT m = d_bodies[id]->merge_target;
//   if (m != kNullptr) {
//     if (d_bodies[m]->merge_target == kNullptr) {
//       // Perform merge.
//       float new_mass = d_bodies[id]->mass + d_bodies[m]->mass;
//       float new_vel_x = (d_bodies[id]->vel_x*d_bodies[id]->mass
//                          + d_bodies[m]->vel_x*d_bodies[m]->mass) / new_mass;
//       float new_vel_y = (d_bodies[id]->vel_y*d_bodies[id]->mass
//                          + d_bodies[m]->vel_y*d_bodies[m]->mass) / new_mass;
//       d_bodies[m]->mass = new_mass;
//       d_bodies[m]->vel_x = new_vel_x;
//       d_bodies[m]->vel_y = new_vel_y;
//       d_bodies[m]->pos_x = (d_bodies[id]->pos_x + d_bodies[m]->pos_x) / 2;
//       d_bodies[m]->pos_y = (d_bodies[id]->pos_y + d_bodies[m]->pos_y) / 2;

//       d_bodies[id]->successful_merge = true;
//     }
//   }
// }

__device__ void Body_update_merge(IndexT id) {
    IndexT m = d_bodies[id]->get_merge_target();
    if (m != kNullptr) {
        if (d_bodies[m]->get_merge_target() == kNullptr) {
            // Perform merge.
            float idmass = d_bodies[id]->get_mass();
            float mmass = d_bodies[m]->get_mass();

            float new_mass = idmass + mmass;
            float idvelx = d_bodies[id]->VelX();
            float idvely = d_bodies[id]->VelY();

            float mvelx = d_bodies[m]->VelX();

            float mvely = d_bodies[m]->VelY();

            float new_vel_x = (idvelx * idmass + mvelx * mmass) / new_mass;
            float new_vel_y = (idvely * idmass + mvely * mmass) / new_mass;
            d_bodies[m]->set_mass(new_mass);
            d_bodies[m]->set_VelX(new_vel_x);
            d_bodies[m]->set_VelY(new_vel_y);
            float idposx = d_bodies[id]->PosX();
            float idposy = d_bodies[id]->PosY();
            float mposx = d_bodies[m]->PosX();
            float mposy = d_bodies[m]->PosY();
            d_bodies[m]->set_PosX((idposx + mposx) / 2);
            d_bodies[m]->set_PosY((idposy + mposy) / 2);

            d_bodies[id]->set_is_successful_merge(true);
        }
    }
}

__device__ void Body_delete_merged(IndexT id) {
    if (d_bodies[id]->get_is_successful_merge()) {
        d_bodies[id]->set_active(false);
    }
}

__device__ void Body_add_to_draw_array(IndexT id) {
    int idx = atomicAdd(&r_draw_counter, 1);
    r_Body_pos_x[idx] = d_bodies[id]->pos_x;
    r_Body_pos_y[idx] = d_bodies[id]->pos_y;
    r_Body_vel_x[idx] = d_bodies[id]->vel_x;
    r_Body_vel_y[idx] = d_bodies[id]->vel_y;
    r_Body_mass[idx] = d_bodies[id]->mass;
}

__device__ void new_Body(IndexT id, float pos_x, float pos_y, float vel_x,
                         float vel_y, float mass) {
    d_bodies[id]->pos_x = pos_x;
    d_bodies[id]->pos_y = pos_y;
    d_bodies[id]->vel_x = vel_x;
    d_bodies[id]->vel_y = vel_y;
    d_bodies[id]->mass = mass;
    d_bodies[id]->is_active = true;
}
__global__ void kernel_initialize_bodies() {
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    curandState rand_state;
    curand_init(kSeed, tid, 0, &rand_state);
    for (int id = tid; id < kNumBodies; id += blockDim.x * gridDim.x) {
        d_bodies[id] = new Body();
        assert(d_bodies[id] != NULL);

        new_Body(id,
                 /*pos_x=*/2 * curand_uniform(&rand_state) - 1,
                 /*pos_y=*/2 * curand_uniform(&rand_state) - 1,
                 /*vel_x=*/(curand_uniform(&rand_state) - 0.5) / 1000,
                 /*vel_y=*/(curand_uniform(&rand_state) - 0.5) / 1000,
                 /*mass=*/(curand_uniform(&rand_state) / 2 + 0.5) * kMaxMass);
    }
}

__global__ void kernel_reset_draw_counters() { r_draw_counter = 0; }

template <void (*func)(IndexT)>
__global__ void parallel_do() {
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    for (int id = tid; id < kNumBodies; id += blockDim.x * gridDim.x) {
        if (d_bodies[id]->active()) {
            func(id);
        }
    }
}

template <void (*func)(IndexT)>
__global__ void parallel_init() {
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    for (int id = tid; id < kNumBodies; id += blockDim.x * gridDim.x) {
        if (d_bodies[id]->active()) {
            func(id);
        }
    }
}

void transfer_data() {
    // Extract data from SoaAlloc data structure.
    kernel_reset_draw_counters<<<1, 1>>>();
    gpuErrchk(cudaDeviceSynchronize());
    parallel_init<&Body_add_to_draw_array><<<kBlocks, kThreads>>>();
    gpuErrchk(cudaDeviceSynchronize());

    // Copy data to host.
    cudaMemcpyFromSymbol(host_Body_pos_x, r_Body_pos_x,
                         sizeof(float) * kNumBodies, 0, cudaMemcpyDeviceToHost);
    cudaMemcpyFromSymbol(host_Body_pos_y, r_Body_pos_y,
                         sizeof(float) * kNumBodies, 0, cudaMemcpyDeviceToHost);
    cudaMemcpyFromSymbol(host_Body_vel_x, r_Body_vel_x,
                         sizeof(float) * kNumBodies, 0, cudaMemcpyDeviceToHost);
    cudaMemcpyFromSymbol(host_Body_vel_y, r_Body_vel_y,
                         sizeof(float) * kNumBodies, 0, cudaMemcpyDeviceToHost);
    cudaMemcpyFromSymbol(host_Body_mass, r_Body_mass,
                         sizeof(float) * kNumBodies, 0, cudaMemcpyDeviceToHost);
    cudaMemcpyFromSymbol(&host_draw_counter, r_draw_counter, sizeof(int), 0,
                         cudaMemcpyDeviceToHost);
}

int checksum() {
    transfer_data();
    int result = 0;

    for (int i = 0; i < kNumBodies; ++i) {
        int Body_checksum =
            static_cast<int>(
                (host_Body_pos_x[i] * 1000 + host_Body_pos_y[i] * 2000 +
                 host_Body_vel_x[i] * 3000 + host_Body_vel_y[i] * 4000)) %
            123456;
        result += Body_checksum;
    }

    return result;
}

int main(int /*argc*/, char ** /*argv*/) {
#ifdef OPTION_RENDER
    init_renderer();
#endif  // OPTION_RENDER

    // Allocate memory.

    cudaMallocManaged(&d_bodies, sizeof(Body *) * kNumBodies);

    // Allocate and create Body objects.
    kernel_initialize_bodies<<<128, 128>>>();
    gpuErrchk(cudaDeviceSynchronize());

#ifdef OPTION_RENDER
    // Compute max_mass.
    float max_mass = 0.0f;
    transfer_data();

    for (int i = 0; i < host_draw_counter; ++i) {
        max_mass += host_Body_mass[i];
    }
#endif  // OPTION_RENDER

    auto time_start = std::chrono::system_clock::now();

    for (int i = 0; i < kIterations; ++i) {
        printf("%i\n", i);
        parallel_do<&Body_compute_force><<<kBlocks, kThreads>>>();
        gpuErrchk(cudaDeviceSynchronize());

        parallel_do<&Body_update><<<kBlocks, kThreads>>>();
        gpuErrchk(cudaDeviceSynchronize());

        parallel_do<&Body_initialize_merge><<<kBlocks, kThreads>>>();
        gpuErrchk(cudaDeviceSynchronize());

        parallel_do<&Body_prepare_merge><<<kBlocks, kThreads>>>();
        gpuErrchk(cudaDeviceSynchronize());

        parallel_do<&Body_update_merge><<<kBlocks, kThreads>>>();
        gpuErrchk(cudaDeviceSynchronize());

        parallel_do<&Body_delete_merged><<<kBlocks, kThreads>>>();
        gpuErrchk(cudaDeviceSynchronize());

    }

    auto time_end = std::chrono::system_clock::now();
    auto elapsed = time_end - time_start;
    auto micros =
        std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();

#ifndef NDEBUG
    printf("Checksum: %i\n", checksum());
    printf("#bodies: %i\n", host_draw_counter);
#endif  // NDEBUG

    printf("%lu\n", micros);

    // Free memory

#ifdef OPTION_RENDER
    close_renderer();
#endif  // OPTION_RENDER

    return 0;
}
