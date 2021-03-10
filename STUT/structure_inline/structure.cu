
#include "structure.h"
#define gpuErrchk(ans)                                                         \
  { gpuAssert((ans), __FILE__, __LINE__); }

inline void gpuAssert(cudaError_t code, const char *file, int line,
                      bool abort = true) {
  if (code != cudaSuccess) {
    fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file,
            line);
    if (abort)
      exit(code);
  }
}
static const int kThreads = 256;

using IndexT = int;

__managed__ Node **dev_nodes;
__managed__ Spring **dev_springs;
__device__ void new_NodeBase(Node *node, float pos_x, float pos_y) {
  node->pos_x = pos_x;
  node->pos_y = pos_y;
  node->num_springs = 0;
  node->type = kTypeNodeBase;

  for (int i = 0; i < kMaxDegree; ++i) {
    node->springs[i] = NULL;
  }
}

__device__ void new_AnchorNode(Node *node, float pos_x, float pos_y) {
  new_NodeBase(node, pos_x, pos_y);
  node->type = kTypeAnchorNode;
}

__device__ void new_AnchorPullNode(Node *node, float pos_x, float pos_y,
                                   float vel_x, float vel_y) {
  new_AnchorNode(node, pos_x, pos_y);
  node->vel_x = vel_x;
  node->vel_y = vel_y;
  node->type = kTypeAnchorPullNode;
}

__device__ void new_Node(Node *node, float pos_x, float pos_y, float mass) {
  new_NodeBase(node, pos_x, pos_y);
  node->mass = mass;
  node->type = kTypeNode;
}

// __device__ float NodeBase_distance_to(Node *node, Node *other) {
//   float dx = node->pos_x - other->pos_x;
//   float dy = node->pos_y - other->pos_y;
//   float dist_sq = dx * dx + dy * dy;
//   return sqrt(dist_sq);
// }

__device__ void NodeBase_add_spring(Node *node, Spring *spring) {
  assert(node != NULL);

  int idx = atomicAdd(&node->num_springs, 1);
  assert(idx + 1 <= kMaxDegree);
  node->springs[idx] = spring;

  // assert(spring->p1 == node || spring->p2 == node);
}

__device__ void new_Spring(Spring *spring, Node *p1, Node *p2,
                           float spring_factor, float max_force) {
  spring->is_active = true;
  spring->p1 = p1;
  spring->p2 = p2;
  spring->factor = spring_factor;
  spring->force = 0.0f;
  spring->max_force = max_force;
  spring->initial_length = p1->distance_to(p2);
  spring->delete_flag = false;
  // if (!(spring->initial_length > 0.0f))
  //   printf("%f \n", spring->initial_length);
  assert(spring->initial_length > 0.0f);

  NodeBase_add_spring(p1, spring);
  NodeBase_add_spring(p2, spring);
}

__device__ void NodeBase_remove_spring(Node *node, Spring *spring) {
  for (int i = 0; i < kMaxDegree; ++i) {
    if (node->springs[i] == spring) {
      node->springs[i] = NULL;
      if (atomicSub(&node->num_springs, 1) == 1) {
        // Deleted last spring.
        node->type = 0;
      }
      return;
    }
  }

  // Spring not found.
  assert(false);
}

__device__ void AnchorPullNode_pull(Node *node) {
  node->pos_x += node->vel_x * kDt;
  node->pos_y += node->vel_y * kDt;
}

__device__ void Spring_self_destruct(Spring *spring) {
  NodeBase_remove_spring(spring->get_p1(), spring);
  NodeBase_remove_spring(spring->get_p2(), spring);
  spring->is_active = false;
}

__device__ void Spring_compute_force(Spring *spring) {
  float dist = spring->get_p1()->distance_to(spring->get_p2());
  float displacement = max(0.0f, dist - spring->get_init_len());
  spring->update_force(displacement);

  if (spring->is_max_force()) {
    spring->get_p1()->remove_spring(spring);
    spring->get_p2()->remove_spring(spring);
    spring->deactivate();
    // Spring_self_destruct(spring);
  }
}

__device__ void Node_move(Node *node) {
  float force_x = 0.0f;
  float force_y = 0.0f;

  for (int i = 0; i < kMaxDegree; ++i) {
    Spring *s = node->spring(i);

    if (s != NULL) {
      Node *from;
      Node *to;

      if (s->get_p1() == node) {
        from = node;
        to = s->get_p2();
      } else {
        assert(s->get_p2() == node);
        from = node;
        to = s->get_p1( );
      }

      // Calculate unit vector.

      float dist = to->distance_to(from);
      float unit_x = to->unit_x(from, dist);
      float unit_y = to->unit_y(from, dist);

      // Apply force.
      force_x += unit_x * s->get_force();
      force_y += unit_y * s->get_force();
    }
  }

  // Calculate new velocity and position.
  node->update_vel_x(force_x);
  node->update_vel_y(force_y);
  node->update_pos_x(force_x);
  node->update_pos_y(force_y);
}

__device__ void NodeBase_initialize_bfs(Node *node) {
  if (node->type == kTypeAnchorNode) {
    node->set_distance(0);
  } else {
    node->set_distance(kMaxDistance); // should be int_max
  }
}

__device__ bool dev_bfs_continue;

__device__ void NodeBase_bfs_visit(Node *node, int distance) {
  if (distance == node->get_distance()) {
    // Continue until all vertices were visited.
    dev_bfs_continue = true;

    for (int i = 0; i < kMaxDegree; ++i) {
      Spring *spring = node->spring(i);

      if (spring != NULL) {
        // Find neighboring vertices.
        Node *n;
        if (node == spring->get_p1()) {
          n = spring->get_p2();
        } else {
          n = spring->get_p1();
        }

        if (n->get_distance() == kMaxDistance) {
          // Set distance on neighboring vertex if unvisited.
          n->set_distance(distance + 1);
        }
      }
    }
  }
}
__device__ void Spring_bfs_delete(Spring *spring) {
  if (spring->delete_flag) {
    spring->get_p1()->remove_spring(spring);
    spring->get_p2()->remove_spring(spring);
    spring->deactivate();
  }
}

__device__ void NodeBase_bfs_set_delete_flags(Node *node) {
  if (node->distance == kMaxDistance) { // should be int_max
    for (int i = 0; i < kMaxDegree; ++i) {
      Spring *spring = node->spring(i);
      if (spring != NULL) {
        spring->delete_flag = true;
        //Spring_bfs_delete(spring);
      }
    }
  }
}

// Only for rendering and checksum computation.
__device__ int dev_num_springs;
__device__ SpringInfo dev_spring_info[kMaxSprings];
int host_num_springs;
SpringInfo host_spring_info[kMaxSprings];

__device__ void Spring_add_to_rendering_array(Spring *spring) {
  int idx = atomicAdd(&dev_num_springs, 1);
  dev_spring_info[idx].p1_x = spring->p1->pos_x;
  dev_spring_info[idx].p1_y = spring->p1->pos_y;
  dev_spring_info[idx].p2_x = spring->p2->pos_x;
  dev_spring_info[idx].p2_y = spring->p2->pos_y;
  dev_spring_info[idx].force = spring->force;
  dev_spring_info[idx].max_force = spring->max_force;
}

__global__ void kernel_AnchorPullNode_pull() {
  for (int i = threadIdx.x + blockDim.x * blockIdx.x; i < kMaxNodes;
       i += blockDim.x * gridDim.x) {
    if (dev_nodes[i]->type == kTypeAnchorPullNode) {
      dev_nodes[i]->pull();
    }
  }
}

__global__ void kernel_Node_move() {
  for (int i = threadIdx.x + blockDim.x * blockIdx.x; i < kMaxNodes;
       i += blockDim.x * gridDim.x) {
    if (dev_nodes[i]->type == kTypeNode) {
      Node_move(dev_nodes[i]);
    }
  }
}

__global__ void kernel_NodeBase_initialize_bfs() {
  for (int i = threadIdx.x + blockDim.x * blockIdx.x; i < kMaxNodes;
       i += blockDim.x * gridDim.x) {
    if (dev_nodes[i]->type != 0) {
      NodeBase_initialize_bfs(dev_nodes[i]);
    }
  }
}

__global__ void kernel_NodeBase_bfs_visit(int dist) {
  for (int i = threadIdx.x + blockDim.x * blockIdx.x; i < kMaxNodes;
       i += blockDim.x * gridDim.x) {
    if (dev_nodes[i]->type != 0) {
      NodeBase_bfs_visit(dev_nodes[i], dist);
    }
  }
}

__global__ void kernel_NodeBase_bfs_set_delete_flags() {
  for (int i = threadIdx.x + blockDim.x * blockIdx.x; i < kMaxNodes;
       i += blockDim.x * gridDim.x) {
    if (dev_nodes[i]->type != 0) {
      NodeBase_bfs_set_delete_flags(dev_nodes[i]);
    }
  }
}

__global__ void kernel_Spring_compute_force() {
  for (int i = threadIdx.x + blockDim.x * blockIdx.x; i < kMaxSprings;
       i += blockDim.x * gridDim.x) {
    if (dev_springs[i]->get_is_active()) {
      Spring_compute_force(dev_springs[i]);
    }
  }
}

__global__ void kernel_Spring_bfs_delete() {
  for (int i = threadIdx.x + blockDim.x * blockIdx.x; i < kMaxSprings;
       i += blockDim.x * gridDim.x) {
    if (dev_springs[i]->get_is_active()) {
      Spring_bfs_delete(dev_springs[i]);
    }
  }
}

__global__ void kernel_Spring_add_to_rendering_array() {
  for (int i = threadIdx.x + blockDim.x * blockIdx.x; i < kMaxSprings;
       i += blockDim.x * gridDim.x) {
    if (dev_springs[i]->get_is_active()) {
      Spring_add_to_rendering_array(dev_springs[i]);
    }
  }
}

__global__ void kernel_initialize_nodes() {
  for (int i = threadIdx.x + blockDim.x * blockIdx.x; i < kMaxNodes;
       i += blockDim.x * gridDim.x) {
    dev_nodes[i] = new Node();

    assert(dev_nodes[i] != NULL);
    dev_nodes[i]->type = 0;
  }
}

__global__ void kernel_initialize_springs() {
  for (int i = threadIdx.x + blockDim.x * blockIdx.x; i < kMaxSprings;
       i += blockDim.x * gridDim.x) {

    dev_springs[i] = new Spring();
    assert(dev_springs[i] != NULL);
    dev_springs[i]->is_active = false;
  }
}

void transfer_data() {
  int zero = 0;
  cudaMemcpyToSymbol(dev_num_springs, &zero, sizeof(int), 0,
                     cudaMemcpyHostToDevice);
  gpuErrchk(cudaDeviceSynchronize());

  kernel_Spring_add_to_rendering_array<<<128, 128>>>();
  gpuErrchk(cudaDeviceSynchronize());

  cudaMemcpyFromSymbol(&host_num_springs, dev_num_springs, sizeof(int), 0,
                       cudaMemcpyDeviceToHost);
  gpuErrchk(cudaDeviceSynchronize());

  cudaMemcpyFromSymbol(host_spring_info, dev_spring_info,
                       sizeof(SpringInfo) * host_num_springs, 0,
                       cudaMemcpyDeviceToHost);
  gpuErrchk(cudaDeviceSynchronize());
}

float checksum() {
  transfer_data();
  float result = 0.0f;

  for (int i = 0; i < host_num_springs; ++i) {
    result += host_spring_info[i].p1_x * host_spring_info[i].p2_y *
              host_spring_info[i].force;
  }

  return result;
}

void compute() {
  kernel_Spring_compute_force<<<(kMaxSprings + kThreads - 1) / kThreads,
                                kThreads>>>();
  gpuErrchk(cudaDeviceSynchronize());

  kernel_Node_move<<<(kMaxNodes + kThreads - 1) / kThreads, kThreads>>>();
  gpuErrchk(cudaDeviceSynchronize());
}

void bfs_and_delete() {
  // Perform BFS to check reachability.
  kernel_NodeBase_initialize_bfs<<<(kMaxNodes + kThreads - 1) / kThreads,
                                   kThreads>>>();
  gpuErrchk(cudaDeviceSynchronize());

  for (int i = 0; i < kMaxDistance; ++i) {
    bool continue_flag = false;
    cudaMemcpyToSymbol(dev_bfs_continue, &continue_flag, sizeof(bool), 0,
                       cudaMemcpyHostToDevice);

    kernel_NodeBase_bfs_visit<<<(kMaxNodes + kThreads - 1) / kThreads,
                                kThreads>>>(i);
    gpuErrchk(cudaDeviceSynchronize());

    cudaMemcpyFromSymbol(&continue_flag, dev_bfs_continue, sizeof(bool), 0,
                         cudaMemcpyDeviceToHost);

    if (!continue_flag)
      break;
  }

  // Delete springs (and nodes).
  kernel_NodeBase_bfs_set_delete_flags<<<(kMaxNodes + kThreads - 1) / kThreads,
                                         kThreads>>>();
  gpuErrchk(cudaDeviceSynchronize());

  kernel_Spring_bfs_delete<<<(kMaxSprings + kThreads - 1) / kThreads,
                             kThreads>>>();
  gpuErrchk(cudaDeviceSynchronize());
}

void step() {
  kernel_AnchorPullNode_pull<<<(kMaxNodes + kThreads - 1) / kThreads,
                               kThreads>>>();
  gpuErrchk(cudaDeviceSynchronize());

  for (int i = 0; i < kNumComputeIterations; ++i) {
    compute();
  }

  bfs_and_delete();
}

void initialize_memory() {
  kernel_initialize_nodes<<<(kMaxNodes + kThreads - 1) / kThreads,
                            kThreads>>>();
  gpuErrchk(cudaDeviceSynchronize());

  kernel_initialize_springs<<<(kMaxSprings + kThreads - 1) / kThreads,
                              kThreads>>>();
  gpuErrchk(cudaDeviceSynchronize());
}

__device__ IndexT dev_tmp_nodes[kMaxNodes];
__device__ IndexT dev_node_counter;

__global__ void kernel_create_nodes(DsNode *nodes, int num_nodes) {
  for (int i = threadIdx.x + blockDim.x * blockIdx.x; i < num_nodes;
       i += blockDim.x * gridDim.x) {
    int idx = atomicAdd(&dev_node_counter, 1);
    assert(dev_nodes != NULL);
    dev_tmp_nodes[i] = idx;

    assert(dev_nodes[idx] != NULL);
    if (nodes[i].type == kTypeNode) {
      new_Node(dev_nodes[idx], nodes[i].pos_x, nodes[i].pos_y, nodes[i].mass);
    } else if (nodes[i].type == kTypeAnchorPullNode) {
      new_AnchorPullNode(dev_nodes[idx], nodes[i].pos_x, nodes[i].pos_y,
                         nodes[i].vel_x, nodes[i].vel_y);
    } else if (nodes[i].type == kTypeAnchorNode) {
      new_AnchorNode(dev_nodes[idx], nodes[i].pos_x, nodes[i].pos_y);
    } else {
      assert(false);
    }
  }
}

__global__ void kernel_create_springs(DsSpring *springs, int num_springs) {
  for (int i = threadIdx.x + blockDim.x * blockIdx.x; i < num_springs;
       i += blockDim.x * gridDim.x) {

    assert(dev_springs[i] != nullptr);

    new_Spring(dev_springs[i], dev_nodes[dev_tmp_nodes[springs[i].p1]],
               dev_nodes[dev_tmp_nodes[springs[i].p2]],
               springs[i].spring_factor, springs[i].max_force);
    // printf("%p \n", dev_springs[i]);
  }
}

void load_dataset(Dataset &dataset) {
  DsNode *host_nodes;
  cudaMalloc(&host_nodes, sizeof(DsNode) * dataset.nodes.size());
  cudaMemcpy(host_nodes, dataset.nodes.data(),
             sizeof(DsNode) * dataset.nodes.size(), cudaMemcpyHostToDevice);

  DsSpring *host_springs;
  cudaMalloc(&host_springs, sizeof(DsSpring) * dataset.springs.size());
  cudaMemcpy(host_springs, dataset.springs.data(),
             sizeof(DsSpring) * dataset.springs.size(), cudaMemcpyHostToDevice);
  gpuErrchk(cudaDeviceSynchronize());

  IndexT zero = 0;
  cudaMemcpyToSymbol(dev_node_counter, &zero, sizeof(IndexT), 0,
                     cudaMemcpyHostToDevice);
  gpuErrchk(cudaDeviceSynchronize());
  assert(dataset.nodes.size() == kMaxNodes);

  // kernel_create_nodes1<<<(kMaxNodes + kThreads - 1) / kThreads, kThreads>>>(
  //     host_nodes, dataset.nodes.size());
  kernel_create_nodes<<<(kMaxNodes + kThreads - 1) / kThreads, kThreads>>>(
      host_nodes, dataset.nodes.size());
  gpuErrchk(cudaDeviceSynchronize());
  // kernel_create_spring1<<<(kMaxSprings + kThreads - 1) / kThreads,
  // kThreads>>>(
  //   host_nodes, dataset.springs.size());
  kernel_create_springs<<<(kMaxSprings + kThreads - 1) / kThreads, kThreads>>>(
      host_springs, dataset.springs.size());
  gpuErrchk(cudaDeviceSynchronize());

  cudaFree(host_nodes);
  cudaFree(host_springs);
}

int main(int /*argc*/, char ** /*argv*/) {
  // Allocate memory.
  using namespace std::chrono;

  cudaDeviceSetLimit(cudaLimitMallocHeapSize, 4ULL * 1024 * 1024 * 1024);
  high_resolution_clock::time_point t1 = high_resolution_clock::now();
  cudaMalloc(&dev_nodes, sizeof(Node *) * kMaxNodes);
  // cudaMemcpyToSymbol(dev_nodes, &host_nodes, sizeof(Node *), 0,
  //                    cudaMemcpyHostToDevice);
  assert(dev_nodes != NULL);
  // printf("%p\n", dev_nodes);

  // Spring *host_springs;
  cudaMalloc(&dev_springs, sizeof(Spring *) * kMaxSprings);
  // cudaMemcpyToSymbol(dev_springs, &host_springs, sizeof(Spring *), 0,
  //                    cudaMemcpyHostToDevice);
  initialize_memory();
  high_resolution_clock::time_point t2 = high_resolution_clock::now();
  duration<double> alloc_time = duration_cast<duration<double>>(t2 - t1);
  printf("alloc_time : %f \n",alloc_time.count() );

  Dataset dataset;
  random_dataset(dataset);
  load_dataset(dataset);

  auto time_start = std::chrono::system_clock::now();

  for (int i = 0; i < kNumSteps; ++i) {
#ifndef NDEBUG
    printf("%i\n", i);
#endif // NDEBUG
    step();
  }

  auto time_end = std::chrono::system_clock::now();
  auto elapsed = time_end - time_start;
  auto micros =
      std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();

  printf("%lu\n", micros);

#ifndef NDEBUG
  printf("Checksum: %f\n", checksum());
#endif // NDEBUG
}
