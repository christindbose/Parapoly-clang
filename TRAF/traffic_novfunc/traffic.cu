#include "traffic.h"

static const int kNumBlockSize = 256;

static const char kCellTypeNormal = 1;
static const char kCellTypeProducer = 2;

using IndexT = int;
using CellPointerT = IndexT;
#include "../dataset.h"

__managed__ Cell **dev_cells;

// Need 2 arrays of both, so we can swap.
__device__ int *d_Car_active;
__device__ int *d_Car_active_2;
__managed__ Car **dev_cars;
__managed__ Car **dev_cars_2;

// For prefix sum array compaction.
__device__ int *d_prefix_sum_temp;
__device__ int *d_prefix_sum_output;
int *h_prefix_sum_temp;
int *h_prefix_sum_output;
int *h_Car_active;
int *h_Car_active_2;

__device__ int d_num_cells;
__device__ int d_num_cars;
__device__ int d_num_cars_2;
int host_num_cells;
int host_num_cars;

// TODO: Consider migrating to SoaAlloc.
TrafficLight *h_traffic_lights;
__managed__ TrafficLight **d_traffic_lights;

// Only for rendering.
__device__ int dev_num_cells;
__device__ float *dev_Cell_pos_x;
__device__ float *dev_Cell_pos_y;
__device__ bool *dev_Cell_occupied;
float *host_Cell_pos_x;
float *host_Cell_pos_y;
bool *host_Cell_occupied;
float *host_data_Cell_pos_x;
float *host_data_Cell_pos_y;
bool *host_data_Cell_occupied;

__device__ void Car_step_extend_path(IndexT self) {
  Cell *cell = dev_cars[self]->get_position();
  Cell *next_cell;

  for (int i = 0; i < dev_cars[self]->get_velocity(); ++i) {
    bool cond = cell->get_is_target();
    if (cell->is_sink() || cond) {
      break;
    }

    next_cell = dev_cars[self]->next_step(cell);
    assert(next_cell != cell);

    if (!next_cell->is_free())
      break;

    cell = next_cell;
    dev_cars[self]->set_path(cell, i);
    int path_len = dev_cars[self]->get_path_length();
    dev_cars[self]->set_path_length(path_len + 1);
  }
  int path_len = dev_cars[self]->get_path_length();
  dev_cars[self]->set_velocity(path_len);
}

__device__ void Car_step_constraint_velocity(IndexT self) {
  // This is actually only needed for the very first iteration, because a car
  // may be positioned on a traffic light cell.
  int vel = dev_cars[self]->get_velocity();
  Cell *cell = dev_cars[self]->get_position();
  if (vel > cell->get_current_max_velocity()) {
    int max_velocity = cell->get_current_max_velocity();
    dev_cars[self]->set_velocity(max_velocity);
  }

  int path_index = 0;
  int distance = 1;

  while (distance <= dev_cars[self]->get_velocity()) {
    // Invariant: Movement of up to `distance - 1` many cells at `velocity_`
    //            is allowed.
    // Now check if next cell can be entered.
    Cell *next_cell = dev_cars[self]->get_path(path_index);

    // Avoid collision.
    if (!next_cell->is_free()) {
      // Cannot enter cell.
      --distance;
      dev_cars[self]->set_velocity(distance);
      break;
    } // else: Can enter next cell.

    int curr_vel = dev_cars[self]->get_velocity();

    if (curr_vel > next_cell->get_current_max_velocity()) {
      // Car is too fast for this cell.
      if (next_cell->get_current_max_velocity() > distance - 1) {
        // Even if we slow down, we would still make progress.
        int max = next_cell->get_current_max_velocity();
        dev_cars[self]->set_velocity(max);
      } else {
        // Do not enter the next cell.
        --distance;
        assert(distance >= 0);

        dev_cars[self]->set_velocity(distance);
        break;
      }
    }

    ++distance;
    ++path_index;
  }

  --distance;

#ifndef NDEBUG
  for (int i = 0; i < dev_cars[self]->get_velocity(); ++i) {
    assert(dev_cars[self]->get_path(i)->is_free());
    assert(i == 0 ||
           dev_cars[self]->get_path(i - 1) != dev_cars[self]->get_path(i));
  }
  // TODO: Check why the cast is necessary.
  assert(distance <= dev_cars[self]->get_velocity());
#endif // NDEBUG
}

__device__ void Car_step_move(IndexT self) {
  Cell *cell = dev_cars[self]->get_position();
  for (int i = 0; i < dev_cars[self]->get_velocity(); ++i) {
    assert(dev_cars[self]->get_path(i) != cell);

    cell = dev_cars[self]->get_path(i);
    assert(cell->is_free());
    Cell *ptr = dev_cars[self]->get_position();
    ptr->release();
    cell->occupy(dev_cars[self]);
    dev_cars[self]->set_position(cell);
  }

  Cell *ptr = dev_cars[self]->get_position();
  bool cond = ptr->is_sink();
  if (cond || ptr->get_is_target()) {
    // Remove car from the simulation. Will be added again in the next
    // iteration.

    ptr->release();
    dev_cars[self]->set_position(nullptr);
    d_Car_active[self] = 0;
  }
}

__device__ void Car_step_slow_down(IndexT self) {
  // 20% change of slowdown.
  int vel = dev_cars[self]->get_velocity();
  if (dev_cars[self]->random_uni() < 0.2 && vel > 0) {

    dev_cars[self]->set_velocity(vel - 1);
  }
}

__device__ IndexT new_Car(int seed, IndexT cell, int max_velocity) {
  IndexT idx = atomicAdd(&d_num_cars, 1);
  assert(idx >= 0 && idx < kMaxNumCars);

  assert(!d_Car_active[idx]);
  dev_cars[idx]->set_position(dev_cells[cell]);
  dev_cars[idx]->set_path_length(0);
  dev_cars[idx]->set_velocity(0);
  dev_cars[idx]->set_max_velocity(max_velocity);
  d_Car_active[idx] = 1;

  assert(dev_cells[cell]->is_free());
  dev_cells[cell]->occupy(dev_cars[idx]);
  curand_init(seed, 0, 0, &dev_cars[idx]->random_state);

  return idx;
}

__device__ void ProducerCell_create_car(IndexT self) {
  assert(dev_cells[self]->type == kCellTypeProducer);
  if (dev_cells[self]->is_free()) {
    float r = curand_uniform(&dev_cells[self]->random_state);
    if (r < kCarAllocationRatio) {
      IndexT new_car = new_Car(
          /*seed=*/curand(&dev_cells[self]->random_state), /*cell=*/self,
          /*max_velocity=*/curand(&dev_cells[self]->random_state) %
                  (kMaxVelocity / 2) +
              kMaxVelocity / 2);
    }
  }
}

__device__ IndexT new_Cell(int max_velocity, float x, float y) {
  IndexT idx = atomicAdd(&d_num_cells, 1);

  dev_cells[idx]->car = nullptr;
  dev_cells[idx]->max_velocity = max_velocity;
  dev_cells[idx]->current_max_velocity = max_velocity;
  dev_cells[idx]->num_incoming = 0;
  dev_cells[idx]->num_outgoing = 0;
  dev_cells[idx]->x = x;
  dev_cells[idx]->y = y;
  dev_cells[idx]->is_target = false;
  dev_cells[idx]->type = kCellTypeNormal;

  return idx;
}

__device__ IndexT new_ProducerCell(int max_velocity, float x, float y,
                                   int seed) {
  IndexT idx = new_Cell(max_velocity, x, y);
  dev_cells[idx]->type = kCellTypeProducer;
  curand_init(seed, 0, 0, &dev_cells[idx]->random_state);

  return idx;
}

__global__ void kernel_traffic_light_step() {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < kNumIntersections;
       i += blockDim.x * gridDim.x) {
    if (d_traffic_lights[i]->get_num_cells() > 0) {

      int timer = d_traffic_lights[i]->get_timer();
      int phase_time = d_traffic_lights[i]->get_phase_time();
      d_traffic_lights[i]->set_timer((timer + 1) % phase_time);

      if (d_traffic_lights[i]->get_timer() == 0) {
        int phase = d_traffic_lights[i]->get_phase();
        assert(d_traffic_lights[i]->get_cell(phase) != nullptr);
        phase = d_traffic_lights[i]->get_phase();
        Cell *ptr = d_traffic_lights[i]->get_cell(phase);

        ptr->set_current_max_velocity(0);
        int phase_2 = d_traffic_lights[i]->get_phase();
        int num_cells = d_traffic_lights[i]->get_num_cells();
        d_traffic_lights[i]->set_phase((phase_2 + 1) % num_cells);
        phase_2 = d_traffic_lights[i]->get_phase();
        ptr = d_traffic_lights[i]->get_cell(phase_2);
        ptr->remove_speed_limit();
      }
    }
    // d_traffic_lights[i]->step();
  }
}

__global__ void kernel_create_nodes() {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < kNumIntersections;
       i += blockDim.x * gridDim.x) {
    curandState_t state;
    curand_init(i, 0, 0, &state);

    assert(d_nodes[i].x >= 0 && d_nodes[i].x <= 1);
    assert(d_nodes[i].y >= 0 && d_nodes[i].y <= 1);

    for (int j = 0; j < d_nodes[i].num_outgoing; ++j) {
      d_nodes[i].cell_out[j] = new_Cell(
          /*max_velocity=*/curand(&state) % (kMaxVelocity / 2) +
              kMaxVelocity / 2,
          d_nodes[i].x, d_nodes[i].y);
    }
  }
}

__device__ IndexT connect_intersections(IndexT from, Node *target,
                                        int incoming_idx,
                                        curandState_t &state) {
  // Create edge.
  float dx = target->x - dev_cells[from]->x;
  float dy = target->y - dev_cells[from]->y;
  float dist = sqrt(dx * dx + dy * dy);
  int steps = dist / kCellLength;
  float step_x = dx / steps;
  float step_y = dy / steps;
  IndexT prev = from;

  for (int j = 0; j < steps; ++j) {
    float new_x = dev_cells[from]->x + j * step_x;
    float new_y = dev_cells[from]->y + j * step_y;
    assert(new_x >= 0 && new_x <= 1);
    assert(new_y >= 0 && new_y <= 1);
    IndexT next;

    if (curand_uniform(&state) < kProducerRatio) {
      next = new_ProducerCell(dev_cells[prev]->max_velocity, new_x, new_y,
                              curand(&state));
    } else {
      next = new_Cell(dev_cells[prev]->max_velocity, new_x, new_y);
    }

    if (curand_uniform(&state) < kTargetRatio) {
      dev_cells[next]->set_target();
    }

    dev_cells[prev]->set_num_outgoing(1);
    dev_cells[prev]->set_outgoing(0, dev_cells[next]);
    dev_cells[next]->set_num_incoming(1);
    dev_cells[next]->set_incoming(0, dev_cells[prev]);

    prev = next;
  }

  // Connect to all outgoing nodes of target.
  dev_cells[prev]->set_num_outgoing(target->num_outgoing);
  for (int i = 0; i < target->num_outgoing; ++i) {
    IndexT next = target->cell_out[i];
    // num_incoming set later.
    dev_cells[prev]->set_outgoing(i, dev_cells[next]);
    dev_cells[next]->set_incoming(incoming_idx, dev_cells[prev]);
  }

  return prev;
}

__global__ void kernel_create_edges() {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < kNumIntersections;
       i += blockDim.x * gridDim.x) {
    curandState_t state;
    curand_init(i, 0, 0, &state);

    for (int k = 0; k < d_nodes[i].num_outgoing; ++k) {
      int target = d_nodes[i].node_out[k];
      int target_pos = d_nodes[i].node_out_pos[k];

      IndexT last = connect_intersections(d_nodes[i].cell_out[k],
                                          &d_nodes[target], target_pos, state);

      dev_cells[last]->set_current_max_velocity(0);
      d_nodes[target].cell_in[target_pos] = last;
    }
  }
}

__global__ void kernel_create_traffic_lights() {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < kNumIntersections;
       i += blockDim.x * gridDim.x) {
    new (d_traffic_lights[i]) TrafficLight(
        /*num_cells=*/d_nodes[i].num_incoming,
        /*phase_time=*/5);

    for (int j = 0; j < d_nodes[i].num_outgoing; ++j) {
      dev_cells[d_nodes[i].cell_out[j]]->set_num_incoming(
          d_nodes[i].num_incoming);
    }

    for (int j = 0; j < d_nodes[i].num_incoming; ++j) {
      d_traffic_lights[i]->set_cell(j, dev_cells[d_nodes[i].cell_in[j]]);
      dev_cells[d_nodes[i].cell_in[j]]->set_current_max_velocity(
          0); // Set to "red".
    }
  }
}

template <class Type, class TypeBase>
__global__ void device_alloc(TypeBase **ptr, int size) {

  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < size;
       i += blockDim.x * gridDim.x) {
    ptr[i] = new Type();
    assert(ptr[i] != nullptr);
  }
}
void create_street_network() {
  int zero = 0;
  cudaMemcpyToSymbol(dev_num_cells, &zero, sizeof(int), 0,
                     cudaMemcpyHostToDevice);
  cudaMalloc(&h_nodes, sizeof(Node) * kNumIntersections);
  cudaMemcpyToSymbol(d_nodes, &h_nodes, sizeof(Node *), 0,
                     cudaMemcpyHostToDevice);
  cudaMalloc(&d_traffic_lights, sizeof(TrafficLight *) * kNumIntersections);

  device_alloc<TrafficLight, TrafficLight>
      <<<(kNumIntersections + kNumBlockSize - 1) / kNumBlockSize,
         kNumBlockSize>>>(d_traffic_lights, kNumIntersections);

  gpuErrchk(cudaDeviceSynchronize());

  // Create basic structure on host.
  create_network_structure();

  kernel_create_nodes<<<(kNumIntersections + kNumBlockSize - 1) / kNumBlockSize,
                        kNumBlockSize>>>();
  gpuErrchk(cudaDeviceSynchronize());

  kernel_create_edges<<<(kNumIntersections + kNumBlockSize - 1) / kNumBlockSize,
                        kNumBlockSize>>>();
  gpuErrchk(cudaDeviceSynchronize());

  kernel_create_traffic_lights<<<(kNumIntersections + kNumBlockSize - 1) /
                                     kNumBlockSize,
                                 kNumBlockSize>>>();
  gpuErrchk(cudaDeviceSynchronize());

  // Allocate helper data structures for rendering.
  cudaMemcpyFromSymbol(&host_num_cells, d_num_cells, sizeof(int), 0,
                       cudaMemcpyDeviceToHost);
  cudaMalloc(&host_Cell_pos_x, sizeof(float) * host_num_cells);
  cudaMemcpyToSymbol(dev_Cell_pos_x, &host_Cell_pos_x, sizeof(float *), 0,
                     cudaMemcpyHostToDevice);
  cudaMalloc(&host_Cell_pos_y, sizeof(float) * host_num_cells);
  cudaMemcpyToSymbol(dev_Cell_pos_y, &host_Cell_pos_y, sizeof(float *), 0,
                     cudaMemcpyHostToDevice);
  cudaMalloc(&host_Cell_occupied, sizeof(bool) * host_num_cells);
  cudaMemcpyToSymbol(dev_Cell_occupied, &host_Cell_occupied, sizeof(bool *), 0,
                     cudaMemcpyHostToDevice);
  host_data_Cell_pos_x = (float *)malloc(sizeof(float) * host_num_cells);
  host_data_Cell_pos_y = (float *)malloc(sizeof(float) * host_num_cells);
  host_data_Cell_occupied = (bool *)malloc(sizeof(bool) * host_num_cells);

#ifndef NDEBUG
  printf("Number of cells: %i\n", host_num_cells);
#endif // NDEBUG
}

void step_traffic_lights() {
  // TODO: Consider migrating this to SoaAlloc.
  kernel_traffic_light_step<<<(kNumIntersections + kNumBlockSize - 1) /
                                  kNumBlockSize,
                              kNumBlockSize>>>();
  gpuErrchk(cudaDeviceSynchronize());
}

__device__ void Cell_add_to_rendering_array(IndexT self) {
  int idx = atomicAdd(&dev_num_cells, 1);
  dev_Cell_pos_x[idx] = dev_cells[self]->x;
  dev_Cell_pos_y[idx] = dev_cells[self]->y;
  dev_Cell_occupied[idx] = !dev_cells[self]->is_free();
}

__global__ void kernel_Cell_add_to_rendering_array() {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < d_num_cells;
       i += blockDim.x * gridDim.x) {
    Cell_add_to_rendering_array(i);
  }
}

void transfer_data() {
  int zero = 0;
  cudaMemcpyToSymbol(dev_num_cells, &zero, sizeof(int), 0,
                     cudaMemcpyHostToDevice);

  kernel_Cell_add_to_rendering_array<<<
      (host_num_cells + kNumBlockSize - 1) / kNumBlockSize, kNumBlockSize>>>();
  gpuErrchk(cudaDeviceSynchronize());

  cudaMemcpy(host_data_Cell_pos_x, host_Cell_pos_x,
             sizeof(float) * host_num_cells, cudaMemcpyDeviceToHost);
  cudaMemcpy(host_data_Cell_pos_y, host_Cell_pos_y,
             sizeof(float) * host_num_cells, cudaMemcpyDeviceToHost);
  cudaMemcpy(host_data_Cell_occupied, host_Cell_occupied,
             sizeof(bool) * host_num_cells, cudaMemcpyDeviceToHost);

  gpuErrchk(cudaDeviceSynchronize());
}

__global__ void kernel_ProducerCell_create_car() {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < d_num_cells;
       i += blockDim.x * gridDim.x) {
    if (dev_cells[i]->type == kCellTypeProducer) {
      ProducerCell_create_car(i);
    }
  }
}

__device__ void Car_step_prepare_path(IndexT self) {
  dev_cars[self]->step_initialize_iteration();
  dev_cars[self]->step_accelerate();
  Car_step_extend_path(self);
  Car_step_constraint_velocity(self);
  Car_step_slow_down(self);
}

__global__ void kernel_Car_step_prepare_path() {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < d_num_cars;
       i += blockDim.x * gridDim.x) {
    if (d_Car_active[i]) {
      Car_step_prepare_path(i);
    }
  }
}

__global__ void kernel_fill_car_indices() {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < d_num_cars;
       i += blockDim.x * gridDim.x) {
    d_Car_active[i] = 0;
    d_Car_active_2[i] = 0;
  }
}

__global__ void kernel_Car_step_move() {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < d_num_cars;
       i += blockDim.x * gridDim.x) {
    if (d_Car_active[i]) {
      Car_step_move(i);
    }
  }
}

__device__ int d_checksum;
__global__ void kernel_compute_checksum() {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < d_num_cars;
       i += blockDim.x * gridDim.x) {
    if (d_Car_active[i]) {
      atomicAdd(&d_checksum, 1);
    }
  }
}

int checksum() {
  int zero = 0;
  cudaMemcpyToSymbol(d_checksum, &zero, sizeof(int), 0, cudaMemcpyHostToDevice);
  kernel_compute_checksum<<<128, 128>>>();

  int result;
  cudaMemcpyFromSymbol(&result, d_checksum, sizeof(int), 0,
                       cudaMemcpyDeviceToHost);
  return result;
}

void step() {
  kernel_ProducerCell_create_car<<<
      (host_num_cells + kNumBlockSize - 1) / kNumBlockSize, kNumBlockSize>>>();
  gpuErrchk(cudaDeviceSynchronize());

  cudaMemcpyFromSymbol(&host_num_cars, d_num_cars, sizeof(int), 0,
                       cudaMemcpyDeviceToHost);
  step_traffic_lights();

  kernel_Car_step_prepare_path<<<
      (host_num_cars + kNumBlockSize - 1) / kNumBlockSize, kNumBlockSize>>>();
  gpuErrchk(cudaDeviceSynchronize());

  kernel_Car_step_move<<<(host_num_cars + kNumBlockSize - 1) / kNumBlockSize,
                         kNumBlockSize>>>();
  gpuErrchk(cudaDeviceSynchronize());
}

void allocate_memory() {

  cudaMalloc(&dev_cells, sizeof(Cell *) * kMaxNumCells);
  device_alloc<Cell, Cell>
      <<<(kMaxNumCells + kNumBlockSize - 1) / kNumBlockSize, kNumBlockSize>>>(
          dev_cells, kMaxNumCells);
  gpuErrchk(cudaDeviceSynchronize());
  cudaMalloc(&dev_cars, sizeof(Car *) * kMaxNumCars);
  cudaMalloc(&dev_cars_2, sizeof(Car *) * kMaxNumCars);
  device_alloc<Car, Car>
      <<<(kMaxNumCars + kNumBlockSize - 1) / kNumBlockSize, kNumBlockSize>>>(
          dev_cars, kMaxNumCars);
  gpuErrchk(cudaDeviceSynchronize());
  device_alloc<Car>
      <<<(kMaxNumCars + kNumBlockSize - 1) / kNumBlockSize, kNumBlockSize>>>(
          dev_cars_2, kMaxNumCars);
  gpuErrchk(cudaDeviceSynchronize());
  cudaMalloc(&h_Car_active, sizeof(int) * kMaxNumCars);
  cudaMemcpyToSymbol(d_Car_active, &h_Car_active, sizeof(int *), 0,
                     cudaMemcpyHostToDevice);

  // No place to use these
  // Car *h_cars_2;
  // cudaMalloc(&h_cars_2, sizeof(Car) * kMaxNumCars);
  // cudaMemcpyToSymbol(dev_cars_2, &h_cars_2, sizeof(Car *), 0,
  //                    cudaMemcpyHostToDevice);

  cudaMalloc(&h_Car_active_2, sizeof(int) * kMaxNumCars);
  cudaMemcpyToSymbol(d_Car_active_2, &h_Car_active_2, sizeof(int *), 0,
                     cudaMemcpyHostToDevice);

  cudaMalloc(&h_prefix_sum_temp, 3 * sizeof(int) * kMaxNumCars);
  cudaMemcpyToSymbol(d_prefix_sum_temp, &h_prefix_sum_temp, sizeof(int *), 0,
                     cudaMemcpyHostToDevice);

  cudaMalloc(&h_prefix_sum_output, sizeof(int) * kMaxNumCars);
  cudaMemcpyToSymbol(d_prefix_sum_output, &h_prefix_sum_output, sizeof(int *),
                     0, cudaMemcpyHostToDevice);

  kernel_fill_car_indices<<<128, 128>>>();
  gpuErrchk(cudaDeviceSynchronize());

  int zero = 0;
  cudaMemcpyToSymbol(d_num_cells, &zero, sizeof(int), 0,
                     cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(d_num_cars, &zero, sizeof(int), 0, cudaMemcpyHostToDevice);
}

__global__ void kernel_compact_initialize() {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < kMaxNumCars;
       i += blockDim.x * gridDim.x) {
    d_Car_active_2[i] = 0;
  }
}

__global__ void kernel_compact_cars() {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < d_num_cars;
       i += blockDim.x * gridDim.x) {
    if (d_Car_active[i]) {
      int target = d_prefix_sum_output[i];

      // Copy i --> target.
      // dev_cars_2[target] = dev_cars[i];
      memcpy(dev_cars_2[target], dev_cars[i], sizeof(Car));
      d_Car_active_2[target] = 1;

      // Update pointer in Cell.
      dev_cars[i]->position->car = dev_cars[target];

      atomicAdd(&d_num_cars_2, 1);
    }
  }
}

__global__ void kernel_compact_swap_pointers() {
  {
    auto *tmp = dev_cars;
    dev_cars = dev_cars_2;
    dev_cars_2 = tmp;
  }

  {
    auto *tmp = d_Car_active;
    d_Car_active = d_Car_active_2;
    d_Car_active_2 = tmp;
  }

  d_num_cars = d_num_cars_2;
}

void compact_car_array() {
  int zero = 0;
  cudaMemcpyToSymbol(d_num_cars_2, &zero, sizeof(int), 0,
                     cudaMemcpyHostToDevice);
  cudaMemcpyFromSymbol(&host_num_cars, d_num_cars, sizeof(int), 0,
                       cudaMemcpyDeviceToHost);

  // TODO: Prefix sum broken for num_objects < 256.
  auto prefix_sum_size = host_num_cars < 256 ? 256 : host_num_cars;
  size_t temp_size = 3 * kMaxNumCars;
  cub::DeviceScan::ExclusiveSum(h_prefix_sum_temp, temp_size, h_Car_active,
                                h_prefix_sum_output, prefix_sum_size);
  gpuErrchk(cudaDeviceSynchronize());

  kernel_compact_initialize<<<(kMaxNumCars + kNumBlockSize - 1) / kNumBlockSize,
                              kNumBlockSize>>>();
  gpuErrchk(cudaDeviceSynchronize());

  kernel_compact_cars<<<(kMaxNumCars + kNumBlockSize - 1) / kNumBlockSize,
                        kNumBlockSize>>>();
  gpuErrchk(cudaDeviceSynchronize());

  kernel_compact_swap_pointers<<<1, 1>>>();
  gpuErrchk(cudaDeviceSynchronize());

  auto *tmp = h_Car_active;
  h_Car_active = h_Car_active_2;
  h_Car_active_2 = tmp;
}

int main(int /*argc*/, char ** /*argv*/) {
  cudaDeviceSetLimit(cudaLimitMallocHeapSize, 4ULL * 1024 * 1024 * 1024);

  high_resolution_clock::time_point t1 = high_resolution_clock::now();

  allocate_memory();
  high_resolution_clock::time_point t2 = high_resolution_clock::now();
  duration<double> alloc_time = duration_cast<duration<double>>(t2 - t1);

  printf("alloc_time : %f\n",alloc_time.count());

  printf("mem alloc done\n");
  create_street_network();

  auto time_start = std::chrono::system_clock::now();

  for (int i = 0; i < kNumIterations; ++i) {
    step();

    compact_car_array();
  }

  auto time_end = std::chrono::system_clock::now();
  auto elapsed = time_end - time_start;
  auto millis =
      std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();

#ifndef NDEBUG
  printf("Checksum: %i\n", checksum());
#endif // NDEBUG

  printf("%lu\n", millis);
}
