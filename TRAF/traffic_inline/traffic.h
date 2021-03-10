#include <assert.h>
#include <curand_kernel.h>
#include <chrono>
#include <cub/cub.cuh>
#include <limits>

#include "../configuration.h"
//#include "util/util.h"
#define ALL __inline__ __device__
class Car;
class TrafficLight;
//class CellBase {
//  public:
//    curandState_t random_state;
//    CellBase *incoming[kMaxDegree];
//    CellBase *outgoing[kMaxDegree];
//    CarBase *car;
//    int num_incoming;
//    int num_outgoing;
//    int max_velocity;
//    int current_max_velocity;
//    float x;
//    float y;
//    bool is_target;
//    char type;
//
//    __noinline__ __device__ virtual float random_uni() = 0;
//    ALL virtual int get_current_max_velocity() = 0;
//
//    ALL virtual int get_max_velocity() = 0;
//
//    ALL virtual void set_current_max_velocity(int v) = 0;
//
//    ALL virtual void remove_speed_limit() = 0;
//    ALL virtual int get_num_incoming() = 0;
//
//    ALL virtual void set_num_incoming(int num) = 0;
//    ALL virtual int get_num_outgoing() = 0;
//    ALL virtual void set_num_outgoing(int num) = 0;
//
//    ALL virtual CellBase *get_incoming(int idx) = 0;
//    ALL virtual void set_incoming(int idx, CellBase *cell) = 0;
//
//    ALL virtual CellBase *get_outgoing(int idx) = 0;
//
//    ALL virtual void set_outgoing(int idx, CellBase *cell) = 0;
//
//    ALL virtual float get_x() = 0;
//
//    ALL virtual float get_y() = 0;
//
//    ALL virtual bool is_free() = 0;
//
//    ALL virtual bool is_sink() = 0;
//
//    ALL virtual bool get_is_target() = 0;
//    ALL virtual void set_target() = 0;
//    ALL virtual void occupy(CarBase *car) = 0;
//
//    ALL virtual void release() = 0;
//};

class Cell {
  public:
    curandState_t random_state;
    Cell *incoming[kMaxDegree];
    Cell *outgoing[kMaxDegree];
    Car *car;
    int num_incoming;
    int num_outgoing;
    int max_velocity;
    int current_max_velocity;
    float x;
    float y;
    bool is_target;
    char type;

    __noinline__ __device__ float random_uni() {
        return curand_uniform(&this->random_state);
    }

    ALL int get_current_max_velocity() { return this->current_max_velocity; }

    ALL int get_max_velocity() { return this->max_velocity; }

    ALL void set_current_max_velocity(int v) { this->current_max_velocity = v; }

    ALL void remove_speed_limit() {
        this->current_max_velocity = this->max_velocity;
    }

    ALL int get_num_incoming() { return this->num_incoming; }

    ALL void set_num_incoming(int num) { this->num_incoming = num; }

    ALL int get_num_outgoing() { return this->num_outgoing; }

    ALL void set_num_outgoing(int num) { this->num_outgoing = num; }

    ALL Cell *get_incoming(int idx) { return this->incoming[idx]; }

    ALL void set_incoming(int idx, Cell *cell) {
        assert(cell != nullptr);
        this->incoming[idx] = cell;
    }

    ALL Cell *get_outgoing(int idx) { return this->outgoing[idx]; }

    ALL void set_outgoing(int idx, Cell *cell) {
        assert(cell != nullptr);
        this->outgoing[idx] = cell;
    }

    ALL float get_x() { return this->x; }

    ALL float get_y() { return this->y; }

    ALL bool is_free() { return this->car == nullptr; }

    ALL bool is_sink() { return this->num_outgoing == 0; }

    ALL bool get_is_target() { return this->is_target; }

    ALL void set_target() { this->is_target = true; }
    ALL void occupy(Car *car) {
        assert(this->is_free());
        this->car = car;
    }

    ALL void release() {
        assert(!this->is_free());
        this->car = nullptr;
    }
};

//class CarBase {
//  public:
//    CellBase *path[kMaxVelocity];
//    CellBase *position;
//    curandState_t random_state;
//    int path_length;
//    int velocity;
//    int max_velocity;
//    ALL virtual void set_path(CellBase *cell, int idx) = 0;
//    ALL virtual CellBase *get_path(int idx) = 0;
//
//    ALL virtual void set_path_length(int len) = 0;
//    ALL virtual int get_path_length() = 0;
//    __noinline__ __device__ virtual int random_int(int a, int b) = 0;
//    __noinline__ __device__ virtual float random_uni() = 0;
//    ALL virtual void step_initialize_iteration() = 0;
//    ALL virtual int get_velocity() = 0;
//    ALL virtual void set_velocity(int v) = 0;
//    ALL virtual void set_max_velocity(int v) = 0;
//    ALL virtual int get_max_velocity() = 0;
//    ALL virtual void set_position(CellBase *cell) = 0;
//    ALL virtual CellBase *get_position() = 0;
//    __noinline__ __device__ virtual void step_slow_down() = 0;
//    __noinline__ __device__ virtual CellBase *next_step(CellBase *position) = 0;
//    __noinline__ __device__ virtual void step_accelerate() = 0;
//};

class Car{
  public:
    Cell *path[kMaxVelocity];
    Cell *position;
    curandState_t random_state;
    int path_length;
    int velocity;
    int max_velocity;
    ALL void set_path(Cell *cell, int idx) { this->path[idx] = cell; }
    ALL Cell *get_path(int idx) { return this->path[idx]; }

    ALL void set_path_length(int len) { this->path_length = len; }
    ALL int get_path_length() { return this->path_length; }
    ALL int random_int(int a, int b) {
        return curand(&this->random_state) % (b - a) + a;
    }
    ALL float random_uni() {
        return curand_uniform(&this->random_state);
    }
    ALL void step_initialize_iteration() {
        // Reset calculated path. This forces cars with a random moving behavior
        // to select a new path in every iteration. Otherwise, cars might get
        // "stuck" on a full network if many cars are waiting for the one in
        // front of them in a cycle.
        this->path_length = 0;
    }
    ALL int get_velocity() { return this->velocity; }
    ALL void set_velocity(int v) { this->velocity = v; }
    ALL void set_max_velocity(int v) { this->max_velocity = v; }

    ALL int get_max_velocity() { return this->max_velocity; }
    ALL void set_position(Cell *cell) { this->position = cell; }

    ALL Cell *get_position() { return this->position; }
    ALL void step_slow_down() {
        // 20% change of slowdown.
        if (curand_uniform(&this->random_state) < 0.2 && this->velocity > 0) {
            this->velocity = this->velocity - 1;
        }
    }
    ALL Cell *next_step(Cell *position) {
        // Almost random walk.
        const uint32_t num_outgoing = position->num_outgoing;
        assert(num_outgoing > 0);

        // Need some kind of return statement here.
        return position->outgoing[this->random_int(0, num_outgoing)];
    }
    ALL void step_accelerate() {
        // Speed up the car by 1 or 2 units.

        int speedup = curand(&this->random_state) % (2 - 0) + 0 + 1;
        this->velocity = this->max_velocity < this->velocity + speedup
                             ? this->max_velocity
                             : this->velocity + speedup;
    }
};

//class TrafficLightBase {
//  protected:
//    CellBase *cells_[kMaxDegree];
//    int num_cells;
//    int timer;
//    int phase_time;
//    int phase;
//
//  public:
//    ALL TrafficLightBase() {}
//    ALL TrafficLightBase(int num_cells_, int phase_time_)
//        : num_cells(num_cells_), timer(0), phase_time(phase_time_), phase(0) {}
//
//    ALL virtual void set_cell(int idx, CellBase *cell) = 0;
//    ALL virtual CellBase *get_cell(int idx) = 0;
//    ALL virtual int get_num_cells() = 0;
//    ALL virtual int get_timer() = 0;
//    ALL virtual int set_timer(int time) = 0;
//    ALL virtual int get_phase_time() = 0;
//    ALL virtual void set_phase(int ph) = 0;
//    ALL virtual int get_phase() = 0;
//};

class TrafficLight {
  protected:
    Cell *cells_[kMaxDegree];
    int num_cells;
    int timer;
    int phase_time;
    int phase;

  public:
    ALL TrafficLight() {}
    ALL TrafficLight(int num_cells_, int phase_time_) {
        num_cells = (num_cells_);
        timer = (0);
        phase_time = (phase_time_);
        phase = (0);
    }

    ALL void set_cell(int idx, Cell *cell) {
        assert(cell != nullptr);
        cells_[idx] = cell;
    }
    ALL Cell *get_cell(int idx) { return cells_[idx]; }
    ALL int get_num_cells() { return num_cells; }
    ALL int get_timer() { return timer; }
    ALL int set_timer(int time) { return this->timer = time; }
    ALL int get_phase_time() { return phase_time; }
    ALL void set_phase(int ph) { this->phase = ph; }
    ALL int get_phase() { return this->phase; }
};
