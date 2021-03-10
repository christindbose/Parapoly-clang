#include <assert.h>
#include <chrono>
#include <curand_kernel.h>
#include <limits>
#include <stdio.h>
#include "../configuration.h"
#include "../dataset.h"
#define ALL __inline__ __device__

class Spring;
//class NodeBase {
//public:
//  SpringBase *springs[kMaxDegree];
//  int num_springs;
//  int distance;
//  float pos_x;
//  float pos_y;
//  float vel_x;
//  float vel_y;
//  float mass;
//  char type;
//  ALL virtual SpringBase *spring(unsigned i) = 0;
//  ALL virtual void pull() = 0;
//  ALL virtual float distance_to(NodeBase *other) = 0;
//  __noinline__  __device__ virtual void remove_spring(SpringBase *spring) = 0;
//  ALL virtual float unit_x(NodeBase *other, float dist) = 0;
//  ALL virtual float unit_y(NodeBase *other, float dist) = 0;
//  ALL virtual void update_vel_x(float force_x) = 0;
//  ALL virtual void update_vel_y(float force_y) = 0;
//  ALL virtual void update_pos_x(float force_x) = 0;
//  ALL virtual void update_pos_y(float force_x) = 0;
//  ALL virtual void set_distance(float dist) = 0;
//  ALL virtual float get_distance() = 0;
//};

class Node{
public:
  Spring *springs[kMaxDegree];
  int num_springs;
  int distance;
  float pos_x;
  float pos_y;
  float vel_x;
  float vel_y;
  float mass;
  char type;
  ALL Spring *spring(unsigned i) { return this->springs[i]; }
  ALL void pull() {
    this->pos_x += this->vel_x * kDt;
    this->pos_y += this->vel_y * kDt;
  }
  ALL float distance_to(Node *other) {
    float dx = this->pos_x - other->pos_x;
    float dy = this->pos_y - other->pos_y;
    float dist_sq = dx * dx + dy * dy;
    return sqrt(dist_sq);
  }
  ALL void remove_spring(Spring *spring) {

    for (int i = 0; i < kMaxDegree; ++i) {
      if (this->springs[i] == spring) {
        this->springs[i] = NULL;
        if (atomicSub(&this->num_springs, 1) == 1) {
          // Deleted last spring.
          this->type = 0;
        }
        return;
      }
    }

    // Spring not found.
    assert(false);
  }
  ALL float unit_x(Node *other, float dist) {
    float dx = this->pos_x - other->pos_x;
    float unit_x = dx / dist;
    return unit_x;
  }
  ALL float unit_y(Node *other, float dist) {
    float dy = this->pos_y - other->pos_y;
    float unit_y = dy / dist;
    return unit_y;
  }
  ALL void update_vel_x(float force_x) {
    this->vel_x += force_x * kDt / this->mass;
    this->vel_x *= 1.0f - kVelocityDampening;
  }
  ALL void update_vel_y(float force_y) {
    this->vel_y += force_y * kDt / this->mass;
    this->vel_y *= 1.0f - kVelocityDampening;
  }
  ALL void update_pos_x(float force_x) {
    this->pos_x += this->vel_x * kDt;
  }
  ALL void update_pos_y(float force_x) {
    this->pos_y += this->vel_y * kDt;
  }
  ALL void set_distance(float dist) { this->distance = dist; }
  ALL float get_distance() { return this->distance; }
};

//class SpringBase {
//public:
//  float factor;
//  float initial_length;
//  float force;
//  float max_force;
//  bool is_active;
//
//  NodeBase *p1;
//  NodeBase *p2;
//  bool delete_flag;
//  ALL virtual void deactivate() =0;
//  ALL virtual void update_force(float displacement) =0;
//  ALL virtual float get_force() = 0;
//  ALL virtual float get_init_len()  = 0;
//  ALL virtual bool get_is_active()  = 0;
//  ALL virtual void set_is_active(bool active)  = 0;
//  ALL virtual NodeBase *get_p1()  = 0;
//  ALL virtual NodeBase *get_p2()  = 0;
//  ALL virtual bool is_max_force()  = 0;
//};

class Spring{
public:
  float factor;
  float initial_length;
  float force;
  float max_force;
  bool is_active;

  Node *p1;
  Node *p2;
  bool delete_flag;

  ALL void deactivate() { this->is_active = false; }
  ALL void update_force(float displacement) {
    this->force = this->factor * displacement;
  }
  ALL float get_force() { return this->force; }
  ALL float get_init_len() { return this->initial_length; }
  ALL bool get_is_active() { return this->is_active; }
  ALL void set_is_active(bool active) { this->is_active = active; }
  ALL Node *get_p1() { return this->p1; }
  ALL Node *get_p2() { return this->p2; }
  ALL bool is_max_force() { return this->force > this->max_force; }
};
