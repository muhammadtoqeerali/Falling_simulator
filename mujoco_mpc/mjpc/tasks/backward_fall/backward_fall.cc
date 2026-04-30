#include "mjpc/tasks/backward_fall/backward_fall.h"

#include <cmath>
#include <string>

#include "mjpc/task.h"
#include "mjpc/utilities.h"

namespace mjpc {

std::string BackwardFallHumanoid::Name() const { return "BackwardFallHumanoid"; }
std::string BackwardFallHumanoid::XmlPath() const {
  return GetModelPath("backward_fall/task.xml");
}

void BackwardFallHumanoid::ResetLocked(const mjModel* model) {
  mode = 0;
  if (parameters.empty()) {
    parameters = {
        0.88,   // upright trunk target cosine proxy / pitch proxy
        -0.75,  // desired backward pelvis vx during fall onset
        0.16,   // desired settle pelvis height
        0.0,    // desired lateral COM offset
        0.0,    // desired settle angular velocity
        1.0     // phase gate: 0 pre-impact, 1 post-impact settle emphasis
    };
  }
}

void BackwardFallHumanoid::ResidualFn::Residual(
    const mjModel* model, const mjData* data, double* residual) const {
  // NOTE:
  // This is a scaffold task. Body and joint indexing may need adaptation to
  // the exact humanoid MJCF used in the user's environment.

  const int pelvis_bid = mj_name2id(model, mjOBJ_BODY, "Pelvis");
  const int head_bid = mj_name2id(model, mjOBJ_BODY, "Head");

  double pelvis_h = 0.0;
  double pelvis_vx = 0.0;
  double pelvis_vy = 0.0;
  double pelvis_vz = 0.0;
  double trunk_pitch_proxy = 0.0;
  double ang_vel = 0.0;
  double stillness = 0.0;

  if (pelvis_bid >= 0) {
    pelvis_h = data->xpos[3 * pelvis_bid + 2];

    mjtNum vel6[6] = {0, 0, 0, 0, 0, 0};
    mj_objectVelocity(model, data, mjOBJ_BODY, pelvis_bid, vel6, 0);
    pelvis_vx = vel6[3];
    pelvis_vy = vel6[4];
    pelvis_vz = vel6[5];
    ang_vel = std::sqrt(vel6[0] * vel6[0] + vel6[1] * vel6[1] + vel6[2] * vel6[2]);
    stillness = std::sqrt(pelvis_vx * pelvis_vx + pelvis_vy * pelvis_vy + pelvis_vz * pelvis_vz);
  }

  if (pelvis_bid >= 0 && head_bid >= 0) {
    double vx = data->xpos[3 * head_bid + 0] - data->xpos[3 * pelvis_bid + 0];
    double vy = data->xpos[3 * head_bid + 1] - data->xpos[3 * pelvis_bid + 1];
    double vz = data->xpos[3 * head_bid + 2] - data->xpos[3 * pelvis_bid + 2];
    double n = std::sqrt(vx * vx + vy * vy + vz * vz) + 1.0e-6;
    trunk_pitch_proxy = vz / n;  // +1 upright, 0 horizontal, -1 inverted
  }

  const double phase_gate = parameters_.size() > 5 ? parameters_[5] : 0.0;
  const double upright_target = parameters_.size() > 0 ? parameters_[0] : 0.88;
  const double backward_vx_target = parameters_.size() > 1 ? parameters_[1] : -0.75;
  const double settle_h_target = parameters_.size() > 2 ? parameters_[2] : 0.16;
  const double lateral_target = parameters_.size() > 3 ? parameters_[3] : 0.0;
  const double settle_angvel_target = parameters_.size() > 4 ? parameters_[4] : 0.0;

  // Pre-impact tracking terms
  residual[0] = (1.0 - phase_gate) * (trunk_pitch_proxy - upright_target);
  residual[1] = (1.0 - phase_gate) * (pelvis_vx - backward_vx_target);
  residual[2] = (1.0 - phase_gate) * (pelvis_vy - lateral_target);
  residual[3] = (1.0 - phase_gate) * (pelvis_h - 0.82);

  // Fall / impact shaping terms
  residual[4] = pelvis_vz;
  residual[5] = ang_vel;
  residual[6] = data->ctrl ? data->ctrl[0] : 0.0;

  // Post-impact settle terms
  residual[7] = phase_gate * stillness;
  residual[8] = phase_gate * (pelvis_h - settle_h_target);
  residual[9] = phase_gate * (ang_vel - settle_angvel_target);
}

}  // namespace mjpc
