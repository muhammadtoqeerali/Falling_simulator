#ifndef MJPC_TASKS_BACKWARD_FALL_BACKWARD_FALL_H_
#define MJPC_TASKS_BACKWARD_FALL_BACKWARD_FALL_H_

#include <memory>
#include <string>

#include "mjpc/task.h"

namespace mjpc {

class BackwardFallHumanoid : public Task {
 public:
  std::string Name() const override;
  std::string XmlPath() const override;

  class ResidualFn : public BaseResidualFn {
   public:
    explicit ResidualFn(const BackwardFallHumanoid* task) : BaseResidualFn(task) {}

    // Residual layout
    // 0: upright tracking error (pre-impact only)
    // 1: backward COM / pelvis velocity target
    // 2: trunk pitch target
    // 3: pelvis height target
    // 4: lateral drift suppression
    // 5: impact smoothness / vertical velocity damping
    // 6: angular velocity regularization
    // 7: control effort
    // 8: settle stillness
    // 9: settle orientation
    void Residual(const mjModel* model, const mjData* data, double* residual) const override;
  };

  BackwardFallHumanoid() : residual_(this) {}

 protected:
  std::unique_ptr<mjpc::ResidualFn> ResidualLocked() const override {
    return std::make_unique<BackwardFallHumanoid::ResidualFn>(this);
  }
  BaseResidualFn* InternalResidual() override { return &residual_; }
  void ResetLocked(const mjModel* model) override;

 private:
  ResidualFn residual_;
};

}  // namespace mjpc

#endif  // MJPC_TASKS_BACKWARD_FALL_BACKWARD_FALL_H_
