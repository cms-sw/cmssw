#ifndef OptimalHelixPlaneCrossing_H
#define OptimalHelixPlaneCrossing_H

#include "TrackingTools/GeomPropagators/interface/HelixBarrelPlaneCrossingByCircle.h"
#include "TrackingTools/GeomPropagators/interface/HelixForwardPlaneCrossing.h"
#include "TrackingTools/GeomPropagators/interface/HelixArbitraryPlaneCrossing.h"

#include <variant>

class OptimalHelixPlaneCrossing {
public:
  using Base = HelixPlaneCrossing;

  template <typename... Args>
  explicit OptimalHelixPlaneCrossing(Plane const &plane, Args &&...args) {
    GlobalVector u = plane.normalVector();
    constexpr float small = 1.e-6;  // for orientation of planes

    if (std::abs(u.z()) < small) {
      // barrel plane:
      // instantiate HelixBarrelPlaneCrossing,
      mem_.emplace<HelixBarrelPlaneCrossingByCircle>(args...);
      base_ = std::get_if<HelixBarrelPlaneCrossingByCircle>(&mem_);
    } else if ((std::abs(u.x()) < small) && (std::abs(u.y()) < small)) {
      // forward plane:
      // instantiate HelixForwardPlaneCrossing
      mem_.emplace<HelixForwardPlaneCrossing>(args...);
      base_ = std::get_if<HelixForwardPlaneCrossing>(&mem_);
    } else {
      // arbitrary plane:
      // instantiate HelixArbitraryPlaneCrossing
      mem_.emplace<HelixArbitraryPlaneCrossing>(args...);
      base_ = std::get_if<HelixArbitraryPlaneCrossing>(&mem_);
    }
  }
  OptimalHelixPlaneCrossing() = delete;
  OptimalHelixPlaneCrossing(const OptimalHelixPlaneCrossing &) = delete;
  OptimalHelixPlaneCrossing &operator=(const OptimalHelixPlaneCrossing &) = delete;
  OptimalHelixPlaneCrossing(OptimalHelixPlaneCrossing &&) = delete;
  OptimalHelixPlaneCrossing &operator=(OptimalHelixPlaneCrossing &&) = delete;

  [[nodiscard]] Base &operator*() noexcept { return *base_; }
  [[nodiscard]] Base const &operator*() const noexcept { return *base_; }

private:
  std::variant<std::monostate, HelixBarrelPlaneCrossingByCircle, HelixForwardPlaneCrossing, HelixArbitraryPlaneCrossing>
      mem_;
  Base *base_;
};

#endif
