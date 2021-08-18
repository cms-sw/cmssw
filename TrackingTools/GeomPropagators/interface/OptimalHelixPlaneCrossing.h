#ifndef OptimalHelixPlaneCrossing_H
#define OptimalHelixPlaneCrossing_H

#include "TrackingTools/GeomPropagators/interface/HelixBarrelPlaneCrossingByCircle.h"
#include "TrackingTools/GeomPropagators/interface/HelixForwardPlaneCrossing.h"
#include "TrackingTools/GeomPropagators/interface/HelixArbitraryPlaneCrossing.h"

#include <memory>

class OptimalHelixPlaneCrossing {
public:
  using Base = HelixPlaneCrossing;

  template <typename... Args>
  OptimalHelixPlaneCrossing(Plane const &plane, Args &&...args) {
    GlobalVector u = plane.normalVector();
    constexpr float small = 1.e-6;  // for orientation of planes

    if (std::abs(u.z()) < small) {
      // barrel plane:
      // instantiate HelixBarrelPlaneCrossing,
      new (get()) HelixBarrelPlaneCrossingByCircle(args...);
    } else if ((std::abs(u.x()) < small) & (std::abs(u.y()) < small)) {
      // forward plane:
      // instantiate HelixForwardPlaneCrossing
      new (get()) HelixForwardPlaneCrossing(args...);
    } else {
      // arbitrary plane:
      // instantiate HelixArbitraryPlaneCrossing
      new (get()) HelixArbitraryPlaneCrossing(args...);
    }
  }

  ~OptimalHelixPlaneCrossing() { get()->~Base(); }

  Base &operator*() { return *get(); }
  Base const &operator*() const { return *get(); }

private:
  Base *get() { return (Base *)&mem; }
  Base const *get() const { return (Base const *)&mem; }

  union Tmp {
    HelixBarrelPlaneCrossingByCircle a;
    HelixForwardPlaneCrossing b;
    HelixArbitraryPlaneCrossing c;
  };
  using aligned_union_t = typename std::aligned_storage<sizeof(Tmp), alignof(Tmp)>::type;
  aligned_union_t mem;
};

#endif
