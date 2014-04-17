#ifndef BasicSingleTrajectoryState_H
#define BasicSingleTrajectoryState_H

#include "TrackingTools/TrajectoryState/interface/BasicTrajectoryState.h"


/** Concrete implementation for the state of one trajectory on a surface.
 */

class BasicSingleTrajectoryState  GCC11_FINAL : public BasicTrajectoryState {
public:
  BasicSingleTrajectoryState() :  BasicTrajectoryState(){}
#if defined( __GXX_EXPERIMENTAL_CXX0X__)
  template<typename... Args>
  BasicSingleTrajectoryState(Args && ...args) : BasicTrajectoryState(std::forward<Args>(args)...){}
#endif
  BasicSingleTrajectoryState* clone() const {
    return new BasicSingleTrajectoryState(*this);
  }

};

#endif
