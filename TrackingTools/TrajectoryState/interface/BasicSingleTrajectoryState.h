#ifndef BasicSingleTrajectoryState_H
#define BasicSingleTrajectoryState_H

#include "TrackingTools/TrajectoryState/interface/BasicTrajectoryState.h"
#include<cassert>


/** Concrete implementation for the state of one trajectory on a surface.
 */

class BasicSingleTrajectoryState  final : public BasicTrajectoryState {
public:
  BasicSingleTrajectoryState() :  BasicTrajectoryState(){}
  template<typename... Args>
    BasicSingleTrajectoryState(Args && ...args) : BasicTrajectoryState(std::forward<Args>(args)...){/* assert(weight()>0);*/}

  pointer clone() const override {
    return build<BasicSingleTrajectoryState>(*this);
  }

  std::vector<TrajectoryStateOnSurface> components() const override;


};

#endif
