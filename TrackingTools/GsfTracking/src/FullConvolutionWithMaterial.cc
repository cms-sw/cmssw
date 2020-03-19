#include "TrackingTools/GsfTracking/interface/FullConvolutionWithMaterial.h"
#include "TrackingTools/GsfTools/interface/MultiTrajectoryStateAssembler.h"

TrajectoryStateOnSurface FullConvolutionWithMaterial::operator()(const TrajectoryStateOnSurface& tsos,
                                                                 const PropagationDirection propDir) const {
  //
  // vector of result states
  MultiTrajectoryStateAssembler result;
  //
  // now add material effects to each component
  //
  for (auto const& iTsos : tsos.components()) {
    // add material
    TrajectoryStateOnSurface updatedTSoS = theMEUpdator->updateState(iTsos, propDir);
    if (updatedTSoS.isValid())
      result.addState(updatedTSoS);
    else
      result.addInvalidState(iTsos.weight());
  }
  return result.combinedState();
}
