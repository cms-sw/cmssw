#include "TrackingTools/GsfTracking/interface/FullConvolutionWithMaterial.h"

#include "TrackingTools/GsfTools/interface/MultiTrajectoryStateAssembler.h"

TrajectoryStateOnSurface
FullConvolutionWithMaterial::operator() (const TrajectoryStateOnSurface& tsos,
					 const PropagationDirection propDir) const {
  //
  // decomposition of input state
  //
  typedef std::vector<TrajectoryStateOnSurface> MultiTSOS;  
  MultiTSOS input(tsos.components());
  //
  // vector of result states
  //
  MultiTrajectoryStateAssembler result;
  //
  // now add material effects to each component
  //
  for ( MultiTSOS::const_iterator iTsos=input.begin();
	iTsos!=input.end(); iTsos++ ) {
    //
    // add material
    //
    TrajectoryStateOnSurface updatedTSoS = 
      theMEUpdator->updateState(*iTsos,propDir);
    if ( updatedTSoS.isValid() )
      result.addState(updatedTSoS);
    else
      result.addInvalidState(iTsos->weight());
  }
  return result.combinedState();
}

//  TimingReport::Item* FullConvolutionWithMaterial::theTimer1(0);
//  TimingReport::Item* FullConvolutionWithMaterial::theTimer2(0);
