#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h" //added
#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"  //added
#include "RecoTracker/SiTrackerMRHTools/interface/MultiTrajectoryMeasurement.h"  //added
#include "TrackingTools/DetLayers/interface/DetLayer.h"    //added
#include "DataFormats/GeometrySurface/interface/BoundSurface.h"   //added
#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHit.h"  //added

MultiTrajectoryMeasurement::MultiTrajectoryMeasurement() {}

MultiTrajectoryMeasurement::MultiTrajectoryMeasurement(TransientTrackingRecHit::ConstRecHitPointer hit,
						       const std::map<int, TSOS>& predictions,
						       const std::map<int, TSOS>& updates,
						       const DetLayer* lay) :
  theRecHits(std::vector<TransientTrackingRecHit::ConstRecHitPointer>(1, hit)),
  theMultiHits(std::map<int,const TransientTrackingRecHit*>()),
  theFilteredStates(updates),
  thePredictedStates(predictions),
  theChi2s(std::map<int,float>()),
  theLayer(lay) {}

MultiTrajectoryMeasurement::MultiTrajectoryMeasurement(std::vector<TransientTrackingRecHit::ConstRecHitPointer>& hits,
						       const std::map<int,const TransientTrackingRecHit*>& multiHits,
						       const std::map<int, TSOS>& predictions,
						       const std::map<int, TSOS>& updates,
						       const std::map<int, float>& chi2s,
						       const DetLayer* lay) :
  theRecHits(hits),
  theMultiHits(multiHits),
  theFilteredStates(updates),
  thePredictedStates(predictions),
  theChi2s(chi2s),

  theLayer(lay) {}

std::vector<TransientTrackingRecHit::ConstRecHitPointer> MultiTrajectoryMeasurement::hits() const {
  
  return theRecHits;
}

std::map<int,const TransientTrackingRecHit*>& MultiTrajectoryMeasurement::multiHits() {

  return theMultiHits;
}

std::map<int, TrajectoryStateOnSurface>& MultiTrajectoryMeasurement::filteredStates() {

  return theFilteredStates;
}

std::map<int, TrajectoryStateOnSurface>& MultiTrajectoryMeasurement::predictedStates() {

return thePredictedStates;
}

std::map<int, float>& MultiTrajectoryMeasurement::chi2s() {

return theChi2s;
}

const BoundSurface& MultiTrajectoryMeasurement::surface() const {
  if(hits().empty()) {
    std::cout << "MultiTrajectoryMeasurement::surface() no hits" << std::endl;
    std::cout << "Program segmentation faults now. Have a nice day." << std::endl; 
  }
  //return hits().front()->det().surface();
  return layer()->surface();
}

const DetLayer* MultiTrajectoryMeasurement::layer() const {
  
  if(theLayer == 0) std::cout << "MultiTrajectoryMeasurement::layer() is 0!" << std::endl;
  return theLayer;
}

