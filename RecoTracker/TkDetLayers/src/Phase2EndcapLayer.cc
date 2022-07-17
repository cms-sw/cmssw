#include "Phase2EndcapLayer.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/GeometrySurface/interface/SimpleDiskBounds.h"

#include "TrackingTools/DetLayers/interface/DetLayerException.h"
#include "TrackingTools/GeomPropagators/interface/HelixForwardPlaneCrossing.h"

#include <array>
#include "DetGroupMerger.h"

using namespace std;

typedef GeometricSearchDet::DetWithState DetWithState;

//hopefully is never called!
const std::vector<const GeometricSearchDet*>& Phase2EndcapLayer::components() const {
  if (not theComponents) {
    auto temp = std::make_unique<std::vector<const GeometricSearchDet*>>();
    temp->reserve(15);  // This number is just an upper bound
    for (auto c : theComps)
      temp->push_back(c);
    std::vector<const GeometricSearchDet*>* expected = nullptr;
    if (theComponents.compare_exchange_strong(expected, temp.get())) {
      //this thread set the value
      temp.release();
    }
  }
  return *theComponents;
}

void Phase2EndcapLayer::fillRingPars(int i) {
  const BoundDisk& ringDisk = static_cast<const BoundDisk&>(theComps[i]->surface());
  ringPars.push_back(tkDetUtil::fillRingParametersFromDisk(ringDisk));
}

Phase2EndcapLayer::Phase2EndcapLayer(vector<const Phase2EndcapRing*>& rings, const bool isOT)
    : RingedForwardLayer(true), isOuterTracker(isOT), theComponents{nullptr} {
  //They should be already R-ordered. TO BE CHECKED!!
  //sort( theRings.begin(), theRings.end(), DetLessR());

  theRingSize = rings.size();
  LogDebug("TkDetLayers") << "Number of rings in Phase2 OT EC layer is " << theRingSize << std::endl;
  setSurface(computeDisk(rings));

  for (unsigned int i = 0; i != rings.size(); ++i) {
    theComps.push_back(rings[i]);
    fillRingPars(i);
    theBasicComps.insert(
        theBasicComps.end(), (*rings[i]).basicComponents().begin(), (*rings[i]).basicComponents().end());
  }

  LogDebug("TkDetLayers") << "==== DEBUG Phase2EndcapLayer =====";
  LogDebug("TkDetLayers") << "r,zed pos  , thickness, innerR, outerR: " << this->position().perp() << " , "
                          << this->position().z() << " , " << this->specificSurface().bounds().thickness() << " , "
                          << this->specificSurface().innerRadius() << " , " << this->specificSurface().outerRadius();
}

BoundDisk* Phase2EndcapLayer::computeDisk(const vector<const Phase2EndcapRing*>& rings) const {
  return tkDetUtil::computeDisk(rings);
}

Phase2EndcapLayer::~Phase2EndcapLayer() {
  for (auto c : theComps)
    delete c;

  delete theComponents.load();
}

void Phase2EndcapLayer::groupedCompatibleDetsV(const TrajectoryStateOnSurface& startingState,
                                               const Propagator& prop,
                                               const MeasurementEstimator& est,
                                               std::vector<DetGroup>& result) const {
  tkDetUtil::groupedCompatibleDetsV(startingState, prop, est, result, theRingSize, theComps, ringPars);
}

float Phase2EndcapLayer::computeWindowSize(const GeomDet* det,
                                           const TrajectoryStateOnSurface& tsos,
                                           const MeasurementEstimator& est) const {
  return tkDetUtil::computeYdirWindowSize(det, tsos, est);
}

bool Phase2EndcapLayer::overlapInR(const TrajectoryStateOnSurface& tsos,
                                   int index,
                                   double ymax,
                                   std::vector<tkDetUtil::RingPar> ringParams) const {
  return tkDetUtil::overlapInR(tsos, index, ymax, ringParams);
}
