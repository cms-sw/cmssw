#include "Phase2EndcapSubDisk.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/GeometrySurface/interface/SimpleDiskBounds.h"

#include "TrackingTools/DetLayers/interface/DetLayerException.h"
#include "TrackingTools/GeomPropagators/interface/HelixForwardPlaneCrossing.h"

#include <array>
#include "DetGroupMerger.h"

using namespace std;

const std::vector<const GeometricSearchDet*>& Phase2EndcapSubDisk::components() const {
  throw cms::Exception("Phase2EndcapSubDisk::components() is not implemented");
}

void Phase2EndcapSubDisk::fillRingPars(int i) {
  const BoundDisk& ringDisk = static_cast<const BoundDisk&>(theComps[i]->surface());
  ringPars.push_back(tkDetUtil::fillRingParametersFromDisk(ringDisk));
}

Phase2EndcapSubDisk::Phase2EndcapSubDisk(vector<const Phase2EndcapSingleRing*>& rings)
    : RingedForwardLayer(true), theComponents{nullptr} {
  theRingSize = rings.size();
  LogDebug("TkDetLayers") << "Number of rings in Phase2EndcapSubDisk is " << theRingSize << std::endl;
  setSurface(computeDisk(rings));

  for (unsigned int i = 0; i != rings.size(); ++i) {
    theComps.push_back(rings[i]);
    fillRingPars(i);
    theBasicComps.insert(
        theBasicComps.end(), (*rings[i]).basicComponents().begin(), (*rings[i]).basicComponents().end());
  }

  LogDebug("TkDetLayers") << "==== DEBUG Phase2EndcapSubDisk =====";
  LogDebug("TkDetLayers") << "r,zed pos  , thickness, innerR, outerR: " << this->position().perp() << " , "
                          << this->position().z() << " , " << this->specificSurface().bounds().thickness() << " , "
                          << this->specificSurface().innerRadius() << " , " << this->specificSurface().outerRadius();
}

BoundDisk* Phase2EndcapSubDisk::computeDisk(const vector<const Phase2EndcapSingleRing*>& rings) const {
  return tkDetUtil::computeDisk(rings);
}

Phase2EndcapSubDisk::~Phase2EndcapSubDisk() {
  for (auto c : theComps)
    delete c;

  delete theComponents.load();
}

void Phase2EndcapSubDisk::groupedCompatibleDetsV(const TrajectoryStateOnSurface& startingState,
                                                 const Propagator& prop,
                                                 const MeasurementEstimator& est,
                                                 std::vector<DetGroup>& result) const {
  tkDetUtil::groupedCompatibleDetsV(startingState, prop, est, result, theRingSize, theComps, ringPars);
  return;
}

float Phase2EndcapSubDisk::computeWindowSize(const GeomDet* det,
                                             const TrajectoryStateOnSurface& tsos,
                                             const MeasurementEstimator& est) const {
  return tkDetUtil::computeYdirWindowSize(det, tsos, est);
}

bool Phase2EndcapSubDisk::overlapInR(const TrajectoryStateOnSurface& tsos,
                                     int index,
                                     double ymax,
                                     std::vector<tkDetUtil::RingPar> ringParams) const {
  return tkDetUtil::overlapInR(tsos, index, ymax, ringParams);
}
