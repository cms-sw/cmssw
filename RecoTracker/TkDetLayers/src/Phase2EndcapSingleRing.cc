#include "Phase2EndcapSingleRing.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "TrackingTools/DetLayers/interface/DetLayerException.h"
#include "TrackingTools/DetLayers/interface/MeasurementEstimator.h"
#include "TrackingTools/GeomPropagators/interface/HelixForwardPlaneCrossing.h"
#include "TrackingTools/DetLayers/interface/rangesIntersect.h"
#include "TrackingTools/DetLayers/interface/ForwardRingDiskBuilderFromDet.h"

#include "LayerCrossingSide.h"
#include "DetGroupMerger.h"
#include "CompatibleDetToGroupAdder.h"

#include "TkDetUtil.h"
#include "DataFormats/GeometryVector/interface/VectorUtil.h"

#include "DetGroupElementZLess.h"

using namespace std;

typedef GeometricSearchDet::DetWithState DetWithState;

Phase2EndcapSingleRing::Phase2EndcapSingleRing(vector<const GeomDet*>& allDets)
    : GeometricSearchDet(true), theDets(allDets.begin(), allDets.end()) {
  theDisk = ForwardRingDiskBuilderFromDet()(theDets);

  theBinFinder = BinFinderType(theDets.front()->surface().position().phi(), theDets.size());

#ifdef EDM_ML_DEBUG
  LogDebug("TkDetLayers") << "DEBUG INFO for Phase2EndcapSingleRing";
  for (vector<const GeomDet*>::const_iterator it = theDets.begin(); it != theDets.end(); it++) {
    LogDebug("TkDetLayers") << "Det detId,phi,z,r: " << (*it)->geographicalId().rawId() << " , "
                            << (*it)->surface().position().phi() << " , " << (*it)->surface().position().z() << " , "
                            << (*it)->surface().position().perp();
  }

#endif
}

Phase2EndcapSingleRing::~Phase2EndcapSingleRing() = default;

const vector<const GeometricSearchDet*>& Phase2EndcapSingleRing::components() const {
  throw DetLayerException("Phase2EndcapSingleRing doesn't have GeometricSearchDet components");
}

pair<bool, TrajectoryStateOnSurface> Phase2EndcapSingleRing::compatible(const TrajectoryStateOnSurface&,
                                                                        const Propagator&,
                                                                        const MeasurementEstimator&) const {
  edm::LogError("TkDetLayers") << "temporary dummy implementation of Phase2EndcapSingleRing::compatible()!!";
  return pair<bool, TrajectoryStateOnSurface>();
}

void Phase2EndcapSingleRing::groupedCompatibleDetsV(const TrajectoryStateOnSurface& tsos,
                                                    const Propagator& prop,
                                                    const MeasurementEstimator& est,
                                                    std::vector<DetGroup>& result) const {
  SubLayerCrossing crossing;

  crossing = computeCrossing(tsos, prop.propagationDirection());

  if (!crossing.isValid())
    return;

  std::vector<DetGroup> closestResult;

  addClosest(tsos, prop, est, crossing, closestResult);
  if (closestResult.empty())
    return;

  DetGroupElement closestGel(closestResult.front().front());

  float phiWindow = tkDetUtil::computeWindowSize(closestGel.det(), closestGel.trajectoryState(), est);

  searchNeighbors(tsos, prop, est, crossing, phiWindow, closestResult, false);

  DetGroupMerger::addSameLevel(std::move(closestResult), result);
}

SubLayerCrossing Phase2EndcapSingleRing::computeCrossing(const TrajectoryStateOnSurface& startingState,
                                                         PropagationDirection propDir) const {
  auto rho = startingState.transverseCurvature();

  HelixPlaneCrossing::PositionType startPos(startingState.globalPosition());
  HelixPlaneCrossing::DirectionType startDir(startingState.globalMomentum());
  HelixForwardPlaneCrossing crossing(startPos, startDir, rho, propDir);

  pair<bool, double> frontPath = crossing.pathLength(*theDisk);
  if (!frontPath.first)
    return SubLayerCrossing();

  GlobalPoint gFrontPoint(crossing.position(frontPath.second));  //There is only one path

  int frontIndex = theBinFinder.binIndex(gFrontPoint.barePhi());
  SubLayerCrossing frontSLC(0, frontIndex, gFrontPoint);

  return frontSLC;
}

bool Phase2EndcapSingleRing::addClosest(const TrajectoryStateOnSurface& tsos,
                                        const Propagator& prop,
                                        const MeasurementEstimator& est,
                                        const SubLayerCrossing& crossing,
                                        vector<DetGroup>& result) const {
  const vector<const GeomDet*>& sub(subLayer(crossing.subLayerIndex()));

  const GeomDet* det(sub[crossing.closestDetIndex()]);

  bool firstgroup = CompatibleDetToGroupAdder::add(*det, tsos, prop, est, result);

  return firstgroup;
}

void Phase2EndcapSingleRing::searchNeighbors(const TrajectoryStateOnSurface& tsos,
                                             const Propagator& prop,
                                             const MeasurementEstimator& est,
                                             const SubLayerCrossing& crossing,
                                             float window,
                                             vector<DetGroup>& result,
                                             bool checkClosest) const {
  const GlobalPoint& gCrossingPos = crossing.position();

  const vector<const GeomDet*>& sLayer(subLayer(crossing.subLayerIndex()));

  int closestIndex = crossing.closestDetIndex();
  int negStartIndex = closestIndex - 1;
  int posStartIndex = closestIndex + 1;

  if (checkClosest) {  // must decide if the closest is on the neg or pos side
    if (Geom::phiLess(gCrossingPos.barePhi(), sLayer[closestIndex]->surface().phi())) {
      posStartIndex = closestIndex;
    } else {
      negStartIndex = closestIndex;
    }
  }

  const BinFinderType& binFinder = theBinFinder;

  typedef CompatibleDetToGroupAdder Adder;
  int half = sLayer.size() / 2;  // to check if dets are called twice....
  for (int idet = negStartIndex; idet >= negStartIndex - half; idet--) {
    const GeomDet& neighborDet = *sLayer[binFinder.binIndex(idet)];
    if (!tkDetUtil::overlapInPhi(gCrossingPos, neighborDet, window))
      break;
    if (!Adder::add(neighborDet, tsos, prop, est, result))
      break;
  }
  for (int idet = posStartIndex; idet < posStartIndex + half; idet++) {
    const GeomDet& neighborDet = *sLayer[binFinder.binIndex(idet)];
    if (!tkDetUtil::overlapInPhi(gCrossingPos, neighborDet, window))
      break;
    if (!Adder::add(neighborDet, tsos, prop, est, result))
      break;
  }
}
