#include "TrackingTools/PatternTools/interface/TempTrajectory.h"
#include "DataFormats/SiStripCluster/interface/SiStripClusterTools.h"
#include "DataFormats/SiStripCluster/interface/SiStripCluster.h"
#include "DataFormats/TrackerRecHit2D/interface/OmniClusterRef.h"
#include "DataFormats/TrackerRecHit2D/interface/BaseTrackerRecHit.h"
#include "DataFormats/SiPixelDetId/interface/PixelSubdetector.h"
#include "Geometry/CommonDetUnit/interface/GeomDet.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "FWCore/Utilities/interface/Likely.h"

namespace {
  template <typename DataContainer>
  unsigned short countTrailingValidHits(DataContainer const& meas) {
    unsigned short n = 0;
    for (auto it = meas.rbegin(); it != meas.rend(); --it) {  // it is not consistent with std...
      if (TempTrajectory::lost(*(*it).recHit()))
        break;
      if ((*it).recHit()->isValid())
        ++n;
    }
    return n;
  }
}  // namespace

TempTrajectory::TempTrajectory(Trajectory&& traj)
    : theChiSquared(0),
      theNumberOfFoundHits(0),
      theNumberOfFoundPixelHits(0),
      theNumberOfLostHits(0),
      theNumberOfCCCBadHits_(0),
      theDirection(traj.direction()),
      theValid(traj.isValid()),
      theNHseed(traj.seedNHits()),
      theNLoops(traj.nLoops()),
      theDPhiCache(traj.dPhiCacheForLoopersReconstruction()),
      theCCCThreshold_(traj.cccThreshold()),
      stopReason_(traj.stopReason()) {
  for (auto& it : traj.measurements()) {
    push(std::move(it));
  }
}

void TempTrajectory::pop() {
  if (!empty()) {
    if (theData.back().recHit()->isValid()) {
      theNumberOfFoundHits--;
      if (badForCCC(theData.back()))
        theNumberOfCCCBadHits_--;
      if (Trajectory::pixel(*(theData.back().recHit())))
        theNumberOfFoundPixelHits--;
    } else if (lost(*(theData.back().recHit()))) {
      theNumberOfLostHits--;
    }
    theData.pop_back();
    theNumberOfTrailingFoundHits = countTrailingValidHits(theData);
  }
}

void TempTrajectory::pushAux(double chi2Increment) {
  const TrajectoryMeasurement& tm = theData.back();
  if (tm.recHit()->isValid()) {
    theNumberOfFoundHits++;
    theNumberOfTrailingFoundHits++;
    if (badForCCC(tm))
      theNumberOfCCCBadHits_++;
    if (Trajectory::pixel(*(tm.recHit())))
      theNumberOfFoundPixelHits++;
  }
  //else if (lost( tm.recHit()) && !inactive(tm.recHit().det())) theNumberOfLostHits++;
  else if (lost(*(tm.recHit()))) {
    theNumberOfLostHits++;
    theNumberOfTrailingFoundHits = 0;
  }

  theChiSquared += chi2Increment;
}

void TempTrajectory::push(const TempTrajectory& segment) {
  assert(segment.theDirection == theDirection);
  assert(segment.theCCCThreshold_ == theCCCThreshold_);

  const int N = segment.measurements().size();
  TrajectoryMeasurement const* tmp[N];
  int i = 0;
  //for (DataContainer::const_iterator it = segment.measurements().rbegin(), ed = segment.measurements().rend(); it != ed; --it)
  for (auto const& tm : segment.measurements())
    tmp[i++] = &tm;
  while (i != 0)
    theData.push_back(*tmp[--i]);
  theNumberOfFoundHits += segment.theNumberOfFoundHits;
  theNumberOfFoundPixelHits += segment.theNumberOfFoundPixelHits;
  theNumberOfLostHits += segment.theNumberOfLostHits;
  theNumberOfCCCBadHits_ += segment.theNumberOfCCCBadHits_;
  theNumberOfTrailingFoundHits = countTrailingValidHits(theData);
  theChiSquared += segment.theChiSquared;
}

void TempTrajectory::join(TempTrajectory& segment) {
  assert(segment.theDirection == theDirection);

  if (theCCCThreshold_ != segment.theCCCThreshold_)
    segment.updateBadForCCC(theCCCThreshold_);
  if (segment.theData.shared()) {
    push(segment);
    segment.theData.clear();  // obey the contract, and increase the chances it will be not shared one day
  } else {
    theData.join(segment.theData);
    theNumberOfFoundHits += segment.theNumberOfFoundHits;
    theNumberOfFoundPixelHits += segment.theNumberOfFoundPixelHits;
    theNumberOfLostHits += segment.theNumberOfLostHits;
    theNumberOfCCCBadHits_ += segment.theNumberOfCCCBadHits_;
    theNumberOfTrailingFoundHits = countTrailingValidHits(theData);
    theChiSquared += segment.theChiSquared;
  }
}

PropagationDirection TempTrajectory::direction() const { return PropagationDirection(theDirection); }

void TempTrajectory::check() const {
  if (theData.size() == 0)
    throw cms::Exception("TrackingTools/PatternTools",
                         "Trajectory::check() - information requested from empty Trajectory");
}

bool TempTrajectory::lost(const TrackingRecHit& hit) {
  if LIKELY (hit.isValid())
    return false;

  //     // A DetLayer is always inactive in this logic.
  //     // The DetLayer is the Det of an invalid RecHit only if no DetUnit
  //     // is compatible with the predicted state, so we don't really expect
  //     // a hit in this case.

  if (hit.geographicalId().rawId() == 0) {
    return false;
  }
  return hit.getType() == TrackingRecHit::missing;
}

bool TempTrajectory::badForCCC(const TrajectoryMeasurement& tm) {
  if (!trackerHitRTTI::isFromDet(*tm.recHit()))
    return false;
  auto const* thit = static_cast<const BaseTrackerRecHit*>(tm.recHit()->hit());
  if (!thit)
    return false;
  if (thit->isPixel() || thit->isPhase2())
    return false;
  if (!tm.updatedState().isValid())
    return false;
  return siStripClusterTools::chargePerCM(thit->rawId(),
                                          thit->firstClusterRef().stripCluster(),
                                          tm.updatedState().localParameters()) < theCCCThreshold_;
}

void TempTrajectory::updateBadForCCC(float ccc_threshold) {
  // If the supplied threshold is the same as the currently cached
  // one, then return the current number of bad hits for CCC,
  // otherwise do a new full rescan.
  if (ccc_threshold == theCCCThreshold_)
    return;

  theCCCThreshold_ = ccc_threshold;
  theNumberOfCCCBadHits_ = 0;
  for (auto const& h : theData) {
    if (badForCCC(h))
      theNumberOfCCCBadHits_++;
  }
}

int TempTrajectory::numberOfCCCBadHits(float ccc_threshold) {
  updateBadForCCC(ccc_threshold);
  return theNumberOfCCCBadHits_;
}

Trajectory TempTrajectory::toTrajectory() const {
  PropagationDirection p = PropagationDirection(theDirection);
  Trajectory traj(p);
  traj.setNLoops(theNLoops);
  traj.setStopReason(stopReason_);
  traj.numberOfCCCBadHits(theCCCThreshold_);

  traj.reserve(theData.size());
  const TrajectoryMeasurement* tmp[theData.size()];
  int i = 0;
  for (DataContainer::const_iterator it = theData.rbegin(), ed = theData.rend(); it != ed; --it)
    tmp[i++] = &(*it);
  while (i != 0)
    traj.push(*tmp[--i]);
  return traj;
}
