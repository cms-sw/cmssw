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

TempTrajectory::TempTrajectory(Trajectory&& traj) : thePayload(std::make_unique<Payload>()) {
  assert(traj.isValid());
  thePayload->theDirection = traj.direction();
  thePayload->theNHseed = traj.seedNHits();
  thePayload->theNLoops = traj.nLoops();
  thePayload->theDPhiCache = traj.dPhiCacheForLoopersReconstruction();
  thePayload->theCCCThreshold_ = traj.cccThreshold();
  thePayload->stopReason_ = traj.stopReason();
  for (auto& it : traj.measurements()) {
    push(it);
  }
  traj.measurements().clear();
}

void TempTrajectory::pop() {
  if (!empty()) {
    if (theData.back().recHit()->isValid()) {
      thePayload->theNumberOfFoundHits--;
      if (badForCCC(theData.back()))
        thePayload->theNumberOfCCCBadHits_--;
      if (Trajectory::pixel(*(theData.back().recHit())))
        thePayload->theNumberOfFoundPixelHits--;
    } else if (lost(*(theData.back().recHit()))) {
      thePayload->theNumberOfLostHits--;
    }
    theData.pop_back();
    thePayload->theNumberOfTrailingFoundHits = countTrailingValidHits(theData);
  }
}

void TempTrajectory::pushAux(double chi2Increment) {
  const TrajectoryMeasurement& tm = theData.back();
  if (tm.recHit()->isValid()) {
    thePayload->theNumberOfFoundHits++;
    thePayload->theNumberOfTrailingFoundHits++;
    if (badForCCC(tm))
      thePayload->theNumberOfCCCBadHits_++;
    if (Trajectory::pixel(*(tm.recHit())))
      thePayload->theNumberOfFoundPixelHits++;
  }
  //else if (lost( tm.recHit()) && !inactive(tm.recHit().det())) theNumberOfLostHits++;
  else if (lost(*(tm.recHit()))) {
    thePayload->theNumberOfLostHits++;
    thePayload->theNumberOfTrailingFoundHits = 0;
  }

  thePayload->theChiSquared += chi2Increment;
}

void TempTrajectory::push(const TempTrajectory& segment) {
  assert(segment.thePayload->theDirection == thePayload->theDirection);
  assert(segment.thePayload->theCCCThreshold_ == thePayload->theCCCThreshold_);

  const int N = segment.measurements().size();
  TrajectoryMeasurement const* tmp[N];
  int i = 0;
  //for (DataContainer::const_iterator it = segment.measurements().rbegin(), ed = segment.measurements().rend(); it != ed; --it)
  for (auto const& tm : segment.measurements())
    tmp[i++] = &tm;
  while (i != 0)
    theData.push_back(*tmp[--i]);
  thePayload->theNumberOfFoundHits += segment.thePayload->theNumberOfFoundHits;
  thePayload->theNumberOfFoundPixelHits += segment.thePayload->theNumberOfFoundPixelHits;
  thePayload->theNumberOfLostHits += segment.thePayload->theNumberOfLostHits;
  thePayload->theNumberOfCCCBadHits_ += segment.thePayload->theNumberOfCCCBadHits_;
  thePayload->theNumberOfTrailingFoundHits = countTrailingValidHits(theData);
  thePayload->theChiSquared += segment.thePayload->theChiSquared;
}

void TempTrajectory::join(TempTrajectory& segment) {
  assert(segment.thePayload->theDirection == thePayload->theDirection);

  if (thePayload->theCCCThreshold_ != segment.thePayload->theCCCThreshold_)
    segment.updateBadForCCC(thePayload->theCCCThreshold_);
  if (segment.theData.shared()) {
    push(segment);
    segment.theData.clear();  // obey the contract, and increase the chances it will be not shared one day
  } else {
    theData.join(segment.theData);
    thePayload->theNumberOfFoundHits += segment.thePayload->theNumberOfFoundHits;
    thePayload->theNumberOfFoundPixelHits += segment.thePayload->theNumberOfFoundPixelHits;
    thePayload->theNumberOfLostHits += segment.thePayload->theNumberOfLostHits;
    thePayload->theNumberOfCCCBadHits_ += segment.thePayload->theNumberOfCCCBadHits_;
    thePayload->theNumberOfTrailingFoundHits = countTrailingValidHits(theData);
    thePayload->theChiSquared += segment.thePayload->theChiSquared;
  }
}

PropagationDirection TempTrajectory::direction() const { return PropagationDirection(thePayload->theDirection); }

void TempTrajectory::check() const {
  if (theData.empty())
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
                                          tm.updatedState().localParameters()) < thePayload->theCCCThreshold_;
}

void TempTrajectory::updateBadForCCC(float ccc_threshold) {
  // If the supplied threshold is the same as the currently cached
  // one, then return the current number of bad hits for CCC,
  // otherwise do a new full rescan.
  if (ccc_threshold == thePayload->theCCCThreshold_)
    return;

  thePayload->theCCCThreshold_ = ccc_threshold;
  thePayload->theNumberOfCCCBadHits_ = 0;
  for (auto const& h : theData) {
    if (badForCCC(h))
      thePayload->theNumberOfCCCBadHits_++;
  }
}

int TempTrajectory::numberOfCCCBadHits(float ccc_threshold) {
  updateBadForCCC(ccc_threshold);
  return thePayload->theNumberOfCCCBadHits_;
}

Trajectory TempTrajectory::toTrajectory() const {
  assert(isValid());
  PropagationDirection p = PropagationDirection(thePayload->theDirection);
  Trajectory traj(p);
  traj.setNLoops(thePayload->theNLoops);
  traj.setStopReason(thePayload->stopReason_);
  traj.numberOfCCCBadHits(thePayload->theCCCThreshold_);

  traj.reserve(theData.size());
  const TrajectoryMeasurement* tmp[theData.size()];
  int i = 0;
  for (DataContainer::const_iterator it = theData.rbegin(), ed = theData.rend(); it != ed; --it)
    tmp[i++] = &(*it);
  while (i != 0)
    traj.push(*tmp[--i]);
  return traj;
}
