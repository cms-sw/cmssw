#include "TrackingTools/TrajectoryState/interface/TrajectoryStateClosestToPoint.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

// Private constructor

TrajectoryStateClosestToPoint::TrajectoryStateClosestToPoint(const FTS& originalFTS, const GlobalPoint& referencePoint)
    : theFTS(originalFTS), theRefPoint(referencePoint), valid(true), theFTSavailable(true), errorIsAvailable(false) {
  auto params = PerigeeConversions::ftsToPerigeeParameters(originalFTS, referencePoint, thePt);
  if (not params) {
    valid = false;
    edm::LogInfo("TrajectoryStateClosestToPoint_PerigeeConversions") << "Track had pt == 0.";
    return;
  }
  theParameters = *params;
  if (theFTS.hasError()) {
    thePerigeeError = PerigeeConversions::ftsToPerigeeError(originalFTS);
    errorIsAvailable = true;
  }
  theField = &(originalFTS.parameters().magneticField());
}

void TrajectoryStateClosestToPoint::calculateFTS() const {
  if (!isValid())
    throw TrajectoryStateException("TrajectoryStateClosestToPoint is invalid and cannot return any parameters");
  GlobalTrajectoryParameters gtp(PerigeeConversions::positionFromPerigee(theParameters, theRefPoint),
                                 PerigeeConversions::momentumFromPerigee(theParameters, thePt, theRefPoint),
                                 theParameters.charge(),
                                 theField);
  if (errorIsAvailable) {
    theFTS = FTS(gtp, PerigeeConversions::curvilinearError(thePerigeeError, gtp));
  } else {
    theFTS = FTS(gtp);
  }
  theFTSavailable = true;
}
