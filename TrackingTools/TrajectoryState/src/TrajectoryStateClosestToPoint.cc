#include "TrackingTools/TrajectoryState/interface/TrajectoryStateClosestToPoint.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

// Private constructor

TrajectoryStateClosestToPoint::
TrajectoryStateClosestToPoint(const FTS& originalFTS, const GlobalPoint& referencePoint) :
  theFTS(originalFTS), theRefPoint(referencePoint),valid(true), theFTSavailable(true) {
  try {
    theParameters = PerigeeConversions::ftsToPerigeeParameters(originalFTS, referencePoint, thePt);
    if (theFTS.hasError()) {
      thePerigeeError = PerigeeConversions::ftsToPerigeeError(originalFTS);
      errorIsAvailable = true;
    } 
    else {
      errorIsAvailable = false;
    }
    theField = &(originalFTS.parameters().magneticField());
  } catch (const cms::Exception &ex) {
    if (ex.category() != "PerigeeConversions") throw;
    edm::LogWarning("TrajectoryStateClosestToPoint_PerigeeConversions") << "Caught exception " << ex.explainSelf() << ".\n";
    valid = false;
  }
}



void TrajectoryStateClosestToPoint::calculateFTS() const {
  if(!isValid()) throw TrajectoryStateException("TrajectoryStateClosestToPoint is invalid and cannot return any parameters");
  GlobalTrajectoryParameters gtp(
				 PerigeeConversions::positionFromPerigee(theParameters, theRefPoint),
				 PerigeeConversions::momentumFromPerigee(theParameters, thePt, theRefPoint),
				 theParameters.charge(), theField);
  if (errorIsAvailable) {
    theFTS = FTS(gtp, PerigeeConversions::curvilinearError(thePerigeeError, gtp));
  } else {
    theFTS = FTS(gtp);
  }
  theFTSavailable = true;
}
