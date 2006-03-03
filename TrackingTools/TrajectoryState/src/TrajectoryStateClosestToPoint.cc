#include "TrackingTools/TrajectoryState/interface/TrajectoryStateClosestToPoint.h"
#include "TrackingTools/TrajectoryState/interface/FakeField.h"

// Private constructor

TrajectoryStateClosestToPoint::
TrajectoryStateClosestToPoint(const FTS& originalFTS, const GlobalPoint& referencePoint) :
  theFTS(originalFTS), theFTSavailable(true), theRefPoint(referencePoint)
{
  theParameters = perigeeConversions.ftsToPerigeeParameters(originalFTS, referencePoint);
  if (theFTS.hasError()) {
    thePerigeeError = perigeeConversions.ftsToPerigeeError(originalFTS);
    errorIsAvailable = true;
  } 
  else {
    errorIsAvailable = false;
  }
}


TrajectoryStateClosestToPoint::
TrajectoryStateClosestToPoint(const reco::helix::Parameters & helixPar, 
	const GlobalPoint& referencePoint) :
  theFTSavailable(false), theRefPoint(referencePoint), errorIsAvailable(false)
{
  theParameters = perigeeConversions.helixToPerigeeParameters(helixPar, referencePoint);
}


TrajectoryStateClosestToPoint::
TrajectoryStateClosestToPoint(const reco::helix::Parameters & helixPar, 
	const reco::helix::Covariance & helixCov, const GlobalPoint& referencePoint) :
  theFTSavailable(false), theRefPoint(referencePoint), errorIsAvailable(true)
{
  theParameters = perigeeConversions.helixToPerigeeParameters(helixPar, referencePoint);
  thePerigeeError = perigeeConversions.helixToPerigeeError(helixPar, helixCov);
}

  /**
   * Public constructor, which is used to convert perigee 
   * parameters to a FreeTrajectoryState. For the case where
   * no error is provided.
   */

TrajectoryStateClosestToPoint::
TrajectoryStateClosestToPoint(const PerigeeTrajectoryParameters& perigeeParameters,
  const GlobalPoint& referencePoint) :
    theFTSavailable(false), theRefPoint(referencePoint), 
    theParameters(perigeeParameters), errorIsAvailable(false)
{}

  /**
   * Public constructor, which is used to convert perigee 
   * parameters to a FreeTrajectoryState. For the case where
   * an error is provided.
   */

TrajectoryStateClosestToPoint::
TrajectoryStateClosestToPoint(const PerigeeTrajectoryParameters& perigeeParameters,
  const PerigeeTrajectoryError& perigeeError, const GlobalPoint& referencePoint):
    theFTSavailable(false), theRefPoint(referencePoint), theParameters(perigeeParameters),
    thePerigeeError(perigeeError), errorIsAvailable(true)
    
{}


void TrajectoryStateClosestToPoint::calculateFTS() const
{
  GlobalTrajectoryParameters gtp(
	    perigeeConversions.positionFromPerigee(theParameters, theRefPoint),
	    perigeeConversions.momentumFromPerigee(theParameters, theRefPoint),
	    theParameters.charge(),
	    TrackingTools::FakeField::Field::field());
  if (errorIsAvailable) {
    theFTS = FTS(gtp, perigeeConversions.curvilinearError(thePerigeeError, gtp));
  } else {
    theFTS = FTS(gtp);
  }
  theFTSavailable = true;
}
