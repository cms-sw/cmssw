#include "TrackingTools/TrajectoryState/interface/TrajectoryStateClosestToPoint.h"

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
  theField = &(originalFTS.parameters().magneticField());
}


TrajectoryStateClosestToPoint::
TrajectoryStateClosestToPoint(const reco::perigee::Parameters & perigeePar, 
	const GlobalPoint& referencePoint, const MagneticField* field) :
  theField(field), theFTSavailable(false), theRefPoint(referencePoint),
  theParameters(perigeePar), errorIsAvailable(false)
{
//   theParameters = perigeeConversions.helixToPerigeeParameters(helixPar, referencePoint);
}


TrajectoryStateClosestToPoint::
TrajectoryStateClosestToPoint(const reco::perigee::Parameters & perigeePar, 
	const reco::perigee::Covariance & perigeeCov, const GlobalPoint& referencePoint,
	const MagneticField* field) :
  theField(field), theFTSavailable(false), theRefPoint(referencePoint), theParameters(perigeePar),
  thePerigeeError(perigeeCov), errorIsAvailable(true)
{
//   theParameters = perigeeConversions.helixToPerigeeParameters(helixPar, referencePoint);
//   thePerigeeError = perigeeConversions.helixToPerigeeError(helixPar, helixCov);
}

  /**
   * Public constructor, which is used to convert perigee 
   * parameters to a FreeTrajectoryState. For the case where
   * no error is provided.
   */

TrajectoryStateClosestToPoint::
TrajectoryStateClosestToPoint(const PerigeeTrajectoryParameters& perigeeParameters,
  const GlobalPoint& referencePoint, const MagneticField* field) :
    theField(field), theFTSavailable(false), theRefPoint(referencePoint), 
    theParameters(perigeeParameters), errorIsAvailable(false)
{}

  /**
   * Public constructor, which is used to convert perigee 
   * parameters to a FreeTrajectoryState. For the case where
   * an error is provided.
   */

TrajectoryStateClosestToPoint::
TrajectoryStateClosestToPoint(const PerigeeTrajectoryParameters& perigeeParameters,
  const PerigeeTrajectoryError& perigeeError, const GlobalPoint& referencePoint,
  const MagneticField* field):
    theField(field), theFTSavailable(false), theRefPoint(referencePoint), theParameters(perigeeParameters),
    thePerigeeError(perigeeError), errorIsAvailable(true)
    
{}


void TrajectoryStateClosestToPoint::calculateFTS() const
{
  GlobalTrajectoryParameters gtp(
	    perigeeConversions.positionFromPerigee(theParameters, theRefPoint),
	    perigeeConversions.momentumFromPerigee(theParameters, theRefPoint),
	    theParameters.charge(), theField);
  if (errorIsAvailable) {
    theFTS = FTS(gtp, perigeeConversions.curvilinearError(thePerigeeError, gtp));
  } else {
    theFTS = FTS(gtp);
  }
  theFTSavailable = true;
}
