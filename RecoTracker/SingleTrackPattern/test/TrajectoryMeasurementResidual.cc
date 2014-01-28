#include "RecoTracker/SingleTrackPattern/test/TrajectoryMeasurementResidual.h"
#include "TrackingTools/TrackFitters/interface/TrajectoryStateCombiner.h"
#include "Geometry/CommonDetUnit/interface/GeomDetUnit.h"
#include "Geometry/CommonTopologies/interface/StripTopology.h"
#include "Geometry/CommonTopologies/interface/PixelTopology.h"
//#include "DataFormats/GeometryCommonDetAlgo/interface/MeasurementVector.h"
//#include "DataFormats/GeometryCommonDetAlgo/interface/MeasurementError.h"
#include "DataFormats/GeometryCommonDetAlgo/interface/MeasurementError.h"
#include "DataFormats/GeometryCommonDetAlgo/interface/MeasurementVector.h"
using namespace std;
TrajectoryMeasurementResidual::TrajectoryMeasurementResidual( const TrajectoryMeasurement& tm) :
  theMeasFrame(false),
  theHit(0),
  theTopol(0),
  theStripTopol(0),
  thePixelTopol(0)
{
  theCombinedPredictedState = TrajectoryStateCombiner().combine( tm.forwardPredictedState(),
								 tm.backwardPredictedState());

    theHit = tm.recHit();
    const GeomDet* det = theHit->det();
    if (det->components().empty()) {
      // GeomDetUnit
      const GeomDetUnit* du = dynamic_cast<const GeomDetUnit*>( det);
      theTopol = &(du->topology());
      theMeasFrame = true;
      // residual in the measurement frame 
      MeasurementPoint theMeasHitPos = theTopol->measurementPosition( theHit->localPosition());
      MeasurementPoint theMeasStatePos = 
	theTopol->measurementPosition( theCombinedPredictedState.localPosition());
      theMeasResidual = theMeasHitPos - theMeasStatePos;
      // Error on the residual in the measurement frame 
      MeasurementError hitMeasError = 
	theTopol->measurementError( theHit->localPosition(), theHit->localPositionError());
      MeasurementError stateMeasError = 
	theTopol->measurementError( theCombinedPredictedState.localPosition(),
				    theCombinedPredictedState.localError().positionError());
      theMeasError = new MeasurementError(hitMeasError.uu()+stateMeasError.uu(),
					  hitMeasError.uv()+stateMeasError.uv(),
					  hitMeasError.vv()+stateMeasError.vv());
      //hitMeasError + stateMeasError; // quadratic sum of errors
      theMeasFrame =true;
      theStripTopol = dynamic_cast<const StripTopology*>(theTopol);
      if (theStripTopol != 0) return;
      thePixelTopol = dynamic_cast<const PixelTopology*>(theTopol);
      if (thePixelTopol == 0) throw cms::Exception("TrajectoryMeasurementResidual",
						   "RecHit GeomDet has Topology of unknown type");

    }
    else      theMeasFrame = false;


   
}

double TrajectoryMeasurementResidual::localXResidual() const
{
  return theHit->localPosition().x() - 
    theCombinedPredictedState.localPosition().x();
}

double TrajectoryMeasurementResidual::localYResidual() const
{

  return theHit->localPosition().y() - 
    theCombinedPredictedState.localPosition().y();
}

double TrajectoryMeasurementResidual::measurementXResidualInPitch() const
{
  checkMeas();
  return  theMeasResidual.x();
}
double TrajectoryMeasurementResidual::measurementYResidualInPitch() const
{
  checkMeas();
  return  theMeasResidual.y();
}

double TrajectoryMeasurementResidual::measurementXResidual() const {
  checkMeas();
  return measurementXResidualInPitch() * xPitch();
}

double TrajectoryMeasurementResidual::measurementYResidual() const {
  checkMeas();
  return measurementYResidualInPitch() * yPitch();
}

double TrajectoryMeasurementResidual::localXError() const 
{
  return sqrt( localError().xx());
}

double TrajectoryMeasurementResidual::localYError() const 
{
  return sqrt( localError().yy());
}

LocalError TrajectoryMeasurementResidual::localError() const 
{
  
  return 
  LocalError(
	     theHit->localPositionError().xx()+ theCombinedPredictedState.localError().positionError().xx(),
 	     theHit->localPositionError().xy()+ theCombinedPredictedState.localError().positionError().xy(),
	     theHit->localPositionError().yy()+ theCombinedPredictedState.localError().positionError().yy());
}


double TrajectoryMeasurementResidual::measurementXError() const
{
  checkMeas();
  return sqrt( theMeasError->uu()) * xPitch();
}
double TrajectoryMeasurementResidual::measurementYError() const
{
  checkMeas();
  return sqrt( theMeasError->vv()) * yPitch();
}

double TrajectoryMeasurementResidual::xPitch() const
{
  if (theStripTopol != 0)  
    return theStripTopol->localPitch( theCombinedPredictedState.localPosition());
else
    return thePixelTopol->pitch().first;
}
double TrajectoryMeasurementResidual::yPitch() const
{
  if (theStripTopol != 0) 
    return theStripTopol->localStripLength(theCombinedPredictedState.localPosition());
  else 
    return thePixelTopol->pitch().second;
}

void TrajectoryMeasurementResidual::checkMeas() const
{
  if (! theMeasFrame) {
    throw cms::Exception("TrajectoryMeasurementResidual",
			 "MeasurementFrame information requested, but GeomDet does not have a Topology");
  }
}
