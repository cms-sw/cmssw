#include "RecoVertex/GaussianSumVertexFit/interface/PerigeeMultiLTS.h"
#include "RecoVertex/VertexTools/interface/PerigeeRefittedTrackState.h"
#include "TrackingTools/TrajectoryState/interface/PerigeeConversions.h"
#include "RecoVertex/VertexPrimitives/interface/VertexException.h"

using namespace std;

PerigeeMultiLTS::PerigeeMultiLTS(const GlobalPoint & linP,
	const reco::TransientTrack & track, const TrajectoryStateOnSurface& tsos)
     : theLinPoint(linP), theTrack(track), theTSOS(tsos),
       theCharge(theTrack.charge()), collapsedStateAvailable(false)
{
  //Extract the TSOS components:

  vector<TrajectoryStateOnSurface> tsosComp = theTSOS.components();
  cout << "PerigeeMultiLTS components: "<<tsosComp.size()<<endl;
  ltComp.reserve(tsosComp.size());
  for (vector<TrajectoryStateOnSurface>::iterator it = tsosComp.begin();
  	it != tsosComp.end(); it++) {
  cout <<(*it).globalPosition()<<endl;
    ltComp.push_back(theLTSfactory.linearizedTrackState(theLinPoint, theTrack, *it ));
  }

}


void PerigeeMultiLTS::prepareCollapsedState() const
{
  if (!collapsedStateAvailable) {
    collapsedStateLT = theLTSfactory.linearizedTrackState(theLinPoint, theTrack, theTSOS);
  }
  collapsedStateAvailable = true;
}


/** Method returning the constant term of the Taylor expansion
 *  of the measurement equation
 */
AlgebraicVector PerigeeMultiLTS::constantTerm() const
{
  if (!collapsedStateAvailable) prepareCollapsedState();
  return collapsedStateLT->constantTerm();
}

/**
 * Method returning the Position Jacobian (Matrix A)
 */
AlgebraicMatrix PerigeeMultiLTS::positionJacobian() const
{
  if (!collapsedStateAvailable) prepareCollapsedState();
  return collapsedStateLT->positionJacobian();
}

/**
 * Method returning the Momentum Jacobian (Matrix B)
 */
AlgebraicMatrix PerigeeMultiLTS::momentumJacobian() const
{
  if (!collapsedStateAvailable) prepareCollapsedState();
  return collapsedStateLT->momentumJacobian();
}

/** Method returning the parameters of the Taylor expansion
 */
AlgebraicVector PerigeeMultiLTS::parametersFromExpansion() const
{
  if (!collapsedStateAvailable) prepareCollapsedState();
  return collapsedStateLT->parametersFromExpansion();
}

/**
 * Method returning the TrajectoryStateClosestToPoint at the point
 * of closest approch to the z-axis (a.k.a. transverse impact point)
 */
TrajectoryStateClosestToPoint PerigeeMultiLTS::predictedState() const
{
  if (!collapsedStateAvailable) prepareCollapsedState();
  const PerigeeLinearizedTrackState* otherP =
  	dynamic_cast<const PerigeeLinearizedTrackState*>(collapsedStateLT.get());
  return otherP->predictedState();
}


bool PerigeeMultiLTS::hasError() const
{
  if (!collapsedStateAvailable) prepareCollapsedState();
  return collapsedStateLT->hasError();
}

AlgebraicVector PerigeeMultiLTS::predictedStateParameters() const
{
  if (!collapsedStateAvailable) prepareCollapsedState();
  return collapsedStateLT->predictedStateParameters();
}

AlgebraicVector PerigeeMultiLTS::predictedStateMomentumParameters() const
{
  if (!collapsedStateAvailable) prepareCollapsedState();
  return collapsedStateLT->predictedStateMomentumParameters();
}

AlgebraicSymMatrix PerigeeMultiLTS::predictedStateWeight() const
{
  if (!collapsedStateAvailable) prepareCollapsedState();
  return collapsedStateLT->predictedStateWeight() ;
}

AlgebraicSymMatrix PerigeeMultiLTS::predictedStateError() const
{
  if (!collapsedStateAvailable) prepareCollapsedState();
  return collapsedStateLT-> predictedStateError();
}

AlgebraicSymMatrix PerigeeMultiLTS::predictedStateMomentumError() const
{
  if (!collapsedStateAvailable) prepareCollapsedState();
  return collapsedStateLT->predictedStateMomentumError() ;
}

bool PerigeeMultiLTS::operator ==(LinearizedTrackState& other)const
{
  const PerigeeMultiLTS* otherP =
  	dynamic_cast<const PerigeeMultiLTS*>(&other);
  if (otherP == 0) {
   throw VertexException("PerigeeMultiLTS: don't know how to compare myself to non-perigee track state");
  }
  return (otherP->track() == theTrack);
}


RefCountedLinearizedTrackState
PerigeeMultiLTS::stateWithNewLinearizationPoint
			(const GlobalPoint & newLP) const
{
  return RefCountedLinearizedTrackState(
  		new PerigeeMultiLTS(newLP, track(), theTSOS));
}

RefCountedRefittedTrackState
PerigeeMultiLTS::createRefittedTrackState(
  	const GlobalPoint & vertexPosition,
	const AlgebraicVector & vectorParameters,
	const AlgebraicSymMatrix & covarianceMatrix) const
{
  PerigeeConversions perigeeConversions;
  TrajectoryStateClosestToPoint refittedTSCP =
        perigeeConversions.trajectoryStateClosestToPoint(
	  vectorParameters, vertexPosition, charge(), covarianceMatrix, theTrack.field());
  return RefCountedRefittedTrackState(new PerigeeRefittedTrackState(refittedTSCP));
}

