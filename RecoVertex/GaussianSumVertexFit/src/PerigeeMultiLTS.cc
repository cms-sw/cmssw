#include "RecoVertex/GaussianSumVertexFit/interface/PerigeeMultiLTS.h"
#include "RecoVertex/VertexTools/interface/PerigeeRefittedTrackState.h"
#include "TrackingTools/TrajectoryState/interface/PerigeeConversions.h"
#include "RecoVertex/VertexPrimitives/interface/VertexException.h"
#include "RecoVertex/VertexTools/interface/PerigeeLinearizedTrackState.h"



using namespace std;

PerigeeMultiLTS::PerigeeMultiLTS(const GlobalPoint & linP,
	const reco::TransientTrack & track, const TrajectoryStateOnSurface& tsos)
     : theLinPoint(linP), theTrack(track), theTSOS(tsos),
       theCharge(theTrack.charge()), collapsedStateAvailable(false)
{
  //Extract the TSOS components:

  vector<TrajectoryStateOnSurface> tsosComp = theTSOS.components();
  //cout << "PerigeeMultiLTS components: "<<tsosComp.size()<<endl;
  ltComp.reserve(tsosComp.size());
  for (vector<TrajectoryStateOnSurface>::iterator it = tsosComp.begin();
  	it != tsosComp.end(); it++) {
  // cout <<(*it).globalPosition()<<endl;
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
const AlgebraicVector5 & PerigeeMultiLTS::constantTerm() const
{
  if (!collapsedStateAvailable) prepareCollapsedState();
  return collapsedStateLT->constantTerm();
}

/**
 * Method returning the Position Jacobian (Matrix A)
 */
const AlgebraicMatrix53 & PerigeeMultiLTS::positionJacobian() const
{
  if (!collapsedStateAvailable) prepareCollapsedState();
  return collapsedStateLT->positionJacobian();
}

/**
 * Method returning the Momentum Jacobian (Matrix B)
 */
const AlgebraicMatrix53 & PerigeeMultiLTS::momentumJacobian() const
{
  if (!collapsedStateAvailable) prepareCollapsedState();
  return collapsedStateLT->momentumJacobian();
}

/** Method returning the parameters of the Taylor expansion
 */
const AlgebraicVector5 & PerigeeMultiLTS::parametersFromExpansion() const
{
  if (!collapsedStateAvailable) prepareCollapsedState();
  return collapsedStateLT->parametersFromExpansion();
}

/**
 * Method returning the TrajectoryStateClosestToPoint at the point
 * of closest approch to the z-axis (a.k.a. transverse impact point)
 */
const TrajectoryStateClosestToPoint & PerigeeMultiLTS::predictedState() const
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

AlgebraicVector5 PerigeeMultiLTS::predictedStateParameters() const
{
  if (!collapsedStateAvailable) prepareCollapsedState();
  return collapsedStateLT->predictedStateParameters();
}

AlgebraicVector3 PerigeeMultiLTS::predictedStateMomentumParameters() const
{
  if (!collapsedStateAvailable) prepareCollapsedState();
  return collapsedStateLT->predictedStateMomentumParameters();
}

AlgebraicSymMatrix55 PerigeeMultiLTS::predictedStateWeight(int & error) const
{
  if (!collapsedStateAvailable) prepareCollapsedState();
  return collapsedStateLT->predictedStateWeight(error) ;
}

AlgebraicSymMatrix55 PerigeeMultiLTS::predictedStateError() const
{
  if (!collapsedStateAvailable) prepareCollapsedState();
  return collapsedStateLT-> predictedStateError();
}

AlgebraicSymMatrix33 PerigeeMultiLTS::predictedStateMomentumError() const
{
  if (!collapsedStateAvailable) prepareCollapsedState();
  return collapsedStateLT->predictedStateMomentumError() ;
}

bool PerigeeMultiLTS::operator ==(LinearizedTrackState<5>& other)const
{
  const PerigeeMultiLTS* otherP =
  	dynamic_cast<const PerigeeMultiLTS*>(&other);
  if (otherP == 0) {
   throw VertexException("PerigeeMultiLTS: don't know how to compare myself to non-perigee track state");
  }
  return (otherP->track() == theTrack);
}


PerigeeMultiLTS::RefCountedLinearizedTrackState
PerigeeMultiLTS::stateWithNewLinearizationPoint
			(const GlobalPoint & newLP) const
{
  return RefCountedLinearizedTrackState(
  		new PerigeeMultiLTS(newLP, track(), theTSOS));
}

PerigeeMultiLTS::RefCountedRefittedTrackState
PerigeeMultiLTS::createRefittedTrackState(
  	const GlobalPoint & vertexPosition,
	const AlgebraicVectorM & vectorParameters,
	const AlgebraicSymMatrixOO & covarianceMatrix) const
{
  TrajectoryStateClosestToPoint refittedTSCP =
        PerigeeConversions::trajectoryStateClosestToPoint(
	  vectorParameters, vertexPosition, charge(), covarianceMatrix, theTrack.field());
  return RefCountedRefittedTrackState(new PerigeeRefittedTrackState(refittedTSCP, vectorParameters));
}

AlgebraicVector5 PerigeeMultiLTS::refittedParamFromEquation(
	const RefCountedRefittedTrackState & theRefittedState) const
{
  AlgebraicVector3 vertexPosition;
  vertexPosition(0) = theRefittedState->position().x();
  vertexPosition(1) = theRefittedState->position().y();
  vertexPosition(2) = theRefittedState->position().z();
  AlgebraicVectorN param = constantTerm() + 
		       positionJacobian() * vertexPosition +
		       momentumJacobian() * theRefittedState->momentumVector();
  if (param(2) >  M_PI) param(2)-= 2*M_PI;
  if (param(2) < -M_PI) param(2)+= 2*M_PI;

  return param;
}


void PerigeeMultiLTS::checkParameters(AlgebraicVector5 & parameters) const
{
  if (parameters(2) >  M_PI) parameters(2)-= 2*M_PI;
  if (parameters(2) < -M_PI) parameters(2)+= 2*M_PI;
}
