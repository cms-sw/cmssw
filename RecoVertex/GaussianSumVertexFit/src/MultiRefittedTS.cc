#include "RecoVertex/GaussianSumVertexFit/interface/MultiRefittedTS.h"
#include "TrackingTools/TrajectoryState/interface/FreeTrajectoryState.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "TrackingTools/GsfTools/interface/BasicMultiTrajectoryState.h"
#include "RecoVertex/VertexPrimitives/interface/VertexException.h"
#include "TrackingTools/PatternTools/interface/TransverseImpactPointExtrapolator.h"
#include "TrackingTools/TransientTrack/interface/TransientTrackFromFTSFactory.h"
#include <cfloat>
using namespace std;


MultiRefittedTS::MultiRefittedTS(const std::vector<RefCountedRefittedTrackState> & prtsComp,
	const Surface & referenceSurface) :
	theComponents(prtsComp), ftsAvailable(false), refSurface(&referenceSurface),
	surf(true) {}


MultiRefittedTS::MultiRefittedTS(const std::vector<RefCountedRefittedTrackState> & prtsComp,
	const GlobalPoint & referencePosition) :
	theComponents(prtsComp), ftsAvailable(false), refPosition(referencePosition),
	surf(false) {}

  /**
   * Transformation into a FreeTrajectoryState
   */

FreeTrajectoryState MultiRefittedTS::freeTrajectoryState() const
{
  if (!ftsAvailable) computeFreeTrajectoryState();
  return fts;
}

void MultiRefittedTS::computeFreeTrajectoryState() const
{
  if (surf) {
    fts =  *(trajectoryStateOnSurface(*refSurface).freeTrajectoryState());
  } else {
    double maxWeight = -1.;
    RTSvector::const_iterator maxIt;
    for (RTSvector::const_iterator it = theComponents.begin(); 
	  it != theComponents.end(); it++) {
      if ( (**it).weight() > maxWeight ) {
	maxWeight = (**it).weight();
	maxIt = it;
      }
    }

    TransverseImpactPointExtrapolator tipe(&((**maxIt).freeTrajectoryState().parameters().magneticField()));
    TrajectoryStateOnSurface initialTSOS = tipe.extrapolate((**maxIt).freeTrajectoryState(), refPosition);

    fts = *(trajectoryStateOnSurface(initialTSOS.surface()).freeTrajectoryState());
  }
  ftsAvailable = true;
}

  /**
   * Vector containing the refitted track parameters. <br>
   * These are (signed transverse curvature, theta, phi,
   *  (signed) transverse , longitudinal impact parameter)
   */

MultiRefittedTS::AlgebraicVectorN MultiRefittedTS::parameters() const
{
  throw VertexException
    ("MultiRefittedTS::freeTrajectoryState(): Don't know how to do that yet...");
}

  /**
   * The covariance matrix
   */

MultiRefittedTS::AlgebraicSymMatrixNN  MultiRefittedTS::covariance() const
{
  throw VertexException
    ("MultiRefittedTS::freeTrajectoryState(): Don't know how to do that yet...");
}

  /**
   * Position at which the momentum is defined.
   */

GlobalPoint MultiRefittedTS::position() const
{
  throw VertexException
    ("MultiRefittedTS::freeTrajectoryState(): Don't know how to do that yet...");
}

  /**
   * Vector containing the parameters describing the momentum as the vertex.
   * These are (signed transverse curvature, theta, phi)
   */

MultiRefittedTS::AlgebraicVectorM MultiRefittedTS::momentumVector() const
{
  throw VertexException
    ("MultiRefittedTS::freeTrajectoryState(): Don't know how to do that yet...");
}

double MultiRefittedTS::weight() const
{
  if (!totalWeightAvailable)
  {
    totalWeight = 0.;
    if (theComponents.empty()) {
      cout << "Asking for weight of empty MultiRefittedTS, returning zero!" << endl;
    }
    for (RTSvector::const_iterator it = theComponents.begin(); 
  	  it != theComponents.end(); it++) {
      totalWeight += (**it).weight();
    }
  }
  return totalWeight;
}


ReferenceCountingPointer<RefittedTrackState<5> > 
MultiRefittedTS::stateWithNewWeight(const double newWeight) const
{
  if (weight() < DBL_MIN) {
  throw VertexException
    ("MultiRefittedTS::stateWithNewWeight(): Can not reweight multi-state with total weight < DBL_MIN");
  }
  double factor = newWeight/weight();

  RTSvector reWeightedRTSC;
  reWeightedRTSC.reserve(theComponents.size());

  for (RTSvector::const_iterator it = theComponents.begin(); 
	it != theComponents.end(); it++) {
    reWeightedRTSC.push_back((**it).stateWithNewWeight((**it).weight()*factor));
  }
  if (surf) {
    return RefCountedRefittedTrackState(new MultiRefittedTS(reWeightedRTSC, *refSurface));
  } else {
    return RefCountedRefittedTrackState(new MultiRefittedTS(reWeightedRTSC, refPosition));
  }
}

  /**
   * Transformation into a TSOS at a given surface
   */
TrajectoryStateOnSurface MultiRefittedTS::trajectoryStateOnSurface(
  		const Surface & surface) const
{
  vector<TrajectoryStateOnSurface> tsosComponents;
  tsosComponents.reserve(theComponents.size());
  for (RTSvector::const_iterator it = theComponents.begin(); 
	it != theComponents.end(); it++) {
    tsosComponents.push_back((**it).trajectoryStateOnSurface(surface));
  }
// #ifndef CMS_NO_COMPLEX_RETURNS
  return TrajectoryStateOnSurface(new BasicMultiTrajectoryState(tsosComponents));
// #else
//   TrajectoryStateOnSurface result(new BasicMultiTrajectoryState(tsosComponents));
//   return result;
// #endif
}

TrajectoryStateOnSurface MultiRefittedTS::trajectoryStateOnSurface(
  		const Surface & surface, const Propagator & propagator) const
{ //fixme... is the propagation done correctly? Is there a gsf propagator?
  vector<TrajectoryStateOnSurface> tsosComponents;
  tsosComponents.reserve(theComponents.size());
  for (RTSvector::const_iterator it = theComponents.begin(); 
	it != theComponents.end(); it++) {
    tsosComponents.push_back((**it).trajectoryStateOnSurface(surface, propagator));
  }
// #ifndef CMS_NO_COMPLEX_RETURNS
  return TrajectoryStateOnSurface(new BasicMultiTrajectoryState(tsosComponents));
// #else
//   TrajectoryStateOnSurface result(new BasicMultiTrajectoryState(tsosComponents));
//   return result;
// #endif
}

reco::TransientTrack MultiRefittedTS::transientTrack() const
{
  TransientTrackFromFTSFactory factory;
  return factory.build(freeTrajectoryState());
}
