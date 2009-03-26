#include "TrackingTools/PatternTools/interface/TSCBLBuilderWithPropagator.h"
#include "TrackingTools/GeomPropagators/interface/AnalyticalPropagator.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

using namespace std;

TSCBLBuilderWithPropagator::TSCBLBuilderWithPropagator (const MagneticField* field) :
  thePropagator(new AnalyticalPropagator(field, anyDirection)) {}

TSCBLBuilderWithPropagator::TSCBLBuilderWithPropagator (const Propagator& u) :
  thePropagator(u.clone()) 
{
  thePropagator->setPropagationDirection(anyDirection);
}


TrajectoryStateClosestToBeamLine
TSCBLBuilderWithPropagator::operator()
	(const FreeTrajectoryState& originalFTS,
	 const reco::BeamSpot& beamSpot) const
{

  return TrajectoryStateClosestToBeamLine();
}
