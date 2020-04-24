#include "TrackingTools/PatternTools/interface/TSCBLBuilderWithPropagator.h"
#include "TrackingTools/GeomPropagators/interface/AnalyticalPropagator.h"
#include "TrackingTools/PatternTools/interface/TrajectoryExtrapolatorToLine.h"
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

  GlobalPoint bspos(beamSpot.position().x(), beamSpot.position().y(), beamSpot.position().z());
  GlobalVector bsvec(beamSpot.dxdz(), beamSpot.dydz(), 1.);
  Line bsline(bspos,bsvec);

  TrajectoryExtrapolatorToLine tetl;

  TrajectoryStateOnSurface tsosfinal = tetl.extrapolate(originalFTS,bsline,*thePropagator);

  if (!tsosfinal.isValid())
    return TrajectoryStateClosestToBeamLine();

  //Compute point on beamline of closest approach
  GlobalPoint tp = tsosfinal.globalPosition(); //position of trajectory closest approach
  GlobalVector hyp(tp.x() - bspos.x(),tp.y() - bspos.y(),tp.z() - bspos.z()); //difference between traj and beamline reference
  double l=bsline.direction().dot(hyp); //length along beamline away from reference point
  GlobalPoint closepoint = bspos + l*bsvec;

  //Get the free state and return the TSCBL
  const FreeTrajectoryState theFTS = *tsosfinal.freeState();
  return TrajectoryStateClosestToBeamLine(theFTS, closepoint, beamSpot);
}
