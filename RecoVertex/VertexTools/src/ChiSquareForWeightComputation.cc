#include "RecoVertex/VertexTools/interface/ChiSquareForWeightComputation.h"
#include "TrackingTools/TransientTrack/interface/TransientTrack.h"
#include "RecoVertex/VertexTools/interface/LinearizedTrackStateFactory.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateClosestToPoint.h"
#include "TrackingTools/TrajectoryState/interface/PerigeeConversions.h"
#include "TrackingTools/PatternTools/interface/TSCPBuilderNoMaterial.h"
#include "RecoVertex/VertexPrimitives/interface/VertexException.h"

using namespace std;

float ChiSquareForWeightComputation::estimate(
      const GlobalPoint & vertex, const reco::TransientTrack & track ) const
{
  LinearizedTrackStateFactory factory;
  GlobalPoint linpt ( vertex.x(), vertex.y(), vertex.z() );
  RefCountedLinearizedTrackState lt
    = factory.linearizedTrackState( linpt, track );
  return estimate( vertex, lt );
}

float ChiSquareForWeightComputation::estimate(
      const GlobalPoint & vertex,
      const RefCountedLinearizedTrackState & lt ) const
{
  TrajectoryStateOnSurface tsos=lt->track().impactPointState();
  TrajectoryStateClosestToPoint tscp=TSCPBuilderNoMaterial() ( tsos, vertex );

  AlgebraicVector d=tscp.perigeeParameters().vector().sub(4,5);
  AlgebraicSymMatrix err=tscp.perigeeError().covarianceMatrix().sub(4,5);

  int ifail;
  err.invert(ifail);
  if ( ifail )
  {
    throw VertexException ( "[ChiSquareForWeightComputation] Matrix inversion failed" );
  }

  return err.similarity ( d );
}
