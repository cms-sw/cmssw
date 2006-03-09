#include "TrackingTools/TrajectoryState/interface/TrajectoryStateTransform.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"

PTrajectoryStateOnDet* 
TrajectoryStateTransform::persistentState( const TrajectoryStateOnSurface& ts,
					   unsigned int detid) const
{
  AlgebraicSymMatrix m = ts.localError().matrix();
  
  int dim = 5; /// should check if corresponds to m

  float localErrors[15];
  int k = 0;
  for (int i=0; i<dim; i++) {
    for (int j=0; j<=i; j++) {
      localErrors[k++] = m[i][j];
    }
  }
  int surfaceSide = static_cast<int>(ts.surfaceSide());

  return new PTrajectoryStateOnDet( ts.localParameters(),
				    localErrors, detid,
				    surfaceSide);
}

TrajectoryStateOnSurface 
TrajectoryStateTransform::transientState( const PTrajectoryStateOnDet& ts,
					  const Surface* surface,
					  const MagneticField* field) const
{
  int dim = 5;
  AlgebraicSymMatrix m(dim);
  int k = 0;
  for (int i=0; i<dim; i++) {
    for (int j=0; j<=i; j++) {
      m[i][j] = ts.errorMatrix()[k++];
    }
  }

  return TrajectoryStateOnSurface( ts.parameters(),
				   LocalTrajectoryError(m), 
				   *surface, field,
				   static_cast<SurfaceSide>(ts.surfaceSide()));

}
