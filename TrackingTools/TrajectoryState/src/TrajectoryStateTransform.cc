#include "TrackingTools/TrajectoryState/interface/TrajectoryStateTransform.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "TrackingTools/TrajectoryParametrization/interface/GlobalTrajectoryParameters.h"
#include "Geometry/CommonDetUnit/interface/TrackingGeometry.h"
#include "Geometry/CommonDetUnit/interface/GeomDet.h"



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

FreeTrajectoryState TrajectoryStateTransform::innerFreeState( const reco::Track& tk,
							      const MagneticField* field) const
{
  Basic3DVector<float> pos( tk.innerPosition());
  GlobalPoint gpos( pos);
  Basic3DVector<float> mom( tk.innerMomentum());
  GlobalVector gmom( mom);
  GlobalTrajectoryParameters par( gpos, gmom, tk.charge(), field);
  CurvilinearTrajectoryError err( tk.extra()->innerStateCovariance());
  return FreeTrajectoryState( par, err);
}

FreeTrajectoryState TrajectoryStateTransform::innerFreeState( const reco::GsfTrack& tk,
							      const MagneticField* field) const
{
  Basic3DVector<float> pos( tk.innerPosition());
  GlobalPoint gpos( pos);
  Basic3DVector<float> mom( tk.innerMomentum());
  GlobalVector gmom( mom);
  GlobalTrajectoryParameters par( gpos, gmom, tk.charge(), field);
  CurvilinearTrajectoryError err( tk.extra()->innerStateCovariance());
  return FreeTrajectoryState( par, err);
}

FreeTrajectoryState TrajectoryStateTransform::outerFreeState( const reco::Track& tk,
							      const MagneticField* field) const
{
  Basic3DVector<float> pos( tk.outerPosition());
  GlobalPoint gpos( pos);
  Basic3DVector<float> mom( tk.outerMomentum());
  GlobalVector gmom( mom);
  GlobalTrajectoryParameters par( gpos, gmom, tk.charge(), field);
  CurvilinearTrajectoryError err( tk.extra()->outerStateCovariance());
  return FreeTrajectoryState( par, err);
}

FreeTrajectoryState TrajectoryStateTransform::outerFreeState( const reco::GsfTrack& tk,
							      const MagneticField* field) const
{
  Basic3DVector<float> pos( tk.outerPosition());
  GlobalPoint gpos( pos);
  Basic3DVector<float> mom( tk.outerMomentum());
  GlobalVector gmom( mom);
  GlobalTrajectoryParameters par( gpos, gmom, tk.charge(), field);
  CurvilinearTrajectoryError err( tk.extra()->outerStateCovariance());
  return FreeTrajectoryState( par, err);
}

TrajectoryStateOnSurface TrajectoryStateTransform::innerStateOnSurface( const reco::Track& tk, 
									const TrackingGeometry& geom,
									const MagneticField* field) const
{
  const Surface& surface = geom.idToDet( DetId( tk.extra()->innerDetId()))->surface();
  return TrajectoryStateOnSurface( innerFreeState( tk, field), surface);
}

TrajectoryStateOnSurface TrajectoryStateTransform::outerStateOnSurface( const reco::Track& tk, 
									const TrackingGeometry& geom,
									const MagneticField* field) const
{
  const Surface& surface = geom.idToDet( DetId( tk.extra()->outerDetId()))->surface();
  return TrajectoryStateOnSurface( outerFreeState( tk, field), surface);
}
