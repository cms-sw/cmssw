#include "TrackingTools/TrajectoryState/interface/TrajectoryStateTransform.h"
#include "DataFormats/TrackReco/interface/Track.h" 
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "TrackingTools/TrajectoryState/interface/FreeTrajectoryState.h" 
#include "DataFormats/GeometrySurface/interface/Surface.h" 
#include "TrackingTools/TrajectoryParametrization/interface/GlobalTrajectoryParameters.h"
#include "Geometry/CommonDetUnit/interface/TrackingGeometry.h"
#include "Geometry/CommonDetUnit/interface/GeomDet.h"


namespace trajectoryStateTransform {

  using namespace SurfaceSideDefinition;
  
  PTrajectoryStateOnDet 
  persistentState( const TrajectoryStateOnSurface& ts,
		   unsigned int detid)
  {
    int surfaceSide = static_cast<int>(ts.surfaceSide());
    auto pt = ts.globalMomentum().perp();
    
    if (ts.hasError()) {
      AlgebraicSymMatrix55 const & m = ts.localError().matrix();
      
      int dim = 5; /// should check if corresponds to m
      float localErrors[15];
      
      int k = 0;
      for (int i=0; i<dim; i++) {
	for (int j=0; j<=i; j++) {
	  localErrors[k++] = m(i,j);
	}
      }
    return PTrajectoryStateOnDet(ts.localParameters(),pt,
				 localErrors, detid,
				 surfaceSide);
    }
    return PTrajectoryStateOnDet(ts.localParameters(),pt,
				 detid,
				 surfaceSide);
  }
  
  TrajectoryStateOnSurface 
  transientState( const PTrajectoryStateOnDet& ts,
		  const Surface* surface,
		  const MagneticField* field)
  {
    AlgebraicSymMatrix55 m;
    bool errInv=true;
    if (ts.hasError()) {
      errInv = false;
      int dim = 5;
      int k = 0;
      for (int i=0; i<dim; i++) {
	for (int j=0; j<=i; j++) {
	  m(i,j) = ts.error(k++);       // NOTE: here we do a cast float => double.     
	}
      }
    }
    
    
    return TrajectoryStateOnSurface( ts.parameters(),
				     errInv ? LocalTrajectoryError(InvalidError()) : LocalTrajectoryError(m),
				     *surface, field,
				     static_cast<SurfaceSide>(ts.surfaceSide()));
    
}
  
  FreeTrajectoryState initialFreeState( const reco::Track& tk,
					const MagneticField* field, bool withErr) 
  {
    Basic3DVector<float> pos( tk.vertex());
    GlobalPoint gpos( pos);
    Basic3DVector<float> mom( tk.momentum());
    GlobalVector gmom( mom);
    GlobalTrajectoryParameters par( gpos, gmom, tk.charge(), field);
    if (!withErr) return FreeTrajectoryState(par);
    CurvilinearTrajectoryError err( tk.covariance());
    return FreeTrajectoryState( par, err);
  }
  
  FreeTrajectoryState innerFreeState( const reco::Track& tk,
				      const MagneticField* field, bool withErr)
  {
    Basic3DVector<float> pos( tk.innerPosition());
    GlobalPoint gpos( pos);
    Basic3DVector<float> mom( tk.innerMomentum());
    GlobalVector gmom( mom);
    GlobalTrajectoryParameters par( gpos, gmom, tk.charge(), field);
    if (!withErr) return FreeTrajectoryState(par);
    CurvilinearTrajectoryError err( tk.extra()->innerStateCovariance());
    return FreeTrajectoryState( par, err);
  }

  
  FreeTrajectoryState outerFreeState( const reco::Track& tk,
				      const MagneticField* field, bool withErr)
  {
    Basic3DVector<float> pos( tk.outerPosition());
    GlobalPoint gpos( pos);
    Basic3DVector<float> mom( tk.outerMomentum());
    GlobalVector gmom( mom);
    GlobalTrajectoryParameters par( gpos, gmom, tk.charge(), field);
    if (!withErr) return FreeTrajectoryState(par);
    CurvilinearTrajectoryError err( tk.extra()->outerStateCovariance());
    return FreeTrajectoryState( par, err);
  }

  
  TrajectoryStateOnSurface innerStateOnSurface( const reco::Track& tk, 
						const TrackingGeometry& geom,
						const MagneticField* field, bool withErr)
  {
    const Surface& surface = geom.idToDet( DetId( tk.extra()->innerDetId()))->surface();
    return TrajectoryStateOnSurface( innerFreeState( tk, field, withErr), surface);
  }
  
  TrajectoryStateOnSurface outerStateOnSurface( const reco::Track& tk, 
						const TrackingGeometry& geom,
						const MagneticField* field, bool withErr)
  {
    const Surface& surface = geom.idToDet( DetId( tk.extra()->outerDetId()))->surface();
    return TrajectoryStateOnSurface( outerFreeState( tk, field, withErr), surface);
  }
  
}
