#include "TkDetUtil.h"
#include "TrackingTools/DetLayers/interface/PhiLess.h"

#include "Geometry/CommonDetUnit/interface/GeomDet.h"
#include "TrackingTools/DetLayers/interface/MeasurementEstimator.h"
#include "DataFormats/GeometrySurface/interface/Plane.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "TrackingTools/DetLayers/interface/rangesIntersect.h"


namespace tkDetUtil {

  namespace {
    struct PhiLess {
      bool operator()(float a, float b) const {
	return Geom::phiLess(a,b);
      }
    };
  }

  bool overlapInPhi( const GlobalPoint& crossPoint,const GeomDet & det, float phiWindow) 
  {
    float phi = crossPoint.barePhi();
    std::pair<float,float> phiRange(phi-phiWindow, phi+phiWindow);
    std::pair<float,float> detPhiRange = det.surface().phiSpan(); 
    //   return rangesIntersect( phiRange, detPhiRange, boost::function<bool(float,float)>(&Geom::phiLess));
    return rangesIntersect( phiRange, detPhiRange, PhiLess());
  }
 

  float computeWindowSize( const GeomDet* det, 
			   const TrajectoryStateOnSurface& tsos, 
			   const MeasurementEstimator& est)
  {
    const Plane& startPlane = det->surface();  
    MeasurementEstimator::Local2DVector maxDistance = 
      est.maximalLocalDisplacement( tsos, startPlane);
    return calculatePhiWindow( maxDistance, tsos, startPlane);
  }



  float 
  calculatePhiWindow( const MeasurementEstimator::Local2DVector& maxDistance, 
		      const TrajectoryStateOnSurface& ts, 
		      const Plane& plane)
  {
    
    LocalPoint start = ts.localPosition();
    float corners[]  =  { plane.toGlobal(LocalPoint( start.x()+maxDistance.x(), start.y()+maxDistance.y() )).barePhi(),
			  plane.toGlobal(LocalPoint( start.x()-maxDistance.x(), start.y()+maxDistance.y() )).barePhi(),
			  plane.toGlobal(LocalPoint( start.x()-maxDistance.x(), start.y()-maxDistance.y() )).barePhi(),
			  plane.toGlobal(LocalPoint( start.x()+maxDistance.x(), start.y()-maxDistance.y() )).barePhi() 
    };
    
    float phimin = corners[0];
    float phimax = phimin;
    for ( int i = 1; i<4; i++) {
      float cPhi = corners[i];
      if ( Geom::phiLess(cPhi, phimin) ) { phimin = cPhi; }
      if ( Geom::phiLess( phimax, cPhi) ) { phimax = cPhi; }
    }
    float phiWindow = phimax - phimin;
    if ( phiWindow < 0.) { phiWindow +=  2.*Geom::pi();}
    
    return phiWindow;
  }




}
