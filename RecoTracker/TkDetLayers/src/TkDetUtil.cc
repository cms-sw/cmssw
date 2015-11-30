#include "TkDetUtil.h"

#include "TrackingTools/DetLayers/interface/MeasurementEstimator.h"
#include "DataFormats/GeometrySurface/interface/Plane.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"

namespace tkDetUtil {

  float computeWindowSize( const GeomDet* det, 
			   const TrajectoryStateOnSurface& tsos, 
			   const MeasurementEstimator& est)
  {
    const Plane& startPlane = det->surface();  
    auto maxDistance =  est.maximalLocalDisplacement( tsos, startPlane);
    return calculatePhiWindow( maxDistance, tsos, startPlane);
  }


  float 
  calculatePhiWindow( const MeasurementEstimator::Local2DVector & imaxDistance, 
		      const TrajectoryStateOnSurface& ts, 
		      const Plane& plane)
  {
    MeasurementEstimator::Local2DVector maxDistance(std::abs(imaxDistance.x()),std::abs(imaxDistance.y()));

    constexpr float tollerance=1.e-6;
    LocalPoint start = ts.localPosition();
    //     std::cout << "plane z " << plane.normalVector() << std::endl;
    float dphi=0;
    if likely(std::abs(1.f-std::abs(plane.normalVector().z()))<tollerance) {
      auto ori = plane.toLocal(GlobalPoint(0.,0.,0.));
      auto x0 = std::abs(start.x() - ori.x());
      auto y0 = std::abs(start.y() - ori.y());
      
      if (y0<maxDistance.y() && x0<maxDistance.x()) return M_PI;

      if (y0>maxDistance.y()) {
        auto phimax = std::atan2(y0 + (x0<maxDistance.x() ? -maxDistance.y() : maxDistance.y()), x0 - maxDistance.x() );
        auto phimin = std::atan2(y0 - maxDistance.y(), x0 + maxDistance.x() );
        if (phimin>phimax) std::cout << "phimess x " << phimin<<','<<phimax << " " << x0 << ',' << maxDistance.x() << " " << y0 << ',' << maxDistance.y() << std::endl;
        dphi=phimax-phimin;
      } else {
        auto phimax = std::atan2(x0 - maxDistance.x(), -y0 - maxDistance.y() );
        auto phimin = std::atan2(x0 - maxDistance.x(), -y0 + maxDistance.y() );
        if (phimin>phimax) std::cout << "phimess y  " << phimin<<','<<phimax << std::endl;
        dphi=phimax-phimin;
      }
      return dphi;
    }
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
    // std::cout << "phiWindow " << phiWindow << ' ' << dphi << ' ' << dphi-phiWindow  << std::endl;
    return phiWindow;
  }




}
