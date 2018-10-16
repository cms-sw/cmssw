#ifndef TkDetLayers_BarrelUtil_h
#define TkDetLayers_BarrelUtil_h


#include "Geometry/CommonDetUnit/interface/GeomDet.h"

#include "TrackingTools/DetLayers/interface/GeometricSearchDet.h"
#include "TrackingTools/DetLayers/interface/MeasurementEstimator.h"
#include "TrackingTools/DetLayers/interface/rangesIntersect.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "DataFormats/GeometryVector/interface/VectorUtil.h"

#pragma GCC visibility push(hidden)
namespace barrelUtil {

  inline
  float calculatePhiWindow( float Xmax, const GeomDet& det,
			    const TrajectoryStateOnSurface& state) {
    
    LocalPoint startPoint = state.localPosition();
    LocalVector shift( Xmax , 0. , 0.);
    LocalPoint shift1 = startPoint + shift;
    LocalPoint shift2 = startPoint + (-shift); 
    //LocalPoint shift2( startPoint); //original code;
    //shift2 -= shift;
    
    auto phi1 = det.surface().toGlobal(shift1).barePhi();
    auto phi2 = det.surface().toGlobal(shift2).barePhi();
    auto phiStart = state.globalPosition().barePhi();
    auto phiWin = std::min(std::abs(phiStart-phi1),std::abs(phiStart-phi2));
    
    return phiWin;
  }
  
  inline
  float computeWindowSize( const GeomDet* det, 
			   const TrajectoryStateOnSurface& tsos, 
			   const MeasurementEstimator& est) {
    auto xmax = 
      est.maximalLocalDisplacement(tsos, det->surface()).x();
    return calculatePhiWindow( xmax, *det, tsos);
  }
  
  
  
  inline
  bool overlap(float phi, const GeometricSearchDet& gsdet, float phiWin) {
    // introduce offset (extrapolated point and true propagated point differ by 0.0003 - 0.00033, 
    // due to thickness of Rod of 1 cm) 
    constexpr float phiOffset = 0.00034;  //...TOBE CHECKED LATER...
    phiWin += phiOffset;
    
    // detector phi range
    std::pair<float,float> phiRange(phi-phiWin, phi+phiWin);
    
    return rangesIntersect(phiRange, gsdet.surface().phiSpan(),
            [](auto x, auto y){ return Geom::phiLess(x,y); });
  } 
  

}


#pragma GCC visibility pop
#endif 
