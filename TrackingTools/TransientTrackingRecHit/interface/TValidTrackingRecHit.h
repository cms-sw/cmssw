#ifndef TValidTrackingRecHit_H
#define TValidTrackingRecHit_H

#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHit.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHitGlobalState.h"
#include "DataFormats/GeometryCommonDetAlgo/interface/ErrorFrameTransformer.h"

class GeomDetUnit;


/**
 *
 *  only for valid hits
 */
class TValidTrackingRecHit : public TransientTrackingRecHit {
public:
  
  TValidTrackingRecHit(const GeomDet & geom) : 
  TrackingRecHit(geom) {}


  template<typename... Args>
  TValidTrackingRecHit(Args && ...args) : 
    TrackingRecHit(std::forward<Args>(args)...) {}

  // to be moved in children
  TrackingRecHit * cloneHit() const { return hit()->clone();}

  // Extension of the TrackingRecHit interface
  virtual const Surface * surface() const GCC11_FINAL {return &(det()->surface());}


  virtual GlobalPoint globalPosition() const GCC11_FINAL {
      return surface()->toGlobal(localPosition());
  }
  
  GlobalError globalPositionError() const GCC11_FINAL { return ErrorFrameTransformer().transform( localPositionError(), *surface() );}
  float errorGlobalR() const GCC11_FINAL { return std::sqrt(globalPositionError().rerr(globalPosition()));}
  float errorGlobalZ() const GCC11_FINAL { return std::sqrt(globalPositionError().czz()); }
  float errorGlobalRPhi() const GCC11_FINAL { return globalPosition().perp()*sqrt(globalPositionError().phierr(globalPosition())); }

  // once cache removed will obsolete the above
  TrackingRecHitGlobalState globalState() const {
    GlobalError  
      globalError = ErrorFrameTransformer::transform( localPositionError(), *surface() );
    auto gp = globalPosition();
    float r = gp.perp();
    float errorRPhi = r*std::sqrt(float(globalError.phierr(gp))); 
    float errorR = std::sqrt(float(globalError.rerr(gp)));
    float errorZ = std::sqrt(float(globalError.czz()));
    return (TrackingRecHitGlobalState){
      gp.basicVector(), r, gp.barePhi(),
	errorR,errorZ,errorRPhi
	};
  }


  /// Returns true if the clone( const TrajectoryStateOnSurface&) method returns an
  /// improved hit, false if it returns an identical copy.
  /// In order to avoid redundent copies one should call canImproveWithTrack() before 
  /// calling clone( const TrajectoryStateOnSurface&).
  virtual bool canImproveWithTrack() const {return false;}

 
/// cluster probability, overloaded by pixel rechits.
  virtual float clusterProbability() const { return 1.f; }

private:
 
  // hide the clone method for ReferenceCounted. Warning: this method is still 
  // accessible via the bas class TrackingRecHit interface!
  virtual TValidTrackingRecHit * clone() const = 0;

};

#endif

