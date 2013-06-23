#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHit.h"
#include "DataFormats/GeometryCommonDetAlgo/interface/ErrorFrameTransformer.h"

#include "Geometry/CommonDetUnit/interface/GeomDetUnit.h"

#include "FWCore/Utilities/interface/Likely.h"


const GeomDetUnit * TransientTrackingRecHit::detUnit() const
{
  return dynamic_cast<const GeomDetUnit*>(det());
}


GlobalPoint TransientTrackingRecHit::globalPosition() const {
  if unlikely(! hasGlobalPosition_){
    globalPosition_ = surface()->toGlobal(localPosition());
    hasGlobalPosition_ = true;
    return globalPosition_;
  }else{
    return globalPosition_;
  }
}


#ifdef TTRH_NOGE
GlobalError TransientTrackingRecHit::globalPositionError() const {
  return ErrorFrameTransformer().transform( localPositionError(), *surface() );
}
#else
GlobalError TransientTrackingRecHit::globalPositionError() const {
  if unlikely(! hasGlobalError_){
    setPositionErrors();
    return globalError_;
  }else{
    return globalError_;
  }

}   
#endif

float 
TransientTrackingRecHit::errorGlobalR() const {
  if unlikely(!hasGlobalError_){
    setPositionErrors();
    return errorR_;
  }else{
    return errorR_;
  }
}

float 
TransientTrackingRecHit::errorGlobalZ() const {
  if unlikely(!hasGlobalError_){
    setPositionErrors();
    return errorZ_;
  }else{
    return errorZ_;
  }
}

float 
TransientTrackingRecHit::errorGlobalRPhi() const {
  if unlikely(!hasGlobalError_){
    setPositionErrors();
    return errorRPhi_;
  }else{
    return errorRPhi_;
  }
}

void
TransientTrackingRecHit::setPositionErrors() const {
#ifdef TTRH_NOGE
GlobalError  
#endif
  globalError_ = ErrorFrameTransformer::transform( localPositionError(), *surface() );
  errorRPhi_ = globalPosition().perp()*sqrt(globalError_.phierr(globalPosition())); 
  errorR_ = sqrt(globalError_.rerr(globalPosition()));
  errorZ_ = sqrt(globalError_.czz());
  hasGlobalError_ = true;
}

TransientTrackingRecHit::ConstRecHitContainer TransientTrackingRecHit::transientHits() const 
{
  // no components by default
  return ConstRecHitContainer();
}

TransientTrackingRecHit::RecHitPointer 
TransientTrackingRecHit::clone( const TrajectoryStateOnSurface& ts) const {
  return RecHitPointer(const_cast<TransientTrackingRecHit*>(this));
}
