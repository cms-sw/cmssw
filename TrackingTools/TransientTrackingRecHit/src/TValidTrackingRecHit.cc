#include "TrackingTools/TransientTrackingRecHit/interface/TValidTrackingRecHit.h"
#include "DataFormats/GeometryCommonDetAlgo/interface/ErrorFrameTransformer.h"

GlobalPoint TValidTrackingRecHit::globalPosition() const {
  if unlikely(! hasGlobalPosition_){
      globalPosition_ = surface()->toGlobal(localPosition());
      hasGlobalPosition_ = true;
    }
  return globalPosition_;
}


GlobalError TValidTrackingRecHit::globalPositionError() const {
  return ErrorFrameTransformer().transform( localPositionError(), *surface() );
}


float 
TValidTrackingRecHit::errorGlobalR() const {
  if unlikely(! hasGlobalError_) setPositionErrors();
  return errorR_;
}

float 
TValidTrackingRecHit::errorGlobalZ() const {
  if unlikely(! hasGlobalError_) setPositionErrors();
  return errorZ_;
}

float 
TValidTrackingRecHit::errorGlobalRPhi() const {
 if unlikely(! hasGlobalError_) setPositionErrors();
 return errorRPhi_;
}

void
TValidTrackingRecHit::setPositionErrors() const {
GlobalError  
  globalError_ = ErrorFrameTransformer::transform( localPositionError(), *surface() );
  errorRPhi_ = globalPosition().perp()*sqrt(globalError_.phierr(globalPosition())); 
  errorR_ = std::sqrt(globalError_.rerr(globalPosition()));
  errorZ_ = std::sqrt(globalError_.czz());
  hasGlobalError_ = true;
}

