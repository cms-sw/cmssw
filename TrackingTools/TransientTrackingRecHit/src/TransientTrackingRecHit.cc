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
    }
  return globalPosition_;
}


GlobalError TransientTrackingRecHit::globalPositionError() const {
  return ErrorFrameTransformer().transform( localPositionError(), *surface() );
}


float 
TransientTrackingRecHit::errorGlobalR() const {
  if unlikely(! hasGlobalError_) setPositionErrors();
  return errorR_;
}

float 
TransientTrackingRecHit::errorGlobalZ() const {
  if unlikely(! hasGlobalError_) setPositionErrors();
  return errorZ_;
}

float 
TransientTrackingRecHit::errorGlobalRPhi() const {
 if unlikely(! hasGlobalError_) setPositionErrors();
 return errorRPhi_;
}

void
TransientTrackingRecHit::setPositionErrors() const {
GlobalError  
  globalError_ = ErrorFrameTransformer::transform( localPositionError(), *surface() );
  errorRPhi_ = globalPosition().perp()*sqrt(globalError_.phierr(globalPosition())); 
  errorR_ = std::sqrt(globalError_.rerr(globalPosition()));
  errorZ_ = std::sqrt(globalError_.czz());
  hasGlobalError_ = true;
}

TransientTrackingRecHit::ConstRecHitContainer TransientTrackingRecHit::transientHits() const 
{
  // no components by default
  return ConstRecHitContainer();
}

TransientTrackingRecHit::RecHitPointer 
TransientTrackingRecHit::clone( const TrajectoryStateOnSurface&) const {
  return RecHitPointer(const_cast<TransientTrackingRecHit*>(this));
}
