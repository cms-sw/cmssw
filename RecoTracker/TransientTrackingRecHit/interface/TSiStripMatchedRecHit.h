#ifndef RECOTRACKER_TRANSIENTRACKINGRECHIT_TSiStripMatchedRecHit_H
#define RECOTRACKER_TRANSIENTRACKINGRECHIT_TSiStripMatchedRecHit_H

#include "TrackingTools/TransientTrackingRecHit/interface/GenericTransientTrackingRecHit.h"
#include "TrackingTools/TransientTrackingRecHit/interface/HelpertRecHit2DLocalPos.h"
#include "RecoLocalTracker/SiStripRecHitConverter/interface/SiStripRecHitMatcher.h"
#include "RecoLocalTracker/ClusterParameterEstimator/interface/StripClusterParameterEstimator.h"
#include<memory>

class TSiStripMatchedRecHit : public GenericTransientTrackingRecHit{
public:

  virtual void getKfComponents( KfComponentsHolder & holder ) const {
      HelpertRecHit2DLocalPos().getKfComponents(holder, *hit(), *det()); 
  }

  virtual AlgebraicSymMatrix parametersError() const {
    return HelpertRecHit2DLocalPos().parError( localPositionError(), *det()); 
  }

  const GeomDetUnit* detUnit() const {return 0;}

  static RecHitPointer build( const GeomDet * geom, const TrackingRecHit * rh, 
			      const SiStripRecHitMatcher *matcher,
			      const StripClusterParameterEstimator* cpe=0,
			      float weight=1., float annealing=1.) {
    return RecHitPointer( new TSiStripMatchedRecHit( geom, rh, matcher,cpe, weight, annealing));
  }

  static RecHitPointer build( const GeomDet * geom, std::auto_ptr<TrackingRecHit> rh, 
			      const SiStripRecHitMatcher *matcher,
			      const StripClusterParameterEstimator* cpe=0,
			      float weight=1., float annealing=1.) {
    return RecHitPointer( new TSiStripMatchedRecHit( geom, rh, matcher,cpe,weight, annealing));
  }

  virtual RecHitPointer clone( const TrajectoryStateOnSurface& ts) const;
  virtual bool canImproveWithTrack() const {return (theMatcher != 0);}
  virtual ConstRecHitContainer 	transientHits () const;
private:
  const SiStripRecHitMatcher* theMatcher; 
  const StripClusterParameterEstimator* theCPE;
  TSiStripMatchedRecHit (const GeomDet * geom, const TrackingRecHit * rh, 
			 const SiStripRecHitMatcher *matcher,
			 const StripClusterParameterEstimator* cpe,
			 float weight, float annealing) : 
     GenericTransientTrackingRecHit(geom, *rh, weight, annealing), theMatcher(matcher),theCPE(cpe) {}

  TSiStripMatchedRecHit (const GeomDet * geom, std::auto_ptr<TrackingRecHit> rh,
			 const SiStripRecHitMatcher *matcher,
			 const StripClusterParameterEstimator* cpe,
			 float weight, float annealing) : 
    GenericTransientTrackingRecHit(geom, rh.release(), weight, annealing), theMatcher(matcher),theCPE(cpe) {}

  virtual TSiStripMatchedRecHit* clone() const {
    return new TSiStripMatchedRecHit(*this);
  }

};



#endif
