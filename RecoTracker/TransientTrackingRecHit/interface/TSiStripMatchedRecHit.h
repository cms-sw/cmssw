#ifndef RECOTRACKER_TRANSIENTRACKINGRECHIT_TSiStripMatchedRecHit_H
#define RECOTRACKER_TRANSIENTRACKINGRECHIT_TSiStripMatchedRecHit_H

#include "TrackingTools/TransientTrackingRecHit/interface/GenericTransientTrackingRecHit.h"
#include "TrackingTools/TransientTrackingRecHit/interface/HelpertRecHit2DLocalPos.h"
#include "RecoLocalTracker/SiStripRecHitConverter/interface/SiStripRecHitMatcher.h"
#include "RecoLocalTracker/ClusterParameterEstimator/interface/StripClusterParameterEstimator.h"
#include<memory>

#include "Geometry/TrackerGeometryBuilder/interface/GluedGeomDet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

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
			      float weight=1., float annealing=1.,
			      bool computeCoarseLocalPosition=false) {
    return RecHitPointer( new TSiStripMatchedRecHit( geom, rh, matcher,cpe, weight, annealing, computeCoarseLocalPosition));
  }

  static RecHitPointer build( const GeomDet * geom, std::auto_ptr<TrackingRecHit> rh, 
			      const SiStripRecHitMatcher *matcher,
			      const StripClusterParameterEstimator* cpe=0,
			      float weight=1., float annealing=1.,
			      bool computeCoarseLocalPosition=false) {
    return RecHitPointer( new TSiStripMatchedRecHit( geom, rh, matcher,cpe,weight, annealing, computeCoarseLocalPosition));
  }

  virtual RecHitPointer clone( const TrajectoryStateOnSurface& ts) const;
  virtual bool canImproveWithTrack() const {return (theMatcher != 0);}
  virtual ConstRecHitContainer 	transientHits () const;

  /// Dummy struct to pass to the constructor to say 'please don't clone the hit'
  struct DontCloneRecHit {};

  /// Build this hit on the heap, but possibly starting from already allocated memory.
  /// if 'memory' is not null, it will call the placed delete, and then the placed new to make a new hit 
  /// if 'memory' is null, it will fill it with a new heap-allocated hit
  /// both at entry and exit of this method any rechit in 'memory' DOES NOT own it's persistent rechit
  static void buildInPlace(std::auto_ptr<TSiStripMatchedRecHit> &memory,
                              const GeomDet * geom, const TrackingRecHit * rh,
                              const SiStripRecHitMatcher *matcher,
                              const StripClusterParameterEstimator* cpe=0,
                              float weight=1., float annealing=1.,
                              bool computeCoarseLocalPosition=false) {
        if (memory.get()) {
            memory->~TSiStripMatchedRecHit(); // call destructor
            new (memory.get()) TSiStripMatchedRecHit( geom, rh, matcher,cpe,weight, annealing, computeCoarseLocalPosition, DontCloneRecHit());
        } else {
            memory.reset(new TSiStripMatchedRecHit( geom, rh, matcher,cpe,weight, annealing, computeCoarseLocalPosition, DontCloneRecHit()));
        }
  }

  /// take ownership of the hit, if it wasn't owned (note: if it was owned, this code will leak it)
  void clonePersistentHit()  { trackingRecHit_ = trackingRecHit_->clone(); }
  /// drop the pointer to the hit, so that it's not deleted by the destructor.
  /// you must call this before deleting the TSiStripMatchedRecHit IF AND ONLY IF it doesn't own the rechit
  void clearPersistentHit() { trackingRecHit_ = 0; }

private:
  const SiStripRecHitMatcher* theMatcher; 
  const StripClusterParameterEstimator* theCPE;
  TSiStripMatchedRecHit (const GeomDet * geom, const TrackingRecHit * rh, 
			 const SiStripRecHitMatcher *matcher,
			 const StripClusterParameterEstimator* cpe,
			 float weight, float annealing,
			 bool computeCoarseLocalPosition) : 
    GenericTransientTrackingRecHit(geom, *rh, weight, annealing), theMatcher(matcher),theCPE(cpe) {
    if (computeCoarseLocalPosition) ComputeCoarseLocalPosition();
  }

  TSiStripMatchedRecHit (const GeomDet * geom, std::auto_ptr<TrackingRecHit> rh,
			 const SiStripRecHitMatcher *matcher,
			 const StripClusterParameterEstimator* cpe,
			 float weight, float annealing,
			 bool computeCoarseLocalPosition) : 
    GenericTransientTrackingRecHit(geom, rh.release(), weight, annealing), theMatcher(matcher),theCPE(cpe) {
    if (computeCoarseLocalPosition) ComputeCoarseLocalPosition();
  }
  TSiStripMatchedRecHit (const GeomDet * geom, const TrackingRecHit * rh, 
			 const SiStripRecHitMatcher *matcher,
			 const StripClusterParameterEstimator* cpe,
			 float weight, float annealing,
			 bool computeCoarseLocalPosition,
                         const DontCloneRecHit &) : 
    GenericTransientTrackingRecHit(geom, const_cast<TrackingRecHit *>(rh), weight, annealing), theMatcher(matcher),theCPE(cpe) {
    if (computeCoarseLocalPosition) ComputeCoarseLocalPosition();
  }

   
    void ComputeCoarseLocalPosition(){
      if (!theCPE || !theMatcher) return;
      const SiStripMatchedRecHit2D *orig = static_cast<const SiStripMatchedRecHit2D *> (trackingRecHit_);
      if (orig && !orig->hasPositionAndError()){
	LogDebug("TSiStripMatchedRecHit")<<"calculating coarse position/error.";
	const GeomDet *det = this->det();
	const GluedGeomDet *gdet = static_cast<const GluedGeomDet *> (det);
	LocalVector tkDir = det->surface().toLocal( det->position()-GlobalPoint(0,0,0));
	
	const SiStripMatchedRecHit2D* better=0;
	
	if(!orig->monoHit()->cluster().isNull()){
	  const SiStripCluster& monoclust   = *orig->monoHit()->cluster();  
	  const SiStripCluster& stereoclust = *orig->stereoHit()->cluster();
	  
	  StripClusterParameterEstimator::LocalValues lvMono = 
	    theCPE->localParameters( monoclust, *gdet->monoDet());
	  StripClusterParameterEstimator::LocalValues lvStereo = 
	    theCPE->localParameters( stereoclust, *gdet->stereoDet());
	  
	  SiStripRecHit2D monoHit = SiStripRecHit2D( lvMono.first, lvMono.second,
						     gdet->monoDet()->geographicalId(),
						     orig->monoHit()->cluster());
	  
	  SiStripRecHit2D stereoHit = SiStripRecHit2D( lvStereo.first, lvStereo.second,
						       gdet->stereoDet()->geographicalId(),
						       orig->stereoHit()->cluster());
	  better =  theMatcher->match(&monoHit,&stereoHit,gdet,tkDir);
	}else{
	  const SiStripCluster& monoclust   = *orig->monoHit()->cluster_regional();  
	  const SiStripCluster& stereoclust = *orig->stereoHit()->cluster_regional();
	  StripClusterParameterEstimator::LocalValues lvMono = 
	    theCPE->localParameters( monoclust, *gdet->monoDet());
	  StripClusterParameterEstimator::LocalValues lvStereo = 
	    theCPE->localParameters( stereoclust, *gdet->stereoDet());
	  
	  SiStripRecHit2D monoHit = SiStripRecHit2D( lvMono.first, lvMono.second,
						     gdet->monoDet()->geographicalId(),
						     orig->monoHit()->cluster_regional());
	  
	  SiStripRecHit2D stereoHit = SiStripRecHit2D( lvStereo.first, lvStereo.second,
						       gdet->stereoDet()->geographicalId(),
						       orig->stereoHit()->cluster_regional());
	  better =  theMatcher->match(&monoHit,&stereoHit,gdet,tkDir);
	  
	}
	if (!better) {
	  edm::LogWarning("TSiStripMatchedRecHit")<<"could not get a matching rechit.";
	}else{
	  trackingRecHit_ = better->clone();
	}
      }
    }

  virtual TSiStripMatchedRecHit* clone() const {
    return new TSiStripMatchedRecHit(*this);
  }

};



#endif
