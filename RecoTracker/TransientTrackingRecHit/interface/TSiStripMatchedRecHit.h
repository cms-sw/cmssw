#ifndef RECOTRACKER_TRANSIENTRACKINGRECHIT_TSiStripMatchedRecHit_H
#define RECOTRACKER_TRANSIENTRACKINGRECHIT_TSiStripMatchedRecHit_H

#include "TrackingTools/TransientTrackingRecHit/interface/GenericTransientTrackingRecHit.h"
#include "TrackingTools/TransientTrackingRecHit/interface/HelpertRecHit2DLocalPos.h"
#include "RecoLocalTracker/SiStripRecHitConverter/interface/SiStripRecHitMatcher.h"
#include "RecoLocalTracker/ClusterParameterEstimator/interface/StripClusterParameterEstimator.h"
#include<memory>

#include "Geometry/TrackerGeometryBuilder/interface/GluedGeomDet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

class TSiStripMatchedRecHit GCC11_FINAL : public GenericTransientTrackingRecHit{
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
			       bool computeCoarseLocalPosition=false) {
    return RecHitPointer( new TSiStripMatchedRecHit( geom, rh, matcher,cpe, computeCoarseLocalPosition));
  }

  static RecHitPointer build( const GeomDet * geom, std::auto_ptr<TrackingRecHit> rh, 
			      const SiStripRecHitMatcher *matcher,
			      const StripClusterParameterEstimator* cpe=0,
			      bool computeCoarseLocalPosition=false) {
    return RecHitPointer( new TSiStripMatchedRecHit( geom, rh, matcher,cpe, computeCoarseLocalPosition));
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
                              bool computeCoarseLocalPosition=false) {
        if (memory.get()) {
            memory->~TSiStripMatchedRecHit(); // call destructor
            new (memory.get()) TSiStripMatchedRecHit( geom, rh, matcher,cpe, computeCoarseLocalPosition, DontCloneRecHit());
        } else {
            memory.reset(new TSiStripMatchedRecHit( geom, rh, matcher,cpe,computeCoarseLocalPosition, DontCloneRecHit()));
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

private:
  TSiStripMatchedRecHit (const GeomDet * geom, const TrackingRecHit * rh, 
			 const SiStripRecHitMatcher *matcher,
			 const StripClusterParameterEstimator* cpe,
			  bool computeCoarseLocalPosition) : 
    GenericTransientTrackingRecHit(geom, *rh), theMatcher(matcher),theCPE(cpe) {
    if (computeCoarseLocalPosition) ComputeCoarseLocalPosition();
  }

  TSiStripMatchedRecHit (const GeomDet * geom, std::auto_ptr<TrackingRecHit> rh,
			 const SiStripRecHitMatcher *matcher,
			 const StripClusterParameterEstimator* cpe,
			 bool computeCoarseLocalPosition) : 
    GenericTransientTrackingRecHit(geom, rh.release()), theMatcher(matcher),theCPE(cpe) {
    if (computeCoarseLocalPosition) ComputeCoarseLocalPosition();
  }
  TSiStripMatchedRecHit (const GeomDet * geom, const TrackingRecHit * rh, 
			 const SiStripRecHitMatcher *matcher,
			 const StripClusterParameterEstimator* cpe,
			 bool computeCoarseLocalPosition,
                         const DontCloneRecHit &) : 
    GenericTransientTrackingRecHit(geom, const_cast<TrackingRecHit *>(rh)), theMatcher(matcher),theCPE(cpe) {
    if (computeCoarseLocalPosition) ComputeCoarseLocalPosition();
  }

private:
  void ComputeCoarseLocalPosition();

  virtual TSiStripMatchedRecHit* clone() const {
    return new TSiStripMatchedRecHit(*this);
  }

};



#endif
