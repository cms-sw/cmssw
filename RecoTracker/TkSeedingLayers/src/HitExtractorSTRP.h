#ifndef RecoTracker_TkSeedingLayers_HitExtractorSTRP_H
#define RecoTracker_TkSeedingLayers_HitExtractorSTRP_H

#include "RecoTracker/TkSeedingLayers/interface/SeedingLayer.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "HitExtractor.h"

#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit2D.h"

#include <vector>
class DetLayer;

namespace edm {
  template< typename T> class ContainerMask;
}

namespace ctfseeding {

class HitExtractorSTRP : public HitExtractor {

public:
  typedef SiStripRecHit2D::ClusterRef SiStripClusterRef;

  HitExtractorSTRP( const DetLayer* detLayer,  SeedingLayer::Side & side, int idLayer);
  virtual ~HitExtractorSTRP(){}

  virtual HitExtractor::Hits hits( const SeedingLayer & sl, const edm::Event& , const edm::EventSetup& ) const;
  virtual HitExtractorSTRP * clone() const { return new HitExtractorSTRP(*this); }

  void useMatchedHits( const edm::InputTag & m) { hasMatchedHits = true; theMatchedHits = m; }
  void useRPhiHits(    const edm::InputTag & m) { hasRPhiHits    = true; theRPhiHits = m; }
  void useStereoHits(  const edm::InputTag & m) { hasStereoHits = true; theStereoHits = m; }
  void useRingSelector(int minRing, int maxRing);
  void useSimpleRphiHitsCleaner(bool use) {hasSimpleRphiHitsCleaner = use;}

  void cleanedOfClusters( const edm::Event& ev, HitExtractor::Hits & hits, bool matched, unsigned int cleanFrom=0) const;

  bool skipThis(TransientTrackingRecHit::ConstRecHitPointer & ptr,edm::Handle<edm::ContainerMask<edmNew::DetSetVector<SiStripCluster> > > & stripClusterMask,
		TransientTrackingRecHit::ConstRecHitPointer & replaceMe) const;

  bool skipThis(OmniClusterRef const& clus, edm::Handle<edm::ContainerMask<edmNew::DetSetVector<SiStripCluster> > > & stripClusterMask) const;

  void project(TransientTrackingRecHit::ConstRecHitPointer & ptr,
	       const SiStripRecHit2D * hit,
	       TransientTrackingRecHit::ConstRecHitPointer & replaceMe) const;
  void setNoProjection() const {failProjection=true;};
  void setMinAbsZ(double minZToSet) {minAbsZ=minZToSet;}
private:
  bool ringRange(int ring) const;
private:
  const DetLayer * theLayer;
  SeedingLayer::Side theSide;
  mutable const SeedingLayer * theSLayer;
  int theIdLayer;
  bool hasMatchedHits; edm::InputTag theMatchedHits;
  bool hasRPhiHits;    edm::InputTag theRPhiHits;
  bool hasStereoHits;  edm::InputTag theStereoHits;
  bool hasRingSelector; int theMinRing, theMaxRing; 
  bool hasSimpleRphiHitsCleaner;
  double minAbsZ;
  mutable bool failProjection;
};

}
#endif
