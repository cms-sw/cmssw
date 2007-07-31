#ifndef RecoTracker_TkSeedingLayers_HitExtractorSTRP_H
#define RecoTracker_TkSeedingLayers_HitExtractorSTRP_H

#include "RecoTracker/TkSeedingLayers/interface/SeedingLayer.h"
#include "FWCore/ParameterSet/interface/InputTag.h"
#include "HitExtractor.h"

#include <vector>
class DetLayer;


namespace ctfseeding {

class HitExtractorSTRP : public HitExtractor {

public:
  HitExtractorSTRP( const DetLayer* detLayer,  SeedingLayer::Side & side, int idLayer);
  virtual ~HitExtractorSTRP(){}

  virtual std::vector<SeedingHit> hits( const SeedingLayer & sl, const edm::Event& , const edm::EventSetup& ) const;
  virtual HitExtractorSTRP * clone() const { return new HitExtractorSTRP(*this); }

  void useMatchedHits( const edm::InputTag & m) { hasMatchedHits = true; theMatchedHits = m; }
  void useRPhiHits(    const edm::InputTag & m) { hasRPhiHits    = true; theRPhiHits = m; }
  void useStereoHits(  const edm::InputTag & m) { hasStereoHits = true; theStereoHits = m; }
  void useRingSelector(int minRing, int maxRing);

private:
  bool ringRange(int ring) const;
  bool ringRangeTEC(const TrackingRecHit& hit) const;
  bool ringRangeNodsTEC(const TrackingRecHit& hit) const;
  bool ringRangeTID(const TrackingRecHit& hit) const;
  bool ringRangeNodsTID(const TrackingRecHit& hit) const;
private:
  const DetLayer * theLayer;
  SeedingLayer::Side theSide;
  int theIdLayer;
  bool hasMatchedHits; edm::InputTag theMatchedHits;
  bool hasRPhiHits;    edm::InputTag theRPhiHits;
  bool hasStereoHits;  edm::InputTag theStereoHits;
  bool hasRingSelector; int theMinRing, theMaxRing; 
};

}
#endif
