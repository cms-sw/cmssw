#ifndef RecoTracker_TkSeedingLayers_HitExtractorPIX_H
#define RecoTracker_TkSeedingLayers_HitExtractorPIX_H

#include "RecoTracker/TkSeedingLayers/interface/SeedingLayer.h"
#include "HitExtractor.h"

#include <string>
#include <vector>

#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHitCollection.h"

namespace ctfseeding {
class HitExtractorPIX : public HitExtractor {
public:
  HitExtractorPIX( SeedingLayer::Side & side, int idLayer, const std::string & hitProducer);
  virtual ~HitExtractorPIX(){}
  virtual HitExtractor::Hits hits(const SeedingLayer & sl, const edm::Event& , const edm::EventSetup& ) const;
  virtual HitExtractorPIX * clone() const { return new HitExtractorPIX(*this); }

private:
  SeedingLayer::Side theSide;
  int theIdLayer;
  std::string theHitProducer; 
};
}
#endif
