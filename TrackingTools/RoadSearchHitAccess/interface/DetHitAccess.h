#ifndef RoadSearch_DetHitAccess_h
#define RoadSearch_DetHitAccess_h

#include <string>

#include "FWCore/Framework/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit2DCollection.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripMatchedRecHit2DCollection.h"
#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHitCollection.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/SiStripDetId/interface/StripSubdetector.h"
#include "DataFormats/SiPixelDetId/interface/PixelSubdetector.h"
#include "DataFormats/SiStripDetId/interface/TECDetId.h"

class DetHitAccess {

 public:

  DetHitAccess(const SiStripRecHit2DCollection* rphiRecHits,
	       const SiStripRecHit2DCollection* stereoRecHits,
	       const SiStripMatchedRecHit2DCollection* matchedRecHits,
	       const SiPixelRecHitCollection* pixelRecHits);
  
  edm::OwnVector<TrackingRecHit> getHitVector(const DetId* detid);

 private:
  
  const SiStripRecHit2DCollection* rphiHits_;
  const SiStripRecHit2DCollection* stereoHits_;
  const SiStripMatchedRecHit2DCollection * matchedHits_;
  const SiPixelRecHitCollection *pixelHits_;
};

#endif
