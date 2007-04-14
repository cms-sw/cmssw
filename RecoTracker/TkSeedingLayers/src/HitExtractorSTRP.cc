#include "HitExtractorSTRP.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerLayerIdAccessor.h"
#include "TrackingTools/DetLayers/interface/DetLayer.h"

#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/Event.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripMatchedRecHit2DCollection.h"

#include "DataFormats/SiStripDetId/interface/TIBDetId.h"
#include "DataFormats/SiStripDetId/interface/TIDDetId.h"
#include "DataFormats/SiStripDetId/interface/TOBDetId.h"
#include "DataFormats/SiStripDetId/interface/TECDetId.h"

using namespace ctfseeding;
using namespace std;
using namespace edm;

HitExtractorSTRP::HitExtractorSTRP( const DetLayer* detLayer, 
    SeedingLayer::Side & side, int idLayer)
  : theLayer(detLayer), theSide(side), theIdLayer(idLayer),
    hasMatchedHits(false), hasRPhiHits(false), hasStereoHits(false),
    hasRingSelector(false), theMinRing(1), theMaxRing(0)
{ }

void HitExtractorSTRP::useRingSelector(int minRing, int maxRing) 
{
  hasRingSelector=true;
  theMinRing=minRing;
  theMaxRing=maxRing; 
}

bool HitExtractorSTRP::ringRange(int ring) const
{
  if (!hasRingSelector) return true;
  else if ( ring >= theMinRing && ring <= theMaxRing) return true;
  else return false;
}

vector<SeedingHit> HitExtractorSTRP::hits(const edm::Event& ev, const edm::EventSetup& es) const
{
  TrackerLayerIdAccessor accessor;
  std::vector<SeedingHit> result;

  //
  // TIB
  //
  if (theLayer->subDetector() == GeomDetEnumerators::TIB) {
    if (hasMatchedHits) {
      edm::Handle<SiStripMatchedRecHit2DCollection> matchedHits;
      ev.getByLabel( theMatchedHits, matchedHits);
      const SiStripMatchedRecHit2DCollection::range range =
        matchedHits->get(accessor.stripTIBLayer(theIdLayer) );
      for(SiStripMatchedRecHit2DCollection::const_iterator it=range.first; it!=range.second; it++){
        result.push_back( SeedingHit(&(*it), es) );
      }
    }
    if (hasRPhiHits) {
//  ...
    }
    if (hasStereoHits) {
//  ...
    }
  }

  //
  // TID
  //
  else if (theLayer->subDetector() == GeomDetEnumerators::TID) {
    if (hasMatchedHits) {
      edm::Handle<SiStripMatchedRecHit2DCollection> matchedHits;
      ev.getByLabel( theMatchedHits, matchedHits);
      const SiStripMatchedRecHit2DCollection::range range =
          matchedHits->get(accessor.stripTIDDisk(theSide,theIdLayer) );
      for(SiStripMatchedRecHit2DCollection::const_iterator it=range.first; it!=range.second; it++){
        int ring = TIDDetId( it->geographicalId() ).ring();
        if (ringRange(ring))result.push_back( SeedingHit(&(*it), es) );
      }
    }
    if (hasRPhiHits) {
//  ...
    }
    if (hasStereoHits) {
//  ...
    }
  }

  //
  // TOB
  //

  //
  // TEC
  //


  return result;
}


