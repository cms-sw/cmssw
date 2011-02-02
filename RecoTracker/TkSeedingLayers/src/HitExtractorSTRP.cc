#include "HitExtractorSTRP.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerLayerIdAccessor.h"
#include "TrackingTools/DetLayers/interface/DetLayer.h"

#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/Event.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripMatchedRecHit2DCollection.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit2DCollection.h"

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
    hasRingSelector(false), theMinRing(1), theMaxRing(0), hasSimpleRphiHitsCleaner(true)
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

bool HitExtractorSTRP::skipThis(const SiStripRecHit2D * hit,
				edm::Handle<edmNew::DetSetVector<TkStripMeasurementDet::SiStripClusterRef> > & stripClusterRefs) const {
  static DetId lastId=hit->geographicalId();
  static edmNew::DetSetVector<TkStripMeasurementDet::SiStripClusterRef>::const_iterator f=stripClusterRefs->find(lastId.rawId());
  
  if (f==stripClusterRefs->end()) return false;
  if (!hit->isValid())  return false;

  if (hit->geographicalId()!=lastId){
    lastId=hit->geographicalId();
    f=stripClusterRefs->find(lastId.rawId());
  }
  
  return (find(f->begin(),f->end(),hit->cluster())!=f->end());
}
bool HitExtractorSTRP::skipThis(const SiStripMatchedRecHit2D * hit,
				edm::Handle<edmNew::DetSetVector<TkStripMeasurementDet::SiStripClusterRef> > & stripClusterRefs) const {
  if (skipThis(hit->stereoHit(),stripClusterRefs)) return true;
  if (skipThis(hit->monoHit(),stripClusterRefs)) return true;
  return false;
}


void HitExtractorSTRP::cleanedOfClusters( const edm::Event& ev, HitExtractor::Hits & hits,
					  bool matched)const{
  edm::Handle<edmNew::DetSetVector<TkStripMeasurementDet::SiStripClusterRef> > stripClusterRefs;
  ev.getByLabel(theSkipClusters,stripClusterRefs);
  std::vector<bool> keep(hits.size(),true);
  HitExtractor::Hits newHits;
  newHits.reserve(hits.size());
  for (unsigned int iH=0;iH!=hits.size();++iH){
    if (matched && skipThis((SiStripMatchedRecHit2D*) hits[iH]->hit(),stripClusterRefs)){
      keep[iH]=false;
      continue;
    }
    if (!matched && skipThis((SiStripRecHit2D*) hits[iH]->hit(),stripClusterRefs)){
      keep[iH]=false;
      continue;
    }
    newHits.push_back(hits[iH]);
  }
  hits.swap(newHits);
}

HitExtractor::Hits HitExtractorSTRP::hits(const SeedingLayer & sl, const edm::Event& ev, const edm::EventSetup& es) const
{
  HitExtractor::Hits result;
  TrackerLayerIdAccessor accessor;
  //
  // TIB
  //
  if (theLayer->subDetector() == GeomDetEnumerators::TIB) {
    if (hasMatchedHits) {
      edm::Handle<SiStripMatchedRecHit2DCollection> matchedHits;
      ev.getByLabel( theMatchedHits, matchedHits);
      range2SeedingHits( *matchedHits, result, accessor.stripTIBLayer(theIdLayer), sl, es); 
      if (skipClusters) cleanedOfClusters(ev,result,true);
    }
    if (hasRPhiHits) {
      edm::Handle<SiStripRecHit2DCollection> rphiHits;
      ev.getByLabel( theRPhiHits, rphiHits);
      if (hasMatchedHits){ 
	if (!hasSimpleRphiHitsCleaner){ // this is a brutal "cleaning". Add something smarter in the future
          range2SeedingHits( *rphiHits, result, accessor.stripTIBLayer(theIdLayer), sl, es); 
	  if (skipClusters) cleanedOfClusters(ev,result,false);
	}
      } else {
        range2SeedingHits( *rphiHits, result, accessor.stripTIBLayer(theIdLayer), sl, es); 
	if (skipClusters) cleanedOfClusters(ev,result,false);
      }
    }
    if (hasStereoHits) {
      edm::Handle<SiStripRecHit2DCollection> stereoHits;
      ev.getByLabel( theStereoHits, stereoHits);
      range2SeedingHits( *stereoHits, result, accessor.stripTIBLayer(theIdLayer), sl, es); 
      if (skipClusters) cleanedOfClusters(ev,result,false);
    }
  }
  
  //
  // TID
  //
  else if (theLayer->subDetector() == GeomDetEnumerators::TID) {
      if (hasMatchedHits) {
          edm::Handle<SiStripMatchedRecHit2DCollection> matchedHits;
          ev.getByLabel( theMatchedHits, matchedHits);
          std::pair<DetId,DetIdTIDSameDiskComparator> getter = accessor.stripTIDDisk(theSide,theIdLayer);
          SiStripMatchedRecHit2DCollection::Range range = matchedHits->equal_range(getter.first, getter.second);
          for (SiStripMatchedRecHit2DCollection::const_iterator it = range.first; it != range.second; ++it) {
              int ring = TIDDetId( it->detId() ).ring();  if (!ringRange(ring)) continue;
              for (SiStripMatchedRecHit2DCollection::DetSet::const_iterator hit = it->begin(), end = it->end(); hit != end; ++hit) {
		result.push_back( sl.hitBuilder()->build(hit) ); 
              }
          }
	  if (skipClusters) cleanedOfClusters(ev,result,true);
      }
      if (hasRPhiHits) {
          edm::Handle<SiStripRecHit2DCollection> rphiHits;
          ev.getByLabel( theRPhiHits, rphiHits);
          std::pair<DetId,DetIdTIDSameDiskComparator> getter = accessor.stripTIDDisk(theSide,theIdLayer);
          SiStripRecHit2DCollection::Range range = rphiHits->equal_range(getter.first, getter.second);
          for (SiStripRecHit2DCollection::const_iterator it = range.first; it != range.second; ++it) {
              int ring = TIDDetId( it->detId() ).ring();  if (!ringRange(ring)) continue;
              if ((SiStripDetId(it->detId()).partnerDetId() != 0) && hasSimpleRphiHitsCleaner) continue;  // this is a brutal "cleaning". Add something smarter in the future
              for (SiStripRecHit2DCollection::DetSet::const_iterator hit = it->begin(), end = it->end(); hit != end; ++hit) {
                  result.push_back( sl.hitBuilder()->build(hit) );
              }
          }
	  if (skipClusters) cleanedOfClusters(ev,result,false);
      }
      if (hasStereoHits) {
          edm::Handle<SiStripRecHit2DCollection> stereoHits;
          ev.getByLabel( theStereoHits, stereoHits);
          std::pair<DetId,DetIdTIDSameDiskComparator> getter = accessor.stripTIDDisk(theSide,theIdLayer);
          SiStripRecHit2DCollection::Range range = stereoHits->equal_range(getter.first, getter.second);
          for (SiStripRecHit2DCollection::const_iterator it = range.first; it != range.second; ++it) {
              int ring = TIDDetId( it->detId() ).ring();  if (!ringRange(ring)) continue;
              for (SiStripRecHit2DCollection::DetSet::const_iterator hit = it->begin(), end = it->end(); hit != end; ++hit) {
                  result.push_back( sl.hitBuilder()->build(hit) );
              }
          }
	  if (skipClusters) cleanedOfClusters(ev,result,false);
      }
  }
  //
  // TOB
  //
  else if (theLayer->subDetector() == GeomDetEnumerators::TOB) {
    if (hasMatchedHits) {
      edm::Handle<SiStripMatchedRecHit2DCollection> matchedHits;
      ev.getByLabel( theMatchedHits, matchedHits);
      range2SeedingHits( *matchedHits, result, accessor.stripTOBLayer(theIdLayer), sl, es); 
      if (skipClusters) cleanedOfClusters(ev,result,true);
    }
    if (hasRPhiHits) {
      edm::Handle<SiStripRecHit2DCollection> rphiHits;
      ev.getByLabel( theRPhiHits, rphiHits);
      if (hasMatchedHits){ 
	if (!hasSimpleRphiHitsCleaner){ // this is a brutal "cleaning". Add something smarter in the future
          range2SeedingHits( *rphiHits, result, accessor.stripTOBLayer(theIdLayer), sl, es); 
	  if (skipClusters) cleanedOfClusters(ev,result,false);
	}
      } else {
        range2SeedingHits( *rphiHits, result, accessor.stripTOBLayer(theIdLayer), sl, es); 
	if (skipClusters) cleanedOfClusters(ev,result,false);
      }
    }
    if (hasStereoHits) {
      edm::Handle<SiStripRecHit2DCollection> stereoHits;
      ev.getByLabel( theStereoHits, stereoHits);
      range2SeedingHits( *stereoHits, result, accessor.stripTOBLayer(theIdLayer), sl, es); 
      if (skipClusters) cleanedOfClusters(ev,result,false);
    }
  }

  //
  // TEC
  //
  else if (theLayer->subDetector() == GeomDetEnumerators::TEC) {
      if (hasMatchedHits) {
          edm::Handle<SiStripMatchedRecHit2DCollection> matchedHits;
          ev.getByLabel( theMatchedHits, matchedHits);
          std::pair<DetId,DetIdTECSameDiskComparator> getter = accessor.stripTECDisk(theSide,theIdLayer);
          SiStripMatchedRecHit2DCollection::Range range = matchedHits->equal_range(getter.first, getter.second);
          for (SiStripMatchedRecHit2DCollection::const_iterator it = range.first; it != range.second; ++it) {
              int ring = TECDetId( it->detId() ).ring();  if (!ringRange(ring)) continue;
              for (SiStripMatchedRecHit2DCollection::DetSet::const_iterator hit = it->begin(), end = it->end(); hit != end; ++hit) {
                  result.push_back(  sl.hitBuilder()->build(hit) );
              }
          }
	  if (skipClusters) cleanedOfClusters(ev,result,true);
      }
      if (hasRPhiHits) {
          edm::Handle<SiStripRecHit2DCollection> rphiHits;
          ev.getByLabel( theRPhiHits, rphiHits);
          std::pair<DetId,DetIdTECSameDiskComparator> getter = accessor.stripTECDisk(theSide,theIdLayer);
          SiStripRecHit2DCollection::Range range = rphiHits->equal_range(getter.first, getter.second);
          for (SiStripRecHit2DCollection::const_iterator it = range.first; it != range.second; ++it) {
              int ring = TECDetId( it->detId() ).ring();  if (!ringRange(ring)) continue;
              if ((SiStripDetId(it->detId()).partnerDetId() != 0) && hasSimpleRphiHitsCleaner) continue;  // this is a brutal "cleaning". Add something smarter in the future
              for (SiStripRecHit2DCollection::DetSet::const_iterator hit = it->begin(), end = it->end(); hit != end; ++hit) {
                  result.push_back( sl.hitBuilder()->build(hit) );
              }
          }
	  if (skipClusters) cleanedOfClusters(ev,result,false);

      }
      if (hasStereoHits) {
          edm::Handle<SiStripRecHit2DCollection> stereoHits;
          ev.getByLabel( theStereoHits, stereoHits);
          std::pair<DetId,DetIdTECSameDiskComparator> getter = accessor.stripTECDisk(theSide,theIdLayer);
          SiStripRecHit2DCollection::Range range = stereoHits->equal_range(getter.first, getter.second);
          for (SiStripRecHit2DCollection::const_iterator it = range.first; it != range.second; ++it) {
              int ring = TECDetId( it->detId() ).ring();  if (!ringRange(ring)) continue;
              for (SiStripRecHit2DCollection::DetSet::const_iterator hit = it->begin(), end = it->end(); hit != end; ++hit) {
                  result.push_back( sl.hitBuilder()->build(hit) );
              }
          }
	  if (skipClusters) cleanedOfClusters(ev,result,false);
      }
  }
  return result;
}


