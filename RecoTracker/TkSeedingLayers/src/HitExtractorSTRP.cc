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

#include "TrackingTools/TransientTrackingRecHit/interface/TrackingRecHitProjector.h"
#include "RecoTracker/TransientTrackingRecHit/interface/ProjectedRecHit2D.h"

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
				edm::Handle<edmNew::DetSetVector<SiStripClusterRef> > & stripClusterRefs) const {
  static DetId lastId=hit->geographicalId();
  static edmNew::DetSetVector<SiStripClusterRef>::const_iterator f=stripClusterRefs->find(lastId.rawId());
  if (hit->geographicalId()!=lastId){
    lastId=hit->geographicalId();
    f=stripClusterRefs->find(lastId.rawId());
  }  
  if (f==stripClusterRefs->end()) return false;
  if (!hit->isValid())  return false;

  bool skipping=(find(f->begin(),f->end(),hit->cluster())!=f->end());
  //if (skipping) LogDebug("HitExtractorSTRP")<<"skipping a hit on :"<<hit->geographicalId().rawId()<<" key: "<<hit->cluster().key();
  return skipping;
}


void HitExtractorSTRP::project(TransientTrackingRecHit::ConstRecHitPointer & ptr,
			       const SiStripRecHit2D * hit,
			       TransientTrackingRecHit::ConstRecHitPointer & replaceMe) const{
  
  if (failProjection) {replaceMe=0; return;}
  TrackingRecHitProjector<ProjectedRecHit2D> proj;
  TransientTrackingRecHit::RecHitPointer sHit=theSLayer->hitBuilder()->build(hit);
  replaceMe=proj.project( *sHit, *ptr->det());
  if (!replaceMe) LogDebug("HitExtractorSTRP")<<"projection failed.";
}

bool HitExtractorSTRP::skipThis(TransientTrackingRecHit::ConstRecHitPointer & ptr,
				edm::Handle<edmNew::DetSetVector<SiStripClusterRef> > & stripClusterRefs,
				TransientTrackingRecHit::ConstRecHitPointer & replaceMe) const {
  const SiStripMatchedRecHit2D * hit = (SiStripMatchedRecHit2D *) ptr->hit();

  bool rejectSt=false,rejectMono=false;
  if (skipThis(hit->stereoHit(),stripClusterRefs))  rejectSt=true;
  if (skipThis(hit->monoHit(),stripClusterRefs))    rejectMono=true;

  if (rejectSt&&rejectMono){
    //only skip if both hits are done
    return true;
  }
  else{
    if (rejectSt) project(ptr,hit->stereoHit(),replaceMe);
    else if (rejectMono) project(ptr,hit->monoHit(),replaceMe);
    if (!replaceMe) return true; //means that the projection failed, and needs to be skipped
    if (rejectSt)
      LogDebug("HitExtractorSTRP")<<"a matched hit is partially masked, and the mono hit got projected onto: "<<replaceMe->hit()->geographicalId().rawId()<<" key: "<<hit->monoHit()->cluster().key();
    else if (rejectMono)
      LogDebug("HitExtractorSTRP")<<"a matched hit is partially masked, and the stereo hit got projected onto: "<<replaceMe->hit()->geographicalId().rawId()<<" key: "<<hit->stereoHit()->cluster().key();
    return false; //means the projection succeeded or nothing to be masked, no need to skip and replaceMe is going to be used anyways.
  }
  return false;
}


void HitExtractorSTRP::cleanedOfClusters( const edm::Event& ev, HitExtractor::Hits & hits,
					  bool matched)const{
  LogDebug("HitExtractorPIX")<<"getting: "<<hits.size()<<" in input.";
  edm::Handle<edmNew::DetSetVector<SiStripClusterRef> > stripClusterRefs;
  ev.getByLabel(theSkipClusters,stripClusterRefs);
  HitExtractor::Hits newHits;
  uint skipped=0;
  uint projected=0;
  newHits.reserve(hits.size());
  TransientTrackingRecHit::ConstRecHitPointer replaceMe;
  for (unsigned int iH=0;iH!=hits.size();++iH){
    replaceMe=hits[iH];
    if (matched && skipThis(hits[iH],stripClusterRefs,replaceMe)){
      LogDebug("HitExtractorSTRP")<<"skipping a matched hit on :"<<hits[iH]->hit()->geographicalId().rawId();
      skipped++;
      continue;
    }
    if (!matched && skipThis((SiStripRecHit2D*) hits[iH]->hit(),stripClusterRefs)){
      LogDebug("HitExtractorSTRP")<<"skipping a hit on :"<<hits[iH]->hit()->geographicalId().rawId()<<" key: ";
      skipped++;
      continue;
    }
    if (replaceMe!=hits[iH]) projected++;
    newHits.push_back(replaceMe);
  }
  LogDebug("HitExtractorPIX")<<"skipped :"<<skipped<<" strip rechits because of clusters and projected: "<<projected;
  hits.swap(newHits);
}

HitExtractor::Hits HitExtractorSTRP::hits(const SeedingLayer & sl, const edm::Event& ev, const edm::EventSetup& es) const
{
  HitExtractor::Hits result;
  TrackerLayerIdAccessor accessor;
  theSLayer=&sl;
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
  LogDebug("HitExtractorSTRP")<<" giving: "<<result.size()<<" out";
  return result;
}


