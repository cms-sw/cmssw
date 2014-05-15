#include "HitExtractorSTRP.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerLayerIdAccessor.h"
#include "TrackingTools/DetLayers/interface/DetLayer.h"

#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/Event.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripMatchedRecHit2DCollection.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit2DCollection.h"

#include "DataFormats/Common/interface/ContainerMask.h"

#include "TrackingTools/TransientTrackingRecHit/interface/TrackingRecHitProjector.h"
#include "RecoTracker/TransientTrackingRecHit/interface/ProjectedRecHit2D.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "FWCore/Framework/interface/ESHandle.h"


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

bool HitExtractorSTRP::skipThis(OmniClusterRef const& clus,
				edm::Handle<edm::ContainerMask<edmNew::DetSetVector<SiStripCluster> > > & stripClusterMask) const {
  //  if (!hit->isValid())  return false;

  return stripClusterMask->mask(clus.key());
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
				edm::Handle<edm::ContainerMask<edmNew::DetSetVector<SiStripCluster> > > & stripClusterMask,
				TransientTrackingRecHit::ConstRecHitPointer & replaceMe) const {
  const SiStripMatchedRecHit2D * hit = (SiStripMatchedRecHit2D *) ptr->hit();

  bool rejectSt   = skipThis(hit->stereoClusterRef(), stripClusterMask);
  bool rejectMono = skipThis(hit->monoClusterRef(),  stripClusterMask);

  if (rejectSt&&rejectMono){
    //only skip if both hits are done
    return true;
  }
  else{
    //FIX use clusters directly
    auto const & s= hit->stereoHit();
    auto const & m= hit->monoHit();
    if (rejectSt) project(ptr,&s,replaceMe);
    else if (rejectMono) project(ptr,&m,replaceMe);
    if (!replaceMe) return true; //means that the projection failed, and needs to be skipped
    if (rejectSt)
      LogDebug("HitExtractorSTRP")<<"a matched hit is partially masked, and the mono hit got projected onto: "<<replaceMe->hit()->geographicalId().rawId()<<" key: "<<hit->monoClusterRef().key();
    else if (rejectMono)
      LogDebug("HitExtractorSTRP")<<"a matched hit is partially masked, and the stereo hit got projected onto: "<<replaceMe->hit()->geographicalId().rawId()<<" key: "<<hit->stereoClusterRef().key();
    return false; //means the projection succeeded or nothing to be masked, no need to skip and replaceMe is going to be used anyways.
  }
  return false;
}


void HitExtractorSTRP::cleanedOfClusters( const edm::Event& ev, HitExtractor::Hits & hits,
					  bool matched,
					  unsigned int cleanFrom)const{
  LogDebug("HitExtractorPIX")<<"getting: "<<hits.size()<<" in input.";
  edm::Handle<edm::ContainerMask<edmNew::DetSetVector<SiStripCluster> > > stripClusterMask;
  ev.getByLabel(theSkipClusters,stripClusterMask);
  HitExtractor::Hits newHits;
  unsigned int skipped=0;
  unsigned int projected=0;
  newHits.reserve(hits.size());
  TransientTrackingRecHit::ConstRecHitPointer replaceMe;
  for (unsigned int iH=cleanFrom;iH<hits.size();++iH){
    if (!hits[iH]->isValid()) continue;
    replaceMe=hits[iH];
    if (matched && skipThis(hits[iH],stripClusterMask,replaceMe)){
      LogDebug("HitExtractorSTRP")<<"skipping a matched hit on :"<<hits[iH]->hit()->geographicalId().rawId();
      skipped++;
      continue;
    }
    if (!matched && skipThis( ((TrackerSingleRecHit const *)(hits[iH]->hit()))->omniClusterRef(),stripClusterMask)){
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
  unsigned int cleanFrom=0;

  //Retrieve tracker topology from geometry
  edm::ESHandle<TrackerTopology> tTopoHandle;
  es.get<IdealGeometryRecord>().get(tTopoHandle);
  const TrackerTopology* const tTopo = tTopoHandle.product();
  
  //
  // TIB
  //
  if (theLayer->subDetector() == GeomDetEnumerators::TIB) {
    if (hasMatchedHits) {
      edm::Handle<SiStripMatchedRecHit2DCollection> matchedHits;
      ev.getByLabel( theMatchedHits, matchedHits);
      if (skipClusters) cleanFrom=result.size();
      range2SeedingHits( *matchedHits, result, accessor.stripTIBLayer(theIdLayer), sl, es); 
      if (skipClusters) cleanedOfClusters(ev,result,true,cleanFrom);
    }
    if (hasRPhiHits) {
      edm::Handle<SiStripRecHit2DCollection> rphiHits;
      ev.getByLabel( theRPhiHits, rphiHits);
      if (hasMatchedHits){ 
	if (!hasSimpleRphiHitsCleaner){ // this is a brutal "cleaning". Add something smarter in the future
	  if (skipClusters) cleanFrom=result.size();
          range2SeedingHits( *rphiHits, result, accessor.stripTIBLayer(theIdLayer), sl, es); 
	  if (skipClusters) cleanedOfClusters(ev,result,false,cleanFrom);
	}
      } else {
	if (skipClusters) cleanFrom=result.size();
        range2SeedingHits( *rphiHits, result, accessor.stripTIBLayer(theIdLayer), sl, es); 
	if (skipClusters) cleanedOfClusters(ev,result,false,cleanFrom);
      }
    }
    if (hasStereoHits) {
      edm::Handle<SiStripRecHit2DCollection> stereoHits;
      ev.getByLabel( theStereoHits, stereoHits);
      if (skipClusters) cleanFrom=result.size();
      range2SeedingHits( *stereoHits, result, accessor.stripTIBLayer(theIdLayer), sl, es); 
      if (skipClusters) cleanedOfClusters(ev,result,false,cleanFrom);
    }
  }
  
  //
  // TID
  //
  else if (theLayer->subDetector() == GeomDetEnumerators::TID) {
      if (hasMatchedHits) {
          edm::Handle<SiStripMatchedRecHit2DCollection> matchedHits;
          ev.getByLabel( theMatchedHits, matchedHits);
	  if (skipClusters) cleanFrom=result.size();
          std::pair<DetId,DetIdTIDSameDiskComparator> getter = accessor.stripTIDDisk(theSide,theIdLayer);
          SiStripMatchedRecHit2DCollection::Range range = matchedHits->equal_range(getter.first, getter.second);
          for (SiStripMatchedRecHit2DCollection::const_iterator it = range.first; it != range.second; ++it) {
              int ring = tTopo->tidRing( it->detId() );  if (!ringRange(ring)) continue;
              for (SiStripMatchedRecHit2DCollection::DetSet::const_iterator hit = it->begin(), end = it->end(); hit != end; ++hit) {
		result.push_back( sl.hitBuilder()->build(hit) ); 
              }
          }
	  if (skipClusters) cleanedOfClusters(ev,result,true,cleanFrom);
      }
      if (hasRPhiHits) {
          edm::Handle<SiStripRecHit2DCollection> rphiHits;
          ev.getByLabel( theRPhiHits, rphiHits);
	  if (skipClusters) cleanFrom=result.size();
          std::pair<DetId,DetIdTIDSameDiskComparator> getter = accessor.stripTIDDisk(theSide,theIdLayer);
          SiStripRecHit2DCollection::Range range = rphiHits->equal_range(getter.first, getter.second);
          for (SiStripRecHit2DCollection::const_iterator it = range.first; it != range.second; ++it) {
              int ring = tTopo->tidRing( it->detId() );  if (!ringRange(ring)) continue;
              if ((SiStripDetId(it->detId()).partnerDetId() != 0) && hasSimpleRphiHitsCleaner) continue;  // this is a brutal "cleaning". Add something smarter in the future
              for (SiStripRecHit2DCollection::DetSet::const_iterator hit = it->begin(), end = it->end(); hit != end; ++hit) {
                  result.push_back( sl.hitBuilder()->build(hit) );
              }
          }
	  if (skipClusters) cleanedOfClusters(ev,result,false,cleanFrom);
      }
      if (hasStereoHits) {
          edm::Handle<SiStripRecHit2DCollection> stereoHits;
          ev.getByLabel( theStereoHits, stereoHits);
	  if (skipClusters) cleanFrom=result.size();
          std::pair<DetId,DetIdTIDSameDiskComparator> getter = accessor.stripTIDDisk(theSide,theIdLayer);
          SiStripRecHit2DCollection::Range range = stereoHits->equal_range(getter.first, getter.second);
          for (SiStripRecHit2DCollection::const_iterator it = range.first; it != range.second; ++it) {
              int ring = tTopo->tidRing( it->detId() );  if (!ringRange(ring)) continue;
              for (SiStripRecHit2DCollection::DetSet::const_iterator hit = it->begin(), end = it->end(); hit != end; ++hit) {
                  result.push_back( sl.hitBuilder()->build(hit) );
              }
          }
	  if (skipClusters) cleanedOfClusters(ev,result,false,cleanFrom);
      }
  }
  //
  // TOB
  //
  else if (theLayer->subDetector() == GeomDetEnumerators::TOB) {
    if (hasMatchedHits) {
      edm::Handle<SiStripMatchedRecHit2DCollection> matchedHits;
      ev.getByLabel( theMatchedHits, matchedHits);
      if (skipClusters) cleanFrom=result.size();
      range2SeedingHits( *matchedHits, result, accessor.stripTOBLayer(theIdLayer), sl, es); 
      if (skipClusters) cleanedOfClusters(ev,result,true,cleanFrom);
    }
    if (hasRPhiHits) {
      edm::Handle<SiStripRecHit2DCollection> rphiHits;
      ev.getByLabel( theRPhiHits, rphiHits);
      if (hasMatchedHits){ 
	if (!hasSimpleRphiHitsCleaner){ // this is a brutal "cleaning". Add something smarter in the future
	  if (skipClusters) cleanFrom=result.size();
          range2SeedingHits( *rphiHits, result, accessor.stripTOBLayer(theIdLayer), sl, es); 
	  if (skipClusters) cleanedOfClusters(ev,result,false,cleanFrom);
	}
      } else {
	if (skipClusters) cleanFrom=result.size();
        range2SeedingHits( *rphiHits, result, accessor.stripTOBLayer(theIdLayer), sl, es); 
	if (skipClusters) cleanedOfClusters(ev,result,false,cleanFrom);
      }
    }
    if (hasStereoHits) {
      edm::Handle<SiStripRecHit2DCollection> stereoHits;
      ev.getByLabel( theStereoHits, stereoHits);
      if (skipClusters) cleanFrom=result.size();
      range2SeedingHits( *stereoHits, result, accessor.stripTOBLayer(theIdLayer), sl, es); 
      if (skipClusters) cleanedOfClusters(ev,result,false,cleanFrom);
    }
  }

  //
  // TEC
  //
  else if (theLayer->subDetector() == GeomDetEnumerators::TEC) {
      if (hasMatchedHits) {
          edm::Handle<SiStripMatchedRecHit2DCollection> matchedHits;
          ev.getByLabel( theMatchedHits, matchedHits);
	  if (skipClusters) cleanFrom=result.size();
          std::pair<DetId,DetIdTECSameDiskComparator> getter = accessor.stripTECDisk(theSide,theIdLayer);
          SiStripMatchedRecHit2DCollection::Range range = matchedHits->equal_range(getter.first, getter.second);
          for (SiStripMatchedRecHit2DCollection::const_iterator it = range.first; it != range.second; ++it) {
              int ring = tTopo->tecRing( it->detId() );  if (!ringRange(ring)) continue;
              for (SiStripMatchedRecHit2DCollection::DetSet::const_iterator hit = it->begin(), end = it->end(); hit != end; ++hit) {
                  result.push_back(  sl.hitBuilder()->build(hit) );
              }
          }
	  if (skipClusters) cleanedOfClusters(ev,result,true,cleanFrom);
      }
      if (hasRPhiHits) {
          edm::Handle<SiStripRecHit2DCollection> rphiHits;
          ev.getByLabel( theRPhiHits, rphiHits);
	  if (skipClusters) cleanFrom=result.size();
          std::pair<DetId,DetIdTECSameDiskComparator> getter = accessor.stripTECDisk(theSide,theIdLayer);
          SiStripRecHit2DCollection::Range range = rphiHits->equal_range(getter.first, getter.second);
          for (SiStripRecHit2DCollection::const_iterator it = range.first; it != range.second; ++it) {
              int ring = tTopo->tecRing( it->detId() );  if (!ringRange(ring)) continue;
              if ((SiStripDetId(it->detId()).partnerDetId() != 0) && hasSimpleRphiHitsCleaner) continue;  // this is a brutal "cleaning". Add something smarter in the future
              for (SiStripRecHit2DCollection::DetSet::const_iterator hit = it->begin(), end = it->end(); hit != end; ++hit) {
                  result.push_back( sl.hitBuilder()->build(hit) );
              }
          }
	  if (skipClusters) cleanedOfClusters(ev,result,false,cleanFrom);

      }
      if (hasStereoHits) {
          edm::Handle<SiStripRecHit2DCollection> stereoHits;
          ev.getByLabel( theStereoHits, stereoHits);
	  if (skipClusters) cleanFrom=result.size();
          std::pair<DetId,DetIdTECSameDiskComparator> getter = accessor.stripTECDisk(theSide,theIdLayer);
          SiStripRecHit2DCollection::Range range = stereoHits->equal_range(getter.first, getter.second);
          for (SiStripRecHit2DCollection::const_iterator it = range.first; it != range.second; ++it) {
              int ring = tTopo->tecRing( it->detId() );  if (!ringRange(ring)) continue;
              for (SiStripRecHit2DCollection::DetSet::const_iterator hit = it->begin(), end = it->end(); hit != end; ++hit) {
                  result.push_back( sl.hitBuilder()->build(hit) );
              }
          }
	  if (skipClusters) cleanedOfClusters(ev,result,false,cleanFrom);
      }
  }
  LogDebug("HitExtractorSTRP")<<" giving: "<<result.size()<<" out";
  return result;
}


