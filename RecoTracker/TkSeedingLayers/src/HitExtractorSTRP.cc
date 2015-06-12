#include "HitExtractorSTRP.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "Geometry/Records/interface/TrackerTopologyRcd.h"
#include "TrackingTools/DetLayers/interface/DetLayer.h"

#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/Event.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "DataFormats/Common/interface/ContainerMask.h"

#include "TrackingTools/TransientTrackingRecHit/interface/TrackingRecHitProjector.h"
#include "RecoTracker/TransientTrackingRecHit/interface/ProjectedRecHit2D.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "Geometry/Records/interface/TrackerTopologyRcd.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "DataFormats/SiStripCluster/interface/SiStripClusterTools.h"

#include<tuple>

#include<iostream>

using namespace ctfseeding;
using namespace std;
using namespace edm;

HitExtractorSTRP::HitExtractorSTRP(GeomDetEnumerators::SubDetector subdet, SeedingLayer::Side & side, int idLayer, float iminGoodCharge):
  theLayerSubDet(subdet), theSide(side), theIdLayer(idLayer),
  minAbsZ(0), theMinRing(1), theMaxRing(0),
  hasMatchedHits(false), hasRPhiHits(false), hasStereoHits(false),
  hasRingSelector(false), hasSimpleRphiHitsCleaner(true)
{ minGoodCharge=iminGoodCharge; if (minGoodCharge>0) skipClusters=true; }

void HitExtractorSTRP::useSkipClusters_(const edm::InputTag & m, edm::ConsumesCollector& iC) {
  theSkipClusters = iC.consumes<SkipClustersCollection>(m);
}

void HitExtractorSTRP::useRingSelector(int minRing, int maxRing) 
{
  hasRingSelector=true;
  theMinRing=minRing;
  theMaxRing=maxRing; 
}

bool HitExtractorSTRP::ringRange(int ring) const
{
  if (!hasRingSelector) return true;
  return (ring >= theMinRing) & (ring <= theMaxRing); 
}

bool HitExtractorSTRP::skipThis(DetId id, OmniClusterRef const& clus,
				edm::Handle<edm::ContainerMask<edmNew::DetSetVector<SiStripCluster> > > & stripClusterMask) const {

  if (maskCluster && (stripClusterMask->mask(clus.key())) ) return true;

  if unlikely(minGoodCharge<=0) return false;
  return siStripClusterTools::chargePerCM(id,*clus.cluster_strip()) <= minGoodCharge;
}



std::pair<bool,ProjectedSiStripRecHit2D *> 
HitExtractorSTRP::skipThis(const TkTransientTrackingRecHitBuilder& ttrhBuilder,
			   TkHitRef matched,
			   edm::Handle<edm::ContainerMask<edmNew::DetSetVector<SiStripCluster> > > & stripClusterMask) const {
  const SiStripMatchedRecHit2D & hit = (SiStripMatchedRecHit2D const&)(matched);
 
  assert(dynamic_cast<SiStripMatchedRecHit2D const*>(&matched));

  auto id = hit.geographicalId();  
  ProjectedSiStripRecHit2D * replaceMe = nullptr;
  bool rejectSt   = skipThis(id,hit.stereoClusterRef(), stripClusterMask);
  bool rejectMono = skipThis(id, hit.monoClusterRef(),  stripClusterMask);

  if ((!rejectSt)&(!rejectMono)){
    // keepit
    return std::make_pair(false,replaceMe);
  }

  if (failProjection || (rejectSt&rejectMono) ){
    //only skip if both hits are done
    return std::make_pair(true,replaceMe);
  }

  // replace with one

  auto cloner = ttrhBuilder.cloner();
  replaceMe = cloner.project(hit, rejectSt, TrajectoryStateOnSurface()).release();
  if (rejectSt)
    LogDebug("HitExtractorSTRP")<<"a matched hit is partially masked, and the mono hit got projected onto: "<<replaceMe->geographicalId().rawId()<<" key: "<<hit.monoClusterRef().key();
  else 
    LogDebug("HitExtractorSTRP")<<"a matched hit is partially masked, and the stereo hit got projected onto: "<<replaceMe->geographicalId().rawId()<<" key: "<<hit.stereoClusterRef().key();

  return std::make_pair(true,replaceMe);
}


void HitExtractorSTRP::cleanedOfClusters( const TkTransientTrackingRecHitBuilder& ttrhBuilder,
					  const edm::Event& ev, HitExtractor::Hits & hits,
					  bool matched,
					  unsigned int cleanFrom) const{
  LogDebug("HitExtractorPIX")<<"getting: "<<hits.size()<<" in input.";
  edm::Handle<SkipClustersCollection> stripClusterMask;
  if (maskCluster) ev.getByToken(theSkipClusters,stripClusterMask);
  unsigned int skipped=0;
  unsigned int projected=0;
  for (unsigned int iH=cleanFrom;iH<hits.size();++iH){
     assert(hits[iH]->isValid());
    auto id = hits[iH]->geographicalId();
    if (matched) {
      bool replace; ProjectedSiStripRecHit2D * replaceMe; std::tie(replace,replaceMe) = skipThis(ttrhBuilder, *hits[iH],stripClusterMask);
      if (replace) {
	if (!replaceMe) {
	  LogDebug("HitExtractorSTRP")<<"skipping a matched hit on :"<<hits[iH]->geographicalId().rawId();
	  skipped++;
	} else projected++;
	hits[iH].reset(replaceMe);
	if (replaceMe==nullptr) assert(hits[iH].empty());
	else assert(hits[iH].isOwn());
      }
    }
    else if (skipThis(id, hits[iH]->firstClusterRef(),stripClusterMask)){
      LogDebug("HitExtractorSTRP")<<"skipping a hit on :"<<hits[iH]->geographicalId().rawId()<<" key: ";
      skipped++;
      hits[iH].reset();
    }
  }
    //  remove empty elements...
  auto last = std::remove_if(hits.begin()+cleanFrom,hits.end(),[]( HitPointer const & p) {return p.empty();});
  hits.resize(last-hits.begin());

  // std::cout << "HitExtractorSTRP " <<"skipped :"<<skipped<<" strip rechits because of clusters and projected: "<<projected << std::endl;
  LogDebug("HitExtractorSTRP")<<"skipped :"<<skipped<<" strip rechits because of clusters and projected: "<<projected;
}

HitExtractor::Hits HitExtractorSTRP::hits(const TkTransientTrackingRecHitBuilder &ttrhBuilder, const edm::Event& ev, const edm::EventSetup& es) const
{
  HitExtractor::Hits result;
  unsigned int cleanFrom=0;

  //Retrieve tracker topology from geometry
  edm::ESHandle<TrackerTopology> tTopoHandle;
  es.get<TrackerTopologyRcd>().get(tTopoHandle);
  const TrackerTopology* const tTopo = tTopoHandle.product();

  //
  // TIB
  //
  if (theLayerSubDet == GeomDetEnumerators::TIB) {
    if (hasMatchedHits) {
      edm::Handle<SiStripMatchedRecHit2DCollection> matchedHits;
      ev.getByToken( theMatchedHits, matchedHits);
      if (skipClusters) cleanFrom=result.size();
      range2SeedingHits( *matchedHits, result, tTopo->tibDetIdLayerComparator(theIdLayer));
      if (skipClusters) cleanedOfClusters(ttrhBuilder, ev,result,true,cleanFrom);
    }
    if (hasRPhiHits) {
      edm::Handle<SiStripRecHit2DCollection> rphiHits;
      ev.getByToken( theRPhiHits, rphiHits);
      if (hasMatchedHits){ 
	if (!hasSimpleRphiHitsCleaner){ // this is a brutal "cleaning". Add something smarter in the future
	  if (skipClusters) cleanFrom=result.size();
          range2SeedingHits( *rphiHits, result, tTopo->tibDetIdLayerComparator(theIdLayer));
	  if (skipClusters) cleanedOfClusters(ttrhBuilder, ev,result,false,cleanFrom);
	}
      } else {
	if (skipClusters) cleanFrom=result.size();
        range2SeedingHits( *rphiHits, result, tTopo->tibDetIdLayerComparator(theIdLayer));
	if (skipClusters) cleanedOfClusters(ttrhBuilder, ev,result,false,cleanFrom);
      }
    }
    if (hasStereoHits) {
      edm::Handle<SiStripRecHit2DCollection> stereoHits;
      ev.getByToken( theStereoHits, stereoHits);
      if (skipClusters) cleanFrom=result.size();
      range2SeedingHits( *stereoHits, result, tTopo->tibDetIdLayerComparator(theIdLayer)); 
      if (skipClusters) cleanedOfClusters(ttrhBuilder, ev,result,false,cleanFrom);
    }
  }
  
  //
  // TID
  //
  else if (theLayerSubDet == GeomDetEnumerators::TID) {
      if (hasMatchedHits) {
          edm::Handle<SiStripMatchedRecHit2DCollection> matchedHits;
          ev.getByToken( theMatchedHits, matchedHits);
	  if (skipClusters) cleanFrom=result.size();
          auto getter = tTopo->tidDetIdWheelComparator(theSide,theIdLayer);
          SiStripMatchedRecHit2DCollection::Range range = matchedHits->equal_range(getter.first, getter.second);
          for (SiStripMatchedRecHit2DCollection::const_iterator it = range.first; it != range.second; ++it) {
              int ring = tTopo->tidRing( it->detId() );  if (!ringRange(ring)) continue;
              for (SiStripMatchedRecHit2DCollection::DetSet::const_iterator hit = it->begin(), end = it->end(); hit != end; ++hit) {
		result.emplace_back(*hit); 
              }
          }
	  if (skipClusters) cleanedOfClusters(ttrhBuilder, ev,result,true,cleanFrom);
      }
      if (hasRPhiHits) {
          edm::Handle<SiStripRecHit2DCollection> rphiHits;
          ev.getByToken( theRPhiHits, rphiHits);
	  if (skipClusters) cleanFrom=result.size();
          auto getter = tTopo->tidDetIdWheelComparator(theSide,theIdLayer);
          SiStripRecHit2DCollection::Range range = rphiHits->equal_range(getter.first, getter.second);
          for (SiStripRecHit2DCollection::const_iterator it = range.first; it != range.second; ++it) {
              int ring = tTopo->tidRing( it->detId() );  if (!ringRange(ring)) continue;
              if ((SiStripDetId(it->detId()).partnerDetId() != 0) && hasSimpleRphiHitsCleaner) continue;  // this is a brutal "cleaning". Add something smarter in the future
              for (SiStripRecHit2DCollection::DetSet::const_iterator hit = it->begin(), end = it->end(); hit != end; ++hit) {
                  result.emplace_back(*hit);
              }
          }
	  if (skipClusters) cleanedOfClusters(ttrhBuilder, ev,result,false,cleanFrom);
      }
      if (hasStereoHits) {
          edm::Handle<SiStripRecHit2DCollection> stereoHits;
          ev.getByToken( theStereoHits, stereoHits);
	  if (skipClusters) cleanFrom=result.size();
          auto getter = tTopo->tidDetIdWheelComparator(theSide,theIdLayer);
          SiStripRecHit2DCollection::Range range = stereoHits->equal_range(getter.first, getter.second);
          for (SiStripRecHit2DCollection::const_iterator it = range.first; it != range.second; ++it) {
              int ring = tTopo->tidRing( it->detId() );  if (!ringRange(ring)) continue;
              for (SiStripRecHit2DCollection::DetSet::const_iterator hit = it->begin(), end = it->end(); hit != end; ++hit) {
                  result.emplace_back(*hit);
              }
          }
	  if (skipClusters) cleanedOfClusters(ttrhBuilder, ev,result,false,cleanFrom);
      }
  }
  //
  // TOB
  //
  else if (theLayerSubDet == GeomDetEnumerators::TOB) {
    if (hasMatchedHits) {
      edm::Handle<SiStripMatchedRecHit2DCollection> matchedHits;
      ev.getByToken( theMatchedHits, matchedHits);
      if (skipClusters) cleanFrom=result.size();
      if (minAbsZ>0.) {
	auto getter = tTopo->tobDetIdLayerComparator(theIdLayer);
	SiStripMatchedRecHit2DCollection::Range range = matchedHits->equal_range(getter.first, getter.second);
	for (SiStripMatchedRecHit2DCollection::const_iterator it = range.first; it != range.second; ++it) {
	  for (SiStripMatchedRecHit2DCollection::DetSet::const_iterator hit = it->begin(), end = it->end(); hit != end; ++hit) {
	    if (fabs(hit->globalPosition().z())>=minAbsZ) result.emplace_back(*hit); 
	  }
	}
      } else {
	range2SeedingHits( *matchedHits, result, tTopo->tobDetIdLayerComparator(theIdLayer));
      }
      if (skipClusters) cleanedOfClusters(ttrhBuilder, ev,result,true,cleanFrom);
    }
    if (hasRPhiHits) {
      edm::Handle<SiStripRecHit2DCollection> rphiHits;
      ev.getByToken( theRPhiHits, rphiHits);
      if (hasMatchedHits){ 
	if (!hasSimpleRphiHitsCleaner){ // this is a brutal "cleaning". Add something smarter in the future
	  if (skipClusters) cleanFrom=result.size();
          range2SeedingHits( *rphiHits, result, tTopo->tobDetIdLayerComparator(theIdLayer));
	  if (skipClusters) cleanedOfClusters(ttrhBuilder, ev,result,false,cleanFrom);
	}
      } else {
	if (skipClusters) cleanFrom=result.size();
        range2SeedingHits( *rphiHits, result, tTopo->tobDetIdLayerComparator(theIdLayer));
	if (skipClusters) cleanedOfClusters(ttrhBuilder, ev,result,false,cleanFrom);
      }
    }
    if (hasStereoHits) {
      edm::Handle<SiStripRecHit2DCollection> stereoHits;
      ev.getByToken( theStereoHits, stereoHits);
      if (skipClusters) cleanFrom=result.size();
      range2SeedingHits( *stereoHits, result, tTopo->tobDetIdLayerComparator(theIdLayer));
      if (skipClusters) cleanedOfClusters(ttrhBuilder, ev,result,false,cleanFrom);
    }
  }

  //
  // TEC
  //
  else if (theLayerSubDet == GeomDetEnumerators::TEC) {
      if (hasMatchedHits) {
          edm::Handle<SiStripMatchedRecHit2DCollection> matchedHits;
          ev.getByToken( theMatchedHits, matchedHits);
	  if (skipClusters) cleanFrom=result.size();
          auto getter = tTopo->tecDetIdWheelComparator(theSide,theIdLayer);
          SiStripMatchedRecHit2DCollection::Range range = matchedHits->equal_range(getter.first, getter.second);
          for (SiStripMatchedRecHit2DCollection::const_iterator it = range.first; it != range.second; ++it) {
              int ring = tTopo->tecRing( it->detId() );  if (!ringRange(ring)) continue;
              for (SiStripMatchedRecHit2DCollection::DetSet::const_iterator hit = it->begin(), end = it->end(); hit != end; ++hit) {
                  result.emplace_back(*hit);
              }
          }
	  if (skipClusters) cleanedOfClusters(ttrhBuilder, ev,result,true,cleanFrom);
      }
      if (hasRPhiHits) {
          edm::Handle<SiStripRecHit2DCollection> rphiHits;
          ev.getByToken( theRPhiHits, rphiHits);
	  if (skipClusters) cleanFrom=result.size();
          auto getter = tTopo->tecDetIdWheelComparator(theSide,theIdLayer);
          SiStripRecHit2DCollection::Range range = rphiHits->equal_range(getter.first, getter.second);
          for (SiStripRecHit2DCollection::const_iterator it = range.first; it != range.second; ++it) {
              int ring = tTopo->tecRing( it->detId() );  if (!ringRange(ring)) continue;
              if ((SiStripDetId(it->detId()).partnerDetId() != 0) && hasSimpleRphiHitsCleaner) continue;  // this is a brutal "cleaning". Add something smarter in the future
              for (SiStripRecHit2DCollection::DetSet::const_iterator hit = it->begin(), end = it->end(); hit != end; ++hit) {
                  result.emplace_back(*hit);
              }
          }
	  if (skipClusters) cleanedOfClusters(ttrhBuilder, ev,result,false,cleanFrom);

      }
      if (hasStereoHits) {
          edm::Handle<SiStripRecHit2DCollection> stereoHits;
          ev.getByToken( theStereoHits, stereoHits);
	  if (skipClusters) cleanFrom=result.size();
          auto getter = tTopo->tecDetIdWheelComparator(theSide,theIdLayer);
          SiStripRecHit2DCollection::Range range = stereoHits->equal_range(getter.first, getter.second);
          for (SiStripRecHit2DCollection::const_iterator it = range.first; it != range.second; ++it) {
              int ring = tTopo->tecRing( it->detId() );  if (!ringRange(ring)) continue;
              for (SiStripRecHit2DCollection::DetSet::const_iterator hit = it->begin(), end = it->end(); hit != end; ++hit) {
                  result.emplace_back(*hit);
              }
          }
	  if (skipClusters) cleanedOfClusters(ttrhBuilder, ev,result,false,cleanFrom);
      }
  }

  LogDebug("HitExtractorSTRP")<<" giving: "<<result.size()<<" out";
  // std::cout << "HitExtractorSTRP "<<" giving: "<<result.size() << " for charge cut " << minGoodCharge << std::endl;
  return result;
}


