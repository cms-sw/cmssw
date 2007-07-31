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

#include <boost/ref.hpp>
#include <boost/bind.hpp>
#include <boost/function.hpp>

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

bool HitExtractorSTRP::ringRangeTID(const TrackingRecHit& hit) const {
  return ringRange(TIDDetId(hit.geographicalId() ).ring());
}

bool HitExtractorSTRP::ringRangeNodsTID(const TrackingRecHit& hit) const {
  int ring = TIDDetId(hit.geographicalId() ).ring();
  return ringRange(ring) &&
    (!hasMatchedHits || (ring!=1 && ring!=2));
}

bool HitExtractorSTRP::ringRangeTEC(const TrackingRecHit& hit) const {
  return ringRange(TECDetId(hit.geographicalId() ).ring());
}

bool HitExtractorSTRP::ringRangeNodsTEC(const TrackingRecHit& hit) const {
  int ring = TECDetId(hit.geographicalId() ).ring();
  return ringRange(ring) &&
    (!hasMatchedHits || (ring!=1 && ring!=2 && ring!=5));
}


namespace {

  template <typename C, typename A, typename B>
  typename C::Range rangeFromPair(C const & v, std::pair<A,B> const & p) {
    return v.equal_range(p.first,p.second);
  }

  template <typename C, typename A, typename B, typename F>
  void foreachHit(C const & v, std::pair<A,B> const & p, F & f) {
    typename C::Range range = rangeFromPair(v,p);
    for(typename C::const_iterator id=range.first; id!=range.second; id++)
      std::for_each((*id).begin(), (*id).end(), boost::ref(f));
  }

  bool True(const TrackingRecHit&) { return true;}

  
  struct Add {
    Add(const SeedingLayer & isl, const edm::EventSetup& ies) : sl(isl), es(ies), cond(True){}
    void operator()(const TrackingRecHit & hit) {
      if (cond(hit)) result.push_back(SeedingHit(&hit, sl, es) );
    }
    
   
    std::vector<SeedingHit> result;
    const SeedingLayer &      sl;
    const edm::EventSetup &   es;
    boost::function<bool(const TrackingRecHit&)> cond;
    
  private:
    // just make sure
    Add(Add const &){}
    Add & operator=(Add&){return *this;}

  }; 

}

vector<SeedingHit> HitExtractorSTRP::hits(const SeedingLayer & sl, const edm::Event& ev, const edm::EventSetup& es) const
{
  TrackerLayerIdAccessor accessor;
  Add add(sl,es);
 
  edm::Handle<SiStripMatchedRecHit2DCollection> matchedHits;
  if (hasMatchedHits) ev.getByLabel( theMatchedHits, matchedHits);
  edm::Handle<SiStripRecHit2DCollection> rphiHits;
  if (hasRPhiHits) ev.getByLabel( theRPhiHits, rphiHits);
  edm::Handle<SiStripRecHit2DCollection> stereoHits;
  if (hasStereoHits) ev.getByLabel( theStereoHits, stereoHits);
  
  //
  // TIB
  //
  if (theLayer->subDetector() == GeomDetEnumerators::TIB) {
    if (hasMatchedHits) {
      foreachHit(*matchedHits,accessor.stripTIBLayer(theIdLayer),add);
    }
    if (hasRPhiHits) {
      //se ho anche i matched non voglio gli rphi dei layer ds
      if ((!hasMatchedHits) || (theIdLayer != 1 && theIdLayer != 2) )
	foreachHit(*rphiHits,accessor.stripTIBLayer(theIdLayer),add );
    }
    if (hasStereoHits) {
     foreachHit(*stereoHits,accessor.stripTIBLayer(theIdLayer),add );
    }
  }

  //
  // TID
  //
  else if (theLayer->subDetector() == GeomDetEnumerators::TID) {
    if (hasMatchedHits) {
      add.cond=boost::bind(HitExtractorSTRP::ringRangeTID,this); // error prone, shall change
      foreachHit(*matchedHits,accessor.stripTIDDisk(theSide,theIdLayer),add);
      add.cond=True;
    }
    if (hasRPhiHits) {
      add.cond=boost::bind(HitExtractorSTRP::ringRangeNodsTID,this); // error prone, shall change
      foreachHit(*rphiHits,accessor.stripTIDDisk(theSide,theIdLayer),add);
      add.cond=True;  
    }
    if (hasStereoHits) {
      add.cond=boost::bind(HitExtractorSTRP::ringRangeTID,this); // error prone, shall change
      foreachHit(*stereoHits,accessor.stripTIDDisk(theSide,theIdLayer),add);
      add.cond=True;  
    }
  }
  //
  // TOB
  //
  else if (theLayer->subDetector() == GeomDetEnumerators::TOB) {
    if (hasMatchedHits) {
	foreachHit(*matchedHits,accessor.stripTOBLayer(theIdLayer),add );
    }
    if (hasRPhiHits) {
      if ((!hasMatchedHits) || (theIdLayer != 1 && theIdLayer != 2) )
	foreachHit(*rphiHits,accessor.stripTOBLayer(theIdLayer),add );
    }
    if (hasStereoHits) {
      foreachHit(*stereoHits,accessor.stripTOBLayer(theIdLayer),add );
    }
  }

  //
  // TEC
  //
  else if (theLayer->subDetector() == GeomDetEnumerators::TEC) {
    if (hasMatchedHits) {
      add.cond=boost::bind(HitExtractorSTRP::ringRangeTEC,this); // error prone, shall change
      foreachHit(*matchedHits,accessor.stripTECDisk(theSide,theIdLayer),add);
      add.cond=True;
    }
    if (hasRPhiHits) {
      add.cond=boost::bind(HitExtractorSTRP::ringRangeNodsTEC,this); // error prone, shall change
      foreachHit(*rphiHits,accessor.stripTECDisk(theSide,theIdLayer),add);
      add.cond=True;  
    }
    if (hasStereoHits) {
      add.cond=boost::bind(HitExtractorSTRP::ringRangeTEC,this); // error prone, shall change
      foreachHit(*stereoHits,accessor.stripTECDisk(theSide,theIdLayer),add);
      add.cond=True;  
    }
    return add.result;
}


