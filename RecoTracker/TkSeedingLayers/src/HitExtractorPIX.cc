#include "HitExtractorPIX.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerLayerIdAccessor.h"
#include "TrackingTools/DetLayers/interface/DetLayer.h"

#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/Event.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"


#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHit.h"

using namespace ctfseeding;
using namespace std;

HitExtractorPIX::HitExtractorPIX(
    SeedingLayer::Side & side, int idLayer, const std::string & hitProducer)
  : theSide(side), theIdLayer(idLayer), theHitProducer(hitProducer)
{ }

HitExtractor::Hits HitExtractorPIX::hits(const SeedingLayer & sl,const edm::Event& ev, const edm::EventSetup& es) const
{
  HitExtractor::Hits result;
  TrackerLayerIdAccessor accessor;
  edm::Handle<SiPixelRecHitCollection> pixelHits;
  ev.getByLabel( theHitProducer, pixelHits);
  if (theSide==SeedingLayer::Barrel) {
    range2SeedingHits( *pixelHits, result, accessor.pixelBarrelLayer(theIdLayer), sl, es );
  } else {
    range2SeedingHits( *pixelHits, result, accessor.pixelForwardDisk(theSide,theIdLayer), sl, es );
  }


  if (skipClusters){
    LogDebug("HitExtractorPIX")<<"getting : "<<result.size()<<" pixel hits.";
    //std::cout<<" skipping"<<std::endl;
    edm::Handle<edmNew::DetSetVector<SiPixelClusterRefNew> > pixelClusterRefs;
    ev.getByLabel(theSkipClusters,pixelClusterRefs);
    std::vector<bool> keep(result.size(),true);
    HitExtractor::Hits newHits;
    uint skipped=0;
    if (result.empty()) return result;
    DetId lookup=result.front()->hit()->geographicalId();
    edmNew::DetSetVector<SiPixelClusterRefNew>::const_iterator f=pixelClusterRefs->find(lookup);
    newHits.reserve(result.size());
    for (unsigned int iH=0;iH!=result.size();++iH){
      if (result[iH]->hit()->geographicalId()!=lookup)
	{
	  lookup=result[iH]->hit()->geographicalId();
	  f=pixelClusterRefs->find(lookup);
	}
      if (result[iH]->hit()->isValid()){
	SiPixelRecHit * concrete = (SiPixelRecHit *) result[iH]->hit();
	if (f!=pixelClusterRefs->end() && find(f->begin(),f->end(),concrete->cluster())!=f->end()){
	  //too much debug LogDebug("HitExtractorPIX")<<"skipping a pixel hit on: "<< result[iH]->hit()->geographicalId().rawId()<<" key: "<<find(f->begin(),f->end(),concrete->cluster())->key();
	  skipped++;
	  continue;
	}
      }
      newHits.push_back(result[iH]);
    }
    result.swap(newHits);
    LogDebug("HitExtractorPIX")<<"skipped :"<<skipped<<" pixel clusters";
  }
  LogDebug("HitExtractorPIX")<<"giving :"<<result.size()<<" rechits out";
  return result;
}
