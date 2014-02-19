#include "HitExtractorPIX.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerLayerIdAccessor.h"
#include "TrackingTools/DetLayers/interface/DetLayer.h"

#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/Event.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"

using namespace ctfseeding;
using namespace std;

HitExtractorPIX::HitExtractorPIX(
    SeedingLayer::Side & side, int idLayer, const std::string & hitProducer, edm::ConsumesCollector& iC)
  : theHitProducer(iC.consumes<SiPixelRecHitCollection>(hitProducer)), theSide(side), theIdLayer(idLayer)
{ }

void HitExtractorPIX::useSkipClusters_(const edm::InputTag & m, edm::ConsumesCollector& iC) {
  theSkipClusters = iC.consumes<SkipClustersCollection>(m);
}

HitExtractor::Hits HitExtractorPIX::hits(const TransientTrackingRecHitBuilder &ttrhBuilder, const edm::Event& ev, const edm::EventSetup& es) const
{
  HitExtractor::Hits result;
  TrackerLayerIdAccessor accessor;
  edm::Handle<SiPixelRecHitCollection> pixelHits;
  ev.getByToken( theHitProducer, pixelHits);
  if (theSide==SeedingLayer::Barrel) {
    range2SeedingHits( *pixelHits, result, accessor.pixelBarrelLayer(theIdLayer), ttrhBuilder, es );
  } else {
    range2SeedingHits( *pixelHits, result, accessor.pixelForwardDisk(theSide,theIdLayer), ttrhBuilder, es );
  }


  if (skipClusters){
    LogDebug("HitExtractorPIX")<<"getting : "<<result.size()<<" pixel hits.";
    //std::cout<<" skipping"<<std::endl;
    edm::Handle<SkipClustersCollection> pixelClusterMask;
    ev.getByToken(theSkipClusters,pixelClusterMask);
    std::vector<bool> keep(result.size(),true);
    HitExtractor::Hits newHits;
    unsigned int skipped=0;
    if (result.empty()) return result;
    newHits.reserve(result.size());
    for (unsigned int iH=0;iH!=result.size();++iH){
      if (result[iH]->hit()->isValid()){
        SiPixelRecHit * concrete = (SiPixelRecHit *) result[iH]->hit();
        assert(pixelClusterMask->refProd().id() == concrete->cluster().id());
        if(pixelClusterMask->mask(concrete->cluster().key())) {
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
