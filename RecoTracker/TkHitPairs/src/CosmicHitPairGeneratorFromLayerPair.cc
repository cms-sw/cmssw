#include "RecoTracker/TkHitPairs/interface/CosmicHitPairGeneratorFromLayerPair.h"
#include "RecoTracker/TkTrackingRegions/interface/TrackingRegion.h"
#include "TrackingTools/DetLayers/interface/DetLayer.h"
#include "RecoTracker/TkHitPairs/interface/OrderedHitPairs.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "TrackingTools/DetLayers/interface/BarrelDetLayer.h"

CosmicHitPairGeneratorFromLayerPair::CosmicHitPairGeneratorFromLayerPair(const LayerWithHits* inner, 
							     const LayerWithHits* outer, 
									 //							     LayerCacheType* layerCache, 
							     const edm::EventSetup& iSetup)
  : TTRHbuilder(0),trackerGeometry(0),
    //theLayerCache(*layerCache), 
    theOuterLayer(outer), theInnerLayer(inner)
{

  edm::ESHandle<TrackerGeometry> tracker;
  iSetup.get<TrackerDigiGeometryRecord>().get(tracker);
  trackerGeometry = tracker.product();
}
void CosmicHitPairGeneratorFromLayerPair::hitPairs(
  const TrackingRegion & region, OrderedHitPairs & result,
  const edm::EventSetup& iSetup)
{
//  static int NSee = 0; static int Ntry = 0; static int Nacc = 0;


  typedef OrderedHitPair::InnerHit InnerHit;
  typedef OrderedHitPair::OuterHit OuterHit;


  if (theInnerLayer->recHits().empty()) return;

  if (theOuterLayer->recHits().empty()) return;
  //  const DetLayer* innerlay=theOuterLayer->layer();
  // const BarrelDetLayer *pippo=dynamic_cast<const BarrelDetLayer*>(theOuterLayer->layer());

  float radius1 =dynamic_cast<const BarrelDetLayer*>(theInnerLayer->layer())->specificSurface().radius();
  float radius2 =dynamic_cast<const BarrelDetLayer*>(theOuterLayer->layer())->specificSurface().radius();

  //check if the seed is from overlaps or not
  bool seedfromoverlaps=(abs(radius1-radius2)<0.1) ? true : false;

 
  vector<OrderedHitPair> allthepairs;
  



  std::vector<const TrackingRecHit*>::const_iterator ohh;
  for(ohh=theOuterLayer->recHits().begin();ohh!=theOuterLayer->recHits().end();ohh++){
    const TkHitPairsCachedHit * oh=new TkHitPairsCachedHit(*ohh,iSetup);
    std::vector<const TrackingRecHit*>::const_iterator ihh;
    for(ihh=theInnerLayer->recHits().begin();ihh!=theInnerLayer->recHits().end();ihh++){
      const TkHitPairsCachedHit * ih=new TkHitPairsCachedHit(*ihh,iSetup);
      
      float z_diff =ih->z()-oh->z();
      float inny=ih->r()*sin(ih->phi());
      float outy=oh->r()*sin(oh->phi());
      float innx=ih->r()*cos(ih->phi());
      float outx=oh->r()*cos(oh->phi());
      float dxdy=abs((outx-innx)/(outy-inny));
      float DeltaR=oh->r()-ih->r();
      
      if( (abs(z_diff)<30)
	  //&&((abs(inny-outy))<30) 
	  &&(dxdy<2)
	  &&(inny*outy>0)
	  && (abs(DeltaR)>0)) {

	if (seedfromoverlaps){
	  //this part of code works for MTCC
	  // for the other geometries must be verified
	  //Overlaps in the difference in z is decreased and the difference in phi is
	  //less than 0.05
	  if ((DeltaR<0)&&(abs(z_diff)<18)&&(abs(ih->phi()-oh->phi())<0.05)&&(dxdy<2)) allthepairs.push_back( OrderedHitPair(ih->RecHit(), oh->RecHit()));
	}
	else  allthepairs.push_back( OrderedHitPair(ih->RecHit(), oh->RecHit()));
    } 
   }
    
  }
  stable_sort(allthepairs.begin(),allthepairs.end(),CompareHitPairsY(iSetup));
  //Seed from overlaps are saved only if 
  //no others have been saved

  if (allthepairs.size()>0) {
    if (seedfromoverlaps) {
      if (result.size()==0) result.push_back(allthepairs[0]);
    }
    else result.push_back(allthepairs[0]);
  }

}

