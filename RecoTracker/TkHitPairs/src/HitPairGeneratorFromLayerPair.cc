#include "RecoTracker/TkHitPairs/interface/HitPairGeneratorFromLayerPair.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "TrackingTools/DetLayers/interface/DetLayer.h"
#include "TrackingTools/DetLayers/interface/BarrelDetLayer.h"
#include "TrackingTools/DetLayers/interface/ForwardDetLayer.h"
#include <TrackingTools/Records/interface/TransientRecHitRecord.h>

#include "RecoTracker/TkTrackingRegions/interface/HitRZCompatibility.h"
#include "RecoTracker/TkTrackingRegions/interface/TrackingRegion.h"
#include "RecoTracker/TkTrackingRegions/interface/TrackingRegionBase.h"
#include "RecoTracker/TkHitPairs/interface/OrderedHitPairs.h"
#include "RecoTracker/TkHitPairs/interface/InnerDeltaPhi.h"
#include "RecoTracker/TkHitPairs/interface/LayerHitMapLoop.h"
#include "RecoTracker/TkHitPairs/interface/RecHitsSortedInPhi.h"

using namespace GeomDetEnumerators;
using namespace std;

typedef PixelRecoRange<float> Range;


HitPairGeneratorFromLayerPair::HitPairGeneratorFromLayerPair(const LayerWithHits* inner, 
							     const LayerWithHits* outer, 
							     LayerCacheType* layerCache, 
							     const edm::EventSetup& iSetup)
  : TTRHbuilder(0),trackerGeometry(0),theLayerCache(*layerCache), 
    theOuterLayer(outer), theInnerLayer(inner)
{
  edm::ESHandle<TrackerGeometry> tracker;
  iSetup.get<TrackerDigiGeometryRecord>().get(tracker);
  trackerGeometry = tracker.product();
}

void HitPairGeneratorFromLayerPair::hitPairs(
  const TrackingRegion & region, OrderedHitPairs & result,
  const edm::EventSetup& iSetup)
{
  if (theInnerLayer->layer()->subDetector() != PixelBarrel &&
      theInnerLayer->layer()->location() == barrel ){
    hitPairsWithErrors(region,result,iSetup);
    return;
  }
  //  static int NSee = 0; static int Ntry = 0; static int Nacc = 0;

  typedef OrderedHitPair::InnerHit InnerHit;
  typedef OrderedHitPair::OuterHit OuterHit;


  const LayerHitMap & innerHitsMap = theLayerCache(theInnerLayer, region,iSetup);
  if (innerHitsMap.empty()) return;
 
  const LayerHitMap & outerHitsMap = theLayerCache(theOuterLayer, region,iSetup);
  if (outerHitsMap.empty()) return;

  innerlay=theInnerLayer->layer();
  outerlay=theOuterLayer->layer();

  
  float outerHitErrorRPhi = (outerlay->location() == barrel) ?
      TrackingRegionBase::hitErrRPhi(
	  dynamic_cast<const BarrelDetLayer*>(outerlay) )
    : TrackingRegionBase::hitErrRPhi(
          dynamic_cast<const ForwardDetLayer*>(outerlay) ) ;




  float zMinOrigin = region.origin().z() - region.originZBound();
  float zMaxOrigin = region.origin().z() + region.originZBound();
  InnerDeltaPhi deltaPhi(*innerlay, region.ptMin(), region.originRBound(),
			 zMinOrigin, zMaxOrigin,iSetup);

  float rzLayer1, rzLayer2;

  
  if (innerlay->location() == barrel) {
    const BarrelDetLayer& bl = 
        dynamic_cast<const BarrelDetLayer&>(*innerlay);
    float halfThickness  = bl.surface().bounds().thickness()/2;
    float radius = bl.specificSurface().radius();
    rzLayer1 = radius-halfThickness;
    rzLayer2 = radius+halfThickness;


  } else {
    float halfThickness  = innerlay->surface().bounds().thickness()/2;
    float zLayer = innerlay->position().z() ;
    rzLayer1 = zLayer-halfThickness;
    rzLayer2 = zLayer+halfThickness;
  }

  const TkHitPairsCachedHit * oh;
  LayerHitMapLoop outerHits = outerHitsMap.loop();
//  static TimingReport::Item * theTimer1 =
//        PixelRecoUtilities::initTiming("--- outerHitloop ",1);
//  TimeMe tm1( *theTimer1, false);

  while ( (oh=outerHits.getHit()) ) {

    float dphi = deltaPhi( (*oh).r(), (*oh).z(), outerHitErrorRPhi);
  
    if (dphi < 0.) continue;
    PixelRecoRange<float> phiRange((*oh).phi()-dphi,(*oh).phi()+dphi);

    const HitRZCompatibility *checkRZ = region.checkRZ(&(*innerlay), oh->RecHit(),iSetup);

    if(!checkRZ) continue;
 
    Range r1 = checkRZ->range(rzLayer1);
    Range r2 = checkRZ->range(rzLayer2);
    Range rzRangeMin = r1.intersection(r2);
    Range rzRangeMax = r1.sum(r2);
 

    if ( ! rzRangeMax.empty() ) { 
      LayerHitMapLoop innerHits = innerHitsMap.loop(phiRange, rzRangeMax );
      const TkHitPairsCachedHit * ih;

//    static TimingReport::Item * theTimer4 =
//      PixelRecoUtilities::initTiming("--- innerHitloop 4",1);
//    TimeMe tm4( *theTimer4, false);

      if (rzRangeMin.empty()) {
        while ( (ih=innerHits.getHit()) ) {
          if ((*checkRZ)( ih->r(), ih->z()) ) 
	    result.push_back( OrderedHitPair( ih->RecHit(), oh->RecHit()));
        }
      } 
      else {

        bool inSafeRange = true;

        innerHits.setSafeRzRange(rzRangeMin, &inSafeRange);

        while ( (ih=innerHits.getHit()) ) {


          if (inSafeRange || (*checkRZ)( ih->r(), ih->z()) )  
	    result.push_back( OrderedHitPair(ih->RecHit(), oh->RecHit()));
          inSafeRange = true;
        }
      }
    }
    delete checkRZ;
  }
  
//  cout << "average size of inner hits: "<<NSee <<" "<<Ntry<<" "<<Nacc<<endl; 

/*
  static bool debug 
      = SimpleConfigurable<int>(0,"TkHitPairs:debugLevel").value() >= 1;

  if (debug) {
    cout << "** HitPairGeneratorFromLayerPair ** ";
    if (theInnerLayer->part()==barrel) cout <<"(B,";
    if (theInnerLayer->part()==forward) cout <<"(F,";
    if (theOuterLayer->part()==barrel) cout <<"B)";
    if (theOuterLayer->part()==forward) cout <<"F)";
    cout << " pairs: "<< result.size();
    cout << " ohits: "<< outerHits.size();
    cout <<" from (Inner,Outer) det hits: ("
         <<theInnerLayer->recHits().size()<<","
         <<theOuterLayer->recHits().size()<<")"
         <<endl;
  }
*/
}

void HitPairGeneratorFromLayerPair::
   hitPairsWithErrors( const TrackingRegion& region,
		       OrderedHitPairs & result,
		       const edm::EventSetup& iSetup)
{
  if(TTRHbuilder == 0){
    edm::ESHandle<TransientTrackingRecHitBuilder> theBuilderHandle;
    iSetup.get<TransientRecHitRecord>().get("WithoutRefit",theBuilderHandle);
    TTRHbuilder = theBuilderHandle.product();
  }

  typedef OrderedHitPair::InnerHit InnerHit;
  typedef OrderedHitPair::OuterHit OuterHit;

  //BM vector<RecHit>     outerHits(region.hits(theOuterLayer));
  vector<const TrackingRecHit*> outerHits(theOuterLayer->recHits());
  //BM  RecHitsSortedInPhi innerSortedHits(region.hits(theInnerLayer));
  RecHitsSortedInPhi innerSortedHits(theInnerLayer->recHits(),trackerGeometry);
				   
  float zMinOrigin = region.origin().z() - region.originZBound();
  float zMaxOrigin = region.origin().z() + region.originZBound();
  InnerDeltaPhi deltaPhi(
      *(theInnerLayer->layer()), region.ptMin(), region.originRBound(),
      zMinOrigin, zMaxOrigin,iSetup);

  typedef vector<const TrackingRecHit*>::const_iterator  HI;
  float nSigmaRZ = sqrt(12.);
  float nSigmaPhi = 3.;
  for (HI oh=outerHits.begin(); oh!= outerHits.end(); oh++) {
    TransientTrackingRecHit::RecHitPointer recHit = TTRHbuilder->build(*oh);
    GlobalPoint hitPos = recHit->globalPosition();
    float phiErr = nSigmaPhi * sqrt(recHit->globalPositionError().phierr(hitPos)); 
    float dphi = deltaPhi( hitPos.perp(), hitPos.z(), hitPos.perp()*phiErr);   

    float phiHit = hitPos.phi();
    vector<const TrackingRecHit*> innerCandid = innerSortedHits.hits(phiHit-dphi,phiHit+dphi);
    const HitRZCompatibility *checkRZ = region.checkRZ(theInnerLayer->layer(), *oh,iSetup);
    if(!checkRZ) continue;

    for (HI ih = innerCandid.begin(); ih != innerCandid.end(); ih++) {
      TransientTrackingRecHit::RecHitPointer recHit = TTRHbuilder->build(&(**ih));
      GlobalPoint innPos = recHit->globalPosition();
      Range allowed = checkRZ->range(innPos.perp());
      Range hitRZ;
      if (theInnerLayer->layer()->location() == barrel) {
        float zErr = nSigmaRZ * sqrt(recHit->globalPositionError().czz());
        hitRZ = Range(innPos.z()-zErr, innPos.z()+zErr);
      } else {
        float rErr = nSigmaRZ * sqrt(recHit->globalPositionError().rerr(innPos));
        hitRZ = Range(innPos.perp()-rErr, innPos.perp()+rErr);
      }
      Range crossRange = allowed.intersection(hitRZ);
      if (! crossRange.empty() ) {
        result.push_back( OrderedHitPair( *ih, *oh ) );
      }
    } 
    delete checkRZ;
  }
}

