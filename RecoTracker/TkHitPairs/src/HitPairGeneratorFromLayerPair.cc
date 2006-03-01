#include "RecoTracker/TkHitPairs/interface/HitPairGeneratorFromLayerPair.h"
#include "RecoTracker/TkTrackingRegions/interface/TrackingRegion.h"

//#include "CommonDet/BasicDet/interface/RecHit.h"
#include "TrackingTools/DetLayers/interface/DetLayer.h"
#include "TrackingTools/DetLayers/interface/BarrelDetLayer.h"
#include "TrackingTools/DetLayers/interface/ForwardDetLayer.h"

#include "RecoTracker/TkHitPairs/interface/OrderedHitPairs.h"
//#include "RecoTracker/TkHitPairs/interface/LayerHitsCache.h"
#include "RecoTracker/TkHitPairs/interface/InnerDeltaPhi.h"
#include "RecoTracker/TkTrackingRegions/interface/HitRZCompatibility.h"

#include "RecoTracker/TkTrackingRegions/interface/TrackingRegionBase.h"
#include "RecoTracker/TkHitPairs/interface/LayerHitMapLoop.h"


typedef PixelRecoRange<float> Range;

void HitPairGeneratorFromLayerPair::hitPairs(
  const TrackingRegion & region, OrderedHitPairs & result)
{
//  static int NSee = 0; static int Ntry = 0; static int Nacc = 0;

  typedef OrderedHitPair::InnerHit InnerHit;
  typedef OrderedHitPair::OuterHit OuterHit;

  //MP
 


  const LayerHitMap & innerHitsMap = theLayerCache(theInnerLayer, region);
  if (innerHitsMap.empty()) return;
  const LayerHitMap & outerHitsMap = theLayerCache(theOuterLayer, region);
  if (outerHitsMap.empty()) return;

  
//   float outerHitErrorRPhi = (theOuterLayer->part() == barrel) ?
//       TrackingRegionBase::hitErrRPhi(
//           dynamic_cast<const BarrelDetLayer*>(theOuterLayer) )
//     : TrackingRegionBase::hitErrRPhi(
//           dynamic_cast<const ForwardDetLayer*>(theOuterLayer) ) ;

  float outerHitErrorRPhi =  
    TrackingRegionBase::hitErrRPhi(innerlay);


  float zMinOrigin = region.origin().z() - region.originZBound();
  float zMaxOrigin = region.origin().z() + region.originZBound();
  InnerDeltaPhi deltaPhi(*innerlay, region.ptMin(), region.originRBound(),
			 zMinOrigin, zMaxOrigin);

  float rzLayer1, rzLayer2;
  //MP
  //  if (theInnerLayer->part() == barrel) {
    const BarrelDetLayer& bl = 
        dynamic_cast<const BarrelDetLayer&>(*innerlay);
    float halfThickness  = bl.surface().bounds().thickness()/2;
    float radius = bl.specificSurface().radius();
    rzLayer1 = radius-halfThickness;
    rzLayer2 = radius+halfThickness;
//   } else {
//     float halfThickness  = theInnerLayer->surface().bounds().thickness()/2;
//     float zLayer = theInnerLayer->position().z() ;
//     rzLayer1 = zLayer-halfThickness;
//     rzLayer2 = zLayer+halfThickness;
//   }

  const TkHitPairsCachedHit * oh;
  LayerHitMapLoop outerHits = outerHitsMap.loop();
//  static TimingReport::Item * theTimer1 =
//        PixelRecoUtilities::initTiming("--- outerHitloop ",1);
//  TimeMe tm1( *theTimer1, false);

  while ( (oh=outerHits.getHit()) ) {
    float dphi = deltaPhi( (*oh).r(), (*oh).z(), outerHitErrorRPhi);
    if (dphi < 0.) continue;
    PixelRecoRange<float> phiRange((*oh).phi()-dphi,(*oh).phi()+dphi);
    const HitRZCompatibility *checkRZ = region.checkRZ(&(*innerlay), oh->RecHit());
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

