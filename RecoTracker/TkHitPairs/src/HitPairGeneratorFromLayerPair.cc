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
  const TrackingRegion & region, OrderedHitPairs & result,
  const edm::EventSetup& iSetup)
{
//  static int NSee = 0; static int Ntry = 0; static int Nacc = 0;

  typedef OrderedHitPair::InnerHit InnerHit;
  typedef OrderedHitPair::OuterHit OuterHit;

  cout<<"d1"<<endl;

  const LayerHitMap & innerHitsMap = theLayerCache(theInnerLayer, region,iSetup);
  if (innerHitsMap.empty()) return;
 
  const LayerHitMap & outerHitsMap = theLayerCache(theOuterLayer, region,iSetup);
  if (outerHitsMap.empty()) return;

  innerlay=theInnerLayer->layer();
  outerlay=theOuterLayer->layer();
 cout<<"d2"<<endl;
  
  float outerHitErrorRPhi = (outerlay->part() == barrel) ?
      TrackingRegionBase::hitErrRPhi(
          dynamic_cast<const BarrelDetLayer*>(outerlay) )
    : TrackingRegionBase::hitErrRPhi(
          dynamic_cast<const ForwardDetLayer*>(outerlay) ) ;




  float zMinOrigin = region.origin().z() - region.originZBound();
  float zMaxOrigin = region.origin().z() + region.originZBound();
  InnerDeltaPhi deltaPhi(*innerlay, region.ptMin(), region.originRBound(),
			 zMinOrigin, zMaxOrigin,iSetup);

  float rzLayer1, rzLayer2;
 cout<<"d3"<<endl;
  
  if (innerlay->part() == barrel) {
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
cout<<"d4"<<endl;
  const TkHitPairsCachedHit * oh;
  LayerHitMapLoop outerHits = outerHitsMap.loop();
//  static TimingReport::Item * theTimer1 =
//        PixelRecoUtilities::initTiming("--- outerHitloop ",1);
//  TimeMe tm1( *theTimer1, false);
 cout<<"d5"<<endl;
  while ( (oh=outerHits.getHit()) ) {
   cout<<"q1"<<endl;
    float dphi = deltaPhi( (*oh).r(), (*oh).z(), outerHitErrorRPhi);
  
    if (dphi < 0.) continue;
    PixelRecoRange<float> phiRange((*oh).phi()-dphi,(*oh).phi()+dphi);
  cout<<"q2"<<endl;
    const HitRZCompatibility *checkRZ = region.checkRZ(&(*innerlay), oh->RecHit(),iSetup);
  cout<<"q3"<<endl;
    if(!checkRZ) continue;
 
    Range r1 = checkRZ->range(rzLayer1);
    Range r2 = checkRZ->range(rzLayer2);
    Range rzRangeMin = r1.intersection(r2);
    Range rzRangeMax = r1.sum(r2);
 
 cout<<"d6"<<endl;
    if ( ! rzRangeMax.empty() ) { 
      LayerHitMapLoop innerHits = innerHitsMap.loop(phiRange, rzRangeMax );
      const TkHitPairsCachedHit * ih;
  cout<<"h1"<<endl;
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
	cout<<"h2"<<endl;
        innerHits.setSafeRzRange(rzRangeMin, &inSafeRange);
	cout<<"h21"<<endl;
        while ( (ih=innerHits.getHit()) ) {
	  cout<<"h3 "<<endl;
	  cout<<inSafeRange<<" "<<(*checkRZ)( ih->r(), ih->z())<<endl;
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

