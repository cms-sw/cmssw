#include "RecoTracker/TkHitPairs/interface/CosmicHitPairGeneratorFromLayerPair.h"
#include "RecoTracker/TkTrackingRegions/interface/TrackingRegion.h"
#include "TrackingTools/DetLayers/interface/DetLayer.h"
#include "RecoTracker/TkHitPairs/interface/OrderedHitPair.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "TrackingTools/DetLayers/interface/BarrelDetLayer.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "RecoTracker/TransientTrackingRecHit/interface/TkTransientTrackingRecHitBuilder.h"
#include "TrackingTools/Records/interface/TransientRecHitRecord.h"

using namespace std;
// typedef TransientTrackingRecHit::ConstRecHitPointer TkHitPairsCachedHit;

CosmicHitPairGeneratorFromLayerPair::CosmicHitPairGeneratorFromLayerPair(const LayerWithHits* inner,
                                                                         const LayerWithHits* outer,
                                                                         const TrackerGeometry& geom)
    : trackerGeometry(&geom), theOuterLayer(outer), theInnerLayer(inner) {}
CosmicHitPairGeneratorFromLayerPair::~CosmicHitPairGeneratorFromLayerPair() {}

void CosmicHitPairGeneratorFromLayerPair::hitPairs(const TrackingRegion& region, OrderedHitPairs& result) {
  //  static int NSee = 0; static int Ntry = 0; static int Nacc = 0;

  typedef OrderedHitPair::InnerRecHit InnerHit;
  typedef OrderedHitPair::OuterRecHit OuterHit;

  if (theInnerLayer->recHits().empty())
    return;

  if (theOuterLayer->recHits().empty())
    return;
  //  const DetLayer* innerlay=theOuterLayer->layer();
  // const BarrelDetLayer *pippo=dynamic_cast<const BarrelDetLayer*>(theOuterLayer->layer());

  // ************ Daniele

  const DetLayer* blay1;
  const DetLayer* blay2;
  blay1 = dynamic_cast<const BarrelDetLayer*>(theInnerLayer->layer());
  blay2 = dynamic_cast<const BarrelDetLayer*>(theOuterLayer->layer());

  bool seedfromoverlaps = false;
  bool InTheBarrel = false;
  bool InTheForward = false;
  if (blay1 && blay2) {
    InTheBarrel = true;
  } else
    InTheForward = true;

  if (InTheBarrel) {
    float radius1 = dynamic_cast<const BarrelDetLayer*>(theInnerLayer->layer())->specificSurface().radius();
    float radius2 = dynamic_cast<const BarrelDetLayer*>(theOuterLayer->layer())->specificSurface().radius();
    seedfromoverlaps = (abs(radius1 - radius2) < 0.1) ? true : false;
  }

  vector<OrderedHitPair> allthepairs;

  for (auto ohh = theOuterLayer->recHits().begin(); ohh != theOuterLayer->recHits().end(); ohh++) {
    for (auto ihh = theInnerLayer->recHits().begin(); ihh != theInnerLayer->recHits().end(); ihh++) {
      auto oh = static_cast<BaseTrackerRecHit const* const>(*ohh);
      auto ih = static_cast<BaseTrackerRecHit const* const>(*ihh);

      float z_diff = ih->globalPosition().z() - oh->globalPosition().z();
      float inny = ih->globalPosition().y();
      float outy = oh->globalPosition().y();
      float innx = ih->globalPosition().x();
      float outx = oh->globalPosition().x();
      ;
      float dxdy = abs((outx - innx) / (outy - inny));
      float DeltaR = oh->globalPosition().perp() - ih->globalPosition().perp();
      ;

      if (InTheBarrel &&
          (abs(z_diff) < 30)  // && (outy > 0.) && (inny > 0.)
          //&&((abs(inny-outy))<30)
          && (dxdy < 2) && (inny * outy > 0) && (abs(DeltaR) > 0)) {
        //	cout << " ******** sono dentro inthebarrel *********** " << endl;
        if (seedfromoverlaps) {
          //this part of code works for MTCC
          // for the other geometries must be verified
          //Overlaps in the difference in z is decreased and the difference in phi is
          //less than 0.05
          if ((DeltaR < 0) && (abs(z_diff) < 18) &&
              (abs(ih->globalPosition().phi() - oh->globalPosition().phi()) < 0.05) && (dxdy < 2))
            result.push_back(OrderedHitPair(ih, oh));
        } else
          result.push_back(OrderedHitPair(ih, oh));
      }
      if (InTheForward && (abs(z_diff) > 1.)) {
        //	cout << " ******** sono dentro intheforward *********** " << endl;
        result.push_back(OrderedHitPair(ih, oh));
      }
    }
  }

  /*
  // uncomment if the sort operation below needs to be called
  class CompareHitPairsY {
  public:
    CompareHitPairsY(const TrackerGeometry& t): tracker{&t} {}
    bool operator()(const OrderedHitPair& h1, const OrderedHitPair& h2) {
      const TrackingRecHit* trh1i = h1.inner()->hit();
      const TrackingRecHit* trh2i = h2.inner()->hit();
      const TrackingRecHit* trh1o = h1.outer()->hit();
      const TrackingRecHit* trh2o = h2.outer()->hit();
      GlobalPoint in1p = tracker->idToDet(trh1i->geographicalId())->surface().toGlobal(trh1i->localPosition());
      GlobalPoint in2p = tracker->idToDet(trh2i->geographicalId())->surface().toGlobal(trh2i->localPosition());
      GlobalPoint ou1p = tracker->idToDet(trh1o->geographicalId())->surface().toGlobal(trh1o->localPosition());
      GlobalPoint ou2p = tracker->idToDet(trh2o->geographicalId())->surface().toGlobal(trh2o->localPosition());
      if (ou1p.y() * ou2p.y() < 0)
        return ou1p.y() > ou2p.y();
      else {
        float dist1 = 100 * std::abs(ou1p.z() - in1p.z()) - std::abs(ou1p.y()) - 0.1 * std::abs(in1p.y());
        float dist2 = 100 * std::abs(ou2p.z() - in2p.z()) - std::abs(ou2p.y()) - 0.1 * std::abs(in2p.y());
        return dist1 < dist2;
      }
    }
    
  private:
    const TrackerGeometry* tracker;
  }; 
  stable_sort(allthepairs.begin(),allthepairs.end(),CompareHitPairsY(*trackerGeometry));
  //Seed from overlaps are saved only if
  //no others have been saved

  if (allthepairs.size()>0) {
    if (seedfromoverlaps) {
      if (result.size()==0) result.push_back(allthepairs[0]);
    }
    else result.push_back(allthepairs[0]);
  }
*/
}
