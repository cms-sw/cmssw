#ifndef RecoTracker_TkTrackingRegions_GlobalTrackingRegionProducerFromBeamSpot_H
#define RecoTracker_TkTrackingRegions_GlobalTrackingRegionProducerFromBeamSpot_H

#include "RecoTracker/TkTrackingRegions/interface/TrackingRegionProducer.h"
#include "RecoTracker/TkTrackingRegions/interface/GlobalTrackingRegion.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "FWCore/ParameterSet/interface/InputTag.h"

class GlobalTrackingRegionProducerFromBeamSpot : public TrackingRegionProducer {

public:

  GlobalTrackingRegionProducerFromBeamSpot(const edm::ParameterSet& cfg) {

    edm::ParameterSet regionPSet = cfg.getParameter<edm::ParameterSet>("RegionPSet");
    thePtMin            = regionPSet.getParameter<double>("ptMin");
    theOriginRadius     = regionPSet.getParameter<double>("originRadius");
    theOriginHalfLength = regionPSet.getParameter<double>("originHalfLength");
    theBeamSpotTag      = regionPSet.getParameter<edm::InputTag>("beamSpot");
    thePrecise          = regionPSet.getParameter<bool>("precise");
  }

  virtual ~GlobalTrackingRegionProducerFromBeamSpot(){}

  virtual std::vector<TrackingRegion* > regions(const edm::Event&ev, const edm::EventSetup&) const {
//    std::cout <<"HERE GlobalTrackingRegionProducerFromBeamSpot !!!" << std::endl;
    std::vector<TrackingRegion* > result;
    edm::Handle<reco::BeamSpot> bsHandle;
    ev.getByLabel( theBeamSpotTag, bsHandle);
    if(bsHandle.isValid()) {
      const reco::BeamSpot & bs = *bsHandle; 
      GlobalPoint origin(bs.x0(), bs.y0(), bs.z0()); 
//      std::cout <<"origin: " << origin << std::endl;
      result.push_back( new GlobalTrackingRegion(
          thePtMin, theOrigin,theOriginRadius, theOriginHalfLength, thePrecise));
    }
    return result;
  }

private:
  double thePtMin;
  GlobalPoint theOrigin;
  double theOriginRadius;
  double theOriginHalfLength;
  edm::InputTag theBeamSpotTag;
  bool thePrecise;
};

#endif

