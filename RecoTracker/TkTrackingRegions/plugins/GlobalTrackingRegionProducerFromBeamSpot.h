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
    theNSigmaZ          = regionPSet.getParameter<double>("nSigmaZ");
    theBeamSpotTag      = regionPSet.getParameter<edm::InputTag>("beamSpot");
    thePrecise          = regionPSet.getParameter<bool>("precise");
  }

  virtual ~GlobalTrackingRegionProducerFromBeamSpot(){}

  virtual std::vector<TrackingRegion* > regions(const edm::Event&ev, const edm::EventSetup&) const {
    std::vector<TrackingRegion* > result;
    edm::Handle<reco::BeamSpot> bsHandle;
    ev.getByLabel( theBeamSpotTag, bsHandle);
    if(bsHandle.isValid()) {

      const reco::BeamSpot & bs = *bsHandle; 

      GlobalPoint origin(bs.x0(), bs.y0(), bs.z0()); 

      result.push_back( new GlobalTrackingRegion( 
          thePtMin, origin, theOriginRadius, theNSigmaZ*bs.sigmaZ(), thePrecise));

    }
    return result;
  }

private:
  double thePtMin;
  double theOriginRadius;
  double theNSigmaZ;
  edm::InputTag theBeamSpotTag;
  bool thePrecise;
};

#endif

