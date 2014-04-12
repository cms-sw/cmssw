#ifndef RecoTracker_TkTrackingRegions_GlobalTrackingRegionProducerFromBeamSpot_H
#define RecoTracker_TkTrackingRegions_GlobalTrackingRegionProducerFromBeamSpot_H

#include "RecoTracker/TkTrackingRegions/interface/TrackingRegionProducer.h"
#include "RecoTracker/TkTrackingRegions/interface/GlobalTrackingRegion.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"

class GlobalTrackingRegionProducerFromBeamSpot : public TrackingRegionProducer {

public:

  GlobalTrackingRegionProducerFromBeamSpot(const edm::ParameterSet& cfg,
	   edm::ConsumesCollector && iC):
    GlobalTrackingRegionProducerFromBeamSpot(cfg, iC)
  {}
  GlobalTrackingRegionProducerFromBeamSpot(const edm::ParameterSet& cfg,
	   edm::ConsumesCollector & iC) {

    edm::ParameterSet regionPSet = cfg.getParameter<edm::ParameterSet>("RegionPSet");
    thePtMin            = regionPSet.getParameter<double>("ptMin");
    theOriginRadius     = regionPSet.getParameter<double>("originRadius");
    if (!regionPSet.existsAs<double>("nSigmaZ") && !regionPSet.existsAs<double>("originHalfLength")) {
        throw cms::Exception("Configuration") << "GlobalTrackingRegionProducerFromBeamSpot: at least one of nSigmaZ, originHalfLength must be present in the cfg.\n";
    }
    theNSigmaZ          = (regionPSet.existsAs<double>("nSigmaZ")          ? regionPSet.getParameter<double>("nSigmaZ")          : 0.0);
    theOriginHalfLength = (regionPSet.existsAs<double>("originHalfLength") ? regionPSet.getParameter<double>("originHalfLength") : 0.0);
    token_beamSpot      = iC.consumes<reco::BeamSpot>(regionPSet.getParameter<edm::InputTag>("beamSpot"));
    thePrecise          = regionPSet.getParameter<bool>("precise");
  }

  virtual ~GlobalTrackingRegionProducerFromBeamSpot(){}

  virtual std::vector<TrackingRegion* > regions(const edm::Event&ev, const edm::EventSetup&) const {
    std::vector<TrackingRegion* > result;
    edm::Handle<reco::BeamSpot> bsHandle;
    ev.getByToken( token_beamSpot, bsHandle);
    if(bsHandle.isValid()) {

      const reco::BeamSpot & bs = *bsHandle; 

      GlobalPoint origin(bs.x0(), bs.y0(), bs.z0()); 

      result.push_back( new GlobalTrackingRegion( 
          thePtMin, origin, theOriginRadius, std::max(theNSigmaZ*bs.sigmaZ(), theOriginHalfLength), thePrecise));

    }
    return result;
  }

private:
  double thePtMin;
  double theOriginRadius;
  double theOriginHalfLength; 
  double theNSigmaZ;
  edm::EDGetTokenT<reco::BeamSpot> 	 token_beamSpot; 
  bool thePrecise;
};

#endif

