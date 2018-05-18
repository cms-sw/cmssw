#ifndef RecoTracker_TkTrackingRegions_GlobalTrackingRegionProducerFromBeamSpot_H
#define RecoTracker_TkTrackingRegions_GlobalTrackingRegionProducerFromBeamSpot_H

#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/Event.h"
#include "RecoTracker/TkTrackingRegions/interface/TrackingRegionProducer.h"
#include "RecoTracker/TkTrackingRegions/interface/GlobalTrackingRegion.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
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
    theUseMS           = (regionPSet.existsAs<bool>("useMultipleScattering") ? regionPSet.getParameter<bool>("useMultipleScattering") : false);

  }

  ~GlobalTrackingRegionProducerFromBeamSpot() override{}

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    {
      edm::ParameterSetDescription desc;

      desc.add<bool>("precise", true);
      desc.add<bool>("useMultipleScattering", false);
      desc.add<double>("nSigmaZ", 4.0);
      desc.add<double>("originHalfLength", 0.0); // this is the default in constructor
      desc.add<double>("originRadius", 0.2);
      desc.add<double>("ptMin", 0.9);
      desc.add<edm::InputTag>("beamSpot", edm::InputTag("offlineBeamSpot"));

      // Only for backwards-compatibility
      edm::ParameterSetDescription descRegion;
      descRegion.add<edm::ParameterSetDescription>("RegionPSet", desc);

      descriptions.add("globalTrackingRegionFromBeamSpot", descRegion);
    }

    {
      edm::ParameterSetDescription desc;

      desc.add<bool>("precise", true);
      desc.add<bool>("useMultipleScattering", false);
      desc.add<double>("nSigmaZ", 0.0); // this is the default in constructor
      desc.add<double>("originHalfLength", 21.2);
      desc.add<double>("originRadius", 0.2);
      desc.add<double>("ptMin", 0.9);
      desc.add<edm::InputTag>("beamSpot", edm::InputTag("offlineBeamSpot"));

      // Only for backwards-compatibility
      edm::ParameterSetDescription descRegion;
      descRegion.add<edm::ParameterSetDescription>("RegionPSet", desc);

      descriptions.add("globalTrackingRegionFromBeamSpotFixedZ", descRegion);
    }
  }

  std::vector<std::unique_ptr<TrackingRegion> > regions(const edm::Event&ev, const edm::EventSetup&) const override {
    std::vector<std::unique_ptr<TrackingRegion> > result;
    edm::Handle<reco::BeamSpot> bsHandle;
    ev.getByToken( token_beamSpot, bsHandle);
    if(bsHandle.isValid()) {

      const reco::BeamSpot & bs = *bsHandle; 

      GlobalPoint origin(bs.x0(), bs.y0(), bs.z0()); 

      result.push_back( std::make_unique<GlobalTrackingRegion>(
          thePtMin, origin, theOriginRadius, std::max(theNSigmaZ*bs.sigmaZ(), theOriginHalfLength), thePrecise,theUseMS));

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
  bool theUseMS;
};

#endif

