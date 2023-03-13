#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/global/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

#include "RecoTracker/PixelTrackFitting/interface/PixelFitter.h"
#include "RecoTracker/PixelTrackFitting/interface/PixelFitterByConformalMappingAndLine.h"

#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHitBuilder.h"
#include "TrackingTools/Records/interface/TransientRecHitRecord.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"

class PixelFitterByConformalMappingAndLineProducer : public edm::global::EDProducer<> {
public:
  explicit PixelFitterByConformalMappingAndLineProducer(const edm::ParameterSet& iConfig)
      : theTTRHBuilderToken(esConsumes(edm::ESInputTag("", iConfig.getParameter<std::string>("TTRHBuilder")))),
        theTrackerToken(esConsumes()),
        theFieldToken(esConsumes()),
        thePutToken(produces<PixelFitter>()),
        theFixImpactParameter(0),
        theUseFixImpactParameter(false) {
    if (iConfig.getParameter<bool>("useFixImpactParameter")) {
      theFixImpactParameter = iConfig.getParameter<double>("fixImpactParameter");
      theUseFixImpactParameter = true;
    }
  }
  ~PixelFitterByConformalMappingAndLineProducer() override {}

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;

    desc.add<std::string>("TTRHBuilder", "PixelTTRHBuilderWithoutAngle");
    desc.add<bool>("useFixImpactParameter", false);
    desc.add<double>("fixImpactParameter", 0.0);
    descriptions.add("pixelFitterByConformalMappingAndLine", desc);
  }

private:
  void produce(edm::StreamID, edm::Event& iEvent, const edm::EventSetup& iSetup) const override;

  const edm::ESGetToken<TransientTrackingRecHitBuilder, TransientRecHitRecord> theTTRHBuilderToken;
  const edm::ESGetToken<TrackerGeometry, TrackerDigiGeometryRecord> theTrackerToken;
  const edm::ESGetToken<MagneticField, IdealMagneticFieldRecord> theFieldToken;
  const edm::EDPutTokenT<PixelFitter> thePutToken;
  double theFixImpactParameter;
  bool theUseFixImpactParameter;
};

void PixelFitterByConformalMappingAndLineProducer::produce(edm::StreamID,
                                                           edm::Event& iEvent,
                                                           const edm::EventSetup& iSetup) const {
  iEvent.emplace(thePutToken,
                 std::make_unique<PixelFitterByConformalMappingAndLine>(&iSetup.getData(theTTRHBuilderToken),
                                                                        &iSetup.getData(theTrackerToken),
                                                                        &iSetup.getData(theFieldToken),
                                                                        theFixImpactParameter,
                                                                        theUseFixImpactParameter));
}

DEFINE_FWK_MODULE(PixelFitterByConformalMappingAndLineProducer);
