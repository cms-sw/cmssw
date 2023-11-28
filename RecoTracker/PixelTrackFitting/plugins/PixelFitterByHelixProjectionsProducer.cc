#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/global/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

#include "RecoTracker/PixelTrackFitting/interface/PixelFitter.h"
#include "RecoTracker/PixelTrackFitting/interface/PixelFitterByHelixProjections.h"

#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "Geometry/Records/interface/TrackerTopologyRcd.h"

class PixelFitterByHelixProjectionsProducer : public edm::global::EDProducer<> {
public:
  explicit PixelFitterByHelixProjectionsProducer(const edm::ParameterSet& iConfig)
      : theTopoToken(esConsumes()),
        theFieldToken(esConsumes()),
        thePutToken(produces<PixelFitter>()),
        thescaleErrorsForBPix1(iConfig.getParameter<bool>("scaleErrorsForBPix1")),
        thescaleFactor(iConfig.getParameter<double>("scaleFactor")) {}
  ~PixelFitterByHelixProjectionsProducer() override {}

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;
    desc.add<bool>("scaleErrorsForBPix1", false);
    desc.add<double>("scaleFactor", 0.65)->setComment("The default value was derived for phase1 pixel");
    descriptions.add("pixelFitterByHelixProjectionsDefault", desc);
  }

private:
  void produce(edm::StreamID, edm::Event& iEvent, const edm::EventSetup& iSetup) const override;

  const edm::ESGetToken<TrackerTopology, TrackerTopologyRcd> theTopoToken;
  const edm::ESGetToken<MagneticField, IdealMagneticFieldRecord> theFieldToken;
  const edm::EDPutTokenT<PixelFitter> thePutToken;
  const bool thescaleErrorsForBPix1;
  const float thescaleFactor;
};

void PixelFitterByHelixProjectionsProducer::produce(edm::StreamID,
                                                    edm::Event& iEvent,
                                                    const edm::EventSetup& iSetup) const {
  iEvent.emplace(
      thePutToken,
      std::make_unique<PixelFitterByHelixProjections>(
          &iSetup.getData(theTopoToken), &iSetup.getData(theFieldToken), thescaleErrorsForBPix1, thescaleFactor));
}

DEFINE_FWK_MODULE(PixelFitterByHelixProjectionsProducer);
