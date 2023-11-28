#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "RecoTracker/PixelTrackFitting/interface/PixelFitter.h"
#include "RecoTracker/PixelTrackFitting/interface/PixelNtupletsFitter.h"
#include "RecoTracker/TkMSParametrization/interface/PixelRecoUtilities.h"

class PixelNtupletsFitterProducer : public edm::global::EDProducer<> {
public:
  explicit PixelNtupletsFitterProducer(const edm::ParameterSet& iConfig)
      : useRiemannFit_(iConfig.getParameter<bool>("useRiemannFit")), idealMagneticFieldToken_(esConsumes()) {
    produces<PixelFitter>();
  }
  ~PixelNtupletsFitterProducer() override {}

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;
    desc.add<bool>("useRiemannFit", false)->setComment("true for Riemann, false for BrokenLine");
    descriptions.add("pixelNtupletsFitterDefault", desc);
  }

private:
  bool useRiemannFit_;
  const edm::ESGetToken<MagneticField, IdealMagneticFieldRecord> idealMagneticFieldToken_;
  void produce(edm::StreamID, edm::Event& iEvent, const edm::EventSetup& iSetup) const override;
};

void PixelNtupletsFitterProducer::produce(edm::StreamID, edm::Event& iEvent, const edm::EventSetup& iSetup) const {
  auto const& idealField = iSetup.getData(idealMagneticFieldToken_);
  float bField = 1 / idealField.inverseBzAtOriginInGeV();
  auto impl = std::make_unique<PixelNtupletsFitter>(bField, &idealField, useRiemannFit_);
  auto prod = std::make_unique<PixelFitter>(std::move(impl));
  iEvent.put(std::move(prod));
}

DEFINE_FWK_MODULE(PixelNtupletsFitterProducer);
