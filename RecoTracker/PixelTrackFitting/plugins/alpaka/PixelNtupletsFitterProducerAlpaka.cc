#include <alpaka/alpaka.hpp>
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/StreamID.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "RecoTracker/PixelTrackFitting/interface/PixelFitter.h"
#include "RecoTracker/PixelTrackFitting/interface/alpaka/PixelNtupletsFitter.h"
#include "RecoTracker/TkMSParametrization/interface/PixelRecoUtilities.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/Event.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EventSetup.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/global/EDProducer.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {
  class PixelNtupletsFitterProducer : public global::EDProducer<> {
  public:
    explicit PixelNtupletsFitterProducer(const edm::ParameterSet& iConfig)
        : useRiemannFit_(iConfig.getParameter<bool>("useRiemannFit")),
          idealMagneticFieldToken_(esConsumes()),
          fitterToken_(produces()) {}
    ~PixelNtupletsFitterProducer() override {}

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
      edm::ParameterSetDescription desc;
      desc.add<bool>("useRiemannFit", false)->setComment("true for Riemann, false for BrokenLine");
      descriptions.addWithDefaultLabel(desc);
    }

  private:
    bool useRiemannFit_;
    const edm::ESGetToken<MagneticField, IdealMagneticFieldRecord> idealMagneticFieldToken_;
    const device::EDPutToken<PixelFitter> fitterToken_;
    void produce(edm::StreamID, device::Event& iEvent, const device::EventSetup& iSetup) const override;
  };

  void PixelNtupletsFitterProducer::produce(edm::StreamID,
                                            device::Event& iEvent,
                                            const device::EventSetup& iSetup) const {
    auto const& idealField = iSetup.getData(idealMagneticFieldToken_);
    float bField = 1 / idealField.inverseBzAtOriginInGeV();
    auto impl = std::make_unique<PixelNtupletsFitter>(iEvent.queue(), bField, &idealField, useRiemannFit_);
    auto prod = std::make_unique<PixelFitter>(std::move(impl));
    iEvent.put(fitterToken_, std::move(prod));
  }
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/MakerMacros.h"
DEFINE_FWK_ALPAKA_MODULE(PixelNtupletsFitterProducer);