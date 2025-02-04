#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "DataFormats/BeamSpot/interface/BeamSpotHost.h"
#include "DataFormats/BeamSpot/interface/BeamSpotPOD.h"
#include "DataFormats/BeamSpot/interface/alpaka/BeamSpotDevice.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EDPutToken.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/global/EDProducer.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  class BeamSpotDeviceProducer : public global::EDProducer<> {
  public:
    BeamSpotDeviceProducer(edm::ParameterSet const& config)
        : EDProducer(config),
          legacyToken_{consumes(config.getParameter<edm::InputTag>("src"))},
          deviceToken_{produces()} {}

    void produce(edm::StreamID, device::Event& event, device::EventSetup const& setup) const override {
      reco::BeamSpot const& beamspot = event.get(legacyToken_);

      BeamSpotHost hostProduct{event.queue()};
      hostProduct->x = beamspot.x0();
      hostProduct->y = beamspot.y0();
      hostProduct->z = beamspot.z0();
      hostProduct->sigmaZ = beamspot.sigmaZ();
      hostProduct->beamWidthX = beamspot.BeamWidthX();
      hostProduct->beamWidthY = beamspot.BeamWidthY();
      hostProduct->dxdz = beamspot.dxdz();
      hostProduct->dydz = beamspot.dydz();
      hostProduct->emittanceX = beamspot.emittanceX();
      hostProduct->emittanceY = beamspot.emittanceY();
      hostProduct->betaStar = beamspot.betaStar();

      if constexpr (std::is_same_v<Device, alpaka::DevCpu>) {
        event.emplace(deviceToken_, std::move(hostProduct));
      } else {
        BeamSpotDevice deviceProduct{event.queue()};
        alpaka::memcpy(event.queue(), deviceProduct.buffer(), hostProduct.const_buffer());
        event.emplace(deviceToken_, std::move(deviceProduct));
      }
    }

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
      edm::ParameterSetDescription desc;
      desc.add("src", edm::InputTag{});
      descriptions.addWithDefaultLabel(desc);
    }

  private:
    const edm::EDGetTokenT<reco::BeamSpot> legacyToken_;
    const device::EDPutToken<BeamSpotDevice> deviceToken_;
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#include "HeterogeneousCore/AlpakaCore/interface/alpaka/MakerMacros.h"
DEFINE_FWK_ALPAKA_MODULE(BeamSpotDeviceProducer);
