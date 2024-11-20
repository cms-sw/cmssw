#include "DataFormats/PortableVertex/interface/alpaka/VertexDeviceCollection.h"
#include "DataFormats/PortableVertex/interface/VertexHostCollection.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/global/EDProducer.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EDPutToken.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/ESGetToken.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "DataFormats/Math/interface/AlgebraicROOTObjects.h"

#define DEBUG_RECOVERTEX_PRIMARYVERTEXPRODUCER_ALPAKA_PORTABLEBEAMSPOTSOAPRODUCER 0

namespace ALPAKA_ACCELERATOR_NAMESPACE {
  /**
   * This class does
   * - consume set of reco::BeamSpot
   * - converting them to a Alpaka-friendly dataformat
   * - put the Alpaka dataformat in the device for later consumption
   */

  class PortableBeamSpotSoAProducer : public global::EDProducer<> {
  public:
    PortableBeamSpotSoAProducer(edm::ParameterSet const& config) {
      theConfig = config;
      beamSpotToken_ = consumes<reco::BeamSpot>(config.getParameter<edm::InputTag>("BeamSpotLabel"));
      devicePutToken_ = produces();
    }

    void produce(edm::StreamID sid, device::Event& iEvent, device::EventSetup const& iSetup) const override {
      // Get input collections from event
      auto beamSpot = iEvent.getHandle(beamSpotToken_).product();

      // Host collections
      portablevertex::BeamSpotHostCollection hostBeamSpot{1, iEvent.queue()};
      auto& bview = hostBeamSpot.view();
      convertBeamSpot(bview[0], *beamSpot);

      // Create device collections and copy into device
      portablevertex::BeamSpotDeviceCollection deviceBeamSpot{1, iEvent.queue()};

      alpaka::memcpy(iEvent.queue(), deviceBeamSpot.buffer(), hostBeamSpot.buffer());

      // And put into the event
      iEvent.emplace(devicePutToken_, std::move(deviceBeamSpot));
    }

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
      edm::ParameterSetDescription desc;
      desc.add<edm::InputTag>("BeamSpotLabel");
      descriptions.addWithDefaultLabel(desc);
    }

  private:
    edm::EDGetTokenT<reco::BeamSpot> beamSpotToken_;
    device::EDPutToken<portablevertex::BeamSpotDeviceCollection> devicePutToken_;
    edm::ParameterSet theConfig;
    static void convertBeamSpot(portablevertex::BeamSpotHostCollection::View::element out, const reco::BeamSpot in);
  };  //PortableBeamSpotSoAProducer declaration

  void PortableBeamSpotSoAProducer::convertBeamSpot(portablevertex::BeamSpotHostCollection::View::element out,
                                                    const reco::BeamSpot in) {
    out.x() = in.position().x();
    out.y() = in.position().y();
#ifdef DEBUG_RECOVERTEX_PRIMARYVERTEXPRODUCER_ALPAKA_PORTABLEBEAMSPOTSOAPRODUCER
    printf(
        "[PortableBeamSpotSoAProducer::convertBeamSpot()], x:%1.5f, y:%1.5f\n", in.position().x(), in.position().y());
#endif
    out.sx() = in.rotatedCovariance3D()(0, 0);
    out.sy() = in.rotatedCovariance3D()(1, 1);
  }
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#include "HeterogeneousCore/AlpakaCore/interface/alpaka/MakerMacros.h"
DEFINE_FWK_ALPAKA_MODULE(PortableBeamSpotSoAProducer);
