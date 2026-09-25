#include <Eigen/Core>
#include <Eigen/Dense>

#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EDPutToken.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/Event.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EventSetup.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/MakerMacros.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/stream/EDProducer.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/host.h"
#include "PhysicsTools/PyTorchAlpaka/interface/TensorCollection.h"
#include "PhysicsTools/PyTorchAlpaka/interface/alpaka/AlpakaModel.h"
#include "DataFormats/VertexGNNReco/interface/VertexGNNSoA.h"
#include "DataFormats/VertexGNNReco/interface/VertexGNNHostCollection.h"
#include "DataFormats/VertexGNNReco/interface/alpaka/VertexGNNDeviceCollection.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::vertexgnn {

  class GNNVertexProducerAlpaka : public stream::EDProducer<> {
  public:
    explicit GNNVertexProducerAlpaka(const edm::ParameterSet& params)
        : EDProducer<>(params),

          trackFeaturesToken_(
              consumes<::vertexgnn::TrackFeaturesHostCollection>(params.getParameter<edm::InputTag>("trackFeatures"))),
          gnnOutputToken_{produces()},
          model_(params.getParameter<edm::FileInPath>("model").fullPath()),
          verbose_(params.getUntrackedParameter<bool>("verbose", false)) {
      checkModel();
    }

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
      edm::ParameterSetDescription desc;
      desc.add<edm::FileInPath>("model", edm::FileInPath("RecoVertex/PrimaryVertexProducer/data/vertexSlotGNN.pt"));
      desc.add<edm::InputTag>("trackFeatures", edm::InputTag("trackFeatureProducer"));
      desc.addUntracked<bool>("verbose", false);
      descriptions.addWithDefaultLabel(desc);
    }

    void produce(device::Event& event, const device::EventSetup& eventSetup) override {
      const auto& hostInput = event.get(trackFeaturesToken_);
      const auto N = hostInput.const_view().metadata().size();

      if (verbose_) {
        edm::LogInfo("GNNVertexProducerAlpaka")
            << "N=" << N << " tracks (from HostCollection), K=" << ::vertexgnn::kNumSlots << " slots";
      }

      TrackFeaturesDeviceCollection deviceInput(event.queue(), N);
      alpaka::memcpy(event.queue(), deviceInput.buffer(), hostInput.buffer());

      GNNOutputDeviceCollection deviceOutput(event.queue(), N);

      auto inputRecords = deviceInput.const_view().records();
      auto outputRecords = deviceOutput.view().records();

      cms::torch::alpakatools::TensorCollection<Queue> inputs(N);
      inputs.add<::vertexgnn::TrackFeaturesSoA>("features",
                                                inputRecords.vz(),
                                                inputRecords.dz(),
                                                inputRecords.pt(),
                                                inputRecords.eta(),
                                                inputRecords.mva(),
                                                inputRecords.pl(),
                                                inputRecords.t_pi(),
                                                inputRecords.t_k(),
                                                inputRecords.t_p(),
                                                inputRecords.s_pi(),
                                                inputRecords.s_k(),
                                                inputRecords.s_p(),
                                                inputRecords.has_time());

      cms::torch::alpakatools::TensorCollection<Queue> outputs(N);
      outputs.add<::vertexgnn::GNNOutputSoA>("A", outputRecords.A());
      outputs.add<::vertexgnn::GNNOutputSoA>("z_hat", outputRecords.z_hat());
      outputs.add<::vertexgnn::GNNOutputSoA>("t_hat", outputRecords.t_hat());
      outputs.add<::vertexgnn::GNNOutputSoA>("p", outputRecords.p());
      outputs.add<::vertexgnn::GNNOutputSoA>("pi", outputRecords.pi());

      if (verbose_) {
        edm::LogInfo("GNNVertexProducerAlpaka") << "Running model_.forward(queue, inputs, outputs) on device...";
      }

      model_.forward(event.queue(), inputs, outputs);

      if (verbose_) {
        edm::LogInfo("GNNVertexProducerAlpaka") << "Inference complete";
      }

      event.emplace(gnnOutputToken_, std::move(deviceOutput));
    }

  private:
    void checkModel() {
      constexpr int nTracks = 64;
      std::vector<::torch::IValue> inputs{::torch::rand({nTracks, ::vertexgnn::kNumFeatures})};
      ::torch::IValue result;
      try {
        result = model_.forward(inputs);
      } catch (const std::exception& e) {
        throw cms::Exception("Configuration")
            << "GNNVertexProducerAlpaka: the model does not accept " << ::vertexgnn::kNumFeatures << " track features\n"
            << e.what();
      }
      if (!result.isTuple() || result.toTuple()->elements().size() != 5) {
        throw cms::Exception("Configuration")
            << "GNNVertexProducerAlpaka: the model must return the 5 tensors A, z_hat, t_hat, p, pi";
      }
      const auto A = result.toTuple()->elements()[0].toTensor();
      const auto slots = (A.dim() == 2) ? A.size(1) : -1;
      if (slots != ::vertexgnn::kNumSlots) {
        throw cms::Exception("Configuration")
            << "GNNVertexProducerAlpaka: the model has " << slots
            << " vertex slots, the build expects kNumSlots = " << ::vertexgnn::kNumSlots;
      }
    }

    const edm::EDGetTokenT<::vertexgnn::TrackFeaturesHostCollection> trackFeaturesToken_;
    const device::EDPutToken<GNNOutputDeviceCollection> gnnOutputToken_;
    torch::AlpakaModel model_;
    const bool verbose_;
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::vertexgnn

DEFINE_FWK_ALPAKA_MODULE(vertexgnn::GNNVertexProducerAlpaka);
