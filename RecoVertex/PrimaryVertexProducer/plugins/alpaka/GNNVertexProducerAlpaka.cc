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
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/stream/FixedQueueEDProducer.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/host.h"
#include "PhysicsTools/PyTorchAlpaka/interface/TensorCollection.h"
#include "PhysicsTools/PyTorchAlpaka/interface/alpaka/AlpakaModel.h"
#include "DataFormats/VertexSoA/interface/VertexGNNSoA.h"
#include "DataFormats/VertexSoA/interface/VertexGNNHostCollection.h"
#include "DataFormats/VertexSoA/interface/alpaka/VertexGNNDeviceCollection.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::vertexgnn {

  class GNNVertexProducerAlpaka : public stream::FixedQueueEDProducer<> {
  public:
    explicit GNNVertexProducerAlpaka(const edm::ParameterSet& params)
        : FixedQueueEDProducer<>(params),
          trackFeaturesToken_(
              consumes<::vertexgnn::TrackFeaturesHostCollection>(params.getParameter<edm::InputTag>("trackFeatures"))),
          gnnOutputToken_{produces()},
          gnnSlotToken_{produces()},
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
      GNNSlotDeviceCollection deviceSlots(event.queue(), ::vertexgnn::kNumSlots);

      auto inputRecords = deviceInput.const_view().records();
      auto outputRecords = deviceOutput.view().records();
      auto slotRecords = deviceSlots.view().records();

      cms::torch::alpakatools::TensorCollection<Queue> inputs;
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

      cms::torch::alpakatools::TensorCollection<Queue> outputs;
      outputs.add<::vertexgnn::GNNOutputSoA>("A", outputRecords.A());
      outputs.add<::vertexgnn::GNNSlotSoA>("z_hat", slotRecords.z_hat());
      outputs.add<::vertexgnn::GNNSlotSoA>("t_hat", slotRecords.t_hat());
      outputs.add<::vertexgnn::GNNSlotSoA>("p", slotRecords.p());
      outputs.add<::vertexgnn::GNNOutputSoA>("pi", outputRecords.pi());

      if (verbose_) {
        edm::LogInfo("GNNVertexProducerAlpaka") << "Running model_.forward(queue, inputs, outputs) on device...";
      }

      model_.forward(event.queue(), inputs, outputs);

      if (verbose_) {
        edm::LogInfo("GNNVertexProducerAlpaka") << "Inference complete";
      }

      event.emplace(gnnOutputToken_, std::move(deviceOutput));
      event.emplace(gnnSlotToken_, std::move(deviceSlots));
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
      for (int i = 1; i <= 3; ++i) {
        const auto t = result.toTuple()->elements()[i].toTensor();
        if (t.dim() != 2 || t.size(0) != ::vertexgnn::kNumSlots || t.size(1) != 1) {
          throw cms::Exception("Configuration")
              << "GNNVertexProducerAlpaka: output " << i << " of the model must have the shape ["
              << ::vertexgnn::kNumSlots << ", 1], one value per vertex slot";
        }
      }
    }

    const edm::EDGetTokenT<::vertexgnn::TrackFeaturesHostCollection> trackFeaturesToken_;
    const device::EDPutToken<GNNOutputDeviceCollection> gnnOutputToken_;
    const device::EDPutToken<GNNSlotDeviceCollection> gnnSlotToken_;
    torch::AlpakaModel model_;
    const bool verbose_;
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::vertexgnn

DEFINE_FWK_ALPAKA_MODULE(vertexgnn::GNNVertexProducerAlpaka);
