#include "DataFormats/OfflineVertexSoA/interface/alpaka/VertexDeviceCollection.h"
#include "DataFormats/BeamSpot/interface/alpaka/BeamSpotDevice.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/stream/EDProducer.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EDPutToken.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/ESGetToken.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

#include "DataFormats/Math/interface/AlgebraicROOTObjects.h"

#include "TracksForDAInBlocksAlgo.h"
#include "DAInBlocksClusterizerAlgo.h"
#include "WeightedVertexFitterAlgo.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {
  /**
   * This class does vertexing by
   * - consuming set of Track
   * - clusterizing them into track clusters
   * - fitting cluster properties to vertex coordinates
   * - produces a device vertex product (Vertex)
   */
  class PrimaryVertexProducerPortable : public stream::EDProducer<> {
  public:
    PrimaryVertexProducerPortable(edm::ParameterSet const& config) : stream::EDProducer<>(config) {
      trackToken_ = consumes(config.getParameter<edm::InputTag>("TrackLabel"));
      beamSpotToken_ = consumes(config.getParameter<edm::InputTag>("BeamSpotLabel"));
      devicePutToken_ = produces();
      blockSize_ = config.getParameter<int32_t>("blockSize");
      blockOverlap_ = config.getParameter<double>("blockOverlap");
      clusterParams_ = {
          .Tmin = config.getParameter<edm::ParameterSet>("TkClusParameters").getParameter<double>("Tmin"),
          .Tpurge = config.getParameter<edm::ParameterSet>("TkClusParameters").getParameter<double>("Tpurge"),
          .Tstop = config.getParameter<edm::ParameterSet>("TkClusParameters").getParameter<double>("Tstop"),
          .vertexSize = config.getParameter<edm::ParameterSet>("TkClusParameters").getParameter<double>("vertexSize"),
          .coolingFactor =
              config.getParameter<edm::ParameterSet>("TkClusParameters").getParameter<double>("coolingFactor"),
          .d0CutOff = config.getParameter<edm::ParameterSet>("TkClusParameters").getParameter<double>("d0CutOff"),
          .dzCutOff = config.getParameter<edm::ParameterSet>("TkClusParameters").getParameter<double>("dzCutOff"),
          .uniquetrkweight =
              config.getParameter<edm::ParameterSet>("TkClusParameters").getParameter<double>("uniquetrkweight"),
          .uniquetrkminp =
              config.getParameter<edm::ParameterSet>("TkClusParameters").getParameter<double>("uniquetrkminp"),
          .zmerge = config.getParameter<edm::ParameterSet>("TkClusParameters").getParameter<double>("zmerge"),
          .zrange = config.getParameter<edm::ParameterSet>("TkClusParameters").getParameter<double>("zrange"),
          .convergence_mode =
              config.getParameter<edm::ParameterSet>("TkClusParameters").getParameter<int>("convergence_mode"),
          .delta_lowT = config.getParameter<edm::ParameterSet>("TkClusParameters").getParameter<double>("delta_lowT"),
          .delta_highT =
              config.getParameter<edm::ParameterSet>("TkClusParameters").getParameter<double>("delta_highT")};
      clusterParams_.uniquetrkminp = clusterParams_.uniquetrkminp * (1 - blockOverlap_);
      fitterParams_ = {
          .useBeamSpotConstraint =
              config.getParameter<edm::ParameterSet>("TkFitterParameters").getParameter<bool>("useBeamSpotConstraint"),
      };
    }

    void produce(device::Event& iEvent, device::EventSetup const& iSetup) override {
      const TrackDeviceCollection& inputtracks = iEvent.get(trackToken_);
      const BeamSpotDevice& beamSpot = iEvent.get(beamSpotToken_);
      int32_t nT = inputtracks.view().nT();
      int32_t nBlocks = nT > blockSize_ ? int32_t((nT - 1) / (blockOverlap_ * blockSize_))
                                        : 1;  // If all fit within a block, no need to split
      // Now the device collections we still need
      TrackDeviceCollection tracksInBlocks(iEvent.queue(), nBlocks * blockSize_);  // As high as needed
      VertexDeviceCollection deviceVertex(
          iEvent.queue(), 1024);  // Hard capped to 1024, though we might want to restrict it for low PU cases

      // run the algorithm
      //// First create the individual blocks
      TracksForDAInBlocksAlgo blockKernel_{};
      blockKernel_.createBlocks(iEvent.queue(), inputtracks, tracksInBlocks, blockSize_, blockOverlap_);
      // Need to have the blocks created before launching the next step
      //// Then run the clusterizer per blocks
      DAInBlocksClusterizerAlgo clusterizerKernel_{iEvent.queue(), blockSize_};
      clusterizerKernel_.clusterize(iEvent.queue(), tracksInBlocks, deviceVertex, clusterParams_, nBlocks, blockSize_);
      clusterizerKernel_.resplit_tracks(
          iEvent.queue(), tracksInBlocks, deviceVertex, clusterParams_, nBlocks, blockSize_);
      clusterizerKernel_.reject_outliers(
          iEvent.queue(), tracksInBlocks, deviceVertex, clusterParams_, nBlocks, blockSize_);
      // Need to have all vertex before arbitrating and deciding what we keep
      clusterizerKernel_.arbitrate(iEvent.queue(), tracksInBlocks, deviceVertex, clusterParams_, nBlocks, blockSize_);
      //// And then fit
      WeightedVertexFitterAlgo fitterKernel_{iEvent.queue(), fitterParams_};
      fitterKernel_.fit(iEvent.queue(), tracksInBlocks, deviceVertex, beamSpot);
      // Put the vertices in the event as a portable collection
      iEvent.emplace(devicePutToken_, std::move(deviceVertex));
    }

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
      edm::ParameterSetDescription desc;
      desc.add<edm::InputTag>("TrackLabel");
      desc.add<edm::InputTag>("BeamSpotLabel");
      desc.add<double>("blockOverlap");
      desc.add<int32_t>("blockSize");
      edm::ParameterSetDescription parf0;
      parf0.add<bool>("useBeamSpotConstraint", true);
      desc.add<edm::ParameterSetDescription>("TkFitterParameters", parf0);
      edm::ParameterSetDescription parc0;
      parc0.add<double>("d0CutOff", 3.0);
      parc0.add<double>("Tmin", 2.0);
      parc0.add<double>("delta_lowT", 0.001);
      parc0.add<double>("zmerge", 0.01);
      parc0.add<double>("dzCutOff", 3.0);
      parc0.add<double>("Tpurge", 2.0);
      parc0.add<int32_t>("convergence_mode", 0);
      parc0.add<double>("delta_highT", 0.01);
      parc0.add<double>("Tstop", 0.5);
      parc0.add<double>("coolingFactor", 0.6);
      parc0.add<double>("vertexSize", 0.006);
      parc0.add<double>("uniquetrkweight", 0.8);
      parc0.add<double>("uniquetrkminp", 0.0);
      parc0.add<double>("zrange", 4.0);
      desc.add<edm::ParameterSetDescription>("TkClusParameters", parc0);
      descriptions.addWithDefaultLabel(desc);
    }

  private:
    device::EDGetToken<TrackDeviceCollection> trackToken_;
    device::EDGetToken<BeamSpotDevice> beamSpotToken_;
    device::EDPutToken<VertexDeviceCollection> devicePutToken_;
    int32_t blockSize_;
    double blockOverlap_;
    FitterParameters fitterParams_;
    ClusterParameters clusterParams_;
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#include "HeterogeneousCore/AlpakaCore/interface/alpaka/MakerMacros.h"
DEFINE_FWK_ALPAKA_MODULE(PrimaryVertexProducerPortable);
