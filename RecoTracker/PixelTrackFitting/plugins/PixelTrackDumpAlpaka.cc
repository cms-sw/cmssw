#include <Eigen/Core>  // needed here by soa layout

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/global/EDAnalyzer.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "DataFormats/VertexSoA/interface/ZVertexHost.h"
#include "DataFormats/TrackSoA/interface/TracksHost.h"

template <typename TrackerTraits>
class PixelTrackDumpAlpakaT : public edm::global::EDAnalyzer<> {
public:
  using TkSoAHost = reco::TracksHost;
  using VertexSoAHost = ZVertexHost;

  explicit PixelTrackDumpAlpakaT(const edm::ParameterSet& iConfig);
  ~PixelTrackDumpAlpakaT() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void analyze(edm::StreamID streamID, edm::Event const& iEvent, const edm::EventSetup& iSetup) const override;
  edm::EDGetTokenT<TkSoAHost> tokenSoATrack_;
  edm::EDGetTokenT<VertexSoAHost> tokenSoAVertex_;
};

template <typename TrackerTraits>
PixelTrackDumpAlpakaT<TrackerTraits>::PixelTrackDumpAlpakaT(const edm::ParameterSet& iConfig) {
  tokenSoATrack_ = consumes(iConfig.getParameter<edm::InputTag>("pixelTrackSrc"));
  tokenSoAVertex_ = consumes(iConfig.getParameter<edm::InputTag>("pixelVertexSrc"));
}

template <typename TrackerTraits>
void PixelTrackDumpAlpakaT<TrackerTraits>::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("pixelTrackSrc", edm::InputTag("pixelTracksAlpaka"));
  desc.add<edm::InputTag>("pixelVertexSrc", edm::InputTag("pixelVerticesAlpaka"));
  descriptions.addWithDefaultLabel(desc);
}

template <typename TrackerTraits>
void PixelTrackDumpAlpakaT<TrackerTraits>::analyze(edm::StreamID streamID,
                                                   edm::Event const& iEvent,
                                                   const edm::EventSetup& iSetup) const {
  auto const& tracks = iEvent.get(tokenSoATrack_);
  assert(tracks.view().quality().data());
  assert(tracks.view().chi2().data());
  assert(tracks.view().nLayers().data());
  assert(tracks.view().eta().data());
  assert(tracks.view().pt().data());
  assert(tracks.view().state().data());
  assert(tracks.view().covariance().data());
  assert(tracks.view().nTracks());

  auto const& vertices = iEvent.get(tokenSoAVertex_);
  assert(vertices.view<reco::ZVertexTracksSoA>().idv().data());
  assert(vertices.view().zv().data());
  assert(vertices.view().wv().data());
  assert(vertices.view().chi2().data());
  assert(vertices.view().ptv2().data());
  assert(vertices.view<reco::ZVertexTracksSoA>().ndof().data());
  assert(vertices.view().sortInd().data());
  assert(vertices.view().nvFinal());
}

using PixelTrackDumpAlpakaPhase1 = PixelTrackDumpAlpakaT<pixelTopology::Phase1>;
using PixelTrackDumpAlpakaPhase2 = PixelTrackDumpAlpakaT<pixelTopology::Phase2>;
using PixelTrackDumpAlpakaHIonPhase1 = PixelTrackDumpAlpakaT<pixelTopology::HIonPhase1>;

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(PixelTrackDumpAlpakaPhase1);
DEFINE_FWK_MODULE(PixelTrackDumpAlpakaPhase2);
DEFINE_FWK_MODULE(PixelTrackDumpAlpakaHIonPhase1);
