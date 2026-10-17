#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/Common/interface/ValueMap.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/VertexGNNReco/interface/VertexGNNSoA.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "TH1F.h"

class GNNTrackInspector : public edm::one::EDAnalyzer<edm::one::SharedResources> {
public:
  explicit GNNTrackInspector(const edm::ParameterSet& iConfig);
  ~GNNTrackInspector() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void beginJob() override;
  void analyze(const edm::Event& iEvent, const edm::EventSetup&) override;

  const edm::InputTag trackSrc_;
  const edm::InputTag pvModule_;
  const uint32_t printFirstN_;
  const bool dropNaNs_;

  edm::EDGetTokenT<reco::TrackCollection> tkTok_;
  edm::EDGetTokenT<edm::ValueMap<float>> slotAssignTok_;
  edm::EDGetTokenT<edm::ValueMap<float>> maxProbTok_;
  edm::EDGetTokenT<edm::ValueMap<float>> piWeight0Tok_;
  edm::EDGetTokenT<edm::ValueMap<float>> piWeight1Tok_;
  edm::EDGetTokenT<edm::ValueMap<float>> piWeight2Tok_;

  TH1F *h_slotAssign_, *h_maxProb_;
  TH1F *h_piWeight0_, *h_piWeight1_, *h_piWeight2_;
};

GNNTrackInspector::GNNTrackInspector(const edm::ParameterSet& iConfig)
    : trackSrc_(iConfig.getParameter<edm::InputTag>("trackSrc")),
      pvModule_(iConfig.getParameter<edm::InputTag>("pvModule")),
      printFirstN_(iConfig.getParameter<uint32_t>("printFirstN")),
      dropNaNs_(iConfig.getParameter<bool>("dropNaNs")) {
  usesResource("TFileService");

  tkTok_ = consumes<reco::TrackCollection>(trackSrc_);

  auto mkTag = [&](const std::string& instance) -> edm::InputTag {
    return edm::InputTag(pvModule_.label(), instance, pvModule_.process());
  };

  slotAssignTok_ = consumes<edm::ValueMap<float>>(mkTag("gnnSlotAssignment"));
  maxProbTok_ = consumes<edm::ValueMap<float>>(mkTag("gnnMaxProb"));
  piWeight0Tok_ = consumes<edm::ValueMap<float>>(mkTag("gnnPiWeight0"));
  piWeight1Tok_ = consumes<edm::ValueMap<float>>(mkTag("gnnPiWeight1"));
  piWeight2Tok_ = consumes<edm::ValueMap<float>>(mkTag("gnnPiWeight2"));
}

void GNNTrackInspector::beginJob() {
  edm::Service<TFileService> fs;
  h_slotAssign_ =
      fs->make<TH1F>("slotAssign", "GNN Slot Assignment;slot;tracks", vertexgnn::kNumSlots, 0, vertexgnn::kNumSlots);
  h_maxProb_ = fs->make<TH1F>("maxProb", "GNN Max Assignment Prob;prob;tracks", 100, 0.0, 1.0);
  h_piWeight0_ = fs->make<TH1F>("piWeight0", "PID Weight #pi;weight;tracks", 100, 0.0, 1.0);
  h_piWeight1_ = fs->make<TH1F>("piWeight1", "PID Weight K;weight;tracks", 100, 0.0, 1.0);
  h_piWeight2_ = fs->make<TH1F>("piWeight2", "PID Weight p;weight;tracks", 100, 0.0, 1.0);
}

void GNNTrackInspector::analyze(const edm::Event& iEvent, const edm::EventSetup&) {
  edm::Handle<reco::TrackCollection> hTracks;
  iEvent.getByToken(tkTok_, hTracks);
  auto const& tracks = *hTracks;
  auto const& slotAssignVM = iEvent.get(slotAssignTok_);
  auto const& maxProbVM = iEvent.get(maxProbTok_);
  auto const& piWeight0VM = iEvent.get(piWeight0Tok_);
  auto const& piWeight1VM = iEvent.get(piWeight1Tok_);
  auto const& piWeight2VM = iEvent.get(piWeight2Tok_);

  const size_t nTk = tracks.size();

  auto get = [&](const edm::ValueMap<float>& vm, size_t i) -> float {
    reco::TrackRef tref(hTracks, i);
    float v = vm[tref];
    return (dropNaNs_ && !std::isfinite(v)) ? 0.f : v;
  };

  for (size_t i = 0; i < std::min<size_t>(nTk, printFirstN_); ++i) {
    float slot = get(slotAssignVM, i);
    float maxP = get(maxProbVM, i);
    float pw0 = get(piWeight0VM, i), pw1 = get(piWeight1VM, i), pw2 = get(piWeight2VM, i);

    auto const& trk = tracks[i];
    edm::LogInfo("GNNTrackInspector") << "track[" << i << "] pt=" << trk.pt() << " eta=" << trk.eta()
                                      << " slot=" << slot << " maxProb=" << maxP << " piWeights=(" << pw0 << "," << pw1
                                      << "," << pw2 << ")";
  }

  auto fillIf = [&](TH1F* h, const edm::ValueMap<float>& vm, size_t i) {
    reco::TrackRef tref(hTracks, i);
    float v = vm[tref];
    if (dropNaNs_ && !std::isfinite(v))
      return;
    h->Fill(v);
  };
  for (size_t i = 0; i < nTk; ++i) {
    fillIf(h_slotAssign_, slotAssignVM, i);
    fillIf(h_maxProb_, maxProbVM, i);
    fillIf(h_piWeight0_, piWeight0VM, i);
    fillIf(h_piWeight1_, piWeight1VM, i);
    fillIf(h_piWeight2_, piWeight2VM, i);
  }
}

void GNNTrackInspector::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("trackSrc", edm::InputTag("generalTracks"))
      ->setComment("The TrackCollection these ValueMaps are keyed against.");
  desc.add<edm::InputTag>("pvModule", edm::InputTag("unsortedOfflinePrimaryVerticesGNN"))
      ->setComment("Module label of the PrimaryVertexProducer that produced gnn* ValueMaps.");
  desc.add<uint32_t>("printFirstN", 10)->setComment("Print first N tracks per event to the Log.");
  desc.add<bool>("dropNaNs", true)->setComment("If true, skip NaNs when filling histos.");
  descriptions.add("GNNTrackInspector", desc);
}
DEFINE_FWK_MODULE(GNNTrackInspector);
