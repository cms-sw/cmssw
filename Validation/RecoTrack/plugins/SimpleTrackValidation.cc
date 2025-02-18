// system include files
#include <memory>

#include "TTree.h"
#include "TFile.h"

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"

#include "SimTracker/TrackerHitAssociation/interface/ClusterTPAssociation.h"
#include "SimDataFormats/PileupSummaryInfo/interface/PileupSummaryInfo.h"
#include "SimDataFormats/Associations/interface/TrackToTrackingParticleAssociator.h"

#include "SimTracker/Common/interface/TrackingParticleSelector.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"

using reco::TrackCollection;

class SimpleTrackValidation : public edm::one::EDAnalyzer<edm::one::SharedResources> {
public:
  explicit SimpleTrackValidation(const edm::ParameterSet&);
  ~SimpleTrackValidation() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void beginJob() override;
  void analyze(const edm::Event&, const edm::EventSetup&) override;
  void endJob() override;

  // Counters
  int global_rt_ = 0;
  int global_at_ = 0;
  int global_st_ = 0;
  int global_dt_ = 0;
  int global_ast_ = 0;

  TrackingParticleSelector tpSelector;
  TTree* output_tree_;
  const std::vector<edm::InputTag> trackLabels_;
  std::vector<edm::EDGetTokenT<edm::View<reco::Track>>> trackTokens_;
  const edm::EDGetTokenT<reco::TrackToTrackingParticleAssociator> trackAssociatorToken_;
  const edm::EDGetTokenT<TrackingParticleCollection> trackingParticleToken_;
};

SimpleTrackValidation::SimpleTrackValidation(const edm::ParameterSet& iConfig)
    : trackLabels_(iConfig.getParameter<std::vector<edm::InputTag>>("trackLabels")),
      trackAssociatorToken_(consumes<reco::TrackToTrackingParticleAssociator>(
          iConfig.getUntrackedParameter<edm::InputTag>("trackAssociator"))),
      trackingParticleToken_(
          consumes<TrackingParticleCollection>(iConfig.getParameter<edm::InputTag>("trackingParticles"))) {
  for (auto& itag : trackLabels_) {
    trackTokens_.push_back(consumes<edm::View<reco::Track>>(itag));
  }
  tpSelector = TrackingParticleSelector(iConfig.getParameter<double>("ptMinTP"),
                                        iConfig.getParameter<double>("ptMaxTP"),
                                        iConfig.getParameter<double>("minRapidityTP"),
                                        iConfig.getParameter<double>("maxRapidityTP"),
                                        iConfig.getParameter<double>("tipTP"),
                                        iConfig.getParameter<double>("lipTP"),
                                        iConfig.getParameter<int>("minHitTP"),
                                        iConfig.getParameter<bool>("signalOnlyTP"),
                                        iConfig.getParameter<bool>("intimeOnlyTP"),
                                        iConfig.getParameter<bool>("chargedOnlyTP"),
                                        iConfig.getParameter<bool>("stableOnlyTP"),
                                        iConfig.getParameter<std::vector<int>>("pdgIdTP"),
                                        iConfig.getParameter<bool>("invertRapidityCutTP"),
                                        iConfig.getParameter<double>("minPhiTP"),
                                        iConfig.getParameter<double>("maxPhiTP"));
}

void SimpleTrackValidation::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  using namespace edm;

  auto const& associatorByHits = iEvent.get(trackAssociatorToken_);
  auto TPCollectionH = iEvent.getHandle(trackingParticleToken_);
  TrackingParticleRefVector tpCollection;

  for (size_t i = 0, size = TPCollectionH->size(); i < size; ++i) {
    auto tp = TrackingParticleRef(TPCollectionH, i);
    if (tpSelector(*tp)) {
      tpCollection.push_back(tp);
    }
  }

  for (const auto& trackToken : trackTokens_) {
    edm::Handle<edm::View<reco::Track>> tracksHandle;
    iEvent.getByToken(trackToken, tracksHandle);
    const edm::View<reco::Track>& tracks = *tracksHandle;

    edm::RefToBaseVector<reco::Track> trackRefs;
    for (edm::View<reco::Track>::size_type i = 0; i < tracks.size(); ++i) {
      trackRefs.push_back(tracks.refAt(i));
    }

    reco::RecoToSimCollection recSimColl = associatorByHits.associateRecoToSim(trackRefs, tpCollection);
    reco::SimToRecoCollection simRecColl = associatorByHits.associateSimToReco(trackRefs, tpCollection);

    int rt = 0;                    // number of reconstructed tracks;
    int at = 0;                    // number of reconstructed tracks associated to a tracking particle
    int ast = 0;                   // number of tracking particles associated to at least a reconstructed track.
    int dt = 0;                    //  number of duplicates (number of sim associated to more than one reco);
    int st = tpCollection.size();  // number of tracking particles;

    for (const auto& track : trackRefs) {
      rt++;
      auto foundTP = recSimColl.find(track);
      if (foundTP != recSimColl.end()) {
        const auto& tp = foundTP->val;
        if (!tp.empty()) {
          at++;
        }
        if (simRecColl.find(tp[0].first) != simRecColl.end()) {
          if (simRecColl[tp[0].first].size() > 1) {
            dt++;
          }
        }
      }
    }
    for (const TrackingParticleRef& tpr : tpCollection) {
      auto foundTrack = simRecColl.find(tpr);
      if (foundTrack != simRecColl.end()) {
        ast++;
      }
    }

    LogPrint("TrackValidator") << "Tag " << trackLabels_[0].label() << " Total simulated " << st
                               << " Associated tracks " << at << " Total reconstructed " << rt;
    global_rt_ += rt;
    global_st_ += st;
    global_at_ += at;
    global_dt_ += dt;
    global_ast_ += ast;
  }
}

void SimpleTrackValidation::beginJob() {
  edm::Service<TFileService> fs;
  output_tree_ = fs->make<TTree>("output", "Simple Track Validation TTree");

  output_tree_->Branch("rt", &global_rt_);
  output_tree_->Branch("at", &global_at_);
  output_tree_->Branch("st", &global_st_);
  output_tree_->Branch("dt", &global_dt_);
  output_tree_->Branch("ast", &global_ast_);
}

void SimpleTrackValidation::endJob() { output_tree_->Fill(); }

void SimpleTrackValidation::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;

  desc.add<std::vector<edm::InputTag>>("trackLabels", {edm::InputTag("generalTracks")});
  desc.add<edm::InputTag>("trackingParticles", edm::InputTag("mix", "MergedTrackTruth"));
  desc.add<edm::InputTag>("trackAssociator", edm::InputTag("trackingParticleRecoTrackAsssociation"));

  // TP Selector parameters
  desc.add<double>("ptMinTP", 0.9);
  desc.add<double>("ptMaxTP", 1e100);
  desc.add<double>("minRapidityTP", -2.4);
  desc.add<double>("maxRapidityTP", 2.4);
  desc.add<double>("tipTP", 3.5);
  desc.add<double>("lipTP", 30.0);
  desc.add<int>("minHitTP", 0);
  desc.add<bool>("signalOnlyTP", true);
  desc.add<bool>("intimeOnlyTP", false);
  desc.add<bool>("chargedOnlyTP", true);
  desc.add<bool>("stableOnlyTP", false);
  desc.add<std::vector<int>>("pdgIdTP", {});
  desc.add<bool>("invertRapidityCutTP", false);
  desc.add<double>("minPhiTP", -3.2);
  desc.add<double>("maxPhiTP", 3.2);

  descriptions.addWithDefaultLabel(desc);
}

DEFINE_FWK_MODULE(SimpleTrackValidation);
