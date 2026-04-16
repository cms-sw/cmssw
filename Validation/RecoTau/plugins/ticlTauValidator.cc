// -*- C++ -*-
//
// Package:    Validation/TICLTauValidator
// Class:      TICLTauValidator 
//
/*

 Description: Validation for tau reconstruction using TICL and tracks

 Implementation:
   Calo Steps (0..5):
     0: CaloParticle - SimTrackster (seed match)
     1: SimTrackster - RecoTrackster (association map)
     2: RecoTrackster - TICLCandidate
     3: RecoTrackster - PF candidate (merged PF)
     4: PF cand - PFJet
     5: PFJet - PFTau usage (skipped if no taus)

    Tracking steps (0..4):
     0: CP - TrackingParticle
     1: TrackingParticle - reco Track (via tracking associator)
     2: reco Track - PF candidate
     3: PF cand - PFJet

   Confusion matrices:
     - dm_reco_vs_gen_jet : DM inferred from leg counts (in jets)
     - dm_reco_vs_gen_tau : DM inferred from leg counts (in taus)
     - dm_reco_vs_gen_hps : DM reconstructed by HPS (PFTau::decayMode())
*/
//
// Original Author:  Andreas Gruber
//         Created:  Tue, 16 Sep 2025 08:48:53 GMT
//
//

#include <string>
#include <vector>
#include <algorithm>
#include <cmath>
#include <sstream>
#include <unordered_map>
#include <unordered_set>
#include <set>
#include <iterator>
#include <array>

#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"

#include "DQMServices/Core/interface/DQMEDAnalyzer.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"

#include "DataFormats/TauReco/interface/PFTau.h"
#include "DataFormats/TauReco/interface/PFTauFwd.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "DataFormats/Math/interface/deltaR.h"
#include "DataFormats/HGCalReco/interface/Trackster.h"
#include "DataFormats/HGCalReco/interface/TICLCandidate.h"
#include "DataFormats/Common/interface/Ptr.h"
#include "DataFormats/JetReco/interface/PFJet.h"
#include "DataFormats/JetReco/interface/PFJetCollection.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "DataFormats/HLTReco/interface/TriggerObject.h"
#include "DataFormats/HLTReco/interface/TriggerFilterObjectWithRefs.h"

#include "SimDataFormats/CaloAnalysis/interface/SimTauCPLink.h"
#include "SimDataFormats/CaloAnalysis/interface/CaloParticle.h"
#include "SimDataFormats/Associations/interface/TICLAssociationMap.h"
#include "SimDataFormats/CaloAnalysis/interface/SimCluster.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticle.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticleFwd.h"
#include "SimDataFormats/Associations/interface/TrackToTrackingParticleAssociator.h"

class TICLTauValidator : public DQMEDAnalyzer {
public:
  explicit TICLTauValidator(const edm::ParameterSet&);
  ~TICLTauValidator() override {}
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  using TracksterToTracksterMap = ticl::AssociationMap<ticl::mapWithSharedEnergyAndScore,
                                                       std::vector<ticl::Trackster>,
                                                       std::vector<ticl::Trackster>>;

  void bookHistograms(DQMStore::IBooker&, edm::Run const&, edm::EventSetup const&) override;
  void analyze(const edm::Event&, const edm::EventSetup&) override;
  
  struct AssocCounts {
    int nSigCh = 0;            // Signal charged (pion) + neutral hadron (K_L) PF cands
    int nSigPho = 0;           // Signal photon + electron PF cands
    int nAssocCalo = 0;         // Matched via TICL chain (endcap PF candidates)
    int nAssocTrack = 0;        // Matched via track -> TP -> CP (any PF with trackRef)
    int nAssocAllParticles = 0; // Either path (calo OR track)
  };

  AssocCounts countAssociatedSignalPFCands(const reco::PFTau& tau,
                                           size_t barrelSize,
                                           const std::unordered_set<unsigned int>& tauCPKeys,
                                           const std::vector<TICLCandidate>& ticlCandidates,
                                           const TracksterToTracksterMap& recoToSimMap,
                                           const std::vector<ticl::Trackster>& simTracksters,
                                           const std::vector<CaloParticle>* caloParticles,
                                           const reco::RecoToSimCollection* trackRecoToSim = nullptr) const;

  std::string folder_;
  double maxAssocScore_;  // smaller = better association
  double hgcalEtaAbsMin_;

  edm::EDGetTokenT<std::vector<SimTauCPLink>> simTauToken_;
  edm::EDGetTokenT<std::vector<CaloParticle>> caloParticlesToken_;
  edm::EDGetTokenT<reco::PFTauCollection>     tauProducerToken_;

  edm::EDGetTokenT<reco::PFCandidateCollection> pfToken_;
  edm::EDGetTokenT<reco::PFCandidateCollection> pfTmpBarrelToken_;

  edm::EDGetTokenT<std::vector<TICLCandidate>>       ticlCandidatesToken_;
  edm::EDGetTokenT<std::vector<ticl::Trackster>>     simTrackstersToken_;
  edm::EDGetTokenT<TracksterToTracksterMap>          allTrkToSimTrkAssocByLCsToken_;
  edm::EDGetTokenT<reco::PFJetCollection>            pfJetsToken_;
  edm::EDGetTokenT<reco::GenParticleCollection>      genParticlesToken_;
  edm::EDGetTokenT<reco::GenParticleCollection> genVisTausToken_;
  edm::EDGetTokenT<TrackingParticleCollection>   trackingParticleToken_;
  edm::EDGetTokenT<TracksterToTracksterMap>      recoToSimAssocByLCsToken_;
  edm::EDGetTokenT<reco::RecoToSimCollection>    trackRecoToSimToken_;
  std::vector<std::string> hltTauFilterLabels_;
  std::string hltProcessName_;
  std::vector<edm::EDGetTokenT<trigger::TriggerFilterObjectWithRefs>> hltFilterTokens_;

  // ---------- constants & helpers ----------
  static constexpr int   kMaxCHLegs    = 3; // charged hadrons per tau
  static constexpr int   kMaxGammaLegs = 4; // photons (1 pair = 1 pi0)
  static constexpr int   kMaxPi0Legs   = kMaxGammaLegs / 2;
  // Calo chain steps: (0..5) 
  static constexpr int   kNSteps       = 6; 
  // Track chain: (0..4)
  static constexpr int   kNTrackSteps  = 5;
  // Combined (AND) chain: calo AND track must both reach an equivalent endpoint
  static constexpr int   kNCombiSteps  = 3;

  static constexpr int kNDMSel = 6;
  static constexpr int kDMSel[kNDMSel] = {0, 1, 2, 5, 10, 11};

  static inline int dmToSelIndex(int dm) {
    for (int i = 0; i < kNDMSel; ++i)
      if (kDMSel[i] == dm) return i;
    return -1;
  }

  // Gen-DM set
  static constexpr int kNDMGen = 5;
  static constexpr int kDMGen[kNDMGen] = {0, 1, 2, 10, 11};

  static inline int dmToGenIndex(int dm) {
    for (int i = 0; i < kNDMGen; ++i)
      if (kDMGen[i] == dm) return i;
    return -1;
  }

  // Expectations per DM
  static inline int expectedChForDM(int dm) {
    switch (dm) {
      case 0:
      case 1:
      case 2:  return 1;
      case 10:
      case 11: return 3;
      case 5:  return 2; // "2-prong" category (unphysical, used when one hadron is missing)
      default: return 0;
    }
  }
  static inline int expectedPi0ForDM(int dm) {
    switch (dm) {
      case 0:  return 0;
      case 1:  return 1;
      case 2:  return 2;
      case 10: return 0;
      case 11: return 1;
      case 5:  return 0;
      default: return 0;
    }
  }

  static inline int chCapForDM(int dm)  { return std::min(expectedChForDM(dm),  kMaxCHLegs); }
  static inline int pi0CapForDM(int dm) { return std::min(expectedPi0ForDM(dm), kMaxGammaLegs); }

  // Any hadronic tau decay mode, excluding leptonic modes
  static inline bool isHadronicDM(int dm) { return dm >= 0 && dm < 16; }

  template<int N>
  struct StepHistsT {
    MonitorElement* den_pt[N]  {nullptr};
    MonitorElement* den_eta[N] {nullptr};
    MonitorElement* num_pt[N]  {nullptr};
    MonitorElement* num_eta[N] {nullptr};
  };

  // per-DM, per-leg step histos
  using CaloStepHists  = StepHistsT<kNSteps>;
  using TrackStepHists = StepHistsT<kNTrackSteps>;
  using CombiStepHists = StepHistsT<kNCombiSteps>;

  struct FakeRateHists {
    MonitorElement* den_pt  = nullptr;
    MonitorElement* den_eta = nullptr;
    MonitorElement* num_pt  = nullptr;
    MonitorElement* num_eta = nullptr;
    MonitorElement* den_dm_pt[kNDMSel]   = {};
    MonitorElement* den_dm_eta[kNDMSel]  = {};
    MonitorElement* num_dm_pt[kNDMSel]   = {};
    MonitorElement* num_dm_eta[kNDMSel]  = {};
    MonitorElement* den_sig_nCh_pt[kMaxCHLegs]    = {};
    MonitorElement* den_sig_nCh_eta[kMaxCHLegs]   = {};
    MonitorElement* num_sig_nCh_pt[kMaxCHLegs]    = {};
    MonitorElement* num_sig_nCh_eta[kMaxCHLegs]   = {};
    MonitorElement* den_sig_nPi0_pt[kMaxPi0Legs]  = {};
    MonitorElement* den_sig_nPi0_eta[kMaxPi0Legs] = {};
    MonitorElement* num_sig_nPi0_pt[kMaxPi0Legs]  = {};
    MonitorElement* num_sig_nPi0_eta[kMaxPi0Legs] = {};
    MonitorElement* den_dm_sig_nCh_pt[kNDMSel][kMaxCHLegs]    = {{}};
    MonitorElement* den_dm_sig_nCh_eta[kNDMSel][kMaxCHLegs]   = {{}};
    MonitorElement* num_dm_sig_nCh_pt[kNDMSel][kMaxCHLegs]    = {{}};
    MonitorElement* num_dm_sig_nCh_eta[kNDMSel][kMaxCHLegs]   = {{}};
    MonitorElement* den_dm_sig_nPi0_pt[kNDMSel][kMaxPi0Legs]  = {{}};
    MonitorElement* den_dm_sig_nPi0_eta[kNDMSel][kMaxPi0Legs] = {{}};
    MonitorElement* num_dm_sig_nPi0_pt[kNDMSel][kMaxPi0Legs]  = {{}};
    MonitorElement* num_dm_sig_nPi0_eta[kNDMSel][kMaxPi0Legs] = {{}};
  };

  void fillFakeRateHists(const reco::PFTau& tau,
                         int dmSelI,
                         int nChFill,
                         int nPi0Fill,
                         bool isGenuine,
                         FakeRateHists& fr);

  std::array<std::array<CaloStepHists,  kMaxCHLegs>,    kNDMGen> chStepHists_{};
  std::array<std::array<CaloStepHists,  kMaxGammaLegs>, kNDMGen> gammaStepHists_{};
  std::array<std::array<TrackStepHists, kMaxCHLegs>,    kNDMGen> chTrackStepHists_{};
  std::array<std::array<CombiStepHists, kMaxCHLegs>,    kNDMGen> chCombiStepHists_{};

  MonitorElement* dm_reco_vs_gen_jet_  = nullptr;
  MonitorElement* dm_reco_vs_gen_tau_  = nullptr;
  MonitorElement* dm_reco_vs_gen_hps_  = nullptr;

  MonitorElement* cp_chHad_pt_all_  = nullptr;
  MonitorElement* cp_chHad_eta_all_ = nullptr;
  MonitorElement* cp_gamma_pt_all_  = nullptr;
  MonitorElement* cp_gamma_eta_all_ = nullptr;

  // per-DM CP context histos
  MonitorElement* cp_chHad_pt_dm_[kNDMGen]  = {};
  MonitorElement* cp_chHad_eta_dm_[kNDMGen] = {};
  MonitorElement* cp_gamma_pt_dm_[kNDMGen]  = {};
  MonitorElement* cp_gamma_eta_dm_[kNDMGen] = {};

  // denominators (gen-level, use kNDMGen)
  MonitorElement* tau_gen_pt_[kNDMGen]  = {};
  MonitorElement* tau_gen_eta_[kNDMGen] = {};

  // numerators (gen-level, use kNDMGen)
  MonitorElement* tau_gen_matched_to_nCh_pt_[kNDMGen][kMaxCHLegs]     = {{}};
  MonitorElement* tau_gen_matched_to_nCh_eta_[kNDMGen][kMaxCHLegs]    = {{}};
  MonitorElement* tau_gen_matched_to_nPi0_pt_[kNDMGen][kMaxGammaLegs] = {{}};
  MonitorElement* tau_gen_matched_to_nPi0_eta_[kNDMGen][kMaxGammaLegs]= {{}};
  MonitorElement* tau_gen_matched_to_all_pt_[kNDMGen]  = {};
  MonitorElement* tau_gen_matched_to_all_eta_[kNDMGen] = {};

  // numerators split into signal and isolation (tau endpoint only, gen-level use kNDMGen)
  MonitorElement* tau_gen_matched_to_nCh_sig_pt_[kNDMGen][kMaxCHLegs]      = {{}};
  MonitorElement* tau_gen_matched_to_nCh_sig_eta_[kNDMGen][kMaxCHLegs]     = {{}};
  MonitorElement* tau_gen_matched_to_nCh_iso_pt_[kNDMGen][kMaxCHLegs]         = {{}};
  MonitorElement* tau_gen_matched_to_nCh_iso_eta_[kNDMGen][kMaxCHLegs]        = {{}};
  MonitorElement* tau_gen_matched_to_nPi0_sig_pt_[kNDMGen][kMaxGammaLegs]  = {{}};
  MonitorElement* tau_gen_matched_to_nPi0_sig_eta_[kNDMGen][kMaxGammaLegs] = {{}};
  MonitorElement* tau_gen_matched_to_nPi0_iso_pt_[kNDMGen][kMaxGammaLegs]     = {{}};
  MonitorElement* tau_gen_matched_to_nPi0_iso_eta_[kNDMGen][kMaxGammaLegs]    = {{}};

  // Tau-level two-fold: >= N charged CPs matched by track / calo / both / either
  MonitorElement* tau_nCh_track_pt_[kNDMGen][kMaxCHLegs]  = {{}};
  MonitorElement* tau_nCh_track_eta_[kNDMGen][kMaxCHLegs] = {{}};
  MonitorElement* tau_nCh_calo_pt_[kNDMGen][kMaxCHLegs]   = {{}};
  MonitorElement* tau_nCh_calo_eta_[kNDMGen][kMaxCHLegs]  = {{}};
  MonitorElement* tau_nCh_both_pt_[kNDMGen][kMaxCHLegs]   = {{}};
  MonitorElement* tau_nCh_both_eta_[kNDMGen][kMaxCHLegs]  = {{}};
  MonitorElement* tau_nCh_either_pt_[kNDMGen][kMaxCHLegs]  = {{}};
  MonitorElement* tau_nCh_either_eta_[kNDMGen][kMaxCHLegs] = {{}};

  MonitorElement* tau_pt_reco_over_gen_[kNDMGen] = {};

  // reco tau shapes per DM (gen-level, use kNDMGen)
  MonitorElement* tau_reco_pt_[kNDMGen]  = {};
  MonitorElement* tau_reco_eta_[kNDMGen] = {};

  // CP-to-PF resolution: 1D ratio histograms (PF pT / CP pT) per DM (gen-level, use kNDMGen)
  MonitorElement* cp_pf_pt_resolution_had_dm_[kNDMGen] = {};  // hadronic (charged hadrons)
  MonitorElement* cp_pf_pt_resolution_em_dm_[kNDMGen]  = {};  // electromagnetic (photons)

  // ---------- Two-fold CP-level efficiency numerators (per-DM, use kNDMGen) ----------
  // trackOnly: CP has a reco track matched via TrackingParticle
  MonitorElement* cp_chHad_trackOnly_pt_dm_[kNDMGen]  = {};
  MonitorElement* cp_chHad_trackOnly_eta_dm_[kNDMGen] = {};
  // caloOnly: CP reached merged PF via TICL chain
  MonitorElement* cp_chHad_caloOnly_pt_dm_[kNDMGen]  = {};
  MonitorElement* cp_chHad_caloOnly_eta_dm_[kNDMGen] = {};
  // trackAndCalo: both criteria satisfied
  MonitorElement* cp_chHad_trackAndCalo_pt_dm_[kNDMGen]  = {};
  MonitorElement* cp_chHad_trackAndCalo_eta_dm_[kNDMGen] = {};
  // Photon equivalents (photons have no track, so trackOnly is always empty;
  // kept for uniform structure; caloOnly = TICL-matched)
  MonitorElement* cp_gamma_caloOnly_pt_dm_[kNDMGen]  = {};
  MonitorElement* cp_gamma_caloOnly_eta_dm_[kNDMGen] = {};

  // ---------- fake rate histograms ----------
  FakeRateHists fakeRate_;
  FakeRateHists fakeRateFilt_;
  FakeRateHists fakeRateCalo_;
  FakeRateHists fakeRateCaloFilt_;
  FakeRateHists fakeRateTrack_;
  FakeRateHists fakeRateTrackFilt_;

};

TICLTauValidator::TICLTauValidator(const edm::ParameterSet& iConfig)
  : folder_( iConfig.getParameter<std::string>("folder") ),
    maxAssocScore_( iConfig.getParameter<double>("maxAssocScore") ),
    hgcalEtaAbsMin_( iConfig.getParameter<double>("hgcalEtaAbsMin") )
{
  simTauToken_        = consumes<std::vector<SimTauCPLink>>( iConfig.getParameter<edm::InputTag>("simTaus") );
  {
    const auto caloParticlesTag = iConfig.getParameter<edm::InputTag>("caloParticles");
    if (!caloParticlesTag.label().empty()) {
      caloParticlesToken_ = mayConsume<std::vector<CaloParticle>>(caloParticlesTag);
    }
  }
  tauProducerToken_   = consumes<reco::PFTauCollection>(      iConfig.getParameter<edm::InputTag>("TauProducer") );

  pfToken_          = consumes<reco::PFCandidateCollection>(  iConfig.getParameter<edm::InputTag>("pf") );
  pfTmpBarrelToken_ = consumes<reco::PFCandidateCollection>(  iConfig.getParameter<edm::InputTag>("pfTmpBarrel") );

  ticlCandidatesToken_ = consumes<std::vector<TICLCandidate>>(   iConfig.getParameter<edm::InputTag>("ticlCandidates") );
  simTrackstersToken_  = consumes<std::vector<ticl::Trackster>>( iConfig.getParameter<edm::InputTag>("simTracksters") );
  allTrkToSimTrkAssocByLCsToken_ = consumes<TracksterToTracksterMap>(
    iConfig.getParameter<edm::InputTag>("simToRecoTracksterAssocByLCs") );
  pfJetsToken_       = consumes<reco::PFJetCollection>(        iConfig.getParameter<edm::InputTag>("jets") );
  genParticlesToken_ = consumes<reco::GenParticleCollection>(  iConfig.getParameter<edm::InputTag>("genParticles") );
  genVisTausToken_   = consumes<reco::GenParticleCollection>(    iConfig.getParameter<edm::InputTag>("genVisTaus") );
  trackingParticleToken_ = consumes<TrackingParticleCollection>( iConfig.getParameter<edm::InputTag>("trackingParticles") );
  recoToSimAssocByLCsToken_ = consumes<TracksterToTracksterMap>(
    iConfig.getParameter<edm::InputTag>("recoToSimTracksterAssocByLCs") );
  trackRecoToSimToken_ = consumes<reco::RecoToSimCollection>(
    iConfig.getParameter<edm::InputTag>("trackRecoToSim") );
  hltTauFilterLabels_ = iConfig.getParameter<std::vector<std::string>>("hltTauFilterLabels");
  hltProcessName_ = iConfig.getParameter<std::string>("hltProcessName");
  for (const auto& label : hltTauFilterLabels_) {
    hltFilterTokens_.push_back(
      mayConsume<trigger::TriggerFilterObjectWithRefs>(edm::InputTag(label, "", hltProcessName_))
    );
  }
}

void TICLTauValidator::bookHistograms(DQMStore::IBooker& ibook,
                                      edm::Run const&,
                                      edm::EventSetup const&) {
  ibook.setCurrentFolder(folder_);

  // ---- booking helpers ----
  // Book a matched pair of (pT, eta) histograms in one call.
  auto bookPtEta = [&](MonitorElement*& hpt, MonitorElement*& heta,
                        const std::string& base, const std::string& title) {
    hpt  = ibook.book1D(base + "_pt",  title + "; pT [GeV]; entries", 60, 0., 120.);
    heta = ibook.book1D(base + "_eta", title + "; eta; entries",       50, -3., 3.);
  };

  // Book a resolution histogram (pT ratio).
  auto bookRes = [&](MonitorElement*& h, const std::string& name, const std::string& title) {
    h = ibook.book1D(name, title + ";pT^{reco}/pT^{gen};entries", 60, 0., 3.);
  };

  // Book den+num for one step of a StepHistsT chain.
  auto bookStepDenNum = [&](MonitorElement*& denPt, MonitorElement*& denEta,
                            MonitorElement*& numPt, MonitorElement*& numEta,
                            const std::string& base, const std::string& title) {
    bookPtEta(denPt, denEta, base + "_den", "Den: " + title);
    bookPtEta(numPt, numEta, base + "_num", "Num: " + title);
  };

  // Context CP histos
  bookPtEta(cp_chHad_pt_all_, cp_chHad_eta_all_, "cp_chHad_all", "Charged CP");
  bookPtEta(cp_gamma_pt_all_, cp_gamma_eta_all_, "cp_gamma_all", "Photon CP");

  auto labelAxes = [](MonitorElement* me){
    if (!me) return;
    if (auto* h2 = me->getTH2F()) {
      // X (reco): 6 bins for {0,1,2,5,10,11}
      std::array<std::string, 6> lblReco = {{
        "1 #pi^{#pm}",                // DM 0
        "1 #pi^{#pm} 1 #pi^{0}",      // DM 1
        "1 #pi^{#pm} 2 #pi^{0}",      // DM 2
        "2 #pi^{#pm}",                // DM 5 (reco-only 2-prong)
        "3 #pi^{#pm}",                // DM 10
        "3 #pi^{#pm} 1 #pi^{0}"       // DM 11
      }};
      // Y (gen): 5 bins for {0,1,2,10,11} (no DM 5)
      std::array<std::string, 5> lblGen = {{
        "1 #pi^{#pm}",                // DM 0
        "1 #pi^{#pm} 1 #pi^{0}",      // DM 1
        "1 #pi^{#pm} 2 #pi^{0}",      // DM 2
        "3 #pi^{#pm}",                // DM 10
        "3 #pi^{#pm} 1 #pi^{0}"       // DM 11
      }};

      auto* xax = h2->GetXaxis();
      auto* yax = h2->GetYaxis();
      for (int i = 1; i <= static_cast<int>(lblReco.size()); ++i)
        xax->SetBinLabel(i, lblReco[i-1].c_str());
      for (int i = 1; i <= static_cast<int>(lblGen.size()); ++i)
        yax->SetBinLabel(i, lblGen[i-1].c_str());
    }
  };

  // Confusion matrices
  dm_reco_vs_gen_jet_ = ibook.book2D(
    "dm_reco_vs_gen_jet",
    "Reco DM in jet vs gen DM;reco DM index;gen DM index",
    6, -0.5, 5.5,
    5, -0.5, 4.5
  );
  labelAxes(dm_reco_vs_gen_jet_);

  dm_reco_vs_gen_tau_ = ibook.book2D(
    "dm_reco_vs_gen_tau",
    "Reco DM in tau vs gen DM;reco DM index;gen DM index",
    6, -0.5, 5.5,
    5, -0.5, 4.5
  );
  labelAxes(dm_reco_vs_gen_tau_);

  dm_reco_vs_gen_hps_ = ibook.book2D(
    "dm_reco_vs_gen_hps",
    "HPS tau decayMode vs gen DM;HPS DM index;gen DM index",
    6, -0.5, 5.5,
    5, -0.5, 4.5
  );
  labelAxes(dm_reco_vs_gen_hps_);

  // steps we save per-leg histos for
  const std::vector<int> stepsToKeep = {0, 1, 2, 3, 4, 5};

  for (int dmI = 0; dmI < kNDMGen; ++dmI) {
    int dm = kDMGen[dmI];
    ibook.setCurrentFolder(folder_ + "/GenDM" + std::to_string(dm));

    const int chCap    = chCapForDM(dm);
    const int gammaCap = std::min(2 * expectedPi0ForDM(dm), kMaxGammaLegs);

    // per-DM CP base histos
    std::string d = std::to_string(dm);
    bookPtEta(cp_chHad_pt_dm_[dmI], cp_chHad_eta_dm_[dmI],
              "cp_chHad_dm" + d, "Charged CP (DM=" + d + ")");
    bookPtEta(cp_gamma_pt_dm_[dmI], cp_gamma_eta_dm_[dmI],
              "cp_gamma_dm" + d, "Photon CP (DM=" + d + ")");

    // CP-to-PF pT resolution per DM
    bookRes(cp_pf_pt_resolution_had_dm_[dmI],
            "cp_pf_pt_resolution_hadronic_dm" + d,
            "Charged hadron CP-to-PF pT resolution (DM=" + d + ")");
    bookRes(cp_pf_pt_resolution_em_dm_[dmI],
            "cp_pf_pt_resolution_em_dm" + d,
            "Photon CP-to-PF pT resolution (DM=" + d + ")");

    // charged legs - calo chain
    for (int li = 0; li < chCap; ++li) {
      for (int s : stepsToKeep) {
        std::string base = "ch_dm" + d + "_leg" + std::to_string(li) + "_step" + std::to_string(s);
        std::string tag  = "charged; DM=" + d + " leg=" + std::to_string(li) + " step=" + std::to_string(s);
        bookStepDenNum(chStepHists_[dmI][li].den_pt[s],  chStepHists_[dmI][li].den_eta[s],
                       chStepHists_[dmI][li].num_pt[s],  chStepHists_[dmI][li].num_eta[s],
                       base, tag);
      }
    }

    // charged legs - track chain
    for (int li = 0; li < chCap; ++li) {
      for (int s = 0; s < kNTrackSteps; ++s) {
        std::string base = "ch_dm" + d + "_leg" + std::to_string(li) + "_trkstep" + std::to_string(s);
        std::string tag  = "charged track-chain; DM=" + d + " leg=" + std::to_string(li) + " trkStep=" + std::to_string(s);
        bookStepDenNum(chTrackStepHists_[dmI][li].den_pt[s],  chTrackStepHists_[dmI][li].den_eta[s],
                       chTrackStepHists_[dmI][li].num_pt[s],  chTrackStepHists_[dmI][li].num_eta[s],
                       base, tag);
      }
    }

    // charged legs - combi (AND) chain
    for (int li = 0; li < chCap; ++li) {
      for (int s = 0; s < kNCombiSteps; ++s) {
        std::string base = "ch_dm" + d + "_leg" + std::to_string(li) + "_combistep" + std::to_string(s);
        std::string tag  = "charged combi-chain; DM=" + d + " leg=" + std::to_string(li) + " combiStep=" + std::to_string(s);
        bookStepDenNum(chCombiStepHists_[dmI][li].den_pt[s],  chCombiStepHists_[dmI][li].den_eta[s],
                       chCombiStepHists_[dmI][li].num_pt[s],  chCombiStepHists_[dmI][li].num_eta[s],
                       base, tag);
      }
    }

    // photon legs - calo chain
    for (int li = 0; li < gammaCap; ++li) {
      for (int s : stepsToKeep) {
        std::string base = "pho_dm" + d + "_leg" + std::to_string(li) + "_step" + std::to_string(s);
        std::string tag  = "photon; DM=" + d + " leg=" + std::to_string(li) + " step=" + std::to_string(s);
        bookStepDenNum(gammaStepHists_[dmI][li].den_pt[s],  gammaStepHists_[dmI][li].den_eta[s],
                       gammaStepHists_[dmI][li].num_pt[s],  gammaStepHists_[dmI][li].num_eta[s],
                       base, tag);
      }
    }
  }

  // tau-level denominators & numerators (gen-level, so use kNDMGen)
  for (int dmI = 0; dmI < kNDMGen; ++dmI) {
    const int dm     = kDMGen[dmI];
    const int chCap  = chCapForDM(dm);
    const int pi0Cap = pi0CapForDM(dm);
    ibook.setCurrentFolder(folder_ + "/GenDM" + std::to_string(dm));

    std::string ds = std::to_string(dm);
    bookPtEta(tau_gen_pt_[dmI], tau_gen_eta_[dmI],
              "tau_dm" + ds + "_den", "DM " + ds + " gen tau");
    bookPtEta(tau_reco_pt_[dmI], tau_reco_eta_[dmI],
              "tau_dm" + ds + "_reco", "DM " + ds + " reco tau");
    bookRes(tau_pt_reco_over_gen_[dmI],
            "tau_dm" + ds + "_pt_reco_over_gen",
            "DM " + ds + " tau: pT_reco/pT_gen");

    // reco (jet/tau) endpoint: combined
    for (int N = 1; N <= chCap; ++N) {
      std::string ns = std::to_string(N);
      bookPtEta(tau_gen_matched_to_nCh_pt_[dmI][N-1], tau_gen_matched_to_nCh_eta_[dmI][N-1],
                "tau_dm" + ds + "_ge" + ns + "ch_num",
                "DM " + ds + " tau: >= " + ns + " charged at reco");
    }
    for (int N = 1; N <= pi0Cap; ++N) {
      std::string ns = std::to_string(N);
      bookPtEta(tau_gen_matched_to_nPi0_pt_[dmI][N-1], tau_gen_matched_to_nPi0_eta_[dmI][N-1],
                "tau_dm" + ds + "_ge" + ns + "pi0_num",
                "DM " + ds + " tau: >= " + ns + " pi0 at reco");
    }
    bookPtEta(tau_gen_matched_to_all_pt_[dmI], tau_gen_matched_to_all_eta_[dmI],
              "tau_dm" + ds + "_all_num",
              "DM " + ds + " tau: all expected charged+pi0 at reco");

    // TAU-only endpoint: signal vs iso
    for (int N = 1; N <= chCap; ++N) {
      std::string ns = std::to_string(N);
      bookPtEta(tau_gen_matched_to_nCh_sig_pt_[dmI][N-1], tau_gen_matched_to_nCh_sig_eta_[dmI][N-1],
                "tau_dm" + ds + "_ge" + ns + "ch_num_signal",
                "DM " + ds + " tau: >= " + ns + " charged in signal");
      bookPtEta(tau_gen_matched_to_nCh_iso_pt_[dmI][N-1], tau_gen_matched_to_nCh_iso_eta_[dmI][N-1],
                "tau_dm" + ds + "_ge" + ns + "ch_num_iso",
                "DM " + ds + " tau: >= " + ns + " charged in isolation");
    }
    for (int N = 1; N <= pi0Cap; ++N) {
      std::string ns = std::to_string(N);
      bookPtEta(tau_gen_matched_to_nPi0_sig_pt_[dmI][N-1], tau_gen_matched_to_nPi0_sig_eta_[dmI][N-1],
                "tau_dm" + ds + "_ge" + ns + "pi0_num_signal",
                "DM " + ds + " tau: >= " + ns + " pi0 in signal");
      bookPtEta(tau_gen_matched_to_nPi0_iso_pt_[dmI][N-1], tau_gen_matched_to_nPi0_iso_eta_[dmI][N-1],
                "tau_dm" + ds + "_ge" + ns + "pi0_num_iso",
                "DM " + ds + " tau: >= " + ns + " pi0 in isolation");
    }

    // Tau-level two-fold: >= N charged CPs matched by track / calo / both / either
    for (int N = 1; N <= chCap; ++N) {
      std::string ns = std::to_string(N);
      bookPtEta(tau_nCh_track_pt_[dmI][N-1], tau_nCh_track_eta_[dmI][N-1],
                "tau_dm" + ds + "_ge" + ns + "ch_track_num",
                "DM " + ds + " tau: >= " + ns + " charged track-matched");
      bookPtEta(tau_nCh_calo_pt_[dmI][N-1], tau_nCh_calo_eta_[dmI][N-1],
                "tau_dm" + ds + "_ge" + ns + "ch_calo_num",
                "DM " + ds + " tau: >= " + ns + " charged calo-matched");
      bookPtEta(tau_nCh_both_pt_[dmI][N-1], tau_nCh_both_eta_[dmI][N-1],
                "tau_dm" + ds + "_ge" + ns + "ch_both_num",
                "DM " + ds + " tau: >= " + ns + " charged track AND calo");
      bookPtEta(tau_nCh_either_pt_[dmI][N-1], tau_nCh_either_eta_[dmI][N-1],
                "tau_dm" + ds + "_ge" + ns + "ch_either_num",
                "DM " + ds + " tau: >= " + ns + " charged track OR calo");
    }

    // Two-fold CP-level efficiency numerators
    bookPtEta(cp_chHad_trackOnly_pt_dm_[dmI], cp_chHad_trackOnly_eta_dm_[dmI],
              "cp_chHad_dm" + ds + "_trackOnly", "DM " + ds + " charged CP: track-matched");
    bookPtEta(cp_chHad_caloOnly_pt_dm_[dmI], cp_chHad_caloOnly_eta_dm_[dmI],
              "cp_chHad_dm" + ds + "_caloOnly", "DM " + ds + " charged CP: calo-matched (TICL)");
    bookPtEta(cp_chHad_trackAndCalo_pt_dm_[dmI], cp_chHad_trackAndCalo_eta_dm_[dmI],
              "cp_chHad_dm" + ds + "_trackAndCalo", "DM " + ds + " charged CP: track AND calo matched");
    bookPtEta(cp_gamma_caloOnly_pt_dm_[dmI], cp_gamma_caloOnly_eta_dm_[dmI],
              "cp_gamma_dm" + ds + "_caloOnly", "DM " + ds + " photon CP: calo-matched (TICL)");
  }

  // ---------- Fake rate histograms ----------
  auto bookFakeRateSet = [&](const std::string& prefix, const std::string& tagExtra, FakeRateHists& fr) {
    ibook.setCurrentFolder(folder_ + "/FakeRate");
    std::string tag = tagExtra.empty() ? "Fake rate" : ("Fake rate (" + tagExtra + ")");
    bookStepDenNum(fr.den_pt, fr.den_eta, fr.num_pt, fr.num_eta, prefix, tag);

    for (int N = 1; N <= kMaxCHLegs; ++N) {
      std::string ns = std::to_string(N);
      bookStepDenNum(fr.den_sig_nCh_pt[N-1], fr.den_sig_nCh_eta[N-1],
                     fr.num_sig_nCh_pt[N-1], fr.num_sig_nCh_eta[N-1],
                     prefix + "_ge" + ns + "ch_sig",
                     tag + " (>= " + ns + " charged in signal)");
    }
    for (int N = 1; N <= kMaxPi0Legs; ++N) {
      std::string ns = std::to_string(N);
      bookStepDenNum(fr.den_sig_nPi0_pt[N-1], fr.den_sig_nPi0_eta[N-1],
                     fr.num_sig_nPi0_pt[N-1], fr.num_sig_nPi0_eta[N-1],
                     prefix + "_ge" + ns + "pi0_sig",
                     tag + " (>= " + ns + " pi0 in signal)");
    }

    for (int i = 0; i < kNDMSel; ++i) {
      int dm = kDMSel[i];
      std::string ds = std::to_string(dm);
      ibook.setCurrentFolder(folder_ + "/GenDM" + ds + "/FakeRate");
      bookStepDenNum(fr.den_dm_pt[i], fr.den_dm_eta[i], fr.num_dm_pt[i], fr.num_dm_eta[i],
                     prefix + "_dm" + ds, tag + " (DM=" + ds + ")");
      for (int N = 1; N <= kMaxCHLegs; ++N) {
        std::string ns = std::to_string(N);
        bookStepDenNum(fr.den_dm_sig_nCh_pt[i][N-1], fr.den_dm_sig_nCh_eta[i][N-1],
                       fr.num_dm_sig_nCh_pt[i][N-1], fr.num_dm_sig_nCh_eta[i][N-1],
                       prefix + "_dm" + ds + "_ge" + ns + "ch_sig",
                       tag + " (DM=" + ds + ", >= " + ns + " charged in signal)");
      }
      for (int N = 1; N <= kMaxPi0Legs; ++N) {
        std::string ns = std::to_string(N);
        bookStepDenNum(fr.den_dm_sig_nPi0_pt[i][N-1], fr.den_dm_sig_nPi0_eta[i][N-1],
                       fr.num_dm_sig_nPi0_pt[i][N-1], fr.num_dm_sig_nPi0_eta[i][N-1],
                       prefix + "_dm" + ds + "_ge" + ns + "pi0_sig",
                       tag + " (DM=" + ds + ", >= " + ns + " pi0 in signal)");
      }
    }
  };

  bookFakeRateSet("fake", "", fakeRate_);
  bookFakeRateSet("fake_chargedIsoPath", "charged iso path taus", fakeRateFilt_);
  bookFakeRateSet("fake_calo", "calo assoc", fakeRateCalo_);
  bookFakeRateSet("fake_calo_chargedIsoPath", "calo assoc, charged iso path taus", fakeRateCaloFilt_);
  bookFakeRateSet("fake_track", "track assoc", fakeRateTrack_);
  bookFakeRateSet("fake_track_chargedIsoPath", "track assoc, charged iso path taus", fakeRateTrackFilt_);
}

void TICLTauValidator::analyze(const edm::Event& iEvent,
                               const edm::EventSetup&) {

  // handles
  edm::Handle<std::vector<SimTauCPLink>> simTaus;            iEvent.getByToken(simTauToken_, simTaus);
  edm::Handle<std::vector<CaloParticle>> caloParticles;
  if (!caloParticlesToken_.isUninitialized())
    iEvent.getByToken(caloParticlesToken_, caloParticles);
  edm::Handle<reco::PFTauCollection>     taus;               iEvent.getByToken(tauProducerToken_, taus);
  edm::Handle<std::vector<ticl::Trackster>> simTracksters;   iEvent.getByToken(simTrackstersToken_, simTracksters);
  edm::Handle<TracksterToTracksterMap>   simToRecoMap;       iEvent.getByToken(allTrkToSimTrkAssocByLCsToken_, simToRecoMap);
  edm::Handle<TracksterToTracksterMap>   recoToSimMap;       iEvent.getByToken(recoToSimAssocByLCsToken_, recoToSimMap);
  edm::Handle<std::vector<TICLCandidate>> ticlCandidates;    iEvent.getByToken(ticlCandidatesToken_, ticlCandidates);
  edm::Handle<reco::PFCandidateCollection> pfMerged;         iEvent.getByToken(pfToken_, pfMerged);
  edm::Handle<reco::PFCandidateCollection> pfTmpBarrel;      iEvent.getByToken(pfTmpBarrelToken_, pfTmpBarrel);
  edm::Handle<reco::PFJetCollection>      pfJets;            iEvent.getByToken(pfJetsToken_, pfJets);
  edm::Handle<reco::GenParticleCollection> genParticles;     iEvent.getByToken(genParticlesToken_, genParticles);
  edm::Handle<reco::GenParticleCollection> genVisTaus;       iEvent.getByToken(genVisTausToken_, genVisTaus);
  edm::Handle<TrackingParticleCollection> trackingParticles;  iEvent.getByToken(trackingParticleToken_, trackingParticles);
  edm::Handle<reco::RecoToSimCollection> trackRecoToSim;       iEvent.getByToken(trackRecoToSimToken_, trackRecoToSim);

  if (!taus.isValid())
    return;

  std::vector<reco::PFTauRef> finalFilterTauRefs;

  // Access HLT filter products and extract PFTau refs
  for (size_t fi = 0; fi < hltTauFilterLabels_.size(); ++fi) {
    edm::Handle<trigger::TriggerFilterObjectWithRefs> filterProduct;
    iEvent.getByToken(hltFilterTokens_[fi], filterProduct);
    if (filterProduct.isValid()) {
      trigger::VRpftau tauRefs;
      filterProduct->getObjects(trigger::TriggerTau, tauRefs);
      if (hltTauFilterLabels_[fi] == "hltHpsDoublePFTau40TrackPt1MediumChargedIsolation")
        finalFilterTauRefs = tauRefs;
      // HLT filter products accessed for final filter tau reference extraction
    } else {
      edm::LogWarning("TICLTauValidator") << "HLT filter " << hltTauFilterLabels_[fi]
                                          << " product not available";
    }
  }

  if (!simTaus.isValid()) {
    edm::LogWarning("TICLTauValidator") << "simTaus invalid, skipping event " << iEvent.id();
    return;
  }

  std::vector<std::string> eventChains;
  std::set<unsigned int> seenCPChargedEvent, seenCPPhotonEvent;
  std::set<unsigned int> seenCPTwoFoldCharged, seenCPTwoFoldPhoton;  // dedup for two-fold numerators

  // Build set of TrackingParticle indices that have at least one matched reco track.
  // Used for CP-level forward track matching (CP -> TP -> reco track).
  std::unordered_set<size_t> tpWithRecoTrack;
  // Reverse map: TP index -> reco track keys (for track chain steps)
  std::unordered_map<size_t, std::vector<size_t>> tpToRecoTrackKeys;
  if (trackRecoToSim.isValid() && trackingParticles.isValid()) {
    for (const auto& entry : *trackRecoToSim) {
      for (const auto& tpMatch : entry.val) {
        const size_t tpKey = tpMatch.first.key();
        tpWithRecoTrack.insert(tpKey);
        tpToRecoTrackKeys[tpKey].push_back(entry.key.key());
      }
    }
  }

  // Map: reco track key -> PF candidate indices in merged collection (for track chain)
  std::unordered_map<size_t, std::vector<size_t>> trackKeyToPFIdx;
  if (pfMerged.isValid()) {
    for (size_t pfi = 0; pfi < pfMerged->size(); ++pfi) {
      const auto& trkRef = (*pfMerged)[pfi].trackRef();
      if (trkRef.isNonnull())
        trackKeyToPFIdx[trkRef.key()].push_back(pfi);
    }
  }

  // Pre-build G4 track -> TP index map
  // Only hard-scatter (eventId == 0) G4 tracks are relevant, so the key is just trackId.
  std::unordered_map<unsigned int, std::vector<size_t>> g4ToTPIdx;
  if (trackingParticles.isValid()) {
    for (size_t tpIdx = 0; tpIdx < trackingParticles->size(); ++tpIdx) {
      for (const auto& g4 : (*trackingParticles)[tpIdx].g4Tracks()) {
        if (g4.eventId().event() != 0)
          continue;
        g4ToTPIdx[g4.trackId()].push_back(tpIdx);
      }
    }
  }

  const size_t barrelSize = pfTmpBarrel.isValid() ? pfTmpBarrel->size() : 0;

  for (const auto& link : *simTaus) {
    const int dmPhys = link.decayMode;
    const int dmSelIdx = dmToSelIndex(dmPhys);
    const int dmGenIdx = dmToGenIndex(dmPhys);
    if (dmSelIdx < 0)
      continue; // ignore non-selected DMs
    if (dmGenIdx < 0)
      continue; // ignore non-physical gen DMs (e.g., DM 5)

    struct PtEta { float pt=0.f, eta=0.f; };
    struct TauRegionFlags { bool signal=false, isolation=false; };
    struct PendingInfo {
      float cpPt=0.f, cpEta=0.f;
      float pfPt=0.f, pfEta=0.f;
      int   pdgId=0; 
      bool  hasCPKinematics=false;
      bool  hasPFKinematics=false;
      bool  trackMatched=false;  
      std::array<bool, kNSteps> stepPass{}; // calo chain steps
      std::array<bool, kNTrackSteps> trackStepPass{};  // track chain steps
      std::array<bool, kNCombiSteps> combiStepPass{};   // AND of calo + track at equivalent endpoints
      std::set<size_t> jets;
      std::set<size_t> trackJets;    // jets containing track-chain PF candidates
      std::set<size_t> trackPFKeys;  // PF keys matched via track chain
      std::unordered_map<size_t, std::vector<int>> dmByJet;
      std::unordered_map<size_t, TauRegionFlags>   tauRegion;
      std::unordered_map<size_t, TauRegionFlags>   trackTauRegion; // tau regions for track-chain PFs
    };

    std::unordered_map<unsigned, PtEta> chKinTicl, phoKinTicl;
    std::unordered_map<unsigned, PendingInfo> pendingHad;
    std::unordered_map<unsigned, PendingInfo> pendingGamma;

    // --- loop leaves ---
    for (const auto& leaf : link.leaves) {
      const int cp_id = leaf.calo_particle_idx();
      if (cp_id < 0 || static_cast<size_t>(cp_id) >= link.calo_particle_leaves.size())
        continue;
      const auto& cpRef = link.calo_particle_leaves[cp_id];
      if (!cpRef.isNonnull())
        continue;
      const auto& cp = *cpRef;

      // keep only CPs in HGCAL 
      if (std::abs(cp.eta()) < hgcalEtaAbsMin_)
        continue;

      const int absPdg = std::abs(cp.pdgId());
      const bool isPhoton        = (absPdg == 22);
      const bool isChargedHadron = (absPdg == 211 || absPdg == 321 || absPdg == 2212);

      // context CP histos (once per event per CP key) + per-DM base histos
      if (isChargedHadron) {
        const bool firstTime = seenCPChargedEvent.insert(cpRef.key()).second;
        if (firstTime) {
          if (cp_chHad_pt_all_)  cp_chHad_pt_all_->Fill(cp.pt());
          if (cp_chHad_eta_all_) cp_chHad_eta_all_->Fill(cp.eta());
          if (dmGenIdx >= 0) {
            if (cp_chHad_pt_dm_[dmGenIdx])  cp_chHad_pt_dm_[dmGenIdx]->Fill(cp.pt());
            if (cp_chHad_eta_dm_[dmGenIdx]) cp_chHad_eta_dm_[dmGenIdx]->Fill(cp.eta());
          }
        }
      }
      if (isPhoton) {
        const bool firstTime = seenCPPhotonEvent.insert(cpRef.key()).second;
        if (firstTime) {
          if (cp_gamma_pt_all_)  cp_gamma_pt_all_->Fill(cp.pt());
          if (cp_gamma_eta_all_) cp_gamma_eta_all_->Fill(cp.eta());
          if (dmGenIdx >= 0) {
            if (cp_gamma_pt_dm_[dmGenIdx])  cp_gamma_pt_dm_[dmGenIdx]->Fill(cp.pt());
            if (cp_gamma_eta_dm_[dmGenIdx]) cp_gamma_eta_dm_[dmGenIdx]->Fill(cp.eta());
          }
        }
      }

      if (isChargedHadron) chKinTicl[cpRef.key()] = {cp.pt(), cp.eta()};
      if (isPhoton)        phoKinTicl[cpRef.key()] = {cp.pt(), cp.eta()};

      // Step 0: CP - SimTrackster
      size_t matchedSimTkIdx = (size_t)-1;
      if (simTracksters.isValid()) {
        for (size_t si = 0; si < simTracksters->size(); ++si) {
          const auto& simTk = (*simTracksters)[si];
          const int seedIdx = simTk.seedIndex();
          if (simTk.seedID() == cpRef.id() && seedIdx >= 0 &&
              static_cast<unsigned>(seedIdx) == cpRef.key()) {
            matchedSimTkIdx = si;
            break;
          }
        }
      } else {
        edm::LogWarning("TICLTauValidator")
          << "simTracksters collection missing ";
      }
      if (matchedSimTkIdx == (size_t)-1)
        continue;

      if (!isChargedHadron && !isPhoton)
        continue;

      auto& pend = (isChargedHadron) ? pendingHad[cpRef.key()] : pendingGamma[cpRef.key()];
      pend.pdgId = cp.pdgId();
      pend.hasCPKinematics = true;
      pend.cpPt  = cp.pt();
      pend.cpEta = cp.eta();
      pend.stepPass[0] = true;


      if (isChargedHadron && trackingParticles.isValid()) {
        // Track Step 0: CP -> TP (find any TP sharing G4 tracks with this CP)
        std::vector<size_t> matchedTPIndices;
        for (const auto& cpG4 : cp.g4Tracks()) {
          if (cpG4.eventId().event() != 0)
            continue;
          auto it = g4ToTPIdx.find(cpG4.trackId());
          if (it != g4ToTPIdx.end()) {
            for (size_t tpIdx : it->second)
              matchedTPIndices.push_back(tpIdx);
          }
        }
        if (!matchedTPIndices.empty())
          pend.trackStepPass[0] = true;

        // Track Step 1: TP -> reco Track
        std::unordered_set<size_t> matchedRecoTrackKeysSet;
        for (size_t tpIdx : matchedTPIndices) {
          auto it = tpToRecoTrackKeys.find(tpIdx);
          if (it != tpToRecoTrackKeys.end()) {
            for (size_t trkKey : it->second)
              matchedRecoTrackKeysSet.insert(trkKey);
          }
        }
        if (!matchedRecoTrackKeysSet.empty()) {
          pend.trackStepPass[1] = true;
          pend.trackMatched = true; 
        }

        // Track Step 2: reco Track -> PF candidate
        for (size_t trkKey : matchedRecoTrackKeysSet) {
          auto it = trackKeyToPFIdx.find(trkKey);
          if (it != trackKeyToPFIdx.end()) {
            for (size_t pfIdx : it->second)
              pend.trackPFKeys.insert(pfIdx);
          }
        }
        if (!pend.trackPFKeys.empty())
          pend.trackStepPass[2] = true;

        // Track Step 3: PF -> PFJet
        if (pfJets.isValid() && pfMerged.isValid()) {
          for (size_t pfIdx : pend.trackPFKeys) {
            reco::PFCandidateRef pfRef(pfMerged, pfIdx);
            for (size_t j = 0; j < pfJets->size(); ++j) {
              const auto& jet = (*pfJets)[j];
              for (const auto& pfPtr : jet.getPFConstituents()) {
                if (pfPtr.isNonnull() &&
                    pfPtr.id() == pfRef.id() && pfPtr.key() == pfRef.key()) {
                  pend.trackJets.insert(j);
                  break;
                }
              }
            }
          }
          if (!pend.trackJets.empty())
            pend.trackStepPass[3] = true;

          // Track Step 4: PFJet -> PFTau (signal region)
          if (taus.isValid()) {
            for (size_t t = 0; t < taus->size(); ++t) {
              const auto& tau = (*taus)[t];
              auto jetRef = tau.jetRef();
              if (!jetRef)
                continue;
              bool jetMatch = false;
              for (size_t j : pend.trackJets) {
                if (jetRef.get() == &(*pfJets)[j]) {
                  jetMatch = true;
                  break;
                }
              }
              if (!jetMatch)
                continue;
              // check if any track-chain PF is in signal/isolation
              for (size_t pfIdx : pend.trackPFKeys) {
                reco::PFCandidateRef pfRef(pfMerged, pfIdx);
                auto& flags = pend.trackTauRegion[t];
                if (!flags.signal) {
                  for (const auto& p : tau.signalPFCands()) {
                    if (p.id() == pfRef.id() && p.key() == pfRef.key()) {
                      flags.signal = true;
                      break;
                    }
                  }
                }
                if (!flags.isolation) {
                  for (const auto& p : tau.isolationPFCands()) {
                    if (p.id() == pfRef.id() && p.key() == pfRef.key()) {
                      flags.isolation = true;
                      break;
                    }
                  }
                }
              }
            }
            bool usedInTau = false;
            for (const auto& tr : pend.trackTauRegion) {
              if (tr.second.signal || tr.second.isolation) {
                usedInTau = true;
                break;
              }
            }
            if (usedInTau)
              pend.trackStepPass[4] = true;
          }
        }
      }

      // Step 1: SimToReco & Step 2: Reco - TICLCandidate
      std::vector<size_t> matchedRecoTkIdx;

      if (!simToRecoMap.isValid()) {
        // Product not there at all
        edm::LogWarning("TICLTauValidator")
            << "Trackster association map is missing.";
      } else if (simToRecoMap->size() == 0 ||
                 matchedSimTkIdx >= simToRecoMap->size()) {
        edm::LogWarning("TICLTauValidator")
            << "Trackster association map is empty "
               "or does not contain entry for sim trackster index "
            << matchedSimTkIdx << ".";
      } else {
        auto const& assocs = (*simToRecoMap)[matchedSimTkIdx];
        for (const auto& m : assocs) {
          if (m.score() <= maxAssocScore_)  // smaller is better
            matchedRecoTkIdx.push_back(m.index());
        }
      }

      if (!matchedRecoTkIdx.empty())
        pend.stepPass[1] = true;

      std::vector<size_t> candIdxs;
      if (ticlCandidates.isValid()) {
        for (size_t ci = 0; ci < ticlCandidates->size(); ++ci) {
          const auto& cand = (*ticlCandidates)[ci];
          bool uses = false;
          for (const auto& tsPtr : cand.tracksters()) {
            if (!tsPtr.isNonnull())
              continue;
            const size_t tkKey = tsPtr.key();
            if (std::find(matchedRecoTkIdx.begin(), matchedRecoTkIdx.end(), tkKey)
                != matchedRecoTkIdx.end()) {
              uses = true;
              break;
            }
          }
          if (uses)
            candIdxs.push_back(ci);
        }
      } else {
        edm::LogWarning("TICLTauValidator")
          << "TICLCandidate collection missing";
      }
      if (!candIdxs.empty())
        pend.stepPass[2] = true;

      // Step 3: TICLCandidate - PF
      if (pfMerged.isValid()) {
        for (size_t ci : candIdxs) {
          // NOTE: we assume that TICL PF candidates are appended after the barrel part
          const size_t pfIdx = barrelSize + ci;
          if (pfIdx >= pfMerged->size())
            continue;
          const auto& pfCand = (*pfMerged)[pfIdx];
          pend.hasPFKinematics = true;
          pend.pfPt  = pfCand.pt();
          pend.pfEta = pfCand.eta();
          pend.stepPass[3] = true;

          // Fill CP-to-PF resolution histograms (per-DM): ratio PF pT / CP pT
          if (cp.pt() > 0.) {
            const double ratio = pfCand.pt() / cp.pt();
            if (isChargedHadron && dmGenIdx >= 0 && cp_pf_pt_resolution_had_dm_[dmGenIdx]) {
              cp_pf_pt_resolution_had_dm_[dmGenIdx]->Fill(ratio);
            }
            if (isPhoton && dmGenIdx >= 0 && cp_pf_pt_resolution_em_dm_[dmGenIdx]) {
              cp_pf_pt_resolution_em_dm_[dmGenIdx]->Fill(ratio);
            }
          }

          // Step 4: PFCand - PFJet membership
          if (pfJets.isValid()) {
            reco::PFCandidateRef pfRef(pfMerged, pfIdx);
            for (size_t j = 0; j < pfJets->size(); ++j) {
              const auto& jet = (*pfJets)[j];
              for (const auto& pfPtr : jet.getPFConstituents()) {
                if (!pfPtr.isNonnull())
                  continue;
                if (pfPtr.id() == pfRef.id() && pfPtr.key() == pfRef.key()) {
                  pend.jets.insert(j);
                  break;
                }
              }
            }
            if (!pend.jets.empty())
              pend.stepPass[4] = true;

            // Step 5: PFJet - HPS tau
            if (taus.isValid()) {
              for (size_t t = 0; t < taus->size(); ++t) {
                const auto& tau = (*taus)[t];
                auto jetRef = tau.jetRef();
                if (!jetRef)
                  continue;
                size_t jetIdx = (size_t)-1;
                for (size_t j : pend.jets) {
                  if (jetRef.get() == &(*pfJets)[j]) {
                    jetIdx = j;
                    break;
                  }
                }
                if (jetIdx == (size_t)-1)
                  continue;

                pend.dmByJet[jetIdx].push_back(tau.decayMode());
                auto& flags = pend.tauRegion[t];
                if (!flags.signal) {
                  for (const auto& p : tau.signalPFCands()) {
                    if (p.id() == pfRef.id() && p.key() == pfRef.key()) {
                      flags.signal = true;
                      break;
                    }
                  }
                }
                if (!flags.isolation) {
                  for (const auto& p : tau.isolationPFCands()) {
                    if (p.id() == pfRef.id() && p.key() == pfRef.key()) {
                      flags.isolation = true;
                      break;
                    }
                  }
                }
              }
            }
          }
        }
      }

      // Compute AND chain: calo AND track must both pass at equivalent endpoint

      pend.combiStepPass[0] = pend.stepPass[3] && pend.trackStepPass[2];
      pend.combiStepPass[1] = pend.stepPass[4] && pend.trackStepPass[3];
      // combiStepPass[2] is computed after the leaf loop once stepPass[5] is known

      std::ostringstream ss;
      ss << "Chain: CP=" << cpRef.key() << " pdgId=" << cp.pdgId()
          << " caloSteps=[";
      for (int s = 0; s < kNSteps; ++s)
        ss << (pend.stepPass[s] ? '1' : '0');
      ss << "] trkSteps=[";
      for (int s = 0; s < kNTrackSteps; ++s)
        ss << (pend.trackStepPass[s] ? '1' : '0');
      ss << "] combiSteps=[";
      for (int s = 0; s < kNCombiSteps; ++s)
        ss << (pend.combiStepPass[s] ? '1' : '0');
      ss << "] jets=[";
      size_t c = 0;
      for (auto j : pend.jets) {
        if (c++) ss << ",";
        ss << j;
      }
      ss << "]";
      eventChains.push_back(ss.str());

      // Fill CP-level two-fold efficiency numerators (once per unique CP key per event)
      if (dmGenIdx >= 0 && pend.hasCPKinematics) {
        if (isChargedHadron && seenCPTwoFoldCharged.insert(cpRef.key()).second) {
          if (pend.trackMatched) {
            if (cp_chHad_trackOnly_pt_dm_[dmGenIdx])  cp_chHad_trackOnly_pt_dm_[dmGenIdx]->Fill(pend.cpPt);
            if (cp_chHad_trackOnly_eta_dm_[dmGenIdx]) cp_chHad_trackOnly_eta_dm_[dmGenIdx]->Fill(pend.cpEta);
          }
          if (pend.stepPass[3]) {
            if (cp_chHad_caloOnly_pt_dm_[dmGenIdx])  cp_chHad_caloOnly_pt_dm_[dmGenIdx]->Fill(pend.cpPt);
            if (cp_chHad_caloOnly_eta_dm_[dmGenIdx]) cp_chHad_caloOnly_eta_dm_[dmGenIdx]->Fill(pend.cpEta);
          }
          if (pend.trackMatched && pend.stepPass[3]) {
            if (cp_chHad_trackAndCalo_pt_dm_[dmGenIdx])  cp_chHad_trackAndCalo_pt_dm_[dmGenIdx]->Fill(pend.cpPt);
            if (cp_chHad_trackAndCalo_eta_dm_[dmGenIdx]) cp_chHad_trackAndCalo_eta_dm_[dmGenIdx]->Fill(pend.cpEta);
          }
        }
        if (isPhoton && seenCPTwoFoldPhoton.insert(cpRef.key()).second) {
          if (pend.stepPass[3]) {
            if (cp_gamma_caloOnly_pt_dm_[dmGenIdx])  cp_gamma_caloOnly_pt_dm_[dmGenIdx]->Fill(pend.cpPt);
            if (cp_gamma_caloOnly_eta_dm_[dmGenIdx]) cp_gamma_caloOnly_eta_dm_[dmGenIdx]->Fill(pend.cpEta);
          }
        }
      }
    } // leaves

    // Jets at step 4: unique jet that contains all PF legs
    std::set<size_t> commonJetsAllPF;
    {
      std::vector<std::set<size_t>> jetSets;
      for (const auto& kv : pendingHad)
        if (kv.second.stepPass[4]) jetSets.push_back(kv.second.jets);
      for (const auto& kv : pendingGamma)
        if (kv.second.stepPass[4]) jetSets.push_back(kv.second.jets);

      if (!jetSets.empty()) {
        commonJetsAllPF = jetSets.front();
        for (size_t i = 1; i < jetSets.size(); ++i) {
          std::set<size_t> tmp;
          std::set_intersection(commonJetsAllPF.begin(), commonJetsAllPF.end(),
                                jetSets[i].begin(), jetSets[i].end(),
                                std::inserter(tmp, tmp.begin()));
          commonJetsAllPF.swap(tmp);
          if (commonJetsAllPF.empty())
            break;
        }
      }
    }
    const bool   hasUniqueJetEndpoint = (commonJetsAllPF.size() == 1);
    const size_t jetEndpoint          = hasUniqueJetEndpoint ? *commonJetsAllPF.begin() : (size_t)-1;

    int nGoodCH_jet     = 0;
    int photonsUsed_jet = 0;
    if (hasUniqueJetEndpoint) {
      for (const auto& kv : pendingHad) {
        const auto& info = kv.second;
        if (info.stepPass[4] && info.jets.count(jetEndpoint))
          ++nGoodCH_jet;
      }
      for (const auto& kv : pendingGamma) {
        const auto& info = kv.second;
        if (info.stepPass[4] && info.jets.count(jetEndpoint))
          ++photonsUsed_jet;
      }
    }
    const int nPi0_jet = photonsUsed_jet / 2;

    // confusion matrix for jets
    if (dm_reco_vs_gen_jet_) {
      int recoDM = -1;
      if      (nGoodCH_jet == 1) {
        if      (nPi0_jet <= 0) recoDM = 0;
        else if (nPi0_jet == 1) recoDM = 1;
        else                    recoDM = 2;
      } else if (nGoodCH_jet == 2) {
        recoDM = 5;
      } else if (nGoodCH_jet == 3) {
        recoDM = (nPi0_jet >= 1 ? 11 : 10);
      }
      const int gSel = dmGenIdx;
      const int rSel = dmToSelIndex(recoDM);
      if (gSel >= 0 && rSel >= 0)
        dm_reco_vs_gen_jet_->Fill(rSel, gSel);
    }

    // taus - step 5
    const bool haveHpsTaus = (taus.isValid() && !taus->empty() && pfJets.isValid());
    int bestTauIdxForLink = -1;

    int nGoodCH_tau_total        = 0;
    int nGoodCH_signal_tau       = 0;
    int nGoodCH_iso_tau          = 0;
    int photonsUsed_tau_total    = 0;
    int photonsUsed_signal_tau   = 0;
    int photonsUsed_iso_tau      = 0;

    if (haveHpsTaus) {
      auto processLegTauEndpoint = [&](PendingInfo& info, bool isPhoton) {
        bool usedSignal = false;
        bool usedIso    = false;

        for (const auto& tr : info.tauRegion) {
          const auto& flags = tr.second;
          if (flags.signal)    usedSignal = true;
          if (flags.isolation) usedIso    = true;
        }

        bool usedTotal = (usedSignal || usedIso);
        info.stepPass[5] = usedTotal;

        if (!usedTotal)
          return;

        if (!isPhoton) {
          ++nGoodCH_tau_total;
          if (usedSignal) ++nGoodCH_signal_tau;
          if (usedIso)    ++nGoodCH_iso_tau;
        } else {
          ++photonsUsed_tau_total;
          if (usedSignal) ++photonsUsed_signal_tau;
          if (usedIso)    ++photonsUsed_iso_tau;
        }
      };

      for (auto& kv : pendingHad)   processLegTauEndpoint(kv.second, false);
      for (auto& kv : pendingGamma) processLegTauEndpoint(kv.second, true);
    } else {
      // No taus: step-5 remains false for all legs
      for (auto& kv : pendingHad)   kv.second.stepPass[5] = false;
      for (auto& kv : pendingGamma) kv.second.stepPass[5] = false;
    }

    // Now that stepPass[5] is known, compute combiStepPass[2] for all legs
    for (auto& kv : pendingHad)
      kv.second.combiStepPass[2] = kv.second.stepPass[5] && kv.second.trackStepPass[4];
    for (auto& kv : pendingGamma)
      kv.second.combiStepPass[2] = kv.second.stepPass[5] && kv.second.trackStepPass[4];

    const int nPi0_tau_total   = photonsUsed_tau_total   / 2;
    const int nPi0_signal_tau  = photonsUsed_signal_tau  / 2;
    const int nPi0_iso_tau     = photonsUsed_iso_tau     / 2;

    // Tau-level two-fold: count charged CPs matched by track / calo / both / either
    int nCh_track = 0, nCh_calo = 0, nCh_both = 0, nCh_either = 0;
    for (const auto& kv : pendingHad) {
      const auto& info = kv.second;
      const bool trk  = info.trackMatched;
      const bool cal  = info.stepPass[3];
      if (trk)          ++nCh_track;
      if (cal)          ++nCh_calo;
      if (trk && cal)   ++nCh_both;
      if (trk || cal)   ++nCh_either;
    }

    // confusion matrix for taus - use reconstructed leg counts
    if (haveHpsTaus && dm_reco_vs_gen_tau_) {
      int recoDM = -1;
      if      (nGoodCH_tau_total == 1) {
        if      (nPi0_tau_total <= 0) recoDM = 0;
        else if (nPi0_tau_total == 1) recoDM = 1;
        else                          recoDM = 2;
      } else if (nGoodCH_tau_total == 2) {
        recoDM = 5;
      } else if (nGoodCH_tau_total == 3) {
        recoDM = (nPi0_tau_total >= 1 ? 11 : 10);
      }
      const int gSel = dmGenIdx;
      const int rSel = dmToSelIndex(recoDM);
      if (gSel >= 0 && rSel >= 0)
        dm_reco_vs_gen_tau_->Fill(rSel, gSel);
    }

    // HPS confusion matrix: use actual hpsPFTau decayMode() vs gen DM
    if (haveHpsTaus && dm_reco_vs_gen_hps_ && pfJets.isValid()) {
      std::unordered_map<size_t, std::vector<size_t>> tausPerJet;
      for (size_t t = 0; t < taus->size(); ++t) {
        auto jr = (*taus)[t].jetRef();
        if (!jr)
          continue;
        for (size_t j = 0; j < pfJets->size(); ++j) {
          if (jr.get() == &(*pfJets)[j]) {
            tausPerJet[j].push_back(t);
            break;
          }
        }
      }

      std::vector<size_t> candTauIdxs;

      // Prefer taus on jet endpoint (if it exists)
      if (hasUniqueJetEndpoint) {
        auto it = tausPerJet.find(jetEndpoint);
        if (it != tausPerJet.end())
          candTauIdxs = it->second;
      }

      // If none, fall back to taus on any jet containing >=1 PF leg of this link
      if (candTauIdxs.empty()) {
        std::unordered_set<size_t> jetsWithAnyLeg;
        for (const auto& kv : pendingHad)
          if (kv.second.stepPass[4])
            jetsWithAnyLeg.insert(kv.second.jets.begin(), kv.second.jets.end());
        for (const auto& kv : pendingGamma)
          if (kv.second.stepPass[4])
            jetsWithAnyLeg.insert(kv.second.jets.begin(), kv.second.jets.end());

        for (auto j : jetsWithAnyLeg) {
          auto it = tausPerJet.find(j);
          if (it != tausPerJet.end())
            candTauIdxs.insert(candTauIdxs.end(), it->second.begin(), it->second.end());
        }
      }

      if (!candTauIdxs.empty()) {
        auto overlapCountForTau = [&](size_t tIdx)->int {
          int cnt = 0;
          for (const auto& kv : pendingHad) {
            auto it = kv.second.tauRegion.find(tIdx);
            if (it != kv.second.tauRegion.end() &&
                (it->second.signal || it->second.isolation)) ++cnt;
          }
          for (const auto& kv : pendingGamma) {
            auto it = kv.second.tauRegion.find(tIdx);
            if (it != kv.second.tauRegion.end() &&
                (it->second.signal || it->second.isolation)) ++cnt;
          }
          return cnt;
        };

        size_t bestTauIdx = candTauIdxs.front();
        int    bestOverlap = overlapCountForTau(bestTauIdx);
        for (size_t t : candTauIdxs) {
          int ov = overlapCountForTau(t);
          if (ov > bestOverlap) {
            bestOverlap = ov;
            bestTauIdx  = t;
          }
        }
        bestTauIdxForLink = static_cast<int>(bestTauIdx);

        int hpsDM = (*taus)[bestTauIdx].decayMode();
        int rSel  = dmToSelIndex(hpsDM);
        int gSel  = dmGenIdx;
        if (rSel >= 0 && gSel >= 0)
          dm_reco_vs_gen_hps_->Fill(rSel, gSel);
      }
    }

    // generic helper to fill per-leg step histos for any chain type
    auto fillStepHists = [](auto& hists, int nSteps, float pt, float eta, const auto& pass) {
      for (int s = 0; s < nSteps; ++s) {
        if (auto* h = hists.den_pt[s])  h->Fill(pt);
        if (auto* h = hists.den_eta[s]) h->Fill(eta);
        if (pass[s]) {
          if (auto* h = hists.num_pt[s])  h->Fill(pt);
          if (auto* h = hists.num_eta[s]) h->Fill(eta);
        }
      }
    };

    // pT-sorted charged legs
    {
      std::vector<std::pair<unsigned, float>> chList;
      chList.reserve(chKinTicl.size());
      for (auto const& kv : chKinTicl)
        chList.emplace_back(kv.first, kv.second.pt);
      std::sort(chList.begin(), chList.end(),
                [](auto const& a, auto const& b){ return a.second > b.second; });

      const int chMaxLegs = std::min(expectedChForDM(dmPhys), kMaxCHLegs);
      int li = 0;
      for (auto const& kv : chList) {
        if (li >= chMaxLegs)
          break;
        unsigned key = kv.first;
        auto it = pendingHad.find(key);
        if (it == pendingHad.end())
          continue;
        const auto& pend = it->second;
        if (!pend.hasCPKinematics)
          continue;
        fillStepHists(chStepHists_[dmGenIdx][li], kNSteps, pend.cpPt, pend.cpEta, pend.stepPass);
        fillStepHists(chTrackStepHists_[dmGenIdx][li], kNTrackSteps, pend.cpPt, pend.cpEta, pend.trackStepPass);
        fillStepHists(chCombiStepHists_[dmGenIdx][li], kNCombiSteps, pend.cpPt, pend.cpEta, pend.combiStepPass);
        ++li;
      }
    }

    // pT-sorted photon legs
    {
      std::vector<std::pair<unsigned, float>> phoList;
      phoList.reserve(phoKinTicl.size());
      for (auto const& kv : phoKinTicl)
        phoList.emplace_back(kv.first, kv.second.pt);
      std::sort(phoList.begin(), phoList.end(),
                [](auto const& a, auto const& b){ return a.second > b.second; });

      const int gammaMaxLegs = std::min(2 * expectedPi0ForDM(dmPhys), kMaxGammaLegs);
      int li = 0;
      for (auto const& kv : phoList) {
        if (li >= gammaMaxLegs)
          break;
        unsigned key = kv.first;
        auto it = pendingGamma.find(key);
        if (it == pendingGamma.end())
          continue;
        const auto& pend = it->second;
        if (!pend.hasCPKinematics)
          continue;
        fillStepHists(gammaStepHists_[dmGenIdx][li], kNSteps, pend.cpPt, pend.cpEta, pend.stepPass);
        ++li;
      }
    }

    // gen-tau for tau-level plots
    double tauPt = 0., tauEta = 0.;
    const reco::GenParticle* bestMotherTau = nullptr;
    double bestMotherPt = -1.;
    unsigned int bestMotherIdx = static_cast<unsigned int>(-1);
    if (genParticles.isValid()) {
      for (const auto& leaf : link.leaves) {
        const int genIdx = leaf.gen_particle_idx();
        if (genIdx < 0 || genIdx >= (int)genParticles->size())
          continue;
        const auto* leafGen = &(*genParticles)[genIdx];

        const reco::GenParticle* cur = leafGen;
        while (cur && cur->numberOfMothers() > 0) {
          const auto* mom = dynamic_cast<const reco::GenParticle*>(cur->mother(0));
          if (!mom)
            break;
          if (std::abs(mom->pdgId()) == 15) {
            if (!bestMotherTau ||
                (mom->statusFlags().isLastCopy() && !bestMotherTau->statusFlags().isLastCopy()) ||
                (mom->statusFlags().isLastCopy() == bestMotherTau->statusFlags().isLastCopy() &&
                 mom->pt() > bestMotherPt)) {
              bestMotherTau = mom;
              bestMotherPt  = mom->pt();
              bestMotherIdx = cur->motherRef().key();
            }
            break;
          }
          cur = mom;
        }
      }
    }
    if (bestMotherTau) {
      // Fallback to mother tau kinematics to avoid default zero values.
      tauPt  = bestMotherTau->pt();
      tauEta = bestMotherTau->eta();
      bool foundVisTau = false;
      if (genVisTaus.isValid()) {
        for (const auto& genVisTau : *genVisTaus) {
          if (genVisTau.motherRef().isNonnull() &&
              genVisTau.motherRef().key() == bestMotherIdx) {
            tauPt  = genVisTau.pt();
            tauEta = genVisTau.eta();
            foundVisTau = true;
            break;
          }
        }
        if (!foundVisTau) {
          edm::LogWarning("TICLTauValidator")
            << "No genVisTau match for bestMotherIdx=" << bestMotherIdx
            << " (dm=" << dmPhys << ", tau pt=" << tauPt
            << ", eta=" << tauEta << ")";
        }
      } else {
        edm::LogWarning("TICLTauValidator")
          << "genVisTaus collection missing/invalid; using mother tau kinematics"
          << " (dm=" << dmPhys << ", tau pt=" << tauPt
          << ", eta=" << tauEta << ")";
      }
    }

    const int expCh  = expectedChForDM(dmPhys);
    const int expPi0 = expectedPi0ForDM(dmPhys);

    const bool genInAcceptance = bestMotherTau && std::abs(tauEta) > hgcalEtaAbsMin_;

    // Fill tau-level denominators for all gen-visible taus in geometric acceptance
    if (genInAcceptance) {
      if (tau_gen_pt_[dmGenIdx])  tau_gen_pt_[dmGenIdx]->Fill(tauPt);
      if (tau_gen_eta_[dmGenIdx]) tau_gen_eta_[dmGenIdx]->Fill(tauEta);
    }

    // tau-level numerators
    if (genInAcceptance) {
      const int chCap  = chCapForDM(dmPhys);
      const int p0Cap  = pi0CapForDM(dmPhys);

      const int nGoodCH_endpoint = haveHpsTaus ? nGoodCH_tau_total : nGoodCH_jet;
      const int nPi0_endpoint    = haveHpsTaus ? nPi0_tau_total    : nPi0_jet;

      // >= N charged legs at reco
      for (int N = 1; N <= chCap; ++N) {
        if (nGoodCH_endpoint >= N) {
          if (auto* h = tau_gen_matched_to_nCh_pt_[dmGenIdx][N-1])  h->Fill(tauPt);
          if (auto* h = tau_gen_matched_to_nCh_eta_[dmGenIdx][N-1]) h->Fill(tauEta);
        }
      }

      // >= N pi0 at reco
      for (int N = 1; N <= p0Cap; ++N) {
        if (nPi0_endpoint >= N) {
          if (auto* h = tau_gen_matched_to_nPi0_pt_[dmGenIdx][N-1])  h->Fill(tauPt);
          if (auto* h = tau_gen_matched_to_nPi0_eta_[dmGenIdx][N-1]) h->Fill(tauEta);
        }
      }

      // ALL expected legs
      if (expCh > 0 || expPi0 > 0) {
        if (nGoodCH_endpoint >= expCh && nPi0_endpoint >= expPi0) {
          if (tau_gen_matched_to_all_pt_[dmGenIdx])  tau_gen_matched_to_all_pt_[dmGenIdx]->Fill(tauPt);
          if (tau_gen_matched_to_all_eta_[dmGenIdx]) tau_gen_matched_to_all_eta_[dmGenIdx]->Fill(tauEta);
        }
      }

      // Tau-level two-fold: >= N charged track/calo/both/either
      for (int N = 1; N <= chCap; ++N) {
        if (nCh_track >= N) {
          if (auto* h = tau_nCh_track_pt_[dmGenIdx][N-1])  h->Fill(tauPt);
          if (auto* h = tau_nCh_track_eta_[dmGenIdx][N-1]) h->Fill(tauEta);
        }
        if (nCh_calo >= N) {
          if (auto* h = tau_nCh_calo_pt_[dmGenIdx][N-1])  h->Fill(tauPt);
          if (auto* h = tau_nCh_calo_eta_[dmGenIdx][N-1]) h->Fill(tauEta);
        }
        if (nCh_both >= N) {
          if (auto* h = tau_nCh_both_pt_[dmGenIdx][N-1])  h->Fill(tauPt);
          if (auto* h = tau_nCh_both_eta_[dmGenIdx][N-1]) h->Fill(tauEta);
        }
        if (nCh_either >= N) {
          if (auto* h = tau_nCh_either_pt_[dmGenIdx][N-1])  h->Fill(tauPt);
          if (auto* h = tau_nCh_either_eta_[dmGenIdx][N-1]) h->Fill(tauEta);
        }
      }

      // signal vs iso split (tau endpoint only)
      if (haveHpsTaus) {
        const int nGoodCH_signal = nGoodCH_signal_tau;
        const int nGoodCH_iso    = nGoodCH_iso_tau;
        const int nPi0_signal    = nPi0_signal_tau;
        const int nPi0_iso       = nPi0_iso_tau;

        for (int N = 1; N <= chCap; ++N) {
          if (nGoodCH_signal >= N) {
            if (auto* h = tau_gen_matched_to_nCh_sig_pt_[dmGenIdx][N-1])  h->Fill(tauPt);
            if (auto* h = tau_gen_matched_to_nCh_sig_eta_[dmGenIdx][N-1]) h->Fill(tauEta);
          }
          if (nGoodCH_iso >= N) {
            if (auto* h = tau_gen_matched_to_nCh_iso_pt_[dmGenIdx][N-1])  h->Fill(tauPt);
            if (auto* h = tau_gen_matched_to_nCh_iso_eta_[dmGenIdx][N-1]) h->Fill(tauEta);
          }
        }
        for (int N = 1; N <= p0Cap; ++N) {
          if (nPi0_signal >= N) {
            if (auto* h = tau_gen_matched_to_nPi0_sig_pt_[dmGenIdx][N-1])  h->Fill(tauPt);
            if (auto* h = tau_gen_matched_to_nPi0_sig_eta_[dmGenIdx][N-1]) h->Fill(tauEta);
          }
          if (nPi0_iso >= N) {
            if (auto* h = tau_gen_matched_to_nPi0_iso_pt_[dmGenIdx][N-1])  h->Fill(tauPt);
            if (auto* h = tau_gen_matched_to_nPi0_iso_eta_[dmGenIdx][N-1]) h->Fill(tauEta);
          }
        }
        // tau resolution
        if (tauPt > 0.) {
          if (bestTauIdxForLink >= 0 && taus.isValid()) {
            const auto& tau = (*taus)[bestTauIdxForLink];
            const double respTau = tau.pt() / tauPt;  // reco / gen
            if (tau_pt_reco_over_gen_[dmGenIdx]) {
              tau_pt_reco_over_gen_[dmGenIdx]->Fill(respTau);
            }
            // reco tau shapes per DM
            if (tau_reco_pt_[dmGenIdx])  tau_reco_pt_[dmGenIdx]->Fill(tau.pt());
            if (tau_reco_eta_[dmGenIdx]) tau_reco_eta_[dmGenIdx]->Fill(tau.eta());
          }
        }
      }
    } 

    // --- per-link summary ---
    std::ostringstream ss;
    ss << "[TICLTauValidator] link summary: "
        << "DM=" << dmPhys
        << " nTiclCh=" << pendingHad.size()
        << " nTiclPho=" << pendingGamma.size()
        << " hasUniqueJetEndpoint=" << (hasUniqueJetEndpoint ? "yes" : "no")
        << " jetEndpoint=" << (hasUniqueJetEndpoint ? static_cast<int>(jetEndpoint) : -1)
        << " nGoodCH_jet=" << nGoodCH_jet
        << " nPi0_jet=" << nPi0_jet
        << " haveHpsTaus=" << (haveHpsTaus ? "yes" : "no")
        << " nGoodCH_tau=" << nGoodCH_tau_total
        << " nPi0_tau=" << nPi0_tau_total
        << " endpoint=" << (haveHpsTaus ? "TAU" : "JET");
    edm::LogVerbatim("TICLTauValidator") << ss.str();

} // links

  // === FAKE RATE: loop reco taus, reverse the TICL association chain ===
  // Build set of CaloParticle keys belonging to any simulated hadronic tau
  std::unordered_set<unsigned int> tauCPKeys;
  for (const auto& link : *simTaus) {
    if (!isHadronicDM(link.decayMode))
      continue;
    for (const auto& leaf : link.leaves) {
      const int cpId = leaf.calo_particle_idx();
      if (cpId < 0 || static_cast<size_t>(cpId) >= link.calo_particle_leaves.size())
        continue;
      const auto& cpRef = link.calo_particle_leaves[cpId];
      if (cpRef.isNonnull())
        tauCPKeys.insert(cpRef.key());
    }
  }

  // Get CaloParticle pointer for barrel TP matching in countAssociatedSignalPFCands
  const std::vector<CaloParticle>* cpPtr = nullptr;
  if (!caloParticlesToken_.isUninitialized() && caloParticles.isValid())
    cpPtr = &(*caloParticles);

  const auto processFakeRateTau = [&](const reco::PFTau& tau,
                                       FakeRateHists& fr,
                                       FakeRateHists& frCalo,
                                       FakeRateHists& frTrack) {
    if (std::abs(tau.eta()) < hgcalEtaAbsMin_)
      return;
    if (tau.signalPFCands().empty())
      return;

    const int hpsDM  = tau.decayMode();
    const int dmSelI = dmToSelIndex(hpsDM);

    int nChSig = 0, nPhoSig = 0;
    for (const auto& pfPtr2 : tau.signalPFCands()) {
      if (!pfPtr2.isNonnull()) continue;
      const int absPdg = std::abs(pfPtr2->pdgId());
      if (absPdg == 211)      ++nChSig;
      else if (absPdg == 22)  ++nPhoSig;
    }
    const int nPi0Sig = nPhoSig / 2;
    const int nChFill = std::min(nChSig, kMaxCHLegs);
    const int nPi0Fill = std::min(nPi0Sig, kMaxPi0Legs);

    auto assocCounts = countAssociatedSignalPFCands(tau, barrelSize, tauCPKeys,
                    *ticlCandidates, *recoToSimMap, *simTracksters,
                    cpPtr,
                    trackRecoToSim.isValid() ? &(*trackRecoToSim) : nullptr);
    const bool isGenuine      = (assocCounts.nAssocAllParticles > 0);
    const bool isGenuineCalo  = (assocCounts.nAssocCalo > 0);
    const bool isGenuineTrack = (assocCounts.nAssocTrack > 0);
    fillFakeRateHists(tau, dmSelI, nChFill, nPi0Fill, isGenuine, fr);
    fillFakeRateHists(tau, dmSelI, nChFill, nPi0Fill, isGenuineCalo, frCalo);
    fillFakeRateHists(tau, dmSelI, nChFill, nPi0Fill, isGenuineTrack, frTrack);
  };

  if (taus.isValid() && pfJets.isValid() && pfMerged.isValid() &&
      ticlCandidates.isValid() && recoToSimMap.isValid() && simTracksters.isValid()) {
    for (size_t t = 0; t < taus->size(); ++t)
      processFakeRateTau((*taus)[t], fakeRate_, fakeRateCalo_, fakeRateTrack_);
  } else {
    edm::LogWarning("TICLTauValidator") << "Fake rate loop skipped: "
                                        << "taus=" << (taus.isValid() ? "ok" : "invalid")
                                        << " pfJets=" << (pfJets.isValid() ? "ok" : "invalid")
                                        << " pfMerged=" << (pfMerged.isValid() ? "ok" : "invalid")
                                        << " ticlCandidates=" << (ticlCandidates.isValid() ? "ok" : "invalid")
                                        << " recoToSimMap=" << (recoToSimMap.isValid() ? "ok" : "invalid")
                                        << " simTracksters=" << (simTracksters.isValid() ? "ok" : "invalid");
  }

  if (!finalFilterTauRefs.empty() && pfJets.isValid() && pfMerged.isValid() &&
      ticlCandidates.isValid() && recoToSimMap.isValid() && simTracksters.isValid()) {
    for (const auto& tauRef : finalFilterTauRefs) {
      if (tauRef.isNonnull())
        processFakeRateTau(*tauRef, fakeRateFilt_, fakeRateCaloFilt_, fakeRateTrackFilt_);
    }
  }

  edm::LogVerbatim("TICLTauValidator")
    << "[TICLTauValidator] processed event " << iEvent.id();
  for (const auto& s : eventChains) {
    edm::LogVerbatim("TICLTauValidator") << "  " << s;
  }
}

void TICLTauValidator::fillFakeRateHists(const reco::PFTau& tau,
                                         int dmSelI,
                                         int nChFill,
                                         int nPi0Fill,
                                         bool isGenuine,
                                         FakeRateHists& fr) {
  auto fillPtEta = [&](MonitorElement* hpt, MonitorElement* heta) {
    if (hpt)  hpt->Fill(tau.pt());
    if (heta) heta->Fill(tau.eta());
  };

  fillPtEta(fr.den_pt, fr.den_eta);
  if (dmSelI >= 0)
    fillPtEta(fr.den_dm_pt[dmSelI], fr.den_dm_eta[dmSelI]);

  for (int N = 1; N <= nChFill; ++N) {
    fillPtEta(fr.den_sig_nCh_pt[N-1], fr.den_sig_nCh_eta[N-1]);
    if (dmSelI >= 0)
      fillPtEta(fr.den_dm_sig_nCh_pt[dmSelI][N-1], fr.den_dm_sig_nCh_eta[dmSelI][N-1]);
  }
  for (int N = 1; N <= nPi0Fill; ++N) {
    fillPtEta(fr.den_sig_nPi0_pt[N-1], fr.den_sig_nPi0_eta[N-1]);
    if (dmSelI >= 0)
      fillPtEta(fr.den_dm_sig_nPi0_pt[dmSelI][N-1], fr.den_dm_sig_nPi0_eta[dmSelI][N-1]);
  }

  if (isGenuine)
    return;

  fillPtEta(fr.num_pt, fr.num_eta);
  if (dmSelI >= 0)
    fillPtEta(fr.num_dm_pt[dmSelI], fr.num_dm_eta[dmSelI]);

  for (int N = 1; N <= nChFill; ++N) {
    fillPtEta(fr.num_sig_nCh_pt[N-1], fr.num_sig_nCh_eta[N-1]);
    if (dmSelI >= 0)
      fillPtEta(fr.num_dm_sig_nCh_pt[dmSelI][N-1], fr.num_dm_sig_nCh_eta[dmSelI][N-1]);
  }
  for (int N = 1; N <= nPi0Fill; ++N) {
    fillPtEta(fr.num_sig_nPi0_pt[N-1], fr.num_sig_nPi0_eta[N-1]);
    if (dmSelI >= 0)
      fillPtEta(fr.num_dm_sig_nPi0_pt[dmSelI][N-1], fr.num_dm_sig_nPi0_eta[dmSelI][N-1]);
  }
}

TICLTauValidator::AssocCounts TICLTauValidator::countAssociatedSignalPFCands(
    const reco::PFTau& tau,
    size_t barrelSize,
    const std::unordered_set<unsigned int>& tauCPKeys,
    const std::vector<TICLCandidate>& ticlCandidates,
    const TracksterToTracksterMap& recoToSimMap,
    const std::vector<ticl::Trackster>& simTracksters,
    const std::vector<CaloParticle>* caloParticles,
    const reco::RecoToSimCollection* trackRecoToSim) const {
  AssocCounts counts;

  for (const auto& pfPtr : tau.signalPFCands()) {
    if (!pfPtr.isNonnull())
      continue;

    const int absPdg = std::abs(pfPtr->pdgId());
    const bool isCharged    = (absPdg == 211);
    const bool isPhoton     = (absPdg == 22);
    const bool isElectron   = (absPdg == 11);
    const bool isNeutralHad = (absPdg == 130);

    if (!isCharged && !isPhoton && !isElectron && !isNeutralHad)
      continue;

    if (isCharged || isNeutralHad)
      ++counts.nSigCh;
    if (isPhoton || isElectron)
      ++counts.nSigPho;

    const size_t pfKey = pfPtr.key();
    bool trackMatched = false;
    bool caloMatched = false;

    // Track matching: try for ANY PF candidate with a valid trackRef
    if (trackRecoToSim && caloParticles) {
      const auto trackRef = pfPtr->trackRef();
      if (trackRef.isNonnull()) {
        auto found = trackRecoToSim->find(edm::RefToBase<reco::Track>(trackRef));
        if (found != trackRecoToSim->end()) {
          for (const auto& tpMatch : found->val) {
            const auto& tpRef = tpMatch.first;
            for (const auto& tpG4 : tpRef->g4Tracks()) {
              for (const auto& cpKeyVal : tauCPKeys) {
                if (cpKeyVal >= caloParticles->size()) continue;
                const auto& cp = (*caloParticles)[cpKeyVal];
                for (const auto& cpG4 : cp.g4Tracks()) {
                  if (tpG4.trackId() == cpG4.trackId() && tpG4.eventId().event() == 0 && cpG4.eventId().event() == 0)
                    trackMatched = true;
                }
              }
            }
            if (trackMatched) break;
          }
        }
      }
    }

    // Calo (TICL) matching: endcap PF candidates only
    if (pfKey >= barrelSize) {
      const size_t ticlIdx = pfKey - barrelSize;
      if (ticlIdx < ticlCandidates.size()) {
        const auto& cand = ticlCandidates[ticlIdx];
        for (const auto& tsPtr : cand.tracksters()) {
          if (!tsPtr.isNonnull())
            continue;
          const size_t recoTkIdx = tsPtr.key();
          if (recoTkIdx >= recoToSimMap.size())
            continue;

          for (const auto& m : recoToSimMap[recoTkIdx]) {
            const double score = m.score();
            const size_t simTkIdx = m.index();
            if (simTkIdx >= simTracksters.size())
              continue;

            const auto& simTk = simTracksters[simTkIdx];
            const int seedIdx = simTk.seedIndex();
            if (seedIdx < 0 || !tauCPKeys.count(static_cast<unsigned int>(seedIdx)))
              continue;

            if (score <= maxAssocScore_) {
              caloMatched = true;
              break;
            }
          }
          if (caloMatched)
            break;
        }
      }
    }

    if (trackMatched) ++counts.nAssocTrack;
    if (caloMatched) ++counts.nAssocCalo;
    if (trackMatched || caloMatched) ++counts.nAssocAllParticles;
  }

  return counts;
}

void TICLTauValidator::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<std::string>("folder", "RecoTauV/ticlTauValidator");

  desc.add<edm::InputTag>("simTaus", edm::InputTag("SimTauProducer"));
  desc.add<edm::InputTag>("caloParticles", edm::InputTag("mix", "MergedCaloTruth"));
  desc.add<edm::InputTag>("TauProducer", edm::InputTag("hpsPFTauProducer"));
  desc.add<edm::InputTag>("pf", edm::InputTag("particleFlow"));
  desc.add<edm::InputTag>("pfTmpBarrel", edm::InputTag("particleFlowTmpBarrel"));
  desc.add<edm::InputTag>("jets", edm::InputTag("ak4PFJets"));
  desc.add<edm::InputTag>("ticlCandidates", edm::InputTag("ticlCandidate"));
  desc.add<edm::InputTag>("simTracksters", edm::InputTag("ticlSimTracksters", "fromCPs"));
  desc.add<edm::InputTag>("simToRecoTracksterAssocByLCs",
                          edm::InputTag("allTrackstersToSimTrackstersAssociationsByLCs",
                                        "ticlSimTrackstersfromCPsToticlCandidate"));
  desc.add<edm::InputTag>("recoToSimTracksterAssocByLCs",
                          edm::InputTag("allTrackstersToSimTrackstersAssociationsByLCs",
                                        "ticlCandidateToticlSimTrackstersfromCPs"));
  desc.add<edm::InputTag>("genParticles", edm::InputTag("genParticles"));
  desc.add<edm::InputTag>("genVisTaus", edm::InputTag("genVisTaus"));
  desc.add<edm::InputTag>("trackingParticles", edm::InputTag("mix", "MergedTrackTruth"));
  desc.add<edm::InputTag>("trackRecoToSim", edm::InputTag("tpToHltGeneralTrackAssociation"));
  desc.add<double>("maxAssocScore", 0.6);
  desc.add<double>("hgcalEtaAbsMin", 1.5);


  desc.add<std::string>("hltProcessName", "HLTX");
  desc.add<std::vector<std::string>>(
    "hltTauFilterLabels",
    {
      "hltHpsDoublePFTau40TrackPt1MediumChargedIsolation"
    });


  descriptions.add("ticlTauValidator", desc);
}

DEFINE_FWK_MODULE(TICLTauValidator);
