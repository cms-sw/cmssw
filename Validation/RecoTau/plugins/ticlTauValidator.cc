// -*- C++ -*-
//
// Package:    Validation/TICLTauValidator
// Class:      TICLTauValidator 
//
/*

 Description: Validation for tau reconstruction using TICL and tracks

 Implementation:

   Calo Steps (0..5):
     0: CaloParticle - sim trackster(s) of its sim TICL candidate
     1: merged (fromCPs) sim trackster - best reco trackster (lowest score, gated by maxAssocScore)
     2: best reco trackster - THE reco TICLCandidate containing it
     3: reco TICLCandidate - PF candidate (merged PF)
     4: PF cand - PFJet
     5: PFJet - PFTau usage (skipped if no taus)

    Tracking steps (0..4):
     0: CP - sim TICL candidate
     1: sim TICL candidate - reco Track(s) (attached at production time, quality-gated)
     2: reco Track - PF candidate
     3: PF cand - PFJet
     4: PFJet - PFTau usage

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
#include <map>
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
#include "DataFormats/HGCalReco/interface/Trackster.h"
#include "DataFormats/HGCalReco/interface/TICLCandidate.h"
#include "DataFormats/Common/interface/Ptr.h"
#include "DataFormats/JetReco/interface/PFJet.h"
#include "DataFormats/JetReco/interface/PFJetCollection.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"

#include "SimDataFormats/CaloAnalysis/interface/SimTauCPLink.h"
#include "SimDataFormats/CaloAnalysis/interface/CaloParticle.h"
#include "SimDataFormats/Associations/interface/TICLAssociationMap.h"

class TICLTauValidator : public DQMEDAnalyzer {
public:
  explicit TICLTauValidator(const edm::ParameterSet&);
  ~TICLTauValidator() override {}
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  using TracksterToTracksterMap = ticl::AssociationMap<ticl::mapWithSharedEnergyAndScore,
                                                       std::vector<ticl::Trackster>,
                                                       std::vector<ticl::Trackster>>;
  using ProductKey = std::pair<edm::ProductID, size_t>;

  void bookHistograms(DQMStore::IBooker&, edm::Run const&, edm::EventSetup const&) override;
  void analyze(const edm::Event&, const edm::EventSetup&) override;
  
  struct AssocCounts {
    int nSigCh = 0;            // Signal charged PF cands
    int nSigPho = 0;           // Signal photon + electron PF cands
    int nAssocCalo = 0;         // Matched via TICL chain (endcap PF candidates)
    int nAssocTrack = 0;        // PF track is a sim-candidate track of a tau CP
    int nAssocAllParticles = 0; // Either path (calo OR track)
  };

  AssocCounts countAssociatedSignalPFCands(const reco::PFTau& tau,
                                           size_t barrelSize,
                                           const std::unordered_set<size_t>& tauSimTracksterIdxs,
                                           const std::set<ProductKey>& tauSimCandTrackKeys,
                                           const std::vector<TICLCandidate>& ticlCandidates,
                                           const TracksterToTracksterMap& recoToSimMap) const;

  std::string folder_;
  double maxAssocScore_;  // smaller = better association
  double hgcalEtaAbsMin_;

  edm::EDGetTokenT<std::vector<SimTauCPLink>> simTauToken_;
  edm::EDGetTokenT<reco::PFTauCollection>     tauProducerToken_;

  edm::EDGetTokenT<reco::PFCandidateCollection> pfToken_;
  edm::EDGetTokenT<reco::PFCandidateCollection> pfTmpBarrelToken_;

  edm::EDGetTokenT<std::vector<TICLCandidate>>       ticlCandidatesToken_;
  edm::EDGetTokenT<std::vector<TICLCandidate>>       simTICLCandidatesToken_;
  edm::EDGetTokenT<std::vector<ticl::Trackster>>     simTrackstersFromCPToken_;
  edm::EDGetTokenT<TracksterToTracksterMap>          allTrkToSimTrkAssocByLCsToken_;
  edm::EDGetTokenT<reco::PFJetCollection>            pfJetsToken_;
  edm::EDGetTokenT<reco::GenParticleCollection>      genParticlesToken_;
  edm::EDGetTokenT<reco::GenParticleCollection> genVisTausToken_;
  edm::EDGetTokenT<TracksterToTracksterMap>      recoToSimAssocByLCsToken_;

  // ---------- constants & helpers ----------
  static constexpr int   kMaxCHLegs    = 3; // charged hadrons per tau
  static constexpr int   kMaxGammaLegs = 4; // truth photons from pi0 decays
  static constexpr int   kMaxPi0Legs   = 2; // truth pi0 equivalents
  // Calo chain steps: (0..5) 
  static constexpr int   kNSteps       = 6; 
  // Track chain: (0..4)
  static constexpr int   kNTrackSteps  = 5;
  // Both-path chain: the same truth CP must be represented through calo AND track paths
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

  // Expectations per DM (gen DMs only; reco-only DM 5 never reaches these)
  static inline int expectedChForDM(int dm) {
    switch (dm) {
      case 0:
      case 1:
      case 2:  return 1;
      case 10:
      case 11: return 3;
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
      default: return 0;
    }
  }

  static inline int chCapForDM(int dm)  { return std::min(expectedChForDM(dm),  kMaxCHLegs); }
  static inline int gammaCapForDM(int dm) { return std::min(2 * expectedPi0ForDM(dm), kMaxGammaLegs); }
  static inline int pi0CapForDM(int dm) { return std::min(expectedPi0ForDM(dm), kMaxPi0Legs); }

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
  };

  void fillFakeRateHists(const reco::PFTau& tau,
                         int dmSelI,
                         bool isGenuine,
                         FakeRateHists& fr);

  // ---------- per-event working structures ----------
  struct TauRegionFlags { bool signal = false, isolation = false; };

  // Per-CP association state, filled while tracing the chains.
  struct PendingInfo {
    float cpPt = 0.f, cpEta = 0.f;
    float pfPt = 0.f;
    bool hasCPKinematics = false;
    bool hasPFKinematics = false;
    bool trackMatched = false;
    std::array<bool, kNSteps> stepPass{};            // calo chain steps
    std::array<bool, kNTrackSteps> trackStepPass{};  // track chain steps
    std::array<bool, kNCombiSteps> combiStepPass{};  // truth CP represented through both paths
    std::set<size_t> jets;         // jets containing calo-chain PF candidates
    std::set<size_t> trackJets;    // jets containing track-chain PF candidates
    std::set<size_t> caloPFKeys;   // PF keys matched via calo chain
    std::set<size_t> trackPFKeys;  // PF keys matched via track chain
    std::unordered_map<size_t, TauRegionFlags> tauRegion;
    std::unordered_map<size_t, TauRegionFlags> trackTauRegion;
  };
  using PendingMap = std::unordered_map<unsigned, PendingInfo>;

  // Event-level collections and lookup maps, built once per event.
  struct EventContext {
    const reco::PFTauCollection* taus = nullptr;
    const reco::PFJetCollection* jets = nullptr;
    const reco::PFCandidateCollection* pfMerged = nullptr;
    const std::vector<TICLCandidate>* ticlCandidates = nullptr;
    const std::vector<TICLCandidate>* simTICLCandidates = nullptr;
    const TracksterToTracksterMap* simToRecoMap = nullptr;
    const TracksterToTracksterMap* recoToSimMap = nullptr;
    const reco::GenParticleCollection* genParticles = nullptr;
    const reco::GenParticleCollection* genVisTaus = nullptr;
    edm::ProductID pfMergedId, jetsId;
    size_t barrelSize = 0;
    std::map<ProductKey, std::vector<size_t>> trackKeyToPFIdx;   // reco track -> merged PF indices
    std::map<ProductKey, std::vector<size_t>> pfKeyToJets;       // PF candidate -> jet indices
    std::vector<std::set<ProductKey>> tauSignalKeys;             // per tau: signal PF keys
    std::vector<std::set<ProductKey>> tauIsoKeys;                // per tau: isolation PF keys
    std::unordered_map<size_t, std::vector<size_t>> tausPerJet;  // jet key -> tau indices
    std::map<std::pair<edm::ProductID, unsigned int>, size_t> cpToSimIdx;  // CaloParticle -> fromCPs index
  };

  // Dedup sets for once-per-event CP histogram fills.
  struct SeenSets {
    std::set<unsigned int> charged, photon;
    std::set<unsigned int> twoFoldCharged, twoFoldPhoton;
  };

  // Per-link counts of truth legs recovered at the various endpoints.
  struct LegCounts {
    int chJet = 0, gammaJet = 0;
    int chTau = 0, chSig = 0, chIso = 0;
    int gammaTau = 0, gammaSig = 0, gammaIso = 0;
    int chTrack = 0, chCalo = 0, chBoth = 0, chEither = 0;
  };

  void buildRecoLookups(EventContext& ctx) const;
  void buildCpToSimIdx(const std::vector<ticl::Trackster>* simTrackstersFromCP, EventContext& ctx) const;
  void jetsForPFIdxs(const std::set<size_t>& pfIdxs, std::set<size_t>& jetsOut, const EventContext& ctx) const;
  void matchTauRegions(const std::set<size_t>& pfIdxs,
                       const std::set<size_t>& legJets,
                       std::unordered_map<size_t, TauRegionFlags>& regions,
                       const EventContext& ctx) const;
  void traceTrackChain(const std::vector<ProductKey>& simCandTrackKeys,
                       PendingInfo& pend,
                       const EventContext& ctx) const;
  int findBestTICLCandidateIdx(size_t simIdx, PendingInfo& pend, const EventContext& ctx) const;
  void traceCaloToTau(int candIdx, PendingInfo& pend, const EventContext& ctx) const;
  void processLeaf(const SimTauCPLink::DecayNav& leaf,
                   const SimTauCPLink& link,
                   int dmGenIdx,
                   const EventContext& ctx,
                   SeenSets& seen,
                   PendingMap& pendingHad,
                   PendingMap& pendingGamma);
  void processLink(const SimTauCPLink& link, const EventContext& ctx, SeenSets& seen);
  int selectBestTauIdx(const PendingMap& pendingHad,
                       const PendingMap& pendingGamma,
                       bool hasUniqueJetEndpoint,
                       size_t jetEndpoint,
                       const EventContext& ctx) const;
  bool genTauKinematics(const SimTauCPLink& link, const EventContext& ctx, double& tauPt, double& tauEta) const;
  void fillLegStepHists(int dmPhys, int dmGenIdx, const PendingMap& pendingHad, const PendingMap& pendingGamma);
  void fillTauLevelHists(int dmPhys,
                         int dmGenIdx,
                         double tauPt,
                         double tauEta,
                         bool havePFTaus,
                         int bestTauIdx,
                         const LegCounts& counts,
                         const EventContext& ctx);
  void processFakeRates(const std::vector<SimTauCPLink>& simTaus, const EventContext& ctx);
  static int coverageDMFromTruthCPs(int nCharged, int nPhotons);

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
  MonitorElement* tau_gen_matched_to_nPi0_pt_[kNDMGen][kMaxPi0Legs]   = {{}};
  MonitorElement* tau_gen_matched_to_nPi0_eta_[kNDMGen][kMaxPi0Legs]  = {{}};
  MonitorElement* tau_gen_matched_to_all_pt_[kNDMGen]  = {};
  MonitorElement* tau_gen_matched_to_all_eta_[kNDMGen] = {};

  // numerators split into signal and isolation (tau endpoint only, gen-level use kNDMGen)
  MonitorElement* tau_gen_matched_to_nCh_sig_pt_[kNDMGen][kMaxCHLegs]   = {{}};
  MonitorElement* tau_gen_matched_to_nCh_sig_eta_[kNDMGen][kMaxCHLegs]  = {{}};
  MonitorElement* tau_gen_matched_to_nCh_iso_pt_[kNDMGen][kMaxCHLegs]   = {{}};
  MonitorElement* tau_gen_matched_to_nCh_iso_eta_[kNDMGen][kMaxCHLegs]  = {{}};
  MonitorElement* tau_gen_matched_to_nPi0_sig_pt_[kNDMGen][kMaxPi0Legs]  = {{}};
  MonitorElement* tau_gen_matched_to_nPi0_sig_eta_[kNDMGen][kMaxPi0Legs] = {{}};
  MonitorElement* tau_gen_matched_to_nPi0_iso_pt_[kNDMGen][kMaxPi0Legs]  = {{}};
  MonitorElement* tau_gen_matched_to_nPi0_iso_eta_[kNDMGen][kMaxPi0Legs] = {{}};

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
  // trackOnly: CP's sim TICL candidate has an associated reco track
  MonitorElement* cp_chHad_trackOnly_pt_dm_[kNDMGen]  = {};
  MonitorElement* cp_chHad_trackOnly_eta_dm_[kNDMGen] = {};
  // caloOnly: CP reached merged PF via TICL chain
  MonitorElement* cp_chHad_caloOnly_pt_dm_[kNDMGen]  = {};
  MonitorElement* cp_chHad_caloOnly_eta_dm_[kNDMGen] = {};
  // trackAndCalo: both criteria satisfied
  MonitorElement* cp_chHad_trackAndCalo_pt_dm_[kNDMGen]  = {};
  MonitorElement* cp_chHad_trackAndCalo_eta_dm_[kNDMGen] = {};
  // Photon equivalents (photons have no track, so trackOnly is always empty;
  MonitorElement* cp_gamma_caloOnly_pt_dm_[kNDMGen]  = {};
  MonitorElement* cp_gamma_caloOnly_eta_dm_[kNDMGen] = {};

  // ---------- fake rate histograms ----------
  FakeRateHists fakeRate_;
  FakeRateHists fakeRateCalo_;
  FakeRateHists fakeRateTrack_;

};

TICLTauValidator::TICLTauValidator(const edm::ParameterSet& iConfig)
  : folder_( iConfig.getParameter<std::string>("folder") ),
    maxAssocScore_( iConfig.getParameter<double>("maxAssocScore") ),
    hgcalEtaAbsMin_( iConfig.getParameter<double>("hgcalEtaAbsMin") )
{
  simTauToken_        = consumes<std::vector<SimTauCPLink>>( iConfig.getParameter<edm::InputTag>("simTaus") );
  tauProducerToken_   = consumes<reco::PFTauCollection>(      iConfig.getParameter<edm::InputTag>("TauProducer") );

  pfToken_          = consumes<reco::PFCandidateCollection>(  iConfig.getParameter<edm::InputTag>("pf") );
  pfTmpBarrelToken_ = consumes<reco::PFCandidateCollection>(  iConfig.getParameter<edm::InputTag>("pfTmpBarrel") );

  ticlCandidatesToken_ = consumes<std::vector<TICLCandidate>>(   iConfig.getParameter<edm::InputTag>("ticlCandidates") );
  simTICLCandidatesToken_ = consumes<std::vector<TICLCandidate>>(iConfig.getParameter<edm::InputTag>("simTICLCandidates") );
  simTrackstersFromCPToken_ = consumes<std::vector<ticl::Trackster>>(iConfig.getParameter<edm::InputTag>("simTracksters") );
  allTrkToSimTrkAssocByLCsToken_ = consumes<TracksterToTracksterMap>(
    iConfig.getParameter<edm::InputTag>("simToRecoTracksterAssocByLCs") );
  pfJetsToken_       = consumes<reco::PFJetCollection>(        iConfig.getParameter<edm::InputTag>("jets") );
  genParticlesToken_ = consumes<reco::GenParticleCollection>(  iConfig.getParameter<edm::InputTag>("genParticles") );
  genVisTausToken_   = consumes<reco::GenParticleCollection>(    iConfig.getParameter<edm::InputTag>("genVisTaus") );
  recoToSimAssocByLCsToken_ = consumes<TracksterToTracksterMap>(
    iConfig.getParameter<edm::InputTag>("recoToSimTracksterAssocByLCs") );
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

  // per-DM, per-leg step histograms
  for (int dmI = 0; dmI < kNDMGen; ++dmI) {
    int dm = kDMGen[dmI];
    ibook.setCurrentFolder(folder_ + "/GenDM" + std::to_string(dm));

    const int chCap    = chCapForDM(dm);
    const int gammaCap = gammaCapForDM(dm);

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
      for (int s = 0; s < kNSteps; ++s) {
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
      for (int s = 0; s < kNSteps; ++s) {
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

    for (int i = 0; i < kNDMSel; ++i) {
      int dm = kDMSel[i];
      std::string ds = std::to_string(dm);
      ibook.setCurrentFolder(folder_ + "/GenDM" + ds + "/FakeRate");
      bookStepDenNum(fr.den_dm_pt[i], fr.den_dm_eta[i], fr.num_dm_pt[i], fr.num_dm_eta[i],
                     prefix + "_dm" + ds, tag + " (DM=" + ds + ")");
    }
  };

  bookFakeRateSet("fake", "", fakeRate_);
  bookFakeRateSet("fake_calo", "calo assoc", fakeRateCalo_);
  bookFakeRateSet("fake_track", "track assoc", fakeRateTrack_);
}

void TICLTauValidator::analyze(const edm::Event& iEvent,
                               const edm::EventSetup&) {
  edm::Handle<std::vector<SimTauCPLink>> simTaus;            iEvent.getByToken(simTauToken_, simTaus);
  edm::Handle<reco::PFTauCollection>     taus;               iEvent.getByToken(tauProducerToken_, taus);
  edm::Handle<TracksterToTracksterMap>   simToRecoMap;       iEvent.getByToken(allTrkToSimTrkAssocByLCsToken_, simToRecoMap);
  edm::Handle<TracksterToTracksterMap>   recoToSimMap;       iEvent.getByToken(recoToSimAssocByLCsToken_, recoToSimMap);
  edm::Handle<std::vector<TICLCandidate>> ticlCandidates;    iEvent.getByToken(ticlCandidatesToken_, ticlCandidates);
  edm::Handle<std::vector<TICLCandidate>> simTICLCandidates; iEvent.getByToken(simTICLCandidatesToken_, simTICLCandidates);
  edm::Handle<std::vector<ticl::Trackster>> simTrackstersFromCP; iEvent.getByToken(simTrackstersFromCPToken_, simTrackstersFromCP);
  edm::Handle<reco::PFCandidateCollection> pfMerged;         iEvent.getByToken(pfToken_, pfMerged);
  edm::Handle<reco::PFCandidateCollection> pfTmpBarrel;      iEvent.getByToken(pfTmpBarrelToken_, pfTmpBarrel);
  edm::Handle<reco::PFJetCollection>      pfJets;            iEvent.getByToken(pfJetsToken_, pfJets);
  edm::Handle<reco::GenParticleCollection> genParticles;     iEvent.getByToken(genParticlesToken_, genParticles);
  edm::Handle<reco::GenParticleCollection> genVisTaus;       iEvent.getByToken(genVisTausToken_, genVisTaus);

  if (!simTaus.isValid()) {
    edm::LogWarning("TICLTauValidator") << "simTaus invalid, skipping event " << iEvent.id();
    return;
  }
  if (!simTICLCandidates.isValid()) {
    edm::LogWarning("TICLTauValidator") << "sim TICL candidate collection missing in event " << iEvent.id();
  }

  EventContext ctx;
  ctx.taus              = taus.isValid() ? taus.product() : nullptr;
  ctx.jets              = pfJets.isValid() ? pfJets.product() : nullptr;
  ctx.pfMerged          = pfMerged.isValid() ? pfMerged.product() : nullptr;
  ctx.ticlCandidates    = ticlCandidates.isValid() ? ticlCandidates.product() : nullptr;
  ctx.simTICLCandidates = simTICLCandidates.isValid() ? simTICLCandidates.product() : nullptr;
  ctx.simToRecoMap      = simToRecoMap.isValid() ? simToRecoMap.product() : nullptr;
  ctx.recoToSimMap      = recoToSimMap.isValid() ? recoToSimMap.product() : nullptr;
  ctx.genParticles      = genParticles.isValid() ? genParticles.product() : nullptr;
  ctx.genVisTaus        = genVisTaus.isValid() ? genVisTaus.product() : nullptr;
  if (pfMerged.isValid())
    ctx.pfMergedId = pfMerged.id();
  if (pfJets.isValid())
    ctx.jetsId = pfJets.id();
  ctx.barrelSize = pfTmpBarrel.isValid() ? pfTmpBarrel->size() : 0;
  buildRecoLookups(ctx);
  buildCpToSimIdx(simTrackstersFromCP.isValid() ? simTrackstersFromCP.product() : nullptr, ctx);

  SeenSets seen;
  for (const auto& link : *simTaus)
    processLink(link, ctx, seen);

  processFakeRates(*simTaus, ctx);

  LogDebug("TICLTauValidator")
      << "event " << iEvent.id()
      << " nSimTaus=" << simTaus->size()
      << " nCPsCharged=" << seen.charged.size()
      << " nCPsPhoton=" << seen.photon.size()
      << " nTaus=" << (ctx.taus ? ctx.taus->size() : 0);
}

void TICLTauValidator::buildRecoLookups(EventContext& ctx) const {
  // Map reco track identity to PF candidate indices in the merged collection.
  if (ctx.pfMerged) {
    for (size_t pfi = 0; pfi < ctx.pfMerged->size(); ++pfi) {
      const auto& trkRef = (*ctx.pfMerged)[pfi].trackRef();
      if (trkRef.isNonnull())
        ctx.trackKeyToPFIdx[{trkRef.id(), trkRef.key()}].push_back(pfi);
    }
  }
  // Map PF candidate identity to the jets containing it.
  if (ctx.jets) {
    for (size_t j = 0; j < ctx.jets->size(); ++j) {
      for (const auto& pfPtr : (*ctx.jets)[j].getPFConstituents())
        if (pfPtr.isNonnull())
          ctx.pfKeyToJets[{pfPtr.id(), pfPtr.key()}].push_back(j);
    }
  }
  // Per-tau signal/isolation PF keys and the tau(s) seeded by each jet.
  if (ctx.taus) {
    ctx.tauSignalKeys.resize(ctx.taus->size());
    ctx.tauIsoKeys.resize(ctx.taus->size());
    for (size_t t = 0; t < ctx.taus->size(); ++t) {
      const auto& tau = (*ctx.taus)[t];
      for (const auto& p : tau.signalPFCands())
        if (p.isNonnull())
          ctx.tauSignalKeys[t].emplace(p.id(), p.key());
      for (const auto& p : tau.isolationPFCands())
        if (p.isNonnull())
          ctx.tauIsoKeys[t].emplace(p.id(), p.key());
      const auto jetRef = tau.jetRef();
      if (ctx.jets && jetRef.isNonnull() && jetRef.id() == ctx.jetsId && jetRef.key() < ctx.jets->size())
        ctx.tausPerJet[jetRef.key()].push_back(t);
    }
  }
}

void TICLTauValidator::buildCpToSimIdx(const std::vector<ticl::Trackster>* simTrackstersFromCP,
                                       EventContext& ctx) const {
  // Map CaloParticle -> index in the (compacted) fromCPs sim-trackster collection.
  // SimTrackstersProducer erases the entries of CaloParticles without deposits from both the
  // fromCPs tracksters and the sim TICL candidates, so those collections
  // are not aligned with the CaloParticle collection.
  // See also RecoHGCal/TICL/plugins/SimTrackstersProducer.cc
  if (!simTrackstersFromCP) {
    edm::LogWarning("TICLTauValidator")
        << "fromCPs sim trackster collection missing; cannot translate CaloParticle keys.";
    return;
  }
  if (ctx.simTICLCandidates && simTrackstersFromCP->size() != ctx.simTICLCandidates->size()) {
    edm::LogWarning("TICLTauValidator")
        << "fromCPs sim tracksters (" << simTrackstersFromCP->size()
        << ") and sim TICL candidates (" << ctx.simTICLCandidates->size()
        << ") are not index-aligned; CP-to-sim candidate matching may be invalid.";
  }
  for (size_t i = 0; i < simTrackstersFromCP->size(); ++i) {
    const auto& simTk = (*simTrackstersFromCP)[i];
    const int seedIdx = simTk.seedIndex();
    if (seedIdx < 0)
      continue;
    const auto [_, inserted] =
        ctx.cpToSimIdx.emplace(std::make_pair(simTk.seedID(), static_cast<unsigned int>(seedIdx)), i);
    if (!inserted) {
      edm::LogWarning("TICLTauValidator")
          << "duplicate fromCPs sim trackster seed (product " << simTk.seedID()
          << ", key " << seedIdx << "); keeping the first index.";
    }
  }
}

void TICLTauValidator::jetsForPFIdxs(const std::set<size_t>& pfIdxs,
                                     std::set<size_t>& jetsOut,
                                     const EventContext& ctx) const {
  for (const size_t pfIdx : pfIdxs) {
    const auto it = ctx.pfKeyToJets.find({ctx.pfMergedId, pfIdx});
    if (it != ctx.pfKeyToJets.end())
      jetsOut.insert(it->second.begin(), it->second.end());
  }
}

void TICLTauValidator::matchTauRegions(const std::set<size_t>& pfIdxs,
                                       const std::set<size_t>& legJets,
                                       std::unordered_map<size_t, TauRegionFlags>& regions,
                                       const EventContext& ctx) const {
  if (!ctx.taus)
    return;
  for (const size_t j : legJets) {
    const auto tIt = ctx.tausPerJet.find(j);
    if (tIt == ctx.tausPerJet.end())
      continue;
    for (const size_t t : tIt->second) {
      auto& flags = regions[t];
      for (const size_t pfIdx : pfIdxs) {
        const ProductKey key{ctx.pfMergedId, pfIdx};
        if (!flags.signal && ctx.tauSignalKeys[t].count(key))
          flags.signal = true;
        if (!flags.isolation && ctx.tauIsoKeys[t].count(key))
          flags.isolation = true;
        if (flags.signal && flags.isolation)
          break;
      }
    }
  }
}

void TICLTauValidator::traceTrackChain(const std::vector<ProductKey>& simCandTrackKeys,
                                       PendingInfo& pend,
                                       const EventContext& ctx) const {
  // Track Step 0: CP -> SimTICLCand. Always true: the SimTICLCandidate is built from the CP.
  pend.trackStepPass[0] = true;

  // Track Step 1: SimTICLCand -> reco Track. The SimTICLCandidate already carries the
  // associated reco track (attached in SimTrackstersProducer, quality-gated).
  std::set<ProductKey> matchedRecoTrackKeys(simCandTrackKeys.begin(), simCandTrackKeys.end());
  if (!matchedRecoTrackKeys.empty()) {
    pend.trackStepPass[1] = true;
    pend.trackMatched = true;
  }
  LogDebug("TICLTauValidator") << "    trk-step1 matched=" << (!matchedRecoTrackKeys.empty());

  // Track Step 2: reco Track -> PF candidate
  for (const auto& trackKey : matchedRecoTrackKeys) {
    const auto it = ctx.trackKeyToPFIdx.find(trackKey);
    if (it != ctx.trackKeyToPFIdx.end())
      pend.trackPFKeys.insert(it->second.begin(), it->second.end());
  }
  pend.trackStepPass[2] = !pend.trackPFKeys.empty();
  LogDebug("TICLTauValidator") << "    trk-step2 matched=" << (!pend.trackPFKeys.empty());

  // Track Step 3: PF -> PFJet
  jetsForPFIdxs(pend.trackPFKeys, pend.trackJets, ctx);
  pend.trackStepPass[3] = !pend.trackJets.empty();
  LogDebug("TICLTauValidator") << "    trk-step3 matched=" << (!pend.trackJets.empty());

  // Track Step 4 (PFJet -> PFTau usage) is evaluated in processLink once the tau is
  // selected; here we only record which taus contain the track-chain PF candidates.
  matchTauRegions(pend.trackPFKeys, pend.trackJets, pend.trackTauRegion, ctx);
}

int TICLTauValidator::findBestTICLCandidateIdx(size_t simIdx, PendingInfo& pend, const EventContext& ctx) const {
  // Step 1: merged (fromCPs) sim trackster -> best reco trackster.
  // The sim->reco association map is keyed by the fromCPs sim-trackster collection,
  // so simIdx must be the (compacted) fromCPs index, not the CaloParticle key.
  // Best match = lowest score (TICLCandidateValidator convention), gated by maxAssocScore_.
  int bestRecoTkIdx = -1;
  if (!ctx.simToRecoMap) {
    edm::LogWarning("TICLTauValidator") << "Trackster association map is missing.";
  } else if (simIdx >= ctx.simToRecoMap->size()) {
    edm::LogWarning("TICLTauValidator")
        << "sim trackster index " << simIdx
        << " is out of range for the sim->reco association map (size "
        << ctx.simToRecoMap->size() << ").";
  } else {
    const auto& assocs = (*ctx.simToRecoMap)[simIdx];
    auto best = std::min_element(assocs.begin(), assocs.end(), [](const auto& a, const auto& b) {
      return a.score() < b.score();
    });
    if (best != assocs.end()) {
      const bool pass = best->score() <= maxAssocScore_;
      LogDebug("TICLTauValidator")
          << "    step1 simIdx=" << simIdx
          << " bestRecoTkIdx=" << best->index()
          << " score=" << best->score()
          << " sharedE=" << best->sharedEnergy()
          << " pass=" << pass;
      if (pass)
        bestRecoTkIdx = static_cast<int>(best->index());
    } else {
      LogDebug("TICLTauValidator") << "    step1 simIdx=" << simIdx << " no associations";
    }
  }
  if (bestRecoTkIdx < 0)
    return -1;
  pend.stepPass[1] = true;

  // Step 2: THE reco TICLCandidate containing the best reco trackster
  // (in TICLv5 there is exactly one trackster per candidate).
  if (!ctx.ticlCandidates) {
    edm::LogWarning("TICLTauValidator") << "TICLCandidate collection missing";
    return -1;
  }
  for (size_t ci = 0; ci < ctx.ticlCandidates->size(); ++ci) {
    for (const auto& tracksterPtr : (*ctx.ticlCandidates)[ci].tracksters()) {
      if (!tracksterPtr.isNonnull() || tracksterPtr.key() != static_cast<size_t>(bestRecoTkIdx))
        continue;
      pend.stepPass[2] = true;
      LogDebug("TICLTauValidator") << "    step2 candIdx=" << ci;
      return static_cast<int>(ci);
    }
  }
  LogDebug("TICLTauValidator") << "    step2 no candidate uses recoTkIdx=" << bestRecoTkIdx;
  return -1;
}

void TICLTauValidator::traceCaloToTau(int candIdx, PendingInfo& pend, const EventContext& ctx) const {
  if (!ctx.pfMerged || candIdx < 0)
    return;
  // Step 3: TICLCandidate -> PF. TICL PF candidates are appended after the barrel part,
  // exactly one per TICLCandidate (PFTICLProducer contract).
  const size_t pfIdx = ctx.barrelSize + static_cast<size_t>(candIdx);
  if (pfIdx >= ctx.pfMerged->size()) {
    LogDebug("TICLTauValidator")
        << "    step3 candIdx=" << candIdx << " pfIdx=" << pfIdx
        << " OUT_OF_RANGE (pfMerged.size=" << ctx.pfMerged->size() << ")";
    return;
  }
  const auto& pfCand = (*ctx.pfMerged)[pfIdx];
  LogDebug("TICLTauValidator")
      << "    step3 candIdx=" << candIdx
      << " pfIdx=" << pfIdx
      << " pt=" << pfCand.pt()
      << " eta=" << pfCand.eta()
      << " phi=" << pfCand.phi()
      << " pdg=" << pfCand.pdgId();
  pend.hasPFKinematics = true;
  pend.pfPt = pfCand.pt();
  pend.stepPass[3] = true;
  pend.caloPFKeys.insert(pfIdx);

  // Step 4: PF -> PFJet
  jetsForPFIdxs(pend.caloPFKeys, pend.jets, ctx);
  pend.stepPass[4] = !pend.jets.empty();
  LogDebug("TICLTauValidator") << "    step4 matched=" << (!pend.jets.empty());

  // Step 5 (PFJet -> PFTau usage) is evaluated in processLink once the tau is selected;
  // here we only record which taus contain the calo-chain PF candidate.
  matchTauRegions(pend.caloPFKeys, pend.jets, pend.tauRegion, ctx);
}

void TICLTauValidator::processLeaf(const SimTauCPLink::DecayNav& leaf,
                                   const SimTauCPLink& link,
                                   int dmGenIdx,
                                   const EventContext& ctx,
                                   SeenSets& seen,
                                   PendingMap& pendingHad,
                                   PendingMap& pendingGamma) {
  const int cp_id = leaf.calo_particle_idx();
  if (cp_id < 0 || static_cast<size_t>(cp_id) >= link.calo_particle_leaves.size())
    return;
  const auto& cpRef = link.calo_particle_leaves[cp_id];
  if (!cpRef.isNonnull())
    return;
  const auto& cp = *cpRef;

  // keep only CPs in HGCAL
  if (std::abs(cp.eta()) < hgcalEtaAbsMin_)
    return;

  const int absPdg = std::abs(cp.pdgId());
  const bool isPhoton        = (absPdg == 22);
  const bool isChargedHadron = (absPdg == 211 || absPdg == 321 || absPdg == 2212);
  if (!isChargedHadron && !isPhoton)
    return;

  // context CP histos (once per event per CP key) + per-DM base histos
  if (isChargedHadron && seen.charged.insert(cpRef.key()).second) {
    if (cp_chHad_pt_all_)  cp_chHad_pt_all_->Fill(cp.pt());
    if (cp_chHad_eta_all_) cp_chHad_eta_all_->Fill(cp.eta());
    if (cp_chHad_pt_dm_[dmGenIdx])  cp_chHad_pt_dm_[dmGenIdx]->Fill(cp.pt());
    if (cp_chHad_eta_dm_[dmGenIdx]) cp_chHad_eta_dm_[dmGenIdx]->Fill(cp.eta());
  }
  if (isPhoton && seen.photon.insert(cpRef.key()).second) {
    if (cp_gamma_pt_all_)  cp_gamma_pt_all_->Fill(cp.pt());
    if (cp_gamma_eta_all_) cp_gamma_eta_all_->Fill(cp.eta());
    if (cp_gamma_pt_dm_[dmGenIdx])  cp_gamma_pt_dm_[dmGenIdx]->Fill(cp.pt());
    if (cp_gamma_eta_dm_[dmGenIdx]) cp_gamma_eta_dm_[dmGenIdx]->Fill(cp.eta());
  }

  auto& pend = isChargedHadron ? pendingHad[cpRef.key()] : pendingGamma[cpRef.key()];
  pend.hasCPKinematics = true;
  pend.cpPt = cp.pt();
  pend.cpEta = cp.eta();

  LogDebug("TICLTauValidator")
      << "  cp key=" << cpRef.key()
      << " pdg=" << cp.pdgId()
      << " pt=" << cp.pt()
      << " eta=" << cp.eta()
      << " phi=" << cp.phi()
      << " type=" << (isChargedHadron ? "hadron" : "photon");

  // Select THE sim TICL candidate for this CP (logically first).
  // The sim TICL candidates are index-aligned with the compacted fromCPs sim
  // tracksters, not with the CaloParticles, so translate the CP key first.
  const auto simIdxIt = ctx.cpToSimIdx.find({cpRef.id(), cpRef.key()});
  const bool hasSimIdx = (simIdxIt != ctx.cpToSimIdx.end());
  const size_t simIdx = hasSimIdx ? simIdxIt->second : 0;
  const TICLCandidate* simCand = nullptr;
  if (hasSimIdx && ctx.simTICLCandidates && simIdx < ctx.simTICLCandidates->size())
    simCand = &(*ctx.simTICLCandidates)[simIdx];

  // Step 0: CP -> sim trackster(s): the CP left usable deposits in HGCAL
  const bool hasSimTracksters = simCand && !simCand->tracksters().empty();
  pend.stepPass[0] = hasSimTracksters;
  LogDebug("TICLTauValidator") << "    step0 matched=" << hasSimTracksters;
  if (!hasSimTracksters)
    return;

  // Track chain (charged legs): sim candidate's reco track(s) -> PF -> jet -> tau
  // (sim-side track list, consistent with TICLCandidateValidator)
  if (isChargedHadron) {
    std::vector<ProductKey> simCandTrackKeys;
    for (const auto& trkPtr : simCand->trackPtrs())
      if (trkPtr.isNonnull())
        simCandTrackKeys.emplace_back(trkPtr.id(), trkPtr.key());
    if (simCand->trackPtr().isNonnull())
      simCandTrackKeys.emplace_back(simCand->trackPtr().id(), simCand->trackPtr().key());
    traceTrackChain(simCandTrackKeys, pend, ctx);
  }

  // Calo chain: merged sim trackster -> best reco trackster -> reco TICLCandidate -> PF -> jet -> tau
  const int candIdx = hasSimIdx ? findBestTICLCandidateIdx(simIdx, pend, ctx) : -1;
  traceCaloToTau(candIdx, pend, ctx);

  // CP-to-PF pT response (per-DM)
  if (pend.hasPFKinematics && cp.pt() > 0.) {
    const double ratio = pend.pfPt / cp.pt();
    if (isChargedHadron && cp_pf_pt_resolution_had_dm_[dmGenIdx])
      cp_pf_pt_resolution_had_dm_[dmGenIdx]->Fill(ratio);
    if (isPhoton && cp_pf_pt_resolution_em_dm_[dmGenIdx])
      cp_pf_pt_resolution_em_dm_[dmGenIdx]->Fill(ratio);
    if (!cp.g4Tracks().empty()) {
      // The CaloParticle pT is taken at the production vertex, while the sim trackster
      // energy comes from the momentum at the HGCAL boundary.
      const auto& g4 = cp.g4Tracks()[0];
      const double bPt = g4.crossedBoundary() ? g4.getMomentumAtBoundary().pt() : -1.;
      LogDebug("TICLTauValidator")
          << "    resp cpKey=" << cpRef.key()
          << " pdg=" << cp.pdgId()
          << " cpPt=" << cp.pt()
          << " boundaryPt=" << bPt
          << " pfPt=" << pend.pfPt
          << " pf/cp=" << ratio
          << " pf/boundary=" << (bPt > 0. ? pend.pfPt / bPt : -1.);
    }
  }

  // Both-path coverage does not require both paths to converge on the same reco object.
  pend.combiStepPass[0] = pend.stepPass[3] && pend.trackStepPass[2];
  pend.combiStepPass[1] = pend.stepPass[4] && pend.trackStepPass[3];
  // combiStepPass[2] is computed in processLink once stepPass[5] is known

  // CP-level two-fold efficiency numerators (once per unique CP key per event)
  if (isChargedHadron && seen.twoFoldCharged.insert(cpRef.key()).second) {
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
  if (isPhoton && seen.twoFoldPhoton.insert(cpRef.key()).second) {
    if (pend.stepPass[3]) {
      if (cp_gamma_caloOnly_pt_dm_[dmGenIdx])  cp_gamma_caloOnly_pt_dm_[dmGenIdx]->Fill(pend.cpPt);
      if (cp_gamma_caloOnly_eta_dm_[dmGenIdx]) cp_gamma_caloOnly_eta_dm_[dmGenIdx]->Fill(pend.cpEta);
    }
  }
}

void TICLTauValidator::processLink(const SimTauCPLink& link, const EventContext& ctx, SeenSets& seen) {
  const int dmPhys = link.decayMode;
  const int dmGenIdx = dmToGenIndex(dmPhys);
  if (dmGenIdx < 0)
    return; // only selected physical gen DMs (excludes e.g. leptonic modes and DM 5)

  LogDebug("TICLTauValidator")
      << "=== link DM=" << dmPhys
      << " nLeaves=" << link.leaves.size()
      << " nCPLeaves=" << link.calo_particle_leaves.size()
      << " ===";

  PendingMap pendingHad, pendingGamma;
  for (const auto& leaf : link.leaves)
    processLeaf(leaf, link, dmGenIdx, ctx, seen, pendingHad, pendingGamma);

  // Jets at step 4: unique jet that contains all PF legs
  std::set<size_t> commonJetsAllPF;
  {
    std::vector<const std::set<size_t>*> jetSets;
    for (const auto& kv : pendingHad)
      if (kv.second.stepPass[4])
        jetSets.push_back(&kv.second.jets);
    for (const auto& kv : pendingGamma)
      if (kv.second.stepPass[4])
        jetSets.push_back(&kv.second.jets);
    if (!jetSets.empty()) {
      commonJetsAllPF = *jetSets.front();
      for (size_t i = 1; i < jetSets.size() && !commonJetsAllPF.empty(); ++i) {
        std::set<size_t> tmp;
        std::set_intersection(commonJetsAllPF.begin(), commonJetsAllPF.end(),
                              jetSets[i]->begin(), jetSets[i]->end(),
                              std::inserter(tmp, tmp.begin()));
        commonJetsAllPF.swap(tmp);
      }
    }
  }
  const bool hasUniqueJetEndpoint = (commonJetsAllPF.size() == 1);
  const size_t jetEndpoint = hasUniqueJetEndpoint ? *commonJetsAllPF.begin() : static_cast<size_t>(-1);

  LegCounts counts;
  if (hasUniqueJetEndpoint) {
    for (const auto& kv : pendingHad)
      if (kv.second.stepPass[4] && kv.second.jets.count(jetEndpoint))
        ++counts.chJet;
    for (const auto& kv : pendingGamma)
      if (kv.second.stepPass[4] && kv.second.jets.count(jetEndpoint))
        ++counts.gammaJet;
    if (dm_reco_vs_gen_jet_) {
      const int coverageIdx = dmToSelIndex(coverageDMFromTruthCPs(counts.chJet, counts.gammaJet));
      if (coverageIdx >= 0)
        dm_reco_vs_gen_jet_->Fill(coverageIdx, dmGenIdx);
    }
  }

  // Select one reco tau for this gen tau before evaluating any tau endpoint.
  const bool havePFTaus = (ctx.taus && !ctx.taus->empty() && ctx.jets);
  const int bestTauIdx =
      havePFTaus ? selectBestTauIdx(pendingHad, pendingGamma, hasUniqueJetEndpoint, jetEndpoint, ctx) : -1;

  // Evaluate the tau endpoint (calo step 5 / track step 4) against the selected tau.
  auto processLegTauEndpoint = [&](PendingInfo& info, bool isPhoton) {
    bool usedSignal = false, usedIso = false, trackUsed = false;
    if (bestTauIdx >= 0) {
      const auto caloIt = info.tauRegion.find(bestTauIdx);
      usedSignal = caloIt != info.tauRegion.end() && caloIt->second.signal;
      usedIso = caloIt != info.tauRegion.end() && caloIt->second.isolation;
      const auto trackIt = info.trackTauRegion.find(bestTauIdx);
      trackUsed = trackIt != info.trackTauRegion.end() && (trackIt->second.signal || trackIt->second.isolation);
    }
    const bool usedTotal = usedSignal || usedIso;
    info.stepPass[5] = usedTotal;
    info.trackStepPass[4] = trackUsed;
    info.combiStepPass[2] = usedTotal && trackUsed;
    if (!usedTotal)
      return;
    if (!isPhoton) {
      ++counts.chTau;
      if (usedSignal) ++counts.chSig;
      if (usedIso)    ++counts.chIso;
    } else {
      ++counts.gammaTau;
      if (usedSignal) ++counts.gammaSig;
      if (usedIso)    ++counts.gammaIso;
    }
  };
  for (auto& kv : pendingHad)   processLegTauEndpoint(kv.second, false);
  for (auto& kv : pendingGamma) processLegTauEndpoint(kv.second, true);

  // Tau-level two-fold: count charged CPs matched by track / calo / both / either
  for (const auto& kv : pendingHad) {
    const bool trk = kv.second.trackMatched;
    const bool cal = kv.second.stepPass[3];
    if (trk)        ++counts.chTrack;
    if (cal)        ++counts.chCalo;
    if (trk && cal) ++counts.chBoth;
    if (trk || cal) ++counts.chEither;
  }

  if (bestTauIdx >= 0) {
    const auto& tau = (*ctx.taus)[bestTauIdx];
    if (dm_reco_vs_gen_tau_) {
      const int coverageIdx = dmToSelIndex(coverageDMFromTruthCPs(counts.chTau, counts.gammaTau));
      if (coverageIdx >= 0)
        dm_reco_vs_gen_tau_->Fill(coverageIdx, dmGenIdx);
    }
    if (dm_reco_vs_gen_hps_) {
      const int recoIdx = dmToSelIndex(tau.decayMode());
      if (recoIdx >= 0)
        dm_reco_vs_gen_hps_->Fill(recoIdx, dmGenIdx);
    }
  }

  fillLegStepHists(dmPhys, dmGenIdx, pendingHad, pendingGamma);

  // gen-tau kinematics for tau-level plots
  double tauPt = 0., tauEta = 0.;
  const bool haveGenTau = genTauKinematics(link, ctx, tauPt, tauEta);
  const bool genInAcceptance = haveGenTau && std::abs(tauEta) > hgcalEtaAbsMin_;
  if (genInAcceptance) {
    if (tau_gen_pt_[dmGenIdx])  tau_gen_pt_[dmGenIdx]->Fill(tauPt);
    if (tau_gen_eta_[dmGenIdx]) tau_gen_eta_[dmGenIdx]->Fill(tauEta);
    fillTauLevelHists(dmPhys, dmGenIdx, tauPt, tauEta, havePFTaus, bestTauIdx, counts, ctx);
  }

  LogDebug("TICLTauValidator")
      << "link DM=" << dmPhys
      << " nCH=" << pendingHad.size()
      << " nPho=" << pendingGamma.size()
      << " jetEndpoint=" << (hasUniqueJetEndpoint ? static_cast<int>(jetEndpoint) : -1)
      << " nGoodCH_jet=" << counts.chJet
      << " nGoodGamma_jet=" << counts.gammaJet
      << " nGoodCH_tau=" << counts.chTau
      << " nGoodGamma_tau=" << counts.gammaTau
      << " endpoint=" << (havePFTaus ? "TAU" : "JET");
}

int TICLTauValidator::selectBestTauIdx(const PendingMap& pendingHad,
                                       const PendingMap& pendingGamma,
                                       bool hasUniqueJetEndpoint,
                                       size_t jetEndpoint,
                                       const EventContext& ctx) const {
  std::set<size_t> candidateTauIdxs;
  if (hasUniqueJetEndpoint) {
    const auto it = ctx.tausPerJet.find(jetEndpoint);
    if (it != ctx.tausPerJet.end())
      candidateTauIdxs.insert(it->second.begin(), it->second.end());
  }
  if (candidateTauIdxs.empty()) {
    std::set<size_t> jetsWithAnyLeg;
    for (const auto& kv : pendingHad) {
      jetsWithAnyLeg.insert(kv.second.jets.begin(), kv.second.jets.end());
      jetsWithAnyLeg.insert(kv.second.trackJets.begin(), kv.second.trackJets.end());
    }
    for (const auto& kv : pendingGamma)
      jetsWithAnyLeg.insert(kv.second.jets.begin(), kv.second.jets.end());
    for (const size_t jetIdx : jetsWithAnyLeg) {
      const auto it = ctx.tausPerJet.find(jetIdx);
      if (it != ctx.tausPerJet.end())
        candidateTauIdxs.insert(it->second.begin(), it->second.end());
    }
  }

  auto regionUsed = [](const std::unordered_map<size_t, TauRegionFlags>& regions, size_t tauIdx) {
    const auto it = regions.find(tauIdx);
    return it != regions.end() && (it->second.signal || it->second.isolation);
  };

  int bestTauIdx = -1;
  int bestOverlap = -1;
  for (const size_t tauIdx : candidateTauIdxs) {
    int overlap = 0;
    for (const auto& kv : pendingHad)
      if (regionUsed(kv.second.tauRegion, tauIdx) || regionUsed(kv.second.trackTauRegion, tauIdx))
        ++overlap;
    for (const auto& kv : pendingGamma)
      if (regionUsed(kv.second.tauRegion, tauIdx))
        ++overlap;
    if (overlap > 0 && overlap > bestOverlap) {
      bestOverlap = overlap;
      bestTauIdx = static_cast<int>(tauIdx);
    }
  }
  return bestTauIdx;
}

bool TICLTauValidator::genTauKinematics(const SimTauCPLink& link,
                                        const EventContext& ctx,
                                        double& tauPt,
                                        double& tauEta) const {
  const reco::GenParticle* bestMotherTau = nullptr;
  double bestMotherPt = -1.;
  unsigned int bestMotherIdx = static_cast<unsigned int>(-1);
  if (ctx.genParticles) {
    for (const auto& leaf : link.leaves) {
      const int genIdx = leaf.gen_particle_idx();
      if (genIdx < 0 || genIdx >= static_cast<int>(ctx.genParticles->size()))
        continue;
      const reco::GenParticle* cur = &(*ctx.genParticles)[genIdx];
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
            bestMotherPt = mom->pt();
            bestMotherIdx = cur->motherRef().key();
          }
          break;
        }
        cur = mom;
      }
    }
  }
  if (!bestMotherTau)
    return false;

  // Fall back to mother tau kinematics to avoid default zero values.
  tauPt = bestMotherTau->pt();
  tauEta = bestMotherTau->eta();
  if (!ctx.genVisTaus) {
    edm::LogWarning("TICLTauValidator") << "genVisTaus collection missing/invalid; using mother tau kinematics"
                                        << " (dm=" << link.decayMode << ", tau pt=" << tauPt
                                        << ", eta=" << tauEta << ")";
    return true;
  }
  for (const auto& genVisTau : *ctx.genVisTaus) {
    if (genVisTau.motherRef().isNonnull() && genVisTau.motherRef().key() == bestMotherIdx) {
      tauPt = genVisTau.pt();
      tauEta = genVisTau.eta();
      return true;
    }
  }
  edm::LogWarning("TICLTauValidator") << "No genVisTau match for bestMotherIdx=" << bestMotherIdx
                                      << " (dm=" << link.decayMode << ", tau pt=" << tauPt
                                      << ", eta=" << tauEta << ")";
  return true;
}

void TICLTauValidator::fillLegStepHists(int dmPhys,
                                        int dmGenIdx,
                                        const PendingMap& pendingHad,
                                        const PendingMap& pendingGamma) {
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

  auto sortedLegs = [](const PendingMap& pending) {
    std::vector<const PendingInfo*> legs;
    legs.reserve(pending.size());
    for (const auto& kv : pending)
      if (kv.second.hasCPKinematics)
        legs.push_back(&kv.second);
    std::sort(legs.begin(), legs.end(),
              [](const PendingInfo* a, const PendingInfo* b) { return a->cpPt > b->cpPt; });
    return legs;
  };

  // leg index = pT rank among the truth CPs of this link
  const int chMaxLegs = chCapForDM(dmPhys);
  int li = 0;
  for (const PendingInfo* pend : sortedLegs(pendingHad)) {
    if (li >= chMaxLegs)
      break;
    fillStepHists(chStepHists_[dmGenIdx][li], kNSteps, pend->cpPt, pend->cpEta, pend->stepPass);
    fillStepHists(chTrackStepHists_[dmGenIdx][li], kNTrackSteps, pend->cpPt, pend->cpEta, pend->trackStepPass);
    fillStepHists(chCombiStepHists_[dmGenIdx][li], kNCombiSteps, pend->cpPt, pend->cpEta, pend->combiStepPass);
    ++li;
  }

  const int gammaMaxLegs = gammaCapForDM(dmPhys);
  li = 0;
  for (const PendingInfo* pend : sortedLegs(pendingGamma)) {
    if (li >= gammaMaxLegs)
      break;
    fillStepHists(gammaStepHists_[dmGenIdx][li], kNSteps, pend->cpPt, pend->cpEta, pend->stepPass);
    ++li;
  }
}

void TICLTauValidator::fillTauLevelHists(int dmPhys,
                                         int dmGenIdx,
                                         double tauPt,
                                         double tauEta,
                                         bool havePFTaus,
                                         int bestTauIdx,
                                         const LegCounts& counts,
                                         const EventContext& ctx) {
  const int chCap = chCapForDM(dmPhys);
  const int pi0Cap = pi0CapForDM(dmPhys);
  const int expCh = expectedChForDM(dmPhys);
  const int expPi0 = expectedPi0ForDM(dmPhys);

  // Without a PFTau collection the PFJet endpoint is used instead.
  const int nGoodCH_endpoint = havePFTaus ? counts.chTau : counts.chJet;
  const int nGoodGamma_endpoint = havePFTaus ? counts.gammaTau : counts.gammaJet;
  // Two recovered truth-photon CPs represent one truth pi0, even if they share one PF candidate.
  const int nPi0Equivalent_endpoint = nGoodGamma_endpoint / 2;

  // >= N charged legs / pi0 equivalents at reco
  for (int N = 1; N <= chCap; ++N) {
    if (nGoodCH_endpoint >= N) {
      if (auto* h = tau_gen_matched_to_nCh_pt_[dmGenIdx][N - 1])  h->Fill(tauPt);
      if (auto* h = tau_gen_matched_to_nCh_eta_[dmGenIdx][N - 1]) h->Fill(tauEta);
    }
  }
  for (int N = 1; N <= pi0Cap; ++N) {
    if (nPi0Equivalent_endpoint >= N) {
      if (auto* h = tau_gen_matched_to_nPi0_pt_[dmGenIdx][N - 1])  h->Fill(tauPt);
      if (auto* h = tau_gen_matched_to_nPi0_eta_[dmGenIdx][N - 1]) h->Fill(tauEta);
    }
  }

  // ALL expected legs
  if ((expCh > 0 || expPi0 > 0) && nGoodCH_endpoint >= expCh && nPi0Equivalent_endpoint >= expPi0) {
    if (tau_gen_matched_to_all_pt_[dmGenIdx])  tau_gen_matched_to_all_pt_[dmGenIdx]->Fill(tauPt);
    if (tau_gen_matched_to_all_eta_[dmGenIdx]) tau_gen_matched_to_all_eta_[dmGenIdx]->Fill(tauEta);
  }

  // Tau-level two-fold: >= N charged track/calo/both/either
  for (int N = 1; N <= chCap; ++N) {
    if (counts.chTrack >= N) {
      if (auto* h = tau_nCh_track_pt_[dmGenIdx][N - 1])  h->Fill(tauPt);
      if (auto* h = tau_nCh_track_eta_[dmGenIdx][N - 1]) h->Fill(tauEta);
    }
    if (counts.chCalo >= N) {
      if (auto* h = tau_nCh_calo_pt_[dmGenIdx][N - 1])  h->Fill(tauPt);
      if (auto* h = tau_nCh_calo_eta_[dmGenIdx][N - 1]) h->Fill(tauEta);
    }
    if (counts.chBoth >= N) {
      if (auto* h = tau_nCh_both_pt_[dmGenIdx][N - 1])  h->Fill(tauPt);
      if (auto* h = tau_nCh_both_eta_[dmGenIdx][N - 1]) h->Fill(tauEta);
    }
    if (counts.chEither >= N) {
      if (auto* h = tau_nCh_either_pt_[dmGenIdx][N - 1])  h->Fill(tauPt);
      if (auto* h = tau_nCh_either_eta_[dmGenIdx][N - 1]) h->Fill(tauEta);
    }
  }

  if (!havePFTaus)
    return;

  // signal vs iso split (tau endpoint only)
  const int nPi0Equivalent_signal = counts.gammaSig / 2;
  const int nPi0Equivalent_iso = counts.gammaIso / 2;
  for (int N = 1; N <= chCap; ++N) {
    if (counts.chSig >= N) {
      if (auto* h = tau_gen_matched_to_nCh_sig_pt_[dmGenIdx][N - 1])  h->Fill(tauPt);
      if (auto* h = tau_gen_matched_to_nCh_sig_eta_[dmGenIdx][N - 1]) h->Fill(tauEta);
    }
    if (counts.chIso >= N) {
      if (auto* h = tau_gen_matched_to_nCh_iso_pt_[dmGenIdx][N - 1])  h->Fill(tauPt);
      if (auto* h = tau_gen_matched_to_nCh_iso_eta_[dmGenIdx][N - 1]) h->Fill(tauEta);
    }
  }
  for (int N = 1; N <= pi0Cap; ++N) {
    if (nPi0Equivalent_signal >= N) {
      if (auto* h = tau_gen_matched_to_nPi0_sig_pt_[dmGenIdx][N - 1])  h->Fill(tauPt);
      if (auto* h = tau_gen_matched_to_nPi0_sig_eta_[dmGenIdx][N - 1]) h->Fill(tauEta);
    }
    if (nPi0Equivalent_iso >= N) {
      if (auto* h = tau_gen_matched_to_nPi0_iso_pt_[dmGenIdx][N - 1])  h->Fill(tauPt);
      if (auto* h = tau_gen_matched_to_nPi0_iso_eta_[dmGenIdx][N - 1]) h->Fill(tauEta);
    }
  }

  // tau response and reco tau shapes per DM
  if (tauPt > 0. && bestTauIdx >= 0 && ctx.taus) {
    const auto& tau = (*ctx.taus)[bestTauIdx];
    if (tau_pt_reco_over_gen_[dmGenIdx])
      tau_pt_reco_over_gen_[dmGenIdx]->Fill(tau.pt() / tauPt);
    if (tau_reco_pt_[dmGenIdx])  tau_reco_pt_[dmGenIdx]->Fill(tau.pt());
    if (tau_reco_eta_[dmGenIdx]) tau_reco_eta_[dmGenIdx]->Fill(tau.eta());
  }
}

int TICLTauValidator::coverageDMFromTruthCPs(int nCharged, int nPhotons) {
  const int nPi0Equivalents = nPhotons / 2;
  if (nCharged == 1) {
    if (nPi0Equivalents <= 0) return 0;
    if (nPi0Equivalents == 1) return 1;
    return 2;
  }
  if (nCharged == 2) return 5;
  if (nCharged == 3) return nPi0Equivalents >= 1 ? 11 : 10;
  return -1;
}






// Fake rate: loop reco taus and reverse the association chain.
void TICLTauValidator::processFakeRates(const std::vector<SimTauCPLink>& simTaus, const EventContext& ctx) {
  if (!ctx.taus || !ctx.jets || !ctx.pfMerged || !ctx.ticlCandidates || !ctx.simTICLCandidates ||
      !ctx.recoToSimMap) {
    edm::LogWarning("TICLTauValidator") << "Fake rate loop skipped:"
        << " taus=" << (ctx.taus ? "ok" : "invalid")
        << " pfJets=" << (ctx.jets ? "ok" : "invalid")
        << " pfMerged=" << (ctx.pfMerged ? "ok" : "invalid")
        << " ticlCandidates=" << (ctx.ticlCandidates ? "ok" : "invalid")
        << " simTICLCandidates=" << (ctx.simTICLCandidates ? "ok" : "invalid")
        << " recoToSimMap=" << (ctx.recoToSimMap ? "ok" : "invalid");
    return;
  }

  // Collect the fromCPs sim-trackster indices of the CaloParticles belonging to any simulated
  // hadronic tau: both the sim TICL candidates and the reco->sim association map are keyed by
  // those (compacted) indices, not by the CaloParticle keys.
  std::unordered_set<size_t> tauSimTracksterIdxs;
  for (const auto& link : simTaus) {
    if (!isHadronicDM(link.decayMode))
      continue;
    for (const auto& leaf : link.leaves) {
      const int cpId = leaf.calo_particle_idx();
      if (cpId < 0 || static_cast<size_t>(cpId) >= link.calo_particle_leaves.size())
        continue;
      const auto& cpRef = link.calo_particle_leaves[cpId];
      if (!cpRef.isNonnull())
        continue;
      const auto it = ctx.cpToSimIdx.find({cpRef.id(), cpRef.key()});
      if (it != ctx.cpToSimIdx.end())
        tauSimTracksterIdxs.insert(it->second);
    }
  }

  // Union of reco-track keys attached to the sim TICL candidates of all tau CPs.
  // Reverse of the efficiency track chain.
  std::set<ProductKey> tauSimCandTrackKeys;
  for (const auto simIdx : tauSimTracksterIdxs) {
    if (simIdx >= ctx.simTICLCandidates->size())
      continue;
    const auto& sc = (*ctx.simTICLCandidates)[simIdx];
    for (const auto& trkPtr : sc.trackPtrs())
      if (trkPtr.isNonnull())
        tauSimCandTrackKeys.emplace(trkPtr.id(), trkPtr.key());
    if (sc.trackPtr().isNonnull())
      tauSimCandTrackKeys.emplace(sc.trackPtr().id(), sc.trackPtr().key());
  }

  for (const auto& tau : *ctx.taus) {
    if (std::abs(tau.eta()) < hgcalEtaAbsMin_)
      continue;
    if (tau.signalPFCands().empty())
      continue;

    const int dmSelI = dmToSelIndex(tau.decayMode());
    const auto assocCounts = countAssociatedSignalPFCands(
        tau, ctx.barrelSize, tauSimTracksterIdxs, tauSimCandTrackKeys, *ctx.ticlCandidates, *ctx.recoToSimMap);
    const bool isGenuine      = (assocCounts.nAssocAllParticles > 0);
    const bool isGenuineCalo  = (assocCounts.nAssocCalo > 0);
    const bool isGenuineTrack = (assocCounts.nAssocTrack > 0);
    LogDebug("TICLTauValidator")
        << "fakeRate tau pt=" << tau.pt() << " eta=" << tau.eta()
        << " dm=" << tau.decayMode()
        << " genuine=" << isGenuine
        << " calo=" << isGenuineCalo
        << " track=" << isGenuineTrack
        << " nSigCh=" << assocCounts.nSigCh
        << " nSigPho=" << assocCounts.nSigPho;
    fillFakeRateHists(tau, dmSelI, isGenuine, fakeRate_);
    fillFakeRateHists(tau, dmSelI, isGenuineCalo, fakeRateCalo_);
    fillFakeRateHists(tau, dmSelI, isGenuineTrack, fakeRateTrack_);
  }
}

void TICLTauValidator::fillFakeRateHists(const reco::PFTau& tau,
                                         int dmSelI,
                                         bool isGenuine,
                                         FakeRateHists& fr) {
  auto fillPtEta = [&](MonitorElement* hpt, MonitorElement* heta) {
    if (hpt)  hpt->Fill(tau.pt());
    if (heta) heta->Fill(tau.eta());
  };

  fillPtEta(fr.den_pt, fr.den_eta);
  if (dmSelI >= 0)
    fillPtEta(fr.den_dm_pt[dmSelI], fr.den_dm_eta[dmSelI]);

  if (isGenuine)
    return;

  fillPtEta(fr.num_pt, fr.num_eta);
  if (dmSelI >= 0)
    fillPtEta(fr.num_dm_pt[dmSelI], fr.num_dm_eta[dmSelI]);

}

TICLTauValidator::AssocCounts TICLTauValidator::countAssociatedSignalPFCands(
    const reco::PFTau& tau,
    size_t barrelSize,
    const std::unordered_set<size_t>& tauSimTracksterIdxs,
    const std::set<ProductKey>& tauSimCandTrackKeys,
    const std::vector<TICLCandidate>& ticlCandidates,
    const TracksterToTracksterMap& recoToSimMap) const {
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

    // Track matching: is the PF candidate's track one of the reco tracks attached
    // to the sim TICL candidates of the tau CPs? (reverse of the efficiency chain)
    {
      const auto trackRef = pfPtr->trackRef();
      if (trackRef.isNonnull() && tauSimCandTrackKeys.count({trackRef.id(), trackRef.key()}))
        trackMatched = true;
    }

    // Calo (TICL) matching: endcap PF candidates only. Reverse of the efficiency chain:
    // the reco candidate's trackster's BEST sim match (lowest score, gated by maxAssocScore_)
    // must be the fromCPs sim trackster of a tau CP.
    if (pfKey >= barrelSize) {
      const size_t ticlIdx = pfKey - barrelSize;
      if (ticlIdx < ticlCandidates.size()) {
        const auto& cand = ticlCandidates[ticlIdx];
        for (const auto& tsPtr : cand.tracksters()) {
          if (!tsPtr.isNonnull())
            continue;
          const size_t recoTkIdx = tsPtr.key();
          if (recoTkIdx >= recoToSimMap.size()) {
            edm::LogWarning("TICLTauValidator")
                << "reco trackster key " << recoTkIdx
                << " is out of range for the reco->sim association map (size "
                << recoToSimMap.size() << "); skipping.";
            continue;
          }
          const auto& assocs = recoToSimMap[recoTkIdx];
          auto best = std::min_element(assocs.begin(), assocs.end(), [](const auto& a, const auto& b) {
            return a.score() < b.score();
          });
          if (best == assocs.end() || best->score() > maxAssocScore_)
            continue;
          if (tauSimTracksterIdxs.count(static_cast<size_t>(best->index()))) {
            caloMatched = true;
            break;
          }
        }
      }
    }

    if (trackMatched) ++counts.nAssocTrack;
    if (caloMatched) ++counts.nAssocCalo;
    if (trackMatched || caloMatched) ++counts.nAssocAllParticles;

    LogDebug("TICLTauValidator")
      << "  pf key=" << pfKey
      << " pdg=" << absPdg
      << " endcap=" << (pfKey >= barrelSize ? 1 : 0)
      << " track=" << trackMatched
      << " calo=" << caloMatched;
  }

  return counts;
}

void TICLTauValidator::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<std::string>("folder", "RecoTauV/ticlTauValidator");

  desc.add<edm::InputTag>("simTaus", edm::InputTag("SimTauProducer"));
  desc.add<edm::InputTag>("TauProducer");
  desc.add<edm::InputTag>("pf", edm::InputTag("particleFlow"));
  desc.add<edm::InputTag>("pfTmpBarrel", edm::InputTag("particleFlowTmpBarrel"));
  desc.add<edm::InputTag>("jets", edm::InputTag("ak4PFJets"));
  desc.add<edm::InputTag>("ticlCandidates", edm::InputTag("ticlCandidate"));
  desc.add<edm::InputTag>("simTICLCandidates", edm::InputTag("ticlSimTracksters"));
  desc.add<edm::InputTag>("simTracksters", edm::InputTag("ticlSimTracksters", "fromCPs"));
  desc.add<edm::InputTag>("simToRecoTracksterAssocByLCs",
                          edm::InputTag("allTrackstersToSimTrackstersAssociationsByLCs",
                                        "ticlSimTrackstersfromCPsToticlCandidate"));
  desc.add<edm::InputTag>("recoToSimTracksterAssocByLCs",
                          edm::InputTag("allTrackstersToSimTrackstersAssociationsByLCs",
                                        "ticlCandidateToticlSimTrackstersfromCPs"));
  desc.add<edm::InputTag>("genParticles", edm::InputTag("genParticles"));
  desc.add<edm::InputTag>("genVisTaus", edm::InputTag("genVisTaus"));
  desc.add<double>("maxAssocScore", 0.6);
  desc.add<double>("hgcalEtaAbsMin", 1.5);


  descriptions.add("ticlTauValidator", desc);
}

DEFINE_FWK_MODULE(TICLTauValidator);
