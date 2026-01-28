// -*- C++ -*-
//
// Package:    Validation/TICLTauValidator
// Class:      TICLTauValidator 
//
/*

 Description: Validation for tau reconstruction within TICL

 Implementation:
   Steps (0..5):
     0: CaloParticle - SimTrackster (seed match)
     1: SimTrackster - RecoTrackster (association map)
     2: RecoTrackster - TICLCandidate
     3: RecoTrackster - PF candidate (merged PF)
     4: PF cand - PFJet
     5: PFJet - PFTau usage (skipped if no taus)

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

#include "SimDataFormats/CaloAnalysis/interface/SimTauCPLink.h"
#include "SimDataFormats/Associations/interface/TICLAssociationMap.h"
#include "SimDataFormats/CaloAnalysis/interface/SimCluster.h"

class TICLTauValidator : public DQMEDAnalyzer {
public:
  explicit TICLTauValidator(const edm::ParameterSet&);
  ~TICLTauValidator() override {}
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void bookHistograms(DQMStore::IBooker&, edm::Run const&, edm::EventSetup const&) override;
  void analyze(const edm::Event&, const edm::EventSetup&) override;

  using TracksterToTracksterMap = ticl::AssociationMap<ticl::mapWithSharedEnergyAndScore,
                                                       std::vector<ticl::Trackster>,
                                                       std::vector<ticl::Trackster>>;

  std::string folder_;
  double maxAssocScore_;  // smaller = better association
  double hgcalEtaAbsMin_;

  edm::EDGetTokenT<std::vector<SimTauCPLink>> simTauToken_;
  edm::EDGetTokenT<reco::PFTauCollection>     tauProducerToken_;

  edm::EDGetTokenT<reco::PFCandidateCollection> pfToken_;
  edm::EDGetTokenT<reco::PFCandidateCollection> pfTmpBarrelToken_;

  edm::EDGetTokenT<std::vector<TICLCandidate>>       ticlCandidatesToken_;
  edm::EDGetTokenT<std::vector<ticl::Trackster>>     simTrackstersToken_;
  edm::EDGetTokenT<TracksterToTracksterMap>          allTrkToSimTrkAssocByLCsToken_;
  edm::EDGetTokenT<reco::PFJetCollection>            pfJetsToken_;
  edm::EDGetTokenT<reco::GenParticleCollection>      genParticlesToken_;

  // ---------- constants & helpers ----------
  static constexpr int   kMaxDM        = 16;
  static constexpr int   kMaxCHLegs    = 3; // charged hadrons per tau
  static constexpr int   kMaxGammaLegs = 4; // photons (1 pair = 1 pi0)
  static constexpr int   kNSteps       = 6; // 0..5

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

  struct StepHists {
    MonitorElement* cp_gen_pt[kNSteps]  {nullptr};
    MonitorElement* cp_gen_eta[kNSteps] {nullptr};
    MonitorElement* cp_gen_matched_pt[kNSteps]  {nullptr};
    MonitorElement* cp_gen_matched_eta[kNSteps] {nullptr};
  };

  // per-DM, per-leg step histos
  std::array<std::array<StepHists, kMaxCHLegs>,    kNDMSel> chStepHists_{};
  std::array<std::array<StepHists, kMaxGammaLegs>, kNDMSel> gammaStepHists_{};

  MonitorElement* dm_reco_vs_gen_jet_  = nullptr;
  MonitorElement* dm_reco_vs_gen_tau_  = nullptr;
  MonitorElement* dm_reco_vs_gen_hps_  = nullptr;

  MonitorElement* cp_chHad_pt_all_  = nullptr;
  MonitorElement* cp_chHad_eta_all_ = nullptr;
  MonitorElement* cp_gamma_pt_all_  = nullptr;
  MonitorElement* cp_gamma_eta_all_ = nullptr;

  // per-DM CP context histos
  MonitorElement* cp_chHad_pt_dm_[kNDMSel]  = {};
  MonitorElement* cp_chHad_eta_dm_[kNDMSel] = {};
  MonitorElement* cp_gamma_pt_dm_[kNDMSel]  = {};
  MonitorElement* cp_gamma_eta_dm_[kNDMSel] = {};

  // denominators
  MonitorElement* tau_gen_pt_[kNDMSel]  = {};
  MonitorElement* tau_gen_eta_[kNDMSel] = {};

  // numerators
  MonitorElement* tau_gen_matched_to_nCh_pt_[kNDMSel][kMaxCHLegs]     = {{}};
  MonitorElement* tau_gen_matched_to_nCh_eta_[kNDMSel][kMaxCHLegs]    = {{}};
  MonitorElement* tau_gen_matched_to_nPi0_pt_[kNDMSel][kMaxGammaLegs] = {{}};
  MonitorElement* tau_gen_matched_to_nPi0_eta_[kNDMSel][kMaxGammaLegs]= {{}};
  MonitorElement* tau_gen_matched_to_all_pt_[kNDMSel]  = {};
  MonitorElement* tau_gen_matched_to_all_eta_[kNDMSel] = {};

  // numerators split into signal and isolation (tau endpoint only)
  MonitorElement* tau_gen_matched_to_nCh_sig_pt_[kNDMSel][kMaxCHLegs]      = {{}};
  MonitorElement* tau_gen_matched_to_nCh_sig_eta_[kNDMSel][kMaxCHLegs]     = {{}};
  MonitorElement* tau_gen_matched_to_nCh_iso_pt_[kNDMSel][kMaxCHLegs]         = {{}};
  MonitorElement* tau_gen_matched_to_nCh_iso_eta_[kNDMSel][kMaxCHLegs]        = {{}};
  MonitorElement* tau_gen_matched_to_nPi0_sig_pt_[kNDMSel][kMaxGammaLegs]  = {{}};
  MonitorElement* tau_gen_matched_to_nPi0_sig_eta_[kNDMSel][kMaxGammaLegs] = {{}};
  MonitorElement* tau_gen_matched_to_nPi0_iso_pt_[kNDMSel][kMaxGammaLegs]     = {{}};
  MonitorElement* tau_gen_matched_to_nPi0_iso_eta_[kNDMSel][kMaxGammaLegs]    = {{}};

  MonitorElement* tau_pt_reco_over_gen_[kNDMSel] = {};

  // reco tau shapes per DM
  MonitorElement* tau_reco_pt_[kNDMSel]  = {};
  MonitorElement* tau_reco_eta_[kNDMSel] = {};

  // CP-to-PF resolution: 1D ratio histograms (PF pT / CP pT) per DM
  MonitorElement* cp_pf_pt_resolution_had_dm_[kNDMSel] = {};  // hadronic (charged hadrons)
  MonitorElement* cp_pf_pt_resolution_em_dm_[kNDMSel]  = {};  // electromagnetic (photons)

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
  simTrackstersToken_  = consumes<std::vector<ticl::Trackster>>( iConfig.getParameter<edm::InputTag>("simTracksters") );
  allTrkToSimTrkAssocByLCsToken_ = consumes<TracksterToTracksterMap>(
    iConfig.getParameter<edm::InputTag>("allTrackstersToSimTrackstersAssociationsByLCs") );
  pfJetsToken_       = consumes<reco::PFJetCollection>(        iConfig.getParameter<edm::InputTag>("jets") );
  genParticlesToken_ = consumes<reco::GenParticleCollection>(  iConfig.getParameter<edm::InputTag>("genParticles") );
}

void TICLTauValidator::bookHistograms(DQMStore::IBooker& ibook,
                                      edm::Run const&,
                                      edm::EventSetup const&) {
  ibook.setCurrentFolder(folder_);

  // Context CP histos
  cp_chHad_pt_all_  = ibook.book1D("cp_chHad_pt_all",  "Charged CP; pT [GeV]; entries", 60, 0., 120.);
  cp_chHad_eta_all_ = ibook.book1D("cp_chHad_eta_all", "Charged CP; eta; entries",       50, -3., 3.);
  cp_gamma_pt_all_  = ibook.book1D("cp_gamma_pt_all",  "Photon CP; pT [GeV]; entries",   60, 0., 120.);
  cp_gamma_eta_all_ = ibook.book1D("cp_gamma_eta_all", "Photon CP; eta; entries",        50, -3., 3.);

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

  // Confusion matrices: 6 reco bins (incl. DM 5), 5 gen bins (no DM 5)
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

  for (int dmI = 0; dmI < kNDMSel; ++dmI) {
    int dm = kDMSel[dmI];
    ibook.setCurrentFolder(folder_ + "/GenDM" + std::to_string(dm));

    const int chCap    = chCapForDM(dm);
    const int gammaCap = std::min(2 * expectedPi0ForDM(dm), kMaxGammaLegs);

    // per-DM CP base histos
    {
      std::ostringstream n, t;
      n << "cp_chHad_dm" << dm << "_pt";
      t << "Charged CP (DM=" << dm << "); pT [GeV]; entries";
      cp_chHad_pt_dm_[dmI] = ibook.book1D(n.str(), t.str(), 60, 0., 120.);
    }
    {
      std::ostringstream n, t;
      n << "cp_chHad_dm" << dm << "_eta";
      t << "Charged CP (DM=" << dm << "); eta; entries";
      cp_chHad_eta_dm_[dmI] = ibook.book1D(n.str(), t.str(), 50, -3., 3.);
    }
    {
      std::ostringstream n, t;
      n << "cp_gamma_dm" << dm << "_pt";
      t << "Photon CP (DM=" << dm << "); pT [GeV]; entries";
      cp_gamma_pt_dm_[dmI] = ibook.book1D(n.str(), t.str(), 60, 0., 120.);
    }
    {
      std::ostringstream n, t;
      n << "cp_gamma_dm" << dm << "_eta";
      t << "Photon CP (DM=" << dm << "); eta; entries";
      cp_gamma_eta_dm_[dmI] = ibook.book1D(n.str(), t.str(), 50, -3., 3.);
    }

    // CP-to-PF pT resolution per DM (ratio: PF pT / CP pT)
    {
      std::ostringstream n, t;
      n << "cp_pf_pt_resolution_hadronic_dm" << dm;
      t << "Charged hadron CP-to-PF pT resolution (DM=" << dm << ");pT^{reco}/pT^{gen};entries";
      cp_pf_pt_resolution_had_dm_[dmI] = ibook.book1D(n.str(), t.str(), 60, 0., 3.);
    }
    {
      std::ostringstream n, t;
      n << "cp_pf_pt_resolution_em_dm" << dm;
      t << "Photon CP-to-PF pT resolution (DM=" << dm << ");pT^{reco}/pT^{gen};entries";
      cp_pf_pt_resolution_em_dm_[dmI] = ibook.book1D(n.str(), t.str(), 60, 0., 3.);
    }

    // charged legs
    for (int li = 0; li < chCap; ++li) {
      for (int s : stepsToKeep) {
        {        
          std::ostringstream n, t; n << "ch_dm" << dm << "_leg" << li << "_step" << s << "_den_pt";
          t << "Den: charged; DM=" << dm << " leg=" << li << " step=" << s << "; pT (CP) [GeV]; entries";
          chStepHists_[dmI][li].cp_gen_pt[s] = ibook.book1D(n.str(), t.str(), 60, 0., 120.);
        }
        {
          std::ostringstream n, t; n << "ch_dm" << dm << "_leg" << li << "_step" << s << "_den_eta";
          t << "Den: charged; DM=" << dm << " leg=" << li << " step=" << s << "; eta (CP); entries";
          chStepHists_[dmI][li].cp_gen_eta[s] = ibook.book1D(n.str(), t.str(), 50, -3., 3.);
        }
        {
          std::ostringstream n, t; n << "ch_dm" << dm << "_leg" << li << "_step" << s << "_num_pt";
          t << "Num: charged; DM=" << dm << " leg=" << li << " step=" << s << "; pT (CP) [GeV]; entries";
          chStepHists_[dmI][li].cp_gen_matched_pt[s] = ibook.book1D(n.str(), t.str(), 60, 0., 120.);
        }
        {
          std::ostringstream n, t; n << "ch_dm" << dm << "_leg" << li << "_step" << s << "_num_eta";
          t << "Num: charged; DM=" << dm << " leg=" << li << " step=" << s << "; eta (CP); entries";
          chStepHists_[dmI][li].cp_gen_matched_eta[s] = ibook.book1D(n.str(), t.str(), 50, -3., 3.);
        }
      }
    }

    // photon legs
    for (int li = 0; li < gammaCap; ++li) {
      for (int s : stepsToKeep) {
        {
          std::ostringstream n, t; n << "pho_dm" << dm << "_leg" << li << "_step" << s << "_den_pt";
          t << "Den: photon; DM=" << dm << " leg=" << li << " step=" << s << "; pT (CP) [GeV]; entries";
          gammaStepHists_[dmI][li].cp_gen_pt[s] = ibook.book1D(n.str(), t.str(), 60, 0., 120.);
        }
        {
          std::ostringstream n, t; n << "pho_dm" << dm << "_leg" << li << "_step" << s << "_den_eta";
          t << "Den: photon; DM=" << dm << " leg=" << li << " step=" << s << "; eta (CP); entries";
          gammaStepHists_[dmI][li].cp_gen_eta[s] = ibook.book1D(n.str(), t.str(), 50, -3., 3.);
        }
        {
          std::ostringstream n, t; n << "pho_dm" << dm << "_leg" << li << "_step" << s << "_num_pt";
          t << "Num: photon; DM=" << dm << " leg=" << li << " step=" << s << "; pT (CP) [GeV]; entries";
          gammaStepHists_[dmI][li].cp_gen_matched_pt[s] = ibook.book1D(n.str(), t.str(), 60, 0., 120.);
        }
        {
          std::ostringstream n, t; n << "pho_dm" << dm << "_leg" << li << "_step" << s << "_num_eta";
          t << "Num: photon; DM=" << dm << " leg=" << li << " step=" << s << "; eta (CP); entries";
          gammaStepHists_[dmI][li].cp_gen_matched_eta[s] = ibook.book1D(n.str(), t.str(), 50, -3., 3.);
        }
      }
    }
  }

  // tau-level denominators & numerators
  for (int dmI = 0; dmI < kNDMSel; ++dmI) {
    const int dm     = kDMSel[dmI];
    const int chCap  = chCapForDM(dm);
    const int pi0Cap = pi0CapForDM(dm);
    ibook.setCurrentFolder(folder_ + "/GenDM" + std::to_string(dm));

    {
      std::ostringstream n1, t1; n1 << "tau_dm" << dm << "_den_pt";
      t1 << "DM " << dm << " gen tau; pT [GeV]; entries";
      tau_gen_pt_[dmI] = ibook.book1D(n1.str(), t1.str(), 60, 0., 120.);
    }
    {
      std::ostringstream n2, t2; n2 << "tau_dm" << dm << "_den_eta";
      t2 << "DM " << dm << " gen tau; eta; entries";
      tau_gen_eta_[dmI] = ibook.book1D(n2.str(), t2.str(), 50, -3., 3.);
    }
    {
      std::ostringstream n1, t1; n1 << "tau_dm" << dm << "_reco_pt";
      t1 << "DM " << dm << " reco tau; pT [GeV]; entries";
      tau_reco_pt_[dmI] = ibook.book1D(n1.str(), t1.str(), 60, 0., 120.);
    }
    {
      std::ostringstream n2, t2; n2 << "tau_dm" << dm << "_reco_eta";
      t2 << "DM " << dm << " reco tau; eta; entries";
      tau_reco_eta_[dmI] = ibook.book1D(n2.str(), t2.str(), 50, -3., 3.);
    }
    {
      std::ostringstream n, t;
      n << "tau_dm" << dm << "_pt_reco_over_gen";
      t << "DM " << dm << " tau: pT_reco/pT_gen; pT_reco/pT_gen; entries";
      tau_pt_reco_over_gen_[dmI] = ibook.book1D(n.str(), t.str(), 60, 0., 3.);
    } 
    // reco (jet/tau) endpoint: combined
    for (int N = 1; N <= chCap; ++N) {
      std::ostringstream npt, tpt, neta, teta;
      npt << "tau_dm" << dm << "_ge" << N << "ch_num_pt";
      tpt << "DM " << dm << " tau: >= " << N << " charged at reco; pT [GeV]; entries";
      neta << "tau_dm" << dm << "_ge" << N << "ch_num_eta";
      teta << "DM " << dm << " tau: >= " << N << " charged at reco; eta; entries";
      tau_gen_matched_to_nCh_pt_[dmI][N-1]  = ibook.book1D(npt.str(),  tpt.str(), 60, 0., 120.);
      tau_gen_matched_to_nCh_eta_[dmI][N-1] = ibook.book1D(neta.str(), teta.str(), 50, -3., 3.);
    }
    for (int N = 1; N <= pi0Cap; ++N) {
      std::ostringstream npt, tpt, neta, teta;
      npt << "tau_dm" << dm << "_ge" << N << "pi0_num_pt";
      tpt << "DM " << dm << " tau: >= " << N << " pi0 at reco; pT [GeV]; entries";
      neta << "tau_dm" << dm << "_ge" << N << "pi0_num_eta";
      teta << "DM " << dm << " tau: >= " << N << " pi0 at reco; eta; entries";
      tau_gen_matched_to_nPi0_pt_[dmI][N-1]  = ibook.book1D(npt.str(),  tpt.str(), 60, 0., 120.);
      tau_gen_matched_to_nPi0_eta_[dmI][N-1] = ibook.book1D(neta.str(), teta.str(), 50, -3., 3.);
    }
    {
      std::ostringstream npt, tpt, neta, teta;
      npt << "tau_dm" << dm << "_all_num_pt";
      tpt << "DM " << dm << " tau: all expected charged+pi0 at reco; pT [GeV]; entries";
      neta << "tau_dm" << dm << "_all_num_eta";
      teta << "DM " << dm << " tau: all expected charged+pi0 at reco; eta; entries";
      tau_gen_matched_to_all_pt_[dmI]  = ibook.book1D(npt.str(),  tpt.str(), 60, 0., 120.);
      tau_gen_matched_to_all_eta_[dmI] = ibook.book1D(neta.str(), teta.str(), 50, -3., 3.);
    }

    // TAU-only endpoint: signal vs iso
    for (int N = 1; N <= chCap; ++N) {
      std::ostringstream nspt, tspt, nseta, tseta;
      std::ostringstream nipt, tipt, nieta, tieta;

      nspt << "tau_dm" << dm << "_ge" << N << "ch_num_signal_pt";
      tspt << "DM " << dm << " tau: >= " << N << " charged in signal; pT [GeV]; entries";
      nseta << "tau_dm" << dm << "_ge" << N << "ch_num_signal_eta";
      tseta << "DM " << dm << " tau: >= " << N << " charged in signal; eta; entries";

      nipt << "tau_dm" << dm << "_ge" << N << "ch_num_iso_pt";
      tipt << "DM " << dm << " tau: >= " << N << " charged in isolation; pT [GeV]; entries";
      nieta << "tau_dm" << dm << "_ge" << N << "ch_num_iso_eta";
      tieta << "DM " << dm << " tau: >= " << N << " charged in isolation; eta; entries";

      tau_gen_matched_to_nCh_sig_pt_[dmI][N-1]  = ibook.book1D(nspt.str(),  tspt.str(), 60, 0., 120.);
      tau_gen_matched_to_nCh_sig_eta_[dmI][N-1] = ibook.book1D(nseta.str(), tseta.str(), 50, -3., 3.);
      tau_gen_matched_to_nCh_iso_pt_[dmI][N-1]     = ibook.book1D(nipt.str(),  tipt.str(), 60, 0., 120.);
      tau_gen_matched_to_nCh_iso_eta_[dmI][N-1]    = ibook.book1D(nieta.str(), tieta.str(), 50, -3., 3.);
    }
    for (int N = 1; N <= pi0Cap; ++N) {
      std::ostringstream nspt, tspt, nseta, tseta;
      std::ostringstream nipt, tipt, nieta, tieta;

      nspt << "tau_dm" << dm << "_ge" << N << "pi0_num_signal_pt";
      tspt << "DM " << dm << " tau: >= " << N << " pi0 in signal; pT [GeV]; entries";
      nseta << "tau_dm" << dm << "_ge" << N << "pi0_num_signal_eta";
      tseta << "DM " << dm << " tau: >= " << N << " pi0 in signal; eta; entries";

      nipt << "tau_dm" << dm << "_ge" << N << "pi0_num_iso_pt";
      tipt << "DM " << dm << " tau: >= " << N << " pi0 in isolation; pT [GeV]; entries";
      nieta << "tau_dm" << dm << "_ge" << N << "pi0_num_iso_eta";
      tieta << "DM " << dm << " tau: >= " << N << " pi0 in isolation; eta; entries";

      tau_gen_matched_to_nPi0_sig_pt_[dmI][N-1]  = ibook.book1D(nspt.str(),  tspt.str(), 60, 0., 120.);
      tau_gen_matched_to_nPi0_sig_eta_[dmI][N-1] = ibook.book1D(nseta.str(), tseta.str(), 50, -3., 3.);
      tau_gen_matched_to_nPi0_iso_pt_[dmI][N-1]     = ibook.book1D(nipt.str(),  tipt.str(), 60, 0., 120.);
      tau_gen_matched_to_nPi0_iso_eta_[dmI][N-1]    = ibook.book1D(nieta.str(), tieta.str(), 50, -3., 3.);
    }

  }
}

void TICLTauValidator::analyze(const edm::Event& iEvent,
                               const edm::EventSetup&) {

  // handles
  edm::Handle<std::vector<SimTauCPLink>> simTaus;            iEvent.getByToken(simTauToken_, simTaus);
  edm::Handle<reco::PFTauCollection>     taus;               iEvent.getByToken(tauProducerToken_, taus);
  edm::Handle<std::vector<ticl::Trackster>> simTracksters;   iEvent.getByToken(simTrackstersToken_, simTracksters);
  edm::Handle<TracksterToTracksterMap>   simToRecoMap;       iEvent.getByToken(allTrkToSimTrkAssocByLCsToken_, simToRecoMap);
  edm::Handle<std::vector<TICLCandidate>> ticlCandidates;    iEvent.getByToken(ticlCandidatesToken_, ticlCandidates);
  edm::Handle<reco::PFCandidateCollection> pfMerged;         iEvent.getByToken(pfToken_, pfMerged);
  edm::Handle<reco::PFCandidateCollection> pfTmpBarrel;      iEvent.getByToken(pfTmpBarrelToken_, pfTmpBarrel);
  edm::Handle<reco::PFJetCollection>      pfJets;            iEvent.getByToken(pfJetsToken_, pfJets);
  edm::Handle<reco::GenParticleCollection> genParticles;     iEvent.getByToken(genParticlesToken_, genParticles);

  if (!simTaus.isValid())
    return;

  std::vector<std::string> eventChains;
  std::set<unsigned int> seenCPChargedEvent, seenCPPhotonEvent;

  for (const auto& link : *simTaus) {
    const int dmPhys = link.decayMode;
    const int dmIdx  = dmToSelIndex(dmPhys);
    if (dmIdx < 0)
      continue; // ignore non-selected DMs

    struct PtEta { float pt=0.f, eta=0.f; };
    struct TauRegionFlags { bool signal=false, isolation=false; };
    struct PendingInfo {
      float cpPt=0.f, cpEta=0.f;
      float pfPt=0.f, pfEta=0.f;
      int   pdgId=0; 
      bool  hasCPKinematics=false;
      bool  hasPFKinematics=false;
      std::array<bool, kNSteps> stepPass{};
      std::set<size_t> jets;
      std::unordered_map<size_t, std::vector<int>> dmByJet;
      std::unordered_map<size_t, TauRegionFlags>   tauRegion;
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
      const bool isChargedHadron = (absPdg == 211); // || absPdg == 321 || absPdg == 2212);

      // context CP histos (once per event per CP key) + per-DM base histos
      if (isChargedHadron) {
        const bool firstTime = seenCPChargedEvent.insert(cpRef.key()).second;
        if (firstTime) {
          if (cp_chHad_pt_all_)  cp_chHad_pt_all_->Fill(cp.pt());
          if (cp_chHad_eta_all_) cp_chHad_eta_all_->Fill(cp.eta());
          if (dmIdx >= 0) {
            if (cp_chHad_pt_dm_[dmIdx])  cp_chHad_pt_dm_[dmIdx]->Fill(cp.pt());
            if (cp_chHad_eta_dm_[dmIdx]) cp_chHad_eta_dm_[dmIdx]->Fill(cp.eta());
          }
        }
      }
      if (isPhoton) {
        const bool firstTime = seenCPPhotonEvent.insert(cpRef.key()).second;
        if (firstTime) {
          if (cp_gamma_pt_all_)  cp_gamma_pt_all_->Fill(cp.pt());
          if (cp_gamma_eta_all_) cp_gamma_eta_all_->Fill(cp.eta());
          if (dmIdx >= 0) {
            if (cp_gamma_pt_dm_[dmIdx])  cp_gamma_pt_dm_[dmIdx]->Fill(cp.pt());
            if (cp_gamma_eta_dm_[dmIdx]) cp_gamma_eta_dm_[dmIdx]->Fill(cp.eta());
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
        const size_t barrelSize = pfTmpBarrel.isValid() ? pfTmpBarrel->size() : 0;
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
            if (isChargedHadron && dmIdx >= 0 && cp_pf_pt_resolution_had_dm_[dmIdx]) {
              cp_pf_pt_resolution_had_dm_[dmIdx]->Fill(ratio);
            }
            if (isPhoton && dmIdx >= 0 && cp_pf_pt_resolution_em_dm_[dmIdx]) {
              cp_pf_pt_resolution_em_dm_[dmIdx]->Fill(ratio);
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

      std::ostringstream ss;
      ss << "Chain: CP=" << cpRef.key() << " pdgId=" << cp.pdgId()
          << " steps=[";
      for (int s = 0; s < kNSteps; ++s)
        ss << (pend.stepPass[s] ? '1' : '0');
      ss << "] jets=[";
      size_t c = 0;
      for (auto j : pend.jets) {
        if (c++) ss << ",";
        ss << j;
      }
      ss << "]";
      eventChains.push_back(ss.str());
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
      const int gSel = dmIdx;
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

    const int nPi0_tau_total   = photonsUsed_tau_total   / 2;
    const int nPi0_signal_tau  = photonsUsed_signal_tau  / 2;
    const int nPi0_iso_tau     = photonsUsed_iso_tau     / 2;

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
      const int gSel = dmIdx;
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
        int gSel  = dmIdx;
        if (rSel >= 0 && gSel >= 0)
          dm_reco_vs_gen_hps_->Fill(rSel, gSel);
      }
    }

    // helper to fill per-leg step histos
    auto fillLegStepsDM = [&](bool isCharged,
                              int dmI,
                              int legIdx,
                              float cpPt, float cpEta,
                              const std::array<bool, kNSteps>& pass){
      if (isCharged) {
        for (int s = 0; s < kNSteps; ++s) {
          if (auto* h = chStepHists_[dmI][legIdx].cp_gen_pt[s])  h->Fill(cpPt);
          if (auto* h = chStepHists_[dmI][legIdx].cp_gen_eta[s]) h->Fill(cpEta);
          if (pass[s]) {
            if (auto* h = chStepHists_[dmI][legIdx].cp_gen_matched_pt[s])  h->Fill(cpPt);
            if (auto* h = chStepHists_[dmI][legIdx].cp_gen_matched_eta[s]) h->Fill(cpEta);
          }
        }
      } else {
        for (int s = 0; s < kNSteps; ++s) {
          if (auto* h = gammaStepHists_[dmI][legIdx].cp_gen_pt[s])  h->Fill(cpPt);
          if (auto* h = gammaStepHists_[dmI][legIdx].cp_gen_eta[s]) h->Fill(cpEta);
          if (pass[s]) {
            if (auto* h = gammaStepHists_[dmI][legIdx].cp_gen_matched_pt[s])  h->Fill(cpPt);
            if (auto* h = gammaStepHists_[dmI][legIdx].cp_gen_matched_eta[s]) h->Fill(cpEta);
          }
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
        fillLegStepsDM(true, dmIdx, li, pend.cpPt, pend.cpEta, pend.stepPass);
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
        fillLegStepsDM(false, dmIdx, li, pend.cpPt, pend.cpEta, pend.stepPass);
        ++li;
      }
    }

    // gen-tau for tau-level plots
    double tauPt = 0., tauEta = 0.;
    const reco::GenParticle* bestMotherTau = nullptr;
    double bestMotherPt = -1.;
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
            }
            break;
          }
          cur = mom;
        }
      }
    }
    if (bestMotherTau) {
      tauPt  = bestMotherTau->pt();
      tauEta = bestMotherTau->eta();
    }

    const int expCh  = expectedChForDM(dmPhys);
    const int expPi0 = expectedPi0ForDM(dmPhys);

    const int nChTicl  = static_cast<int>(pendingHad.size());
    const int nPhoTicl = static_cast<int>(pendingGamma.size());

    const bool tauInAcceptance =
      (expCh  == 0 || nChTicl  >= expCh) &&
      (expPi0 == 0 || nPhoTicl >= 2 * expPi0);

    // Fill tau-level denominators only for taus with enough TICL legs
    if (bestMotherTau && tauInAcceptance) {
      if (tau_gen_pt_[dmIdx])  tau_gen_pt_[dmIdx]->Fill(tauPt);
      if (tau_gen_eta_[dmIdx]) tau_gen_eta_[dmIdx]->Fill(tauEta);
    }

    // tau-level numerators
    if (bestMotherTau && tauInAcceptance) {
      const int chCap  = chCapForDM(dmPhys);
      const int p0Cap  = pi0CapForDM(dmPhys);

      const int nGoodCH_endpoint = haveHpsTaus ? nGoodCH_tau_total : nGoodCH_jet;
      const int nPi0_endpoint    = haveHpsTaus ? nPi0_tau_total    : nPi0_jet;

      // >= N charged legs at reco
      for (int N = 1; N <= chCap; ++N) {
        if (nGoodCH_endpoint >= N) {
          if (auto* h = tau_gen_matched_to_nCh_pt_[dmIdx][N-1])  h->Fill(tauPt);
          if (auto* h = tau_gen_matched_to_nCh_eta_[dmIdx][N-1]) h->Fill(tauEta);
        }
      }

      // >= N pi0 at reco
      for (int N = 1; N <= p0Cap; ++N) {
        if (nPi0_endpoint >= N) {
          if (auto* h = tau_gen_matched_to_nPi0_pt_[dmIdx][N-1])  h->Fill(tauPt);
          if (auto* h = tau_gen_matched_to_nPi0_eta_[dmIdx][N-1]) h->Fill(tauEta);
        }
      }

      // ALL expected legs
      if (expCh > 0 || expPi0 > 0) {
        if (nGoodCH_endpoint >= expCh && nPi0_endpoint >= expPi0) {
          if (tau_gen_matched_to_all_pt_[dmIdx])  tau_gen_matched_to_all_pt_[dmIdx]->Fill(tauPt);
          if (tau_gen_matched_to_all_eta_[dmIdx]) tau_gen_matched_to_all_eta_[dmIdx]->Fill(tauEta);
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
            if (auto* h = tau_gen_matched_to_nCh_sig_pt_[dmIdx][N-1])  h->Fill(tauPt);
            if (auto* h = tau_gen_matched_to_nCh_sig_eta_[dmIdx][N-1]) h->Fill(tauEta);
          }
          if (nGoodCH_iso >= N) {
            if (auto* h = tau_gen_matched_to_nCh_iso_pt_[dmIdx][N-1])  h->Fill(tauPt);
            if (auto* h = tau_gen_matched_to_nCh_iso_eta_[dmIdx][N-1]) h->Fill(tauEta);
          }
        }
        for (int N = 1; N <= p0Cap; ++N) {
          if (nPi0_signal >= N) {
            if (auto* h = tau_gen_matched_to_nPi0_sig_pt_[dmIdx][N-1])  h->Fill(tauPt);
            if (auto* h = tau_gen_matched_to_nPi0_sig_eta_[dmIdx][N-1]) h->Fill(tauEta);
          }
          if (nPi0_iso >= N) {
            if (auto* h = tau_gen_matched_to_nPi0_iso_pt_[dmIdx][N-1])  h->Fill(tauPt);
            if (auto* h = tau_gen_matched_to_nPi0_iso_eta_[dmIdx][N-1]) h->Fill(tauEta);
          }
        }
        // tau resolution
        if (tauPt > 0.) {
          if (bestTauIdxForLink >= 0 && taus.isValid()) {
            const auto& tau = (*taus)[bestTauIdxForLink];
            const double respTau = tau.pt() / tauPt;  // reco / gen
            if (tau_pt_reco_over_gen_[dmIdx]) {
              tau_pt_reco_over_gen_[dmIdx]->Fill(respTau);
            }
            // reco tau shapes per DM
            if (tau_reco_pt_[dmIdx])  tau_reco_pt_[dmIdx]->Fill(tau.pt());
            if (tau_reco_eta_[dmIdx]) tau_reco_eta_[dmIdx]->Fill(tau.eta());
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

  edm::LogVerbatim("TICLTauValidator")
    << "[TICLTauValidator] processed event " << iEvent.id();
  for (const auto& s : eventChains) {
    edm::LogVerbatim("TICLTauValidator") << "  " << s;
  }
}

void TICLTauValidator::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<std::string>("folder", "RecoTauV/ticlTauValidator");

  desc.add<edm::InputTag>("simTaus", edm::InputTag("SimTauProducer"));
  desc.add<edm::InputTag>("TauProducer", edm::InputTag("hpsPFTauProducer"));
  desc.add<edm::InputTag>("pf", edm::InputTag("particleFlow"));
  desc.add<edm::InputTag>("pfTmpBarrel", edm::InputTag("particleFlowTmpBarrel"));
  desc.add<edm::InputTag>("jets", edm::InputTag("ak4PFJets"));
  desc.add<edm::InputTag>("ticlCandidates", edm::InputTag("ticlCandidate"));
  desc.add<edm::InputTag>("simTracksters", edm::InputTag("ticlSimTracksters", "fromCPs"));
  desc.add<edm::InputTag>("allTrackstersToSimTrackstersAssociationsByLCs",
                          edm::InputTag("allTrackstersToSimTrackstersAssociationsByLCs",
                                        "ticlSimTrackstersfromCPsToticlCandidate"));
  desc.add<edm::InputTag>("genParticles", edm::InputTag("genParticles"));
  desc.add<double>("maxAssocScore", 0.6);
  desc.add<double>("hgcalEtaAbsMin", 1.5);

  descriptions.add("ticlTauValidator", desc);
}

DEFINE_FWK_MODULE(TICLTauValidator);
