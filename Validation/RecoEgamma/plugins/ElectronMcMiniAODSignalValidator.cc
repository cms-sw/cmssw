// user include files
#include "Validation/RecoEgamma/plugins/ElectronMcMiniAODSignalValidator.h"
#include "CLHEP/Units/GlobalPhysicalConstants.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

// user include files

using namespace reco;
using namespace pat;

ElectronMcSignalValidatorMiniAOD::ElectronMcSignalValidatorMiniAOD(const edm::ParameterSet& iConfig)
    : ElectronDqmAnalyzerBase(iConfig) {
  mcTruthCollection_ = consumes<edm::View<reco::GenParticle> >(
      iConfig.getParameter<edm::InputTag>("mcTruthCollection"));  // prunedGenParticles
  electronToken_ =
      consumes<pat::ElectronCollection>(iConfig.getParameter<edm::InputTag>("electrons"));  // slimmedElectrons
  electronTokenEndcaps_ =
      consumes<pat::ElectronCollection>(iConfig.getParameter<edm::InputTag>("electrons_endcaps"));  // slimmedElectrons

  edm::ParameterSet histosSet = iConfig.getParameter<edm::ParameterSet>("histosCfg");
  edm::ParameterSet isolationSet = iConfig.getParameter<edm::ParameterSet>("isolationCfg");

  maxPt_ = iConfig.getParameter<double>("MaxPt");
  maxAbsEta_ = iConfig.getParameter<double>("MaxAbsEta");
  deltaR_ = iConfig.getParameter<double>("DeltaR");
  deltaR2_ = deltaR_ * deltaR_;
  matchingIDs_ = iConfig.getParameter<std::vector<int> >("MatchingID");
  matchingMotherIDs_ = iConfig.getParameter<std::vector<int> >("MatchingMotherID");
  outputInternalPath_ = iConfig.getParameter<std::string>("OutputFolderName");

  // histos bining and limits

  xyz_nbin = histosSet.getParameter<int>("Nbinxyz");

  pt_nbin = histosSet.getParameter<int>("Nbinpt");
  pt2D_nbin = histosSet.getParameter<int>("Nbinpt2D");
  pteff_nbin = histosSet.getParameter<int>("Nbinpteff");
  pt_max = histosSet.getParameter<double>("Ptmax");

  fhits_nbin = histosSet.getParameter<int>("Nbinfhits");
  fhits_max = histosSet.getParameter<double>("Fhitsmax");

  eta_nbin = histosSet.getParameter<int>("Nbineta");
  eta2D_nbin = histosSet.getParameter<int>("Nbineta2D");
  eta_min = histosSet.getParameter<double>("Etamin");
  eta_max = histosSet.getParameter<double>("Etamax");

  detamatch_nbin = histosSet.getParameter<int>("Nbindetamatch");
  detamatch2D_nbin = histosSet.getParameter<int>("Nbindetamatch2D");
  detamatch_min = histosSet.getParameter<double>("Detamatchmin");
  detamatch_max = histosSet.getParameter<double>("Detamatchmax");

  dphi_nbin = histosSet.getParameter<int>("Nbindphi");
  dphi_min = histosSet.getParameter<double>("Dphimin");
  dphi_max = histosSet.getParameter<double>("Dphimax");

  dphimatch_nbin = histosSet.getParameter<int>("Nbindphimatch");
  dphimatch2D_nbin = histosSet.getParameter<int>("Nbindphimatch2D");
  dphimatch_min = histosSet.getParameter<double>("Dphimatchmin");
  dphimatch_max = histosSet.getParameter<double>("Dphimatchmax");

  hoe_nbin = histosSet.getParameter<int>("Nbinhoe");
  hoe_min = histosSet.getParameter<double>("Hoemin");
  hoe_max = histosSet.getParameter<double>("Hoemax");

  mee_nbin = histosSet.getParameter<int>("Nbinmee");
  mee_min = histosSet.getParameter<double>("Meemin");
  mee_max = histosSet.getParameter<double>("Meemax");

  poptrue_nbin = histosSet.getParameter<int>("Nbinpoptrue");
  poptrue_min = histosSet.getParameter<double>("Poptruemin");
  poptrue_max = histosSet.getParameter<double>("Poptruemax");

  set_EfficiencyFlag = histosSet.getParameter<bool>("EfficiencyFlag");
  set_StatOverflowFlag = histosSet.getParameter<bool>("StatOverflowFlag");

  ele_nbin = histosSet.getParameter<int>("NbinELE");
  ele_min = histosSet.getParameter<double>("ELE_min");
  ele_max = histosSet.getParameter<double>("ELE_max");

  // so to please coverity...

  h1_recEleNum = nullptr;

  h1_ele_vertexPt = nullptr;
  h1_ele_vertexEta = nullptr;
  h1_ele_vertexPt_nocut = nullptr;

  h1_scl_SigIEtaIEta_mAOD = nullptr;
  h1_scl_SigIEtaIEta_mAOD_barrel = nullptr;
  h1_scl_SigIEtaIEta_mAOD_endcaps = nullptr;

  h2_ele_foundHitsVsEta = nullptr;
  h2_ele_foundHitsVsEta_mAOD = nullptr;

  h2_ele_PoPtrueVsEta = nullptr;
  h2_ele_sigmaIetaIetaVsPt = nullptr;

  h1_ele_HoE_mAOD = nullptr;
  h1_ele_HoE_mAOD_barrel = nullptr;
  h1_ele_HoE_mAOD_endcaps = nullptr;
  h1_ele_mee_all = nullptr;
  h1_ele_mee_os = nullptr;

  h1_ele_fbrem_mAOD = nullptr;
  h1_ele_fbrem_mAOD_barrel = nullptr;
  h1_ele_fbrem_mAOD_endcaps = nullptr;

  h1_ele_dEtaSc_propVtx_mAOD = nullptr;
  h1_ele_dEtaSc_propVtx_mAOD_barrel = nullptr;
  h1_ele_dEtaSc_propVtx_mAOD_endcaps = nullptr;
  h1_ele_dPhiCl_propOut_mAOD = nullptr;
  h1_ele_dPhiCl_propOut_mAOD_barrel = nullptr;
  h1_ele_dPhiCl_propOut_mAOD_endcaps = nullptr;

  h1_ele_chargedHadronRelativeIso_mAOD = nullptr;
  h1_ele_chargedHadronRelativeIso_mAOD_barrel = nullptr;
  h1_ele_chargedHadronRelativeIso_mAOD_endcaps = nullptr;
  h1_ele_neutralHadronRelativeIso_mAOD = nullptr;
  h1_ele_neutralHadronRelativeIso_mAOD_barrel = nullptr;
  h1_ele_neutralHadronRelativeIso_mAOD_endcaps = nullptr;
  h1_ele_photonRelativeIso_mAOD = nullptr;
  h1_ele_photonRelativeIso_mAOD_barrel = nullptr;
  h1_ele_photonRelativeIso_mAOD_endcaps = nullptr;
}

ElectronMcSignalValidatorMiniAOD::~ElectronMcSignalValidatorMiniAOD() {}

void ElectronMcSignalValidatorMiniAOD::bookHistograms(DQMStore::IBooker& iBooker,
                                                      edm::Run const&,
                                                      edm::EventSetup const&) {
  iBooker.setCurrentFolder(outputInternalPath_);

  setBookIndex(-1);
  setBookPrefix("h");
  setBookEfficiencyFlag(set_EfficiencyFlag);
  setBookStatOverflowFlag(set_StatOverflowFlag);

  // rec event collections sizes
  h1_recEleNum = bookH1(iBooker, "recEleNum", "# rec electrons", ele_nbin, ele_min, ele_max, "N_{ele}");
  // matched electrons
  setBookPrefix("h_mc");
  setBookPrefix("h_ele");
  h1_ele_vertexPt =
      bookH1withSumw2(iBooker, "vertexPt", "ele transverse momentum", pt_nbin, 0., pt_max, "p_{T vertex} (GeV/c)");
  h1_ele_vertexEta = bookH1withSumw2(iBooker, "vertexEta", "ele momentum eta", eta_nbin, eta_min, eta_max, "#eta");
  h1_ele_vertexPt_nocut =
      bookH1withSumw2(iBooker, "vertexPt_nocut", "pT of prunned electrons", pt_nbin, 0., 20., "p_{T vertex} (GeV/c)");
  h2_ele_PoPtrueVsEta = bookH2withSumw2(iBooker,
                                        "PoPtrueVsEta",
                                        "ele momentum / gen momentum vs eta",
                                        eta2D_nbin,
                                        eta_min,
                                        eta_max,
                                        50,
                                        poptrue_min,
                                        poptrue_max);
  h2_ele_sigmaIetaIetaVsPt =
      bookH2(iBooker, "sigmaIetaIetaVsPt", "SigmaIetaIeta vs pt", 100, 0., pt_max, 100, 0., 0.05);

  // all electrons
  setBookPrefix("h_ele");
  h1_ele_mee_all = bookH1withSumw2(iBooker,
                                   "mee_all",
                                   "ele pairs invariant mass, all reco electrons",
                                   mee_nbin,
                                   mee_min,
                                   mee_max,
                                   "m_{ee} (GeV/c^{2})",
                                   "Events",
                                   "ELE_LOGY E1 P");
  h1_ele_mee_os = bookH1withSumw2(iBooker,
                                  "mee_os",
                                  "ele pairs invariant mass, opp. sign",
                                  mee_nbin,
                                  mee_min,
                                  mee_max,
                                  "m_{e^{+}e^{-}} (GeV/c^{2})",
                                  "Events",
                                  "ELE_LOGY E1 P");

  // matched electron, superclusters
  setBookPrefix("h_scl");
  h1_scl_SigIEtaIEta_mAOD = bookH1withSumw2(iBooker,
                                            "SigIEtaIEta_mAOD",
                                            "ele supercluster sigma ieta ieta",
                                            100,
                                            0.,
                                            0.05,
                                            "#sigma_{i#eta i#eta}",
                                            "Events",
                                            "ELE_LOGY E1 P");
  h1_scl_SigIEtaIEta_mAOD_barrel = bookH1withSumw2(iBooker,
                                                   "SigIEtaIEta_mAOD_barrel",
                                                   "ele supercluster sigma ieta ieta, barrel",
                                                   100,
                                                   0.,
                                                   0.05,
                                                   "#sigma_{i#eta i#eta}",
                                                   "Events",
                                                   "ELE_LOGY E1 P");
  h1_scl_SigIEtaIEta_mAOD_endcaps = bookH1withSumw2(iBooker,
                                                    "SigIEtaIEta_mAOD_endcaps",
                                                    "ele supercluster sigma ieta ieta, endcaps",
                                                    100,
                                                    0.,
                                                    0.05,
                                                    "#sigma_{i#eta i#eta}",
                                                    "Events",
                                                    "ELE_LOGY E1 P");

  // matched electron, gsf tracks
  setBookPrefix("h_ele");
  h2_ele_foundHitsVsEta = bookH2(iBooker,
                                 "foundHitsVsEta",
                                 "ele track # found hits vs eta",
                                 eta2D_nbin,
                                 eta_min,
                                 eta_max,
                                 fhits_nbin,
                                 0.,
                                 fhits_max);
  h2_ele_foundHitsVsEta_mAOD = bookH2(iBooker,
                                      "foundHitsVsEta_mAOD",
                                      "ele track # found hits vs eta",
                                      eta2D_nbin,
                                      eta_min,
                                      eta_max,
                                      fhits_nbin,
                                      0.,
                                      fhits_max);

  // matched electrons, matching
  setBookPrefix("h_ele");
  h1_ele_HoE_mAOD = bookH1withSumw2(iBooker,
                                    "HoE_mAOD",
                                    "ele hadronic energy / em energy",
                                    hoe_nbin,
                                    hoe_min,
                                    hoe_max,
                                    "H/E",
                                    "Events",
                                    "ELE_LOGY E1 P");
  h1_ele_HoE_mAOD_barrel = bookH1withSumw2(iBooker,
                                           "HoE_mAOD_barrel",
                                           "ele hadronic energy / em energy, barrel",
                                           hoe_nbin,
                                           hoe_min,
                                           hoe_max,
                                           "H/E",
                                           "Events",
                                           "ELE_LOGY E1 P");
  h1_ele_HoE_mAOD_endcaps = bookH1withSumw2(iBooker,
                                            "HoE_mAOD_endcaps",
                                            "ele hadronic energy / em energy, endcaps",
                                            hoe_nbin,
                                            hoe_min,
                                            hoe_max,
                                            "H/E",
                                            "Events",
                                            "ELE_LOGY E1 P");
  h1_ele_dEtaSc_propVtx_mAOD = bookH1withSumw2(iBooker,
                                               "dEtaSc_propVtx_mAOD",
                                               "ele #eta_{sc} - #eta_{tr}, prop from vertex",
                                               detamatch_nbin,
                                               detamatch_min,
                                               detamatch_max,
                                               "#eta_{sc} - #eta_{tr}",
                                               "Events",
                                               "ELE_LOGY E1 P");
  h1_ele_dEtaSc_propVtx_mAOD_barrel = bookH1withSumw2(iBooker,
                                                      "dEtaSc_propVtx_mAOD_barrel",
                                                      "ele #eta_{sc} - #eta_{tr}, prop from vertex, barrel",
                                                      detamatch_nbin,
                                                      detamatch_min,
                                                      detamatch_max,
                                                      "#eta_{sc} - #eta_{tr}",
                                                      "Events",
                                                      "ELE_LOGY E1 P");
  h1_ele_dEtaSc_propVtx_mAOD_endcaps = bookH1withSumw2(iBooker,
                                                       "dEtaSc_propVtx_mAOD_endcaps",
                                                       "ele #eta_{sc} - #eta_{tr}, prop from vertex, endcaps",
                                                       detamatch_nbin,
                                                       detamatch_min,
                                                       detamatch_max,
                                                       "#eta_{sc} - #eta_{tr}",
                                                       "Events",
                                                       "ELE_LOGY E1 P");
  h1_ele_dPhiCl_propOut_mAOD = bookH1withSumw2(iBooker,
                                               "dPhiCl_propOut_mAOD",
                                               "ele #phi_{cl} - #phi_{tr}, prop from outermost",
                                               dphimatch_nbin,
                                               dphimatch_min,
                                               dphimatch_max,
                                               "#phi_{seedcl} - #phi_{tr} (rad)",
                                               "Events",
                                               "ELE_LOGY E1 P");
  h1_ele_dPhiCl_propOut_mAOD_barrel = bookH1withSumw2(iBooker,
                                                      "dPhiCl_propOut_mAOD_barrel",
                                                      "ele #phi_{cl} - #phi_{tr}, prop from outermost, barrel",
                                                      dphimatch_nbin,
                                                      dphimatch_min,
                                                      dphimatch_max,
                                                      "#phi_{seedcl} - #phi_{tr} (rad)",
                                                      "Events",
                                                      "ELE_LOGY E1 P");
  h1_ele_dPhiCl_propOut_mAOD_endcaps = bookH1withSumw2(iBooker,
                                                       "dPhiCl_propOut_mAOD_endcaps",
                                                       "ele #phi_{cl} - #phi_{tr}, prop from outermost, endcaps",
                                                       dphimatch_nbin,
                                                       dphimatch_min,
                                                       dphimatch_max,
                                                       "#phi_{seedcl} - #phi_{tr} (rad)",
                                                       "Events",
                                                       "ELE_LOGY E1 P");

  // fbrem
  h1_ele_fbrem_mAOD = bookH1withSumw2(
      iBooker, "fbrem_mAOD", "ele brem fraction, mode of GSF components", 100, 0., 1., "P_{in} - P_{out} / P_{in}");
  h1_ele_fbrem_mAOD_barrel = bookH1withSumw2(iBooker,
                                             "fbrem_mAOD_barrel",
                                             "ele brem fraction for barrel, mode of GSF components",
                                             100,
                                             0.,
                                             1.,
                                             "P_{in} - P_{out} / P_{in}");
  h1_ele_fbrem_mAOD_endcaps = bookH1withSumw2(iBooker,
                                              "fbrem_mAOD_endcaps",
                                              "ele brem franction for endcaps, mode of GSF components",
                                              100,
                                              0.,
                                              1.,
                                              "P_{in} - P_{out} / P_{in}");

  // -- pflow over pT
  h1_ele_chargedHadronRelativeIso_mAOD = bookH1withSumw2(iBooker,
                                                         "chargedHadronRelativeIso_mAOD",
                                                         "chargedHadronRelativeIso",
                                                         100,
                                                         0.0,
                                                         2.,
                                                         "chargedHadronRelativeIso",
                                                         "Events",
                                                         "ELE_LOGY E1 P");
  h1_ele_chargedHadronRelativeIso_mAOD_barrel = bookH1withSumw2(iBooker,
                                                                "chargedHadronRelativeIso_mAOD_barrel",
                                                                "chargedHadronRelativeIso for barrel",
                                                                100,
                                                                0.0,
                                                                2.,
                                                                "chargedHadronRelativeIso_barrel",
                                                                "Events",
                                                                "ELE_LOGY E1 P");
  h1_ele_chargedHadronRelativeIso_mAOD_endcaps = bookH1withSumw2(iBooker,
                                                                 "chargedHadronRelativeIso_mAOD_endcaps",
                                                                 "chargedHadronRelativeIso for endcaps",
                                                                 100,
                                                                 0.0,
                                                                 2.,
                                                                 "chargedHadronRelativeIso_endcaps",
                                                                 "Events",
                                                                 "ELE_LOGY E1 P");
  h1_ele_neutralHadronRelativeIso_mAOD = bookH1withSumw2(iBooker,
                                                         "neutralHadronRelativeIso_mAOD",
                                                         "neutralHadronRelativeIso",
                                                         100,
                                                         0.0,
                                                         2.,
                                                         "neutralHadronRelativeIso",
                                                         "Events",
                                                         "ELE_LOGY E1 P");
  h1_ele_neutralHadronRelativeIso_mAOD_barrel = bookH1withSumw2(iBooker,
                                                                "neutralHadronRelativeIso_mAOD_barrel",
                                                                "neutralHadronRelativeIso for barrel",
                                                                100,
                                                                0.0,
                                                                2.,
                                                                "neutralHadronRelativeIso_barrel",
                                                                "Events",
                                                                "ELE_LOGY E1 P");
  h1_ele_neutralHadronRelativeIso_mAOD_endcaps = bookH1withSumw2(iBooker,
                                                                 "neutralHadronRelativeIso_mAOD_endcaps",
                                                                 "neutralHadronRelativeIso for endcaps",
                                                                 100,
                                                                 0.0,
                                                                 2.,
                                                                 "neutralHadronRelativeIso_endcaps",
                                                                 "Events",
                                                                 "ELE_LOGY E1 P");
  h1_ele_photonRelativeIso_mAOD = bookH1withSumw2(iBooker,
                                                  "photonRelativeIso_mAOD",
                                                  "photonRelativeIso",
                                                  100,
                                                  0.0,
                                                  2.,
                                                  "photonRelativeIso",
                                                  "Events",
                                                  "ELE_LOGY E1 P");
  h1_ele_photonRelativeIso_mAOD_barrel = bookH1withSumw2(iBooker,
                                                         "photonRelativeIso_mAOD_barrel",
                                                         "photonRelativeIso for barrel",
                                                         100,
                                                         0.0,
                                                         2.,
                                                         "photonRelativeIso_barrel",
                                                         "Events",
                                                         "ELE_LOGY E1 P");
  h1_ele_photonRelativeIso_mAOD_endcaps = bookH1withSumw2(iBooker,
                                                          "photonRelativeIso_mAOD_endcaps",
                                                          "photonRelativeIso for endcaps",
                                                          100,
                                                          0.0,
                                                          2.,
                                                          "photonRelativeIso_endcaps",
                                                          "Events",
                                                          "ELE_LOGY E1 P");
}

void ElectronMcSignalValidatorMiniAOD::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  // get collections
  auto electrons = iEvent.getHandle(electronToken_);
  auto electrons_endcaps = iEvent.getHandle(electronTokenEndcaps_);
  auto genParticles = iEvent.getHandle(mcTruthCollection_);

  edm::Handle<pat::ElectronCollection> mergedElectrons;

  edm::LogInfo("ElectronMcSignalValidatorMiniAOD::analyze")
      << "Treating event " << iEvent.id() << " with " << electrons.product()->size() << " electrons";
  edm::LogInfo("ElectronMcSignalValidatorMiniAOD::analyze")
      << "Treating event " << iEvent.id() << " with " << electrons_endcaps.product()->size()
      << " multi slimmed electrons";

  h1_recEleNum->Fill((*electrons).size());

  //===============================================
  // all rec electrons
  //===============================================

  pat::Electron gsfElectron;
  pat::ElectronCollection::const_iterator el1;
  std::vector<pat::Electron>::const_iterator el3;
  std::vector<pat::Electron>::const_iterator el4;

  //===============================================
  // get a vector with EB  & EE
  //===============================================
  std::vector<pat::Electron> localCollection;

  // looking for EB
  for (el1 = electrons->begin(); el1 != electrons->end(); el1++) {
    if (el1->isEB()) {
      localCollection.push_back(*el1);
    }
  }

  // looking for EE
  for (el1 = electrons_endcaps->begin(); el1 != electrons_endcaps->end(); el1++) {
    if (el1->isEE()) {
      localCollection.push_back(*el1);
    }
  }

  for (el3 = localCollection.begin(); el3 != localCollection.end(); el3++) {
    for (el4 = el3 + 1; el4 != localCollection.end(); el4++) {
      math::XYZTLorentzVector p12 = el3->p4() + el4->p4();
      float mee2 = p12.Dot(p12);
      h1_ele_mee_all->Fill(sqrt(mee2));
      if (el3->charge() * el4->charge() < 0.) {
        h1_ele_mee_os->Fill(sqrt(mee2));
      }
    }
  }

  //===============================================
  // charge mis-ID
  //===============================================

  bool matchingMotherID;

  //===============================================
  // association mc-reco
  //===============================================

  for (size_t i = 0; i < genParticles->size(); i++) {
    // select requested mother matching gen particle
    // always include single particle with no mother
    const Candidate* mother = (*genParticles)[i].mother(0);
    matchingMotherID = false;
    for (unsigned int ii = 0; ii < matchingMotherIDs_.size(); ii++) {
      if (mother == nullptr) {
        matchingMotherID = true;
      } else if (mother->pdgId() == matchingMotherIDs_[ii]) {
        if (mother->numberOfDaughters() <= 2) {
          matchingMotherID = true;
        }
      }  // end of mother if test

    }  // end of for loop
    if (!matchingMotherID) {
      continue;
    }

    // electron preselection
    if ((*genParticles)[i].pt() > maxPt_ || std::abs((*genParticles)[i].eta()) > maxAbsEta_) {
      continue;
    }

    // find best matched electron
    bool okGsfFound = false;
    bool passMiniAODSelection = true;
    double gsfOkRatio = 999999.;
    bool isEBflag = false;
    bool isEEflag = false;
    pat::Electron bestGsfElectron;

    for (el3 = localCollection.begin(); el3 != localCollection.end(); el3++) {
      double dphi = el3->phi() - (*genParticles)[i].phi();
      if (std::abs(dphi) > CLHEP::pi) {
        dphi = dphi < 0 ? (CLHEP::twopi) + dphi : dphi - CLHEP::twopi;
      }
      double deltaR2 = (el3->eta() - (*genParticles)[i].eta()) * (el3->eta() - (*genParticles)[i].eta()) + dphi * dphi;
      if (deltaR2 < deltaR2_) {
        if ((((*genParticles)[i].pdgId() == 11) && (el3->charge() < 0.)) ||
            (((*genParticles)[i].pdgId() == -11) && (el3->charge() > 0.))) {
          double tmpGsfRatio = el3->p() / (*genParticles)[i].p();
          if (std::abs(tmpGsfRatio - 1) < std::abs(gsfOkRatio - 1)) {
            gsfOkRatio = tmpGsfRatio;
            bestGsfElectron = *el3;
            okGsfFound = true;
          }
        }
      }
    }  // end *electrons loop

    if (okGsfFound) {
      //------------------------------------
      // analysis when the mc track is found
      //------------------------------------
      passMiniAODSelection = bestGsfElectron.pt() >= 5.;
      double one_over_pt = 1. / bestGsfElectron.pt();
      isEBflag = bestGsfElectron.isEB();
      isEEflag = bestGsfElectron.isEE();

      // electron related distributions
      h1_ele_vertexPt->Fill(bestGsfElectron.pt());
      h1_ele_vertexEta->Fill(bestGsfElectron.eta());
      if ((bestGsfElectron.scSigmaIEtaIEta() == 0.) && (bestGsfElectron.fbrem() == 0.))
        h1_ele_vertexPt_nocut->Fill(bestGsfElectron.pt());
      // track related distributions
      h2_ele_foundHitsVsEta->Fill(bestGsfElectron.eta(), bestGsfElectron.gsfTrack()->numberOfValidHits());

      // generated distributions for matched electrons
      h2_ele_PoPtrueVsEta->Fill(bestGsfElectron.eta(), bestGsfElectron.p() / (*genParticles)[i].p());

      if (passMiniAODSelection) {  // Pt > 5.
        h2_ele_sigmaIetaIetaVsPt->Fill(bestGsfElectron.pt(), bestGsfElectron.scSigmaIEtaIEta());

        // supercluster related distributions
        h1_scl_SigIEtaIEta_mAOD->Fill(bestGsfElectron.scSigmaIEtaIEta());
        h1_ele_dEtaSc_propVtx_mAOD->Fill(bestGsfElectron.deltaEtaSuperClusterTrackAtVtx());
        h1_ele_dPhiCl_propOut_mAOD->Fill(bestGsfElectron.deltaPhiSeedClusterTrackAtCalo());

        // track related distributions
        h2_ele_foundHitsVsEta_mAOD->Fill(bestGsfElectron.eta(), bestGsfElectron.gsfTrack()->numberOfValidHits());

        // match distributions
        h1_ele_HoE_mAOD->Fill(bestGsfElectron.hcalOverEcal());

        // fbrem
        h1_ele_fbrem_mAOD->Fill(bestGsfElectron.fbrem());

        // -- pflow over pT

        h1_ele_chargedHadronRelativeIso_mAOD->Fill(bestGsfElectron.pfIsolationVariables().sumChargedHadronPt *
                                                   one_over_pt);
        h1_ele_neutralHadronRelativeIso_mAOD->Fill(bestGsfElectron.pfIsolationVariables().sumNeutralHadronEt *
                                                   one_over_pt);
        h1_ele_photonRelativeIso_mAOD->Fill(bestGsfElectron.pfIsolationVariables().sumPhotonEt * one_over_pt);

        if (isEBflag) {
          // supercluster related distributions
          h1_scl_SigIEtaIEta_mAOD_barrel->Fill(bestGsfElectron.scSigmaIEtaIEta());
          h1_ele_dEtaSc_propVtx_mAOD_barrel->Fill(bestGsfElectron.deltaEtaSuperClusterTrackAtVtx());
          h1_ele_dPhiCl_propOut_mAOD_barrel->Fill(bestGsfElectron.deltaPhiSeedClusterTrackAtCalo());
          // match distributions
          h1_ele_HoE_mAOD_barrel->Fill(bestGsfElectron.hcalOverEcal());
          // fbrem
          h1_ele_fbrem_mAOD_barrel->Fill(bestGsfElectron.fbrem());

          h1_ele_chargedHadronRelativeIso_mAOD_barrel->Fill(bestGsfElectron.pfIsolationVariables().sumChargedHadronPt *
                                                            one_over_pt);
          h1_ele_neutralHadronRelativeIso_mAOD_barrel->Fill(bestGsfElectron.pfIsolationVariables().sumNeutralHadronEt *
                                                            one_over_pt);
          h1_ele_photonRelativeIso_mAOD_barrel->Fill(bestGsfElectron.pfIsolationVariables().sumPhotonEt * one_over_pt);
        }

        // supercluster related distributions
        else if (isEEflag) {
          h1_scl_SigIEtaIEta_mAOD_endcaps->Fill(bestGsfElectron.scSigmaIEtaIEta());
          h1_ele_dEtaSc_propVtx_mAOD_endcaps->Fill(bestGsfElectron.deltaEtaSuperClusterTrackAtVtx());
          h1_ele_dPhiCl_propOut_mAOD_endcaps->Fill(bestGsfElectron.deltaPhiSeedClusterTrackAtCalo());
          // match distributions
          h1_ele_HoE_mAOD_endcaps->Fill(bestGsfElectron.hcalOverEcal());
          // fbrem
          h1_ele_fbrem_mAOD_endcaps->Fill(bestGsfElectron.fbrem());
          h1_ele_chargedHadronRelativeIso_mAOD_endcaps->Fill(bestGsfElectron.pfIsolationVariables().sumChargedHadronPt *
                                                             one_over_pt);
          h1_ele_neutralHadronRelativeIso_mAOD_endcaps->Fill(bestGsfElectron.pfIsolationVariables().sumNeutralHadronEt *
                                                             one_over_pt);
          h1_ele_photonRelativeIso_mAOD_endcaps->Fill(bestGsfElectron.pfIsolationVariables().sumPhotonEt * one_over_pt);
        }
      }
    }
    //} // end loop i_elec
  }  // end loop size_t i
}
