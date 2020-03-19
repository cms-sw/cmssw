#include "Validation/RecoEgamma/plugins/EgammaObjects.h"

#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/EgammaReco/interface/BasicCluster.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/EgammaReco/interface/ClusterShape.h"
#include "DataFormats/EgammaCandidates/interface/Photon.h"
#include "DataFormats/EgammaCandidates/interface/PhotonFwd.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectronFwd.h"

#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"

#include "TF1.h"

EgammaObjects::EgammaObjects(const edm::ParameterSet& ps) {
  particleID = ps.getParameter<int>("particleID");
  EtCut = ps.getParameter<int>("EtCut");

  if (particleID == 22)
    particleString = "Photon";
  else if (particleID == 11)
    particleString = "Electron";
  else
    throw(std::runtime_error(
        "\n\nEgammaObjects: Only 11 = Photon and 22 = Electron are acceptable parictleIDs! Exiting...\n\n"));

  loadCMSSWObjects(ps);
  loadHistoParameters(ps);

  rootFile_ = TFile::Open(ps.getParameter<std::string>("outputFile").c_str(), "RECREATE");

  hist_EtaEfficiency_ = nullptr;
  hist_EtaNumRecoOverNumTrue_ = nullptr;
  hist_deltaEtaVsEt_ = nullptr;
  hist_deltaEtaVsE_ = nullptr;
  hist_deltaEtaVsEta_ = nullptr;
  hist_deltaEtaVsPhi_ = nullptr;
  hist_resolutionEtaVsEt_ = nullptr;
  hist_resolutionEtaVsE_ = nullptr;
  hist_resolutionEtaVsEta_ = nullptr;
  hist_resolutionEtaVsPhi_ = nullptr;

  hist_Phi_ = nullptr;
  hist_PhiOverTruth_ = nullptr;
  hist_PhiEfficiency_ = nullptr;
  hist_PhiNumRecoOverNumTrue_ = nullptr;
  hist_deltaPhiVsEt_ = nullptr;
  hist_deltaPhiVsE_ = nullptr;
  hist_deltaPhiVsEta_ = nullptr;
  hist_deltaPhiVsPhi_ = nullptr;
  hist_resolutionPhiVsEt_ = nullptr;
  hist_resolutionPhiVsE_ = nullptr;
  hist_resolutionPhiVsEta_ = nullptr;
  hist_resolutionPhiVsPhi_ = nullptr;

  hist_All_recoMass_ = nullptr;
  hist_BarrelOnly_recoMass_ = nullptr;
  hist_EndcapOnly_recoMass_ = nullptr;
  hist_Mixed_recoMass_ = nullptr;

  hist_recoMass_withBackgroud_NoEtCut_ = nullptr;
  hist_recoMass_withBackgroud_5EtCut_ = nullptr;
  hist_recoMass_withBackgroud_10EtCut_ = nullptr;
  hist_recoMass_withBackgroud_20EtCut_ = nullptr;

  _TEMP_scatterPlot_EtOverTruthVsEt_ = nullptr;
  _TEMP_scatterPlot_EtOverTruthVsE_ = nullptr;
  _TEMP_scatterPlot_EtOverTruthVsEta_ = nullptr;
  _TEMP_scatterPlot_EtOverTruthVsPhi_ = nullptr;

  _TEMP_scatterPlot_EOverTruthVsEt_ = nullptr;
  _TEMP_scatterPlot_EOverTruthVsE_ = nullptr;
  _TEMP_scatterPlot_EOverTruthVsEta_ = nullptr;
  _TEMP_scatterPlot_EOverTruthVsPhi_ = nullptr;

  _TEMP_scatterPlot_deltaEtaVsEt_ = nullptr;
  _TEMP_scatterPlot_deltaEtaVsE_ = nullptr;
  _TEMP_scatterPlot_deltaEtaVsEta_ = nullptr;
  _TEMP_scatterPlot_deltaEtaVsPhi_ = nullptr;

  _TEMP_scatterPlot_deltaPhiVsEt_ = nullptr;
  _TEMP_scatterPlot_deltaPhiVsE_ = nullptr;
  _TEMP_scatterPlot_deltaPhiVsEta_ = nullptr;
  _TEMP_scatterPlot_deltaPhiVsPhi_ = nullptr;
}

void EgammaObjects::loadCMSSWObjects(const edm::ParameterSet& ps) {
  MCTruthCollectionT_ = consumes<edm::HepMCProduct>(ps.getParameter<edm::InputTag>("MCTruthCollection"));
  RecoCollectionT_ = consumes<reco::GsfElectronCollection>(ps.getParameter<edm::InputTag>("RecoCollection"));
}

void EgammaObjects::loadHistoParameters(const edm::ParameterSet& ps) {
  hist_min_Et_ = ps.getParameter<double>("hist_min_Et");
  hist_max_Et_ = ps.getParameter<double>("hist_max_Et");
  hist_bins_Et_ = ps.getParameter<int>("hist_bins_Et");

  hist_min_E_ = ps.getParameter<double>("hist_min_E");
  hist_max_E_ = ps.getParameter<double>("hist_max_E");
  hist_bins_E_ = ps.getParameter<int>("hist_bins_E");

  hist_min_Eta_ = ps.getParameter<double>("hist_min_Eta");
  hist_max_Eta_ = ps.getParameter<double>("hist_max_Eta");
  hist_bins_Eta_ = ps.getParameter<int>("hist_bins_Eta");

  hist_min_Phi_ = ps.getParameter<double>("hist_min_Phi");
  hist_max_Phi_ = ps.getParameter<double>("hist_max_Phi");
  hist_bins_Phi_ = ps.getParameter<int>("hist_bins_Phi");

  hist_min_EtOverTruth_ = ps.getParameter<double>("hist_min_EtOverTruth");
  hist_max_EtOverTruth_ = ps.getParameter<double>("hist_max_EtOverTruth");
  hist_bins_EtOverTruth_ = ps.getParameter<int>("hist_bins_EtOverTruth");

  hist_min_EOverTruth_ = ps.getParameter<double>("hist_min_EOverTruth");
  hist_max_EOverTruth_ = ps.getParameter<double>("hist_max_EOverTruth");
  hist_bins_EOverTruth_ = ps.getParameter<int>("hist_bins_EOverTruth");

  hist_min_EtaOverTruth_ = ps.getParameter<double>("hist_min_EtaOverTruth");
  hist_max_EtaOverTruth_ = ps.getParameter<double>("hist_max_EtaOverTruth");
  hist_bins_EtaOverTruth_ = ps.getParameter<int>("hist_bins_EtaOverTruth");

  hist_min_PhiOverTruth_ = ps.getParameter<double>("hist_min_PhiOverTruth");
  hist_max_PhiOverTruth_ = ps.getParameter<double>("hist_max_PhiOverTruth");
  hist_bins_PhiOverTruth_ = ps.getParameter<int>("hist_bins_PhiOverTruth");

  hist_min_deltaEta_ = ps.getParameter<double>("hist_min_deltaEta");
  hist_max_deltaEta_ = ps.getParameter<double>("hist_max_deltaEta");
  hist_bins_deltaEta_ = ps.getParameter<int>("hist_bins_deltaEta");

  hist_min_deltaPhi_ = ps.getParameter<double>("hist_min_deltaPhi");
  hist_max_deltaPhi_ = ps.getParameter<double>("hist_max_deltaPhi");
  hist_bins_deltaPhi_ = ps.getParameter<int>("hist_bins_deltaPhi");

  hist_min_recoMass_ = ps.getParameter<double>("hist_min_recoMass");
  hist_max_recoMass_ = ps.getParameter<double>("hist_max_recoMass");
  hist_bins_recoMass_ = ps.getParameter<int>("hist_bins_recoMass");
}

EgammaObjects::~EgammaObjects() { delete rootFile_; }

void EgammaObjects::beginJob() {
  TH1::SetDefaultSumw2(true);

  createBookedHistoObjects();
  createTempHistoObjects();
}

void EgammaObjects::createBookedHistoObjects() {
  hist_Et_ =
      new TH1D("hist_Et_", ("Et Distribution of " + particleString).c_str(), hist_bins_Et_, hist_min_Et_, hist_max_Et_);
  hist_EtOverTruth_ = new TH1D("hist_EtOverTruth_",
                               ("Reco Et over True Et of " + particleString).c_str(),
                               hist_bins_EtOverTruth_,
                               hist_min_EtOverTruth_,
                               hist_max_EtOverTruth_);
  hist_EtEfficiency_ = new TH1D("hist_EtEfficiency_",
                                ("# of True " + particleString + " Reconstructed over # of True " + particleString +
                                 " VS Et of " + particleString)
                                    .c_str(),
                                hist_bins_Et_,
                                hist_min_Et_,
                                hist_max_Et_);
  hist_EtNumRecoOverNumTrue_ = new TH1D(
      "hist_EtNumRecoOverNumTrue_",
      ("# of Reco " + particleString + " over # of True " + particleString + " VS Et of " + particleString).c_str(),
      hist_bins_Et_,
      hist_min_Et_,
      hist_max_Et_);

  hist_E_ =
      new TH1D("hist_E_", ("E Distribution of " + particleString).c_str(), hist_bins_E_, hist_min_E_, hist_max_E_);
  hist_EOverTruth_ = new TH1D("hist_EOverTruth_",
                              ("Reco E over True E of " + particleString).c_str(),
                              hist_bins_EOverTruth_,
                              hist_min_EOverTruth_,
                              hist_max_EOverTruth_);
  hist_EEfficiency_ = new TH1D(
      "hist_EEfficiency_",
      ("# of True " + particleString + " Reconstructed over # of True " + particleString + " VS E of " + particleString)
          .c_str(),
      hist_bins_E_,
      hist_min_E_,
      hist_max_E_);
  hist_ENumRecoOverNumTrue_ = new TH1D(
      "hist_ENumRecoOverNumTrue_",
      ("# of Reco " + particleString + " over # of True " + particleString + " VS E of " + particleString).c_str(),
      hist_bins_E_,
      hist_min_E_,
      hist_max_E_);

  hist_Eta_ = new TH1D(
      "hist_Eta_", ("Eta Distribution of " + particleString).c_str(), hist_bins_Eta_, hist_min_Eta_, hist_max_Eta_);
  hist_EtaOverTruth_ = new TH1D("hist_EtaOverTruth_",
                                ("Reco Eta over True Eta of " + particleString).c_str(),
                                hist_bins_EtaOverTruth_,
                                hist_min_EtaOverTruth_,
                                hist_max_EtaOverTruth_);
  hist_EtaEfficiency_ = new TH1D("hist_EtaEfficiency_",
                                 ("# of True " + particleString + " Reconstructed over # of True " + particleString +
                                  " VS Eta of " + particleString)
                                     .c_str(),
                                 hist_bins_Eta_,
                                 hist_min_Eta_,
                                 hist_max_Eta_);
  hist_EtaNumRecoOverNumTrue_ = new TH1D(
      "hist_EtaNumRecoOverNumTrue_",
      ("# of Reco " + particleString + " over # of True " + particleString + " VS Eta of " + particleString).c_str(),
      hist_bins_Eta_,
      hist_min_Eta_,
      hist_max_Eta_);

  hist_Phi_ = new TH1D(
      "hist_Phi_", ("Phi Distribution of " + particleString).c_str(), hist_bins_Phi_, hist_min_Phi_, hist_max_Phi_);
  hist_PhiOverTruth_ = new TH1D("hist_PhiOverTruth_",
                                ("Reco Phi over True Phi of " + particleString).c_str(),
                                hist_bins_PhiOverTruth_,
                                hist_min_PhiOverTruth_,
                                hist_max_PhiOverTruth_);
  hist_PhiEfficiency_ = new TH1D("hist_PhiEfficiency_",
                                 ("# of True " + particleString + " Reconstructed over # of True " + particleString +
                                  " VS Phi of " + particleString)
                                     .c_str(),
                                 hist_bins_Phi_,
                                 hist_min_Phi_,
                                 hist_max_Phi_);
  hist_PhiNumRecoOverNumTrue_ = new TH1D(
      "hist_PhiNumRecoOverNumTrue_",
      ("# of Reco " + particleString + " over # of True " + particleString + " VS Phi of " + particleString).c_str(),
      hist_bins_Phi_,
      hist_min_Phi_,
      hist_max_Phi_);

  std::string recoParticleName;

  if (particleID == 22)
    recoParticleName = "Higgs";
  else if (particleID == 11)
    recoParticleName = "Z";

  hist_All_recoMass_ = new TH1D("hist_All_recoMass_",
                                (recoParticleName + " Mass from " + particleString + " in All Regions").c_str(),
                                hist_bins_recoMass_,
                                hist_min_recoMass_,
                                hist_max_recoMass_);
  hist_BarrelOnly_recoMass_ = new TH1D("hist_BarrelOnly_recoMass_",
                                       (recoParticleName + " Mass from " + particleString + " in Barrel").c_str(),
                                       hist_bins_recoMass_,
                                       hist_min_recoMass_,
                                       hist_max_recoMass_);
  hist_EndcapOnly_recoMass_ = new TH1D("hist_EndcapOnly_recoMass_",
                                       (recoParticleName + " Mass from " + particleString + " in EndCap").c_str(),
                                       hist_bins_recoMass_,
                                       hist_min_recoMass_,
                                       hist_max_recoMass_);
  hist_Mixed_recoMass_ = new TH1D("hist_Mixed_recoMass_",
                                  (recoParticleName + " Mass from " + particleString + " in Split Detectors").c_str(),
                                  hist_bins_recoMass_,
                                  hist_min_recoMass_,
                                  hist_max_recoMass_);

  hist_recoMass_withBackgroud_NoEtCut_ =
      new TH1D("hist_recoMass_withBackgroud_NoEtCut_",
               (recoParticleName + " Mass from " + particleString + " with Background, No Et Cut").c_str(),
               hist_bins_recoMass_,
               hist_min_recoMass_,
               hist_max_recoMass_);
  hist_recoMass_withBackgroud_5EtCut_ =
      new TH1D("hist_recoMass_withBackgroud_5EtCut_",
               (recoParticleName + " Mass from " + particleString + " with Background, 5 Et Cut").c_str(),
               hist_bins_recoMass_,
               hist_min_recoMass_,
               hist_max_recoMass_);
  hist_recoMass_withBackgroud_10EtCut_ =
      new TH1D("hist_recoMass_withBackgroud_10EtCut_",
               (recoParticleName + " Mass from " + particleString + " with Background, 10 Et Cut").c_str(),
               hist_bins_recoMass_,
               hist_min_recoMass_,
               hist_max_recoMass_);
  hist_recoMass_withBackgroud_20EtCut_ =
      new TH1D("hist_recoMass_withBackgroud_20EtCut_",
               (recoParticleName + " Mass from " + particleString + " with Background, 20 Et Cut").c_str(),
               hist_bins_recoMass_,
               hist_min_recoMass_,
               hist_max_recoMass_);
}

void EgammaObjects::createTempHistoObjects() {
  _TEMP_scatterPlot_EtOverTruthVsEt_ = new TH2D("_TEMP_scatterPlot_EtOverTruthVsEt_",
                                                "_TEMP_scatterPlot_EtOverTruthVsEt_",
                                                hist_bins_Et_,
                                                hist_min_Et_,
                                                hist_max_Et_,
                                                hist_bins_EtOverTruth_,
                                                hist_min_EtOverTruth_,
                                                hist_max_EtOverTruth_);
  _TEMP_scatterPlot_EtOverTruthVsE_ = new TH2D("_TEMP_scatterPlot_EtOverTruthVsE_",
                                               "_TEMP_scatterPlot_EtOverTruthVsE_",
                                               hist_bins_E_,
                                               hist_min_E_,
                                               hist_max_E_,
                                               hist_bins_EtOverTruth_,
                                               hist_min_EtOverTruth_,
                                               hist_max_EtOverTruth_);
  _TEMP_scatterPlot_EtOverTruthVsEta_ = new TH2D("_TEMP_scatterPlot_EtOverTruthVsEta_",
                                                 "_TEMP_scatterPlot_EtOverTruthVsEta_",
                                                 hist_bins_Eta_,
                                                 hist_min_Eta_,
                                                 hist_max_Eta_,
                                                 hist_bins_EtOverTruth_,
                                                 hist_min_EtOverTruth_,
                                                 hist_max_EtOverTruth_);
  _TEMP_scatterPlot_EtOverTruthVsPhi_ = new TH2D("_TEMP_scatterPlot_EtOverTruthVsPhi_",
                                                 "_TEMP_scatterPlot_EtOverTruthVsPhi_",
                                                 hist_bins_Phi_,
                                                 hist_min_Phi_,
                                                 hist_max_Phi_,
                                                 hist_bins_EtOverTruth_,
                                                 hist_min_EtOverTruth_,
                                                 hist_max_EtOverTruth_);

  _TEMP_scatterPlot_EOverTruthVsEt_ = new TH2D("_TEMP_scatterPlot_EOverTruthVsEt_",
                                               "_TEMP_scatterPlot_EOverTruthVsEt_",
                                               hist_bins_Et_,
                                               hist_min_Et_,
                                               hist_max_Et_,
                                               hist_bins_EOverTruth_,
                                               hist_min_EOverTruth_,
                                               hist_max_EOverTruth_);
  _TEMP_scatterPlot_EOverTruthVsE_ = new TH2D("_TEMP_scatterPlot_EOverTruthVsE_",
                                              "_TEMP_scatterPlot_EOverTruthVsE_",
                                              hist_bins_E_,
                                              hist_min_E_,
                                              hist_max_E_,
                                              hist_bins_EOverTruth_,
                                              hist_min_EOverTruth_,
                                              hist_max_EOverTruth_);
  _TEMP_scatterPlot_EOverTruthVsEta_ = new TH2D("_TEMP_scatterPlot_EOverTruthVsEta_",
                                                "_TEMP_scatterPlot_EOverTruthVsEta_",
                                                hist_bins_Eta_,
                                                hist_min_Eta_,
                                                hist_max_Eta_,
                                                hist_bins_EOverTruth_,
                                                hist_min_EOverTruth_,
                                                hist_max_EOverTruth_);
  _TEMP_scatterPlot_EOverTruthVsPhi_ = new TH2D("_TEMP_scatterPlot_EOverTruthVsPhi_",
                                                "_TEMP_scatterPlot_EOverTruthVsPhi_",
                                                hist_bins_Phi_,
                                                hist_min_Phi_,
                                                hist_max_Phi_,
                                                hist_bins_EOverTruth_,
                                                hist_min_EOverTruth_,
                                                hist_max_EOverTruth_);

  _TEMP_scatterPlot_deltaEtaVsEt_ = new TH2D("_TEMP_scatterPlot_deltaEtaVsEt_",
                                             "_TEMP_scatterPlot_deltaEtaVsEt_",
                                             hist_bins_Et_,
                                             hist_min_Et_,
                                             hist_max_Et_,
                                             hist_bins_deltaEta_,
                                             hist_min_deltaEta_,
                                             hist_max_deltaEta_);
  _TEMP_scatterPlot_deltaEtaVsE_ = new TH2D("_TEMP_scatterPlot_deltaEtaVsE_",
                                            "_TEMP_scatterPlot_deltaEtaVsE_",
                                            hist_bins_E_,
                                            hist_min_E_,
                                            hist_max_E_,
                                            hist_bins_deltaEta_,
                                            hist_min_deltaEta_,
                                            hist_max_deltaEta_);
  _TEMP_scatterPlot_deltaEtaVsEta_ = new TH2D("_TEMP_scatterPlot_deltaEtaVsEta_",
                                              "_TEMP_scatterPlot_deltaEtaVsEta_",
                                              hist_bins_Eta_,
                                              hist_min_Eta_,
                                              hist_max_Eta_,
                                              hist_bins_deltaEta_,
                                              hist_min_deltaEta_,
                                              hist_max_deltaEta_);
  _TEMP_scatterPlot_deltaEtaVsPhi_ = new TH2D("_TEMP_scatterPlot_deltaEtaVsPhi_",
                                              "_TEMP_scatterPlot_deltaEtaVsPhi_",
                                              hist_bins_Phi_,
                                              hist_min_Phi_,
                                              hist_max_Phi_,
                                              hist_bins_deltaEta_,
                                              hist_min_deltaEta_,
                                              hist_max_deltaEta_);

  _TEMP_scatterPlot_deltaPhiVsEt_ = new TH2D("_TEMP_scatterPlot_deltaPhiVsEt_",
                                             "_TEMP_scatterPlot_deltaPhiVsEt_",
                                             hist_bins_Et_,
                                             hist_min_Et_,
                                             hist_max_Et_,
                                             hist_bins_deltaPhi_,
                                             hist_min_deltaPhi_,
                                             hist_max_deltaPhi_);
  _TEMP_scatterPlot_deltaPhiVsE_ = new TH2D("_TEMP_scatterPlot_deltaPhiVsE_",
                                            "_TEMP_scatterPlot_deltaPhiVsE_",
                                            hist_bins_E_,
                                            hist_min_E_,
                                            hist_max_E_,
                                            hist_bins_deltaPhi_,
                                            hist_min_deltaPhi_,
                                            hist_max_deltaPhi_);
  _TEMP_scatterPlot_deltaPhiVsEta_ = new TH2D("_TEMP_scatterPlot_deltaPhiVsEta_",
                                              "_TEMP_scatterPlot_deltaPhiVsEta_",
                                              hist_bins_Eta_,
                                              hist_min_Eta_,
                                              hist_max_Eta_,
                                              hist_bins_deltaPhi_,
                                              hist_min_deltaPhi_,
                                              hist_max_deltaPhi_);
  _TEMP_scatterPlot_deltaPhiVsPhi_ = new TH2D("_TEMP_scatterPlot_deltaPhiVsPhi_",
                                              "_TEMP_scatterPlot_deltaPhiVsPhi_",
                                              hist_bins_Phi_,
                                              hist_min_Phi_,
                                              hist_max_Phi_,
                                              hist_bins_deltaPhi_,
                                              hist_min_deltaPhi_,
                                              hist_max_deltaPhi_);
}

void EgammaObjects::analyze(const edm::Event& evt, const edm::EventSetup& es) {
  if (particleID == 22)
    analyzePhotons(evt, es);
  else if (particleID == 11)
    analyzeElectrons(evt, es);
}

void EgammaObjects::analyzePhotons(const edm::Event& evt, const edm::EventSetup& es) {
  edm::Handle<reco::PhotonCollection> pPhotons;
  evt.getByToken(RecoCollectionT_, pPhotons);
  if (!pPhotons.isValid()) {
    Labels l;
    labelsForToken(RecoCollectionT_, l);
    edm::LogError("EgammaObjects") << "Error! can't get collection with label " << l.module;
  }

  const reco::PhotonCollection* photons = pPhotons.product();
  std::vector<reco::Photon> photonsMCMatched;

  for (reco::PhotonCollection::const_iterator aClus = photons->begin(); aClus != photons->end(); aClus++) {
    if (aClus->et() >= EtCut) {
      hist_Et_->Fill(aClus->et());
      hist_E_->Fill(aClus->energy());
      hist_Eta_->Fill(aClus->eta());
      hist_Phi_->Fill(aClus->phi());
    }
  }

  for (int firstPhoton = 0, numPhotons = photons->size(); firstPhoton < numPhotons - 1; firstPhoton++)
    for (int secondPhoton = firstPhoton + 1; secondPhoton < numPhotons; secondPhoton++) {
      reco::Photon pOne = (*photons)[firstPhoton];
      reco::Photon pTwo = (*photons)[secondPhoton];

      double recoMass = findRecoMass(pOne, pTwo);

      hist_recoMass_withBackgroud_NoEtCut_->Fill(recoMass);

      if (pOne.et() >= 5 && pTwo.et() >= 5)
        hist_recoMass_withBackgroud_5EtCut_->Fill(recoMass);

      if (pOne.et() >= 10 && pTwo.et() >= 10)
        hist_recoMass_withBackgroud_10EtCut_->Fill(recoMass);

      if (pOne.et() >= 20 && pTwo.et() >= 20)
        hist_recoMass_withBackgroud_20EtCut_->Fill(recoMass);
    }

  edm::Handle<edm::HepMCProduct> pMCTruth;
  evt.getByToken(MCTruthCollectionT_, pMCTruth);
  if (!pMCTruth.isValid()) {
    Labels l;
    labelsForToken(MCTruthCollectionT_, l);
    edm::LogError("EgammaObjects") << "Error! can't get collection with label " << l.module;
  }

  const HepMC::GenEvent* genEvent = pMCTruth->GetEvent();

  for (HepMC::GenEvent::particle_const_iterator currentParticle = genEvent->particles_begin();
       currentParticle != genEvent->particles_end();
       currentParticle++) {
    if (abs((*currentParticle)->pdg_id()) == 22 && (*currentParticle)->status() == 1 &&
        (*currentParticle)->momentum().e() /
                cosh(ecalEta((*currentParticle)->momentum().eta(),
                             (*currentParticle)->production_vertex()->position().z() / 10.,
                             (*currentParticle)->production_vertex()->position().perp() / 10.)) >=
            EtCut) {
      HepMC::FourVector vtx = (*currentParticle)->production_vertex()->position();
      double phiTrue = (*currentParticle)->momentum().phi();
      double etaTrue = ecalEta((*currentParticle)->momentum().eta(), vtx.z() / 10., vtx.perp() / 10.);
      double eTrue = (*currentParticle)->momentum().e();
      double etTrue = (*currentParticle)->momentum().e() / cosh(etaTrue);

      double etaCurrent, etaFound = -999;
      double phiCurrent, phiFound = -999;
      double etCurrent, etFound = -999;
      double eCurrent, eFound = -999;

      reco::Photon bestMatchPhoton;

      double closestParticleDistance = 999;

      for (reco::PhotonCollection::const_iterator aClus = photons->begin(); aClus != photons->end(); aClus++) {
        if (aClus->et() > EtCut) {
          etaCurrent = aClus->eta();
          phiCurrent = aClus->phi();
          etCurrent = aClus->et();
          eCurrent = aClus->energy();

          double deltaPhi = phiCurrent - phiTrue;
          if (deltaPhi > Geom::pi())
            deltaPhi -= 2. * Geom::pi();
          if (deltaPhi < -Geom::pi())
            deltaPhi += 2. * Geom::pi();
          double deltaR = std::sqrt(std::pow(etaCurrent - etaTrue, 2) + std::pow(deltaPhi, 2));

          if (deltaR < closestParticleDistance) {
            etFound = etCurrent;
            eFound = eCurrent;
            etaFound = etaCurrent;
            phiFound = phiCurrent;
            closestParticleDistance = deltaR;
            bestMatchPhoton = *aClus;
          }
        }
      }

      if (closestParticleDistance < 0.05 && etFound / etTrue > .5 && etFound / etTrue < 1.5) {
        hist_EtOverTruth_->Fill(etFound / etTrue);
        hist_EOverTruth_->Fill(eFound / eTrue);
        hist_EtaOverTruth_->Fill(etaFound / etaTrue);
        hist_PhiOverTruth_->Fill(phiFound / phiTrue);

        hist_EtEfficiency_->Fill(etTrue);
        hist_EEfficiency_->Fill(eTrue);
        hist_EtaEfficiency_->Fill(etaTrue);
        hist_PhiEfficiency_->Fill(phiTrue);

        double deltaPhi = phiFound - phiTrue;
        if (deltaPhi > Geom::pi())
          deltaPhi -= 2. * Geom::pi();
        if (deltaPhi < -Geom::pi())
          deltaPhi += 2. * Geom::pi();

        _TEMP_scatterPlot_EtOverTruthVsEt_->Fill(etFound, etFound / etTrue);
        _TEMP_scatterPlot_EtOverTruthVsE_->Fill(eFound, etFound / etTrue);
        _TEMP_scatterPlot_EtOverTruthVsEta_->Fill(etaFound, etFound / etTrue);
        _TEMP_scatterPlot_EtOverTruthVsPhi_->Fill(phiFound, etFound / etTrue);

        _TEMP_scatterPlot_EOverTruthVsEt_->Fill(etFound, eFound / eTrue);
        _TEMP_scatterPlot_EOverTruthVsE_->Fill(eFound, eFound / eTrue);
        _TEMP_scatterPlot_EOverTruthVsEta_->Fill(etaFound, eFound / eTrue);
        _TEMP_scatterPlot_EOverTruthVsPhi_->Fill(phiFound, eFound / eTrue);

        _TEMP_scatterPlot_deltaEtaVsEt_->Fill(etFound, etaFound - etaTrue);
        _TEMP_scatterPlot_deltaEtaVsE_->Fill(eFound, etaFound - etaTrue);
        _TEMP_scatterPlot_deltaEtaVsEta_->Fill(etaFound, etaFound - etaTrue);
        _TEMP_scatterPlot_deltaEtaVsPhi_->Fill(phiFound, etaFound - etaTrue);

        _TEMP_scatterPlot_deltaPhiVsEt_->Fill(etFound, deltaPhi);
        _TEMP_scatterPlot_deltaPhiVsE_->Fill(eFound, deltaPhi);
        _TEMP_scatterPlot_deltaPhiVsEta_->Fill(etaFound, deltaPhi);
        _TEMP_scatterPlot_deltaPhiVsPhi_->Fill(phiFound, deltaPhi);

        photonsMCMatched.push_back(bestMatchPhoton);
      }

      hist_EtNumRecoOverNumTrue_->Fill(etTrue);
      hist_ENumRecoOverNumTrue_->Fill(eTrue);
      hist_EtaNumRecoOverNumTrue_->Fill(etaTrue);
      hist_PhiNumRecoOverNumTrue_->Fill(phiTrue);
    }
  }

  if (photonsMCMatched.size() == 2) {
    reco::Photon pOne = photonsMCMatched[0];
    reco::Photon pTwo = photonsMCMatched[1];

    double recoMass = findRecoMass(pOne, pTwo);

    hist_All_recoMass_->Fill(recoMass);

    if (pOne.superCluster()->seed()->algo() == 1 && pTwo.superCluster()->seed()->algo() == 1)
      hist_BarrelOnly_recoMass_->Fill(recoMass);
    else if (pOne.superCluster()->seed()->algo() == 0 && pTwo.superCluster()->seed()->algo() == 0)
      hist_EndcapOnly_recoMass_->Fill(recoMass);
    else
      hist_Mixed_recoMass_->Fill(recoMass);
  }
}

void EgammaObjects::analyzeElectrons(const edm::Event& evt, const edm::EventSetup& es) {
  edm::Handle<reco::GsfElectronCollection> pElectrons;
  evt.getByToken(RecoCollectionT_, pElectrons);
  if (!pElectrons.isValid()) {
    Labels l;
    labelsForToken(RecoCollectionT_, l);
    edm::LogError("DOEPlotsProducerElectrons") << "Error! can't get collection with label " << l.module;
  }

  const reco::GsfElectronCollection* electrons = pElectrons.product();
  std::vector<reco::GsfElectron> electronsMCMatched;

  for (reco::GsfElectronCollection::const_iterator aClus = electrons->begin(); aClus != electrons->end(); aClus++) {
    if (aClus->et() >= EtCut) {
      hist_Et_->Fill(aClus->et());
      hist_E_->Fill(aClus->energy());
      hist_Eta_->Fill(aClus->eta());
      hist_Phi_->Fill(aClus->phi());
    }
  }

  for (int firstElectron = 0, numElectrons = electrons->size(); firstElectron < numElectrons - 1; firstElectron++)
    for (int secondElectron = firstElectron + 1; secondElectron < numElectrons; secondElectron++) {
      reco::GsfElectron eOne = (*electrons)[firstElectron];
      reco::GsfElectron eTwo = (*electrons)[secondElectron];

      double recoMass = findRecoMass(eOne, eTwo);

      hist_recoMass_withBackgroud_NoEtCut_->Fill(recoMass);

      if (eOne.et() >= 5 && eTwo.et() >= 5)
        hist_recoMass_withBackgroud_5EtCut_->Fill(recoMass);

      if (eOne.et() >= 10 && eTwo.et() >= 10)
        hist_recoMass_withBackgroud_10EtCut_->Fill(recoMass);

      if (eOne.et() >= 20 && eTwo.et() >= 20)
        hist_recoMass_withBackgroud_20EtCut_->Fill(recoMass);
    }

  edm::Handle<edm::HepMCProduct> pMCTruth;
  evt.getByToken(MCTruthCollectionT_, pMCTruth);
  if (!pMCTruth.isValid()) {
    Labels l;
    labelsForToken(MCTruthCollectionT_, l);
    edm::LogError("DOEPlotsProducerElectrons") << "Error! can't get collection with label " << l.module;
  }

  const HepMC::GenEvent* genEvent = pMCTruth->GetEvent();
  for (HepMC::GenEvent::particle_const_iterator currentParticle = genEvent->particles_begin();
       currentParticle != genEvent->particles_end();
       currentParticle++) {
    if (abs((*currentParticle)->pdg_id()) == 11 && (*currentParticle)->status() == 1 &&
        (*currentParticle)->momentum().e() / cosh((*currentParticle)->momentum().eta()) >= EtCut) {
      double phiTrue = (*currentParticle)->momentum().phi();
      double etaTrue = (*currentParticle)->momentum().eta();
      double eTrue = (*currentParticle)->momentum().e();
      double etTrue = (*currentParticle)->momentum().e() / cosh(etaTrue);

      double etaCurrent, etaFound = -999;
      double phiCurrent, phiFound = -999;
      double etCurrent, etFound = -999;
      double eCurrent, eFound = -999;

      reco::GsfElectron bestMatchElectron;

      double closestParticleDistance = 999;

      for (reco::GsfElectronCollection::const_iterator aClus = electrons->begin(); aClus != electrons->end(); aClus++) {
        if (aClus->et() > EtCut) {
          etaCurrent = aClus->eta();
          phiCurrent = aClus->phi();
          etCurrent = aClus->et();
          eCurrent = aClus->energy();

          double deltaPhi = phiCurrent - phiTrue;
          if (deltaPhi > Geom::pi())
            deltaPhi -= 2. * Geom::pi();
          if (deltaPhi < -Geom::pi())
            deltaPhi += 2. * Geom::pi();
          double deltaR = std::sqrt(std::pow(etaCurrent - etaTrue, 2) + std::pow(deltaPhi, 2));

          if (deltaR < closestParticleDistance) {
            etFound = etCurrent;
            eFound = eCurrent;
            etaFound = etaCurrent;
            phiFound = phiCurrent;
            closestParticleDistance = deltaR;
            bestMatchElectron = *aClus;
          }
        }
      }

      if (closestParticleDistance < 0.05 && etFound / etTrue > .5 && etFound / etTrue < 1.5) {
        hist_EtOverTruth_->Fill(etFound / etTrue);
        hist_EOverTruth_->Fill(eFound / eTrue);
        hist_EtaOverTruth_->Fill(etaFound / etaTrue);
        hist_PhiOverTruth_->Fill(phiFound / phiTrue);

        hist_EtEfficiency_->Fill(etTrue);
        hist_EEfficiency_->Fill(eTrue);
        hist_EtaEfficiency_->Fill(etaTrue);
        hist_PhiEfficiency_->Fill(phiTrue);

        double deltaPhi = phiFound - phiTrue;
        if (deltaPhi > Geom::pi())
          deltaPhi -= 2. * Geom::pi();
        if (deltaPhi < -Geom::pi())
          deltaPhi += 2. * Geom::pi();

        _TEMP_scatterPlot_EtOverTruthVsEt_->Fill(etFound, etFound / etTrue);
        _TEMP_scatterPlot_EtOverTruthVsE_->Fill(eFound, etFound / etTrue);
        _TEMP_scatterPlot_EtOverTruthVsEta_->Fill(etaFound, etFound / etTrue);
        _TEMP_scatterPlot_EtOverTruthVsPhi_->Fill(phiFound, etFound / etTrue);

        _TEMP_scatterPlot_EOverTruthVsEt_->Fill(etFound, eFound / eTrue);
        _TEMP_scatterPlot_EOverTruthVsE_->Fill(eFound, eFound / eTrue);
        _TEMP_scatterPlot_EOverTruthVsEta_->Fill(etaFound, eFound / eTrue);
        _TEMP_scatterPlot_EOverTruthVsPhi_->Fill(phiFound, eFound / eTrue);

        _TEMP_scatterPlot_deltaEtaVsEt_->Fill(etFound, etaFound - etaTrue);
        _TEMP_scatterPlot_deltaEtaVsE_->Fill(eFound, etaFound - etaTrue);
        _TEMP_scatterPlot_deltaEtaVsEta_->Fill(etaFound, etaFound - etaTrue);
        _TEMP_scatterPlot_deltaEtaVsPhi_->Fill(phiFound, etaFound - etaTrue);

        _TEMP_scatterPlot_deltaPhiVsEt_->Fill(etFound, deltaPhi);
        _TEMP_scatterPlot_deltaPhiVsE_->Fill(eFound, deltaPhi);
        _TEMP_scatterPlot_deltaPhiVsEta_->Fill(etaFound, deltaPhi);
        _TEMP_scatterPlot_deltaPhiVsPhi_->Fill(phiFound, deltaPhi);

        electronsMCMatched.push_back(bestMatchElectron);
      }

      hist_EtNumRecoOverNumTrue_->Fill(etTrue);
      hist_ENumRecoOverNumTrue_->Fill(eTrue);
      hist_EtaNumRecoOverNumTrue_->Fill(etaTrue);
      hist_PhiNumRecoOverNumTrue_->Fill(phiTrue);
    }
  }

  if (electronsMCMatched.size() == 2) {
    reco::GsfElectron eOne = electronsMCMatched[0];
    reco::GsfElectron eTwo = electronsMCMatched[1];

    double recoMass = findRecoMass(eOne, eTwo);

    hist_All_recoMass_->Fill(recoMass);

    if (eOne.superCluster()->seed()->algo() == 1 && eTwo.superCluster()->seed()->algo() == 1)
      hist_BarrelOnly_recoMass_->Fill(recoMass);
    else if (eOne.superCluster()->seed()->algo() == 0 && eTwo.superCluster()->seed()->algo() == 0)
      hist_EndcapOnly_recoMass_->Fill(recoMass);
    else
      hist_Mixed_recoMass_->Fill(recoMass);
  }
}

double EgammaObjects::findRecoMass(const reco::Photon& pOne, const reco::Photon& pTwo) {
  double cosTheta = (cos(pOne.superCluster()->phi() - pTwo.superCluster()->phi()) +
                     sinh(pOne.superCluster()->eta()) * sinh(pTwo.superCluster()->eta())) /
                    (cosh(pOne.superCluster()->eta()) * cosh(pTwo.superCluster()->eta()));

  double recoMass = sqrt(2 * (pOne.superCluster())->energy() * (pTwo.superCluster())->energy() * (1 - cosTheta));

  return recoMass;
}

double EgammaObjects::findRecoMass(const reco::GsfElectron& eOne, const reco::GsfElectron& eTwo) {
  double cosTheta = (cos(eOne.caloPosition().phi() - eTwo.caloPosition().phi()) +
                     sinh(eOne.caloPosition().eta()) * sinh(eTwo.caloPosition().eta())) /
                    (cosh(eOne.caloPosition().eta()) * cosh(eTwo.caloPosition().eta()));

  double recoMass = sqrt(2 * eOne.caloEnergy() * eTwo.caloEnergy() * (1 - cosTheta));

  return recoMass;
}

float EgammaObjects::ecalEta(float EtaParticle, float Zvertex, float plane_Radius) {
  const float R_ECAL = 136.5;
  const float Z_Endcap = 328.0;
  const float etaBarrelEndcap = 1.479;

  if (EtaParticle != 0.) {
    float Theta = 0.0;
    float ZEcal = (R_ECAL - plane_Radius) * sinh(EtaParticle) + Zvertex;

    if (ZEcal != 0.0)
      Theta = atan(R_ECAL / ZEcal);
    if (Theta < 0.0)
      Theta = Theta + Geom::pi();

    float ETA = -log(tan(0.5 * Theta));

    if (std::abs(ETA) > etaBarrelEndcap) {
      float Zend = Z_Endcap;
      if (EtaParticle < 0.0)
        Zend = -Zend;
      float Zlen = Zend - Zvertex;
      float RR = Zlen / sinh(EtaParticle);
      Theta = atan((RR + plane_Radius) / Zend);
      if (Theta < 0.0)
        Theta = Theta + Geom::pi();
      ETA = -log(tan(0.5 * Theta));
    }

    return ETA;
  } else {
    edm::LogWarning("") << "[EgammaObjects::ecalEta] Warning: Eta equals to zero, not correcting";
    return EtaParticle;
  }
}

void EgammaObjects::endJob() {
  rootFile_->cd();
  rootFile_->mkdir(particleString.c_str());

  getDeltaResHistosViaSlicing();
  getEfficiencyHistosViaDividing();
  fitHistos();

  applyLabels();
  setDrawOptions();
  saveHistos();
  rootFile_->Close();
}

void EgammaObjects::getDeltaResHistosViaSlicing() {
  _TEMP_scatterPlot_EtOverTruthVsEt_->FitSlicesY(nullptr, 1, hist_bins_Et_, 10, "QRG3");
  _TEMP_scatterPlot_EtOverTruthVsE_->FitSlicesY(nullptr, 1, hist_bins_E_, 10, "QRG3");
  _TEMP_scatterPlot_EtOverTruthVsEta_->FitSlicesY(nullptr, 1, hist_bins_Eta_, 10, "QRG2");
  _TEMP_scatterPlot_EtOverTruthVsPhi_->FitSlicesY(nullptr, 1, hist_bins_Phi_, 10, "QRG2");

  _TEMP_scatterPlot_EOverTruthVsEt_->FitSlicesY(nullptr, 1, hist_bins_Et_, 10, "QRG3");
  _TEMP_scatterPlot_EOverTruthVsE_->FitSlicesY(nullptr, 1, hist_bins_E_, 10, "QRG3");
  _TEMP_scatterPlot_EOverTruthVsEta_->FitSlicesY(nullptr, 1, hist_bins_Eta_, 10, "QRG2");
  _TEMP_scatterPlot_EOverTruthVsPhi_->FitSlicesY(nullptr, 1, hist_bins_Phi_, 10, "QRG2");

  _TEMP_scatterPlot_deltaEtaVsEt_->FitSlicesY(nullptr, 1, hist_bins_Et_, 10, "QRG3");
  _TEMP_scatterPlot_deltaEtaVsE_->FitSlicesY(nullptr, 1, hist_bins_E_, 10, "QRG3");
  _TEMP_scatterPlot_deltaEtaVsEta_->FitSlicesY(nullptr, 1, hist_bins_Eta_, 10, "QRG2");
  _TEMP_scatterPlot_deltaEtaVsPhi_->FitSlicesY(nullptr, 1, hist_bins_Phi_, 10, "QRG2");

  _TEMP_scatterPlot_deltaPhiVsEt_->FitSlicesY(nullptr, 1, hist_bins_Et_, 10, "QRG3");
  _TEMP_scatterPlot_deltaPhiVsE_->FitSlicesY(nullptr, 1, hist_bins_E_, 10, "QRG3");
  _TEMP_scatterPlot_deltaPhiVsEta_->FitSlicesY(nullptr, 1, hist_bins_Eta_, 10, "QRG2");
  _TEMP_scatterPlot_deltaPhiVsPhi_->FitSlicesY(nullptr, 1, hist_bins_Phi_, 10, "QRG2");

  hist_EtOverTruthVsEt_ = (TH1D*)gDirectory->Get("_TEMP_scatterPlot_EtOverTruthVsEt__1");
  hist_EtOverTruthVsE_ = (TH1D*)gDirectory->Get("_TEMP_scatterPlot_EtOverTruthVsE__1");
  hist_EtOverTruthVsEta_ = (TH1D*)gDirectory->Get("_TEMP_scatterPlot_EtOverTruthVsEta__1");
  hist_EtOverTruthVsPhi_ = (TH1D*)gDirectory->Get("_TEMP_scatterPlot_EtOverTruthVsPhi__1");

  hist_EOverTruthVsEt_ = (TH1D*)gDirectory->Get("_TEMP_scatterPlot_EOverTruthVsEt__1");
  hist_EOverTruthVsE_ = (TH1D*)gDirectory->Get("_TEMP_scatterPlot_EOverTruthVsE__1");
  hist_EOverTruthVsEta_ = (TH1D*)gDirectory->Get("_TEMP_scatterPlot_EOverTruthVsEta__1");
  hist_EOverTruthVsPhi_ = (TH1D*)gDirectory->Get("_TEMP_scatterPlot_EOverTruthVsPhi__1");

  hist_deltaEtaVsEt_ = (TH1D*)gDirectory->Get("_TEMP_scatterPlot_deltaEtaVsEt__1");
  hist_deltaEtaVsE_ = (TH1D*)gDirectory->Get("_TEMP_scatterPlot_deltaEtaVsE__1");
  hist_deltaEtaVsEta_ = (TH1D*)gDirectory->Get("_TEMP_scatterPlot_deltaEtaVsEta__1");
  hist_deltaEtaVsPhi_ = (TH1D*)gDirectory->Get("_TEMP_scatterPlot_deltaEtaVsPhi__1");

  hist_deltaPhiVsEt_ = (TH1D*)gDirectory->Get("_TEMP_scatterPlot_deltaPhiVsEt__1");
  hist_deltaPhiVsE_ = (TH1D*)gDirectory->Get("_TEMP_scatterPlot_deltaPhiVsE__1");
  hist_deltaPhiVsEta_ = (TH1D*)gDirectory->Get("_TEMP_scatterPlot_deltaPhiVsEta__1");
  hist_deltaPhiVsPhi_ = (TH1D*)gDirectory->Get("_TEMP_scatterPlot_deltaPhiVsPhi__1");

  hist_EtOverTruthVsEt_->SetNameTitle("hist_EtOverTruthVsEt_",
                                      ("Reco Et over True Et VS Et of " + particleString).c_str());
  hist_EtOverTruthVsE_->SetNameTitle("hist_EtOverTruthVsE_",
                                     ("Reco Et over True Et VS E of " + particleString).c_str());
  hist_EtOverTruthVsEta_->SetNameTitle("hist_EtOverTruthVsEta_",
                                       ("Reco Et over True Et VS Eta of " + particleString).c_str());
  hist_EtOverTruthVsPhi_->SetNameTitle("hist_EtOverTruthVsPhi_",
                                       ("Reco Et over True Et VS Phi of " + particleString).c_str());

  hist_EOverTruthVsEt_->SetNameTitle("hist_EOverTruthVsEt_", ("Reco E over True E VS Et of " + particleString).c_str());
  hist_EOverTruthVsE_->SetNameTitle("hist_EOverTruthVsE_", ("Reco E over True E VS E of " + particleString).c_str());
  hist_EOverTruthVsEta_->SetNameTitle("hist_EOverTruthVsEta_",
                                      ("Reco E over True E VS Eta of " + particleString).c_str());
  hist_EOverTruthVsPhi_->SetNameTitle("hist_EOverTruthVsPhi_",
                                      ("Reco E over True E VS Phi of " + particleString).c_str());

  hist_deltaEtaVsEt_->SetNameTitle("hist_deltaEtaVsEt_", ("delta Eta VS Et of " + particleString).c_str());
  hist_deltaEtaVsE_->SetNameTitle("hist_deltaEtaVsE_", ("delta Eta VS E of " + particleString).c_str());
  hist_deltaEtaVsEta_->SetNameTitle("hist_deltaEtaVsEta_", ("delta Eta VS Eta of " + particleString).c_str());
  hist_deltaEtaVsPhi_->SetNameTitle("hist_deltaEtaVsPhi_", ("delta Eta VS Phi of " + particleString).c_str());

  hist_deltaPhiVsEt_->SetNameTitle("hist_deltaPhiVsEt_", ("delta Phi VS Et of " + particleString).c_str());
  hist_deltaPhiVsE_->SetNameTitle("hist_deltaPhiVsE_", ("delta Phi VS E of " + particleString).c_str());
  hist_deltaPhiVsEta_->SetNameTitle("hist_deltaPhiVsEta_", ("delta Phi VS Eta of " + particleString).c_str());
  hist_deltaPhiVsPhi_->SetNameTitle("hist_deltaPhiVsPhi_", ("delta Phi VS Phi of " + particleString).c_str());

  hist_resolutionEtVsEt_ = (TH1D*)gDirectory->Get("_TEMP_scatterPlot_EtOverTruthVsEt__2");
  hist_resolutionEtVsE_ = (TH1D*)gDirectory->Get("_TEMP_scatterPlot_EtOverTruthVsE__2");
  hist_resolutionEtVsEta_ = (TH1D*)gDirectory->Get("_TEMP_scatterPlot_EtOverTruthVsEta__2");
  hist_resolutionEtVsPhi_ = (TH1D*)gDirectory->Get("_TEMP_scatterPlot_EtOverTruthVsPhi__2");

  hist_resolutionEVsEt_ = (TH1D*)gDirectory->Get("_TEMP_scatterPlot_EOverTruthVsEt__2");
  hist_resolutionEVsE_ = (TH1D*)gDirectory->Get("_TEMP_scatterPlot_EOverTruthVsE__2");
  hist_resolutionEVsEta_ = (TH1D*)gDirectory->Get("_TEMP_scatterPlot_EOverTruthVsEta__2");
  hist_resolutionEVsPhi_ = (TH1D*)gDirectory->Get("_TEMP_scatterPlot_EOverTruthVsPhi__2");

  hist_resolutionEtaVsEt_ = (TH1D*)gDirectory->Get("_TEMP_scatterPlot_deltaEtaVsEt__2");
  hist_resolutionEtaVsE_ = (TH1D*)gDirectory->Get("_TEMP_scatterPlot_deltaEtaVsE__2");
  hist_resolutionEtaVsEta_ = (TH1D*)gDirectory->Get("_TEMP_scatterPlot_deltaEtaVsEta__2");
  hist_resolutionEtaVsPhi_ = (TH1D*)gDirectory->Get("_TEMP_scatterPlot_deltaEtaVsPhi__2");

  hist_resolutionPhiVsEt_ = (TH1D*)gDirectory->Get("_TEMP_scatterPlot_deltaPhiVsEt__2");
  hist_resolutionPhiVsE_ = (TH1D*)gDirectory->Get("_TEMP_scatterPlot_deltaPhiVsE__2");
  hist_resolutionPhiVsEta_ = (TH1D*)gDirectory->Get("_TEMP_scatterPlot_deltaPhiVsEta__2");
  hist_resolutionPhiVsPhi_ = (TH1D*)gDirectory->Get("_TEMP_scatterPlot_deltaPhiVsPhi__2");

  hist_resolutionEtVsEt_->SetNameTitle("hist_resolutionEtVsEt_",
                                       ("#sigma of Reco Et over True Et VS Et of " + particleString).c_str());
  hist_resolutionEtVsE_->SetNameTitle("hist_resolutionEtVsE_",
                                      ("#sigma of Reco Et over True Et VS E of " + particleString).c_str());
  hist_resolutionEtVsEta_->SetNameTitle("hist_resolutionEtVsEta_",
                                        ("#sigma of Reco Et over True Et VS Eta of " + particleString).c_str());
  hist_resolutionEtVsPhi_->SetNameTitle("hist_resolutionEtVsPhi_",
                                        ("#sigma of Reco Et over True Et VS Phi of " + particleString).c_str());

  hist_resolutionEVsEt_->SetNameTitle("hist_resolutionEVsEt_",
                                      ("#sigma of Reco E over True E VS Et of " + particleString).c_str());
  hist_resolutionEVsE_->SetNameTitle("hist_resolutionEVsE_",
                                     ("#sigma of Reco E over True E VS E of " + particleString).c_str());
  hist_resolutionEVsEta_->SetNameTitle("hist_resolutionEVsEta_",
                                       ("#sigma of Reco E over True E VS Eta of " + particleString).c_str());
  hist_resolutionEVsPhi_->SetNameTitle("hist_resolutionEVsPhi_",
                                       ("#sigma of Reco E over True E VS Phi of " + particleString).c_str());

  hist_resolutionEtaVsEt_->SetNameTitle("hist_resolutionEtaVsEt_",
                                        ("#sigma of delta Eta VS Et of " + particleString).c_str());
  hist_resolutionEtaVsE_->SetNameTitle("hist_resolutionEtaVsE_",
                                       ("#sigma of delta Eta VS E of " + particleString).c_str());
  hist_resolutionEtaVsEta_->SetNameTitle("hist_resolutionEtaVsEta_",
                                         ("#sigma of delta Eta VS Eta of " + particleString).c_str());
  hist_resolutionEtaVsPhi_->SetNameTitle("hist_resolutionEtaVsPhi_",
                                         ("#sigma of delta Eta VS Phi of " + particleString).c_str());

  hist_resolutionPhiVsEt_->SetNameTitle("hist_resolutionPhiVsEt_",
                                        ("#sigma of delta Phi VS Et of " + particleString).c_str());
  hist_resolutionPhiVsE_->SetNameTitle("hist_resolutionPhiVsE_",
                                       ("#sigma of delta Phi VS E of " + particleString).c_str());
  hist_resolutionPhiVsEta_->SetNameTitle("hist_resolutionPhiVsEta_",
                                         ("#sigma of delta Phi VS Eta of " + particleString).c_str());
  hist_resolutionPhiVsPhi_->SetNameTitle("hist_resolutionPhiVsPhi_",
                                         ("#sigma of delta Phi VS Phi of " + particleString).c_str());
}

void EgammaObjects::getEfficiencyHistosViaDividing() {
  hist_EtEfficiency_->Divide(hist_EtEfficiency_, hist_EtNumRecoOverNumTrue_, 1, 1);
  hist_EEfficiency_->Divide(hist_EEfficiency_, hist_ENumRecoOverNumTrue_, 1, 1);
  hist_EtaEfficiency_->Divide(hist_EtaEfficiency_, hist_EtaNumRecoOverNumTrue_, 1, 1);
  hist_PhiEfficiency_->Divide(hist_PhiEfficiency_, hist_PhiNumRecoOverNumTrue_, 1, 1);

  hist_EtNumRecoOverNumTrue_->Divide(hist_Et_, hist_EtNumRecoOverNumTrue_, 1, 1);
  hist_ENumRecoOverNumTrue_->Divide(hist_E_, hist_ENumRecoOverNumTrue_, 1, 1);
  hist_EtaNumRecoOverNumTrue_->Divide(hist_Eta_, hist_EtaNumRecoOverNumTrue_, 1, 1);
  hist_PhiNumRecoOverNumTrue_->Divide(hist_Phi_, hist_PhiNumRecoOverNumTrue_, 1, 1);
}

void EgammaObjects::fitHistos() {
  //Use our own copy for thread safety
  TF1 gaus("mygaus", "gaus");
  hist_EtOverTruth_->Fit(&gaus, "QEM");
  //  hist_EtNumRecoOverNumTrue_->Fit("pol1","QEM");

  hist_EOverTruth_->Fit(&gaus, "QEM");
  //  hist_ENumRecoOverNumTrue_->Fit("pol1","QEM");

  hist_EtaOverTruth_->Fit(&gaus, "QEM");
  //  hist_EtaNumRecoOverNumTrue_->Fit("pol1","QEM");

  hist_PhiOverTruth_->Fit(&gaus, "QEM");
  //  hist_PhiNumRecoOverNumTrue_->Fit("pol1","QEM");

  /*
  hist_EtOverTruthVsEt_->Fit("pol1","QEM");
  hist_EtOverTruthVsEta_->Fit("pol1","QEM");
  hist_EtOverTruthVsPhi_->Fit("pol1","QEM");
  hist_resolutionEtVsEt_->Fit("pol1","QEM");
  hist_resolutionEtVsEta_->Fit("pol1","QEM");
  hist_resolutionEtVsPhi_->Fit("pol1","QEM");

  hist_EOverTruthVsEt_->Fit("pol1","QEM");
  hist_EOverTruthVsEta_->Fit("pol1","QEM");
  hist_EOverTruthVsPhi_->Fit("pol1","QEM");
  hist_resolutionEVsEt_->Fit("pol1","QEM");
  hist_resolutionEVsEta_->Fit("pol1","QEM");
  hist_resolutionEVsPhi_->Fit("pol1","QEM");

  hist_deltaEtaVsEt_->Fit("pol1","QEM");
  hist_deltaEtaVsEta_->Fit("pol1","QEM");
  hist_deltaEtaVsPhi_->Fit("pol1","QEM");
  hist_resolutionEtaVsEt_->Fit("pol1","QEM");
  hist_resolutionEtaVsEta_->Fit("pol1","QEM");
  hist_resolutionEtaVsPhi_->Fit("pol1","QEM");

  hist_deltaPhiVsEt_->Fit("pol1","QEM");
  hist_deltaPhiVsEta_->Fit("pol1","QEM");
  hist_deltaPhiVsPhi_->Fit("pol1","QEM");
  hist_resolutionPhiVsEt_->Fit("pol1","QEM");
  hist_resolutionPhiVsEta_->Fit("pol1","QEM");
  hist_resolutionPhiVsPhi_->Fit("pol1","QEM");
  */
}

void EgammaObjects::applyLabels() {
  hist_Et_->GetXaxis()->SetTitle("Et (GeV)");
  hist_Et_->GetYaxis()->SetTitle("# per Et Bin");
  hist_EtOverTruth_->GetXaxis()->SetTitle("Reco Et/True Et");
  hist_EtOverTruth_->GetYaxis()->SetTitle("# per Ratio Bin");
  hist_EtEfficiency_->GetXaxis()->SetTitle("Et (GeV)");
  hist_EtEfficiency_->GetYaxis()->SetTitle("# True Reconstructed/# True per Et Bin");
  hist_EtNumRecoOverNumTrue_->GetXaxis()->SetTitle("Et (GeV)");
  hist_EtNumRecoOverNumTrue_->GetYaxis()->SetTitle("# Reco/# True per Et Bin");
  hist_EtOverTruthVsEt_->GetXaxis()->SetTitle("Et (GeV)");
  hist_EtOverTruthVsEt_->GetYaxis()->SetTitle("Reco Et/True Et per Et Bin");
  hist_EtOverTruthVsE_->GetXaxis()->SetTitle("E (GeV)");
  hist_EtOverTruthVsE_->GetYaxis()->SetTitle("Reco Et/True Et per E Bin");
  hist_EtOverTruthVsEta_->GetXaxis()->SetTitle("#eta (Radians)");
  hist_EtOverTruthVsEta_->GetYaxis()->SetTitle("Reco Et/True Et per Eta Bin");
  hist_EtOverTruthVsPhi_->GetXaxis()->SetTitle("#phi (Radians)");
  hist_EtOverTruthVsPhi_->GetYaxis()->SetTitle("Reco Et/True Et per Phi Bin");
  hist_resolutionEtVsEt_->GetXaxis()->SetTitle("Et (GeV)");
  hist_resolutionEtVsEt_->GetYaxis()->SetTitle("#sigma of Reco Et/True Et per Et Bin");
  hist_resolutionEtVsE_->GetXaxis()->SetTitle("E (GeV)");
  hist_resolutionEtVsE_->GetYaxis()->SetTitle("#sigma of Reco Et/True Et per E Bin");
  hist_resolutionEtVsEta_->GetXaxis()->SetTitle("#eta (Radians)");
  hist_resolutionEtVsEta_->GetYaxis()->SetTitle("#sigma of Reco Et/True Et per Eta Bin");
  hist_resolutionEtVsPhi_->GetXaxis()->SetTitle("#phi (Radians)");
  hist_resolutionEtVsPhi_->GetYaxis()->SetTitle("#sigma of Reco Et/True Et per Phi Bin");

  hist_E_->GetXaxis()->SetTitle("E (GeV)");
  hist_E_->GetYaxis()->SetTitle("# per E Bin");
  hist_EOverTruth_->GetXaxis()->SetTitle("Reco E/True E");
  hist_EOverTruth_->GetYaxis()->SetTitle("# per Ratio Bin");
  hist_EEfficiency_->GetXaxis()->SetTitle("E (GeV)");
  hist_EEfficiency_->GetYaxis()->SetTitle("# True Reconstructed/# True per E Bin");
  hist_ENumRecoOverNumTrue_->GetXaxis()->SetTitle("E (GeV)");
  hist_ENumRecoOverNumTrue_->GetYaxis()->SetTitle("# Reco/# True per E Bin");
  hist_EOverTruthVsEt_->GetXaxis()->SetTitle("Et (GeV)");
  hist_EOverTruthVsEt_->GetYaxis()->SetTitle("Reco E/True E per Et Bin");
  hist_EOverTruthVsE_->GetXaxis()->SetTitle("E (GeV)");
  hist_EOverTruthVsE_->GetYaxis()->SetTitle("Reco E/True E per E Bin");
  hist_EOverTruthVsEta_->GetXaxis()->SetTitle("#eta (Radians)");
  hist_EOverTruthVsEta_->GetYaxis()->SetTitle("Reco E/True E per Eta Bin");
  hist_EOverTruthVsPhi_->GetXaxis()->SetTitle("#phi (Radians)");
  hist_EOverTruthVsPhi_->GetYaxis()->SetTitle("Reco E/True E per Phi Bin");
  hist_resolutionEVsEt_->GetXaxis()->SetTitle("Et (GeV)");
  hist_resolutionEVsEt_->GetYaxis()->SetTitle("#sigma of Reco E/True E per Et Bin");
  hist_resolutionEVsE_->GetXaxis()->SetTitle("E (GeV)");
  hist_resolutionEVsE_->GetYaxis()->SetTitle("#sigma of Reco E/True E per E Bin");
  hist_resolutionEVsEta_->GetXaxis()->SetTitle("#eta (Radians)");
  hist_resolutionEVsEta_->GetYaxis()->SetTitle("#sigma of Reco E/True E per Eta Bin");
  hist_resolutionEVsPhi_->GetXaxis()->SetTitle("#phi (Radians)");
  hist_resolutionEVsPhi_->GetYaxis()->SetTitle("#sigma of Reco E/True E per Phi Bin");

  hist_Eta_->GetXaxis()->SetTitle("#eta (Radians)");
  hist_Eta_->GetYaxis()->SetTitle("# per Eta Bin");
  hist_EtaOverTruth_->GetXaxis()->SetTitle("Reco Eta/True Eta");
  hist_EtaOverTruth_->GetYaxis()->SetTitle("# per Ratio Bin");
  hist_EtaEfficiency_->GetXaxis()->SetTitle("#eta (Radians)");
  hist_EtaEfficiency_->GetYaxis()->SetTitle("# True Reconstructed/# True per Eta Bin");
  hist_EtaNumRecoOverNumTrue_->GetXaxis()->SetTitle("#eta (Radians)");
  hist_EtaNumRecoOverNumTrue_->GetYaxis()->SetTitle("# Reco/# True per Eta Bin");
  hist_deltaEtaVsEt_->GetXaxis()->SetTitle("Et (GeV)");
  hist_deltaEtaVsEt_->GetYaxis()->SetTitle("Reco Eta - True Eta per Et Bin");
  hist_deltaEtaVsE_->GetXaxis()->SetTitle("E (GeV)");
  hist_deltaEtaVsE_->GetYaxis()->SetTitle("Reco Eta - True Eta per E Bin");
  hist_deltaEtaVsEta_->GetXaxis()->SetTitle("#eta (Radians)");
  hist_deltaEtaVsEta_->GetYaxis()->SetTitle("Reco Eta - True Eta per Eta Bin");
  hist_deltaEtaVsPhi_->GetXaxis()->SetTitle("#phi (Radians)");
  hist_deltaEtaVsPhi_->GetYaxis()->SetTitle("Reco Eta - True Eta per Phi Bin");
  hist_resolutionEtaVsEt_->GetXaxis()->SetTitle("Et (GeV)");
  hist_resolutionEtaVsEt_->GetYaxis()->SetTitle("#sigma of Reco Eta - True Eta per Et Bin");
  hist_resolutionEtaVsE_->GetXaxis()->SetTitle("E (GeV)");
  hist_resolutionEtaVsE_->GetYaxis()->SetTitle("#sigma of Reco Eta - True Eta per E Bin");
  hist_resolutionEtaVsEta_->GetXaxis()->SetTitle("#eta (Radians)");
  hist_resolutionEtaVsEta_->GetYaxis()->SetTitle("#sigma of Reco Eta - True Eta per Eta Bin");
  hist_resolutionEtaVsPhi_->GetXaxis()->SetTitle("#phi (Radians)");
  hist_resolutionEtaVsPhi_->GetYaxis()->SetTitle("#sigma of Reco Eta - True Eta per Phi Bin");

  hist_Phi_->GetXaxis()->SetTitle("#phi (Radians)");
  hist_Phi_->GetYaxis()->SetTitle("# per Phi Bin");
  hist_PhiOverTruth_->GetXaxis()->SetTitle("Reco Phi/True Phi");
  hist_PhiOverTruth_->GetYaxis()->SetTitle("# per Ratio Bin");
  hist_PhiEfficiency_->GetXaxis()->SetTitle("#phi (Radians)");
  hist_PhiEfficiency_->GetYaxis()->SetTitle("# True Reconstructed/# True per Phi Bin");
  hist_PhiNumRecoOverNumTrue_->GetXaxis()->SetTitle("#Phi (Radians)");
  hist_PhiNumRecoOverNumTrue_->GetYaxis()->SetTitle("# Reco/# True per Phi Bin");
  hist_deltaPhiVsEt_->GetXaxis()->SetTitle("Et (GeV)");
  hist_deltaPhiVsEt_->GetYaxis()->SetTitle("Reco Phi - True Phi per Et Bin");
  hist_deltaPhiVsE_->GetXaxis()->SetTitle("E (GeV)");
  hist_deltaPhiVsE_->GetYaxis()->SetTitle("Reco Phi - True Phi per E Bin");
  hist_deltaPhiVsEta_->GetXaxis()->SetTitle("#eta (Radians)");
  hist_deltaPhiVsEta_->GetYaxis()->SetTitle("Reco Phi - True Phi per Eta Bin");
  hist_deltaPhiVsPhi_->GetXaxis()->SetTitle("#phi (Radians)");
  hist_deltaPhiVsPhi_->GetYaxis()->SetTitle("Reco Phi - True Phi per Phi Bin");
  hist_resolutionPhiVsEt_->GetXaxis()->SetTitle("Et (GeV)");
  hist_resolutionPhiVsEt_->GetYaxis()->SetTitle("#sigma of Reco Phi - True Phi per Et Bin");
  hist_resolutionPhiVsE_->GetXaxis()->SetTitle("E (GeV)");
  hist_resolutionPhiVsE_->GetYaxis()->SetTitle("#sigma of Reco Phi - True Phi per E Bin");
  hist_resolutionPhiVsEta_->GetXaxis()->SetTitle("#eta (Radians)");
  hist_resolutionPhiVsEta_->GetYaxis()->SetTitle("#sigma of Reco Phi - True Phi per Eta Bin");
  hist_resolutionPhiVsPhi_->GetXaxis()->SetTitle("#phi (Radians)");
  hist_resolutionPhiVsPhi_->GetYaxis()->SetTitle("#sigma of Reco Phi - True Phi per Phi Bin");

  std::string recoParticleName;

  if (particleID == 22)
    recoParticleName = "Reco Higgs";
  else if (particleID == 11)
    recoParticleName = "Reco Z";

  hist_All_recoMass_->GetXaxis()->SetTitle((recoParticleName + " Mass (GeV)").c_str());
  hist_All_recoMass_->GetYaxis()->SetTitle("# of Reco Masses per Mass Bin");
  hist_BarrelOnly_recoMass_->GetXaxis()->SetTitle((recoParticleName + " Mass (GeV)").c_str());
  hist_BarrelOnly_recoMass_->GetYaxis()->SetTitle("# of Reco Masses per Mass Bin");
  hist_EndcapOnly_recoMass_->GetXaxis()->SetTitle((recoParticleName + " Mass (GeV)").c_str());
  hist_EndcapOnly_recoMass_->GetYaxis()->SetTitle("# of Reco Masses per Mass Bin");
  hist_Mixed_recoMass_->GetXaxis()->SetTitle((recoParticleName + " Mass (GeV)").c_str());
  hist_Mixed_recoMass_->GetYaxis()->SetTitle("# of Reco Masses per Mass Bin");
  hist_recoMass_withBackgroud_NoEtCut_->GetXaxis()->SetTitle((recoParticleName + " Mass (GeV)").c_str());
  hist_recoMass_withBackgroud_NoEtCut_->GetYaxis()->SetTitle("# of Reco Masses per Mass Bin");
  hist_recoMass_withBackgroud_5EtCut_->GetXaxis()->SetTitle((recoParticleName + " Mass (GeV)").c_str());
  hist_recoMass_withBackgroud_5EtCut_->GetYaxis()->SetTitle("# of Reco Masses per Mass Bin");
  hist_recoMass_withBackgroud_10EtCut_->GetXaxis()->SetTitle((recoParticleName + " Mass (GeV)").c_str());
  hist_recoMass_withBackgroud_10EtCut_->GetYaxis()->SetTitle("# of Reco Masses per Mass Bin");
  hist_recoMass_withBackgroud_20EtCut_->GetXaxis()->SetTitle((recoParticleName + " Mass (GeV)").c_str());
  hist_recoMass_withBackgroud_20EtCut_->GetYaxis()->SetTitle("# of Reco Masses per Mass Bin");
}

void EgammaObjects::setDrawOptions() {
  hist_Et_->SetOption("e");
  hist_EtOverTruth_->SetOption("e");
  hist_EtEfficiency_->SetOption("e");
  hist_EtNumRecoOverNumTrue_->SetOption("e");
  hist_EtOverTruthVsEt_->SetOption("e");
  hist_EtOverTruthVsE_->SetOption("e");
  hist_EtOverTruthVsEta_->SetOption("e");
  hist_EtOverTruthVsPhi_->SetOption("e");
  hist_resolutionEtVsEt_->SetOption("e");
  hist_resolutionEtVsE_->SetOption("e");
  hist_resolutionEtVsEta_->SetOption("e");
  hist_resolutionEtVsPhi_->SetOption("e");

  hist_E_->SetOption("e");
  hist_EOverTruth_->SetOption("e");
  hist_EEfficiency_->SetOption("e");
  hist_ENumRecoOverNumTrue_->SetOption("e");
  hist_EOverTruthVsEt_->SetOption("e");
  hist_EOverTruthVsE_->SetOption("e");
  hist_EOverTruthVsEta_->SetOption("e");
  hist_EOverTruthVsPhi_->SetOption("e");
  hist_resolutionEVsEt_->SetOption("e");
  hist_resolutionEVsE_->SetOption("e");
  hist_resolutionEVsEta_->SetOption("e");
  hist_resolutionEVsPhi_->SetOption("e");

  hist_Eta_->SetOption("e");
  hist_EtaOverTruth_->SetOption("e");
  hist_EtaEfficiency_->SetOption("e");
  hist_EtaNumRecoOverNumTrue_->SetOption("e");
  hist_deltaEtaVsEt_->SetOption("e");
  hist_deltaEtaVsE_->SetOption("e");
  hist_deltaEtaVsEta_->SetOption("e");
  hist_deltaEtaVsPhi_->SetOption("e");
  hist_resolutionEtaVsEt_->SetOption("e");
  hist_resolutionEtaVsE_->SetOption("e");
  hist_resolutionEtaVsEta_->SetOption("e");
  hist_resolutionEtaVsPhi_->SetOption("e");

  hist_Phi_->SetOption("e");
  hist_PhiOverTruth_->SetOption("e");
  hist_PhiEfficiency_->SetOption("e");
  hist_PhiNumRecoOverNumTrue_->SetOption("e");
  hist_deltaPhiVsEt_->SetOption("e");
  hist_deltaPhiVsE_->SetOption("e");
  hist_deltaPhiVsEta_->SetOption("e");
  hist_deltaPhiVsPhi_->SetOption("e");
  hist_resolutionPhiVsEt_->SetOption("e");
  hist_resolutionPhiVsE_->SetOption("e");
  hist_resolutionPhiVsEta_->SetOption("e");
  hist_resolutionPhiVsPhi_->SetOption("e");

  hist_All_recoMass_->SetOption("e");
  hist_BarrelOnly_recoMass_->SetOption("e");
  hist_EndcapOnly_recoMass_->SetOption("e");
  hist_Mixed_recoMass_->SetOption("e");
  hist_recoMass_withBackgroud_NoEtCut_->SetOption("e");
  hist_recoMass_withBackgroud_5EtCut_->SetOption("e");
  hist_recoMass_withBackgroud_10EtCut_->SetOption("e");
  hist_recoMass_withBackgroud_20EtCut_->SetOption("e");
}

void EgammaObjects::saveHistos() {
  rootFile_->cd();
  rootFile_->GetDirectory(particleString.c_str())->mkdir("ET");
  rootFile_->cd(("/" + particleString + "/ET").c_str());

  hist_Et_->Write();
  hist_EtOverTruth_->Write();
  hist_EtEfficiency_->Write();
  hist_EtNumRecoOverNumTrue_->Write();
  hist_EtOverTruthVsEt_->Write();
  hist_EtOverTruthVsE_->Write();
  hist_EtOverTruthVsEta_->Write();
  hist_EtOverTruthVsPhi_->Write();
  hist_resolutionEtVsEt_->Write();
  hist_resolutionEtVsE_->Write();
  hist_resolutionEtVsEta_->Write();
  hist_resolutionEtVsPhi_->Write();

  rootFile_->cd();
  rootFile_->GetDirectory(particleString.c_str())->mkdir("E");
  rootFile_->cd(("/" + particleString + "/E").c_str());

  hist_E_->Write();
  hist_EOverTruth_->Write();
  hist_EEfficiency_->Write();
  hist_ENumRecoOverNumTrue_->Write();
  hist_EOverTruthVsEt_->Write();
  hist_EOverTruthVsE_->Write();
  hist_EOverTruthVsEta_->Write();
  hist_EOverTruthVsPhi_->Write();
  hist_resolutionEVsEt_->Write();
  hist_resolutionEVsE_->Write();
  hist_resolutionEVsEta_->Write();
  hist_resolutionEVsPhi_->Write();

  rootFile_->cd();
  rootFile_->GetDirectory(particleString.c_str())->mkdir("Eta");
  rootFile_->cd(("/" + particleString + "/Eta").c_str());

  hist_Eta_->Write();
  hist_EtaOverTruth_->Write();
  hist_EtaEfficiency_->Write();
  hist_EtaNumRecoOverNumTrue_->Write();
  hist_deltaEtaVsEt_->Write();
  hist_deltaEtaVsE_->Write();
  hist_deltaEtaVsEta_->Write();
  hist_deltaEtaVsPhi_->Write();
  hist_resolutionEtaVsEt_->Write();
  hist_resolutionEtaVsE_->Write();
  hist_resolutionEtaVsEta_->Write();
  hist_resolutionEtaVsPhi_->Write();

  rootFile_->cd();
  rootFile_->GetDirectory(particleString.c_str())->mkdir("Phi");
  rootFile_->cd(("/" + particleString + "/Phi").c_str());

  hist_Phi_->Write();
  hist_PhiOverTruth_->Write();
  hist_PhiEfficiency_->Write();
  hist_PhiNumRecoOverNumTrue_->Write();
  hist_deltaPhiVsEt_->Write();
  hist_deltaPhiVsE_->Write();
  hist_deltaPhiVsEta_->Write();
  hist_deltaPhiVsPhi_->Write();
  hist_resolutionPhiVsEt_->Write();
  hist_resolutionPhiVsE_->Write();
  hist_resolutionPhiVsEta_->Write();
  hist_resolutionPhiVsPhi_->Write();

  std::string recoParticleName;

  if (particleID == 22)
    recoParticleName = "HiggsRecoMass";
  else if (particleID == 11)
    recoParticleName = "ZRecoMass";

  rootFile_->cd();
  rootFile_->GetDirectory(particleString.c_str())->mkdir(recoParticleName.c_str());
  rootFile_->cd(("/" + particleString + "/" + recoParticleName).c_str());

  hist_All_recoMass_->Write();
  hist_BarrelOnly_recoMass_->Write();
  hist_EndcapOnly_recoMass_->Write();
  hist_Mixed_recoMass_->Write();
  hist_recoMass_withBackgroud_NoEtCut_->Write();
  hist_recoMass_withBackgroud_5EtCut_->Write();
  hist_recoMass_withBackgroud_10EtCut_->Write();
  hist_recoMass_withBackgroud_20EtCut_->Write();

  rootFile_->cd();
  rootFile_->GetDirectory(particleString.c_str())->mkdir("_TempScatterPlots");
  rootFile_->cd(("/" + particleString + "/_TempScatterPlots").c_str());

  _TEMP_scatterPlot_EtOverTruthVsEt_->Write();
  _TEMP_scatterPlot_EtOverTruthVsE_->Write();
  _TEMP_scatterPlot_EtOverTruthVsEta_->Write();
  _TEMP_scatterPlot_EtOverTruthVsPhi_->Write();

  _TEMP_scatterPlot_EOverTruthVsEt_->Write();
  _TEMP_scatterPlot_EOverTruthVsE_->Write();
  _TEMP_scatterPlot_EOverTruthVsEta_->Write();
  _TEMP_scatterPlot_EOverTruthVsPhi_->Write();

  _TEMP_scatterPlot_deltaEtaVsEt_->Write();
  _TEMP_scatterPlot_deltaEtaVsE_->Write();
  _TEMP_scatterPlot_deltaEtaVsEta_->Write();
  _TEMP_scatterPlot_deltaEtaVsPhi_->Write();

  _TEMP_scatterPlot_deltaPhiVsEt_->Write();
  _TEMP_scatterPlot_deltaPhiVsE_->Write();
  _TEMP_scatterPlot_deltaPhiVsEta_->Write();
  _TEMP_scatterPlot_deltaPhiVsPhi_->Write();

  rootFile_->cd();
}
