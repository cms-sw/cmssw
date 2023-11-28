/*class TauValidation
 *  
 *  Class to fill dqm monitor elements from existing EDM file
 *
 */
#include "Validation/EventGenerator/interface/TauValidation.h"
#include "CLHEP/Units/defs.h"
#include "CLHEP/Units/PhysicalConstants.h"
#include "DataFormats/Math/interface/LorentzVector.h"
#include "Validation/EventGenerator/interface/TauDecay_GenParticle.h"
#include "Validation/EventGenerator/interface/PdtPdgMini.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "DataFormats/HepMCCandidate/interface/GenParticleFwd.h"
#include <iostream>
#include "Validation/EventGenerator/interface/DQMHelper.h"
using namespace edm;

TauValidation::TauValidation(const edm::ParameterSet &iPSet)
    :  //  wmanager_(iPSet,consumesCollector())
      genparticleCollection_(iPSet.getParameter<edm::InputTag>("genparticleCollection")),
      NMODEID(TauDecay::NMODEID - 1),  // fortran to C++ index
      zsbins(20),
      zsmin(-0.5),
      zsmax(0.5) {
  genparticleCollectionToken_ = consumes<reco::GenParticleCollection>(genparticleCollection_);
  fPDGTableToken = esConsumes<edm::Transition::BeginRun>();
}

TauValidation::~TauValidation() {}

void TauValidation::dqmBeginRun(const edm::Run &r, const edm::EventSetup &c) {
  fPDGTable = c.getHandle(fPDGTableToken);
}

void TauValidation::bookHistograms(DQMStore::IBooker &i, edm::Run const &, edm::EventSetup const &) {
  ///Setting the DQM top directories
  DQMHelper dqm(&i);
  i.setCurrentFolder("Generator/Tau");
  // Number of analyzed events
  nTaus = dqm.book1dHisto("nTaus", "n analyzed Taus", 1, 0., 1., "bin", "Number of #tau's found");
  nPrimeTaus =
      dqm.book1dHisto("nPrimeTaus", "n analyzed prime Taus", 1, 0., 1., "bin", "Number of #tau's from Gauge Bosons");

  //Kinematics
  TauPt = dqm.book1dHisto("TauPt", "Tau pT", 100, 0, 100, "P_{T}^{#tau}", "Number of #tau's from Gauge Bosons");
  TauEta = dqm.book1dHisto("TauEta", "Tau eta", 100, -2.5, 2.5, "#eta^{#tau}", "Number of #tau's from Gauge Bosons");
  TauPhi = dqm.book1dHisto("TauPhi", "Tau phi", 100, -3.14, 3.14, "#phi^{#tau}", "Number of #tau's from Gauge Bosons");
  TauProngs = dqm.book1dHisto("TauProngs", "Tau n prongs", 7, 0, 7, "N_{prongs}", "Number of #tau's from Gauge Bosons");
  TauDecayChannels = dqm.book1dHisto(
      "TauDecayChannels", "Tau decay channels", 13, 0, 13, "Tau POG Decay Mode", "Number of #tau's from Gauge Bosons");
  TauDecayChannels->setBinLabel(1 + undetermined, "?");
  TauDecayChannels->setBinLabel(1 + electron, "e");
  TauDecayChannels->setBinLabel(1 + muon, "mu");
  TauDecayChannels->setBinLabel(1 + pi, "#pi^{#pm}");
  TauDecayChannels->setBinLabel(1 + rho, "#rho^{#pm}");
  TauDecayChannels->setBinLabel(1 + a1, "a_{1}^{#pm}");
  TauDecayChannels->setBinLabel(1 + pi1pi0, "#pi^{#pm}#pi^{0}");
  TauDecayChannels->setBinLabel(1 + pinpi0, "#pi^{#pm}n#pi^{0}");
  TauDecayChannels->setBinLabel(1 + tripi, "3#pi^{#pm}");
  TauDecayChannels->setBinLabel(1 + tripinpi0, "3#pi^{#pm}n#pi^{0}");
  TauDecayChannels->setBinLabel(1 + K, "K");
  TauDecayChannels->setBinLabel(1 + Kstar, "K^{*}");
  TauDecayChannels->setBinLabel(1 + stable, "Stable");

  TauMothers = dqm.book1dHisto("TauMothers", "Tau mother particles", 10, 0, 10, "Mother of #tau", "Number of #tau's");

  TauMothers->setBinLabel(1 + other, "?");
  TauMothers->setBinLabel(1 + B, "B Decays");
  TauMothers->setBinLabel(1 + D, "D Decays");
  TauMothers->setBinLabel(1 + gamma, "#gamma");
  TauMothers->setBinLabel(1 + Z, "Z");
  TauMothers->setBinLabel(1 + W, "W");
  TauMothers->setBinLabel(1 + HSM, "H_{SM}/h^{0}");
  TauMothers->setBinLabel(1 + H0, "H^{0}");
  TauMothers->setBinLabel(1 + A0, "A^{0}");
  TauMothers->setBinLabel(1 + Hpm, "H^{#pm}");

  DecayLength = dqm.book1dHisto(
      "DecayLength", "#tau Decay Length", 100, -20, 20, "L_{#tau} (cm)", "Number of #tau's from Gauge Bosons");
  LifeTime = dqm.book1dHisto(
      "LifeTime", "#tau LifeTime ", 500, 0, 10000E-15, "#tau_{#tau} (s)", "Number of #tau's from Gauge Bosons");

  TauSpinEffectsW_X = dqm.book1dHisto(
      "TauSpinEffectsWX", "X for pion", 50, 0, 1, "X", "Number of #tau#rightarrow#pi#nu from W^{#pm} Bosons");
  TauSpinEffectsHpm_X = dqm.book1dHisto(
      "TauSpinEffectsHpmX", "X for pion", 50, 0, 1, "X", "Number of #tau#rightarrow#pi#nu from H^{#pm} Bosons");

  TauSpinEffectsW_eX = dqm.book1dHisto(
      "TauSpinEffectsWeX", "X for e", 50, 0, 1, "X", "Number of #tau#rightarrowe#nu#nu from W^{#pm} Bosons");
  TauSpinEffectsHpm_eX = dqm.book1dHisto(
      "TauSpinEffectsHpmeX", "X for e", 50, 0, 1, "X", "Number of #tau#rightarrowe#nu#nu from H^{#pm} Bosons");

  TauSpinEffectsW_muX = dqm.book1dHisto(
      "TauSpinEffectsWmuX", "X for mu", 50, 0, 1, "X", "Number of #tau#rightarrow#mu#nu#nu from W^{#pm} Bosons");
  TauSpinEffectsHpm_muX = dqm.book1dHisto(
      "TauSpinEffectsHpmmuX", "X for mue", 50, 0, 1, "X", "Number of #tau#rightarrow#mu#nu#nu from H^{#pm} Bosons");

  TauSpinEffectsW_UpsilonRho = dqm.book1dHisto("TauSpinEffectsWUpsilonRho",
                                               "#Upsilon for #rho",
                                               50,
                                               -1,
                                               1,
                                               "#Upsilon",
                                               "Number of #tau#rightarrow#rho#nu from Gauge Bosons");
  TauSpinEffectsHpm_UpsilonRho = dqm.book1dHisto("TauSpinEffectsHpmUpsilonRho",
                                                 "#Upsilon for #rho",
                                                 50,
                                                 -1,
                                                 1,
                                                 "#Upsilon",
                                                 "Number of #tau#rightarrow#rho#nu from Gauge Bosons");

  TauSpinEffectsW_UpsilonA1 = dqm.book1dHisto("TauSpinEffectsWUpsilonA1",
                                              "#Upsilon for a1",
                                              50,
                                              -1,
                                              1,
                                              "#Upsilon",
                                              "Number of #tau#rightarrow#pi#pi#pi#nu from Gauge Bosons");
  TauSpinEffectsHpm_UpsilonA1 = dqm.book1dHisto("TauSpinEffectsHpmUpsilonA1",
                                                "#Upsilon for a1",
                                                50,
                                                -1,
                                                1,
                                                "#Upsilon",
                                                "Number of #tau#rightarrow#pi#pi#pi#nu from Gauge Bosons");

  TauSpinEffectsH_pipiAcoplanarity =
      dqm.book1dHisto("TauSpinEffectsH_pipiAcoplanarity",
                      "H Acoplanarity for #pi^{-}#pi^{+}",
                      50,
                      0,
                      2 * TMath::Pi(),
                      "Acoplanarity",
                      "Number of H#rightarrow#tau^{-}(#rightarrow#pi^{-}#nu)#tau^{+}(#rightarrow#pi^{+}#nu) Events");

  TauSpinEffectsH_pipiAcollinearity =
      dqm.book1dHisto("TauSpinEffectsH_pipiAcollinearity",
                      "H Acollinearity for #pi^{-}#pi^{+}",
                      50,
                      0,
                      TMath::Pi(),
                      "Acollinearity",
                      "Number of H#rightarrow#tau^{-}(#rightarrow#pi^{-}#nu)#tau^{+}(#rightarrow#pi^{+}#nu) Events");
  TauSpinEffectsH_pipiAcollinearityzoom =
      dqm.book1dHisto("TauSpinEffectsH_pipiAcollinearityzoom",
                      "H Acollinearity for #pi^{-}#pi^{+}",
                      50,
                      3,
                      TMath::Pi(),
                      "Acollinearity",
                      "Number of H#rightarrow#tau^{-}(#rightarrow#pi^{-}#nu)#tau^{+}(#rightarrow#pi^{+}#nu) Events");

  TauSpinEffectsZ_MVis =
      dqm.book1dHisto("TauSpinEffectsZMVis",
                      "Mass of pi+ pi-",
                      25,
                      0,
                      1.1,
                      "M_{#pi^{+}#pi^{-}} (GeV)",
                      "Number of Z#rightarrow#tau^{-}(#rightarrow#pi^{-}#nu)#tau^{+}(#rightarrow#pi^{+}#nu) Events");
  TauSpinEffectsH_MVis =
      dqm.book1dHisto("TauSpinEffectsHMVis",
                      "Mass of pi+ pi-",
                      25,
                      0,
                      1.1,
                      "M_{#pi^{+}#pi^{-}} (GeV)",
                      "Number of H#rightarrow#tau^{-}(#rightarrow#pi^{-}#nu)#tau^{+}(#rightarrow#pi^{+}#nu) Events");

  TauSpinEffectsZ_Zs =
      dqm.book1dHisto("TauSpinEffectsZZs",
                      "Z_{s}",
                      zsbins,
                      zsmin,
                      zsmax,
                      "Z_{s}",
                      "Number of Z#rightarrow#tau^{-}(#rightarrow#pi^{-}#nu)#tau^{+}(#rightarrow#pi^{+}#nu Events");
  TauSpinEffectsH_Zs =
      dqm.book1dHisto("TauSpinEffectsHZs",
                      "Z_{s}",
                      zsbins,
                      zsmin,
                      zsmax,
                      "Z_{s}",
                      "Number of H#rightarrow#tau^{-}(#rightarrow#pi^{-}#nu)#tau^{+}(#rightarrow#pi^{+}#nu Events");

  TauSpinEffectsZ_X = dqm.book1dHisto(
      "TauSpinEffectsZX", "X for pion of #tau^{-}", 25, 0, 1.0, "X", "Number of #tau#rightarrow#pi#nu from Z Bosons");
  TauSpinEffectsZ_X50to75 = dqm.book1dHisto("TauSpinEffectsZX50to75",
                                            "X for pion of #tau^{-} (50GeV-75GeV)",
                                            10,
                                            0,
                                            1.0,
                                            "X",
                                            "Number of #tau#rightarrow#pi#nu from Z(50GeV<M<75GeV) Bosons");
  TauSpinEffectsZ_X75to88 = dqm.book1dHisto("TauSpinEffectsZX75to88",
                                            "X for pion of #tau^{-} (75GeV-88GeV)",
                                            10,
                                            0,
                                            1.0,
                                            "X",
                                            "Number of #tau#rightarrow#pi#nu from Z(75GeV<M<88GeV) Bosons");
  TauSpinEffectsZ_X88to100 = dqm.book1dHisto("TauSpinEffectsZX88to100",
                                             "X for pion of #tau^{-} (88GeV-100GeV)",
                                             10,
                                             0,
                                             1.0,
                                             "X",
                                             "Number of #tau#rightarrow#pi#nu from Z(88GeV<M<100GeV) Bosons");
  TauSpinEffectsZ_X100to120 = dqm.book1dHisto("TauSpinEffectsZX100to120",
                                              "X for pion of #tau^{-} (100GeV-120GeV)",
                                              10,
                                              0,
                                              1.0,
                                              "X",
                                              "Number of #tau#rightarrow#pi#nu from Z(100GeV<M<120GeV) Bosons");
  TauSpinEffectsZ_X120UP = dqm.book1dHisto("TauSpinEffectsZX120UP",
                                           "X for pion of #tau^{-} (>120GeV)",
                                           10,
                                           0,
                                           1.0,
                                           "X",
                                           "Number of #tau#rightarrow#pi#nu from Z(120GeV<MGeV) Bosons");

  TauSpinEffectsH_X = dqm.book1dHisto(
      "TauSpinEffectsH_X", "X for pion of #tau^{-}", 25, 0, 1.0, "X", "Number of #tau#rightarrow#pi#nu from H Bosons");

  TauSpinEffectsZ_Xf = dqm.book1dHisto("TauSpinEffectsZXf",
                                       "X for pion of forward emitted #tau^{-}",
                                       25,
                                       0,
                                       1.0,
                                       "X_{f}",
                                       "Number of #tau#rightarrow#pi#nu from Z Bosons");
  TauSpinEffectsH_Xf = dqm.book1dHisto("TauSpinEffectsHXf",
                                       "X for pion of forward emitted #tau^{-}",
                                       25,
                                       0,
                                       1.0,
                                       "X_{f}",
                                       "Number of #tau#rightarrow#pi#nu from H Bosons");

  TauSpinEffectsZ_Xb = dqm.book1dHisto("TauSpinEffectsZXb",
                                       "X for pion of backward emitted #tau^{-}",
                                       25,
                                       0,
                                       1.0,
                                       "X_{b}",
                                       "Number of #tau#rightarrow#pi#nu from Z Bosons");
  TauSpinEffectsH_Xb = dqm.book1dHisto("TauSpinEffectsHXb",
                                       "X for pion of backward emitted #tau^{-}",
                                       25,
                                       0,
                                       1.0,
                                       "X_{b}",
                                       "Number of #tau#rightarrow#pi#nu from H Bosons");

  TauSpinEffectsZ_eX = dqm.book1dHisto(
      "TauSpinEffectsZeX", "X for e", 50, 0, 1, "X", "Number of #tau#rightarrowe#nu#nu from Gauge Bosons");
  TauSpinEffectsH_eX = dqm.book1dHisto(
      "TauSpinEffectsHeX", "X for e", 50, 0, 1, "X", "Number of #tau#rightarrowe#nu#nu from Gauge Bosons");

  TauSpinEffectsZ_muX = dqm.book1dHisto(
      "TauSpinEffectsZmuX", "X for mu", 50, 0, 1, "X", "Number of #tau#rightarrow#mu#nu#nu from Gauge Bosons");
  TauSpinEffectsH_muX = dqm.book1dHisto(
      "TauSpinEffectsHmuX", "X for mu", 50, 0, 1, "X", "Number of #tau#rightarrow#mu#nu#nu from Gauge Bosons");

  TauSpinEffectsH_rhorhoAcoplanarityminus =
      dqm.book1dHisto("TauSpinEffectsH_rhorhoAcoplanarityminus",
                      "#phi^{*-} (acoplanarity) for Higgs #rightarrow #rho-#rho (y_{1}*y_{2}<0)",
                      32,
                      0,
                      2 * TMath::Pi(),
                      "#phi^{*-} (Acoplanarity)",
                      "Number of H#rightarrow#tau^{-}(#rightarrow#rho^{-}#nu)#tau^{+}(#rightarrow#rho^{+}#nu) Events");
  TauSpinEffectsH_rhorhoAcoplanarityplus =
      dqm.book1dHisto("TauSpinEffectsH_rhorhoAcoplanarityplus",
                      "#phi^{*+} (acoplanarity) for Higgs #rightarrow #rho-#rho (y_{1}*y_{2}>0)",
                      32,
                      0,
                      2 * TMath::Pi(),
                      "#phi^{*+} (Acoplanarity)",
                      "Number of H#rightarrow#tau^{-}(#rightarrow#rho^{-}#nu)#tau^{+}(#rightarrow#rho^{+}#nu) Events");

  TauFSRPhotonsN = dqm.book1dHisto("TauFSRPhotonsN",
                                   "FSR Photons radiating from/with tau (Gauge Boson)",
                                   5,
                                   -0.5,
                                   4.5,
                                   "N^{FSR Photons radiating from/with #tau}",
                                   "Number of #tau's from Gauge Bosons");
  TauFSRPhotonsPt = dqm.book1dHisto("TauFSRPhotonsPt",
                                    "Pt of FSR Photons radiating from/with tau (Gauge Boson)",
                                    100,
                                    0,
                                    100,
                                    "P_{t}^{FSR Photons radiating from/with #tau [per #tau]} (GeV)",
                                    "Number of #tau's from Gauge Bosons");
  TauFSRPhotonsPtSum = dqm.book1dHisto("TauFSRPhotonsPtSum",
                                       "Pt of FSR Photons radiating from/with tau (Gauge Boson)",
                                       100,
                                       0,
                                       100,
                                       "P_{t}^{FSR Photons radiating from/with #tau [per #tau]} (GeV)",
                                       "Number of #tau's from Gauge Bosons");

  TauBremPhotonsN = dqm.book1dHisto("TauBremPhotonsN",
                                    "Brem. Photons radiating in tau decay",
                                    5,
                                    -0.5,
                                    4.5,
                                    "N FSR Photons radiating from/with tau",
                                    "Number of #tau's from Gauge Bosons");
  TauBremPhotonsPt = dqm.book1dHisto("TauBremPhotonsPt",
                                     "Sum Brem Pt ",
                                     100,
                                     0,
                                     100,
                                     "P_{t}^{Brem. Photons radiating in tau decay} (GeV)",
                                     "Number of #tau's from Gauge Bosons");
  TauBremPhotonsPtSum = dqm.book1dHisto("TauBremPhotonsPtSum",
                                        "Sum of Brem Pt ",
                                        100,
                                        0,
                                        100,
                                        "Sum P_{t}^{Brem. Photons radiating in tau decay} (GeV)",
                                        "Number of #tau's from Gauge Bosons");

  MODEID = dqm.book1dHisto("JAKID", "JAK ID", NMODEID + 1, -0.5, NMODEID + 0.5);
  for (unsigned int j = 0; j < NMODEID + 1; j++) {
    MODEInvMass.push_back(std::vector<MonitorElement *>());
    std::string tmp = "JAKID";
    tmp += std::to_string(j);
    MODEInvMass.at(j).push_back(dqm.book1dHisto("M" + tmp,
                                                "M_{" + TauDecay::DecayMode(j) + "} (GeV)",
                                                80,
                                                0,
                                                2.0,
                                                "M_{" + TauDecay::DecayMode(j) + "} (GeV)",
                                                "Number of #tau's from Gauge Bosons"));
    MODEID->setBinLabel(1 + j, TauDecay::DecayMode(j));
    if (j == TauDecay::MODE_3PI || j == TauDecay::MODE_PI2PI0 || j == TauDecay::MODE_KPIK ||
        j == TauDecay::MODE_KPIPI) {
      MODEInvMass.at(j).push_back(dqm.book1dHisto("M13" + tmp,
                                                  "M_{13," + TauDecay::DecayMode(j) + "} (GeV)",
                                                  80,
                                                  0,
                                                  2.0,
                                                  "M_{13," + TauDecay::DecayMode(j) + "} (GeV)",
                                                  "Number of #tau's from Gauge Bosons"));
      MODEInvMass.at(j).push_back(dqm.book1dHisto("M23" + tmp,
                                                  "M_{23," + TauDecay::DecayMode(j) + "} (GeV)",
                                                  80,
                                                  0,
                                                  2.0,
                                                  "M_{23," + TauDecay::DecayMode(j) + "} (GeV)",
                                                  "Number of #tau's from Gauge Bosons"));
      MODEInvMass.at(j).push_back(dqm.book1dHisto("M12" + tmp,
                                                  "M_{12," + TauDecay::DecayMode(j) + "} (GeV)",
                                                  80,
                                                  0,
                                                  2.0,
                                                  "M_{12," + TauDecay::DecayMode(j) + "} (GeV)",
                                                  "Number of #tau's from Gauge Bosons"));
    }
  }
  return;
}

void TauValidation::analyze(const edm::Event &iEvent, const edm::EventSetup &iSetup) {
  ///Gathering the reco::GenParticleCollection information
  edm::Handle<reco::GenParticleCollection> genParticles;
  iEvent.getByToken(genparticleCollectionToken_, genParticles);

  double weight = 1.0;  //=   wmanager_.weight(iEvent);
  //////////////////////////////////////////////
  // find taus
  for (reco::GenParticleCollection::const_iterator iter = genParticles->begin(); iter != genParticles->end(); ++iter) {
    if (abs(iter->pdgId()) == PdtPdgMini::Z0 || abs(iter->pdgId()) == PdtPdgMini::Higgs0) {
      spinEffectsZH(&(*iter), weight);
    }
    if (abs(iter->pdgId()) == 15) {
      if (isLastTauinChain(&(*iter))) {
        nTaus->Fill(0.5, weight);
        int mother = tauMother(&(*iter), weight);
        if (mother > -1) {  // exclude B, D and other non-signal decay modes
          nPrimeTaus->Fill(0.5, weight);
          TauPt->Fill(iter->pt(), weight);
          TauEta->Fill(iter->eta(), weight);
          TauPhi->Fill(iter->phi(), weight);
          photons(&(*iter), weight);
          ///////////////////////////////////////////////
          // Adding MODEID and Mass information
          TauDecay_GenParticle TD;
          unsigned int jak_id, TauBitMask;
          if (TD.AnalyzeTau(&(*iter), jak_id, TauBitMask, false, false)) {
            MODEID->Fill(jak_id, weight);
            TauProngs->Fill(TD.nProng(TauBitMask), weight);
            tauDecayChannel(&(*iter), jak_id, TauBitMask, weight);
            if (jak_id <= NMODEID) {
              int tcharge = iter->pdgId() / abs(iter->pdgId());
              std::vector<const reco::GenParticle *> part = TD.Get_TauDecayProducts();
              spinEffectsWHpm(&(*iter), mother, jak_id, part, weight);
              TLorentzVector LVQ(0, 0, 0, 0);
              TLorentzVector LVS12(0, 0, 0, 0);
              TLorentzVector LVS13(0, 0, 0, 0);
              TLorentzVector LVS23(0, 0, 0, 0);
              bool haspart1 = false;
              TVector3 PV, SV;
              bool hasDL(false);
              for (unsigned int i = 0; i < part.size(); i++) {
                if (abs(part.at(i)->pdgId()) != PdtPdgMini::nu_tau && TD.isTauFinalStateParticle(part.at(i)->pdgId()) &&
                    !hasDL) {
                  hasDL = true;
                  TLorentzVector tlv(iter->px(), iter->py(), iter->pz(), iter->energy());
                  PV = TVector3(iter->vx(), iter->vy(), iter->vz());
                  SV = TVector3(part.at(i)->vx(), part.at(i)->vy(), part.at(i)->vz());
                  TVector3 DL = SV - PV;
                  DecayLength->Fill(DL.Dot(tlv.Vect()) / tlv.P(), weight);
                  double c(2.99792458E8), Ltau(DL.Mag() / 100) /*cm->m*/, beta(iter->p() / iter->mass());
                  LifeTime->Fill(Ltau / (c * beta), weight);
                }

                if (TD.isTauFinalStateParticle(part.at(i)->pdgId()) && abs(part.at(i)->pdgId()) != PdtPdgMini::nu_e &&
                    abs(part.at(i)->pdgId()) != PdtPdgMini::nu_mu && abs(part.at(i)->pdgId()) != PdtPdgMini::nu_tau) {
                  TLorentzVector LV(part.at(i)->px(), part.at(i)->py(), part.at(i)->pz(), part.at(i)->energy());
                  LVQ += LV;
                  if (jak_id == TauDecay::MODE_3PI || jak_id == TauDecay::MODE_PI2PI0 ||
                      jak_id == TauDecay::MODE_KPIK || jak_id == TauDecay::MODE_KPIPI) {
                    if ((tcharge == part.at(i)->pdgId() / abs(part.at(i)->pdgId()) && TD.nProng(TauBitMask) == 3) ||
                        ((jak_id == TauDecay::MODE_3PI || jak_id == TauDecay::MODE_PI2PI0) &&
                         TD.nProng(TauBitMask) == 1 && abs(part.at(i)->pdgId()) == PdtPdgMini::pi_plus)) {
                      LVS13 += LV;
                      LVS23 += LV;
                    } else {
                      LVS12 += LV;
                      if (!haspart1 && ((jak_id == TauDecay::MODE_3PI || jak_id == TauDecay::MODE_PI2PI0) ||
                                        ((jak_id != TauDecay::MODE_3PI || jak_id == TauDecay::MODE_PI2PI0) &&
                                         abs(part.at(i)->pdgId()) == PdtPdgMini::K_plus))) {
                        LVS13 += LV;
                        haspart1 = true;
                      } else {
                        LVS23 += LV;
                      }
                    }
                  }
                }
              }
              part.clear();
              MODEInvMass.at(jak_id).at(0)->Fill(LVQ.M(), weight);
              if (jak_id == TauDecay::MODE_3PI || jak_id == TauDecay::MODE_PI2PI0 || jak_id == TauDecay::MODE_KPIK ||
                  jak_id == TauDecay::MODE_KPIPI) {
                MODEInvMass.at(jak_id).at(1)->Fill(LVS13.M(), weight);
                MODEInvMass.at(jak_id).at(2)->Fill(LVS23.M(), weight);
                MODEInvMass.at(jak_id).at(3)->Fill(LVS12.M(), weight);
              }
            }
          } else {
            MODEID->Fill(jak_id, weight);
          }
        }
      }
    }
  }
}  //analyze

const reco::GenParticle *TauValidation::GetMother(const reco::GenParticle *tau) {
  for (unsigned int i = 0; i < tau->numberOfMothers(); i++) {
    const reco::GenParticle *mother = static_cast<const reco::GenParticle *>(tau->mother(i));
    if (mother->pdgId() == tau->pdgId())
      return GetMother(mother);
    return mother;
  }
  return tau;
}

const std::vector<const reco::GenParticle *> TauValidation::GetMothers(const reco::GenParticle *boson) {
  std::vector<const reco::GenParticle *> mothers;
  for (unsigned int i = 0; i < boson->numberOfMothers(); i++) {
    const reco::GenParticle *mother = static_cast<const reco::GenParticle *>(boson->mother(i));
    if (mother->pdgId() == boson->pdgId())
      return GetMothers(mother);
    mothers.push_back(mother);
  }
  return mothers;
}

int TauValidation::findMother(const reco::GenParticle *tau) { return TauValidation::GetMother(tau)->pdgId(); }

bool TauValidation::isLastTauinChain(const reco::GenParticle *tau) {
  for (unsigned int i = 0; i < tau->numberOfDaughters(); i++) {
    if (tau->daughter(i)->pdgId() == tau->pdgId())
      return false;
  }
  return true;
}

void TauValidation::findTauList(const reco::GenParticle *tau, std::vector<const reco::GenParticle *> &TauList) {
  TauList.insert(TauList.begin(), tau);
  for (unsigned int i = 0; i < tau->numberOfMothers(); i++) {
    const reco::GenParticle *mother = static_cast<const reco::GenParticle *>(tau->mother(i));
    if (mother->pdgId() == tau->pdgId()) {
      findTauList(mother, TauList);
    }
  }
}

void TauValidation::findFSRandBrem(const reco::GenParticle *p,
                                   bool doBrem,
                                   std::vector<const reco::GenParticle *> &ListofFSR,
                                   std::vector<const reco::GenParticle *> &ListofBrem) {
  // note this code split the FSR and Brem based one if the tau decays into a tau+photon or not with the Fortran Tauola Interface, this is not 100% correct because photos puts the tau with the regular tau decay products.
  if (abs(p->pdgId()) == 15) {
    if (isLastTauinChain(p)) {
      doBrem = true;
    } else {
      doBrem = false;
    }
  }
  int photo_ID = 22;
  for (unsigned int i = 0; i < p->numberOfDaughters(); i++) {
    const reco::GenParticle *dau = static_cast<const reco::GenParticle *>(p->daughter(i));
    if (abs((dau)->pdgId()) == abs(photo_ID) && !doBrem) {
      ListofFSR.push_back(dau);
    }
    if (abs((dau)->pdgId()) == abs(photo_ID) && doBrem) {
      ListofBrem.push_back(dau);
    }
    if (abs((dau)->pdgId()) != 111 && abs((dau)->pdgId()) != 221) {  // remove pi0 and eta decays
      findFSRandBrem(dau, doBrem, ListofFSR, ListofBrem);
    }
  }
}

void TauValidation::FindPhotosFSR(const reco::GenParticle *p,
                                  std::vector<const reco::GenParticle *> &ListofFSR,
                                  double &BosonScale) {
  BosonScale = 0.0;
  const reco::GenParticle *m = GetMother(p);
  int mother_pid = m->pdgId();
  if (m->pdgId() != p->pdgId()) {
    for (unsigned int i = 0; i < m->numberOfDaughters(); i++) {
      const reco::GenParticle *dau = static_cast<const reco::GenParticle *>(m->daughter(i));
      if (abs(dau->pdgId()) == 22) {
        ListofFSR.push_back(dau);
      }
    }
  }
  if (abs(mother_pid) == 24)
    BosonScale = 1.0;  // W
  if (abs(mother_pid) == 23)
    BosonScale = 2.0;  // Z;
  if (abs(mother_pid) == 22)
    BosonScale = 2.0;  // gamma;
  if (abs(mother_pid) == 25)
    BosonScale = 2.0;  // HSM;
  if (abs(mother_pid) == 35)
    BosonScale = 2.0;  // H0;
  if (abs(mother_pid) == 36)
    BosonScale = 2.0;  // A0;
  if (abs(mother_pid) == 37)
    BosonScale = 1.0;  //Hpm;
}

int TauValidation::tauMother(const reco::GenParticle *tau, double weight) {
  if (abs(tau->pdgId()) != 15)
    return -3;
  int mother_pid = findMother(tau);
  if (mother_pid == -2)
    return -2;
  int label = other;
  if (abs(mother_pid) == 24)
    label = W;
  if (abs(mother_pid) == 23)
    label = Z;
  if (abs(mother_pid) == 22)
    label = gamma;
  if (abs(mother_pid) == 25)
    label = HSM;
  if (abs(mother_pid) == 35)
    label = H0;
  if (abs(mother_pid) == 36)
    label = A0;
  if (abs(mother_pid) == 37)
    label = Hpm;
  int mother_shortpid = (abs(mother_pid) % 10000);
  if (mother_shortpid > 500 && mother_shortpid < 600)
    label = B;
  if (mother_shortpid > 400 && mother_shortpid < 500)
    label = D;
  TauMothers->Fill(label, weight);
  if (label == B || label == D || label == other)
    return -1;
  return mother_pid;
}

int TauValidation::tauDecayChannel(const reco::GenParticle *tau, int jak_id, unsigned int TauBitMask, double weight) {
  int channel = undetermined;
  if (tau->status() == 1)
    channel = stable;
  int allCount = 0, eCount = 0, muCount = 0, pi0Count = 0, piCount = 0, rhoCount = 0, a1Count = 0, KCount = 0,
      KstarCount = 0;

  countParticles(tau, allCount, eCount, muCount, pi0Count, piCount, rhoCount, a1Count, KCount, KstarCount);

  // resonances
  if (KCount >= 1)
    channel = K;
  if (KstarCount >= 1)
    channel = Kstar;
  if (a1Count >= 1)
    channel = a1;
  if (rhoCount >= 1)
    channel = rho;
  if (channel != undetermined && weight != 0.0)
    TauDecayChannels->Fill(channel, weight);

  // final state products
  if (piCount == 1 && pi0Count == 0)
    channel = pi;
  if (piCount == 1 && pi0Count == 1)
    channel = pi1pi0;
  if (piCount == 1 && pi0Count > 1)
    channel = pinpi0;
  if (piCount == 3 && pi0Count == 0)
    channel = tripi;
  if (piCount == 3 && pi0Count > 0)
    channel = tripinpi0;
  if (eCount == 1)
    channel = electron;
  if (muCount == 1)
    channel = muon;
  if (weight != 0.0)
    TauDecayChannels->Fill(channel, weight);
  return channel;
}

void TauValidation::countParticles(const reco::GenParticle *p,
                                   int &allCount,
                                   int &eCount,
                                   int &muCount,
                                   int &pi0Count,
                                   int &piCount,
                                   int &rhoCount,
                                   int &a1Count,
                                   int &KCount,
                                   int &KstarCount) {
  for (unsigned int i = 0; i < p->numberOfDaughters(); i++) {
    const reco::GenParticle *dau = static_cast<const reco::GenParticle *>(p->daughter(i));
    int pid = dau->pdgId();
    allCount++;
    if (abs(pid) == 11)
      eCount++;
    else if (abs(pid) == 13)
      muCount++;
    else if (abs(pid) == 111)
      pi0Count++;
    else if (abs(pid) == 211)
      piCount++;
    else if (abs(pid) == 213)
      rhoCount++;
    else if (abs(pid) == 20213)
      a1Count++;
    else if (abs(pid) == 321)
      KCount++;
    else if (abs(pid) == 323)
      KstarCount++;
    countParticles(dau, allCount, eCount, muCount, pi0Count, piCount, rhoCount, a1Count, KCount, KstarCount);
  }
}

void TauValidation::spinEffectsWHpm(
    const reco::GenParticle *tau, int mother, int decay, std::vector<const reco::GenParticle *> &part, double weight) {
  // polarization only for 1-prong hadronic taus with no neutral pions
  if (decay == TauDecay::MODE_PION || decay == TauDecay::MODE_MUON || decay == TauDecay::MODE_ELECTRON) {
    TLorentzVector momP4 = motherP4(tau);
    TLorentzVector pionP4 = leadingPionP4(tau);
    pionP4.Boost(-1 * momP4.BoostVector());
    double energy = pionP4.E() / (momP4.M() / 2);
    if (decay == TauDecay::MODE_PION) {
      if (abs(mother) == 24)
        TauSpinEffectsW_X->Fill(energy, weight);
      else if (abs(mother) == 37)
        TauSpinEffectsHpm_X->Fill(energy, weight);
    } else if (decay == TauDecay::MODE_MUON) {
      if (abs(mother) == 24)
        TauSpinEffectsW_muX->Fill(energy, weight);
      else if (abs(mother) == 37)
        TauSpinEffectsHpm_muX->Fill(energy, weight);
    } else if (decay == TauDecay::MODE_ELECTRON) {
      if (abs(mother) == 24)
        TauSpinEffectsW_eX->Fill(energy, weight);
      else if (abs(mother) == 37)
        TauSpinEffectsHpm_eX->Fill(energy, weight);
    }
  } else if (decay == TauDecay::MODE_PIPI0) {
    TLorentzVector rho(0, 0, 0, 0), pi(0, 0, 0, 0);
    for (unsigned int i = 0; i < part.size(); i++) {
      TLorentzVector LV(part.at(i)->px(), part.at(i)->py(), part.at(i)->pz(), part.at(i)->energy());
      if (abs(part.at(i)->pdgId()) == PdtPdgMini::pi_plus) {
        pi += LV;
        rho += LV;
      } else if (abs(part.at(i)->pdgId()) == PdtPdgMini::pi0) {
        rho += LV;
      }
    }
    if (abs(mother) == 24)
      TauSpinEffectsW_UpsilonRho->Fill(2 * pi.P() / rho.P() - 1, weight);
    else if (abs(mother) == 37)
      TauSpinEffectsHpm_UpsilonRho->Fill(2 * pi.P() / rho.P() - 1, weight);
  } else if (decay == TauDecay::MODE_3PI || decay == TauDecay::MODE_PI2PI0) {  // only for pi2pi0 for now
    TLorentzVector a1(0, 0, 0, 0), pi_p(0, 0, 0, 0), pi_m(0, 0, 0, 0);
    int nplus(0), nminus(0);
    for (unsigned int i = 0; i < part.size(); i++) {
      TLorentzVector LV(part.at(i)->px(), part.at(i)->py(), part.at(i)->pz(), part.at(i)->energy());
      if (part.at(i)->pdgId() == PdtPdgMini::pi_plus) {
        pi_p += LV;
        a1 += LV;
        nplus++;
      } else if (part.at(i)->pdgId() == PdtPdgMini::pi_minus) {
        pi_m += LV;
        a1 += LV;
        nminus++;
      }
    }
    double gamma = 0;
    if (nplus + nminus == 3 && nplus == 1)
      gamma = 2 * pi_p.P() / a1.P() - 1;
    else if (nplus + nminus == 3 && nminus == 1)
      gamma = 2 * pi_m.P() / a1.P() - 1;
    else {
      pi_p += pi_m;
      gamma = 2 * pi_p.P() / a1.P() - 1;
    }
    if (abs(mother) == 24)
      TauSpinEffectsW_UpsilonA1->Fill(gamma, weight);
    else if (abs(mother) == 37)
      TauSpinEffectsHpm_UpsilonA1->Fill(gamma, weight);
  }
}

void TauValidation::spinEffectsZH(const reco::GenParticle *boson, double weight) {
  int ntau(0);
  for (unsigned int i = 0; i < boson->numberOfDaughters(); i++) {
    const reco::GenParticle *dau = static_cast<const reco::GenParticle *>(boson->daughter(i));
    if (ntau == 1 && dau->pdgId() == 15)
      return;
    if (boson->pdgId() != 15 && abs(dau->pdgId()) == 15)
      ntau++;
  }
  if (ntau != 2)
    return;
  if (abs(boson->pdgId()) == PdtPdgMini::Z0 || abs(boson->pdgId()) == PdtPdgMini::Higgs0) {
    TLorentzVector tautau(0, 0, 0, 0);
    TLorentzVector pipi(0, 0, 0, 0);
    TLorentzVector taum(0, 0, 0, 0);
    TLorentzVector taup(0, 0, 0, 0);
    TLorentzVector rho_plus, rho_minus, pi_rhominus, pi0_rhominus, pi_rhoplus, pi0_rhoplus, pi_plus, pi_minus;
    bool hasrho_minus(false), hasrho_plus(false), haspi_minus(false), haspi_plus(false);
    int nSinglePionDecays(0);
    double x1(0), x2(0);
    TLorentzVector Zboson(boson->px(), boson->py(), boson->pz(), boson->energy());
    for (unsigned int i = 0; i < boson->numberOfDaughters(); i++) {
      const reco::GenParticle *dau = static_cast<const reco::GenParticle *>(boson->daughter(i));
      int pid = dau->pdgId();
      if (abs(findMother(dau)) != 15 && abs(pid) == 15) {
        TauDecay_GenParticle TD;
        unsigned int jak_id, TauBitMask;
        if (TD.AnalyzeTau(dau, jak_id, TauBitMask, false, false)) {
          std::vector<const reco::GenParticle *> part = TD.Get_TauDecayProducts();
          if (jak_id == TauDecay::MODE_PION || jak_id == TauDecay::MODE_MUON || jak_id == TauDecay::MODE_ELECTRON) {
            if (jak_id == TauDecay::MODE_PION)
              nSinglePionDecays++;
            TLorentzVector LVtau(dau->px(), dau->py(), dau->pz(), dau->energy());
            tautau += LVtau;
            TLorentzVector LVpi = leadingPionP4(dau);
            pipi += LVpi;
            const HepPDT::ParticleData *pd = fPDGTable->particle(dau->pdgId());
            int charge = (int)pd->charge();
            LVtau.Boost(-1 * Zboson.BoostVector());
            LVpi.Boost(-1 * Zboson.BoostVector());

            if (jak_id == TauDecay::MODE_MUON) {
              if (abs(boson->pdgId()) == PdtPdgMini::Z0)
                TauSpinEffectsZ_muX->Fill(LVpi.P() / LVtau.E(), weight);
              if (abs(boson->pdgId()) == PdtPdgMini::Higgs0)
                TauSpinEffectsH_muX->Fill(LVpi.P() / LVtau.E(), weight);
            }
            if (jak_id == TauDecay::MODE_ELECTRON) {
              if (abs(boson->pdgId()) == PdtPdgMini::Z0)
                TauSpinEffectsZ_eX->Fill(LVpi.P() / LVtau.E(), weight);
              if (abs(boson->pdgId()) == PdtPdgMini::Higgs0)
                TauSpinEffectsH_eX->Fill(LVpi.P() / LVtau.E(), weight);
            }

            if (jak_id == TauDecay::MODE_PION) {
              if (abs(boson->pdgId()) == PdtPdgMini::Z0) {
                TauSpinEffectsZ_X->Fill(LVpi.P() / LVtau.E(), weight);
                if (50.0 < Zboson.M() && Zboson.M() < 75.0)
                  TauSpinEffectsZ_X50to75->Fill(LVpi.P() / LVtau.E(), weight);
                if (75.0 < Zboson.M() && Zboson.M() < 88.0)
                  TauSpinEffectsZ_X75to88->Fill(LVpi.P() / LVtau.E(), weight);
                if (88.0 < Zboson.M() && Zboson.M() < 100.0)
                  TauSpinEffectsZ_X88to100->Fill(LVpi.P() / LVtau.E(), weight);
                if (100.0 < Zboson.M() && Zboson.M() < 120.0)
                  TauSpinEffectsZ_X100to120->Fill(LVpi.P() / LVtau.E(), weight);
                if (120.0 < Zboson.M())
                  TauSpinEffectsZ_X120UP->Fill(LVpi.P() / LVtau.E(), weight);
              }
              if (abs(boson->pdgId()) == PdtPdgMini::Higgs0)
                TauSpinEffectsH_X->Fill(LVpi.P() / LVtau.E(), weight);
            }
            if (charge < 0) {
              x1 = LVpi.P() / LVtau.E();
              taum = LVtau;
            } else {
              x2 = LVpi.P() / LVtau.E();
            }
          }
          TLorentzVector LVtau(dau->px(), dau->py(), dau->pz(), dau->energy());
          if (pid == 15)
            taum = LVtau;
          if (pid == -15)
            taup = LVtau;
          if (jak_id == TauDecay::MODE_PIPI0) {
            for (unsigned int i = 0; i < part.size(); i++) {
              int pid_d = part.at(i)->pdgId();
              if (abs(pid_d) == 211 || abs(pid_d) == 111) {
                TLorentzVector LV(part.at(i)->px(), part.at(i)->py(), part.at(i)->pz(), part.at(i)->energy());
                if (pid == 15) {
                  hasrho_minus = true;
                  if (pid_d == -211) {
                    pi_rhominus = LV;
                  }
                  if (abs(pid_d) == 111) {
                    pi0_rhominus = LV;
                  }
                }
                if (pid == -15) {
                  hasrho_plus = true;
                  if (pid_d == 211) {
                    pi_rhoplus = LV;
                  }
                  if (abs(pid_d) == 111) {
                    pi0_rhoplus = LV;
                  }
                }
              }
            }
          }
          if (jak_id == TauDecay::MODE_PION) {
            for (unsigned int i = 0; i < part.size(); i++) {
              int pid_d = part.at(i)->pdgId();
              if (abs(pid_d) == 211) {
                TLorentzVector LV(part.at(i)->px(), part.at(i)->py(), part.at(i)->pz(), part.at(i)->energy());
                if (pid == 15) {
                  haspi_minus = true;
                  if (pid_d == -211) {
                    pi_minus = LV;
                  }
                }
                if (pid == -15) {
                  haspi_plus = true;
                  if (pid_d == 211) {
                    pi_plus = LV;
                  }
                }
              }
            }
          }
        }
      }
    }
    if (hasrho_minus && hasrho_plus) {
      //compute rhorho
      rho_minus = pi_rhominus;
      rho_minus += pi0_rhominus;
      rho_plus = pi_rhoplus;
      rho_plus += pi0_rhoplus;
      TLorentzVector rhorho = rho_minus;
      rhorho += rho_plus;

      // boost to rhorho cm
      TLorentzVector pi_rhoplusb = pi_rhoplus;
      pi_rhoplusb.Boost(-1 * rhorho.BoostVector());
      TLorentzVector pi0_rhoplusb = pi0_rhoplus;
      pi0_rhoplusb.Boost(-1 * rhorho.BoostVector());
      TLorentzVector pi_rhominusb = pi_rhominus;
      pi_rhominusb.Boost(-1 * rhorho.BoostVector());
      TLorentzVector pi0_rhominusb = pi0_rhominus;
      pi0_rhominusb.Boost(-1 * rhorho.BoostVector());

      // compute n+/-
      TVector3 n_plus = pi_rhoplusb.Vect().Cross(pi0_rhoplusb.Vect());
      TVector3 n_minus = pi_rhominusb.Vect().Cross(pi0_rhominusb.Vect());

      // compute the acoplanarity
      double Acoplanarity = acos(n_plus.Dot(n_minus) / (n_plus.Mag() * n_minus.Mag()));
      if (pi_rhominusb.Vect().Dot(n_plus) > 0) {
        Acoplanarity *= -1;
        Acoplanarity += 2 * TMath::Pi();
      }

      // now boost to tau frame
      pi_rhoplus.Boost(-1 * taup.BoostVector());
      pi0_rhoplus.Boost(-1 * taup.BoostVector());
      pi_rhominus.Boost(-1 * taum.BoostVector());
      pi0_rhominus.Boost(-1 * taum.BoostVector());

      // compute y1 and y2
      double y1 = (pi_rhoplus.E() - pi0_rhoplus.E()) / (pi_rhoplus.E() + pi0_rhoplus.E());
      double y2 = (pi_rhominus.E() - pi0_rhominus.E()) / (pi_rhominus.E() + pi0_rhominus.E());

      // fill histograms
      if (abs(boson->pdgId()) == PdtPdgMini::Higgs0 && y1 * y2 < 0)
        TauSpinEffectsH_rhorhoAcoplanarityminus->Fill(Acoplanarity, weight);
      if (abs(boson->pdgId()) == PdtPdgMini::Higgs0 && y1 * y2 > 0)
        TauSpinEffectsH_rhorhoAcoplanarityplus->Fill(Acoplanarity, weight);
    }
    if (haspi_minus && haspi_plus) {
      TLorentzVector tauporig = taup;
      TLorentzVector taumorig = taum;

      // now boost to Higgs frame
      pi_plus.Boost(-1 * Zboson.BoostVector());
      pi_minus.Boost(-1 * Zboson.BoostVector());

      taup.Boost(-1 * Zboson.BoostVector());
      taum.Boost(-1 * Zboson.BoostVector());

      if (abs(boson->pdgId()) == PdtPdgMini::Higgs0) {
        TauSpinEffectsH_pipiAcollinearity->Fill(
            acos(pi_plus.Vect().Dot(pi_minus.Vect()) / (pi_plus.P() * pi_minus.P())));
        TauSpinEffectsH_pipiAcollinearityzoom->Fill(
            acos(pi_plus.Vect().Dot(pi_minus.Vect()) / (pi_plus.P() * pi_minus.P())));
      }

      double proj_m = taum.Vect().Dot(pi_minus.Vect()) / (taum.P() * taum.P());
      double proj_p = taup.Vect().Dot(pi_plus.Vect()) / (taup.P() * taup.P());
      TVector3 Tau_m = taum.Vect();
      TVector3 Tau_p = taup.Vect();
      Tau_m *= proj_m;
      Tau_p *= proj_p;
      TVector3 Pit_m = pi_minus.Vect() - Tau_m;
      TVector3 Pit_p = pi_plus.Vect() - Tau_p;

      double Acoplanarity = acos(Pit_m.Dot(Pit_p) / (Pit_p.Mag() * Pit_m.Mag()));
      TVector3 n = Pit_p.Cross(Pit_m);
      if (n.Dot(Tau_m) / Tau_m.Mag() > 0) {
        Acoplanarity *= -1;
        Acoplanarity += 2 * TMath::Pi();
      }
      // fill histograms
      if (abs(boson->pdgId()) == PdtPdgMini::Higgs0)
        TauSpinEffectsH_pipiAcoplanarity->Fill(Acoplanarity, weight);
      taup = tauporig;
      taum = taumorig;
    }
    if (nSinglePionDecays == 2 && tautau.M() != 0) {
      for (int i = 0; i < zsbins; i++) {
        double zslow = ((double)i) * (zsmax - zsmin) / ((double)zsbins) + zsmin;
        double zsup = ((double)i + 1) * (zsmax - zsmin) / ((double)zsbins) + zsmin;
        double aup = Zstoa(zsup), alow = Zstoa(zslow);
        if (x2 - x1 > alow && x2 - x1 < aup) {
          double zs = (zsup + zslow) / 2;
          if (abs(boson->pdgId()) == PdtPdgMini::Z0)
            TauSpinEffectsZ_Zs->Fill(zs, weight);
          if (abs(boson->pdgId()) == PdtPdgMini::Higgs0)
            TauSpinEffectsH_Zs->Fill(zs, weight);
          break;
        }
      }
      if (abs(boson->pdgId()) == PdtPdgMini::Z0)
        TauSpinEffectsZ_MVis->Fill(pipi.M() / tautau.M(), weight);
      if (abs(boson->pdgId()) == PdtPdgMini::Higgs0)
        TauSpinEffectsH_MVis->Fill(pipi.M() / tautau.M(), weight);

      if (x1 != 0) {
        const std::vector<const reco::GenParticle *> m = GetMothers(boson);
        int q(0), qbar(0);
        TLorentzVector Z(0, 0, 0, 0);
        for (unsigned int i = 0; i < m.size(); i++) {
          if (m.at(i)->pdgId() == PdtPdgMini::d || m.at(i)->pdgId() == PdtPdgMini::u) {
            q++;
          }
          if (m.at(i)->pdgId() == PdtPdgMini::anti_d || m.at(i)->pdgId() == PdtPdgMini::anti_u) {
            qbar++;
          }
        }
        if (q == 1 && qbar == 1) {  // assume q has largest E (valence vs see quarks)
          if (taum.Vect().Dot(Zboson.Vect()) / (Zboson.P() * taum.P()) > 0) {
            if (abs(boson->pdgId()) == PdtPdgMini::Z0)
              TauSpinEffectsZ_Xf->Fill(x1, weight);
            if (abs(boson->pdgId()) == PdtPdgMini::Higgs0)
              TauSpinEffectsH_Xf->Fill(x1, weight);
          } else {
            if (abs(boson->pdgId()) == PdtPdgMini::Z0)
              TauSpinEffectsZ_Xb->Fill(x1, weight);
            if (abs(boson->pdgId()) == PdtPdgMini::Higgs0)
              TauSpinEffectsH_Xb->Fill(x1, weight);
          }
        }
      }
    }
  }
}

double TauValidation::Zstoa(double zs) {
  double a = 1 - sqrt(fabs(1.0 - 2 * fabs(zs)));
  if (zs < 0) {
    a *= -1.0;
  }
  return a;
}

double TauValidation::leadingPionMomentum(const reco::GenParticle *tau, double weight) {
  return leadingPionP4(tau).P();
}

TLorentzVector TauValidation::leadingPionP4(const reco::GenParticle *tau) {
  TLorentzVector p4(0, 0, 0, 0);
  for (unsigned int i = 0; i < tau->numberOfDaughters(); i++) {
    const reco::GenParticle *dau = static_cast<const reco::GenParticle *>(tau->daughter(i));
    int pid = dau->pdgId();
    if (abs(pid) == 15)
      return leadingPionP4(dau);
    if (!(abs(pid) == 211 || abs(pid) == 13 || abs(pid) == 11))
      continue;
    if (dau->p() > p4.P())
      p4 = TLorentzVector(dau->px(), dau->py(), dau->pz(), dau->energy());
  }
  return p4;
}

TLorentzVector TauValidation::motherP4(const reco::GenParticle *tau) {
  const reco::GenParticle *m = GetMother(tau);
  return TLorentzVector(m->px(), m->py(), m->pz(), m->energy());
}

double TauValidation::visibleTauEnergy(const reco::GenParticle *tau) {
  TLorentzVector p4(tau->px(), tau->py(), tau->pz(), tau->energy());
  for (unsigned int i = 0; i < tau->numberOfDaughters(); i++) {
    const reco::GenParticle *dau = static_cast<const reco::GenParticle *>(tau->daughter(i));
    int pid = dau->pdgId();
    if (abs(pid) == 15)
      return visibleTauEnergy(dau);
    if (abs(pid) == 12 || abs(pid) == 14 || abs(pid) == 16) {
      p4 -= TLorentzVector(dau->px(), dau->py(), dau->pz(), dau->energy());
    }
  }
  return p4.E();
}

void TauValidation::photons(const reco::GenParticle *tau, double weight) {
  // Find First tau in chain
  std::vector<const reco::GenParticle *> TauList;
  findTauList(tau, TauList);

  // Get List of Gauge Boson to tau(s) FSR and Brem
  bool passedW = false;
  std::vector<const reco::GenParticle *> ListofFSR;
  ListofFSR.clear();
  std::vector<const reco::GenParticle *> ListofBrem;
  ListofBrem.clear();
  std::vector<const reco::GenParticle *> FSR_photos;
  FSR_photos.clear();
  double BosonScale(1);
  if (!TauList.empty()) {
    TauValidation::findFSRandBrem(TauList.at(0), passedW, ListofFSR, ListofBrem);
    TauValidation::FindPhotosFSR(TauList.at(0), FSR_photos, BosonScale);

    // Add the Tau Brem. information
    TauBremPhotonsN->Fill(ListofBrem.size(), weight);
    double photonPtSum = 0;
    for (unsigned int i = 0; i < ListofBrem.size(); i++) {
      photonPtSum += ListofBrem.at(i)->pt();
      TauBremPhotonsPt->Fill(ListofBrem.at(i)->pt(), weight);
    }
    TauBremPhotonsPtSum->Fill(photonPtSum, weight);

    // Now add the Gauge Boson FSR information
    if (BosonScale != 0) {
      TauFSRPhotonsN->Fill(ListofFSR.size(), weight);
      photonPtSum = 0;
      for (unsigned int i = 0; i < ListofFSR.size(); i++) {
        photonPtSum += ListofFSR.at(i)->pt();
        TauFSRPhotonsPt->Fill(ListofFSR.at(i)->pt(), weight);
      }
      double FSR_photosSum(0);
      for (unsigned int i = 0; i < FSR_photos.size(); i++) {
        FSR_photosSum += FSR_photos.at(i)->pt();
        TauFSRPhotonsPt->Fill(FSR_photos.at(i)->pt() / BosonScale, weight * BosonScale);
      }
      TauFSRPhotonsPtSum->Fill(photonPtSum + FSR_photosSum / BosonScale, weight);
    }
  }
}
