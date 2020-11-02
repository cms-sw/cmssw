#include "Geometry/HcalCommonData/interface/HcalHitRelabeller.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "Validation/HcalHits/interface/HcalSimHitsValidation.h"

HcalSimHitsValidation::HcalSimHitsValidation(edm::ParameterSet const &conf) {
  // DQM ROOT output
  outputFile_ = conf.getUntrackedParameter<std::string>("outputFile", "myfile.root");
  testNumber_ = conf.getUntrackedParameter<bool>("TestNumber", false);
  auxPlots_ = conf.getUntrackedParameter<bool>("auxiliaryPlots", false);

  // register for data access
  g4Label_ = conf.getUntrackedParameter<std::string>("ModuleLabel", "g4SimHits");
  hcalHits_ = conf.getUntrackedParameter<std::string>("HcalHitCollection", "HcalHits");
  ebHits_ = conf.getUntrackedParameter<std::string>("EBHitCollection", "EcalHitsEB");
  eeHits_ = conf.getUntrackedParameter<std::string>("EEHitCollection", "EcalHitsEE");

  // import sampling factors
  hf1_ = conf.getParameter<double>("hf1");
  hf2_ = conf.getParameter<double>("hf2");

  tok_evt_ = consumes<edm::HepMCProduct>(edm::InputTag("generatorSmeared"));
  tok_hcal_ = consumes<edm::PCaloHitContainer>(edm::InputTag(g4Label_, hcalHits_));
  tok_ecalEB_ = consumes<edm::PCaloHitContainer>(edm::InputTag(g4Label_, ebHits_));
  tok_ecalEE_ = consumes<edm::PCaloHitContainer>(edm::InputTag(g4Label_, eeHits_));
  tok_HRNDC_ = esConsumes<HcalDDDRecConstants, HcalRecNumberingRecord, edm::Transition::BeginRun>();

  if (!outputFile_.empty()) {
    edm::LogInfo("OutputInfo") << " Hcal SimHit Task histograms will be saved to '" << outputFile_.c_str() << "'";
  } else {
    edm::LogInfo("OutputInfo") << " Hcal SimHit Task histograms will NOT be saved";
  }

  nevtot = 0;
}

HcalSimHitsValidation::~HcalSimHitsValidation() {}

void HcalSimHitsValidation::bookHistograms(DQMStore::IBooker &ib, edm::Run const &run, edm::EventSetup const &es) {
  const auto &pHRNDC = es.getData(tok_HRNDC_);
  hcons = &pHRNDC;
  maxDepthHB_ = hcons->getMaxDepth(0);
  maxDepthHE_ = hcons->getMaxDepth(1);
  maxDepthHF_ = hcons->getMaxDepth(2);
  maxDepthHO_ = hcons->getMaxDepth(3);

  // Get Phi segmentation from geometry, use the max phi number so that all iphi
  // values are included.

  int NphiMax = hcons->getNPhi(0);

  NphiMax = (hcons->getNPhi(1) > NphiMax ? hcons->getNPhi(1) : NphiMax);
  NphiMax = (hcons->getNPhi(2) > NphiMax ? hcons->getNPhi(2) : NphiMax);
  NphiMax = (hcons->getNPhi(3) > NphiMax ? hcons->getNPhi(3) : NphiMax);

  // Center the iphi bins on the integers
  // float iphi_min = 0.5;
  // float iphi_max = NphiMax + 0.5;
  // int iphi_bins = (int) (iphi_max - iphi_min);

  int iEtaHBMax = hcons->getEtaRange(0).second;
  int iEtaHEMax = std::max(hcons->getEtaRange(1).second, 1);
  int iEtaHFMax = hcons->getEtaRange(2).second;
  int iEtaHOMax = hcons->getEtaRange(3).second;

  // Retain classic behavior, all plots have same ieta range.
  // Comment out	code to	allow each subdetector to have its on range

  int iEtaMax = (iEtaHBMax > iEtaHEMax ? iEtaHBMax : iEtaHEMax);
  iEtaMax = (iEtaMax > iEtaHFMax ? iEtaMax : iEtaHFMax);
  iEtaMax = (iEtaMax > iEtaHOMax ? iEtaMax : iEtaHOMax);

  iEtaHBMax = iEtaMax;
  iEtaHEMax = iEtaMax;
  iEtaHFMax = iEtaMax;
  iEtaHOMax = iEtaMax;

  // Give an empty bin around the subdet ieta range to make it clear that all
  // ieta rings have been included
  float ieta_min_HB = -iEtaHBMax - 1.5;
  float ieta_max_HB = iEtaHBMax + 1.5;
  int ieta_bins_HB = (int)(ieta_max_HB - ieta_min_HB);

  float ieta_min_HE = -iEtaHEMax - 1.5;
  float ieta_max_HE = iEtaHEMax + 1.5;
  int ieta_bins_HE = (int)(ieta_max_HE - ieta_min_HE);

  float ieta_min_HF = -iEtaHFMax - 1.5;
  float ieta_max_HF = iEtaHFMax + 1.5;
  int ieta_bins_HF = (int)(ieta_max_HF - ieta_min_HF);

  float ieta_min_HO = -iEtaHOMax - 1.5;
  float ieta_max_HO = iEtaHOMax + 1.5;
  int ieta_bins_HO = (int)(ieta_max_HO - ieta_min_HO);

  Char_t histo[200];

  ib.setCurrentFolder("HcalHitsV/HcalSimHitTask");

  if (auxPlots_) {
    // General counters
    for (int depth = 0; depth <= maxDepthHB_; depth++) {
      if (depth == 0) {
        sprintf(histo, "N_HB");
      } else {
        sprintf(histo, "N_HB%d", depth);
      }

      Nhb.push_back(ib.book1D(histo, histo, 2600, 0., 2600.));
    }
    for (int depth = 0; depth <= maxDepthHE_; depth++) {
      if (depth == 0) {
        sprintf(histo, "N_HE");
      } else {
        sprintf(histo, "N_HE%d", depth);
      }

      Nhe.push_back(ib.book1D(histo, histo, 2600, 0., 2600.));
    }

    sprintf(histo, "N_HO");
    Nho = ib.book1D(histo, histo, 2200, 0., 2200.);

    for (int depth = 0; depth <= maxDepthHF_; depth++) {
      if (depth == 0) {
        sprintf(histo, "N_HF");
      } else {
        sprintf(histo, "N_HF%d", depth);
      }

      Nhf.push_back(ib.book1D(histo, histo, 1800, 0., 1800.));
    }

    // Mean energy vs iEta TProfiles
    for (int depth = 0; depth <= maxDepthHB_; depth++) {
      if (depth == 0) {
        sprintf(histo, "emean_vs_ieta_HB");
      } else {
        sprintf(histo, "emean_vs_ieta_HB%d", depth);
      }

      emean_vs_ieta_HB.push_back(
          ib.bookProfile(histo, histo, ieta_bins_HB, ieta_min_HB, ieta_max_HB, -10., 2000., " "));
    }
    for (int depth = 0; depth <= maxDepthHE_; depth++) {
      if (depth == 0) {
        sprintf(histo, "emean_vs_ieta_HE");
      } else {
        sprintf(histo, "emean_vs_ieta_HE%d", depth);
      }

      emean_vs_ieta_HE.push_back(
          ib.bookProfile(histo, histo, ieta_bins_HE, ieta_min_HE, ieta_max_HE, -10., 2000., " "));
    }

    sprintf(histo, "emean_vs_ieta_HO");
    emean_vs_ieta_HO = ib.bookProfile(histo, histo, ieta_bins_HO, ieta_min_HO, ieta_max_HO, -10., 2000., " ");

    for (int depth = 0; depth <= maxDepthHF_; depth++) {
      if (depth == 0) {
        sprintf(histo, "emean_vs_ieta_HF");
      } else {
        sprintf(histo, "emean_vs_ieta_HF%d", depth);
      }

      emean_vs_ieta_HF.push_back(
          ib.bookProfile(histo, histo, ieta_bins_HF, ieta_min_HF, ieta_max_HF, -10., 2000., " "));
    }

    // Occupancy vs. iEta TH1Fs
    for (int depth = 0; depth <= maxDepthHB_; depth++) {
      if (depth == 0) {
        sprintf(histo, "occupancy_vs_ieta_HB");
      } else {
        sprintf(histo, "occupancy_vs_ieta_HB%d", depth);
      }

      occupancy_vs_ieta_HB.push_back(ib.book1D(histo, histo, ieta_bins_HB, ieta_min_HB, ieta_max_HB));
    }
    for (int depth = 0; depth <= maxDepthHE_; depth++) {
      if (depth == 0) {
        sprintf(histo, "occupancy_vs_ieta_HE");
      } else {
        sprintf(histo, "occupancy_vs_ieta_HE%d", depth);
      }

      occupancy_vs_ieta_HE.push_back(ib.book1D(histo, histo, ieta_bins_HE, ieta_min_HE, ieta_max_HE));
    }

    sprintf(histo, "occupancy_vs_ieta_HO");
    occupancy_vs_ieta_HO = ib.book1D(histo, histo, ieta_bins_HO, ieta_min_HO, ieta_max_HO);

    for (int depth = 0; depth <= maxDepthHF_; depth++) {
      if (depth == 0) {
        sprintf(histo, "occupancy_vs_ieta_HF");
      } else {
        sprintf(histo, "occupancy_vs_ieta_HF%d", depth);
      }

      occupancy_vs_ieta_HF.push_back(ib.book1D(histo, histo, ieta_bins_HF, ieta_min_HF, ieta_max_HF));
    }

    // Energy spectra
    for (int depth = 0; depth <= maxDepthHB_; depth++) {
      if (depth == 0) {
        sprintf(histo, "HcalSimHitTask_energy_of_simhits_HB");
      } else {
        sprintf(histo, "HcalSimHitTask_energy_of_simhits_HB%d", depth);
      }

      meSimHitsEnergyHB.push_back(ib.book1D(histo, histo, 510, -0.1, 5.));
    }
    for (int depth = 0; depth <= maxDepthHE_; depth++) {
      if (depth == 0) {
        sprintf(histo, "HcalSimHitTask_energy_of_simhits_HE");
      } else {
        sprintf(histo, "HcalSimHitTask_energy_of_simhits_HE%d", depth);
      }

      meSimHitsEnergyHE.push_back(ib.book1D(histo, histo, 510, -0.1, 5.));
    }

    sprintf(histo, "HcalSimHitTask_energy_of_simhits_HO");
    meSimHitsEnergyHO = ib.book1D(histo, histo, 510, -0.1, 5.);

    for (int depth = 0; depth <= maxDepthHF_; depth++) {
      if (depth == 0) {
        sprintf(histo, "HcalSimHitTask_energy_of_simhits_HF");
      } else {
        sprintf(histo, "HcalSimHitTask_energy_of_simhits_HF%d", depth);
      }

      meSimHitsEnergyHF.push_back(ib.book1D(histo, histo, 1010, -5., 500.));
    }

  }  // auxPlots_

  // Energy in Cone
  sprintf(histo, "HcalSimHitTask_En_simhits_cone_profile_vs_ieta_all_depths");
  meEnConeEtaProfile = ib.bookProfile(histo, histo, ieta_bins_HF, ieta_min_HF, ieta_max_HF, -10., 200., " ");

  sprintf(histo, "HcalSimHitTask_En_simhits_cone_profile_vs_ieta_all_depths_E");
  meEnConeEtaProfile_E = ib.bookProfile(histo, histo, ieta_bins_HF, ieta_min_HF, ieta_max_HF, -10., 200., " ");

  sprintf(histo, "HcalSimHitTask_En_simhits_cone_profile_vs_ieta_all_depths_EH");
  meEnConeEtaProfile_EH = ib.bookProfile(histo, histo, ieta_bins_HF, ieta_min_HF, ieta_max_HF, -10., 200., " ");
}

void HcalSimHitsValidation::endJob() {
  if (auxPlots_) {
    for (int i = 1; i <= occupancy_vs_ieta_HB[0]->getNbinsX(); i++) {
      int ieta = i - 43;  // -41 -1, 1 41

      float phi_factor;

      if (std::abs(ieta) <= 20)
        phi_factor = 72.;
      else if (std::abs(ieta) < 40)
        phi_factor = 36.;
      else
        phi_factor = 18.;

      float cnorm;

      // Occupancy vs. iEta TH1Fs
      for (int depth = 0; depth <= maxDepthHB_; depth++) {
        cnorm = occupancy_vs_ieta_HB[depth]->getBinContent(i) / (phi_factor * nevtot);
        occupancy_vs_ieta_HB[depth]->setBinContent(i, cnorm);
      }
      for (int depth = 0; depth <= maxDepthHE_; depth++) {
        cnorm = occupancy_vs_ieta_HE[depth]->getBinContent(i) / (phi_factor * nevtot);
        occupancy_vs_ieta_HE[depth]->setBinContent(i, cnorm);
      }

      cnorm = occupancy_vs_ieta_HO->getBinContent(i) / (phi_factor * nevtot);
      occupancy_vs_ieta_HO->setBinContent(i, cnorm);

      for (int depth = 0; depth <= maxDepthHF_; depth++) {
        cnorm = occupancy_vs_ieta_HF[depth]->getBinContent(i) / (phi_factor * nevtot);
        occupancy_vs_ieta_HF[depth]->setBinContent(i, cnorm);
      }
    }
  }

  // let's see if this breaks anything
  // if ( outputFile_.size() != 0 && dbe_ ) dbe_->save(outputFile_);
}

void HcalSimHitsValidation::analyze(edm::Event const &ev, edm::EventSetup const &c) {
  using namespace edm;
  using namespace std;

  //===========================================================================
  // Getting SimHits
  //===========================================================================

  double phi_MC = -999.;  // phi of initial particle from HepMC
  double eta_MC = -999.;  // eta of initial particle from HepMC

  edm::Handle<edm::HepMCProduct> evtMC;
  ev.getByToken(tok_evt_, evtMC);  // generator in late 310_preX
  if (!evtMC.isValid()) {
    std::cout << "no HepMCProduct found" << std::endl;
  }

  // MC particle with highest pt is taken as a direction reference
  double maxPt = -99999.;
  int npart = 0;

  const HepMC::GenEvent *myGenEvent = evtMC->GetEvent();
  for (HepMC::GenEvent::particle_const_iterator p = myGenEvent->particles_begin(); p != myGenEvent->particles_end();
       ++p) {
    double phip = (*p)->momentum().phi();
    double etap = (*p)->momentum().eta();
    double pt = (*p)->momentum().perp();
    if (pt > maxPt) {
      npart++;
      maxPt = pt;
      phi_MC = phip;
      eta_MC = etap;
    }
  }

  double partR = 0.3;

  // Hcal SimHits

  // Approximate calibration constants
  const float calib_HB = 120.;
  const float calib_HE = 190.;
  const float calib_HF1 = hf1_;  // 1.0/0.383;
  const float calib_HF2 = hf2_;  // 1.0/0.368;

  edm::Handle<PCaloHitContainer> hcalHits;
  ev.getByToken(tok_hcal_, hcalHits);
  const PCaloHitContainer *SimHitResult = hcalHits.product();

  float eta_diff;
  float etaMax = 9999;
  int ietaMax = 0;

  double HcalCone = 0;

  c.get<CaloGeometryRecord>().get(geometry);

  for (std::vector<PCaloHit>::const_iterator SimHits = SimHitResult->begin(); SimHits != SimHitResult->end();
       ++SimHits) {
    HcalDetId cell;
    if (testNumber_)
      cell = HcalHitRelabeller::relabel(SimHits->id(), hcons);
    else
      cell = HcalDetId(SimHits->id());

    auto cellGeometry = geometry->getSubdetectorGeometry(cell)->getGeometry(cell);
    double etaS = cellGeometry->getPosition().eta();
    double phiS = cellGeometry->getPosition().phi();
    double en = SimHits->energy();

    int sub = cell.subdet();
    int depth = cell.depth();
    double ieta = cell.ieta();

    // Energy in Cone
    double r = dR(eta_MC, phi_MC, etaS, phiS);

    if (r < partR) {
      eta_diff = std::abs(eta_MC - etaS);
      if (eta_diff < etaMax) {
        etaMax = eta_diff;
        ietaMax = cell.ieta();
      }
      // Approximation of calibration
      if (sub == 1)
        HcalCone += en * calib_HB;
      else if (sub == 2)
        HcalCone += en * calib_HE;
      else if (sub == 4 && (depth == 1 || depth == 3))
        HcalCone += en * calib_HF1;
      else if (sub == 4 && (depth == 2 || depth == 4))
        HcalCone += en * calib_HF2;
    }

    if (auxPlots_) {
      // HB
      if (sub == 1) {
        meSimHitsEnergyHB[0]->Fill(en);
        meSimHitsEnergyHB[depth]->Fill(en);

        emean_vs_ieta_HB[0]->Fill(double(ieta), en);
        emean_vs_ieta_HB[depth]->Fill(double(ieta), en);

        occupancy_vs_ieta_HB[0]->Fill(double(ieta));
        occupancy_vs_ieta_HB[depth]->Fill(double(ieta));
      }
      // HE
      if (sub == 2 && maxDepthHE_ > 0) {
        meSimHitsEnergyHE[0]->Fill(en);
        meSimHitsEnergyHE[depth]->Fill(en);

        emean_vs_ieta_HE[0]->Fill(double(ieta), en);
        emean_vs_ieta_HE[depth]->Fill(double(ieta), en);

        occupancy_vs_ieta_HE[0]->Fill(double(ieta));
        occupancy_vs_ieta_HE[depth]->Fill(double(ieta));
      }
      // HO
      if (sub == 3) {
        meSimHitsEnergyHO->Fill(en);

        emean_vs_ieta_HO->Fill(double(ieta), en);

        occupancy_vs_ieta_HO->Fill(double(ieta));
      }
      // HF
      if (sub == 4) {
        meSimHitsEnergyHF[0]->Fill(en);
        meSimHitsEnergyHF[depth]->Fill(en);

        emean_vs_ieta_HF[0]->Fill(double(ieta), en);
        emean_vs_ieta_HF[depth]->Fill(double(ieta), en);

        occupancy_vs_ieta_HF[0]->Fill(double(ieta));
        occupancy_vs_ieta_HF[depth]->Fill(double(ieta));
      }

    }  // auxPlots_

  }  // Loop over SimHits

  // Ecal EB SimHits
  double EcalCone = 0;

  if (!ebHits_.empty()) {
    edm::Handle<PCaloHitContainer> ecalEBHits;
    ev.getByToken(tok_ecalEB_, ecalEBHits);
    const PCaloHitContainer *SimHitResultEB = ecalEBHits.product();

    for (std::vector<PCaloHit>::const_iterator SimHits = SimHitResultEB->begin(); SimHits != SimHitResultEB->end();
         ++SimHits) {
      EBDetId EBid = EBDetId(SimHits->id());

      auto cellGeometry = geometry->getSubdetectorGeometry(EBid)->getGeometry(EBid);
      double etaS = cellGeometry->getPosition().eta();
      double phiS = cellGeometry->getPosition().phi();
      double en = SimHits->energy();

      double r = dR(eta_MC, phi_MC, etaS, phiS);

      if (r < partR)
        EcalCone += en;
    }
  }  // ebHits_

  // Ecal EE SimHits
  if (!eeHits_.empty()) {
    edm::Handle<PCaloHitContainer> ecalEEHits;
    ev.getByToken(tok_ecalEE_, ecalEEHits);
    const PCaloHitContainer *SimHitResultEE = ecalEEHits.product();

    for (std::vector<PCaloHit>::const_iterator SimHits = SimHitResultEE->begin(); SimHits != SimHitResultEE->end();
         ++SimHits) {
      EEDetId EEid = EEDetId(SimHits->id());

      auto cellGeometry = geometry->getSubdetectorGeometry(EEid)->getGeometry(EEid);
      double etaS = cellGeometry->getPosition().eta();
      double phiS = cellGeometry->getPosition().phi();
      double en = SimHits->energy();

      double r = dR(eta_MC, phi_MC, etaS, phiS);

      if (r < partR)
        EcalCone += en;
    }
  }  // eeHits_

  if (ietaMax != 0) {  // If ietaMax == 0, there were no good HCAL SimHits
    meEnConeEtaProfile->Fill(double(ietaMax), HcalCone);
    meEnConeEtaProfile_E->Fill(double(ietaMax), EcalCone);
    meEnConeEtaProfile_EH->Fill(double(ietaMax), HcalCone + EcalCone);
  }

  nevtot++;
}

double HcalSimHitsValidation::dR(double eta1, double phi1, double eta2, double phi2) {
  double PI = 3.1415926535898;
  double deltaphi = phi1 - phi2;
  if (phi2 > phi1) {
    deltaphi = phi2 - phi1;
  }
  if (deltaphi > PI) {
    deltaphi = 2. * PI - deltaphi;
  }
  double deltaeta = eta2 - eta1;
  double tmp = sqrt(deltaeta * deltaeta + deltaphi * deltaphi);
  return tmp;
}

double HcalSimHitsValidation::phi12(double phi1, double en1, double phi2, double en2) {
  // weighted mean value of phi1 and phi2

  double tmp;
  double PI = 3.1415926535898;
  double a1 = phi1;
  double a2 = phi2;

  if (a1 > 0.5 * PI && a2 < 0.)
    a2 += 2 * PI;
  if (a2 > 0.5 * PI && a1 < 0.)
    a1 += 2 * PI;
  tmp = (a1 * en1 + a2 * en2) / (en1 + en2);
  if (tmp > PI)
    tmp -= 2. * PI;

  return tmp;
}

double HcalSimHitsValidation::dPhiWsign(double phi1, double phi2) {
  // clockwise      phi2 w.r.t phi1 means "+" phi distance
  // anti-clockwise phi2 w.r.t phi1 means "-" phi distance

  double PI = 3.1415926535898;
  double a1 = phi1;
  double a2 = phi2;
  double tmp = a2 - a1;
  if (a1 * a2 < 0.) {
    if (a1 > 0.5 * PI)
      tmp += 2. * PI;
    if (a2 > 0.5 * PI)
      tmp -= 2. * PI;
  }
  return tmp;
}

DEFINE_FWK_MODULE(HcalSimHitsValidation);
