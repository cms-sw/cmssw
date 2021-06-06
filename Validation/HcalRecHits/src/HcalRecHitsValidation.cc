#include "DataFormats/HcalDetId/interface/HcalSubdetector.h"
#include "Geometry/HcalCommonData/interface/HcalHitRelabeller.h"
#include "Geometry/HcalTowerAlgo/interface/HcalGeometry.h"
#include "Validation/HcalRecHits/interface/HcalRecHitsValidation.h"

HcalRecHitsValidation::HcalRecHitsValidation(edm::ParameterSet const &conf)
    : topFolderName_(conf.getParameter<std::string>("TopFolderName")) {
  // DQM ROOT output
  outputFile_ = conf.getUntrackedParameter<std::string>("outputFile", "myfile.root");

  if (!outputFile_.empty()) {
    edm::LogInfo("OutputInfo") << " Hcal RecHit Task histograms will be saved to '" << outputFile_.c_str() << "'";
  } else {
    edm::LogInfo("OutputInfo") << " Hcal RecHit Task histograms will NOT be saved";
  }

  nevtot = 0;

  hcalselector_ = conf.getUntrackedParameter<std::string>("hcalselector", "all");
  ecalselector_ = conf.getUntrackedParameter<std::string>("ecalselector", "yes");
  sign_ = conf.getUntrackedParameter<std::string>("sign", "*");
  mc_ = conf.getUntrackedParameter<std::string>("mc", "yes");
  testNumber_ = conf.getParameter<bool>("TestNumber");

  // Collections
  tok_hbhe_ = consumes<HBHERecHitCollection>(conf.getUntrackedParameter<edm::InputTag>("HBHERecHitCollectionLabel"));
  tok_hf_ = consumes<HFRecHitCollection>(conf.getUntrackedParameter<edm::InputTag>("HFRecHitCollectionLabel"));
  tok_ho_ = consumes<HORecHitCollection>(conf.getUntrackedParameter<edm::InputTag>("HORecHitCollectionLabel"));

  // register for data access
  tok_evt_ = consumes<edm::HepMCProduct>(edm::InputTag("generatorSmeared"));
  edm::InputTag EBRecHitCollectionLabel = conf.getParameter<edm::InputTag>("EBRecHitCollectionLabel");
  tok_EB_ = consumes<EBRecHitCollection>(EBRecHitCollectionLabel);
  edm::InputTag EERecHitCollectionLabel = conf.getParameter<edm::InputTag>("EERecHitCollectionLabel");
  tok_EE_ = consumes<EERecHitCollection>(EERecHitCollectionLabel);

  tok_hh_ = consumes<edm::PCaloHitContainer>(conf.getUntrackedParameter<edm::InputTag>("SimHitCollectionLabel"));

  tok_HRNDC_ = esConsumes<HcalDDDRecConstants, HcalRecNumberingRecord>();
  tok_Geom_ = esConsumes<CaloGeometry, CaloGeometryRecord>();

  subdet_ = 5;
  if (hcalselector_ == "noise")
    subdet_ = 0;
  if (hcalselector_ == "HB")
    subdet_ = 1;
  if (hcalselector_ == "HE")
    subdet_ = 2;
  if (hcalselector_ == "HO")
    subdet_ = 3;
  if (hcalselector_ == "HF")
    subdet_ = 4;
  if (hcalselector_ == "all")
    subdet_ = 5;
  if (hcalselector_ == "ZS")
    subdet_ = 6;

  iz = 1;
  if (sign_ == "-")
    iz = -1;
  if (sign_ == "*")
    iz = 0;

  imc = 1;
  if (mc_ == "no")
    imc = 0;
}

HcalRecHitsValidation::~HcalRecHitsValidation() {}

void HcalRecHitsValidation::bookHistograms(DQMStore::IBooker &ib, edm::Run const &run, edm::EventSetup const &es) {
  Char_t histo[200];

  ib.setCurrentFolder(topFolderName_);

  //======================= Now various cases one by one ===================

  // Histograms drawn for single pion scan
  if (subdet_ != 0 && imc != 0) {  // just not for noise
    sprintf(histo, "HcalRecHitTask_En_rechits_cone_profile_vs_ieta_all_depths");
    meEnConeEtaProfile = ib.bookProfile(histo, histo, 83, -41.5, 41.5, -100., 2000., " ");

    sprintf(histo, "HcalRecHitTask_En_rechits_cone_profile_vs_ieta_all_depths_E");
    meEnConeEtaProfile_E = ib.bookProfile(histo, histo, 83, -41.5, 41.5, -100., 2000., " ");

    sprintf(histo, "HcalRecHitTask_En_rechits_cone_profile_vs_ieta_all_depths_EH");
    meEnConeEtaProfile_EH = ib.bookProfile(histo, histo, 83, -41.5, 41.5, -100., 2000., " ");
  }

  // ************** HB **********************************
  if (subdet_ == 1 || subdet_ == 5) {
    sprintf(histo, "HcalRecHitTask_M2Log10Chi2_of_rechits_HB");  //   Chi2
    meRecHitsM2Chi2HB = ib.book1D(histo, histo, 120, -2., 10.);

    sprintf(histo, "HcalRecHitTask_Log10Chi2_vs_energy_profile_HB");
    meLog10Chi2profileHB = ib.bookProfile(histo, histo, 300, -5., 295., -2., 9.9, " ");

    sprintf(histo, "HcalRecHitTask_energy_of_rechits_HB");
    meRecHitsEnergyHB = ib.book1D(histo, histo, 2010, -10., 2000.);

    sprintf(histo, "HcalRecHitTask_timing_vs_energy_profile_HB");
    meTEprofileHB = ib.bookProfile(histo, histo, 150, -5., 295., -48., 92., " ");

    sprintf(histo, "HcalRecHitTask_timing_vs_energy_profile_Low_HB");
    meTEprofileHB_Low = ib.bookProfile(histo, histo, 150, -5., 295., -48., 92., " ");

    sprintf(histo, "HcalRecHitTask_timing_vs_energy_profile_High_HB");
    meTEprofileHB_High = ib.bookProfile(histo, histo, 150, -5., 295., 48., 92., " ");

    if (imc != 0) {
      sprintf(histo, "HcalRecHitTask_energy_rechits_vs_simhits_HB");
      meRecHitSimHitHB = ib.book2D(histo, histo, 120, 0., 1.2, 300, 0., 150.);
      sprintf(histo, "HcalRecHitTask_energy_rechits_vs_simhits_profile_HB");
      meRecHitSimHitProfileHB = ib.bookProfile(histo, histo, 120, 0., 1.2, 0., 500., " ");
    }
  }

  // ********************** HE ************************************
  if (subdet_ == 2 || subdet_ == 5) {
    sprintf(histo, "HcalRecHitTask_M2Log10Chi2_of_rechits_HE");  //   Chi2
    meRecHitsM2Chi2HE = ib.book1D(histo, histo, 120, -2., 10.);

    sprintf(histo, "HcalRecHitTask_Log10Chi2_vs_energy_profile_HE");
    meLog10Chi2profileHE = ib.bookProfile(histo, histo, 1000, -5., 995., -2., 9.9, " ");

    sprintf(histo, "HcalRecHitTask_energy_of_rechits_HE");
    meRecHitsEnergyHE = ib.book1D(histo, histo, 2010, -10., 2000.);

    sprintf(histo, "HcalRecHitTask_timing_vs_energy_profile_Low_HE");
    meTEprofileHE_Low = ib.bookProfile(histo, histo, 80, -5., 75., -48., 92., " ");

    sprintf(histo, "HcalRecHitTask_timing_vs_energy_profile_HE");
    meTEprofileHE = ib.bookProfile(histo, histo, 200, -5., 2995., -48., 92., " ");

    if (imc != 0) {
      sprintf(histo, "HcalRecHitTask_energy_rechits_vs_simhits_HE");
      meRecHitSimHitHE = ib.book2D(histo, histo, 120, 0., 0.6, 300, 0., 150.);
      sprintf(histo, "HcalRecHitTask_energy_rechits_vs_simhits_profile_HE");
      meRecHitSimHitProfileHE = ib.bookProfile(histo, histo, 120, 0., 0.6, 0., 500., " ");
    }
  }

  // ************** HO ****************************************
  if (subdet_ == 3 || subdet_ == 5) {
    sprintf(histo, "HcalRecHitTask_energy_of_rechits_HO");
    meRecHitsEnergyHO = ib.book1D(histo, histo, 2010, -10., 2000.);

    sprintf(histo, "HcalRecHitTask_timing_vs_energy_profile_HO");
    meTEprofileHO = ib.bookProfile(histo, histo, 60, -5., 55., -48., 92., " ");

    sprintf(histo, "HcalRecHitTask_timing_vs_energy_profile_High_HO");
    meTEprofileHO_High = ib.bookProfile(histo, histo, 100, -5., 995., -48., 92., " ");

    if (imc != 0) {
      sprintf(histo, "HcalRecHitTask_energy_rechits_vs_simhits_HO");
      meRecHitSimHitHO = ib.book2D(histo, histo, 150, 0., 1.5, 350, 0., 350.);
      sprintf(histo, "HcalRecHitTask_energy_rechits_vs_simhits_profile_HO");
      meRecHitSimHitProfileHO = ib.bookProfile(histo, histo, 150, 0., 1.5, 0., 500., " ");
    }
  }

  // ********************** HF ************************************
  if (subdet_ == 4 || subdet_ == 5) {
    sprintf(histo, "HcalRecHitTask_energy_of_rechits_HF");
    meRecHitsEnergyHF = ib.book1D(histo, histo, 2010, -10., 2000.);

    sprintf(histo, "HcalRecHitTask_timing_vs_energy_profile_Low_HF");
    meTEprofileHF_Low = ib.bookProfile(histo, histo, 100, -5., 195., -48., 92., " ");

    sprintf(histo, "HcalRecHitTask_timing_vs_energy_profile_HF");
    meTEprofileHF = ib.bookProfile(histo, histo, 200, -5., 995., -48., 92., " ");

    if (imc != 0) {
      sprintf(histo, "HcalRecHitTask_energy_rechits_vs_simhits_HF");
      meRecHitSimHitHF = ib.book2D(histo, histo, 50, 0., 50., 150, 0., 150.);
      sprintf(histo, "HcalRecHitTask_energy_rechits_vs_simhits_HFL");
      meRecHitSimHitHFL = ib.book2D(histo, histo, 50, 0., 50., 150, 0., 150.);
      sprintf(histo, "HcalRecHitTask_energy_rechits_vs_simhits_HFS");
      meRecHitSimHitHFS = ib.book2D(histo, histo, 50, 0., 50., 150, 0., 150.);
      sprintf(histo, "HcalRecHitTask_energy_rechits_vs_simhits_profile_HF");
      meRecHitSimHitProfileHF = ib.bookProfile(histo, histo, 50, 0., 50., 0., 500., " ");
      sprintf(histo, "HcalRecHitTask_energy_rechits_vs_simhits_profile_HFL");
      meRecHitSimHitProfileHFL = ib.bookProfile(histo, histo, 50, 0., 50., 0., 500., " ");
      sprintf(histo, "HcalRecHitTask_energy_rechits_vs_simhits_profile_HFS");
      meRecHitSimHitProfileHFS = ib.bookProfile(histo, histo, 50, 0., 50., 0., 500., " ");
    }
  }
}

void HcalRecHitsValidation::analyze(edm::Event const &ev, edm::EventSetup const &c) {
  using namespace edm;

  const HcalDDDRecConstants *hcons = &c.getData(tok_HRNDC_);

  // cuts for each subdet_ector mimiking  "Scheme B"
  //  double cutHB = 0.9, cutHE = 1.4, cutHO = 1.1, cutHFL = 1.2, cutHFS = 1.8;

  // energy in HCAL
  double eHcal = 0.;
  double eHcalCone = 0.;
  double eHcalConeHB = 0.;
  double eHcalConeHE = 0.;
  double eHcalConeHO = 0.;
  double eHcalConeHF = 0.;
  double eHcalConeHFL = 0.;
  double eHcalConeHFS = 0.;

  // Total numbet of RecHits in HCAL, in the cone, above 1 GeV theshold
  int nrechits = 0;
  int nrechitsCone = 0;
  int nrechitsThresh = 0;

  // energy in ECAL
  double eEcal = 0.;
  double eEcalB = 0.;
  double eEcalE = 0.;
  double eEcalCone = 0.;
  int numrechitsEcal = 0;

  // MC info
  double phi_MC = -999999.;  // phi of initial particle from HepMC
  double eta_MC = -999999.;  // eta of initial particle from HepMC

  // HCAL energy around MC eta-phi at all depths;
  double partR = 0.3;

  if (imc != 0) {
    edm::Handle<edm::HepMCProduct> evtMC;
    ev.getByToken(tok_evt_, evtMC);  // generator in late 310_preX
    if (!evtMC.isValid()) {
      edm::LogInfo("HcalRecHitsValidation") << "no HepMCProduct found";
    } else {
      //    std::cout << "*** source HepMCProduct found"<< std::endl;
    }

    // MC particle with highest pt is taken as a direction reference
    double maxPt = -99999.;
    int npart = 0;
    const HepMC::GenEvent *myGenEvent = evtMC->GetEvent();
    for (HepMC::GenEvent::particle_const_iterator p = myGenEvent->particles_begin(); p != myGenEvent->particles_end();
         ++p) {
      double phip = (*p)->momentum().phi();
      double etap = (*p)->momentum().eta();
      //    phi_MC = phip;
      //    eta_MC = etap;
      double pt = (*p)->momentum().perp();
      if (pt > maxPt) {
        npart++;
        maxPt = pt;
        phi_MC = phip;
        eta_MC = etap;
      }
    }
    //  std::cout << "*** Max pT = " << maxPt <<  std::endl;
  }

  //   std::cout << "*** 2" << std::endl;

  geometry_ = &c.getData(tok_Geom_);
  ;

  // Fill working vectors of HCAL RecHits quantities (all of these are drawn)
  fillRecHitsTmp(subdet_, ev);

  //  std::cout << "*** 3" << std::endl;

  //===========================================================================
  // IN ALL other CASES : ieta-iphi maps
  //===========================================================================

  // ECAL
  if (ecalselector_ == "yes" && (subdet_ == 1 || subdet_ == 2 || subdet_ == 5)) {
    Handle<EBRecHitCollection> rhitEB;

    EcalRecHitCollection::const_iterator RecHit;
    EcalRecHitCollection::const_iterator RecHitEnd;

    if (ev.getByToken(tok_EB_, rhitEB)) {
      RecHit = rhitEB.product()->begin();
      RecHitEnd = rhitEB.product()->end();

      for (; RecHit != RecHitEnd; ++RecHit) {
        EBDetId EBid = EBDetId(RecHit->id());

        auto cellGeometry = geometry_->getSubdetectorGeometry(EBid)->getGeometry(EBid);
        double eta = cellGeometry->getPosition().eta();
        double phi = cellGeometry->getPosition().phi();
        double en = RecHit->energy();
        eEcal += en;
        eEcalB += en;

        double r = dR(eta_MC, phi_MC, eta, phi);
        if (r < partR) {
          eEcalCone += en;
          numrechitsEcal++;
        }
      }
    }

    Handle<EERecHitCollection> rhitEE;

    if (ev.getByToken(tok_EE_, rhitEE)) {
      RecHit = rhitEE.product()->begin();
      RecHitEnd = rhitEE.product()->end();

      for (; RecHit != RecHitEnd; ++RecHit) {
        EEDetId EEid = EEDetId(RecHit->id());

        auto cellGeometry = geometry_->getSubdetectorGeometry(EEid)->getGeometry(EEid);
        double eta = cellGeometry->getPosition().eta();
        double phi = cellGeometry->getPosition().phi();
        double en = RecHit->energy();
        eEcal += en;
        eEcalE += en;

        double r = dR(eta_MC, phi_MC, eta, phi);
        if (r < partR) {
          eEcalCone += en;
          numrechitsEcal++;
        }
      }
    }
  }  // end of ECAL selection

  //     std::cout << "*** 4" << std::endl;

  //===========================================================================
  // SUBSYSTEMS,
  //===========================================================================

  if ((subdet_ != 6) && (subdet_ != 0)) {
    //       std::cout << "*** 6" << std::endl;

    double HcalCone = 0.;

    int ietaMax = 9999;
    double etaMax = 9999.;

    //   CYCLE over cells ====================================================

    for (unsigned int i = 0; i < cen.size(); i++) {
      int sub = csub[i];
      int depth = cdepth[i];
      double eta = ceta[i];
      double phi = cphi[i];
      double en = cen[i];
      double t = ctime[i];
      int ieta = cieta[i];
      double chi2 = cchi2[i];

      double chi2_log10 = 9.99;  // initial value above histos limits , keep it if chi2 <= 0.
      if (chi2 > 0.)
        chi2_log10 = log10(chi2);

      nrechits++;
      eHcal += en;
      if (en > 1.)
        nrechitsThresh++;

      double r = dR(eta_MC, phi_MC, eta, phi);
      if (r < partR) {
        if (sub == 1)
          eHcalConeHB += en;
        if (sub == 2)
          eHcalConeHE += en;
        if (sub == 3)
          eHcalConeHO += en;
        if (sub == 4) {
          eHcalConeHF += en;
          if (depth == 1)
            eHcalConeHFL += en;
          else
            eHcalConeHFS += en;
        }
        eHcalCone += en;
        nrechitsCone++;

        HcalCone += en;

        // alternative: ietamax -> closest to MC eta  !!!
        float eta_diff = fabs(eta_MC - eta);
        if (eta_diff < etaMax) {
          etaMax = eta_diff;
          ietaMax = ieta;
        }
      }

      // The energy and overall timing histos are drawn while
      // the ones split by depth are not
      if (sub == 1 && (subdet_ == 1 || subdet_ == 5)) {
        meRecHitsM2Chi2HB->Fill(chi2_log10);
        meLog10Chi2profileHB->Fill(en, chi2_log10);

        meRecHitsEnergyHB->Fill(en);

        meTEprofileHB_Low->Fill(en, t);
        meTEprofileHB->Fill(en, t);
        meTEprofileHB_High->Fill(en, t);
      }
      if (sub == 2 && (subdet_ == 2 || subdet_ == 5)) {
        meRecHitsM2Chi2HE->Fill(chi2_log10);
        meLog10Chi2profileHE->Fill(en, chi2_log10);

        meRecHitsEnergyHE->Fill(en);

        meTEprofileHE_Low->Fill(en, t);
        meTEprofileHE->Fill(en, t);
      }
      if (sub == 4 && (subdet_ == 4 || subdet_ == 5)) {
        meRecHitsEnergyHF->Fill(en);

        meTEprofileHF_Low->Fill(en, t);
        meTEprofileHF->Fill(en, t);
      }
      if (sub == 3 && (subdet_ == 3 || subdet_ == 5)) {
        meRecHitsEnergyHO->Fill(en);

        meTEprofileHO->Fill(en, t);
        meTEprofileHO_High->Fill(en, t);
      }
    }

    if (imc != 0) {
      meEnConeEtaProfile->Fill(double(ietaMax), HcalCone);  //
      meEnConeEtaProfile_E->Fill(double(ietaMax), eEcalCone);
      meEnConeEtaProfile_EH->Fill(double(ietaMax), HcalCone + eEcalCone);
    }

    //     std::cout << "*** 7" << std::endl;
  }

  // SimHits vs. RecHits
  if (subdet_ > 0 && subdet_ < 6 && imc != 0) {  // not noise

    edm::Handle<PCaloHitContainer> hcalHits;
    if (ev.getByToken(tok_hh_, hcalHits)) {
      const PCaloHitContainer *SimHitResult = hcalHits.product();

      double enSimHits = 0.;
      double enSimHitsHB = 0.;
      double enSimHitsHE = 0.;
      double enSimHitsHO = 0.;
      double enSimHitsHF = 0.;
      double enSimHitsHFL = 0.;
      double enSimHitsHFS = 0.;
      // sum of SimHits in the cone

      for (std::vector<PCaloHit>::const_iterator SimHits = SimHitResult->begin(); SimHits != SimHitResult->end();
           ++SimHits) {
        int sub, depth;
        HcalDetId cell;

        if (testNumber_)
          cell = HcalHitRelabeller::relabel(SimHits->id(), hcons);
        else
          cell = HcalDetId(SimHits->id());

        sub = cell.subdet();
        depth = cell.depth();

        if (sub != subdet_ && subdet_ != 5)
          continue;  // If we are not looking at all of the subdetectors and the
                     // simhit doesn't come from the specific subdetector of
                     // interest, then we won't do any thing with it

        const HcalGeometry *cellGeometry =
            dynamic_cast<const HcalGeometry *>(geometry_->getSubdetectorGeometry(DetId::Hcal, cell.subdet()));
        double etaS = cellGeometry->getPosition(cell).eta();
        double phiS = cellGeometry->getPosition(cell).phi();
        double en = SimHits->energy();

        double r = dR(eta_MC, phi_MC, etaS, phiS);

        if (r < partR) {  // just energy in the small cone

          enSimHits += en;
          if (sub == static_cast<int>(HcalBarrel))
            enSimHitsHB += en;
          if (sub == static_cast<int>(HcalEndcap))
            enSimHitsHE += en;
          if (sub == static_cast<int>(HcalOuter))
            enSimHitsHO += en;
          if (sub == static_cast<int>(HcalForward)) {
            enSimHitsHF += en;
            if (depth == 1)
              enSimHitsHFL += en;
            else
              enSimHitsHFS += en;
          }
        }
      }

      // Now some histos with SimHits

      if (subdet_ == 4 || subdet_ == 5) {
        meRecHitSimHitHF->Fill(enSimHitsHF, eHcalConeHF);
        meRecHitSimHitProfileHF->Fill(enSimHitsHF, eHcalConeHF);

        meRecHitSimHitHFL->Fill(enSimHitsHFL, eHcalConeHFL);
        meRecHitSimHitProfileHFL->Fill(enSimHitsHFL, eHcalConeHFL);
        meRecHitSimHitHFS->Fill(enSimHitsHFS, eHcalConeHFS);
        meRecHitSimHitProfileHFS->Fill(enSimHitsHFS, eHcalConeHFS);
      }
      if (subdet_ == 1 || subdet_ == 5) {
        meRecHitSimHitHB->Fill(enSimHitsHB, eHcalConeHB);
        meRecHitSimHitProfileHB->Fill(enSimHitsHB, eHcalConeHB);
      }
      if (subdet_ == 2 || subdet_ == 5) {
        meRecHitSimHitHE->Fill(enSimHitsHE, eHcalConeHE);
        meRecHitSimHitProfileHE->Fill(enSimHitsHE, eHcalConeHE);
      }
      if (subdet_ == 3 || subdet_ == 5) {
        meRecHitSimHitHO->Fill(enSimHitsHO, eHcalConeHO);
        meRecHitSimHitProfileHO->Fill(enSimHitsHO, eHcalConeHO);
      }
    }
  }

  nevtot++;
}

///////////////////////////////////////////////////////////////////////////////
void HcalRecHitsValidation::fillRecHitsTmp(int subdet_, edm::Event const &ev) {
  using namespace edm;

  // initialize data vectors
  csub.clear();
  cen.clear();
  ceta.clear();
  cphi.clear();
  ctime.clear();
  cieta.clear();
  ciphi.clear();
  cdepth.clear();
  cz.clear();
  cchi2.clear();

  if (subdet_ == 1 || subdet_ == 2 || subdet_ == 5 || subdet_ == 6 || subdet_ == 0) {
    // HBHE
    edm::Handle<HBHERecHitCollection> hbhecoll;
    if (ev.getByToken(tok_hbhe_, hbhecoll)) {
      for (HBHERecHitCollection::const_iterator j = hbhecoll->begin(); j != hbhecoll->end(); j++) {
        HcalDetId cell(j->id());
        const HcalGeometry *cellGeometry =
            dynamic_cast<const HcalGeometry *>(geometry_->getSubdetectorGeometry(DetId::Hcal, cell.subdet()));
        double eta = cellGeometry->getPosition(cell).eta();
        double phi = cellGeometry->getPosition(cell).phi();
        double zc = cellGeometry->getPosition(cell).z();
        int sub = cell.subdet();
        int depth = cell.depth();
        int inteta = cell.ieta();
        int intphi = cell.iphi() - 1;
        double en = j->energy();
        double t = j->time();
        double chi2 = j->chi2();

        if ((iz > 0 && eta > 0.) || (iz < 0 && eta < 0.) || iz == 0) {
          csub.push_back(sub);
          cen.push_back(en);
          ceta.push_back(eta);
          cphi.push_back(phi);
          ctime.push_back(t);
          cieta.push_back(inteta);
          ciphi.push_back(intphi);
          cdepth.push_back(depth);
          cz.push_back(zc);
          cchi2.push_back(chi2);
        }
      }
    }
  }

  if (subdet_ == 4 || subdet_ == 5 || subdet_ == 6 || subdet_ == 0) {
    // HF
    edm::Handle<HFRecHitCollection> hfcoll;
    if (ev.getByToken(tok_hf_, hfcoll)) {
      for (HFRecHitCollection::const_iterator j = hfcoll->begin(); j != hfcoll->end(); j++) {
        HcalDetId cell(j->id());
        auto cellGeometry = geometry_->getSubdetectorGeometry(cell)->getGeometry(cell);

        double eta = cellGeometry->getPosition().eta();
        double phi = cellGeometry->getPosition().phi();
        double zc = cellGeometry->getPosition().z();
        int sub = cell.subdet();
        int depth = cell.depth();
        int inteta = cell.ieta();
        int intphi = cell.iphi() - 1;
        double en = j->energy();
        double t = j->time();

        if ((iz > 0 && eta > 0.) || (iz < 0 && eta < 0.) || iz == 0) {
          csub.push_back(sub);
          cen.push_back(en);
          ceta.push_back(eta);
          cphi.push_back(phi);
          ctime.push_back(t);
          cieta.push_back(inteta);
          ciphi.push_back(intphi);
          cdepth.push_back(depth);
          cz.push_back(zc);
          cchi2.push_back(0.);
        }
      }
    }
  }

  // HO
  if (subdet_ == 3 || subdet_ == 5 || subdet_ == 6 || subdet_ == 0) {
    edm::Handle<HORecHitCollection> hocoll;
    if (ev.getByToken(tok_ho_, hocoll)) {
      for (HORecHitCollection::const_iterator j = hocoll->begin(); j != hocoll->end(); j++) {
        HcalDetId cell(j->id());
        auto cellGeometry = geometry_->getSubdetectorGeometry(cell)->getGeometry(cell);

        double eta = cellGeometry->getPosition().eta();
        double phi = cellGeometry->getPosition().phi();
        double zc = cellGeometry->getPosition().z();
        int sub = cell.subdet();
        int depth = cell.depth();
        int inteta = cell.ieta();
        int intphi = cell.iphi() - 1;
        double t = j->time();
        double en = j->energy();

        if ((iz > 0 && eta > 0.) || (iz < 0 && eta < 0.) || iz == 0) {
          csub.push_back(sub);
          cen.push_back(en);
          ceta.push_back(eta);
          cphi.push_back(phi);
          ctime.push_back(t);
          cieta.push_back(inteta);
          ciphi.push_back(intphi);
          cdepth.push_back(depth);
          cz.push_back(zc);
          cchi2.push_back(0.);
        }
      }
    }
  }
}

double HcalRecHitsValidation::dR(double eta1, double phi1, double eta2, double phi2) {
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

double HcalRecHitsValidation::phi12(double phi1, double en1, double phi2, double en2) {
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

double HcalRecHitsValidation::dPhiWsign(double phi1, double phi2) {
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

DEFINE_FWK_MODULE(HcalRecHitsValidation);
