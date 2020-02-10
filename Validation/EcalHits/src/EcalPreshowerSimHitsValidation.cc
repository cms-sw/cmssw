/*
 * \file EcalPreshowerSimHitsValidation.cc
 *
 * \author C.Rovelli
 *
 */

#include "DQMServices/Core/interface/DQMStore.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "Validation/EcalHits/interface/EcalPreshowerSimHitsValidation.h"
#include <DataFormats/EcalDetId/interface/EEDetId.h>
#include <DataFormats/EcalDetId/interface/ESDetId.h>

using namespace cms;
using namespace edm;
using namespace std;

EcalPreshowerSimHitsValidation::EcalPreshowerSimHitsValidation(const edm::ParameterSet &ps)
    :

      HepMCLabel(ps.getParameter<std::string>("moduleLabelMC")),
      g4InfoLabel(ps.getParameter<std::string>("moduleLabelG4")),
      EEHitsCollection(ps.getParameter<std::string>("EEHitsCollection")),
      ESHitsCollection(ps.getParameter<std::string>("ESHitsCollection")) {
  HepMCToken = consumes<edm::HepMCProduct>(HepMCLabel);
  EEHitsToken =
      consumes<edm::PCaloHitContainer>(edm::InputTag(std::string(g4InfoLabel), std::string(EEHitsCollection)));
  ESHitsToken =
      consumes<edm::PCaloHitContainer>(edm::InputTag(std::string(g4InfoLabel), std::string(ESHitsCollection)));
  // verbosity switch
  verbose_ = ps.getUntrackedParameter<bool>("verbose", false);

  // get hold of back-end interface
  dbe_ = nullptr;
  dbe_ = edm::Service<DQMStore>().operator->();

  menESHits1zp_ = nullptr;
  menESHits2zp_ = nullptr;
  menESHits1zm_ = nullptr;
  menESHits2zm_ = nullptr;

  meESEnergyHits1zp_ = nullptr;
  meESEnergyHits2zp_ = nullptr;
  meESEnergyHits1zm_ = nullptr;
  meESEnergyHits2zm_ = nullptr;

  meEShitLog10Energy_ = nullptr;
  meEShitLog10EnergyNorm_ = nullptr;

  meE1alphaE2zp_ = nullptr;
  meE1alphaE2zm_ = nullptr;
  meEEoverESzp_ = nullptr;
  meEEoverESzm_ = nullptr;

  me2eszpOver1eszp_ = nullptr;
  me2eszmOver1eszm_ = nullptr;

  Char_t histo[200];

  if (dbe_) {
    dbe_->setCurrentFolder("EcalHitsV/EcalSimHitsValidation");
    dbe_->setScope(MonitorElementData::Scope::RUN);

    sprintf(histo, "ES hits layer 1 multiplicity z+");
    menESHits1zp_ = dbe_->book1D(histo, histo, 50, 0., 50.);

    sprintf(histo, "ES hits layer 2 multiplicity z+");
    menESHits2zp_ = dbe_->book1D(histo, histo, 50, 0., 50.);

    sprintf(histo, "ES hits layer 1 multiplicity z-");
    menESHits1zm_ = dbe_->book1D(histo, histo, 50, 0., 50.);

    sprintf(histo, "ES hits layer 2 multiplicity z-");
    menESHits2zm_ = dbe_->book1D(histo, histo, 50, 0., 50.);

    sprintf(histo, "ES hits energy layer 1 z+");
    meESEnergyHits1zp_ = dbe_->book1D(histo, histo, 100, 0., 0.05);

    sprintf(histo, "ES hits energy layer 2 z+");
    meESEnergyHits2zp_ = dbe_->book1D(histo, histo, 100, 0., 0.05);

    sprintf(histo, "ES hits energy layer 1 z-");
    meESEnergyHits1zm_ = dbe_->book1D(histo, histo, 100, 0., 0.05);

    sprintf(histo, "ES hits energy layer 2 z-");
    meESEnergyHits2zm_ = dbe_->book1D(histo, histo, 100, 0., 0.05);

    sprintf(histo, "ES hits log10energy spectrum");
    meEShitLog10Energy_ = dbe_->book1D(histo, histo, 140, -10., 4.);

    sprintf(histo, "ES hits log10energy spectrum vs normalized energy");
    meEShitLog10EnergyNorm_ = dbe_->bookProfile(histo, histo, 140, -10., 4., 100, 0., 1.);

    sprintf(histo, "ES E1+07E2 z+");
    meE1alphaE2zp_ = dbe_->book1D(histo, histo, 100, 0., 0.05);

    sprintf(histo, "ES E1+07E2 z-");
    meE1alphaE2zm_ = dbe_->book1D(histo, histo, 100, 0., 0.05);

    sprintf(histo, "EE vs ES z+");
    meEEoverESzp_ = dbe_->bookProfile(histo, histo, 250, 0., 500., 200, 0., 200.);

    sprintf(histo, "EE vs ES z-");
    meEEoverESzm_ = dbe_->bookProfile(histo, histo, 250, 0., 500., 200, 0., 200.);

    sprintf(histo, "ES ene2oEne1 z+");
    me2eszpOver1eszp_ = dbe_->book1D(histo, histo, 50, 0., 10.);

    sprintf(histo, "ES ene2oEne1 z-");
    me2eszmOver1eszm_ = dbe_->book1D(histo, histo, 50, 0., 10.);
  }
}

EcalPreshowerSimHitsValidation::~EcalPreshowerSimHitsValidation() {}

void EcalPreshowerSimHitsValidation::beginJob() {}

void EcalPreshowerSimHitsValidation::endJob() {}

void EcalPreshowerSimHitsValidation::analyze(const edm::Event &e, const edm::EventSetup &c) {
  edm::LogInfo("EventInfo") << " Run = " << e.id().run() << " Event = " << e.id().event();

  edm::Handle<edm::HepMCProduct> MCEvt;
  e.getByToken(HepMCToken, MCEvt);

  edm::Handle<edm::PCaloHitContainer> EcalHitsEE;
  e.getByToken(EEHitsToken, EcalHitsEE);

  edm::Handle<edm::PCaloHitContainer> EcalHitsES;
  e.getByToken(ESHitsToken, EcalHitsES);

  std::vector<PCaloHit> theEECaloHits;
  if (EcalHitsEE.isValid()) {
    theEECaloHits.insert(theEECaloHits.end(), EcalHitsEE->begin(), EcalHitsEE->end());
  }

  std::vector<PCaloHit> theESCaloHits;
  if (EcalHitsES.isValid()) {
    theESCaloHits.insert(theESCaloHits.end(), EcalHitsES->begin(), EcalHitsES->end());
  }

  double ESEnergy_ = 0.;
  // std::map<unsigned int, std::vector<PCaloHit>,std::less<unsigned int> >
  // CaloHitMap;

  // endcap
  double EEetzp_ = 0.;
  double EEetzm_ = 0.;
  for (std::vector<PCaloHit>::iterator isim = theEECaloHits.begin(); isim != theEECaloHits.end(); ++isim) {
    EEDetId eeid(isim->id());
    if (eeid.zside() > 0)
      EEetzp_ += isim->energy();
    if (eeid.zside() < 0)
      EEetzm_ += isim->energy();
  }

  uint32_t nESHits1zp = 0;
  uint32_t nESHits1zm = 0;
  uint32_t nESHits2zp = 0;
  uint32_t nESHits2zm = 0;
  double ESet1zp_ = 0.;
  double ESet2zp_ = 0.;
  double ESet1zm_ = 0.;
  double ESet2zm_ = 0.;
  std::vector<double> econtr(140, 0.);

  for (std::vector<PCaloHit>::iterator isim = theESCaloHits.begin(); isim != theESCaloHits.end(); ++isim) {
    // CaloHitMap[(*isim).id()].push_back((*isim));

    ESDetId esid(isim->id());

    LogDebug("HitInfo") << " CaloHit " << isim->getName() << "\n"
                        << " DetID = " << isim->id() << " ESDetId: z side " << esid.zside() << "  plane "
                        << esid.plane() << esid.six() << ',' << esid.siy() << ':' << esid.strip() << "\n"
                        << " Time = " << isim->time() << "\n"
                        << " Track Id = " << isim->geantTrackId() << "\n"
                        << " Energy = " << isim->energy();

    ESEnergy_ += isim->energy();
    if (isim->energy() > 0) {
      meEShitLog10Energy_->Fill(log10(isim->energy()));
      int log10i = int((log10(isim->energy()) + 10.) * 10.);
      if (log10i >= 0 && log10i < 140)
        econtr[log10i] += isim->energy();
    }

    if (esid.plane() == 1) {
      if (esid.zside() > 0) {
        nESHits1zp++;
        ESet1zp_ += isim->energy();
        if (meESEnergyHits1zp_)
          meESEnergyHits1zp_->Fill(isim->energy());
      } else if (esid.zside() < 0) {
        nESHits1zm++;
        ESet1zm_ += isim->energy();
        if (meESEnergyHits1zm_)
          meESEnergyHits1zm_->Fill(isim->energy());
      }
    } else if (esid.plane() == 2) {
      if (esid.zside() > 0) {
        nESHits2zp++;
        ESet2zp_ += isim->energy();
        if (meESEnergyHits2zp_)
          meESEnergyHits2zp_->Fill(isim->energy());
      } else if (esid.zside() < 0) {
        nESHits2zm++;
        ESet2zm_ += isim->energy();
        if (meESEnergyHits2zm_)
          meESEnergyHits2zm_->Fill(isim->energy());
      }
    }
  }

  if (menESHits1zp_)
    menESHits1zp_->Fill(nESHits1zp);
  if (menESHits1zm_)
    menESHits1zm_->Fill(nESHits1zm);

  if (menESHits2zp_)
    menESHits2zp_->Fill(nESHits2zp);
  if (menESHits2zm_)
    menESHits2zm_->Fill(nESHits2zm);

  if (meEShitLog10EnergyNorm_ && ESEnergy_ != 0) {
    for (int i = 0; i < 140; i++) {
      meEShitLog10EnergyNorm_->Fill(-10. + (float(i) + 0.5) / 10., econtr[i] / ESEnergy_);
    }
  }

  for (HepMC::GenEvent::particle_const_iterator p = MCEvt->GetEvent()->particles_begin();
       p != MCEvt->GetEvent()->particles_end();
       ++p) {
    double htheta = (*p)->momentum().theta();
    double heta = -99999.;
    if (tan(htheta * 0.5) > 0) {
      heta = -log(tan(htheta * 0.5));
    }

    if (heta > 1.653 && heta < 2.6) {
      if (meE1alphaE2zp_)
        meE1alphaE2zp_->Fill(ESet1zp_ + 0.7 * ESet2zp_);
      if (meEEoverESzp_)
        meEEoverESzp_->Fill((ESet1zp_ + 0.7 * ESet2zp_) / 0.00009, EEetzp_);
      if ((me2eszpOver1eszp_) && (ESet1zp_ != 0.))
        me2eszpOver1eszp_->Fill(ESet2zp_ / ESet1zp_);
    }
    if (heta < -1.653 && heta > -2.6) {
      if (meE1alphaE2zm_)
        meE1alphaE2zm_->Fill(ESet1zm_ + 0.7 * ESet2zm_);
      if (meEEoverESzm_)
        meEEoverESzm_->Fill((ESet1zm_ + 0.7 * ESet2zm_) / 0.00009, EEetzm_);
      if ((me2eszmOver1eszm_) && (ESet1zm_ != 0.))
        me2eszmOver1eszm_->Fill(ESet2zm_ / ESet1zm_);
    }
  }
}

//  LocalWords:  EcalSimHitsValidation
