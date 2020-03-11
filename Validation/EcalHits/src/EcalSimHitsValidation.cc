/*
 * \file EcalSimHitsValidation.cc
 *
 * \author C.Rovelli
 *
 */

#include "DQMServices/Core/interface/DQMStore.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "Validation/EcalHits/interface/EcalSimHitsValidation.h"
#include <DataFormats/EcalDetId/interface/EBDetId.h>
#include <DataFormats/EcalDetId/interface/EEDetId.h>
#include <DataFormats/EcalDetId/interface/ESDetId.h>

using namespace cms;
using namespace edm;
using namespace std;

EcalSimHitsValidation::EcalSimHitsValidation(const edm::ParameterSet &ps)
    : g4InfoLabel(ps.getParameter<std::string>("moduleLabelG4")),
      HepMCToken(consumes<edm::HepMCProduct>(ps.getParameter<std::string>("moduleLabelMC"))) {
  EBHitsCollectionToken =
      consumes<edm::PCaloHitContainer>(edm::InputTag(g4InfoLabel, ps.getParameter<std::string>("EBHitsCollection")));
  EEHitsCollectionToken =
      consumes<edm::PCaloHitContainer>(edm::InputTag(g4InfoLabel, ps.getParameter<std::string>("EEHitsCollection")));
  ESHitsCollectionToken =
      consumes<edm::PCaloHitContainer>(edm::InputTag(g4InfoLabel, ps.getParameter<std::string>("ESHitsCollection")));

  // verbosity switch
  verbose_ = ps.getUntrackedParameter<bool>("verbose", false);
}

void EcalSimHitsValidation::bookHistograms(DQMStore::IBooker &ib, edm::Run const &, edm::EventSetup const &c) {
  ib.setCurrentFolder("EcalHitsV/EcalSimHitsValidation");
  ib.setScope(MonitorElementData::Scope::RUN);

  std::string histo = "EcalSimHitsValidation Gun Momentum";
  meGunEnergy_ = ib.book1D(histo, histo, 100, 0., 1000.);

  histo = "EcalSimHitsValidation Gun Eta";
  meGunEta_ = ib.book1D(histo, histo, 700, -3.5, 3.5);

  histo = "EcalSimHitsValidation Gun Phi";
  meGunPhi_ = ib.book1D(histo, histo, 360, 0., 360.);

  histo = "EcalSimHitsValidation Barrel fraction of energy";
  meEBEnergyFraction_ = ib.book1D(histo, histo, 100, 0., 1.1);

  histo = "EcalSimHitsValidation Endcap fraction of energy";
  meEEEnergyFraction_ = ib.book1D(histo, histo, 100, 0., 1.1);

  histo = "EcalSimHitsValidation Preshower fraction of energy";
  meESEnergyFraction_ = ib.book1D(histo, histo, 60, 0., 0.001);
}

void EcalSimHitsValidation::analyze(const edm::Event &e, const edm::EventSetup &c) {
  edm::LogInfo("EventInfo") << " Run = " << e.id().run() << " Event = " << e.id().event();

  std::vector<PCaloHit> theEBCaloHits;
  std::vector<PCaloHit> theEECaloHits;
  std::vector<PCaloHit> theESCaloHits;

  edm::Handle<edm::HepMCProduct> MCEvt;
  edm::Handle<edm::PCaloHitContainer> EcalHitsEB;
  edm::Handle<edm::PCaloHitContainer> EcalHitsEE;
  edm::Handle<edm::PCaloHitContainer> EcalHitsES;

  e.getByToken(HepMCToken, MCEvt);
  e.getByToken(EBHitsCollectionToken, EcalHitsEB);
  e.getByToken(EEHitsCollectionToken, EcalHitsEE);
  e.getByToken(ESHitsCollectionToken, EcalHitsES);

  for (HepMC::GenEvent::particle_const_iterator p = MCEvt->GetEvent()->particles_begin();
       p != MCEvt->GetEvent()->particles_end();
       ++p) {
    double htheta = (*p)->momentum().theta();
    double heta = -99999.;
    if (tan(htheta * 0.5) > 0) {
      heta = -log(tan(htheta * 0.5));
    }
    double hphi = (*p)->momentum().phi();
    hphi = (hphi >= 0) ? hphi : hphi + 2 * M_PI;
    hphi = hphi / M_PI * 180.;

    LogDebug("EventInfo") << "Particle gun type form MC = " << abs((*p)->pdg_id()) << "\n"
                          << "Energy = " << (*p)->momentum().e() << " Eta = " << heta << " Phi = " << hphi;

    meGunEnergy_->Fill((*p)->momentum().e());
    meGunEta_->Fill(heta);
    meGunPhi_->Fill(hphi);
  }

  double EBEnergy_ = 0.;
  if (EcalHitsEB.isValid()) {
    theEBCaloHits.insert(theEBCaloHits.end(), EcalHitsEB->begin(), EcalHitsEB->end());
    for (std::vector<PCaloHit>::iterator isim = theEBCaloHits.begin(); isim != theEBCaloHits.end(); ++isim) {
      EBEnergy_ += isim->energy();
    }
  }

  double EEEnergy_ = 0.;
  if (EcalHitsEE.isValid()) {
    theEECaloHits.insert(theEECaloHits.end(), EcalHitsEE->begin(), EcalHitsEE->end());
    for (std::vector<PCaloHit>::iterator isim = theEECaloHits.begin(); isim != theEECaloHits.end(); ++isim) {
      EEEnergy_ += isim->energy();
    }
  }

  double ESEnergy_ = 0.;
  if (EcalHitsES.isValid()) {
    theESCaloHits.insert(theESCaloHits.end(), EcalHitsES->begin(), EcalHitsES->end());
    for (std::vector<PCaloHit>::iterator isim = theESCaloHits.begin(); isim != theESCaloHits.end(); ++isim) {
      ESEnergy_ += isim->energy();
    }
  }

  double etot = EBEnergy_ + EEEnergy_ + ESEnergy_;
  double fracEB = 0.0;
  double fracEE = 0.0;
  double fracES = 0.0;

  if (etot > 0.0) {
    fracEB = EBEnergy_ / etot;
    fracEE = EEEnergy_ / etot;
    fracES = ESEnergy_ / etot;
  }

  meEBEnergyFraction_->Fill(fracEB);
  meEEEnergyFraction_->Fill(fracEE);
  meESEnergyFraction_->Fill(fracES);
}
