/*
 * \file EcalSimHitsValidation.cc
 *
 * \author C.Rovelli
 *
*/

#include "Validation/EcalHits/interface/EcalSimHitsValidation.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "SimDataFormats/CaloHit/interface/PCaloHit.h"

#include "DQMServices/Core/interface/MonitorElement.h"

#include "FWCore/Framework/interface/MakerMacros.h"

EcalSimHitsValidation::EcalSimHitsValidation(const edm::ParameterSet& ps)
{
  HepMCToken = consumes<edm::HepMCProduct>(edm::InputTag(ps.getParameter<std::string>("moduleLabelMC")));

  std::string g4InfoLabel(ps.getParameter<std::string>("moduleLabelG4"));
  EBHitsCollectionToken = consumes<edm::PCaloHitContainer>(edm::InputTag(g4InfoLabel, ps.getParameter<std::string>("EBHitsCollection")));
  EEHitsCollectionToken = consumes<edm::PCaloHitContainer>(edm::InputTag(g4InfoLabel, ps.getParameter<std::string>("EEHitsCollection")));
  ESHitsCollectionToken = consumes<edm::PCaloHitContainer>(edm::InputTag(g4InfoLabel, ps.getParameter<std::string>("ESHitsCollection")));

  meGunEnergy_ = 0;
  meGunEta_    = 0;   
  meGunPhi_    = 0;   
  meEBEnergyFraction_  = 0;
  meEEEnergyFraction_  = 0;
  meESEnergyFraction_  = 0;
}

EcalSimHitsValidation::~EcalSimHitsValidation()
{
}

void
EcalSimHitsValidation::bookHistograms(DQMStore::IBooker& _ibooker, edm::Run const&, edm::EventSetup const&)
{
  _ibooker.setCurrentFolder("EcalHitsV/EcalSimHitsValidation");

  std::string name;
  
  name = "EcalSimHitsValidation Gun Momentum";
  meGunEnergy_ = _ibooker.book1D(name, name, 100, 0., 1000.);
  
  name = "EcalSimHitsValidation Gun Eta";
  meGunEta_ = _ibooker.book1D(name, name, 700, -3.5, 3.5);
  
  name = "EcalSimHitsValidation Gun Phi";
  meGunPhi_ = _ibooker.book1D(name, name, 360, 0., 360.);

  name = "EcalSimHitsValidation Barrel fraction of energy";
  meEBEnergyFraction_ = _ibooker.book1D(name, name, 100 , 0. , 1.1);
  
  name = "EcalSimHitsValidation Endcap fraction of energy";
  meEEEnergyFraction_ = _ibooker.book1D(name, name, 100 , 0. , 1.1);
  
  name = "EcalSimHitsValidation Preshower fraction of energy";
  meESEnergyFraction_ = _ibooker.book1D(name, name, 60 , 0. , 0.001);
}

void
EcalSimHitsValidation::analyze(edm::Event const& e, edm::EventSetup const&)
{
  edm::LogInfo("EventInfo") << " Run = " << e.id().run() << " Event = " << e.id().event();
  
  std::vector<PCaloHit>  theEBCaloHits;
  std::vector<PCaloHit>  theEECaloHits;
  std::vector<PCaloHit>  theESCaloHits;

  edm::Handle<edm::HepMCProduct> MCEvt;
  edm::Handle<edm::PCaloHitContainer> EcalHitsEB;
  edm::Handle<edm::PCaloHitContainer> EcalHitsEE;
  edm::Handle<edm::PCaloHitContainer> EcalHitsES;

  e.getByToken(HepMCToken, MCEvt);
  e.getByToken(EBHitsCollectionToken, EcalHitsEB);
  e.getByToken(EEHitsCollectionToken, EcalHitsEE);
  e.getByToken(ESHitsCollectionToken, EcalHitsES);

  for ( HepMC::GenEvent::particle_const_iterator p = MCEvt->GetEvent()->particles_begin();
        p != MCEvt->GetEvent()->particles_end(); ++p ) {

    double htheta = (*p)->momentum().theta();
    double heta = -99999.;
    if( tan(htheta * 0.5) > 0 ) {
      heta = -log(tan(htheta * 0.5));
    }
    double hphi = (*p)->momentum().phi();
    hphi = (hphi>=0) ? hphi : hphi+2*M_PI;
    hphi = hphi / M_PI * 180.;

    LogDebug("EventInfo") << "Particle gun type form MC = " << abs((*p)->pdg_id()) << "\n" << "Energy = "<< (*p)->momentum().e() << " Eta = " << heta << " Phi = " << hphi;

    if (meGunEnergy_) meGunEnergy_->Fill((*p)->momentum().e());
    if (meGunEta_)    meGunEta_   ->Fill(heta);
    if (meGunPhi_)    meGunPhi_   ->Fill(hphi);

  }
  
  double EBEnergy_ = 0.;
  if ( EcalHitsEB.isValid() ) {
    theEBCaloHits.insert(theEBCaloHits.end(), EcalHitsEB->begin(), EcalHitsEB->end());
    for (std::vector<PCaloHit>::iterator isim = theEBCaloHits.begin();
         isim != theEBCaloHits.end(); ++isim){
      EBEnergy_ += isim->energy();
    }
  }

  double EEEnergy_ = 0.;
  if ( EcalHitsEE.isValid() ) {
    theEECaloHits.insert(theEECaloHits.end(), EcalHitsEE->begin(), EcalHitsEE->end());
    for (std::vector<PCaloHit>::iterator isim = theEECaloHits.begin();
         isim != theEECaloHits.end(); ++isim){
      EEEnergy_ += isim->energy();
    }
  }
  
  double ESEnergy_ = 0.;
  if ( EcalHitsES.isValid() ) {
    theESCaloHits.insert(theESCaloHits.end(), EcalHitsES->begin(), EcalHitsES->end());
    for (std::vector<PCaloHit>::iterator isim = theESCaloHits.begin();
         isim != theESCaloHits.end(); ++isim){
      ESEnergy_ += isim->energy();
    }
  }
  
  double etot = EBEnergy_ + EEEnergy_ + ESEnergy_ ;
  double fracEB = 0.0;
  double fracEE = 0.0;
  double fracES = 0.0;
  
  if (etot>0.0) { 
    fracEB  = EBEnergy_/etot; 
    fracEE  = EEEnergy_/etot; 
    fracES  = ESEnergy_/etot; 
  }
  
  if (meEBEnergyFraction_) meEBEnergyFraction_->Fill(fracEB);
  
  if (meEEEnergyFraction_) meEEEnergyFraction_->Fill(fracEE);
  
  if (meESEnergyFraction_) meESEnergyFraction_->Fill(fracES);
}

DEFINE_FWK_MODULE(EcalSimHitsValidation);
