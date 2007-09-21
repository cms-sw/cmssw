/*
 * \file EcalPreshowerSimHitsValidation.cc
 *
 * \author C.Rovelli
 *
*/

#include <DataFormats/EcalDetId/interface/EEDetId.h>
#include <DataFormats/EcalDetId/interface/ESDetId.h>
#include "FWCore/Utilities/interface/Exception.h"
#include "Validation/EcalHits/interface/EcalPreshowerSimHitsValidation.h"

using namespace cms;
using namespace edm;
using namespace std;

EcalPreshowerSimHitsValidation::EcalPreshowerSimHitsValidation(const edm::ParameterSet& ps):

  HepMCLabel(ps.getParameter<std::string>("moduleLabelMC")),
  g4InfoLabel(ps.getParameter<std::string>("moduleLabelG4")),
  EEHitsCollection(ps.getParameter<std::string>("EEHitsCollection")),
  ESHitsCollection(ps.getParameter<std::string>("ESHitsCollection")){

  // verbosity switch
  verbose_ = ps.getUntrackedParameter<bool>("verbose", false);
 
  if ( verbose_ ) {
    std::cout << " verbose switch is ON" << std::endl;
  } else {
    std::cout << " verbose switch is OFF" << std::endl;
  }

  // get hold of back-end interface
  dbe_ = 0;
  dbe_ = edm::Service<DaqMonitorBEInterface>().operator->();
  if ( dbe_ ) {
    if ( verbose_ ) { dbe_->setVerbose(1); } 
    else            { dbe_->setVerbose(0); }
  }

  if ( dbe_ ) {
    if ( verbose_ ) dbe_->showDirStructure();
  }


  menESHits1zp_ = 0;     
  menESHits2zp_ = 0;     
  menESHits1zm_ = 0;     
  menESHits2zm_ = 0;     
                                    
  meESEnergyHits1zp_ = 0;
  meESEnergyHits2zp_ = 0;
  meESEnergyHits1zm_ = 0;
  meESEnergyHits2zm_ = 0;

  meE1alphaE2zp_ = 0;
  meE1alphaE2zm_ = 0;
  meEEoverESzp_  = 0;
  meEEoverESzm_  = 0;

  me2eszpOver1eszp_ = 0;
  me2eszmOver1eszm_ = 0;


  Char_t histo[200];
 
  if ( dbe_ ) {
    dbe_->setCurrentFolder("EcalSimHitsValidation");
  
    sprintf (histo, "ES hits layer 1 multiplicity z+" ) ;
    menESHits1zp_ = dbe_->book1D(histo, histo, 50, 0., 50. ) ;

    sprintf (histo, "ES hits layer 2 multiplicity z+" ) ;
    menESHits2zp_ = dbe_->book1D(histo, histo, 50, 0., 50. ) ;

    sprintf (histo, "ES hits layer 1 multiplicity z-" ) ;
    menESHits1zm_ = dbe_->book1D(histo, histo, 50, 0., 50. ) ;

    sprintf (histo, "ES hits layer 2 multiplicity z-" ) ;
    menESHits2zm_ = dbe_->book1D(histo, histo, 50, 0., 50. ) ;

    sprintf (histo, "ES hits energy layer 1 z+" ) ;
    meESEnergyHits1zp_ = dbe_->book1D(histo, histo, 100, 0., 0.001 ) ;

    sprintf (histo, "ES hits energy layer 2 z+" ) ;
    meESEnergyHits2zp_ = dbe_->book1D(histo, histo, 100, 0., 0.001 ) ;

    sprintf (histo, "ES hits energy layer 1 z-" ) ;
    meESEnergyHits1zm_ = dbe_->book1D(histo, histo, 100, 0., 0.001 ) ;

    sprintf (histo, "ES hits energy layer 2 z-" ) ;
    meESEnergyHits2zm_ = dbe_->book1D(histo, histo, 100, 0., 0.001 ) ;

    sprintf (histo, "ES E1+0.7*E2 z+" ) ;
    meE1alphaE2zp_ = dbe_->book1D(histo, histo, 100, 0., 0.001);

    sprintf (histo, "ES E1+0.7*E2 z-" ) ;
    meE1alphaE2zm_ = dbe_->book1D(histo, histo, 100, 0., 0.001);

    sprintf (histo, "EE vs ES z+" ) ;
    meEEoverESzp_ = dbe_->bookProfile(histo, histo, 150, 0., 300., 80, 0., 80.);

    sprintf (histo, "EE vs ES z-" ) ;
    meEEoverESzm_ = dbe_->bookProfile(histo, histo, 150, 0., 300., 80, 0., 80.);

    sprintf (histo, "ES ene2oEne1 z+" ) ;
    me2eszpOver1eszp_ = dbe_->book1D(histo, histo, 50, 0., 10.);

    sprintf (histo, "ES ene2oEne1 z-" ) ;
    me2eszmOver1eszm_ = dbe_->book1D(histo, histo, 50, 0., 10.);
  }
}

EcalPreshowerSimHitsValidation::~EcalPreshowerSimHitsValidation(){

}

void EcalPreshowerSimHitsValidation::beginJob(const edm::EventSetup& c){

}

void EcalPreshowerSimHitsValidation::endJob(){

}

void EcalPreshowerSimHitsValidation::analyze(const edm::Event& e, const edm::EventSetup& c){

  edm::LogInfo("EventInfo") << " Run = " << e.id().run() << " Event = " << e.id().event();
  
  edm::Handle<edm::HepMCProduct> MCEvt;
  e.getByLabel(HepMCLabel, MCEvt);

  edm::Handle<edm::PCaloHitContainer> EcalHitsEE;
  e.getByLabel(g4InfoLabel,EEHitsCollection,EcalHitsEE);

  edm::Handle<edm::PCaloHitContainer> EcalHitsES;
  e.getByLabel(g4InfoLabel,ESHitsCollection,EcalHitsES);

  std::vector<PCaloHit> theEECaloHits;  
  theEECaloHits.insert(theEECaloHits.end(), EcalHitsEE->begin(), EcalHitsEE->end());

  std::vector<PCaloHit> theESCaloHits;
  theESCaloHits.insert(theESCaloHits.end(), EcalHitsES->begin(), EcalHitsES->end());

  double ESEnergy_ = 0.;
  std::map<unsigned int, std::vector<PCaloHit>,std::less<unsigned int> > CaloHitMap;


  // endcap
  double EEetzp_ = 0.;
  double EEetzm_ = 0.;
  for (std::vector<PCaloHit>::iterator isim = theEECaloHits.begin(); isim != theEECaloHits.end(); ++isim){
    EEDetId eeid (isim->id()) ;
    if (eeid.zside() > 0 ) EEetzp_ += isim->energy();
    if (eeid.zside() < 0 ) EEetzm_ += isim->energy();
  }
  
  
  uint32_t nESHits1zp = 0;
  uint32_t nESHits1zm = 0;
  uint32_t nESHits2zp = 0;
  uint32_t nESHits2zm = 0;
  double ESet1zp_ = 0.;
  double ESet2zp_ = 0.;
  double ESet1zm_ = 0.;
  double ESet2zm_ = 0.;
  
  for (std::vector<PCaloHit>::iterator isim = theESCaloHits.begin();
       isim != theESCaloHits.end(); ++isim){
    CaloHitMap[(*isim).id()].push_back((*isim));
    
    ESDetId esid (isim->id()) ;
    
    LogDebug("HitInfo")
      << " CaloHit " << isim->getName() << "\n" 
      << " DetID = "<<isim->id()<< " ESDetId: z side " << esid.zside() << "  plane " << esid.plane() << esid.six() << ',' << esid.siy() << ':' << esid.strip() << "\n"
      << " Time = " << isim->time() << "\n"
      << " Track Id = " << isim->geantTrackId() << "\n"
      << " Energy = " << isim->energy();
    
    ESEnergy_ += isim->energy();
    
    if (esid.plane() == 1 ) { 
      if (esid.zside() > 0 ) {
	nESHits1zp++ ;
	ESet1zp_ += isim->energy();
	if (meESEnergyHits1zp_) meESEnergyHits1zp_->Fill(isim->energy()) ; 
      }
      else if (esid.zside() < 0 ) {
	nESHits1zm++ ; 
	ESet1zm_ += isim->energy();
	if (meESEnergyHits1zm_) meESEnergyHits1zm_->Fill(isim->energy()) ; 
      }
    }
    else if (esid.plane() == 2 ) {
      if (esid.zside() > 0 ) {
	nESHits2zp++ ; 
	ESet2zp_ += isim->energy();
	if (meESEnergyHits2zp_) meESEnergyHits2zp_->Fill(isim->energy()) ; 
      }
      else if (esid.zside() < 0 ) {
	nESHits2zm++ ; 
	ESet2zm_ += isim->energy();
	if (meESEnergyHits2zm_) meESEnergyHits2zm_->Fill(isim->energy()) ; 
      }
    }
    
  }
  
  if (menESHits1zp_) menESHits1zp_->Fill(nESHits1zp);
  if (menESHits1zm_) menESHits1zm_->Fill(nESHits1zm);
  
  if (menESHits2zp_) menESHits2zp_->Fill(nESHits2zp);
  if (menESHits2zm_) menESHits2zm_->Fill(nESHits2zm);
  
  for ( HepMC::GenEvent::particle_const_iterator p = MCEvt->GetEvent()->particles_begin();
	p != MCEvt->GetEvent()->particles_end(); ++p ) {
    
    double htheta = (*p)->momentum().theta();
    double heta = -log(tan(htheta * 0.5));
    
    if ( heta > 1.653 && heta < 2.6 ) {

      if (meE1alphaE2zp_) meE1alphaE2zp_->Fill(ESet1zp_+0.7*ESet2zp_);
      if (meEEoverESzp_)  meEEoverESzp_ ->Fill((ESet1zp_+0.7*ESet2zp_)/0.00009, EEetzp_);
      if ((me2eszpOver1eszp_) && (ESet1zp_ != 0.)) me2eszpOver1eszp_->Fill(ESet2zp_/ESet1zp_);
    }
    if ( heta < -1.653 && heta > -2.6 ) {
      if (meE1alphaE2zm_) meE1alphaE2zm_->Fill(ESet1zm_+0.7*ESet2zm_);
      if (meEEoverESzm_)  meEEoverESzm_ ->Fill((ESet1zm_+0.7*ESet2zm_)/0.00009, EEetzm_);
      if ((me2eszmOver1eszm_) && (ESet1zm_ != 0.)) me2eszmOver1eszm_->Fill(ESet2zm_/ESet1zm_);
    }
  }
}

