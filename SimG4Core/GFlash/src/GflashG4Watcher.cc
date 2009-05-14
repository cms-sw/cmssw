#include "G4Step.hh"
#include "G4Electron.hh"
#include "G4Positron.hh"
#include "G4VProcess.hh"
#include "G4SDManager.hh"
#include "G4PrimaryParticle.hh"
#include "G4PrimaryVertex.hh"

#include "SimG4CMS/Calo/interface/CaloG4HitCollection.h"
#include "SimG4Core/Notification/interface/BeginOfEvent.h"
#include "SimG4Core/Notification/interface/EndOfEvent.h"

#include "SimG4Core/GFlash/interface/GflashG4Watcher.h"
#include "SimG4Core/GFlash/interface/GflashObjects.h"

#include <TVector2.h>

const double radLength = 8.9; // mm
const double rMoliere = 21.9; // mm
// constructors and destructor
//
GflashG4Watcher::GflashG4Watcher(const edm::ParameterSet& p) {

  edm::ParameterSet myP = p.getParameter<edm::ParameterSet>("GflashG4Watcher");
  histFileName_ = myP.getParameter<std::string>("histFileName");
  histFile_ = new TFile(histFileName_.c_str(),"RECREATE");

  TH1::AddDirectory(kTRUE);

  gflashObject_ = new GflashObject;
  watcherTree_ = new TTree("watcherTree","Watcher Tree Variable");
  watcherTree_->Branch("gflashObject","GflashObject",&gflashObject_,6400,99);
  watcherTree_->SetAutoSave();

  longitudinal_ = new TH1F("longitudinal","Logitudinal profile;X_{0};Energy",100,0.0,50.0);
  lateral_r_ = new TH1F("lateral_r","Lateral profile;r_{M};Number of hits",300,0.0,3.0);
  showerStartingPosition_ = new TH1F("showerStartingPosition","Shower starting position;r(cm);Number of hits",100,120.0,170.0);
  nHits_ = new TH1F("nHits","Number of hits;N_{hit};Events",30,4000.0,7000.0);
  hitEnergy_ = new TH1F("hitEnergy","Energy of hits;Energy (MeV);Number of hits",100,0.0,10.0);
  rzHits_ = new TH2F("rzHits","r vs. z of hits;z (X_{0});r_{M}",100,0.0,50.0,300,0.0,3.0);
  incEnergy_ = new TH1F("incEnergy","Incoming energy;energy (GeV);Events",100,0.0,100.0);
  outEnergy_ = new TH1F("outEnergy","Outgoing energy;energy (GeV);Events",100,0.0,100.0);

}


GflashG4Watcher::~GflashG4Watcher() {
  histFile_->cd();
  histFile_->Write();
  histFile_->Close();
}


void GflashG4Watcher::update(const BeginOfEvent* g4Event){
  inc_flag = false;
  inc_energy = 0;
  inc_direction *= 0;
  inc_position *= 0;
  gflashObject_->Init();
}

void GflashG4Watcher::update(const EndOfEvent* g4Event){

  if(!inc_flag) return;

  const G4Event* evt = (*g4Event)();
  double primP = evt->GetPrimaryVertex(0)->GetPrimary(0)->GetMomentum().mag();
  double primM = evt->GetPrimaryVertex(0)->GetPrimary(0)->GetMass();
  double primE = std::sqrt(primP*primP + primM+primM);

  incEnergy_->Fill(inc_energy/GeV);

  if(inc_energy < 0.95*primE) return;


  // Now fill GflashObject

  gflashObject_->energy = inc_energy;
  gflashObject_->direction.SetXYZ(inc_direction.x(),inc_direction.y(),inc_direction.z());
  gflashObject_->position.SetXYZ(inc_position.x(),inc_position.y(),inc_position.z());
  showerStartingPosition_->Fill(inc_position.rho()/cm);


  double outEnergy = 0.0;

  for(std::vector<GflashHit>::iterator it = gflashObject_->hits.begin();
      it != gflashObject_->hits.end(); it++){
    TVector3 diff = it->position - gflashObject_->position;
    double angle = diff.Angle(gflashObject_->direction);
    double diff_z = std::abs(diff.Mag()*std::cos(angle));
    double diff_r = std::abs(diff.Mag()*std::sin(angle));

    lateral_r_->Fill(diff_r/rMoliere,it->energy);
    rzHits_->Fill(diff_z/radLength,diff_r/rMoliere,it->energy);
    hitEnergy_->Fill(it->energy);
    longitudinal_->Fill(diff_z/radLength,it->energy);

    outEnergy += it->energy;
  }

  nHits_->Fill(gflashObject_->hits.size());
  outEnergy_->Fill(outEnergy/GeV);

  watcherTree_->Fill();

}

void GflashG4Watcher::update(const G4Step* aStep){

  if(aStep == NULL) return;

  if(inc_flag){
    if(aStep->GetTotalEnergyDeposit() > 1.0e-6){
      G4ThreeVector hitPosition = aStep->GetPreStepPoint()->GetPosition();
      GflashHit gHit;
      gHit.energy = aStep->GetTotalEnergyDeposit();
      gHit.position.SetXYZ(hitPosition.x(),hitPosition.y(),hitPosition.z());
      gflashObject_->hits.push_back(gHit);
    }
  }
  else {
    G4bool trigger = aStep->GetPreStepPoint()->GetKineticEnergy() > 1.0*GeV;
    trigger = trigger && (aStep->GetTrack()->GetDefinition() == G4Electron::ElectronDefinition() || 
			  aStep->GetTrack()->GetDefinition() == G4Positron::PositronDefinition());

    G4LogicalVolume* lv = aStep->GetPreStepPoint()->GetPhysicalVolume()->GetLogicalVolume();
    trigger = trigger && (lv->GetRegion()->GetName() == "GflashRegion");

    std::size_t pos1 = lv->GetName().find("EBRY");
    std::size_t pos2 = lv->GetName().find("EFRY");
    trigger = trigger && (pos1 != std::string::npos || pos2 != std::string::npos);

    if(trigger){
      inc_energy = aStep->GetPreStepPoint()->GetKineticEnergy();
      inc_direction = aStep->GetPreStepPoint()->GetMomentumDirection();
      inc_position = aStep->GetPreStepPoint()->GetPosition();
      inc_flag = true;
    }
  }

}


//define this as a plug-in
#include "SimG4Core/Watcher/interface/SimWatcherFactory.h"
#include "FWCore/PluginManager/interface/ModuleDef.h"

DEFINE_SEAL_MODULE ();
DEFINE_SIMWATCHER(GflashG4Watcher);

