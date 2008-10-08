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


  em_incE       = new TH1F("em_incE","Incoming energy at Ecal;E (GeV);Number of Events",500,0.0,500.0);
  em_ssp_rho    = new TH1F("em_ssp_rho","Shower starting position;#rho (cm);Number of Events",100,100.0,200.0);
  em_ssp_z      = new TH1F("em_ssp_z","Shower starting position;z (cm);Number of Events",800,-400.0,400.0);
  em_long       = new TH1F("em_long","Longitudinal Profile;Radiation Length;Number of Spots",100,0.0,50.0);
  em_lateral    = new TH2F("em_lateral","Lateral Profile vs. Shower Depth;Radiation Length;Moliere Radius",100,0.0,50.0,100,0.0,3.0);
  em_long_sd    = new TH1F("em_long_sd","Longitudinal Profile in Sensitive Detector;Radiation Length;Number of Spots",100,0.0,50.0);
  em_lateral_sd = new TH2F("em_lateral_sd","Lateral Profile vs. Shower Depth in Sensitive Detector;Radiation Length;Moliere Radius",100,0.0,50.0,100,0.0,3.0);

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
}

void GflashG4Watcher::update(const EndOfEvent* g4Event){

  if(!inc_flag) return;

  const G4Event* evt = (*g4Event)();
  double primP = evt->GetPrimaryVertex(0)->GetPrimary(0)->GetMomentum().mag();
  double primM = evt->GetPrimaryVertex(0)->GetPrimary(0)->GetMass();
  double primE = std::sqrt(primP*primP + primM+primM);

  em_incE->Fill(inc_energy/GeV);

  if(inc_energy < 0.95*primE) return;

  em_ssp_rho->Fill(inc_position.rho()/cm);
  em_ssp_z->Fill(inc_position.z()/cm);

}

void GflashG4Watcher::update(const G4Step* aStep){

  if(aStep == NULL) return;

  if(inc_flag){
    if(aStep->GetTotalEnergyDeposit() > 1.0e-6){
      G4ThreeVector hitPosition = aStep->GetPreStepPoint()->GetPosition();
      double hitEnergy = aStep->GetTotalEnergyDeposit();

      G4ThreeVector diff = hitPosition - inc_position;
      double angle = diff.angle(inc_direction);
      double diff_z = std::abs(diff.mag() * std::cos(angle));
      double diff_r = std::abs(diff.mag() * std::sin(angle));

      em_long->Fill(diff_z/radLength,hitEnergy/GeV);
      em_lateral->Fill(diff_z/radLength,diff_r/rMoliere,hitEnergy/GeV);

      G4VSensitiveDetector* aSensitive = aStep->GetPreStepPoint()->GetSensitiveDetector();
      if(aSensitive){
	em_long_sd->Fill(diff_z/radLength,hitEnergy/GeV);
	em_lateral_sd->Fill(diff_z/radLength,diff_r/rMoliere,hitEnergy/GeV);
      } // if(aSensitive)

    } // if(aStep->GetTotalEnergyDeposit() > 1.0e-6)

  } // if(inc_flag)
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
  } // else

}


//define this as a plug-in
#include "SimG4Core/Watcher/interface/SimWatcherFactory.h"
#include "FWCore/PluginManager/interface/ModuleDef.h"

DEFINE_SEAL_MODULE ();
DEFINE_SIMWATCHER(GflashG4Watcher);

