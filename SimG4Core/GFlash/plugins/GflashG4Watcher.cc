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

using namespace CLHEP;

// constructors and destructor
//
GflashG4Watcher::GflashG4Watcher(const edm::ParameterSet& p) {

  edm::ParameterSet myP = p.getParameter<edm::ParameterSet>("GflashG4Watcher");
  histFileName_ = myP.getParameter<std::string>("histFileName");
  recoEnergyScaleEB_ = myP.getParameter<double>("recoEnergyScaleEB");
  recoEnergyScaleEE_ = myP.getParameter<double>("recoEnergyScaleEE");


  histFile_ = new TFile(histFileName_.c_str(),"RECREATE");

  TH1::AddDirectory(kTRUE);


  em_incE        = new TH1F("em_incE","Incoming energy at Ecal;E (GeV);Number of Events",500,0.0,500.0);
  em_vtx_rho     = new TH1F("em_vtx_rho","vertex position;#rho (cm);Number of Events",100,0.0,10.0);
  em_vtx_z       = new TH1F("em_vtx_z","vertex position;z (cm);Number of Events",100,-10.0,10.0);

  eb_ssp_rho     = new TH1F("eb_ssp_rho","Shower starting position;#rho (cm);Number of Events",200,0.0,200.0);
  eb_hit_long    = new TH1F("eb_hit_long","longitudinal hit position;shower depth (cm);Number of energy weighted hits",400,0.0,200.0);
  eb_hit_lat     = new TH1F("eb_hit_lat","lateral hit position;arm (cm);Number of energy weighted hits",100,0.0,5.0);
  eb_hit_rz      = new TH2F("eb_hit_rz","hit position along the shower direction;shower depth (cm);arm (cm)",400,0.0,200.0,100,0.0,5.0);
  eb_hit_long_sd = new TH1F("eb_hit_long_sd","longitudinal hit position in Sensitive Detector;shower depth (cm);Number of energy weighted hits",400,0.0,200.0);
  eb_hit_lat_sd  = new TH1F("eb_hit_lat_sd","lateral hit position in Sensitive Detector;arm (cm);Number of energy weighted hits",100,0.0,5.0);
  eb_hit_rz_sd   = new TH2F("eb_hit_rz_sd","hit position along the shower direction in Sensitive Detector;shower depth (cm);arm (cm)",400,0.0,200.0,100,0.0,5.0);

  ee_ssp_z       = new TH1F("ee_ssp_z","Shower starting position;z (cm);Number of Events",800,-400.0,400.0);
  ee_hit_long    = new TH1F("ee_hit_long","longitudinal hit position;shower depth (cm);Number of energy weighted hits",800,0.0,400.0);
  ee_hit_lat     = new TH1F("ee_hit_lat","lateral hit position;arm (cm);Number of energy weighted hits",100,0.0,5.0);
  ee_hit_rz      = new TH2F("ee_hit_rz","hit position along the shower direction;shower depth (cm);arm (cm)",800,0.0,400.0,100,0.0,5.0);
  ee_hit_long_sd = new TH1F("ee_hit_long_sd","longitudinal hit position in Sensitive Detector;shower depth (cm);Number of energy weighted hits",800,0.0,400.0);
  ee_hit_lat_sd  = new TH1F("ee_hit_lat_sd","lateral hit position in Sensitive Detector;arm (cm);Number of energy weighted hits",100,0.0,5.0);
  ee_hit_rz_sd   = new TH2F("ee_hit_rz_sd","hit position along the shower direction in Sensitive Detector;shower depth (cm);arm (cm)",800,0.0,400.0,100,0.0,5.0);

}


GflashG4Watcher::~GflashG4Watcher() {
  histFile_->cd();
  histFile_->Write();
  histFile_->Close();
}


void GflashG4Watcher::update(const BeginOfEvent* g4Event){

  inc_flag = false;

  const G4Event* evt = (*g4Event)();
  inc_vertex = evt->GetPrimaryVertex(0)->GetPosition();
  inc_position = inc_vertex;
  inc_direction = evt->GetPrimaryVertex(0)->GetPrimary(0)->GetMomentum().unit();
  inc_energy = evt->GetPrimaryVertex(0)->GetPrimary(0)->GetMomentum().mag();
  out_energy = 0;

  em_incE->Fill(inc_energy/GeV);
  em_vtx_rho->Fill(inc_vertex.rho()/cm);
  em_vtx_z->Fill(inc_vertex.z()/cm);

  if(std::abs(inc_direction.eta()) < 1.5) eb_ssp_rho->Fill(inc_position.rho()/cm);
  else ee_ssp_z->Fill(inc_position.z()/cm);

}


void GflashG4Watcher::update(const EndOfEvent* g4Event){ }


void GflashG4Watcher::update(const G4Step* aStep){

  if(aStep == NULL) return;

  double hitEnergy = aStep->GetTotalEnergyDeposit();

  if(hitEnergy < 1.0e-6) return;

  bool inEB = std::abs(inc_direction.eta()) < 1.5;

  out_energy += hitEnergy; // to check outgoing energy

  // This is to calculate shower depth and arm of hits from the shower direction
  G4ThreeVector hitPosition = aStep->GetPreStepPoint()->GetPosition();
  G4ThreeVector diff = hitPosition - inc_position;
  double angle = diff.angle(inc_direction);
  double diff_z = std::abs(diff.mag() * std::cos(angle));
  double diff_r = std::abs(diff.mag() * std::sin(angle));

  G4VSensitiveDetector* aSensitive = aStep->GetPreStepPoint()->GetSensitiveDetector();

  if(inEB){ // showers in barrel crystals
    hitEnergy *= recoEnergyScaleEB_;
    eb_hit_long->Fill(diff_z/cm,hitEnergy/GeV);
    eb_hit_lat->Fill(diff_r/cm,hitEnergy/GeV);
    eb_hit_rz->Fill(diff_z/cm,diff_r/cm,hitEnergy/GeV);
    if(aSensitive){
      eb_hit_long_sd->Fill(diff_z/cm,hitEnergy/GeV);
      eb_hit_lat_sd->Fill(diff_r/cm,hitEnergy/GeV);
      eb_hit_rz_sd->Fill(diff_z/cm,diff_r/cm,hitEnergy/GeV);
    }
  }
  else{ // showers in endcap crystals
    hitEnergy *= recoEnergyScaleEE_;
    ee_hit_long->Fill(diff_z/cm,hitEnergy/GeV);
    ee_hit_lat->Fill(diff_r/cm,hitEnergy/GeV);
    ee_hit_rz->Fill(diff_z/cm,diff_r/cm,hitEnergy/GeV);
    if(aSensitive){
      ee_hit_long_sd->Fill(diff_z/cm,hitEnergy/GeV);
      ee_hit_lat_sd->Fill(diff_r/cm,hitEnergy/GeV);
      ee_hit_rz_sd->Fill(diff_z/cm,diff_r/cm,hitEnergy/GeV);
    }
  }


}


//define this as a plug-in
#include "SimG4Core/Watcher/interface/SimWatcherFactory.h"
#include "FWCore/PluginManager/interface/ModuleDef.h"


DEFINE_SIMWATCHER(GflashG4Watcher);

