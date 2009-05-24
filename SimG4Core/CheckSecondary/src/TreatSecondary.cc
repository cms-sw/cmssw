#include "SimG4Core/CheckSecondary/interface/TreatSecondary.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "G4Step.hh"
#include "G4Track.hh"
#include "G4VProcess.hh"
#include "G4HCofThisEvent.hh"
#include "CLHEP/Units/GlobalSystemOfUnits.h"
#include "CLHEP/Units/GlobalPhysicalConstants.h"

#include <cmath>
#include <iostream>
#include <iomanip>

TreatSecondary::TreatSecondary(const edm::ParameterSet &p): typeEnumerator(0) {

  verbosity     = p.getUntrackedParameter<int>("Verbosity",0);
  killAfter     = p.getUntrackedParameter<int>("KillAfter", -1);
  minDeltaE     = p.getUntrackedParameter<double>("MinimumDeltaE", 10.0)*MeV;
  
  edm::LogInfo("CheckSecondary") << "Instantiate CheckSecondary with Flag"
				 << " for Killing track after "<< killAfter 
				 << " hadronic interactions\nDefine inelastic"
				 << " if > 1 seondary or change in KE > "
				 << minDeltaE << " MeV\n";

  typeEnumerator = new G4ProcessTypeEnumerator();
} 
   
TreatSecondary::~TreatSecondary() {
  if (typeEnumerator) delete typeEnumerator;
}

void TreatSecondary::initTrack(const G4Track * thTk) {

  step   = 0;
  nsecL  = 0;
  nHad   = 0;
  eTrack = thTk->GetKineticEnergy()/MeV;
  LogDebug("CheckSecondary") << "TreatSecondary::initTrack:Track: " 
			     << thTk->GetTrackID() << " Type: " 
			     << thTk->GetDefinition()->GetParticleName() 
			     << " KE "    << thTk->GetKineticEnergy()/GeV 
			     << " GeV p " << thTk->GetMomentum().mag()/GeV
			     << " GeV daughter of particle " 
			     << thTk->GetParentID();
}

std::vector<math::XYZTLorentzVector> TreatSecondary::tracks(const G4Step*aStep,
							    std::string & name,
							    int & procid,
                                                            bool & hadrInt,
							    double & deltaE,
							    std::vector<int> & charges) {

  step++;
  procid  = -1;
  name    = "Unknown";
  hadrInt = false;
  deltaE  = 0;
  std::vector<math::XYZTLorentzVector> secondaries;
  charges.clear();

  if (aStep != NULL) {
    G4TrackVector* tkV  = const_cast<G4TrackVector*>(aStep->GetSecondary());
    G4Track*       thTk = aStep->GetTrack();
    const G4StepPoint* preStepPoint  = aStep->GetPreStepPoint();
    const G4StepPoint* postStepPoint = aStep->GetPostStepPoint();
    double eTrackNew = thTk->GetKineticEnergy()/MeV;
    deltaE = eTrack-eTrackNew;
    eTrack = eTrackNew;
    if (tkV != 0) {
      int nsc  = (*tkV).size();
      const G4VProcess*  proc = 0;
      if (postStepPoint) proc = postStepPoint->GetProcessDefinedStep();
      procid = typeEnumerator->processIdLong(proc);
      G4ProcessType type   = fNotDefined;
      if (proc) {
	type = proc->GetProcessType();
	name = proc->GetProcessName();
      }
      int           sec   = nsc - nsecL;
      LogDebug("CheckSecondary") << sec << " secondaries in step " 
				 << thTk->GetCurrentStepNumber() 
				 << " of track "  << thTk->GetTrackID() 
				 << " from " << name << " of type " 
				 << type << " ID " << procid << " (" 
				 << typeEnumerator->processG4Name(procid) 
				 << ")"; 

      G4TrackStatus state = thTk->GetTrackStatus();
      if (state == fAlive || state == fStopButAlive) sec++;

      if (type == fHadronic || type == fPhotolepton_hadron || type == fDecay) {
	if (deltaE > minDeltaE || sec > 1) {
	  hadrInt = true;
	  nHad++;
	}
	LogDebug("CheckSecondary") << "Hadronic Interaction " << nHad
				   << " of Type " << type << " with "
				   << sec << " secondaries from process "
				   << proc->GetProcessName() << " Delta E "
				   << deltaE << " Flag " << hadrInt;
      }
      if (hadrInt) {
	math::XYZTLorentzVector secondary;
	if (state == fAlive || state == fStopButAlive) {
	  G4ThreeVector pp    = postStepPoint->GetMomentum();
	  double        ee    = postStepPoint->GetTotalEnergy();
	  secondary = math::XYZTLorentzVector(pp.x(),pp.y(),pp.z(),ee);
	  secondaries.push_back(secondary);
	  int           charge = (int)(postStepPoint->GetCharge());
	  charges.push_back(charge);
	}
	for (int i=nsecL; i<nsc; i++) {
	  G4Track*      tk = (*tkV)[i];
	  G4ThreeVector pp = tk->GetMomentum();
	  double        ee = tk->GetTotalEnergy();
	  secondary = math::XYZTLorentzVector(pp.x(),pp.y(),pp.z(),ee);
	  secondaries.push_back(secondary);
	  int           charge = (int)(tk->GetDefinition()->GetPDGCharge());
	  charges.push_back(charge);
	}
      }
      
      if (killAfter >= 0 && nHad >= killAfter)
	thTk->SetTrackStatus(fStopAndKill);

      if (verbosity > 0) {
	sec = 0;
	if (state == fAlive || state == fStopButAlive) {
	  sec++;
	  LogDebug("CheckSecondary") << "Secondary: " << sec << " ID " 
				     << thTk->GetTrackID() << " Status " 
				     << thTk->GetTrackStatus() << " Particle " 
				     <<thTk->GetDefinition()->GetParticleName()
				     << " Position "
				     << postStepPoint->GetPosition() << " KE " 
				     << postStepPoint->GetKineticEnergy() 
				     << " Time " 
				     << postStepPoint->GetGlobalTime();
	}
	for (int i=nsecL; i<nsc; i++) {
	  sec++;
	  G4Track* tk = (*tkV)[i];
	  LogDebug("CheckSecondary") << "Secondary: " << sec << " ID " 
				     << tk->GetTrackID() << " Status " 
				     << tk->GetTrackStatus() << " Particle " 
				     << tk->GetDefinition()->GetParticleName()
				     << " Position "  << tk->GetPosition()
				     << " KE " << tk->GetKineticEnergy() 
				     << " Time " << tk->GetGlobalTime();
	}
      }
      nsecL  = nsc;
    }

    if (verbosity > 1)
      LogDebug("CheckSecondary") << "Track: " << thTk->GetTrackID() 
				 << " Status " << thTk->GetTrackStatus()
				 << " Particle " 
				 << thTk->GetDefinition()->GetParticleName()
				 << " at " << preStepPoint->GetPosition()
				 << " Step: " << step << " KE " 
				 << thTk->GetKineticEnergy()/GeV << " GeV; p " 
				 << thTk->GetMomentum().mag()/GeV 
				 << " GeV/c; Step Length "
				 << aStep->GetStepLength()<< " Energy Deposit "
				 << aStep->GetTotalEnergyDeposit()/MeV
				 << " MeV; Interaction " << hadrInt;
  }
  return secondaries;
}
