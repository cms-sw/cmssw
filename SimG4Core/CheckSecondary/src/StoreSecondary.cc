#include "SimG4Core/CheckSecondary/interface/StoreSecondary.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "SimG4Core/Notification/interface/BeginOfEvent.h"
#include "SimG4Core/Notification/interface/BeginOfTrack.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "G4Step.hh"
#include "G4Track.hh"
#include "G4VProcess.hh"
#include "G4HCofThisEvent.hh"
#include "CLHEP/Units/SystemOfUnits.h"
#include "CLHEP/Units/PhysicalConstants.h"

#include <cmath>
#include <iostream>
#include <iomanip>

StoreSecondary::StoreSecondary(const edm::ParameterSet &p) {

  edm::ParameterSet m_p = p.getParameter<edm::ParameterSet>("StoreSecondary");
  verbosity         = m_p.getUntrackedParameter<int>("Verbosity",  0);
  killAfter         = m_p.getUntrackedParameter<int>("KillAfter", -1);

  produces<std::vector<math::XYZTLorentzVector> >("SecondaryMomenta");
  produces<std::vector<int> >("SecondaryParticles");
  produces<std::vector<std::string> >("SecondaryProcesses");

  edm::LogInfo("StoreSecondary") << "Instantiate StoreSecondary with Flag "
				 << "for Killing track after "<< killAfter;
} 
   
StoreSecondary::~StoreSecondary() {
}

void StoreSecondary::produce(edm::Event& e, const edm::EventSetup&) {

  std::auto_ptr<std::vector<math::XYZTLorentzVector> > secMom(new std::vector<math::XYZTLorentzVector>);
  *secMom = secondaries;
  e.put(secMom, "SecondaryMomenta");

  std::auto_ptr<std::vector<int> > secNumber(new std::vector<int>);
  *secNumber = nsecs;
  e.put(secNumber, "SecondaryParticles");

  std::auto_ptr<std::vector<std::string> > secProc(new std::vector<std::string>);
  *secProc = procs;
  e.put(secProc, "SecondaryProcesses");

  if (verbosity > 0) {
    std::cout << "StoreSecondary:: Event " << e.id() << " with "
	      << nsecs.size() << " hadronic collisions with secondaries"
	      << " produced in each step\n";
    for (unsigned int i= 0; i < nsecs.size(); i++) 
      std::cout << " " << nsecs[i] << " from " << procs[i];
    std::cout << "\n and " << secondaries.size() << " secondaries produced "
	      << "in the first interactions: \n";
    for (unsigned int i= 0; i < secondaries.size(); i++) 
      std::cout << "Secondary " << i << " " << secondaries[i] << "\n";
  }
}

void StoreSecondary::update(const BeginOfEvent *) {

  track = 0;
  nsecs.clear();
  procs.clear();
  secondaries.clear();
}

void StoreSecondary::update(const BeginOfTrack * trk) {

  track++;
  const G4Track * thTk = (*trk)();
  if (nsecs.size() == 0 && thTk->GetParentID() <= 0) storeIt = true;
  else                                               storeIt = false;
  if (verbosity > 0)
    std::cout << "Track: " << track << "  " << thTk->GetTrackID() 
	      << " Type: " << thTk->GetDefinition()->GetParticleName() 
	      << " KE "    << thTk->GetKineticEnergy()/GeV 
	      << " GeV p " << thTk->GetMomentum().mag()/GeV
	      << " GeV daughter of particle " << thTk->GetParentID() 
	      << " Store " << storeIt << "\n";
  nsecL = 0;
  nHad  = 0;
}

void StoreSecondary::update(const G4Step * aStep) {

  if (aStep != NULL) {
    G4TrackVector* tkV  = const_cast<G4TrackVector*>(aStep->GetSecondary());
    G4Track*       thTk = aStep->GetTrack();
    const G4StepPoint* preStepPoint  = aStep->GetPreStepPoint();
    const G4StepPoint* postStepPoint = aStep->GetPostStepPoint();
    if (tkV != 0) {
      int nsec = (*tkV).size();
      const G4VProcess*  proc = 0;
      if (postStepPoint) proc = postStepPoint->GetProcessDefinedStep();
      G4ProcessType type  = fNotDefined;
      std::string   name  = "Unknown";
      if (proc) {
	type = proc->GetProcessType();
	name = proc->GetProcessName();
      }
      int           sec   = nsec - nsecL;
      if (verbosity > 0) {
	std::cout << sec << " secondaries in step " 
		  << thTk->GetCurrentStepNumber() << " of track " 
		  << thTk->GetTrackID() << " from " << name << " of type " 
		  << type << "\n"; 
      }

      G4TrackStatus state = thTk->GetTrackStatus();
      if (state == fAlive || state == fStopButAlive) sec++;

      if (type == fHadronic || type == fPhotolepton_hadron || type == fDecay) {
	nHad++;
	if (verbosity > 0) std::cout << "Hadronic Interaction " << nHad
				     << " of Type " << type << " with "
				     << sec << " secondaries from process "
				     << proc->GetProcessName() << "\n";
	if (storeIt) {
	  nsecs.push_back(sec);
	  procs.push_back(name);
	  if (nHad == 1) {
	    math::XYZTLorentzVector secondary;
	    if (state == fAlive || state == fStopButAlive) {
	      G4ThreeVector pp = postStepPoint->GetMomentum();
	      double        ee = postStepPoint->GetTotalEnergy();
	      secondary = math::XYZTLorentzVector(pp.x(),pp.y(),pp.z(),ee);
	      secondaries.push_back(secondary);
	    }
	    for (int i=nsecL; i<nsec; i++) {
	      G4Track*      tk = (*tkV)[i];
	      G4ThreeVector pp = tk->GetMomentum();
	      double        ee = tk->GetTotalEnergy();
	      secondary = math::XYZTLorentzVector(pp.x(),pp.y(),pp.z(),ee);
	      secondaries.push_back(secondary);
	    }
	  }
	}
      }

      if (killAfter >= 0 && nHad >= killAfter) {
	for (int i=nsecL; i<nsec; i++) {
	  G4Track* tk = (*tkV)[i];
	  tk->SetTrackStatus(fStopAndKill);
	  std::cout << "StoreSecondary::Kill Secondary " << i << " ID "
		    << tk->GetDefinition() << " p "
		    << tk->GetMomentum().mag()/MeV << " MeV/c\n";
	}
	thTk->SetTrackStatus(fStopAndKill);
      }
      nsecL = nsec;
    }

    if (verbosity > 1)
      std::cout << "Track: " << thTk->GetTrackID() << " Status "
		<< thTk->GetTrackStatus() << " Particle " 
		<< thTk->GetDefinition()->GetParticleName()<< " at "
		<< preStepPoint->GetPosition() << " Step Number "
		<< thTk->GetCurrentStepNumber() << " KE " 
		<< thTk->GetKineticEnergy()/GeV << " GeV; Momentum " 
		<< thTk->GetMomentum().mag()/GeV << " GeV/c; Step Length "
		<< aStep->GetStepLength() << " Energy Deposit "
		<< aStep->GetTotalEnergyDeposit()/MeV << " MeV\n";
  }
}
