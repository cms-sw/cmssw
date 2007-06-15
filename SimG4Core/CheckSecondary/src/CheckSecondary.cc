#include "SimG4Core/CheckSecondary/interface/CheckSecondary.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "SimG4Core/Notification/interface/BeginOfEvent.h"
#include "SimG4Core/Notification/interface/EndOfEvent.h"
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

CheckSecondary::CheckSecondary(const edm::ParameterSet &p): typeEnumerator(0),
							    nsec(0),procids(0),
							    px(0),py(0),pz(0),
							    mass(0),procs(0),
							    file(0),tree(0) {

  edm::ParameterSet m_p = p.getParameter<edm::ParameterSet>("CheckSecondary");
  verbosity         = m_p.getUntrackedParameter<int>("Verbosity",0);
  killAfter         = m_p.getUntrackedParameter<int>("KillAfter", -1);
  suppressHeavy     = m_p.getUntrackedParameter<bool>("SuppressHeavy", false);
  pmaxIon           = m_p.getUntrackedParameter<double>("IonThreshold", 50.0)*MeV;
  pmaxProton        = m_p.getUntrackedParameter<double>("ProtonThreshold", 50.0)*MeV;
  pmaxNeutron       = m_p.getUntrackedParameter<double>("NeutronThreshold", 50.0)*MeV;
  std::string saveFile = m_p.getUntrackedParameter<std::string>("SaveInFile", "None");

  nsec               = new std::vector<int>();
  px                 = new std::vector<double>();
  py                 = new std::vector<double>();
  pz                 = new std::vector<double>();
  mass               = new std::vector<double>();
  procids            = new std::vector<int>();
  procs              = new std::vector<std::string>();

  edm::LogInfo("CheckSecondary") << "Instantiate CheckSecondary with Flag "
				 << "for Killing track after "<< killAfter 
				 << " hadronic interactions\nSuppression Flag "
				 << suppressHeavy << " protons below " 
				 << pmaxProton << " MeV/c, neutrons below "
				 << pmaxNeutron << " and ions below " 
				 << pmaxIon << " MeV/c\n";

  typeEnumerator = new G4ProcessTypeEnumerator();
  if (saveFile != "None") {
    saveToTree = true;
    tree = bookTree (saveFile);
    edm::LogInfo("CheckSecondary") << "First hadronic interaction information"
				   << " to be saved in file " << saveFile;
  } else {
    saveToTree = false;
    edm::LogInfo("CheckSecondary") << "First hadronic interaction information"
				   << " not saved";
  }
  count = 0;
} 
   
CheckSecondary::~CheckSecondary() {
  if (saveToTree)     endTree();
  if (typeEnumerator) delete typeEnumerator;
  if (nsec)           delete nsec;
  if (px)             delete px;
  if (py)             delete py;
  if (pz)             delete pz;
  if (mass)           delete mass;
  if (procs)          delete procs;
  if (procids)        delete procids;
}

TTree* CheckSecondary::bookTree(std::string fileName) {

  file = new TFile (fileName.c_str(), "RECREATE");
  file->cd();

  TTree * t1 = new TTree("T1", "Secondary Particle Information");
  t1->Branch("SecondaryPx",       "std::vector<double>",      &px);
  t1->Branch("SecondaryPy",       "std::vector<double>",      &py);
  t1->Branch("SecondaryPz",       "std::vector<double>",      &pz);
  t1->Branch("SecondaryMass",     "std::vector<double>",      &mass);
  t1->Branch("NumberSecondaries", "std::vector<int>",         &nsec);
  t1->Branch("ProcessID",         "std::vector<int>",         &procids);
  t1->Branch("ProcessNames",      "std::vector<std::string>", &procs);
  return t1;
}

void CheckSecondary::endTree() {

  edm::LogInfo("CheckSecondary") << "Save the Secondary Tree " 
				 << tree->GetName() << " (" << tree
				 << ") in file " << file->GetName() << " ("
				 << file << ")";
  file->cd();
  tree->Write();
  file->Close();
  delete file;
}

void CheckSecondary::update(const BeginOfEvent * evt) {

  int iev = (*evt)()->GetEventID();
  if (verbosity > 0)
    std::cout << "CheckSecondary::=====> Begin of event = " << iev << "\n";

  count++;
  track = 0;
  (*nsec).clear();
  (*procs).clear();
  (*procids).clear();
  (*px).clear();
  (*py).clear();
  (*pz).clear();
  (*mass).clear();
}

void CheckSecondary::update(const BeginOfTrack * trk) {

  track++;
  const G4Track * thTk = (*trk)();
  if (verbosity > 0)
    std::cout << "Track: " << track << "  " << thTk->GetTrackID() 
	      << " Type: " << thTk->GetDefinition()->GetParticleName() 
	      << " KE "    << thTk->GetKineticEnergy()/GeV 
	      << " GeV p " << thTk->GetMomentum().mag()/GeV
	      << " GeV daughter of particle " << thTk->GetParentID() << "\n";
  step  = 0;
  nsecL = 0;
  nHad  = 0;
}

void CheckSecondary::update(const G4Step * aStep) {

  step++;
  bool intr = false;
  if (aStep != NULL) {
    G4TrackVector* tkV  = const_cast<G4TrackVector*>(aStep->GetSecondary());
    G4Track*       thTk = aStep->GetTrack();
    const G4StepPoint* preStepPoint  = aStep->GetPreStepPoint();
    const G4StepPoint* postStepPoint = aStep->GetPostStepPoint();
    if (tkV != 0) {
      int nsc  = (*tkV).size();
      const G4VProcess*  proc = 0;
      if (postStepPoint) proc = postStepPoint->GetProcessDefinedStep();
      int           procid = typeEnumerator->processIdLong(proc);
      G4ProcessType type   = fNotDefined;
      std::string   name   = "Unknown";
      if (proc) {
	type = proc->GetProcessType();
	name = proc->GetProcessName();
      }
      int           sec   = nsc - nsecL;
      if (verbosity > 0) {
	std::cout << sec << " secondaries in step " 
		  << thTk->GetCurrentStepNumber() << " of track " 
		  << thTk->GetTrackID() << " from " << name << " of type " 
		  << type << " ID " << procid << " (" 
		  << typeEnumerator->processG4Name(procid) << ")\n"; 
      }

      G4TrackStatus state = thTk->GetTrackStatus();
      if (state == fAlive || state == fStopButAlive) sec++;

      if (type == fHadronic || type == fPhotolepton_hadron || type == fDecay) {
	nHad++;
	if (verbosity > 0) std::cout << "Hadronic Interaction " << nHad
				     << " of Type " << type << " with "
				     << sec << " secondaries from process "
				     << proc->GetProcessName() << "\n";
	(*nsec).push_back(sec);
	(*procs).push_back(name);
	(*procids).push_back(procid);
	if (nHad == 1) {
	  if (state == fAlive || state == fStopButAlive) {
	    G4ThreeVector pp    = postStepPoint->GetMomentum();
	    double        ee    = postStepPoint->GetTotalEnergy();
	    double        msq   = ee*ee-pp.mag2();
	    double        m     = (msq <= 0. ? 0. : sqrt(msq));
	    (*px).push_back(pp.x());
	    (*py).push_back(pp.y());
	    (*pz).push_back(pp.z());
	    (*mass).push_back(m);
	  }
	  for (int i=nsecL; i<nsc; i++) {
	    G4Track*      tk = (*tkV)[i];
	    G4ThreeVector pp = tk->GetMomentum();
	    double        ee = tk->GetTotalEnergy();
	    double        msq   = ee*ee-pp.mag2();
	    double        m     = (msq <= 0. ? 0. : sqrt(msq));
	    (*px).push_back(pp.x());
	    (*py).push_back(pp.y());
	    (*pz).push_back(pp.z());
	    (*mass).push_back(m);
	  }
	}
      }

      for (int i=nsecL; i<nsc; i++) {
	G4Track* tk = (*tkV)[i];
	bool   ok  = true;
	if (suppressHeavy) {
	  double pp  = tk->GetMomentum().mag()/MeV;
	  int    pdg = tk->GetDefinition()->GetPDGEncoding();
	  if (((pdg/1000000000 == 1 && ((pdg/10000)%100) > 0 &&
		((pdg/10)%100) > 0)) && (pp<pmaxIon)) ok = false;
	  if ((pdg == 2212) && (pp < pmaxProton))     ok = false;
	  if ((pdg == 2112) && (pp < pmaxNeutron))    ok = false;
	}
	if ((killAfter >= 0 && nHad >= killAfter) || (!ok)) {
	  tk->SetTrackStatus(fStopAndKill);
	  if (verbosity > 0)
	    std::cout << "CheckSecondary::Kill Secondary " << i << " ID "
		      << tk->GetDefinition() << " p "
		      << tk->GetMomentum().mag()/MeV << " MeV/c\n";
	}
	if (verbosity > 0)
	  std::cout << "Secondary: " << i << " ID " << tk->GetTrackID()
		    << " Status " << tk->GetTrackStatus() << " Particle " 
		    << tk->GetDefinition()->GetParticleName() << " Position "
		    << tk->GetPosition() << " KE " << tk->GetKineticEnergy() 
		    << " Time " << tk->GetGlobalTime() << "\n";
      }

      if (nsc > nsecL || type == fHadronic || type == fPhotolepton_hadron || 
	  type == fDecay) intr = true;

      if (killAfter >= 0 && nHad >= killAfter)
	thTk->SetTrackStatus(fStopAndKill);
      nsecL = nsc;
    }

    if (verbosity > 1)
      std::cout << "Track: " << thTk->GetTrackID() << " Status "
		<< thTk->GetTrackStatus() << " Particle " 
		<< thTk->GetDefinition()->GetParticleName()<< " at "
		<< preStepPoint->GetPosition() << " Step: " << step
		<< " KE " << thTk->GetKineticEnergy()/GeV << " GeV; p " 
		<< thTk->GetMomentum().mag()/GeV << " GeV/c; Step Length "
		<< aStep->GetStepLength() << " Energy Deposit "
		<< aStep->GetTotalEnergyDeposit()/MeV << " MeV; Interaction "
		<< intr << " Pointer " << tkV << "\n";
  }
}

void CheckSecondary::update(const EndOfEvent * evt) {

  count++;
  if (verbosity > 0) {
    std::cout << "=====> Event " << (*evt)()->GetEventID() << " with "
	      << (*nsec).size() << " hadronic collisions with secondaries"
	      << " produced in each step\n";
    for (unsigned int i= 0; i < (*nsec).size(); i++) 
      std::cout << " " << (*nsec)[i] << " from " << (*procs)[i] << " ID "
		<< (*procids)[i] << " (" 
		<< typeEnumerator->processG4Name((*procids)[i]) << ")\n";
    std::cout << "And " << (*mass).size() << " secondaries produced "
	      << "in the first interactions: \n";
    for (unsigned int i= 0; i < (*mass).size(); i++) 
      std::cout << "Secondary " << i << " (" << (*px)[i] << ", " << (*py)[i]
		<< ", " << (*pz)[i] << ", " << (*mass)[i] << ")\n";
  }
  if (saveToTree) {
    tree->Fill();
  }
}
