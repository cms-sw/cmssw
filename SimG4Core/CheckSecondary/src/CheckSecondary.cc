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
#include "CLHEP/Units/GlobalSystemOfUnits.h"
#include "CLHEP/Units/GlobalPhysicalConstants.h"

#include <cmath>
#include <iostream>
#include <iomanip>

CheckSecondary::CheckSecondary(const edm::ParameterSet &p): treatSecondary(0),
							    typeEnumerator(0),
							    nsec(0),procids(0),
							    px(0),py(0),pz(0),
							    mass(0),deltae(0),
							    procs(0),file(0),
							    tree(0) {

  edm::ParameterSet m_p = p.getParameter<edm::ParameterSet>("CheckSecondary");
  std::string saveFile = m_p.getUntrackedParameter<std::string>("SaveInFile", "None");
  treatSecondary = new TreatSecondary(m_p);
  typeEnumerator = new G4ProcessTypeEnumerator();

  nsec               = new std::vector<int>();
  px                 = new std::vector<double>();
  py                 = new std::vector<double>();
  pz                 = new std::vector<double>();
  mass               = new std::vector<double>();
  deltae             = new std::vector<double>();
  procids            = new std::vector<int>();
  procs              = new std::vector<std::string>();

  if (saveFile != "None") {
    saveToTree = true;
    tree = bookTree (saveFile);
    edm::LogInfo("CheckSecondary") << "Instantiate CheckSecondary with first"
				   << " hadronic interaction information"
				   << " to be saved in file " << saveFile;
  } else {
    saveToTree = false;
    edm::LogInfo("CheckSecondary") << "Instantiate CheckSecondary with first"
				   << " hadronic interaction information"
				   << " not saved";
  }
} 
   
CheckSecondary::~CheckSecondary() {
  if (saveToTree)     endTree();
  if (nsec)           delete nsec;
  if (px)             delete px;
  if (py)             delete py;
  if (pz)             delete pz;
  if (mass)           delete mass;
  if (deltae)         delete deltae;
  if (procs)          delete procs;
  if (procids)        delete procids;
  if (typeEnumerator) delete typeEnumerator;
  if (treatSecondary) delete treatSecondary;
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
  t1->Branch("DeltaEinInteract",  "std::vector<double>",      &deltae);
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
  LogDebug("CheckSecondary") << "CheckSecondary::=====> Begin of event = " 
			     << iev;

  (*nsec).clear();
  (*procs).clear();
  (*procids).clear();
  (*deltae).clear();
  (*px).clear();
  (*py).clear();
  (*pz).clear();
  (*mass).clear();
}

void CheckSecondary::update(const BeginOfTrack * trk) {

  const G4Track * thTk = (*trk)();
  treatSecondary->initTrack(thTk);
  if (thTk->GetParentID() <= 0) storeIt = true;
  else                          storeIt = false;
  nHad  = 0;
  LogDebug("CheckSecondary") << "CheckSecondary:: Track " << thTk->GetTrackID()
			     << " Parent " << thTk->GetParentID() << " Flag "
			     << storeIt;
}

void CheckSecondary::update(const G4Step * aStep) {

  std::string      name;
  int              procID;
  bool             hadrInt;
  double           deltaE;
  std::vector<int> charge;
  std::vector<math::XYZTLorentzVector> tracks = treatSecondary->tracks(aStep,
								       name,
								       procID,
								       hadrInt,
								       deltaE,
								       charge);
  if (storeIt && hadrInt) {
    double pInit = (aStep->GetPreStepPoint()->GetMomentum()).mag();
    double pEnd  = (aStep->GetPostStepPoint()->GetMomentum()).mag();
    nHad++;
    int  sec = (int)(tracks.size());
    LogDebug("CheckSecondary") << "CheckSecondary:: Hadronic Interaction "
			       << nHad << " of type " << name << " ID "
			       << procID << " with " << sec << " secondaries "
			       << " and Momentum (MeV/c) at start " << pInit
			       << " and at end " << pEnd;
    (*nsec).push_back(sec);
    (*procs).push_back(name);
    (*procids).push_back(procID);
    (*deltae).push_back(deltaE);
    if (nHad == 1) {
      for (int i=0; i<sec; i++) {
	double m = tracks[i].M();
	if (charge[i]<0) m = -m;
	(*px).push_back(tracks[i].Px());
	(*py).push_back(tracks[i].Py());
	(*pz).push_back(tracks[i].Pz());
	(*mass).push_back(m);
      }
    }
  }
}

void CheckSecondary::update(const EndOfEvent * evt) {

  LogDebug("CheckSecondary") << "CheckSecondary::EndofEvent =====> Event " 
			     << (*evt)()->GetEventID() << " with "
			     << (*nsec).size() << " hadronic collisions with"
			     << " secondaries produced in each step";
  for (unsigned int i= 0; i < (*nsec).size(); i++) 
    LogDebug("CheckSecondary") << " " << (*nsec)[i] << " from " << (*procs)[i]
			       << " ID " << (*procids)[i] << " (" 
			       << typeEnumerator->processG4Name((*procids)[i])
			       << ") deltaE = " << (*deltae)[i] << " MeV";
  LogDebug("CheckSecondary") << "And " << (*mass).size() << " secondaries "
			     << "produced in the first interactions";
  for (unsigned int i= 0; i < (*mass).size(); i++) 
    LogDebug("CheckSecondary") << "Secondary " << i << " (" << (*px)[i] << ", "
			       << (*py)[i] << ", " << (*pz)[i] << ", " 
			       << (*mass)[i] << ")";

  if (saveToTree) tree->Fill();
}
