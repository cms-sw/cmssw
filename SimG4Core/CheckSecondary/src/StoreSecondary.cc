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
#include "CLHEP/Units/GlobalSystemOfUnits.h"
#include "CLHEP/Units/GlobalPhysicalConstants.h"

#include <cmath>
#include <iostream>
#include <iomanip>

StoreSecondary::StoreSecondary(const edm::ParameterSet &p) {

  edm::ParameterSet m_p = p.getParameter<edm::ParameterSet>("StoreSecondary");
  treatSecondary    = new TreatSecondary (m_p);

  produces<std::vector<math::XYZTLorentzVector> >("SecondaryMomenta");
  produces<std::vector<int> >("SecondaryParticles");
  //  produces<std::vector<std::string> >("SecondaryProcesses");

  edm::LogInfo("CheckSecondary") << "Instantiate StoreSecondary to store "
				 << "secondaries after 1st hadronic inelastic"
				 << " interaction";
} 
   
StoreSecondary::~StoreSecondary() {
  delete treatSecondary;
}

void StoreSecondary::produce(edm::Event& e, const edm::EventSetup&) {

  std::auto_ptr<std::vector<math::XYZTLorentzVector> > secMom(new std::vector<math::XYZTLorentzVector>);
  *secMom = secondaries;
  e.put(secMom, "SecondaryMomenta");

  std::auto_ptr<std::vector<int> > secNumber(new std::vector<int>);
  *secNumber = nsecs;
  e.put(secNumber, "SecondaryParticles");

  /*
  std::auto_ptr<std::vector<std::string> > secProc(new std::vector<std::string>);
  *secProc = procs;
  e.put(secProc, "SecondaryProcesses");
  */

  LogDebug("CheckSecondary") << "StoreSecondary:: Event " << e.id() << " with "
			     << nsecs.size() << " hadronic collisions with "
			     << "secondaries produced in each step";
  for (unsigned int i= 0; i < nsecs.size(); i++) 
    LogDebug("CheckSecondary") << " " << nsecs[i] << " from " << procs[i];
  LogDebug("CheckSecondary") << " and " << secondaries.size() << " secondaries"
			     << " produced in the first interactions:";
  for (unsigned int i= 0; i < secondaries.size(); i++) 
    LogDebug("CheckSecondary") << "Secondary " << i << " " << secondaries[i];
}

void StoreSecondary::update(const BeginOfEvent *) {

  nsecs.clear();
  procs.clear();
  secondaries.clear();
}

void StoreSecondary::update(const BeginOfTrack * trk) {

  const G4Track * thTk = (*trk)();
  treatSecondary->initTrack(thTk);
  if (nsecs.size() == 0 && thTk->GetParentID() <= 0) storeIt = true;
  else                                               storeIt = false;
  nHad  = 0;
}

void StoreSecondary::update(const G4Step * aStep) {

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
  if (hadrInt) {
    nHad++;
    if (storeIt) {
      int sec = (int)(tracks.size());
      nsecs.push_back(sec);
      procs.push_back(name);
      if (nHad == 1) {
	for (int i=0; i<sec; i++) 
	  secondaries.push_back(tracks[i]);
      }
    }
  }
}
