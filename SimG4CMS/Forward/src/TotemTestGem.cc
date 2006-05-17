// -*- C++ -*-
//
// Package:     Forward
// Class  :     TotemTestGem
//
// Implementation:
//     <Notes on implementation>
//
// Original Author: 
//         Created:  Tue May 16 10:14:34 CEST 2006
// $Id$
//

// system include files
#include <iostream>
#include <iomanip>
#include <cmath>

// user include files
#include "SimG4Core/Notification/interface/BeginOfJob.h"
#include "SimG4Core/Notification/interface/BeginOfEvent.h"
#include "SimG4Core/Notification/interface/EndOfEvent.h"
 
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "SimG4CMS/Forward/interface/TotemTestGem.h"
#include "SimG4CMS/Forward/interface/TotemG4Hit.h"
#include "SimG4CMS/Forward/interface/TotemG4HitCollection.h"

#include "G4SDManager.hh"
#include "G4Step.hh"
#include "G4Track.hh"
#include "G4VProcess.hh"
#include "G4HCofThisEvent.hh"
#include "CLHEP/Units/SystemOfUnits.h"
#include "CLHEP/Units/PhysicalConstants.h"

//
// constructors and destructor
//

TotemTestGem::TotemTestGem(const edm::ParameterSet &p):
  tuplesManager(0), tuples(0) {
  
  edm::ParameterSet m_Anal = p.getParameter<edm::ParameterSet>("TotemTestGem");
  names        = m_Anal.getParameter<std::vector<std::string> >("Names");
  fileName     = m_Anal.getParameter<std::string>("FileName");
 
  edm::LogInfo("ForwardSim") << "TotemTestGem:: Initialised as observer of "
			     << "begin of job, begin/end events and of G4step";
  edm::LogInfo("HcalSim") << "TotemTestGem:===>>>  Book user"
			  << " Histograms and Root tree";
}

TotemTestGem::~TotemTestGem() {
}

void TotemTestGem::update(const BeginOfJob * job) {

  // Ntuples
  tuplesManager.reset(new TotemTestHistoManager(fileName));
}

void TotemTestGem::update(const BeginOfEvent * evt) {
                                                                               
  // create tuple object
  tuples = new TotemTestHistoClass();

  int iev = (*evt)()->GetEventID();
  LogDebug("ForwardSim") << "TotemTestGem: Begin of event = " << iev;
}

void TotemTestGem::update(const G4Step * aStep) {}

void TotemTestGem::update(const EndOfEvent * evt) {  

  int evtnum = (*evt)()->GetEventID();
  LogDebug("ForwardSim") << "TotemTestGem:: Fill event " << evtnum;
  tuples->setEVT(evtnum);

  // access to the G4 hit collections 
  G4HCofThisEvent* allHC = (*evt)()->GetHCofThisEvent();
  
  int ihit = 0;
  for (unsigned int in=0; in<names.size(); in++) {
    int HCid = G4SDManager::GetSDMpointer()->GetCollectionID(names[in]);
    TotemG4HitCollection* theHC = (TotemG4HitCollection*) allHC->GetHC(HCid);
    LogDebug("ForwardSim") << "TotemTestGem :: Hit Collection for " <<names[in]
			   << " of ID " << HCid << " is obtained at " << theHC;

    if (HCid >= 0 && theHC > 0) {
      int nentries = theHC->entries();
      LogDebug("ForwardSim") << "TotemTestGem :: " << names[in] << " with "
			     << nentries << " entries";
      for (ihit = 0; ihit <nentries; ihit++) {
	TotemG4Hit* aHit = (*theHC)[ihit];

	int UID     = aHit->getUnitID();
	int Ptype   = aHit->getParticleType();
	int TID     = aHit->getTrackID();
	int PID     = aHit->getParentId();
	float ELoss = aHit->getEnergyLoss();
	float PABS  = aHit->getPabs();
	float x     = aHit->getEntry().x();
	float y     = aHit->getEntry().y();
	float z     = aHit->getEntry().z();
	float vx    = aHit->getVx();
	float vy    = aHit->getVy();
	float vz    = aHit->getVz();
	tuples->fillHit(UID, Ptype, TID, PID, ELoss, PABS, vx, vy, vz, x, y,z);
      }
    }
  }
 
  // Writing the data to the Tree
  tuplesManager->fillTree(tuples); // (no need to delete it...)
  tuples = 0; // but avoid to reuse it...
  LogDebug("ForwardSim") << "TotemTestGem:: --- after fillTree";
}
