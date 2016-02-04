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
// $Id: TotemTestGem.cc,v 1.3 2009/05/25 06:48:53 fabiocos Exp $
//

// system include files
#include <iostream>
#include <iomanip>
#include <cmath>

// user include files
#include "SimG4Core/Notification/interface/BeginOfEvent.h"
#include "SimG4Core/Notification/interface/EndOfEvent.h"
 
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "SimG4CMS/Forward/interface/TotemTestGem.h"
#include "SimG4CMS/Forward/interface/TotemG4HitCollection.h"

#include "G4SDManager.hh"
#include "G4HCofThisEvent.hh"
#include "CLHEP/Units/GlobalSystemOfUnits.h"
#include "CLHEP/Units/GlobalPhysicalConstants.h"

//
// constructors and destructor
//

TotemTestGem::TotemTestGem(const edm::ParameterSet &p) {
  
  edm::ParameterSet m_Anal = p.getParameter<edm::ParameterSet>("TotemTestGem");
  names        = m_Anal.getParameter<std::vector<std::string> >("Names");
 
  edm::LogInfo("ForwardSim") << "TotemTestGem:: Initialised as observer of "
			     << "begin of job, begin/end events and of G4step";
}

TotemTestGem::~TotemTestGem() {
}
 
//
// member functions
//

void TotemTestGem::produce(edm::Event& e, const edm::EventSetup&) {

  std::auto_ptr<TotemTestHistoClass> product(new TotemTestHistoClass);
  fillEvent(*product);
  e.put(product);
}

void TotemTestGem::update(const BeginOfEvent * evt) {

  int iev = (*evt)()->GetEventID();
  LogDebug("ForwardSim") << "TotemTestGem: Begin of event = " << iev;
  clear();

}

void TotemTestGem::update(const EndOfEvent * evt) {  

  evtnum = (*evt)()->GetEventID();
  LogDebug("ForwardSim") << "TotemTestGem:: Fill event " << evtnum;

  // access to the G4 hit collections 
  G4HCofThisEvent* allHC = (*evt)()->GetHCofThisEvent();
  
  int nhit = 0;
  for (unsigned int in=0; in<names.size(); in++) {
    int HCid = G4SDManager::GetSDMpointer()->GetCollectionID(names[in]);
    TotemG4HitCollection* theHC = (TotemG4HitCollection*) allHC->GetHC(HCid);
    LogDebug("ForwardSim") << "TotemTestGem :: Hit Collection for " <<names[in]
			   << " of ID " << HCid << " is obtained at " << theHC;

    if (HCid >= 0 && theHC > 0) {
      int nentries = theHC->entries();
      LogDebug("ForwardSim") << "TotemTestGem :: " << names[in] << " with "
			     << nentries << " entries";
      for (int ihit = 0; ihit <nentries; ihit++) {
	TotemG4Hit* aHit = (*theHC)[ihit];
	hits.push_back(aHit);
      }
      nhit += nentries;
    }
  }
 
  // Writing the data to the Tree
  LogDebug("ForwardSim") << "TotemTestGem:: --- after fillTree with " << nhit
			 << " Hits";
}
 
void TotemTestGem::fillEvent(TotemTestHistoClass& product) {

  product.setEVT(evtnum);
  
  for (unsigned ihit = 0; ihit < hits.size(); ihit++) {
    TotemG4Hit* aHit = hits[ihit];
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
    product.fillHit(UID, Ptype, TID, PID, ELoss, PABS, vx, vy, vz, x, y,z);
  }
} 

void TotemTestGem::clear() {

  evtnum = 0;
  hits.clear();
}
