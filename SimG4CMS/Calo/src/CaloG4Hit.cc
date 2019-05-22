///////////////////////////////////////////////////////////////////////////////
// File: CaloG4Hit.cc
// Description: Transient Hit class for the calorimeters
///////////////////////////////////////////////////////////////////////////////
#include "SimG4CMS/Calo/interface/CaloG4Hit.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <iostream>

#include "G4SystemOfUnits.hh"

G4ThreadLocal G4Allocator<CaloG4Hit>* fpCaloG4HitAllocator = nullptr;

CaloG4Hit::CaloG4Hit() {
  setEntry(0., 0., 0.);
  setEntryLocal(0., 0., 0.);
  elem = 0.;
  hadr = 0.;
  theIncidentEnergy = 0.;
}

CaloG4Hit::~CaloG4Hit() {}

CaloG4Hit::CaloG4Hit(const CaloG4Hit& right) {
  entry = right.entry;
  entryLocal = right.entryLocal;
  pos = right.pos;
  elem = right.elem;
  hadr = right.hadr;
  theIncidentEnergy = right.theIncidentEnergy;
  hitID = right.hitID;
}

const CaloG4Hit& CaloG4Hit::operator=(const CaloG4Hit& right) {
  entry = right.entry;
  entryLocal = right.entryLocal;
  pos = right.pos;
  elem = right.elem;
  hadr = right.hadr;
  theIncidentEnergy = right.theIncidentEnergy;
  hitID = right.hitID;

  return *this;
}

void CaloG4Hit::addEnergyDeposit(double em, double hd) {
  elem += em;
  hadr += hd;
}

void CaloG4Hit::addEnergyDeposit(const CaloG4Hit& aHit) { addEnergyDeposit(aHit.getEM(), aHit.getHadr()); }

void CaloG4Hit::Print() { edm::LogVerbatim("CaloSim") << (*this); }

std::ostream& operator<<(std::ostream& os, const CaloG4Hit& hit) {
  os << " Data of this CaloG4Hit are:"
     << "\n"
     << " HitID: " << hit.getID() << "\n"
     << " EnergyDeposit of EM particles = " << hit.getEM() << "\n"
     << " EnergyDeposit of HD particles = " << hit.getHadr() << "\n"
     << " Energy of primary particle    = " << hit.getIncidentEnergy() / MeV << " (MeV)"
     << "\n"
     << " Entry point in Calorimeter (global) : " << hit.getEntry() << "   (local) " << hit.getEntryLocal() << "\n"
     << " Position of Hit (global) : " << hit.getPosition() << "\n"
     << "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n";
  return os;
}
