///////////////////////////////////////////////////////////////////////////////
// File: BscG4Hit.cc
// Date: 02.2006
// Description: Transient Hit class for the Bsc
///////////////////////////////////////////////////////////////////////////////
#include "SimG4CMS/Forward/interface/BscG4Hit.h"
#include <iostream>

BscG4Hit::BscG4Hit() : entry(0., 0., 0.), entrylp(0., 0., 0.), exitlp(0., 0., 0.) {
  elem = 0.f;
  hadr = 0.f;
  theIncidentEnergy = 0.f;
  theTimeSlice = 0.;
  theTrackID = -1;
  theUnitID = 0;
  thePabs = 0.f;
  theTof = 0.f;
  theEnergyLoss = 0.f;
  theParticleType = 0;
  theUnitID = 0;
  theTrackID = -1;
  theThetaAtEntry = -10000.f;
  thePhiAtEntry = -10000.f;
  theParentId = 0;
  theProcessId = 0;

  theX = 0.f;
  theY = 0.f;
  theZ = 0.f;
  theVx = 0.f;
  theVy = 0.f;
  theVz = 0.f;
}

BscG4Hit::~BscG4Hit() {}

BscG4Hit::BscG4Hit(const BscG4Hit& right) {
  theUnitID = right.theUnitID;

  theTrackID = right.theTrackID;
  theTof = right.theTof;
  theEnergyLoss = right.theEnergyLoss;
  theParticleType = right.theParticleType;
  thePabs = right.thePabs;
  elem = right.elem;
  hadr = right.hadr;
  theIncidentEnergy = right.theIncidentEnergy;
  theTimeSlice = right.theTimeSlice;
  entry = right.entry;
  entrylp = right.entrylp;
  exitlp = right.exitlp;
  theThetaAtEntry = right.theThetaAtEntry;
  thePhiAtEntry = right.thePhiAtEntry;
  theParentId = right.theParentId;
  theProcessId = right.theProcessId;

  theX = right.theX;
  theY = right.theY;
  theZ = right.theZ;

  theVx = right.theVx;
  theVy = right.theVy;
  theVz = right.theVz;
}

const BscG4Hit& BscG4Hit::operator=(const BscG4Hit& right) {
  theUnitID = right.theUnitID;

  theTrackID = right.theTrackID;
  theTof = right.theTof;
  theEnergyLoss = right.theEnergyLoss;
  theParticleType = right.theParticleType;
  thePabs = right.thePabs;
  elem = right.elem;
  hadr = right.hadr;
  theIncidentEnergy = right.theIncidentEnergy;
  theTimeSlice = right.theTimeSlice;
  entry = right.entry;
  entrylp = right.entrylp;
  exitlp = right.exitlp;
  theThetaAtEntry = right.theThetaAtEntry;
  thePhiAtEntry = right.thePhiAtEntry;
  theParentId = right.theParentId;
  theProcessId = right.theProcessId;

  theX = right.theX;
  theY = right.theY;
  theZ = right.theZ;

  theVx = right.theVx;
  theVy = right.theVy;
  theVz = right.theVz;

  return *this;
}

void BscG4Hit::setEntry(const G4ThreeVector& v) {
  entry = v;
  theX = v.x();
  theY = v.y();
  theZ = v.z();
}

void BscG4Hit::addEnergyDeposit(const BscG4Hit& aHit) {
  elem += aHit.getEM();
  hadr += aHit.getHadr();
  theEnergyLoss = elem + hadr;
}

void BscG4Hit::Print() { std::cout << (*this); }

void BscG4Hit::addEnergyDeposit(float em, float hd) {
  elem += em;
  hadr += hd;
  theEnergyLoss = elem + hadr;
}

void BscG4Hit::setHitPosition(const G4ThreeVector& v) {
  theX = v.x();
  theY = v.y();
  theZ = v.z();
}

void BscG4Hit::setVertexPosition(const G4ThreeVector& v) {
  theVx = v.x();
  theVy = v.y();
  theVz = v.z();
}

std::ostream& operator<<(std::ostream& os, const BscG4Hit& hit) {
  os << " Data of this BscG4Hit are:" << std::endl
     << " hitEntryLocalP: " << hit.getEntryLocalP() << std::endl
     << " hitExitLocalP: " << hit.getExitLocalP() << std::endl
     << " Time slice ID: " << hit.getTimeSliceID() << std::endl
     << " Time slice : " << hit.getTimeSlice() << std::endl
     << " Tof : " << hit.getTof() << std::endl
     << " EnergyDeposit = " << hit.getEnergyDeposit() << std::endl
     << " elmenergy = " << hit.getEM() << std::endl
     << " hadrenergy = " << hit.getHadr() << std::endl
     << " EnergyLoss = " << hit.getEnergyLoss() << std::endl
     << " ParticleType = " << hit.getParticleType() << std::endl
     << " Pabs = " << hit.getPabs() << std::endl
     << " Energy of primary particle (ID = " << hit.getTrackID() << ") = " << hit.getIncidentEnergy() << " (MeV)"
     << std::endl
     << " Entry point in Bsc unit number " << hit.getUnitID() << " is: " << hit.getEntry() << " (mm)" << std::endl;
  os << "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~" << std::endl;
  return os;
}
