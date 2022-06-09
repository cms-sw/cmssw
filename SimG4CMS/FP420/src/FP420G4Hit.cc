///////////////////////////////////////////////////////////////////////////////
// File: FP420G4Hit.cc
// Date: 02.2006
// Description: Transient Hit class for the FP420
///////////////////////////////////////////////////////////////////////////////
#include "SimG4CMS/FP420/interface/FP420G4Hit.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <iostream>

FP420G4Hit::FP420G4Hit() : entry(0) {
  entrylp(0);
  exitlp(0);
  elem = 0.;
  hadr = 0.;
  theIncidentEnergy = 0.;
  theTimeSlice = 0.;
  theTrackID = -1;
  theUnitID = 0;
  thePabs = 0.;
  theTof = 0.;
  theEnergyLoss = 0.;
  theParticleType = 0;
  theUnitID = 0;
  theTrackID = -1;
  theThetaAtEntry = -10000.;
  thePhiAtEntry = -10000.;
  theParentId = 0;

  theX = 0.;
  theY = 0.;
  theZ = 0.;
  theVx = 0.;
  theVy = 0.;
  theVz = 0.;
}

FP420G4Hit::~FP420G4Hit() {}

FP420G4Hit::FP420G4Hit(const FP420G4Hit& right) {
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

  theX = right.theX;
  theY = right.theY;
  theZ = right.theZ;

  theVx = right.theVx;
  theVy = right.theVy;
  theVz = right.theVz;
}

const FP420G4Hit& FP420G4Hit::operator=(const FP420G4Hit& right) {
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

  theX = right.theX;
  theY = right.theY;
  theZ = right.theZ;

  theVx = right.theVx;
  theVy = right.theVy;
  theVz = right.theVz;

  return *this;
}

void FP420G4Hit::addEnergyDeposit(const FP420G4Hit& aHit) {
  elem += aHit.getEM();
  hadr += aHit.getHadr();
}

void FP420G4Hit::Print() { edm::LogVerbatim("FP420") << (*this); }

G4ThreeVector FP420G4Hit::getEntry() const { return entry; }
void FP420G4Hit::setEntry(const G4ThreeVector& xyz) { entry = xyz; }

G4ThreeVector FP420G4Hit::getEntryLocalP() const { return entrylp; }
void FP420G4Hit::setEntryLocalP(const G4ThreeVector& xyz1) { entrylp = xyz1; }

G4ThreeVector FP420G4Hit::getExitLocalP() const { return exitlp; }
void FP420G4Hit::setExitLocalP(const G4ThreeVector& xyz1) { exitlp = xyz1; }

double FP420G4Hit::getEM() const { return elem; }
void FP420G4Hit::setEM(double e) { elem = e; }

double FP420G4Hit::getHadr() const { return hadr; }
void FP420G4Hit::setHadr(double e) { hadr = e; }

double FP420G4Hit::getIncidentEnergy() const { return theIncidentEnergy; }
void FP420G4Hit::setIncidentEnergy(double e) { theIncidentEnergy = e; }

unsigned int FP420G4Hit::getTrackID() const { return theTrackID; }
void FP420G4Hit::setTrackID(int i) { theTrackID = i; }

unsigned int FP420G4Hit::getUnitID() const { return theUnitID; }
void FP420G4Hit::setUnitID(unsigned int i) { theUnitID = i; }

double FP420G4Hit::getTimeSlice() const { return theTimeSlice; }
void FP420G4Hit::setTimeSlice(double d) { theTimeSlice = d; }
int FP420G4Hit::getTimeSliceID() const { return (int)theTimeSlice; }

void FP420G4Hit::addEnergyDeposit(double em, double hd) {
  elem += em;
  hadr += hd;
}

double FP420G4Hit::getEnergyDeposit() const { return elem + hadr; }

float FP420G4Hit::getPabs() const { return thePabs; }
float FP420G4Hit::getTof() const { return theTof; }
float FP420G4Hit::getEnergyLoss() const { return theEnergyLoss; }
int FP420G4Hit::getParticleType() const { return theParticleType; }

void FP420G4Hit::setPabs(float e) { thePabs = e; }
void FP420G4Hit::setTof(float e) { theTof = e; }
void FP420G4Hit::addEnergyLoss(float e) { theEnergyLoss += e; }
void FP420G4Hit::setEnergyLoss(float e) { theEnergyLoss = e; }
void FP420G4Hit::setParticleType(short i) { theParticleType = i; }

float FP420G4Hit::getThetaAtEntry() const { return theThetaAtEntry; }
float FP420G4Hit::getPhiAtEntry() const { return thePhiAtEntry; }

void FP420G4Hit::setThetaAtEntry(float t) { theThetaAtEntry = t; }
void FP420G4Hit::setPhiAtEntry(float f) { thePhiAtEntry = f; }

float FP420G4Hit::getX() const { return theX; }
void FP420G4Hit::setX(float t) { theX = t; }

float FP420G4Hit::getY() const { return theY; }
void FP420G4Hit::setY(float t) { theY = t; }

float FP420G4Hit::getZ() const { return theZ; }
void FP420G4Hit::setZ(float t) { theZ = t; }

int FP420G4Hit::getParentId() const { return theParentId; }
void FP420G4Hit::setParentId(int p) { theParentId = p; }

float FP420G4Hit::getVx() const { return theVx; }
void FP420G4Hit::setVx(float t) { theVx = t; }

float FP420G4Hit::getVy() const { return theVy; }
void FP420G4Hit::setVy(float t) { theVy = t; }

float FP420G4Hit::getVz() const { return theVz; }
void FP420G4Hit::setVz(float t) { theVz = t; }

std::ostream& operator<<(std::ostream& os, const FP420G4Hit& hit) {
  os << " Data of this FP420G4Hit are:" << std::endl
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
     << " ParticleType = " << hit.getParticleType() << std::endl
     << " Theta at Entry = " << hit.getThetaAtEntry() << std::endl
     << " Phi at Entry = " << hit.getPhiAtEntry() << std::endl
     << " X at Entry = " << hit.getX() << std::endl
     << " Y at Entry = " << hit.getY() << std::endl
     << " Z at Entry = " << hit.getZ() << std::endl
     << " VtxX = " << hit.getVx() << std::endl
     << " VtxY = " << hit.getVy() << std::endl
     << " VtxZ = " << hit.getVz() << std::endl
     << " parentID = " << hit.getParentId() << std::endl
     << " Pabs = " << hit.getPabs() << std::endl
     << " Energy of primary particle (ID = " << hit.getTrackID() << ") = " << hit.getIncidentEnergy() << " (MeV)"
     << std::endl
     << " Entry point in FP420 unit number " << hit.getUnitID() << " is: " << hit.getEntry() << " (mm)" << std::endl;
  os << "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~" << std::endl;
  return os;
}
