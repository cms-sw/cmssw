// File: CaloG4Hit.cc
// Date: 11.10.02
// Description: Transient Hit class for the calorimeters
#include "SimG4CMS/PPS/interface/TotemRPG4Hit.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <iostream>

TotemRPG4Hit::TotemRPG4Hit() : entry(0) {
  theIncidentEnergy = 0.0;
  theTrackID = -1;
  theUnitID = 0;
  theTimeSlice = 0.0;

  thePabs = 0.0;
  theTof = 0.0;
  theEnergyLoss = 0.0;
  theParticleType = 0;
  theX = 0.0;
  theY = 0.0;
  theZ = 0.0;
  theParentId = 0;
  theVx = 0.0;
  theVy = 0.0;
  theVz = 0.0;
  p_x = p_y = p_z = 0.0;
}

TotemRPG4Hit::TotemRPG4Hit(const TotemRPG4Hit& right) {
  theIncidentEnergy = right.theIncidentEnergy;
  theTrackID = right.theTrackID;
  theUnitID = right.theUnitID;
  theTimeSlice = right.theTimeSlice;
  entry = right.entry;

  thePabs = right.thePabs;
  theTof = right.theTof;
  theEnergyLoss = right.theEnergyLoss;
  theParticleType = right.theParticleType;
  theX = right.theX;
  theY = right.theY;
  theZ = right.theZ;

  theVx = right.theVx;
  theVy = right.theVy;
  theVz = right.theVz;

  theParentId = right.theParentId;
}

const TotemRPG4Hit& TotemRPG4Hit::operator=(const TotemRPG4Hit& right) {
  theIncidentEnergy = right.theIncidentEnergy;
  theTrackID = right.theTrackID;
  theUnitID = right.theUnitID;
  theTimeSlice = right.theTimeSlice;
  entry = right.entry;

  thePabs = right.thePabs;
  theTof = right.theTof;
  theEnergyLoss = right.theEnergyLoss;
  theParticleType = right.theParticleType;
  theX = right.theX;
  theY = right.theY;
  theZ = right.theZ;

  theVx = right.theVx;
  theVy = right.theVy;
  theVz = right.theVz;

  theParentId = right.theParentId;

  return *this;
}

void TotemRPG4Hit::Print() { edm::LogInfo("TotemRP") << (*this); }

G4ThreeVector TotemRPG4Hit::getEntry() const { return entry; }
void TotemRPG4Hit::setEntry(G4ThreeVector xyz) { entry = xyz; }

G4ThreeVector TotemRPG4Hit::getExit() const { return exit; }
void TotemRPG4Hit::setExit(G4ThreeVector xyz) { exit = xyz; }

G4ThreeVector TotemRPG4Hit::getLocalEntry() const { return local_entry; }
void TotemRPG4Hit::setLocalEntry(const G4ThreeVector& xyz) { local_entry = xyz; }
G4ThreeVector TotemRPG4Hit::getLocalExit() const { return local_exit; }
void TotemRPG4Hit::setLocalExit(const G4ThreeVector& xyz) { local_exit = xyz; }

double TotemRPG4Hit::getIncidentEnergy() const { return theIncidentEnergy; }
void TotemRPG4Hit::setIncidentEnergy(double e) { theIncidentEnergy = e; }

unsigned int TotemRPG4Hit::getTrackID() const { return theTrackID; }
void TotemRPG4Hit::setTrackID(int i) { theTrackID = i; }

int TotemRPG4Hit::getUnitID() const { return theUnitID; }
void TotemRPG4Hit::setUnitID(unsigned int i) { theUnitID = i; }

double TotemRPG4Hit::getTimeSlice() const { return theTimeSlice; }
void TotemRPG4Hit::setTimeSlice(double d) { theTimeSlice = d; }
int TotemRPG4Hit::getTimeSliceID() const { return (int)theTimeSlice; }

double TotemRPG4Hit::getPabs() const { return thePabs; }
double TotemRPG4Hit::getTof() const { return theTof; }
double TotemRPG4Hit::getEnergyLoss() const { return theEnergyLoss; }
int TotemRPG4Hit::getParticleType() const { return theParticleType; }

void TotemRPG4Hit::setPabs(double e) { thePabs = e; }
void TotemRPG4Hit::setTof(double e) { theTof = e; }
void TotemRPG4Hit::setEnergyLoss(double e) { theEnergyLoss = e; }
void TotemRPG4Hit::addEnergyLoss(double e) { theEnergyLoss += e; }
void TotemRPG4Hit::setParticleType(short i) { theParticleType = i; }

double TotemRPG4Hit::getThetaAtEntry() const { return theThetaAtEntry; }
double TotemRPG4Hit::getPhiAtEntry() const { return thePhiAtEntry; }

void TotemRPG4Hit::setThetaAtEntry(double t) { theThetaAtEntry = t; }
void TotemRPG4Hit::setPhiAtEntry(double f) { thePhiAtEntry = f; }

double TotemRPG4Hit::getX() const { return theX; }
void TotemRPG4Hit::setX(double t) { theX = t; }

double TotemRPG4Hit::getY() const { return theY; }
void TotemRPG4Hit::setY(double t) { theY = t; }

double TotemRPG4Hit::getZ() const { return theZ; }
void TotemRPG4Hit::setZ(double t) { theZ = t; }

int TotemRPG4Hit::getParentId() const { return theParentId; }
void TotemRPG4Hit::setParentId(int p) { theParentId = p; }

double TotemRPG4Hit::getVx() const { return theVx; }
void TotemRPG4Hit::setVx(double t) { theVx = t; }

double TotemRPG4Hit::getVy() const { return theVy; }
void TotemRPG4Hit::setVy(double t) { theVy = t; }

double TotemRPG4Hit::getVz() const { return theVz; }
void TotemRPG4Hit::setVz(double t) { theVz = t; }

void TotemRPG4Hit::set_p_x(double p) { p_x = p; }
void TotemRPG4Hit::set_p_y(double p) { p_y = p; }
void TotemRPG4Hit::set_p_z(double p) { p_z = p; }

double TotemRPG4Hit::get_p_x() const { return p_x; }
double TotemRPG4Hit::get_p_y() const { return p_y; }
double TotemRPG4Hit::get_p_z() const { return p_z; }

std::ostream& operator<<(std::ostream& os, const TotemRPG4Hit& hit) {
  os << " Data of this TotemRPG4Hit are:" << std::endl
     << " Time slice ID: " << hit.getTimeSliceID() << std::endl
     << " EnergyDeposit = " << hit.getEnergyLoss()
     << std::endl
     //    << " EnergyDeposit of HD particles = " << hit.getHadr() << std::endl
     << " Energy of primary particle (ID = " << hit.getTrackID() << ") = " << hit.getIncidentEnergy() << " (MeV)"
     << std::endl
     << " Entry point in Totem unit number " << hit.getUnitID() << " is: " << hit.getEntry() << " (mm)" << std::endl;
  os << "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~" << std::endl;
  return os;
}
