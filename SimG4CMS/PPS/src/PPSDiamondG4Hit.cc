// Author
//Seyed Mohsen Etesami setesami@cern.ch
#include "SimG4CMS/PPS/interface/PPSDiamondG4Hit.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <iostream>

PPSDiamondG4Hit::PPSDiamondG4Hit() : entry(0), exit(0), local_entry(0), local_exit(0) {
  theIncidentEnergy = 0.0;
  theTrackID = -1;
  theUnitID = 0;
  theTimeSlice = 0.0;
  theGlobaltimehit = 0.0;
  theX = 0.0;
  theY = 0.0;
  theZ = 0.0;
  thePabs = 0.0;
  theTof = 0.0;
  theEnergyLoss = 0.0;
  theParticleType = 0;
  theParentId = 0;
  theVx = 0.0;
  theVy = 0.0;
  theVz = 0.0;
  p_x = p_y = p_z = 0.0;
  theThetaAtEntry = 0;
  thePhiAtEntry = 0;
}

PPSDiamondG4Hit::~PPSDiamondG4Hit() {}

PPSDiamondG4Hit::PPSDiamondG4Hit(const PPSDiamondG4Hit& right) {
  entry = right.entry;
  exit = right.exit;
  local_entry = right.local_entry;
  local_exit = right.local_exit;
  theIncidentEnergy = right.theIncidentEnergy;
  theTrackID = right.theTrackID;
  theUnitID = right.theUnitID;
  theTimeSlice = right.theTimeSlice;
  theGlobaltimehit = right.theGlobaltimehit;
  theX = right.theX;
  theY = right.theY;
  theZ = right.theZ;
  thePabs = right.thePabs;
  theTof = right.theTof;
  theEnergyLoss = right.theEnergyLoss;
  theParticleType = right.theParticleType;
  theParentId = right.theParentId;
  theVx = right.theVx;
  theVy = right.theVy;
  theVz = right.theVz;
  p_x = right.p_x;
  p_y = right.p_y;
  p_z = right.p_z;
  theThetaAtEntry = right.theThetaAtEntry;
  thePhiAtEntry = right.thePhiAtEntry;
}

const PPSDiamondG4Hit& PPSDiamondG4Hit::operator=(const PPSDiamondG4Hit& right) {
  entry = right.entry;
  exit = right.exit;
  local_entry = right.local_entry;
  local_exit = right.local_exit;
  theIncidentEnergy = right.theIncidentEnergy;
  theTrackID = right.theTrackID;
  theUnitID = right.theUnitID;
  theTimeSlice = right.theTimeSlice;
  theGlobaltimehit = right.theGlobaltimehit;
  theX = right.theX;
  theY = right.theY;
  theZ = right.theZ;
  thePabs = right.thePabs;
  theTof = right.theTof;
  theEnergyLoss = right.theEnergyLoss;
  theParticleType = right.theParticleType;
  theParentId = right.theParentId;
  theVx = right.theVx;
  theVy = right.theVy;
  theVz = right.theVz;
  p_x = right.p_x;
  p_y = right.p_y;
  p_z = right.p_z;
  theThetaAtEntry = right.theThetaAtEntry;
  thePhiAtEntry = right.thePhiAtEntry;

  return *this;
}

void PPSDiamondG4Hit::Print() { edm::LogInfo("PPSSimDiamond") << (*this); }

const G4ThreeVector& PPSDiamondG4Hit::getEntry() const { return entry; }
void PPSDiamondG4Hit::setEntry(const G4ThreeVector& xyz) { entry = xyz; }

const G4ThreeVector& PPSDiamondG4Hit::getExit() const { return exit; }
void PPSDiamondG4Hit::setExit(const G4ThreeVector& xyz) { exit = xyz; }

const G4ThreeVector& PPSDiamondG4Hit::getLocalEntry() const { return local_entry; }
void PPSDiamondG4Hit::setLocalEntry(const G4ThreeVector& xyz) { local_entry = xyz; }
const G4ThreeVector& PPSDiamondG4Hit::getLocalExit() const { return local_exit; }
void PPSDiamondG4Hit::setLocalExit(const G4ThreeVector& xyz) { local_exit = xyz; }

double PPSDiamondG4Hit::getIncidentEnergy() const { return theIncidentEnergy; }
void PPSDiamondG4Hit::setIncidentEnergy(double e) { theIncidentEnergy = e; }

unsigned int PPSDiamondG4Hit::getTrackID() const { return theTrackID; }
void PPSDiamondG4Hit::setTrackID(int i) { theTrackID = i; }

int PPSDiamondG4Hit::getUnitID() const { return theUnitID; }
void PPSDiamondG4Hit::setUnitID(unsigned int i) { theUnitID = i; }

double PPSDiamondG4Hit::getTimeSlice() const { return theTimeSlice; }
void PPSDiamondG4Hit::setTimeSlice(double d) { theTimeSlice = d; }
int PPSDiamondG4Hit::getTimeSliceID() const { return (int)theTimeSlice; }

double PPSDiamondG4Hit::getPabs() const { return thePabs; }
double PPSDiamondG4Hit::getTof() const { return theTof; }
double PPSDiamondG4Hit::getEnergyLoss() const { return theEnergyLoss; }
int PPSDiamondG4Hit::getParticleType() const { return theParticleType; }

void PPSDiamondG4Hit::setPabs(double e) { thePabs = e; }
void PPSDiamondG4Hit::setTof(double e) { theTof = e; }
void PPSDiamondG4Hit::setEnergyLoss(double e) { theEnergyLoss = e; }
void PPSDiamondG4Hit::addEnergyLoss(double e) { theEnergyLoss += e; }
void PPSDiamondG4Hit::setParticleType(short i) { theParticleType = i; }

double PPSDiamondG4Hit::getThetaAtEntry() const { return theThetaAtEntry; }
double PPSDiamondG4Hit::getPhiAtEntry() const { return thePhiAtEntry; }

void PPSDiamondG4Hit::setThetaAtEntry(double t) { theThetaAtEntry = t; }
void PPSDiamondG4Hit::setPhiAtEntry(double f) { thePhiAtEntry = f; }

double PPSDiamondG4Hit::getX() const { return theX; }
void PPSDiamondG4Hit::setX(double t) { theX = t; }

double PPSDiamondG4Hit::getY() const { return theY; }
void PPSDiamondG4Hit::setY(double t) { theY = t; }

double PPSDiamondG4Hit::getZ() const { return theZ; }
void PPSDiamondG4Hit::setZ(double t) { theZ = t; }

int PPSDiamondG4Hit::getParentId() const { return theParentId; }
void PPSDiamondG4Hit::setParentId(int p) { theParentId = p; }

double PPSDiamondG4Hit::getVx() const { return theVx; }
void PPSDiamondG4Hit::setVx(double t) { theVx = t; }

double PPSDiamondG4Hit::getVy() const { return theVy; }
void PPSDiamondG4Hit::setVy(double t) { theVy = t; }

double PPSDiamondG4Hit::getVz() const { return theVz; }
void PPSDiamondG4Hit::setVz(double t) { theVz = t; }

void PPSDiamondG4Hit::set_p_x(double p) { p_x = p; }
void PPSDiamondG4Hit::set_p_y(double p) { p_y = p; }
void PPSDiamondG4Hit::set_p_z(double p) { p_z = p; }

double PPSDiamondG4Hit::get_p_x() const { return p_x; }
double PPSDiamondG4Hit::get_p_y() const { return p_y; }
double PPSDiamondG4Hit::get_p_z() const { return p_z; }

double PPSDiamondG4Hit::getGlobalTimehit() const { return theGlobaltimehit; }
void PPSDiamondG4Hit::setGlobalTimehit(double h) { theGlobaltimehit = h; }

std::ostream& operator<<(std::ostream& os, const PPSDiamondG4Hit& hit) {
  os << " Data of this PPSDiamondG4Hit are:" << std::endl
     << " Time slice ID: " << hit.getTimeSliceID() << std::endl
     << " EnergyDeposit = " << hit.getEnergyLoss() << std::endl
     << " Energy of primary particle (ID = " << hit.getTrackID() << ") = " << hit.getIncidentEnergy() << " (MeV)"
     << "\n"
     << " Local entry and exit points in PPS unit number " << hit.getUnitID() << " are: " << hit.getEntry() << " (mm)"
     << hit.getExit() << " (mm)"
     << "\n"
     << "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~" << std::endl;
  return os;
}
