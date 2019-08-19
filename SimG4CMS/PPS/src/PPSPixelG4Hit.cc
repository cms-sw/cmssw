// -*- C++ -*-
//
// Package:     PPS
// Class  :     PPSPixelG4Hit
//
// Implementation:
//     <Notes on implementation>
//
// Original Author:
//         Created:  Tue May 16 10:14:34 CEST 2006
//

// system include files

// user include files
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "SimG4CMS/PPS/interface/PPSPixelG4Hit.h"
#include <iostream>

//
// constructors and destructor
//

PPSPixelG4Hit::PPSPixelG4Hit() : MeanPosition(0), theEntryPoint(0), theExitPoint(0) {
  elem = 0.;
  hadr = 0.;
  theIncidentEnergy = 0.;
  theTrackID = -1;
  theUnitID = 0;
  theTimeSlice = 0.;

  theX = 0.;
  theY = 0.;
  theZ = 0.;
  thePabs = 0.;
  theTof = 0.;
  theEnergyLoss = 0.;
  theParticleType = 0;
  theThetaAtEntry = 0.;
  thePhiAtEntry = 0.;
  theParentId = 0;
  theVx = 0.;
  theVy = 0.;
  theVz = 0.;
  thePx = 0;
  thePy = 0;
  thePz = 0;
  theVPx = 0;
  theVPy = 0;
  theVPz = 0;
}

PPSPixelG4Hit::PPSPixelG4Hit(const PPSPixelG4Hit& right) {
  MeanPosition = right.MeanPosition;

  elem = right.elem;
  hadr = right.hadr;
  theIncidentEnergy = right.theIncidentEnergy;
  theTrackID = right.theTrackID;
  theUnitID = right.theUnitID;
  theTimeSlice = right.theTimeSlice;

  theX = right.theX;
  theY = right.theY;
  theZ = right.theZ;
  thePabs = right.thePabs;
  theTof = right.theTof;
  theEnergyLoss = right.theEnergyLoss;
  theParticleType = right.theParticleType;

  theThetaAtEntry = right.theThetaAtEntry;
  thePhiAtEntry = right.thePhiAtEntry;
  theEntryPoint = right.theEntryPoint;
  theExitPoint = right.theExitPoint;
  theParentId = right.theParentId;
  theVx = right.theVx;
  theVy = right.theVy;
  theVz = right.theVz;
  thePx = right.thePx;
  thePy = right.thePy;
  thePz = right.thePz;
  theVPx = right.theVPx;
  theVPy = right.theVPy;
  theVPz = right.theVPz;
}

const PPSPixelG4Hit& PPSPixelG4Hit::operator=(const PPSPixelG4Hit& right) {
  MeanPosition = right.MeanPosition;
  elem = right.elem;
  hadr = right.hadr;
  theIncidentEnergy = right.theIncidentEnergy;
  theTrackID = right.theTrackID;
  theUnitID = right.theUnitID;
  theTimeSlice = right.theTimeSlice;

  theX = right.theX;
  theY = right.theY;
  theZ = right.theZ;
  thePabs = right.thePabs;
  theTof = right.theTof;
  theEnergyLoss = right.theEnergyLoss;
  theParticleType = right.theParticleType;

  theThetaAtEntry = right.theThetaAtEntry;
  thePhiAtEntry = right.thePhiAtEntry;
  theEntryPoint = right.theEntryPoint;
  theExitPoint = right.theExitPoint;
  theParentId = right.theParentId;
  theVx = right.theVx;
  theVy = right.theVy;
  theVz = right.theVz;
  thePx = right.thePx;
  thePy = right.thePy;
  thePz = right.thePz;
  theVPx = right.theVPx;
  theVPy = right.theVPy;
  theVPz = right.theVPz;

  return *this;
}

void PPSPixelG4Hit::addEnergyDeposit(const PPSPixelG4Hit& aHit) {
  elem += aHit.getEM();
  hadr += aHit.getHadr();
}

void PPSPixelG4Hit::Print() { edm::LogInfo("PPSPixelG4Hit") << (*this); }

const G4ThreeVector& PPSPixelG4Hit::getEntryPoint() const { return theEntryPoint; }

void PPSPixelG4Hit::setEntryPoint(const G4ThreeVector& xyz) { theEntryPoint = xyz; }

const G4ThreeVector& PPSPixelG4Hit::getExitPoint() const { return theExitPoint; }

void PPSPixelG4Hit::setExitPoint(const G4ThreeVector& xyz) { theExitPoint = xyz; }

double PPSPixelG4Hit::getEM() const { return elem; }
void PPSPixelG4Hit::setEM(double e) { elem = e; }

double PPSPixelG4Hit::getHadr() const { return hadr; }
void PPSPixelG4Hit::setHadr(double e) { hadr = e; }

double PPSPixelG4Hit::getIncidentEnergy() const { return theIncidentEnergy; }
void PPSPixelG4Hit::setIncidentEnergy(double e) { theIncidentEnergy = e; }

int PPSPixelG4Hit::getTrackID() const { return theTrackID; }
void PPSPixelG4Hit::setTrackID(int i) { theTrackID = i; }

uint32_t PPSPixelG4Hit::getUnitID() const { return theUnitID; }
void PPSPixelG4Hit::setUnitID(uint32_t i) { theUnitID = i; }

double PPSPixelG4Hit::getTimeSlice() const { return theTimeSlice; }
void PPSPixelG4Hit::setTimeSlice(double d) { theTimeSlice = d; }
int PPSPixelG4Hit::getTimeSliceID() const { return (int)theTimeSlice; }

void PPSPixelG4Hit::addEnergyDeposit(double em, double hd) {
  elem += em;
  hadr += hd;
}

double PPSPixelG4Hit::getEnergyDeposit() const { return elem + hadr; }

float PPSPixelG4Hit::getPabs() const { return thePabs; }
float PPSPixelG4Hit::getTof() const { return theTof; }
float PPSPixelG4Hit::getEnergyLoss() const { return theEnergyLoss; }
int PPSPixelG4Hit::getParticleType() const { return theParticleType; }
float PPSPixelG4Hit::getPx() const { return thePx; }
float PPSPixelG4Hit::getPy() const { return thePy; }
float PPSPixelG4Hit::getPz() const { return thePz; }
float PPSPixelG4Hit::getVPx() const { return theVPx; }
float PPSPixelG4Hit::getVPy() const { return theVPy; }
float PPSPixelG4Hit::getVPz() const { return theVPz; }

void PPSPixelG4Hit::setPabs(float e) { thePabs = e; }
void PPSPixelG4Hit::setPx(float e) { thePx = e; }
void PPSPixelG4Hit::setPy(float e) { thePy = e; }
void PPSPixelG4Hit::setPz(float e) { thePz = e; }
void PPSPixelG4Hit::setVPx(float e) { theVPx = e; }
void PPSPixelG4Hit::setVPy(float e) { theVPy = e; }
void PPSPixelG4Hit::setVPz(float e) { theVPz = e; }
void PPSPixelG4Hit::setTof(float e) { theTof = e; }
void PPSPixelG4Hit::setEnergyLoss(float e) { theEnergyLoss = e; }
void PPSPixelG4Hit::setParticleType(short i) { theParticleType = i; }

float PPSPixelG4Hit::getThetaAtEntry() const { return theThetaAtEntry; }
float PPSPixelG4Hit::getPhiAtEntry() const { return thePhiAtEntry; }

void PPSPixelG4Hit::setThetaAtEntry(float t) { theThetaAtEntry = t; }
void PPSPixelG4Hit::setPhiAtEntry(float f) { thePhiAtEntry = f; }

float PPSPixelG4Hit::getX() const { return theX; }
void PPSPixelG4Hit::setX(float t) { theX = t; }

float PPSPixelG4Hit::getY() const { return theY; }
void PPSPixelG4Hit::setY(float t) { theY = t; }

float PPSPixelG4Hit::getZ() const { return theZ; }
void PPSPixelG4Hit::setZ(float t) { theZ = t; }

int PPSPixelG4Hit::getParentId() const { return theParentId; }
void PPSPixelG4Hit::setParentId(int p) { theParentId = p; }

float PPSPixelG4Hit::getVx() const { return theVx; }
void PPSPixelG4Hit::setVx(float t) { theVx = t; }

float PPSPixelG4Hit::getVy() const { return theVy; }
void PPSPixelG4Hit::setVy(float t) { theVy = t; }

float PPSPixelG4Hit::getVz() const { return theVz; }
void PPSPixelG4Hit::setVz(float t) { theVz = t; }

std::ostream& operator<<(std::ostream& os, const PPSPixelG4Hit& hit) {
  os << " Data of this PPSPixelG4Hit are:\n"
     << " Time slice ID: " << hit.getTimeSliceID() << "\n"
     << " EnergyDeposit = " << hit.getEnergyLoss() << "\n"
     << " Energy of primary particle (ID = " << hit.getTrackID() << ") = " << hit.getIncidentEnergy() << " (MeV)"
     << "\n"
     << " Local entry and exit points in PPS unit number " << hit.getUnitID() << " are: " << hit.getEntryPoint()
     << " (mm)" << hit.getExitPoint() << " (mm)"
     << "\n"
     << " Global posizion in PPS unit number " << hit.getUnitID() << " are: " << hit.getMeanPosition() << " (mm)"
     << "\n"
     << "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n";
  return os;
}
