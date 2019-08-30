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

PPSPixelG4Hit::PPSPixelG4Hit() : MeanPosition_(0), theEntryPoint_(0), theExitPoint_(0) {
  elem_ = 0.;
  hadr_ = 0.;
  theIncidentEnergy_ = 0.;
  theTrackID_ = -1;
  theUnitID_ = 0;
  theTimeSlice_ = 0.;

  theX_ = 0.;
  theY_ = 0.;
  theZ_ = 0.;
  thePabs_ = 0.;
  theTof_ = 0.;
  theEnergyLoss_ = 0.;
  theParticleType_ = 0;
  theThetaAtEntry_ = 0.;
  thePhiAtEntry_ = 0.;
  theParentId_ = 0;
  theVx_ = 0.;
  theVy_ = 0.;
  theVz_ = 0.;
  thePx_ = 0;
  thePy_ = 0;
  thePz_ = 0;
  theVPx_ = 0;
  theVPy_ = 0;
  theVPz_ = 0;
}

PPSPixelG4Hit::PPSPixelG4Hit(const PPSPixelG4Hit& right) {
  MeanPosition_ = right.MeanPosition_;

  elem_ = right.elem_;
  hadr_ = right.hadr_;
  theIncidentEnergy_ = right.theIncidentEnergy_;
  theTrackID_ = right.theTrackID_;
  theUnitID_ = right.theUnitID_;
  theTimeSlice_ = right.theTimeSlice_;

  theX_ = right.theX_;
  theY_ = right.theY_;
  theZ_ = right.theZ_;
  thePabs_ = right.thePabs_;
  theTof_ = right.theTof_;
  theEnergyLoss_ = right.theEnergyLoss_;
  theParticleType_ = right.theParticleType_;

  theThetaAtEntry_ = right.theThetaAtEntry_;
  thePhiAtEntry_ = right.thePhiAtEntry_;
  theEntryPoint_ = right.theEntryPoint_;
  theExitPoint_ = right.theExitPoint_;
  theParentId_ = right.theParentId_;
  theVx_ = right.theVx_;
  theVy_ = right.theVy_;
  theVz_ = right.theVz_;
  thePx_ = right.thePx_;
  thePy_ = right.thePy_;
  thePz_ = right.thePz_;
  theVPx_ = right.theVPx_;
  theVPy_ = right.theVPy_;
  theVPz_ = right.theVPz_;
}

const PPSPixelG4Hit& PPSPixelG4Hit::operator=(const PPSPixelG4Hit& right) {
  MeanPosition_ = right.MeanPosition_;
  elem_ = right.elem_;
  hadr_ = right.hadr_;
  theIncidentEnergy_ = right.theIncidentEnergy_;
  theTrackID_ = right.theTrackID_;
  theUnitID_ = right.theUnitID_;
  theTimeSlice_ = right.theTimeSlice_;

  theX_ = right.theX_;
  theY_ = right.theY_;
  theZ_ = right.theZ_;
  thePabs_ = right.thePabs_;
  theTof_ = right.theTof_;
  theEnergyLoss_ = right.theEnergyLoss_;
  theParticleType_ = right.theParticleType_;

  theThetaAtEntry_ = right.theThetaAtEntry_;
  thePhiAtEntry_ = right.thePhiAtEntry_;
  theEntryPoint_ = right.theEntryPoint_;
  theExitPoint_ = right.theExitPoint_;
  theParentId_ = right.theParentId_;
  theVx_ = right.theVx_;
  theVy_ = right.theVy_;
  theVz_ = right.theVz_;
  thePx_ = right.thePx_;
  thePy_ = right.thePy_;
  thePz_ = right.thePz_;
  theVPx_ = right.theVPx_;
  theVPy_ = right.theVPy_;
  theVPz_ = right.theVPz_;

  return *this;
}

void PPSPixelG4Hit::addEnergyDeposit(const PPSPixelG4Hit& aHit) {
  elem_ += aHit.getEM();
  hadr_ += aHit.getHadr();
}

void PPSPixelG4Hit::Print() { edm::LogInfo("PPSPixelG4Hit") << (*this); }

const G4ThreeVector& PPSPixelG4Hit::getEntryPoint() const { return theEntryPoint_; }

void PPSPixelG4Hit::setEntryPoint(const G4ThreeVector& xyz) { theEntryPoint_ = xyz; }

const G4ThreeVector& PPSPixelG4Hit::getExitPoint() const { return theExitPoint_; }

void PPSPixelG4Hit::setExitPoint(const G4ThreeVector& xyz) { theExitPoint_ = xyz; }

double PPSPixelG4Hit::getEM() const { return elem_; }
void PPSPixelG4Hit::setEM(double e) { elem_ = e; }

double PPSPixelG4Hit::getHadr() const { return hadr_; }
void PPSPixelG4Hit::setHadr(double e) { hadr_ = e; }

double PPSPixelG4Hit::getIncidentEnergy() const { return theIncidentEnergy_; }
void PPSPixelG4Hit::setIncidentEnergy(double e) { theIncidentEnergy_ = e; }

int PPSPixelG4Hit::getTrackID() const { return theTrackID_; }
void PPSPixelG4Hit::setTrackID(int i) { theTrackID_ = i; }

uint32_t PPSPixelG4Hit::getUnitID() const { return theUnitID_; }
void PPSPixelG4Hit::setUnitID(uint32_t i) { theUnitID_ = i; }

double PPSPixelG4Hit::getTimeSlice() const { return theTimeSlice_; }
void PPSPixelG4Hit::setTimeSlice(double d) { theTimeSlice_ = d; }
int PPSPixelG4Hit::getTimeSliceID() const { return (int)theTimeSlice_; }

void PPSPixelG4Hit::addEnergyDeposit(double em, double hd) {
  elem_ += em;
  hadr_ += hd;
}

double PPSPixelG4Hit::getEnergyDeposit() const { return elem_ + hadr_; }

float PPSPixelG4Hit::getPabs() const { return thePabs_; }
float PPSPixelG4Hit::getTof() const { return theTof_; }
float PPSPixelG4Hit::getEnergyLoss() const { return theEnergyLoss_; }
int PPSPixelG4Hit::getParticleType() const { return theParticleType_; }
float PPSPixelG4Hit::getPx() const { return thePx_; }
float PPSPixelG4Hit::getPy() const { return thePy_; }
float PPSPixelG4Hit::getPz() const { return thePz_; }
float PPSPixelG4Hit::getVPx() const { return theVPx_; }
float PPSPixelG4Hit::getVPy() const { return theVPy_; }
float PPSPixelG4Hit::getVPz() const { return theVPz_; }

void PPSPixelG4Hit::setPabs(float e) { thePabs_ = e; }
void PPSPixelG4Hit::setPx(float e) { thePx_ = e; }
void PPSPixelG4Hit::setPy(float e) { thePy_ = e; }
void PPSPixelG4Hit::setPz(float e) { thePz_ = e; }
void PPSPixelG4Hit::setVPx(float e) { theVPx_ = e; }
void PPSPixelG4Hit::setVPy(float e) { theVPy_ = e; }
void PPSPixelG4Hit::setVPz(float e) { theVPz_ = e; }
void PPSPixelG4Hit::setTof(float e) { theTof_ = e; }
void PPSPixelG4Hit::setEnergyLoss(float e) { theEnergyLoss_ = e; }
void PPSPixelG4Hit::setParticleType(short i) { theParticleType_ = i; }

float PPSPixelG4Hit::getThetaAtEntry() const { return theThetaAtEntry_; }
float PPSPixelG4Hit::getPhiAtEntry() const { return thePhiAtEntry_; }

void PPSPixelG4Hit::setThetaAtEntry(float t) { theThetaAtEntry_ = t; }
void PPSPixelG4Hit::setPhiAtEntry(float f) { thePhiAtEntry_ = f; }

float PPSPixelG4Hit::getX() const { return theX_; }
void PPSPixelG4Hit::setX(float t) { theX_ = t; }

float PPSPixelG4Hit::getY() const { return theY_; }
void PPSPixelG4Hit::setY(float t) { theY_ = t; }

float PPSPixelG4Hit::getZ() const { return theZ_; }
void PPSPixelG4Hit::setZ(float t) { theZ_ = t; }

int PPSPixelG4Hit::getParentId() const { return theParentId_; }
void PPSPixelG4Hit::setParentId(int p) { theParentId_ = p; }

float PPSPixelG4Hit::getVx() const { return theVx_; }
void PPSPixelG4Hit::setVx(float t) { theVx_ = t; }

float PPSPixelG4Hit::getVy() const { return theVy_; }
void PPSPixelG4Hit::setVy(float t) { theVy_ = t; }

float PPSPixelG4Hit::getVz() const { return theVz_; }
void PPSPixelG4Hit::setVz(float t) { theVz_ = t; }

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
