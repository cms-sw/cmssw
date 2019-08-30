// Date: 11.10.02
// Description: Transient Hit class for the calorimeters
#include "SimG4CMS/PPS/interface/TotemRPG4Hit.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <iostream>

TotemRPG4Hit::TotemRPG4Hit() : entry_(0) {
  theIncidentEnergy_ = 0.0;
  theTrackID_ = -1;
  theUnitID_ = 0;
  theTimeSlice_ = 0.0;

  thePabs_ = 0.0;
  theTof_ = 0.0;
  theEnergyLoss_ = 0.0;
  theParticleType_ = 0;
  theX_ = 0.0;
  theY_ = 0.0;
  theZ_ = 0.0;
  theParentId_ = 0;
  theVx_ = 0.0;
  theVy_ = 0.0;
  theVz_ = 0.0;
  thePx_ = thePy_ = thePz_ = 0.0;
}

TotemRPG4Hit::TotemRPG4Hit(const TotemRPG4Hit& right) {
  theIncidentEnergy_ = right.theIncidentEnergy_;
  theTrackID_ = right.theTrackID_;
  theUnitID_ = right.theUnitID_;
  theTimeSlice_ = right.theTimeSlice_;
  entry_ = right.entry_;

  thePabs_ = right.thePabs_;
  theTof_ = right.theTof_;
  theEnergyLoss_ = right.theEnergyLoss_;
  theParticleType_ = right.theParticleType_;
  theX_ = right.theX_;
  theY_ = right.theY_;
  theZ_ = right.theZ_;

  theVx_ = right.theVx_;
  theVy_ = right.theVy_;
  theVz_ = right.theVz_;

  theParentId_ = right.theParentId_;
}

const TotemRPG4Hit& TotemRPG4Hit::operator=(const TotemRPG4Hit& right) {
  theIncidentEnergy_ = right.theIncidentEnergy_;
  theTrackID_ = right.theTrackID_;
  theUnitID_ = right.theUnitID_;
  theTimeSlice_ = right.theTimeSlice_;
  entry_ = right.entry_;

  thePabs_ = right.thePabs_;
  theTof_ = right.theTof_;
  theEnergyLoss_ = right.theEnergyLoss_;
  theParticleType_ = right.theParticleType_;
  theX_ = right.theX_;
  theY_ = right.theY_;
  theZ_ = right.theZ_;

  theVx_ = right.theVx_;
  theVy_ = right.theVy_;
  theVz_ = right.theVz_;

  theParentId_ = right.theParentId_;

  return *this;
}

void TotemRPG4Hit::Print() { edm::LogInfo("TotemRP") << (*this); }

G4ThreeVector TotemRPG4Hit::getEntry() const { return entry_; }
void TotemRPG4Hit::setEntry(G4ThreeVector xyz) { entry_ = xyz; }

G4ThreeVector TotemRPG4Hit::getExit() const { return exit_; }
void TotemRPG4Hit::setExit(G4ThreeVector xyz) { exit_ = xyz; }

G4ThreeVector TotemRPG4Hit::getLocalEntry() const { return local_entry_; }
void TotemRPG4Hit::setLocalEntry(const G4ThreeVector& xyz) { local_entry_ = xyz; }
G4ThreeVector TotemRPG4Hit::getLocalExit() const { return local_exit_; }
void TotemRPG4Hit::setLocalExit(const G4ThreeVector& xyz) { local_exit_ = xyz; }

double TotemRPG4Hit::getIncidentEnergy() const { return theIncidentEnergy_; }
void TotemRPG4Hit::setIncidentEnergy(double e) { theIncidentEnergy_ = e; }

unsigned int TotemRPG4Hit::getTrackID() const { return theTrackID_; }
void TotemRPG4Hit::setTrackID(int i) { theTrackID_ = i; }

int TotemRPG4Hit::getUnitID() const { return theUnitID_; }
void TotemRPG4Hit::setUnitID(unsigned int i) { theUnitID_ = i; }

double TotemRPG4Hit::getTimeSlice() const { return theTimeSlice_; }
void TotemRPG4Hit::setTimeSlice(double d) { theTimeSlice_ = d; }
int TotemRPG4Hit::getTimeSliceID() const { return (int)theTimeSlice_; }

double TotemRPG4Hit::getPabs() const { return thePabs_; }
double TotemRPG4Hit::getTof() const { return theTof_; }
double TotemRPG4Hit::getEnergyLoss() const { return theEnergyLoss_; }
int TotemRPG4Hit::getParticleType() const { return theParticleType_; }

void TotemRPG4Hit::setPabs(double e) { thePabs_ = e; }
void TotemRPG4Hit::setTof(double e) { theTof_ = e; }
void TotemRPG4Hit::setEnergyLoss(double e) { theEnergyLoss_ = e; }
void TotemRPG4Hit::addEnergyLoss(double e) { theEnergyLoss_ += e; }
void TotemRPG4Hit::setParticleType(short i) { theParticleType_ = i; }

double TotemRPG4Hit::getThetaAtEntry() const { return theThetaAtEntry_; }
double TotemRPG4Hit::getPhiAtEntry() const { return thePhiAtEntry_; }

void TotemRPG4Hit::setThetaAtEntry(double t) { theThetaAtEntry_ = t; }
void TotemRPG4Hit::setPhiAtEntry(double f) { thePhiAtEntry_ = f; }

double TotemRPG4Hit::getX() const { return theX_; }
void TotemRPG4Hit::setX(double t) { theX_ = t; }

double TotemRPG4Hit::getY() const { return theY_; }
void TotemRPG4Hit::setY(double t) { theY_ = t; }

double TotemRPG4Hit::getZ() const { return theZ_; }
void TotemRPG4Hit::setZ(double t) { theZ_ = t; }

int TotemRPG4Hit::getParentId() const { return theParentId_; }
void TotemRPG4Hit::setParentId(int p) { theParentId_ = p; }

double TotemRPG4Hit::getVx() const { return theVx_; }
void TotemRPG4Hit::setVx(double t) { theVx_ = t; }

double TotemRPG4Hit::getVy() const { return theVy_; }
void TotemRPG4Hit::setVy(double t) { theVy_ = t; }

double TotemRPG4Hit::getVz() const { return theVz_; }
void TotemRPG4Hit::setVz(double t) { theVz_ = t; }

void TotemRPG4Hit::setPx(double p) { thePx_ = p; }
void TotemRPG4Hit::setPy(double p) { thePy_ = p; }
void TotemRPG4Hit::setPz(double p) { thePz_ = p; }

double TotemRPG4Hit::getPx() const { return thePx_; }
double TotemRPG4Hit::getPy() const { return thePy_; }
double TotemRPG4Hit::getPz() const { return thePz_; }

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
