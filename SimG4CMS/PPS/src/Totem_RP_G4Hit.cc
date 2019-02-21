///////////////////////////////////////////////////////////////////////////////
// File: CaloG4Hit.cc
// Date: 11.10.02
// Description: Transient Hit class for the calorimeters
///////////////////////////////////////////////////////////////////////////////
#include "SimG4CMS/PPS/interface/Totem_RP_G4Hit.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <iostream>

Totem_RP_G4Hit::Totem_RP_G4Hit():entry(0)
{
  theIncidentEnergy = 0.0;
  theTrackID = -1;
  theUnitID  =  0;
  theTimeSlice = 0.0;

  thePabs =0.0;
  theTof=0.0;
  theEnergyLoss=0.0;
  theParticleType=0;
  theX = 0.0;
  theY = 0.0;
  theZ = 0.0;
  theParentId=0;
  theVx = 0.0;
  theVy = 0.0;
  theVz = 0.0;
  p_x = p_y = p_z = 0.0;
}


Totem_RP_G4Hit::~Totem_RP_G4Hit()
{
}

Totem_RP_G4Hit::Totem_RP_G4Hit(const Totem_RP_G4Hit &right)
{
  theIncidentEnergy  = right.theIncidentEnergy;
  theTrackID = right.theTrackID;
  theUnitID = right.theUnitID;
  theTimeSlice = right.theTimeSlice;
  entry    = right.entry;

  thePabs =right.thePabs;
  theTof=right.theTof ;
  theEnergyLoss=right.theEnergyLoss   ;
  theParticleType=right.theParticleType ;
  theX = right.theX;
  theY = right.theY;
  theZ = right.theZ;

  theVx = right.theVx;
  theVy = right.theVy;
  theVz = right.theVz;

  theParentId = right.theParentId;
}


const Totem_RP_G4Hit& Totem_RP_G4Hit::operator=(const Totem_RP_G4Hit &right)
{
  theIncidentEnergy = right.theIncidentEnergy;
  theTrackID = right.theTrackID;
  theUnitID = right.theUnitID;
  theTimeSlice = right.theTimeSlice;
  entry = right.entry;
 
  thePabs =right.thePabs;
  theTof=right.theTof ;
  theEnergyLoss=right.theEnergyLoss   ;
  theParticleType=right.theParticleType ;
  theX = right.theX;
  theY = right.theY;
  theZ = right.theZ;

  theVx = right.theVx;
  theVy = right.theVy;
  theVz = right.theVz;

  theParentId = right.theParentId;
  
  return *this;
}


void Totem_RP_G4Hit::Print() {
  edm::LogInfo("TotemRP") << (*this);
}


Hep3Vector Totem_RP_G4Hit::getEntry() const {return entry;}
void Totem_RP_G4Hit::setEntry(Hep3Vector xyz) { entry = xyz; }

Hep3Vector Totem_RP_G4Hit::getExit() const {return exit;}
void Totem_RP_G4Hit::setExit(Hep3Vector xyz) { exit = xyz; }

Hep3Vector Totem_RP_G4Hit::getLocalEntry() const {return local_entry;}
void Totem_RP_G4Hit::setLocalEntry(const Hep3Vector &xyz) { local_entry = xyz;}
Hep3Vector Totem_RP_G4Hit::getLocalExit() const {return local_exit;}
void Totem_RP_G4Hit::setLocalExit(const Hep3Vector &xyz) { local_exit = xyz;}

double Totem_RP_G4Hit::getIncidentEnergy() const {return theIncidentEnergy; }
void Totem_RP_G4Hit::setIncidentEnergy (double e) {theIncidentEnergy  = e; }

unsigned int Totem_RP_G4Hit::getTrackID() const {return theTrackID; }
void Totem_RP_G4Hit::setTrackID (int i) { theTrackID = i; }

int Totem_RP_G4Hit::getUnitID() const {return theUnitID; }
void Totem_RP_G4Hit::setUnitID (unsigned int i) { theUnitID = i; }

double Totem_RP_G4Hit::getTimeSlice() const {return theTimeSlice; }
void Totem_RP_G4Hit::setTimeSlice (double d) {theTimeSlice = d;}
int Totem_RP_G4Hit::getTimeSliceID() const {return (int)theTimeSlice;}

double Totem_RP_G4Hit::getPabs() const {return thePabs;}
double Totem_RP_G4Hit::getTof() const {return theTof;}
double Totem_RP_G4Hit::getEnergyLoss() const {return theEnergyLoss;}
int Totem_RP_G4Hit::getParticleType() const {return theParticleType;}

void Totem_RP_G4Hit::setPabs(double e) {thePabs = e;}
void Totem_RP_G4Hit::setTof(double e) {theTof = e;}
void Totem_RP_G4Hit::setEnergyLoss(double e) {theEnergyLoss = e;}
void Totem_RP_G4Hit::addEnergyLoss(double e) {theEnergyLoss += e;}
void Totem_RP_G4Hit::setParticleType(short i) {theParticleType = i;}

double Totem_RP_G4Hit::getThetaAtEntry() const {return theThetaAtEntry;}   
double Totem_RP_G4Hit::getPhiAtEntry() const{ return thePhiAtEntry;}

void Totem_RP_G4Hit::setThetaAtEntry(double t) {theThetaAtEntry = t;}
void Totem_RP_G4Hit::setPhiAtEntry(double f) {thePhiAtEntry = f ;}

double Totem_RP_G4Hit::getX() const{ return theX;}
void Totem_RP_G4Hit::setX(double t){theX = t;}

double Totem_RP_G4Hit::getY() const{ return theY;}
void Totem_RP_G4Hit::setY(double t){theY = t;}

double Totem_RP_G4Hit::getZ() const{ return theZ;}
void Totem_RP_G4Hit::setZ(double t){theZ = t;}

int Totem_RP_G4Hit::getParentId() const {return theParentId;}
void Totem_RP_G4Hit::setParentId(int p){theParentId = p;}

double Totem_RP_G4Hit::getVx() const{ return theVx;}
void Totem_RP_G4Hit::setVx(double t){theVx = t;}

double Totem_RP_G4Hit::getVy() const{ return theVy;}
void Totem_RP_G4Hit::setVy(double t){theVy = t;}

double Totem_RP_G4Hit::getVz() const{ return theVz;}
void Totem_RP_G4Hit::setVz(double t){theVz = t;}

void Totem_RP_G4Hit::set_p_x(double p) {p_x = p;}
void Totem_RP_G4Hit::set_p_y(double p) {p_y = p;}
void Totem_RP_G4Hit::set_p_z(double p) {p_z = p;}
  
double Totem_RP_G4Hit::get_p_x() const {return p_x;}
double Totem_RP_G4Hit::get_p_y() const {return p_y;}
double Totem_RP_G4Hit::get_p_z() const {return p_z;}


std::ostream& operator<<(std::ostream& os, const Totem_RP_G4Hit& hit) {
  os << " Data of this Totem_RP_G4Hit are:" << std::endl
     << " Time slice ID: " << hit.getTimeSliceID() << std::endl
     << " EnergyDeposit = " << hit.getEnergyLoss() << std::endl
  //    << " EnergyDeposit of HD particles = " << hit.getHadr() << std::endl
     << " Energy of primary particle (ID = " << hit.getTrackID()
     << ") = " << hit.getIncidentEnergy() << " (MeV)"<<std::endl
     << " Entry point in Totem unit number " << hit.getUnitID()
     << " is: " << hit.getEntry() << " (mm)" << std::endl;
  os << "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
     << std::endl;
  return os;
}
