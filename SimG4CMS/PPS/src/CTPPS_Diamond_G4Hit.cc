///////////////////////////////////////////////////////////////////////////////
// Author
//Seyed Mohsen Etesami setesami@cern.ch
//
///////////////////////////////////////////////////////////////////////////////
#include "SimG4CMS/PPS/interface/CTPPS_Diamond_G4Hit.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <iostream>

CTPPS_Diamond_G4Hit::CTPPS_Diamond_G4Hit():entry(0),exit(0),local_entry(0),local_exit(0)
{
  theIncidentEnergy = 0.0;
  theTrackID = -1;
  theUnitID  =  0;
  theTimeSlice = 0.0;
  theGlobaltimehit=0.0;
  theX = 0.0;
  theY = 0.0;
  theZ = 0.0;
  thePabs =0.0;
  theTof=0.0;
  theEnergyLoss=0.0;
  theParticleType=0;
  theParentId=0;
  theVx = 0.0;
  theVy = 0.0;
  theVz = 0.0;
  p_x = p_y = p_z = 0.0;
  theThetaAtEntry=0;
  thePhiAtEntry=0;
}


CTPPS_Diamond_G4Hit::~CTPPS_Diamond_G4Hit()
{
}

CTPPS_Diamond_G4Hit::CTPPS_Diamond_G4Hit(const CTPPS_Diamond_G4Hit &right)
{
  entry              = right.entry;
  exit               = right.exit;
  local_entry        = right.local_entry;
  local_exit         = right.local_exit;
  theIncidentEnergy  = right.theIncidentEnergy;
  theTrackID         = right.theTrackID;
  theUnitID          = right.theUnitID;
  theTimeSlice       = right.theTimeSlice;
  theGlobaltimehit   = right.theGlobaltimehit;
  theX               = right.theX;
  theY               = right.theY;
  theZ               = right.theZ;
  thePabs            = right.thePabs;
  theTof             = right.theTof ;
  theEnergyLoss      = right.theEnergyLoss   ;
  theParticleType    = right.theParticleType ;
  theParentId        = right.theParentId;
  theVx              = right.theVx;
  theVy              = right.theVy;
  theVz              = right.theVz;
  p_x                = right.p_x;
  p_y 	       	     = right.p_y;
  p_z  	       	     = right.p_z;
  theThetaAtEntry    = right.theThetaAtEntry;
  thePhiAtEntry      = right.thePhiAtEntry;
}


const CTPPS_Diamond_G4Hit& CTPPS_Diamond_G4Hit::operator=(const CTPPS_Diamond_G4Hit &right)
{

  entry              = right.entry;
  exit               = right.exit;
  local_entry        = right.local_entry;
  local_exit         = right.local_exit;
  theIncidentEnergy  = right.theIncidentEnergy;
  theTrackID         = right.theTrackID;
  theUnitID          = right.theUnitID;
  theTimeSlice       = right.theTimeSlice;
  theGlobaltimehit   = right.theGlobaltimehit;
  theX               = right.theX;
  theY               = right.theY;
  theZ               = right.theZ;
  thePabs            = right.thePabs;
  theTof             = right.theTof ;
  theEnergyLoss      = right.theEnergyLoss   ;
  theParticleType    = right.theParticleType ;
  theParentId        = right.theParentId;
  theVx              = right.theVx;
  theVy              = right.theVy;
  theVz              = right.theVz;
  p_x                = right.p_x;
  p_y                = right.p_y;
  p_z                = right.p_z;
  theThetaAtEntry    = right.theThetaAtEntry;
  thePhiAtEntry      = right.thePhiAtEntry;
  
  return *this;
}


void CTPPS_Diamond_G4Hit::Print() 
{
  edm::LogInfo("CTPPSSimDiamond") << (*this);
}


Hep3Vector CTPPS_Diamond_G4Hit::getEntry() const {return entry;}
void CTPPS_Diamond_G4Hit::setEntry(Hep3Vector xyz) { entry = xyz; }

Hep3Vector CTPPS_Diamond_G4Hit::getExit() const {return exit;}
void CTPPS_Diamond_G4Hit::setExit(Hep3Vector xyz) { exit = xyz; }

Hep3Vector CTPPS_Diamond_G4Hit::getLocalEntry() const {return local_entry;}
void CTPPS_Diamond_G4Hit::setLocalEntry(const Hep3Vector &xyz) { local_entry = xyz;}
Hep3Vector CTPPS_Diamond_G4Hit::getLocalExit() const {return local_exit;}
void CTPPS_Diamond_G4Hit::setLocalExit(const Hep3Vector &xyz) { local_exit = xyz;}

double CTPPS_Diamond_G4Hit::getIncidentEnergy() const {return theIncidentEnergy; }
void CTPPS_Diamond_G4Hit::setIncidentEnergy (double e) {theIncidentEnergy  = e; }

unsigned int CTPPS_Diamond_G4Hit::getTrackID() const {return theTrackID; }
void CTPPS_Diamond_G4Hit::setTrackID (int i) { theTrackID = i; }

int CTPPS_Diamond_G4Hit::getUnitID() const {return theUnitID; }
void CTPPS_Diamond_G4Hit::setUnitID (unsigned int i) { theUnitID = i; }

double CTPPS_Diamond_G4Hit::getTimeSlice() const {return theTimeSlice; }
void CTPPS_Diamond_G4Hit::setTimeSlice (double d) {theTimeSlice = d;}
int CTPPS_Diamond_G4Hit::getTimeSliceID() const {return (int)theTimeSlice;}

double CTPPS_Diamond_G4Hit::getPabs() const {return thePabs;}
double CTPPS_Diamond_G4Hit::getTof() const {return theTof;}
double CTPPS_Diamond_G4Hit::getEnergyLoss() const {return theEnergyLoss;}
int CTPPS_Diamond_G4Hit::getParticleType() const {return theParticleType;}

void CTPPS_Diamond_G4Hit::setPabs(double e) {thePabs = e;}
void CTPPS_Diamond_G4Hit::setTof(double e) {theTof = e;}
void CTPPS_Diamond_G4Hit::setEnergyLoss(double e) {theEnergyLoss = e;}
void CTPPS_Diamond_G4Hit::addEnergyLoss(double e) {theEnergyLoss += e;}
void CTPPS_Diamond_G4Hit::setParticleType(short i) {theParticleType = i;}

double CTPPS_Diamond_G4Hit::getThetaAtEntry() const {return theThetaAtEntry;}   
double CTPPS_Diamond_G4Hit::getPhiAtEntry() const{ return thePhiAtEntry;}

void CTPPS_Diamond_G4Hit::setThetaAtEntry(double t) {theThetaAtEntry = t;}
void CTPPS_Diamond_G4Hit::setPhiAtEntry(double f) {thePhiAtEntry = f ;}

double CTPPS_Diamond_G4Hit::getX() const{ return theX;}
void CTPPS_Diamond_G4Hit::setX(double t){theX = t;}

double CTPPS_Diamond_G4Hit::getY() const{ return theY;}
void CTPPS_Diamond_G4Hit::setY(double t){theY = t;}

double CTPPS_Diamond_G4Hit::getZ() const{ return theZ;}
void CTPPS_Diamond_G4Hit::setZ(double t){theZ = t;}

int CTPPS_Diamond_G4Hit::getParentId() const {return theParentId;}
void CTPPS_Diamond_G4Hit::setParentId(int p){theParentId = p;}

double CTPPS_Diamond_G4Hit::getVx() const{ return theVx;}
void CTPPS_Diamond_G4Hit::setVx(double t){theVx = t;}

double CTPPS_Diamond_G4Hit::getVy() const{ return theVy;}
void CTPPS_Diamond_G4Hit::setVy(double t){theVy = t;}

double CTPPS_Diamond_G4Hit::getVz() const{ return theVz;}
void CTPPS_Diamond_G4Hit::setVz(double t){theVz = t;}

void CTPPS_Diamond_G4Hit::set_p_x(double p) {p_x = p;}
void CTPPS_Diamond_G4Hit::set_p_y(double p) {p_y = p;}
void CTPPS_Diamond_G4Hit::set_p_z(double p) {p_z = p;}
  
double CTPPS_Diamond_G4Hit::get_p_x() const {return p_x;}
double CTPPS_Diamond_G4Hit::get_p_y() const {return p_y;}
double CTPPS_Diamond_G4Hit::get_p_z() const {return p_z;}


double CTPPS_Diamond_G4Hit::getGlobalTimehit() const {return theGlobaltimehit; }
void CTPPS_Diamond_G4Hit::setGlobalTimehit(double h) {theGlobaltimehit = h;}
 

std::ostream& operator<<(std::ostream& os, const CTPPS_Diamond_G4Hit& hit) {
  os << " Data of this CTPPS_Diamond_G4Hit are:" << std::endl
     << " Time slice ID: " << hit.getTimeSliceID() << std::endl
     << " EnergyDeposit = " << hit.getEnergyLoss() << std::endl
     << " Energy of primary particle (ID = " << hit.getTrackID()
     << ") = " << hit.getIncidentEnergy() << " (MeV)" << "\n"
     << " Local entry and exit points in PPS unit number " << hit.getUnitID()
     << " are: " << hit.getEntry() << " (mm)" << hit.getExit() << " (mm)" <<"\n"
     << "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
     << std::endl;
  return os;
}
