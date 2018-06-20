///////////////////////////////////////////////////////////////////////////////
// File: BscG4Hit.cc
// Date: 02.2006
// Description: Transient Hit class for the Bsc
///////////////////////////////////////////////////////////////////////////////
#include "SimG4CMS/Forward/interface/BscG4Hit.h"
#include <iostream>

BscG4Hit::BscG4Hit():entry(0) {

  entrylp(0);
  exitlp(0);
  elem     = 0.;
  hadr     = 0.;
  theIncidentEnergy = 0.;
  theTimeSlice = 0.;
  theTrackID = -1;
  theUnitID  =  0;
  thePabs =0.;
  theTof=0. ;
  theEnergyLoss=0.   ;
  theParticleType=0 ;
  theUnitID=0;
  theTrackID=-1;
  theThetaAtEntry=-10000. ;
  thePhiAtEntry=-10000. ;
  theParentId=0;
  
  theX = 0.;
  theY = 0.;
  theZ = 0.;
  theVx = 0.;
  theVy = 0.;
  theVz = 0.;
}


BscG4Hit::~BscG4Hit(){}


BscG4Hit::BscG4Hit(const BscG4Hit &right) {
  theUnitID         = right.theUnitID;
  
  theTrackID        = right.theTrackID;
  theTof            = right.theTof ;
  theEnergyLoss      = right.theEnergyLoss   ;
  theParticleType    = right.theParticleType ;
  thePabs            = right.thePabs;
  elem               = right.elem;
  hadr               = right.hadr;
  theIncidentEnergy  = right.theIncidentEnergy;
  theTimeSlice       = right.theTimeSlice;
  entry              = right.entry;
  entrylp            = right.entrylp;
  exitlp             = right.exitlp;
  theThetaAtEntry    = right.theThetaAtEntry;
  thePhiAtEntry      = right.thePhiAtEntry;
  theParentId        = right.theParentId;
  
  theX = right.theX;
  theY = right.theY;
  theZ = right.theZ;
  
  theVx = right.theVx;
  theVy = right.theVy;
  theVz = right.theVz;
  
  
}


const BscG4Hit& BscG4Hit::operator=(const BscG4Hit &right) {
  theUnitID         = right.theUnitID;
  
  theTrackID        = right.theTrackID;
  theTof            = right.theTof ;
  theEnergyLoss      = right.theEnergyLoss   ;
  theParticleType    = right.theParticleType ;
  thePabs            = right.thePabs;
  elem               = right.elem;
  hadr               = right.hadr;
  theIncidentEnergy  = right.theIncidentEnergy;
  theTimeSlice       = right.theTimeSlice;
  entry              = right.entry;
  entrylp            = right.entrylp;
  exitlp             = right.exitlp;
  theThetaAtEntry    = right.theThetaAtEntry;
  thePhiAtEntry      = right.thePhiAtEntry;
  theParentId        = right.theParentId;
  
  theX = right.theX;
  theY = right.theY;
  theZ = right.theZ;
  
  theVx = right.theVx;
  theVy = right.theVy;
  theVz = right.theVz;
  
  
  return *this;
}


void BscG4Hit::addEnergyDeposit(const BscG4Hit& aHit) {

  elem += aHit.getEM();
  hadr += aHit.getHadr();
}


void BscG4Hit::Print() {
  std::cout << (*this);
}

G4ThreeVector   BscG4Hit::getEntry() const           {return entry;}
void         BscG4Hit::setEntry(const G4ThreeVector& xyz)   { entry    = xyz; }

G4ThreeVector    BscG4Hit::getEntryLocalP() const           {return entrylp;}
void         BscG4Hit::setEntryLocalP(const G4ThreeVector& xyz1)   { entrylp    = xyz1; }

G4ThreeVector     BscG4Hit::getExitLocalP() const           {return exitlp;}
void         BscG4Hit::setExitLocalP(const G4ThreeVector& xyz1)   { exitlp    = xyz1; }

float        BscG4Hit::getEM() const              {return elem; }
void         BscG4Hit::setEM (float e)            { elem     = e; }
      
float        BscG4Hit::getHadr() const            {return hadr; }
void         BscG4Hit::setHadr (float e)          { hadr     = e; }
      
float        BscG4Hit::getIncidentEnergy() const  {return theIncidentEnergy; }
void         BscG4Hit::setIncidentEnergy (float e){theIncidentEnergy  = e; }

int          BscG4Hit::getTrackID() const         {return theTrackID; }
void         BscG4Hit::setTrackID (int i)         { theTrackID = i; }

uint32_t     BscG4Hit::getUnitID() const          {return theUnitID; }
void         BscG4Hit::setUnitID (uint32_t i)     { theUnitID = i; }

double       BscG4Hit::getTimeSlice() const       {return theTimeSlice; }
void         BscG4Hit::setTimeSlice (double d)    { theTimeSlice = d; }
int          BscG4Hit::getTimeSliceID() const     {return (int)theTimeSlice;}

void         BscG4Hit::addEnergyDeposit(float em, float hd)
{elem  += em; hadr += hd; theEnergyLoss += em + hd;}

float        BscG4Hit::getEnergyDeposit() const   {return elem+hadr;}

float BscG4Hit::getPabs() const {return thePabs;}
float BscG4Hit::getTof() const {return theTof;}
float BscG4Hit::getEnergyLoss() const {return theEnergyLoss;}
int BscG4Hit::getParticleType() const {return theParticleType;}

void BscG4Hit::setPabs(float e) {thePabs = e;}
void BscG4Hit::setTof(float e) {theTof = e;}
void BscG4Hit::setEnergyLoss(float e) {theEnergyLoss = e;}
void BscG4Hit::setParticleType(int i) {theParticleType = i;}

float BscG4Hit::getThetaAtEntry() const {return theThetaAtEntry;}   
float BscG4Hit::getPhiAtEntry() const{ return thePhiAtEntry;}

void BscG4Hit::setThetaAtEntry(float t){theThetaAtEntry = t;}
void BscG4Hit::setPhiAtEntry(float f) {thePhiAtEntry = f ;}

float BscG4Hit::getX() const{ return theX;}
void BscG4Hit::setX(float t){theX = t;}

float BscG4Hit::getY() const{ return theY;}
void BscG4Hit::setY(float t){theY = t;}

float BscG4Hit::getZ() const{ return theZ;}
void BscG4Hit::setZ(float t){theZ = t;}

int BscG4Hit::getParentId() const {return theParentId;}
void BscG4Hit::setParentId(int p){theParentId = p;}

float BscG4Hit::getVx() const{ return theVx;}
void BscG4Hit::setVx(float t){theVx = t;}

float BscG4Hit::getVy() const{ return theVy;}
void BscG4Hit::setVy(float t){theVy = t;}

float BscG4Hit::getVz() const{ return theVz;}
void BscG4Hit::setVz(float t){theVz = t;}

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
     << " Energy of primary particle (ID = " << hit.getTrackID()
     << ") = " << hit.getIncidentEnergy() << " (MeV)"<<std::endl
     << " Entry point in Bsc unit number " << hit.getUnitID()
     << " is: " << hit.getEntry() << " (mm)" << std::endl;
  os << "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
     << std::endl;
  return os;

}


