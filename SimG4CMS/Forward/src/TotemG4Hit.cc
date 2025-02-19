// -*- C++ -*-
//
// Package:     Forward
// Class  :     TotemG4Hit
//
// Implementation:
//     <Notes on implementation>
//
// Original Author: 
//         Created:  Tue May 16 10:14:34 CEST 2006
// $Id: TotemG4Hit.cc,v 1.3 2007/11/20 12:37:21 fabiocos Exp $
//

// system include files

// user include files
#include "SimG4CMS/Forward/interface/TotemG4Hit.h"

//
// constructors and destructor
//

TotemG4Hit::TotemG4Hit(){

  setEntry(0.,0.,0.);
  theEntryPoint.SetCoordinates(0.,0.,0.);
  theExitPoint.SetCoordinates(0.,0.,0.);

  elem              = 0.;
  hadr              = 0.;
  theIncidentEnergy = 0.;
  theTrackID        = -1;
  theUnitID         =  0;
  theTimeSlice      = 0.;

  theX              = 0.;
  theY              = 0.;
  theZ              = 0.;
  thePabs           = 0.;
  theTof            = 0.;
  theEnergyLoss     = 0.;
  theParticleType   = 0;
  theThetaAtEntry   = 0.;
  thePhiAtEntry     = 0.;
  theParentId       = 0;
  theVx             = 0.;
  theVy             = 0.;
  theVz             = 0.;
}

TotemG4Hit::~TotemG4Hit() {}

TotemG4Hit::TotemG4Hit(const TotemG4Hit &right) {

  entry             = right.entry;
  elem              = right.elem;
  hadr              = right.hadr;
  theIncidentEnergy = right.theIncidentEnergy;
  theTrackID        = right.theTrackID;
  theUnitID         = right.theUnitID;
  theTimeSlice      = right.theTimeSlice;

  theX              = right.theX;
  theY              = right.theY;
  theZ              = right.theZ;
  thePabs           = right.thePabs;
  theTof            = right.theTof;
  theEnergyLoss     = right.theEnergyLoss;
  theParticleType   = right.theParticleType;

  theThetaAtEntry   = right.theThetaAtEntry;
  thePhiAtEntry     = right.thePhiAtEntry;
  theEntryPoint     = right.theEntryPoint;
  theExitPoint      = right.theExitPoint;
  theParentId       = right.theParentId;
  theVx             = right.theVx;
  theVy             = right.theVy;
  theVz             = right.theVz;
}


const TotemG4Hit& TotemG4Hit::operator=(const TotemG4Hit &right) {

  entry             = right.entry;
  elem              = right.elem;
  hadr              = right.hadr;
  theIncidentEnergy = right.theIncidentEnergy;
  theTrackID        = right.theTrackID;
  theUnitID         = right.theUnitID;
  theTimeSlice      = right.theTimeSlice;
 
  theX              = right.theX;
  theY              = right.theY;
  theZ              = right.theZ;
  thePabs           = right.thePabs;
  theTof            = right.theTof ;
  theEnergyLoss     = right.theEnergyLoss   ;
  theParticleType   = right.theParticleType ;

  theThetaAtEntry   = right.theThetaAtEntry;
  thePhiAtEntry     = right.thePhiAtEntry;
  theEntryPoint     = right.theEntryPoint;
  theExitPoint      = right.theExitPoint;
  theParentId       = right.theParentId;
  theVx             = right.theVx;
  theVy             = right.theVy;
  theVz             = right.theVz;

  return *this;
}

void TotemG4Hit::addEnergyDeposit(const TotemG4Hit& aHit) {

  elem += aHit.getEM();
  hadr += aHit.getHadr();
}


void TotemG4Hit::Print() {
  std::cout << (*this);
}


math::XYZPoint TotemG4Hit::getEntry() const           {return entry;}

double     TotemG4Hit::getEM() const              {return elem; }
void       TotemG4Hit::setEM (double e)           { elem     = e; }
      
double     TotemG4Hit::getHadr() const            {return hadr; }
void       TotemG4Hit::setHadr (double e)         { hadr     = e; }
      
double     TotemG4Hit::getIncidentEnergy() const  {return theIncidentEnergy; }
void       TotemG4Hit::setIncidentEnergy(double e) {theIncidentEnergy  = e; }

int        TotemG4Hit::getTrackID() const         {return theTrackID; }
void       TotemG4Hit::setTrackID (int i)         { theTrackID = i; }

uint32_t   TotemG4Hit::getUnitID() const          {return theUnitID; }
void       TotemG4Hit::setUnitID (uint32_t i)     { theUnitID = i; }

double     TotemG4Hit::getTimeSlice() const       {return theTimeSlice; }
void       TotemG4Hit::setTimeSlice (double d)    { theTimeSlice = d; }
int        TotemG4Hit::getTimeSliceID() const     {return (int)theTimeSlice;}

void       TotemG4Hit::addEnergyDeposit(double em, double hd) {elem += em;  hadr += hd;}

double     TotemG4Hit::getEnergyDeposit() const   {return elem+hadr;}

float      TotemG4Hit::getPabs() const            {return thePabs;}
float      TotemG4Hit::getTof() const             {return theTof;}
float      TotemG4Hit::getEnergyLoss() const      {return theEnergyLoss;}
int        TotemG4Hit::getParticleType() const    {return theParticleType;}

void       TotemG4Hit::setPabs(float e)           {thePabs = e;}
void       TotemG4Hit::setTof(float e)            {theTof = e;}
void       TotemG4Hit::setEnergyLoss(float e)     {theEnergyLoss = e;}
void       TotemG4Hit::setParticleType(short i)   {theParticleType = i;}

float      TotemG4Hit::getThetaAtEntry() const    {return theThetaAtEntry;}   
float      TotemG4Hit::getPhiAtEntry() const      {return thePhiAtEntry;}

void       TotemG4Hit::setThetaAtEntry(float t)   {theThetaAtEntry = t;}
void       TotemG4Hit::setPhiAtEntry(float f)     {thePhiAtEntry = f ;}

float      TotemG4Hit::getX() const               {return theX;}
void       TotemG4Hit::setX(float t)              {theX = t;}

float      TotemG4Hit::getY() const               {return theY;}
void       TotemG4Hit::setY(float t)              {theY = t;}

float      TotemG4Hit::getZ() const               {return theZ;}
void       TotemG4Hit::setZ(float t)              {theZ = t;}

int        TotemG4Hit::getParentId() const        {return theParentId;}
void       TotemG4Hit::setParentId(int p)         {theParentId = p;}

float      TotemG4Hit::getVx() const              {return theVx;}
void       TotemG4Hit::setVx(float t)             {theVx = t;}

float      TotemG4Hit::getVy() const              {return theVy;}
void       TotemG4Hit::setVy(float t)             {theVy = t;}

float      TotemG4Hit::getVz() const              {return theVz;}
void       TotemG4Hit::setVz(float t)             {theVz = t;}

std::ostream& operator<<(std::ostream& os, const TotemG4Hit& hit) {
  os << " Data of this TotemG4Hit are:\n" 
     << " Time slice ID: " << hit.getTimeSliceID() << "\n"
     << " EnergyDeposit = " << hit.getEnergyLoss() << "\n"
     << " Energy of primary particle (ID = " << hit.getTrackID()
     << ") = " << hit.getIncidentEnergy() << " (MeV)" << "\n"
     << " Entry point in Totem unit number " << hit.getUnitID()
     << " is: " << hit.getEntry() << " (mm)" << "\n"
     << "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n";
  return os;
}


