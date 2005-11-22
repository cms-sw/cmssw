///////////////////////////////////////////////////////////////////////////////
// File: CaloG4Hit.cc
// Description: Transient Hit class for the calorimeters
///////////////////////////////////////////////////////////////////////////////
#include "SimG4CMS/Calo/interface/CaloG4Hit.h"
#include <iostream>

CaloG4Hit::CaloG4Hit():entry(0), entryLocal(0) {

  elem     = 0.;
  hadr     = 0.;
  theIncidentEnergy = 0.;
}

CaloG4Hit::~CaloG4Hit(){}

CaloG4Hit::CaloG4Hit(const CaloG4Hit &right) {

  entry             = right.entry;
  entryLocal        = right.entryLocal;
  pos               = right.pos;
  elem              = right.elem;
  hadr              = right.hadr;
  theIncidentEnergy = right.theIncidentEnergy;
  hitID             = right.hitID;
}

const CaloG4Hit& CaloG4Hit::operator=(const CaloG4Hit &right) {

  entry             = right.entry;
  entryLocal        = right.entryLocal;
  pos               = right.pos;
  elem              = right.elem;
  hadr              = right.hadr;
  theIncidentEnergy = right.theIncidentEnergy;
  hitID             = right.hitID;
 
  return *this;
}

void CaloG4Hit::addEnergyDeposit(double em, double hd) {

  elem += em ;
  hadr += hd;
}

void CaloG4Hit::addEnergyDeposit(const CaloG4Hit& aHit) {

  addEnergyDeposit(aHit.getEM(),aHit.getHadr());
}


void CaloG4Hit::Print() {
  std::cout << (*this);
}

std::ostream& operator<<(std::ostream& os, const CaloG4Hit& hit) {
  os << " Data of this CaloG4Hit are:" << std::endl
     << " HitID: " << hit.getID() << std::endl
     << " EnergyDeposit of EM particles = " << hit.getEM() << std::endl
     << " EnergyDeposit of HD particles = " << hit.getHadr() << std::endl
     << " Energy of primary particle    = " << hit.getIncidentEnergy()/MeV 
     << " (MeV)"<< std::endl
     << " Entry point in Calorimeter (global) : " << hit.getEntry() 
     << "   (local) " << hit.getEntryLocal() << std::endl
     << " Position of Hit (global) : " << hit.getPosition() << std::endl
     << "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
     << std::endl;
  return os;
}


