#include "SimG4CMS/Calo/interface/CaloGVHit.h"
#include <iostream>

#include "G4SystemOfUnits.hh"

CaloGVHit::CaloGVHit(){
  elem     = 0.;
  hadr     = 0.;
}

CaloGVHit::~CaloGVHit(){}

CaloGVHit::CaloGVHit(const CaloGVHit &right) {
  elem              = right.elem;
  hadr              = right.hadr;
  hitID             = right.hitID;
}

const CaloGVHit& CaloGVHit::operator=(const CaloGVHit &right) {
  elem              = right.elem;
  hadr              = right.hadr;
  hitID             = right.hitID;
  return *this;
}

void CaloGVHit::addEnergyDeposit(double em, double hd) {
  elem += em ;
  hadr += hd;
}

void CaloGVHit::addEnergyDeposit(const CaloGVHit& aHit) {

  addEnergyDeposit(aHit.getEM(),aHit.getHadr());
}

std::ostream& operator<<(std::ostream& os, const CaloGVHit& hit) {
  os << " Data of this CaloGVHit are:" << "\n"
     << " HitID: " << hit.getID() << "\n"
     << " EnergyDeposit of EM particles = " << hit.getEM() << "\n"
     << " EnergyDeposit of HD particles = " << hit.getHadr() << "\n"
     << "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n";
  return os;
}
