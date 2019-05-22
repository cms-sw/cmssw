#include "SimG4CMS/Calo/interface/CaloGVHit.h"
#include <iostream>

#include "G4SystemOfUnits.hh"

CaloGVHit::CaloGVHit() {
  eventID_ = 0;
  elem_ = 0.;
  hadr_ = 0.;
}

CaloGVHit::~CaloGVHit() {}

CaloGVHit::CaloGVHit(const CaloGVHit& right) {
  eventID_ = right.eventID_;
  elem_ = right.elem_;
  hadr_ = right.hadr_;
  hitID_ = right.hitID_;
}

const CaloGVHit& CaloGVHit::operator=(const CaloGVHit& right) {
  eventID_ = right.eventID_;
  elem_ = right.elem_;
  hadr_ = right.hadr_;
  hitID_ = right.hitID_;
  return *this;
}

void CaloGVHit::addEnergyDeposit(double em, double hd) {
  elem_ += em;
  hadr_ += hd;
}

void CaloGVHit::addEnergyDeposit(const CaloGVHit& aHit) { addEnergyDeposit(aHit.getEM(), aHit.getHadr()); }

std::ostream& operator<<(std::ostream& os, const CaloGVHit& hit) {
  os << " Data of this CaloGVHit are:"
     << " EventID: " << hit.getEventID() << " HitID: " << hit.getID() << " EnergyDeposit (EM): " << hit.getEM()
     << " (Had): " << hit.getHadr() << "\n";
  return os;
}
