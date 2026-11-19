///////////////////////////////////////////////////////////////////////////////
// File: CaloSlaveSD.cc
// Date: 10.02
// Description: Persistent component of Sensitive Detector class for
//              calorimeters
// Modifications:
///////////////////////////////////////////////////////////////////////////////

#include "SimDataFormats/SimHitMaker/interface/CaloSlaveSD.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <iostream>

//#define EDM_ML_DEBUG

CaloSlaveSD::CaloSlaveSD(std::string n) : name_(n) {
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HitBuildInfo") << "CaloSlaveSD Called with name " << n << "\n";
#endif
}

CaloSlaveSD::~CaloSlaveSD() {}

void CaloSlaveSD::Initialize() {
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HitBuildInfo") << " initialize CaloSlaveSD " << name_ << "\n";
#endif
  hits_.clear();
}

bool CaloSlaveSD::format() {
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HitBuildInfo") << " CaloSlaveSD " << name_ << "formatting " << hits_.size() << " hits.";
#endif
  return true;
}

bool CaloSlaveSD::processHits(uint32_t unitID, double eDepEM, double eDepHad, double tSlice, int tkID, uint16_t depth) {
  PCaloHit aCal = PCaloHit(unitID, eDepEM, eDepHad, tSlice, tkID, depth);
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HitBuildInfo") << " Sent Hit " << aCal << " to ROU " << name_;
#endif
  hits_.push_back(aCal);
  return true;
}

void CaloSlaveSD::Clean() {
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HitBuildIndo") << "CaloSlaveSD " << name_ << " cleaning the collection";
#endif
  Collection().swap(hits_);
}

void CaloSlaveSD::ReserveMemory(unsigned int size) {
  if (hits_.capacity() < size) {
    hits_.reserve(size);
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HitBuildIndo") << "CaloSlaveSD Reserve " << size << " words for " << name_;
#endif
  }
}
