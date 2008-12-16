///////////////////////////////////////////////////////////////////////////////
// File: CaloSlaveSD.cc
// Date: 10.02
// Description: Persistent component of Sensitive Detector class for 
//              calorimeters
// Modifications: 
///////////////////////////////////////////////////////////////////////////////

#include "SimDataFormats/SimHitMaker/interface/CaloSlaveSD.h"

#include <iostream>

CaloSlaveSD::CaloSlaveSD(std::string n) : name_(n) {

  LogDebug("HitBuildInfo") << "CaloSlaveSD Called with name " << n << "\n";

}

CaloSlaveSD::~CaloSlaveSD() {}

void CaloSlaveSD::Initialize() {

  LogDebug("HitBuildInfo") << " initialize CaloSlaveSD "<< name_ << "\n";
  hits_.clear();
}

bool CaloSlaveSD::format() {

  LogDebug("HitBuildInfo") << " CaloSlaveSD " << name_ << "formatting " 
			   << hits_.size() << " hits.";
  return true;
}

bool CaloSlaveSD::processHits(uint32_t unitID, double eDepEM, double eDepHad, 
			      double tSlice, int tkID, uint16_t depth) {
  
  PCaloHit aCal = PCaloHit (unitID, eDepEM, eDepHad, tSlice, tkID, depth);
  LogDebug("HitBuildInfo") <<" Sent Hit " << aCal << " to ROU " << name_;
  hits_.push_back(aCal);
  return true;
} 

void CaloSlaveSD::Clean() {

  LogDebug("HitBuildIndo") << "CaloSlaveSD " << name_ << " cleaning the collection";
  Collection().swap(hits_);

}

void CaloSlaveSD::ReserveMemory(unsigned int size) {
  
  if ( hits_.capacity() < size ) hits_.reserve(size);

}
