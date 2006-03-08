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

CaloSlaveSD::~CaloSlaveSD() { 
}

void CaloSlaveSD::Initialize() {

  LogDebug("HitBuildInfo") << " initialize CaloSlaveSD "<< name_ << "\n";
  hits_.clear();
}

bool CaloSlaveSD::format() {

  LogDebug("HitBuildInfo") << " CaloSlaveSD " << name_ << "formatting " << hits_.size() << " hits." << "\n";
  return true;
}

bool CaloSlaveSD::processHits(unsigned int unitID, double eDep, double tSlice, 
			      int tkID) {
  
  PCaloHit aCal = PCaloHit (unitID, eDep, tSlice, tkID);
  LogDebug("HitBuildInfo") <<" Sent Hit " << aCal << " to ROU " << name_ << "\n";
  hits_.push_back(aCal);
  return true;
} 
