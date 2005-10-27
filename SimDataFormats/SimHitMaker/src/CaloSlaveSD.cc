///////////////////////////////////////////////////////////////////////////////
// File: CaloSlaveSD.cc
// Date: 10.02
// Description: Persistent component of Sensitive Detector class for 
//              calorimeters
// Modifications: 
///////////////////////////////////////////////////////////////////////////////

#include "SimDataFormats/SimHitMaker/interface/CaloSlaveSD.h"

#include <iostream>
#define debug

CaloSlaveSD::CaloSlaveSD(std::string n) : name_(n) {

#ifdef debug
  std::cout << "CaloSlaveSD Called with name " << n << std::endl;
#endif
}

CaloSlaveSD::~CaloSlaveSD() { 
}

void CaloSlaveSD::Initialize() {

#ifdef debug
  std::cout << " initialize CaloSlaveSD "<< name_ << std::endl;
#endif
  hits_.clear();
}

bool CaloSlaveSD::format() {

#ifdef debug
  std::cout << " CaloSlaveSD " << name_ << "formatting " << hits_.size() << " hits."
       << std::endl;
#endif
  return true;
}

bool CaloSlaveSD::processHits(unsigned int unitID, double eDep, double tSlice, 
			      int tkID) {
  
  PCaloHit aCal = PCaloHit (unitID, eDep, tSlice, tkID);
#ifdef debug
  std::cout <<" Sent Hit " << aCal << " to ROU " << name_ << std::endl;
#endif
  hits_.push_back(aCal);
  return true;
} 
