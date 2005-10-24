///////////////////////////////////////////////////////////////////////////////
// File: EcalEndcapNumberingScheme.cc
// Description: Numbering scheme for endcap electromagnetic calorimeter
///////////////////////////////////////////////////////////////////////////////
#include "SimG4CMS/Calo/interface/EcalEndcapNumberingScheme.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"

#include "CLHEP/Units/SystemOfUnits.h"
#include <iostream>

//#define debug


EcalEndcapNumberingScheme::EcalEndcapNumberingScheme() {
  std::cout << "Creating EcalEndcapNumberingScheme" << std::endl;
}

EcalEndcapNumberingScheme::~EcalEndcapNumberingScheme() {
  std::cout << "Deleting EcalEndcapNumberingScheme" << std::endl;
}

unsigned int EcalEndcapNumberingScheme::getUnitID(const G4Step* aStep) const {

  G4StepPoint* PreStepPoint = aStep->GetPreStepPoint(); 
  const G4VTouchable* touch = aStep->GetPreStepPoint()->GetTouchable();
  G4ThreeVector HitPoint    = PreStepPoint->GetPosition();

  int PVid = touch->GetReplicaNumber(0);
  int MVid = 1;
  if (touch->GetHistoryDepth() > 0) 
    MVid = touch->GetReplicaNumber(1);
  else 
    std::cout << "ECalEndcapNumberingScheme::getUnitID: Null pointer to "
		 << "alveole ! Use default id=1 " << std::endl;

  int zside= HitPoint.z()>0 ? 1: -1;
  int eta  = MVid;
  int phi  = PVid;

  //pack it into an integer
  // to be consistent with EEDetId definition
  unsigned int intindex = EEDetId(1,1,zside).rawId();
#ifdef debug
  std::cout << "EcalEndcapNumberingScheme: det = " << det << " zside = " 
	       << zside << " super crystal = " << eta << " crystal = " << phi
	       << " packed index = 0x" << hex << intindex << dec << std::endl;
#endif
  return intindex;

}
