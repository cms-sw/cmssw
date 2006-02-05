///////////////////////////////////////////////////////////////////////////////
// File: EcalEndcapNumberingScheme.cc
// Description: Numbering scheme for endcap electromagnetic calorimeter
///////////////////////////////////////////////////////////////////////////////
#include "SimG4CMS/Calo/interface/EcalEndcapNumberingScheme.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"

#include "CLHEP/Units/SystemOfUnits.h"
#include <iostream>

EcalEndcapNumberingScheme::EcalEndcapNumberingScheme(int iv) : 
  EcalNumberingScheme(iv) {
  if (verbosity>0) 
    std::cout << "Creating EcalEndcapNumberingScheme" << std::endl;
}

EcalEndcapNumberingScheme::~EcalEndcapNumberingScheme() {
  if (verbosity>0) 
    std::cout << "Deleting EcalEndcapNumberingScheme" << std::endl;
}

uint32_t EcalEndcapNumberingScheme::getUnitID(const G4Step* aStep) const {

  G4StepPoint* PreStepPoint = aStep->GetPreStepPoint(); 
  const G4VTouchable* touch = aStep->GetPreStepPoint()->GetTouchable();
  G4ThreeVector HitPoint    = PreStepPoint->GetPosition();

  int PVid = touch->GetReplicaNumber(0);
  int MVid = 1;
  if (touch->GetHistoryDepth() > 0) 
    MVid = touch->GetReplicaNumber(1);
  else 
    if (verbosity>0) 
      std::cout << "ECalEndcapNumberingScheme::getUnitID: Null pointer to "
		<< "alveole ! Use default id=1 " << std::endl;

  int zside= HitPoint.z()>0 ? 1: -1;
  int module_number  = MVid;
  int crystal_number  = PVid;

  //pack it into an integer
  // to be consistent with EEDetId definition
  //FIXME to checked and most surely corrected
  uint32_t intindex = EEDetId(module_number,crystal_number,zside,EEDetId::SCCRYSTALMODE).rawId();

  if (verbosity>1) 
    std::cout << "EcalEndcapNumberingScheme: zside = "  << zside 
	      << " super crystal = " << module_number << " crystal = " << crystal_number
	      << " packed index = 0x" << std::hex << intindex << std::dec 
	      << std::endl;
  return intindex;

}

float EcalEndcapNumberingScheme::energyInMatrix(int nCellInEta, int nCellInPhi,
						int centralEta, int centralPhi,
						int centralZ, MapType& themap){

  int   ncristals   = 0;
  float totalEnergy = 0.;
        
  int goBackInEta = nCellInEta/2;
  int goBackInPhi = nCellInPhi/2;
  int startEta    = centralEta-goBackInEta;
  int startPhi    = centralPhi-goBackInPhi;
  
  for (int ieta=startEta; ieta<startEta+nCellInEta; ieta++) {
    for (int iphi=startPhi; iphi<startPhi+nCellInPhi; iphi++) {
      
      uint32_t index = EEDetId(ieta,iphi,centralZ).rawId();
      totalEnergy   += themap[index];
      ncristals     += 1;
      if (verbosity > 2)
	std::cout << "EcalEndcapNumberingScheme: ieta - iphi - E = " << ieta 
		  << "  " << iphi << " "  << themap[index] << std::endl;
    }
  }
        
  if (verbosity > 1)
    std::cout << "EcalEndcapNumberingScheme: energy in " << nCellInEta 
	      << " cells in eta times " << nCellInPhi 
	      << " cells in phi matrix = " << totalEnergy
	      << " for " << ncristals << " crystals" << std::endl;
  return totalEnergy;

}   
