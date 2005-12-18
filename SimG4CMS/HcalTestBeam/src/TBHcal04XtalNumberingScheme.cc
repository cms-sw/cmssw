///////////////////////////////////////////////////////////////////////////////
// File: TBHcal04XtalNumberingScheme.cc
// Description: Numbering scheme for 2004 crystal calorimeter
///////////////////////////////////////////////////////////////////////////////
#include "SimG4CMS/HcalTestBeam/interface/TBHcal04XtalNumberingScheme.h"
#include "SimG4CMS/Calo/interface/HcalTestNumberingScheme.h"

using namespace std;

TBHcal04XtalNumberingScheme::TBHcal04XtalNumberingScheme(int iv) : 
  EcalNumberingScheme(iv) {
  if (verbosity>0) 
    std::cout << "Creating TBHcal04XtalNumberingScheme" << std::endl;
}

TBHcal04XtalNumberingScheme::~TBHcal04XtalNumberingScheme() {
  if (verbosity>0) 
    std::cout << "Deleting TBHcal04XtalNumberingScheme" << std::endl;
}

uint32_t TBHcal04XtalNumberingScheme::getUnitID(const G4Step* aStep) const {

  const G4VTouchable* touch = aStep->GetPreStepPoint()->GetTouchable();
  int idx = touch->GetReplicaNumber(0);
  int idl = 0;
  if (touch->GetHistoryDepth() > 0) idl = touch->GetReplicaNumber(1);
  int  det = 10;
  uint32_t idunit = HcalTestNumberingScheme::packHcalIndex(det,0,1,idl,idx,1);

  if (verbosity>1) 
    std::cout << "TBHcal04XtalNumberingScheme : Crystal " << idx 
	      << " Layer " << idl << " UnitID = 0x" << std::hex << idunit
	      << std::dec << std::endl;
  
  return idunit;

}
