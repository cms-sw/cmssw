#include "SimG4CMS/Forward/interface/BHMNumberingScheme.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "CLHEP/Units/GlobalSystemOfUnits.h"
#include "globals.hh"

BHMNumberingScheme::BHMNumberingScheme() {
  LogDebug("BHMSim") << " Creating BHMNumberingScheme" ;
}

BHMNumberingScheme::~BHMNumberingScheme() {
  LogDebug("BHMSim") << " Deleting BHMNumberingScheme" ;
}

int BHMNumberingScheme::detectorLevel(const G4Step* aStep) const {

  //Find number of levels
  const G4VTouchable* touch = aStep->GetPreStepPoint()->GetTouchable();
  int level = 0;
  if (touch) level = ((touch->GetHistoryDepth())+1);
  return level;
}

void BHMNumberingScheme::detectorLevel(const G4Step* aStep, int& level,
                                       int* copyno, G4String* name) const {

  //Get name and copy numbers
  if (level > 0) {
    const G4VTouchable* touch = aStep->GetPreStepPoint()->GetTouchable();
    for (int ii = 0; ii < level; ii++) {
      int i      = level - ii - 1;
      name[ii]   = touch->GetVolume(i)->GetName();
      copyno[ii] = touch->GetReplicaNumber(i);
    }
  }
}

unsigned int BHMNumberingScheme::getUnitID(const G4Step* aStep) const {

  unsigned intindex=0;
  int level = detectorLevel(aStep);

  LogDebug("BHMSim") << "BHMNumberingScheme number of levels= " << level;
  if (level > 0) {
    int*      copyno = new int[level];
    G4String* name   = new G4String[level];
    detectorLevel(aStep, level, copyno, name);

    if (level > 3) {
      int subdet  = copyno[0];
      int zside   = copyno[3];
      int station = copyno[1];
      intindex = packIndex (subdet, zside, station);
      LogDebug("BHMSim") << "BHMNumberingScheme : subdet " << subdet 
			 << " zside "  << zside << " station " << station; 
    }
    delete[] copyno;
    delete[] name;
  }
  LogDebug("BHMSim") << "BHMNumberingScheme : UnitID 0x" << std::hex 
		     << intindex << std::dec;

  return intindex;
  
}

unsigned BHMNumberingScheme::packIndex(int subdet, int zside, int station) {

  unsigned int idx = ((6<<28)|(subdet&0x7)<<25); // Use 6 as the detector name
  idx |= ((zside&0x3)<<5) | (station&0x1F);   // bits 0-4:station 5-6:side
  LogDebug("BHMSim") << "BHM packing: subdet " << subdet 
		     << " zside  " << zside << " station " << station  
		     << "-> 0x" << std::hex << idx << std::dec;
  return idx;
}

void BHMNumberingScheme::unpackIndex(const unsigned int& idx, int& subdet,
				     int& zside, int& station) {

  subdet  = (idx>>25)>>0x7;
  zside   = (idx>>5)&0x3;
  station = idx&0x1F;                                           
  LogDebug("BHMSim") << " Bsc unpacking: 0x " << std::hex << idx << std::dec 
		     << " -> subdet " << subdet  << " zside  " << zside 
		     << " station " << station  ;
}
