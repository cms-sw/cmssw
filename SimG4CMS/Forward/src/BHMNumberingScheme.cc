#include "SimG4CMS/Forward/interface/BHMNumberingScheme.h"
#include "SimG4CMS/Forward/interface/ForwardName.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "CLHEP/Units/GlobalSystemOfUnits.h"
#include "globals.hh"

BHMNumberingScheme::BHMNumberingScheme() { LogDebug("BHMSim") << " Creating BHMNumberingScheme"; }

unsigned int BHMNumberingScheme::getUnitID(const G4Step* aStep) const {
  unsigned intindex = 0;

  //Find number of levels
  const G4VTouchable* touch = aStep->GetPreStepPoint()->GetTouchable();
  int level = (touch) ? ((touch->GetHistoryDepth()) + 1) : 0;

  LogDebug("BHMSim") << "BHMNumberingScheme number of levels= " << level;
  if (level > 3) {
    int subdet = touch->GetReplicaNumber(level - 1);
    int zside = touch->GetReplicaNumber(level - 4);
    int station = touch->GetReplicaNumber(level - 2);
    intindex = packIndex(subdet, zside, station);
    LogDebug("BHMSim") << "BHMNumberingScheme : subdet " << subdet << " zside " << zside << " station " << station;
  }
  LogDebug("BHMSim") << "BHMNumberingScheme : UnitID 0x" << std::hex << intindex << std::dec;

  return intindex;
}

unsigned int BHMNumberingScheme::packIndex(int subdet, int zside, int station) {
  unsigned int idx = ((6 << 28) | (subdet & 0x7) << 25);  // Use 6 as the detector name
  idx |= ((zside & 0x3) << 5) | (station & 0x1F);         // bits 0-4:station 5-6:side
  LogDebug("BHMSim") << "BHM packing: subdet " << subdet << " zside  " << zside << " station " << station << "-> 0x"
                     << std::hex << idx << std::dec;
  return idx;
}

void BHMNumberingScheme::unpackIndex(const unsigned int& idx, int& subdet, int& zside, int& station) {
  subdet = (idx >> 25) >> 0x7;
  zside = (idx >> 5) & 0x3;
  station = idx & 0x1F;
  LogDebug("BHMSim") << " Bsc unpacking: 0x " << std::hex << idx << std::dec << " -> subdet " << subdet << " zside  "
                     << zside << " station " << station;
}
