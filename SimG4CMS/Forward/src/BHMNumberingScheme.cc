#include "SimG4CMS/Forward/interface/BHMNumberingScheme.h"
#include "SimG4CMS/Forward/interface/ForwardName.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "CLHEP/Units/GlobalSystemOfUnits.h"
#include "globals.hh"

namespace {
  int detectorLevel(const G4Step* aStep) {
    //Find number of levels
    const G4VTouchable* touch = aStep->GetPreStepPoint()->GetTouchable();
    return (touch) ? ((touch->GetHistoryDepth()) + 1) : 0;
  }

  std::vector<int> detectorLevelCopyNo(const G4Step* aStep, int level) {
    //Get copy numbers
    std::vector<int> copyno;
    if (level > 0) {
      copyno.reserve(level);
      const G4VTouchable* touch = aStep->GetPreStepPoint()->GetTouchable();
      for (int ii = 0; ii < level; ++ii) {
        int i = level - ii - 1;
        copyno.push_back(touch->GetReplicaNumber(i));
      }
    }
    return copyno;
  }
}  // namespace

namespace BHMNumberingScheme {
  unsigned int getUnitID(const G4Step* aStep) {
    unsigned intindex = 0;
    int level = detectorLevel(aStep);

    LogDebug("BHMSim") << "BHMNumberingScheme number of levels= " << level;
    if (level > 0) {
      auto copyno = detectorLevelCopyNo(aStep, level);

      if (level > 3) {
        int subdet = copyno[0];
        int zside = copyno[3];
        int station = copyno[1];
        intindex = packIndex(subdet, zside, station);
        LogDebug("BHMSim") << "BHMNumberingScheme : subdet " << subdet << " zside " << zside << " station " << station;
      }
    }
    LogDebug("BHMSim") << "BHMNumberingScheme : UnitID 0x" << std::hex << intindex << std::dec;

    return intindex;
  }

  unsigned int packIndex(int subdet, int zside, int station) {
    unsigned int idx = ((6 << 28) | (subdet & 0x7) << 25);  // Use 6 as the detector name
    idx |= ((zside & 0x3) << 5) | (station & 0x1F);         // bits 0-4:station 5-6:side
    LogDebug("BHMSim") << "BHM packing: subdet " << subdet << " zside  " << zside << " station " << station << "-> 0x"
                       << std::hex << idx << std::dec;
    return idx;
  }

  void unpackIndex(const unsigned int& idx, int& subdet, int& zside, int& station) {
    subdet = (idx >> 25) >> 0x7;
    zside = (idx >> 5) & 0x3;
    station = idx & 0x1F;
    LogDebug("BHMSim") << " Bsc unpacking: 0x " << std::hex << idx << std::dec << " -> subdet " << subdet << " zside  "
                       << zside << " station " << station;
  }

}  // namespace BHMNumberingScheme
