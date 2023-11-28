///////////////////////////////////////////////////////////////////////////////
// File: BscNumberingScheme.cc
// Date: 02.2006
// Description: Numbering scheme for Bsc
// Modifications:
///////////////////////////////////////////////////////////////////////////////
#include "SimG4CMS/Forward/interface/BscNumberingScheme.h"
#include "SimG4CMS/Forward/interface/ForwardName.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
//
#include "CLHEP/Units/GlobalSystemOfUnits.h"
#include "globals.hh"

namespace {
  int detectorLevel(const G4Step* aStep) {
    //Find number of levels
    const G4VTouchable* touch = aStep->GetPreStepPoint()->GetTouchable();
    return (touch) ? ((touch->GetHistoryDepth()) + 1) : 0;
  }

  void detectorLevel(const G4Step* aStep, int level, std::vector<int>& copyno, std::vector<G4String>& name) {
    //Get name and copy numbers
    if (level > 0) {
      copyno.reserve(level);
      name.reserve(level);
      const G4VTouchable* touch = aStep->GetPreStepPoint()->GetTouchable();
      for (int ii = 0; ii < level; ++ii) {
        int i = level - ii - 1;
        name.push_back(ForwardName::getName(touch->GetVolume(i)->GetName()));
        copyno.push_back(touch->GetReplicaNumber(i));
      }
    }
  }
}  // namespace

namespace BscNumberingScheme {
  unsigned int getUnitID(const G4Step* aStep) {
    unsigned int intindex = 0;
    int level = detectorLevel(aStep);

    LogDebug("BscSim") << "BscNumberingScheme number of levels= " << level;

    if (level > 0) {
      std::vector<int> copyno;
      std::vector<G4String> name;
      detectorLevel(aStep, level, copyno, name);

      int det = 0;
      int zside = 0;
      int station = 0;
      for (int ich = 0; ich < level; ich++) {
        // new and old set up configurations are possible:
        if (name[ich] == "BSC1" || name[ich] == "BSC2") {
          zside = copyno[ich] - 1;
        } else if (name[ich] == "BSCTrap") {
          det = 0;
          station = 2 * (copyno[ich] - 1);
        } else if (name[ich] == "BSCTubs") {
          det = 1;
          station = copyno[ich] - 1;
        } else if (name[ich] == "BSCTTop") {
          ++station;
        } else if (name[ich] == "BSC2Pad") {
          det = 2;
          station = copyno[ich] - 1;
        }

        LogDebug("BscSim") << "BscNumberingScheme  "
                           << "ich=" << ich << "copyno" << copyno[ich] << "name=" << name[ich];
      }
      intindex = packBscIndex(zside, det, station);
      LogDebug("BscSim") << "BscNumberingScheme : det " << det << " zside " << zside << " station " << station
                         << " UnitID 0x" << std::hex << intindex << std::dec;

      for (int ich = 0; ich < level; ich++)
        LogDebug("BscSim") << " name = " << name[ich] << " copy = " << copyno[ich];

      LogDebug("BscSim") << " packed index = 0x" << std::hex << intindex << std::dec;
    }

    return intindex;
  }

  unsigned int packBscIndex(int zside, int det, int station) {
    unsigned int idx = 6 << 28;  // autre numero que les detecteurs existants
    idx += (zside << 5) & 32;    // vaut 0 ou 1 bit 5
    idx += (det << 3) & 24;      //bit 3-4    det:0-1-2    2 bits:0-1
    idx += (station & 7);        //bits 0-2   station:0-7=8-->2**3 =8   3 bits:0-2
    LogDebug("BscSim") << "Bsc packing: det " << det << " zside  " << zside << " station " << station << "-> 0x"
                       << std::hex << idx << std::dec;

    //  unpackBscIndex(idx);
    return idx;
  }

  void unpackBscIndex(const unsigned int& idx) {
    int zside, det, station;
    zside = (idx & 32) >> 5;
    det = (idx & 24) >> 3;
    station = idx & 7;
    LogDebug("BscSim") << " Bsc unpacking: 0x " << std::hex << idx << std::dec << " -> det " << det << " zside  "
                       << zside << " station " << station;
  }
}  // namespace BscNumberingScheme
