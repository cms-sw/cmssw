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

BscNumberingScheme::BscNumberingScheme() { LogDebug("BscSim") << " Creating BscNumberingScheme"; }

unsigned int BscNumberingScheme::getUnitID(const G4Step* aStep) const {
  unsigned int intindex = 0;

  //Find number of levels
  const G4VTouchable* touch = aStep->GetPreStepPoint()->GetTouchable();
  int level = (touch) ? ((touch->GetHistoryDepth()) + 1) : 0;

  LogDebug("BscSim") << "BscNumberingScheme number of levels= " << level;

  if (level > 0) {
    int det = 0;
    int zside = 0;
    int station = 0;
    for (int ich = 0; ich < level; ich++) {
      int copyno = touch->GetReplicaNumber(level - ich - 1);
      G4String name = ForwardName::getName(touch->GetVolume(level - ich - 1)->GetName());
      // new and old set up configurations are possible:
      if (name == "BSC1" || name == "BSC2") {
        zside = copyno - 1;
      } else if (name == "BSCTrap") {
        det = 0;
        station = 2 * (copyno - 1);
      } else if (name == "BSCTubs") {
        det = 1;
        station = copyno - 1;
      } else if (name == "BSCTTop") {
        ++station;
      } else if (name == "BSC2Pad") {
        det = 2;
        station = copyno - 1;
      }

      LogDebug("BscSim") << "BscNumberingScheme  "
                         << "ich=" << ich << "copyno" << copyno << "name=" << name;
    }
    intindex = packBscIndex(zside, det, station);
    LogDebug("BscSim") << "BscNumberingScheme : det " << det << " zside " << zside << " station " << station
                       << " UnitID 0x" << std::hex << intindex << std::dec;

    for (int ich = 0; ich < level; ich++) {
      G4String name = ForwardName::getName(touch->GetVolume(level - ich - 1)->GetName());
      int copyno = touch->GetReplicaNumber(level - ich - 1);
      LogDebug("BscSim") << " name = " << name << " copy = " << copyno;
    }
    LogDebug("BscSim") << " packed index = 0x" << std::hex << intindex << std::dec;
  }

  return intindex;
}

unsigned int BscNumberingScheme::packBscIndex(int zside, int det, int station) {
  unsigned int idx = 6 << 28;  // autre numero que les detecteurs existants
  idx += (zside << 5) & 32;    // vaut 0 ou 1 bit 5
  idx += (det << 3) & 24;      //bit 3-4    det:0-1-2    2 bits:0-1
  idx += (station & 7);        //bits 0-2   station:0-7=8-->2**3 =8   3 bits:0-2
  LogDebug("BscSim") << "Bsc packing: det " << det << " zside  " << zside << " station " << station << "-> 0x"
                     << std::hex << idx << std::dec;

  //  unpackBscIndex(idx);
  return idx;
}

void BscNumberingScheme::unpackBscIndex(const unsigned int& idx) {
  int zside, det, station;
  zside = (idx & 32) >> 5;
  det = (idx & 24) >> 3;
  station = idx & 7;
  LogDebug("BscSim") << " Bsc unpacking: 0x " << std::hex << idx << std::dec << " -> det " << det << " zside  " << zside
                     << " station " << station;
}
