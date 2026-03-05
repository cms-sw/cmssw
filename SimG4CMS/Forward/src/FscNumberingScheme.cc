///////////////////////////////////////////////////////////////////////////////
// File: FscNumberingScheme.cc
// Date: 02.2026
// Description: Numbering scheme for Fsc
///////////////////////////////////////////////////////////////////////////////
#include "SimG4CMS/Forward/interface/FscNumberingScheme.h"
#include "DataFormats/HcalDetId/interface/HcalZDCDetId.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#define EDM_ML_DEBUG

namespace FscNumberingScheme {
  unsigned int getUnitID(const G4Step* aStep) {
    unsigned int intindex = 0;
    const G4VTouchable* touch = aStep->GetPreStepPoint()->GetTouchable();
    int level = (touch) ? ((touch->GetHistoryDepth()) + 1) : 0;
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("FscSim") << "FscNumberingScheme number of levels= " << level;
#endif
    if (level > 0) {
      int phi = touch->GetReplicaNumber(0);
      int num = touch->GetReplicaNumber(1);
      bool zside = ((num % 2) == 1);
      int stn = ((num - 1) / 2);
      int chn = HcalZDCDetId::fscChannel(stn, phi);
#ifdef EDM_ML_DEBUG
      edm::LogVerbatim("FscSim") << "FscNumberingScheme::zside = " << zside << " num = " << num << " stn = " << stn << " phi = " << phi << " chn = " << chn;
#endif
      intindex = HcalZDCDetId(HcalZDCDetId::FSC, zside, chn).rawId();
    }
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("FscSim") << "FscNumberingScheme::index 0x" << std::hex << intindex << std::dec;
#endif
    return intindex;
  }

  unsigned int packFscIndex(bool zside, int stn, int phi) {
    int chn = HcalZDCDetId::fscChannel(stn, phi);
    unsigned int idx = HcalZDCDetId(HcalZDCDetId::FSC, zside, chn).rawId();
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("FscSim") << "Fsc packing:zside  " << zside << " station " << stn << " phi " << phi << "-> 0x" << std::hex << idx << std::dec;
#endif
    return idx;
  }

  void unpackFscIndex(const unsigned int& idx, int& zside, int& stn, int& phi) {
    HcalZDCDetId id(idx);
    int chn = id.channel();
    zside = id.zside();
    stn = HcalZDCDetId::fscStationFromChannel(chn);
    phi = HcalZDCDetId::fscPhiFromChannel(chn);
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("FscSim") << " Fsc unpacking: 0x " << std::hex << idx << std::dec << " -> zside  " << zside << " station " << stn << " phi " << phi;
#endif
  }
}  // namespace FscNumberingScheme
