#include "SimPPS/RPDigiProducer/plugins/DeadChannelsManager.h"
#include "CondFormats/CTPPSReadoutObjects/interface/TotemSymbId.h"
#include "CondFormats/CTPPSReadoutObjects/interface/TotemDAQMapping.h"
#include "DataFormats/CTPPSDetId/interface/TotemRPDetId.h"
#include "SimPPS/RPDigiProducer/plugins/RPDisplacementGenerator.h"
#include <map>
#include "FWCore/MessageLogger/interface/MessageLogger.h"

DeadChannelsManager::DeadChannelsManager() { analysisMaskPresent = false; }

DeadChannelsManager::DeadChannelsManager(edm::ESHandle<TotemAnalysisMask> _analysisMask) {
  analysisMask = _analysisMask;
  analysisMaskPresent = true;
}

bool DeadChannelsManager::isChannelDead(RPDetId detectorId, unsigned short stripNumber) {
  unsigned int symbolicId = RPDisplacementGenerator::rawToDecId(detectorId) * 10;  //convert to symbolic ID

  unsigned int vfat = stripNumber / 128;
  symbolicId += vfat;                      //add vfatID to symbolic ID
  stripNumber = stripNumber - vfat * 128;  //convert strip number to a number from range <0; 127>
  TotemSymbID totemSymbolicId;
  totemSymbolicId.symbolicID = symbolicId;
  if (analysisMaskPresent) {
    std::map<TotemSymbID, TotemVFATAnalysisMask>::const_iterator vfatIter =
        analysisMask->analysisMask.find(totemSymbolicId);
    if (vfatIter != analysisMask->analysisMask.end()) {
      TotemVFATAnalysisMask vfatMask = vfatIter->second;
      //if channel is dead return true
      if (vfatMask.fullMask || vfatMask.maskedChannels.find(stripNumber) != vfatMask.maskedChannels.end()) {
        return true;
      }
    }
  }
  return false;
}

void DeadChannelsManager::displayMap() {
  if (analysisMaskPresent) {
    std::map<TotemSymbID, TotemVFATAnalysisMask>::const_iterator vfatIter;
    for (vfatIter = analysisMask->analysisMask.begin(); vfatIter != analysisMask->analysisMask.end(); vfatIter++) {
      LogDebug("PPSDigiProducer::DeadChannelsManager") << vfatIter->first.symbolicID << "\n";
      TotemVFATAnalysisMask am = vfatIter->second;
      if (am.fullMask) {
        LogDebug("PPSDigiProducer::DeadChannelsManager") << "   full mask\n";
      } else {
        std::set<unsigned char>::iterator setIterator;
        for (setIterator = am.maskedChannels.begin(); setIterator != am.maskedChannels.end(); setIterator++) {
          LogDebug("PPSDigiProducer::DeadChannelsManager") << "   " << (int)(*setIterator) << "\n";
        }
      }
    }
  }
}
