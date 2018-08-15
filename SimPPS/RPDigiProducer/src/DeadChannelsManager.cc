#include "SimPPS/RPDigiProducer/interface/DeadChannelsManager.h"
#include "CondFormats/CTPPSReadoutObjects/interface/TotemSymbId.h"
#include "CondFormats/CTPPSReadoutObjects/interface/TotemDAQMapping.h"
#include "DataFormats/CTPPSDetId/interface/TotemRPDetId.h"
#include "SimPPS/RPDigiProducer/interface/RPDisplacementGenerator.h"
#include <map>

DeadChannelsManager::DeadChannelsManager() {
	analysisMaskPresent = false;
}

DeadChannelsManager::DeadChannelsManager(edm::ESHandle<TotemAnalysisMask> _analysisMask) {
	analysisMask = _analysisMask;
	analysisMaskPresent = true;
}
//-------------------------------- static members ---------------------------------------
//const uint32_t startArmBit = 24, maskArm = 0x1, startStationBit = 22, maskStation = 0x3, startRPBit = 19, maskRP = 0x7 ;
//static const uint32_t startPlaneBit = 15, maskPlane = 0xF; 

bool DeadChannelsManager::isChannelDead(RPDetId detectorId, unsigned short stripNumber) {
	unsigned int symbolicId = RPDisplacementGenerator::rawToDecId(detectorId) * 10; //convert to symbolic ID

  unsigned int vfat = stripNumber / 128;
	symbolicId += vfat; //add vfatID to symbolic ID
	stripNumber = stripNumber - vfat * 128; //convert strip number to a number from range <0; 127>
	TotemSymbID totemSymbolicId;
	//totemSymbolicId.subSystem = TotemSymbID::RP;
	totemSymbolicId.symbolicID = symbolicId;
	if (analysisMaskPresent) {
		std::map<TotemSymbID, TotemVFATAnalysisMask>::const_iterator vfatIter = analysisMask->analysisMask.find(
		        totemSymbolicId);
		if (vfatIter != analysisMask->analysisMask.end()) {
			TotemVFATAnalysisMask vfatMask = vfatIter->second;
			//if channel is dead return true
			if (vfatMask.fullMask || vfatMask.maskedChannels.find(stripNumber)
			        != vfatMask.maskedChannels.end()) {
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
			std::cout << vfatIter->first.symbolicID << "\n";
			TotemVFATAnalysisMask am = vfatIter->second;
			if (am.fullMask) {
				std::cout << "   full mask\n";
			} else {
				std::set<unsigned char>::iterator setIterator;
				for (setIterator = am.maskedChannels.begin(); setIterator != am.maskedChannels.end(); setIterator++) {
					std::cout << "   " << (int) (*setIterator) << "\n";
				}
			}

		}
	}
}
