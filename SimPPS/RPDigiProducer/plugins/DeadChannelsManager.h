#ifndef SimPPS_RPDigiProducer_DEAD_CHANNELS_MANAGER
#define SimPPS_RPDigiProducer_DEAD_CHANNELS_MANAGER

#include "FWCore/Framework/interface/ESHandle.h"
#include "SimPPS/RPDigiProducer/interface/RPSimTypes.h"
#include "CondFormats/CTPPSReadoutObjects/interface/TotemAnalysisMask.h"

/*
 * This purpose of this class is to answer the question whether a channel (given by detectorId
 * and stripNumber) is dead or not. This class uses analysisMask which is provided
 * by DAQMappingSourceXML.
 * @author Jakub Smajek
 */
class DeadChannelsManager {
private:
  edm::ESHandle<TotemAnalysisMask> analysisMask;
  bool analysisMaskPresent;  //this variable indicates whether analysisMask is present or not

public:
  /**
	 * This constructor allows us to set analysisMask. The analysisMask can be read from
	 * EventSetup.
	 */
  DeadChannelsManager(edm::ESHandle<TotemAnalysisMask> analysisMask);
  DeadChannelsManager();
  /**
	 * This function answers the question whether given channel is dead or not.
	 * RPDetId - detector ID given in raw form, this function has to convert raw ID to symbolic
	 * stripNumber - ID of the strip, it is a number from range <0; 511>, this function has to convert
	 * it into a vfat ID and a number from range <0; 127>
	 *
	 * It is assumed that:
	 * channels 0 - 127 are in vfat number 0
	 * channels 128 - 255 are in vfat number 1
	 * channels 256 - 383 are in vfat number 2
	 * channels 384 - 511 are in vfat number 3
	 */
  bool isChannelDead(RPDetId detectorId, unsigned short stripNumber);
  void displayMap();
  static uint32_t rawToDecId(uint32_t raw);
};

#endif
