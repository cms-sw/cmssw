#ifndef SimFastTiming_FastTimingCommon_MTDDigitizerTraits_h
#define SimFastTiming_FastTimingCommon_MTDDigitizerTraits_h

#include "DataFormats/FTLDigi/interface/FTLDigiCollections.h"

#include "SimFastTiming/FastTimingCommon/interface/BTLTileDeviceSim.h"
#include "SimFastTiming/FastTimingCommon/interface/BTLBarDeviceSim.h"
#include "SimFastTiming/FastTimingCommon/interface/BTLElectronicsSim.h"
#include "SimFastTiming/FastTimingCommon/interface/ETLDeviceSim.h"
#include "SimFastTiming/FastTimingCommon/interface/ETLElectronicsSim.h"


class BTLTileDigitizerTraits
{
 public:
   
  // --- The BTL DIGI collection
  typedef BTLDigiCollection DigiCollection;
      
  // --- The BTL sensor response simulation for the tile geometry
  typedef BTLTileDeviceSim DeviceSim;
      
  // --- The BTL electronics simulation
  typedef BTLElectronicsSim ElectronicsSim;

};


class BTLBarDigitizerTraits
{
 public:

  // --- The BTL DIGI collection
  typedef BTLDigiCollection DigiCollection;

  // --- The BTL sensor response simulation for the bar geometry
  typedef BTLBarDeviceSim DeviceSim;

  // --- The BTL electronics simulation
  typedef BTLElectronicsSim ElectronicsSim;

};


class ETLDigitizerTraits 
{
 public:
   
  // --- The ETL DIGI collection
  typedef ETLDigiCollection DigiCollection;
      
  // --- The ETL sensor response simulation
  typedef ETLDeviceSim DeviceSim;
      
  // --- The ETL sensor response simulation
  typedef ETLElectronicsSim ElectronicsSim;

};

#endif
