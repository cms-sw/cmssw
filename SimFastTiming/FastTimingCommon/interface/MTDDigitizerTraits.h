#ifndef SimFastTiming_FastTimingCommon_MTDDigitizerTraits_h
#define SimFastTiming_FastTimingCommon_MTDDigitizerTraits_h

#include "DataFormats/FTLDigi/interface/FTLDigiCollections.h"

#include "SimFastTiming/FastTimingCommon/interface/BTLDeviceSim.h"
#include "SimFastTiming/FastTimingCommon/interface/ETLDeviceSim.h"
#include "SimFastTiming/FastTimingCommon/interface/BTLElectronicsSim.h"
#include "SimFastTiming/FastTimingCommon/interface/ETLElectronicsSim.h"


class BTLDigitizerTraits 
{
 public:
   
  // --- The BTL DIGI collection
  typedef BTLDigiCollection DigiCollection;
      
  // --- The BTL sensor response simulation
  typedef BTLDeviceSim DeviceSim;
      
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
