#include "SimMuon/GEMDigitizer/src/GEMTiming.h"


GEMTiming::GEMTiming(const edm::ParameterSet& config)
  : numberOfStripsPerPartition_(config.getParameter<int>("numberOfStripsPerPartition"))
{
}


const std::vector<float>& 
GEMTiming::getCalibrationTimeVector(const uint32_t detId)
{
  auto iter = calibrationTimeMap_.find(detId);
  if(iter == calibrationTimeMap_.end())
  {
    throw cms::Exception("DataCorrupt") 
      << "GEMTiming::getCalibrationTimeVector() - no calibration time information for DetId " << detId << "\n";

  }
  if((iter->second).size() != (unsigned) numberOfStripsPerPartition_)
  {
    throw cms::Exception("DataCorrupt") 
      << "GEMTiming::getCalibrationTimeVector() - calibration time information in a wrong format for DetId " << detId << "\n";
  }
  return iter->second;
}
