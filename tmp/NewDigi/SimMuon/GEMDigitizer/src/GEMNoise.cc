#include "SimMuon/GEMDigitizer/src/GEMNoise.h"


GEMNoise::GEMNoise(const edm::ParameterSet& config)
  : numberOfStripsPerPartition_(config.getParameter<int>("numberOfStripsPerPartition"))
{
}


const std::vector<float>& 
GEMNoise::getNoiseRateVector(const uint32_t detId) const
{
  auto iter = noiseRateMap_.find(detId);
  if(iter == noiseRateMap_.end())
  {
    throw cms::Exception("DataCorrupt") 
      << "GEMNoise::getNoiseRateVector() - no noise rate information for DetId " << detId << "\n";
  }
  if((iter->second).size() != (unsigned) numberOfStripsPerPartition_)
  {
    throw cms::Exception("DataCorrupt") 
      << "GEMNoise::getNoiseRateVector() - noise rate information in a wrong format for DetId " << detId << "\n";
  }
  return iter->second;
}
