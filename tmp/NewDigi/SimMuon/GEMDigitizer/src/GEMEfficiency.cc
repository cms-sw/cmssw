#include "SimMuon/GEMDigitizer/src/GEMEfficiency.h" 

GEMEfficiency::GEMEfficiency(const edm::ParameterSet& config)
  : numberOfStripsPerPartition_(config.getParameter<int>("numberOfStripsPerPartition"))
{
}


const std::vector<float> &
GEMEfficiency::getEfficiencyVector(const uint32_t detId) 
{
  auto iter = efficiencyMap_.find(detId);
  if(iter == efficiencyMap_.end())
  {
    throw cms::Exception("DataCorrupt") 
      << "GEMEfficiency::getEfficiencyVector() - no efficiency information for DetId " << detId << "\n";
  }
  if((iter->second).size() != (unsigned) numberOfStripsPerPartition_)
  {
    throw cms::Exception("DataCorrupt") 
      << "GEMEfficiency::getEfficiencyVector() - efficiency information in a wrong format for DetId " << detId << "\n";
  }
  return iter->second;
}

