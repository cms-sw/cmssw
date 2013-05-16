#include "SimMuon/GEMDigitizer/src/GEMClustering.h"


GEMClustering::GEMClustering(const edm::ParameterSet& config)
  : numberOfStripsPerPartition_(config.getParameter<int>("numberOfStripsPerPartition"))
{
}


const std::vector<int>& 
GEMClustering::getClusterSizeVector(const uint32_t detId) const
{
  auto iter = clusterSizeMap_.find(detId);
  if(iter == clusterSizeMap_.end())
  {
    throw cms::Exception("DataCorrupt") 
      << "GEMClustering::getClusterSizeVector() - no cluster size information for DetId " << detId << "\n";
  }
  if((iter->second).size() != (unsigned) numberOfStripsPerPartition_)
  {
    throw cms::Exception("DataCorrupt") 
      << "GEMClustering::getClusterSizeVector() - cluster size information in a wrong format for DetId " << detId << "\n";
  }
  return iter->second;
}
