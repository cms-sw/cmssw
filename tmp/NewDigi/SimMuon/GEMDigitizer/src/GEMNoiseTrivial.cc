#include "SimMuon/GEMDigitizer/src/GEMNoiseTrivial.h"


GEMNoiseTrivial::GEMNoiseTrivial(const edm::ParameterSet& config)
  : GEMNoise(config)
{
  std::cout << ">>> Using noise model: GEMNoiseTrivial" << std::endl;
}


const std::set< std::pair<int, int> > 
GEMNoiseTrivial::simulateNoise(const GEMEtaPartition* roll)
{
  return std::set< std::pair<int, int> >(); 
}
