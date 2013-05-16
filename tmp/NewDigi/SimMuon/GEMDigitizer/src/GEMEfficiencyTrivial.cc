#include "SimMuon/GEMDigitizer/src/GEMEfficiencyTrivial.h"


GEMEfficiencyTrivial::GEMEfficiencyTrivial(const edm::ParameterSet& config)
  : GEMEfficiency(config)
{
  std::cout << ">>> Using efficiency model: GEMEfficiencyTrivial" << std::endl;
}


const bool 
GEMEfficiencyTrivial::isGoodDetId(const uint32_t detId)
{
  return true;
}





