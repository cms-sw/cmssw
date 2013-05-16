#include "SimMuon/GEMDigitizer/src/GEMTimingTrivial.h"


GEMTimingTrivial::GEMTimingTrivial(const edm::ParameterSet& config)
  : GEMTiming(config)
{
  std::cout << ">>> Using timing model: GEMTimingTrivial" << std::endl;
}


const int 
GEMTimingTrivial::getSimHitBx(const PSimHit* simhit)
{
  return 0;
}

