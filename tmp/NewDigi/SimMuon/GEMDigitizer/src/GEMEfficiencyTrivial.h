#ifndef GEMDigitizer_GEMEfficiencyTrivial_h
#define GEMDigitizer_GEMEfficiencyTrivial_h

/** \class GEMEfficiencyTrivial
 *
 *  Class for the GEM strip response simulation based 
 *  on a trivial model, namely all strips 100% efficient
 *
 *  \author Sven Dildick
 */

#include "SimMuon/GEMDigitizer/src/GEMEfficiency.h" 
#include "CLHEP/Random/RandomEngine.h"
#include "CLHEP/Random/RandFlat.h"

class GEMEfficiencyTrivial : public GEMEfficiency
{
 public:
  GEMEfficiencyTrivial(const edm::ParameterSet&);

  ~GEMEfficiencyTrivial() {}

  void setRandomEngine(CLHEP::HepRandomEngine& eng) {}

  void setUp(std::vector<GEMStripEfficiency::StripEfficiencyItem>) {}

  const bool isGoodDetId(const uint32_t);
};

#endif
