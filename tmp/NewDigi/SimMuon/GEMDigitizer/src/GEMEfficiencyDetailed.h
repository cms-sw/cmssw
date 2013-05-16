#ifndef GEMDigitizer_GEMEfficiencyDetailed_h
#define GEMDigitizer_GEMEfficiencyDetailed_h

/** \class GEMEfficiencyDetailed
 *
 *  Class for the GEM strip efficiency simulation 
 *  based on a parametrized model
 *
 *  \author Sven Dildick
 */

#include "SimMuon/GEMDigitizer/src/GEMEfficiency.h" 
#include "CLHEP/Random/RandomEngine.h"
#include "CLHEP/Random/RandFlat.h"

class GEMEfficiencyDetailed : public GEMEfficiency
{
 public:
  GEMEfficiencyDetailed(const edm::ParameterSet&);

  ~GEMEfficiencyDetailed();

  void setRandomEngine(CLHEP::HepRandomEngine& eng);

  void setUp(std::vector<GEMStripEfficiency::StripEfficiencyItem>); 

  const bool isGoodDetId(const uint32_t);
    
 private:
  CLHEP::RandFlat* flat_;
  double averageEfficiency_;
};

#endif
