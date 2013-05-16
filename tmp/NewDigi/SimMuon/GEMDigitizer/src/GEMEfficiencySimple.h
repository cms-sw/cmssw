#ifndef GEMDigitizer_GEMEfficiencySimple_h
#define GEMDigitizer_GEMEfficiencySimple_h

/** \class GEMEfficiencySimple
 *
 *  Class for the GEM strip efficiency simulation 
 *  based on a parametrized model
 *
 *  \author Sven Dildick
 */

#include "SimMuon/GEMDigitizer/src/GEMEfficiency.h" 
#include "CLHEP/Random/RandomEngine.h"
#include "CLHEP/Random/RandFlat.h"

class GEMEfficiencySimple : public GEMEfficiency
{
 public:
  GEMEfficiencySimple(const edm::ParameterSet&);

  ~GEMEfficiencySimple();

  void setRandomEngine(CLHEP::HepRandomEngine& eng);

  void setUp(std::vector<GEMStripEfficiency::StripEfficiencyItem>); 

  const bool isGoodDetId(const uint32_t);
    
 private:
  CLHEP::RandFlat* flat_;
  double averageEfficiency_;
};

#endif
