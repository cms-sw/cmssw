#ifndef GEMDigitizer_GEMNoiseSimple_h
#define GEMDigitizer_GEMNoiseSimple_h

/** \class GEMNoiseSimple
 *
 *  Class for the GEM strip noise simulation based on a average model
 *
 *  \author Sven Dildick
 */

#include "SimMuon/GEMDigitizer/src/GEMNoise.h" 

class PSimHit;
namespace CLHEP
{
  class RandFlat;
  class RandPoissonQ;
}

class GEMNoiseSimple : public GEMNoise
{
 public:

  GEMNoiseSimple(const edm::ParameterSet& config);

  ~GEMNoiseSimple();

  void setRandomEngine(CLHEP::HepRandomEngine& eng); 
    
  void setUp(std::vector<GEMStripNoise::StripNoiseItem>);

  const std::set<std::pair<int, int> > simulateNoise(const GEMEtaPartition*); 

 private:
  CLHEP::RandFlat* flat1_;
  CLHEP::RandFlat* flat2_;
  CLHEP::RandPoissonQ* poisson_;

  double bxWidth_;
  int maxBunch_;
  int minBunch_;
  double averageNoiseRate_;
};

#endif
