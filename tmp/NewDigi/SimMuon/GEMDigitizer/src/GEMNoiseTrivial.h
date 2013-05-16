#ifndef GEMDigitizer_GEMNoiseTrivial_h
#define GEMDigitizer_GEMNoiseTrivial_h

/** \class GEMNoiseTrivial
 *
 *  Class for the GEM strip noise simulation based
 *  on a trivial model, namely no noise.
 *
 *  \author Sven Dildick
 */

#include "SimMuon/GEMDigitizer/src/GEMNoise.h" 

class PSimHit;

class GEMNoiseTrivial : public GEMNoise
{
 public:

  GEMNoiseTrivial(const edm::ParameterSet& config);

  ~GEMNoiseTrivial() {}

  void setRandomEngine(CLHEP::HepRandomEngine& eng) {}
    
  void setUp(std::vector<GEMStripNoise::StripNoiseItem>) {}

  const std::set< std::pair<int, int> > simulateNoise(const GEMEtaPartition*);
};

#endif
