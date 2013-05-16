#ifndef GEMDigitizer_GEMTimingTrivial_h
#define GEMDigitizer_GEMTimingTrivial_h

/** \class GEMTimingTrivial
 *
 *  Class for the GEM strip response simulation based 
 *  on a trivial model that assumes perfect timing
 *
 *  \author Sven Dildick
 */

#include "SimMuon/GEMDigitizer/src/GEMTiming.h" 

class GEMTimingTrivial : public GEMTiming
{
 public:
  GEMTimingTrivial(const edm::ParameterSet&);

  ~GEMTimingTrivial() {}
    
  void setRandomEngine(CLHEP::HepRandomEngine& eng) {}

  void setUp(std::vector<GEMStripTiming::StripTimingItem>) {}
 
  const int getSimHitBx(const PSimHit*);

 private:
  
};

#endif
