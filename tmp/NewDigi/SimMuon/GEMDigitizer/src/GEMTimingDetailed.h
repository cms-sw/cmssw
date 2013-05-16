#ifndef GEMDigitizer_GEMTimingDetailed_h
#define GEMDigitizer_GEMTimingDetailed_h

/** \class GEMTimingDetailed
 *
 *  Class for the GEM strip response simulation based on a trivial model
 *  Perfect timing
 *
 *  \author Sven Dildick
 */

#include "SimMuon/GEMDigitizer/src/GEMTiming.h" 

namespace CLHEP
{
  class RandGaussQ;
}

class GEMTimingDetailed : public GEMTiming
{
 public:

  GEMTimingDetailed(const edm::ParameterSet&);

  ~GEMTimingDetailed();

  void setRandomEngine(CLHEP::HepRandomEngine& eng);
  
  void setUp(std::vector<GEMStripTiming::StripTimingItem>);

  const int getSimHitBx(const PSimHit*);

 private:
  CLHEP::RandGaussQ* gauss1_;
  CLHEP::RandGaussQ* gauss2_;

  double timeCalibrationOffset_;
  double timeResolution_;
  double averageShapingTime_;
  double timeJitter_;
  double signalPropagationSpeed_;
  bool cosmics_;
  double bxWidth_;
  int numberOfStripsPerPartition_;
};

#endif
