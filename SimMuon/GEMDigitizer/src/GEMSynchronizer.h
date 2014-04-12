#ifndef GEMDigitizer_GEMSynchronizer_h
#define GEMDigitizer_GEMSynchronizer_h

/** \class GEMSynchronizer
 *
 *  Class for the GEM strip response simulation based
 *  on a parametrized model
 *
 *  \author Vadim Khotilovich
 */

#include <FWCore/Framework/interface/Frameworkfwd.h>

class PSimHit;
class GEMSimSetUp;

namespace CLHEP {
  class HepRandomEngine;
}

class GEMSynchronizer
{
public:

  GEMSynchronizer(const edm::ParameterSet& config);

  ~GEMSynchronizer();

  int getSimHitBx(const PSimHit*, CLHEP::HepRandomEngine*);

  void setGEMSimSetUp(GEMSimSetUp *simsetup) { simSetUp_ = simsetup; }

  GEMSimSetUp* getGEMSimSetUp() { return simSetUp_; }

private:

  double timeResolution_;
  double averageShapingTime_;
  double timeJitter_;
  double signalPropagationSpeed_;
  bool cosmics_;
  double bxwidth_;
  int minBunch_;

  GEMSimSetUp * simSetUp_;
};
#endif
