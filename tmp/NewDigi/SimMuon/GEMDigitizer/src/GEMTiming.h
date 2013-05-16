#ifndef GEMDigitizer_GEMTiming_h
#define GEMDigitizer_GEMTiming_h

/** \class GEMTiming
 *
 *  Abstract base class for the GEM timing simulation 
 *
 *  \author Sven Dildick
 */

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "SimDataFormats/TrackingHit/interface/PSimHit.h"
#include "Geometry/GEMGeometry/interface/GEMGeometry.h"
#include "CLHEP/Random/RandomEngine.h"
#include "CondFormats/GEMObjects/interface/GEMStripTiming.h" 
//#include "CondFormats/DataRecord/interface/GEMStripTimingRcd.h" 

#include <iostream>
#include <map>
#include <vector>

namespace CLHEP
{
  class HepRandomEngine;
}

namespace 
{
  // "magic" parameter for cosmics
  const double COSMIC_PAR =  37.62;
}
class GEMTiming
{
 public:

  virtual ~GEMTiming() {};
    
  void setGeometry(const GEMGeometry* geom) {geometry_ = geom;}

  const GEMGeometry * getGeometry() const {return geometry_;}
  
  virtual void setRandomEngine(CLHEP::HepRandomEngine& eng) = 0;

  virtual void setUp(std::vector<GEMStripTiming::StripTimingItem>) = 0; 
  
  virtual const int getSimHitBx(const PSimHit*) = 0;

  const std::map< uint32_t, std::vector<float> >& getCalibrationTimeMap() const {return calibrationTimeMap_;}
  
  const std::vector<float>& getCalibrationTimeVector(const uint32_t);

 protected:

  GEMTiming(const edm::ParameterSet&);

  int numberOfStripsPerPartition_;

  const GEMGeometry * geometry_;

  std::map< uint32_t, std::vector<float> > calibrationTimeMap_;
};

#endif

