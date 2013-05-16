#ifndef GEMDigitizer_GEMNoise_h
#define GEMDigitizer_GEMNoise_h

/** \class GEMNoise
 *
 *  Abstract base class for the GEM strip noise simulation 
 *
 *  \author Sven Dildick
 */

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "SimDataFormats/TrackingHit/interface/PSimHit.h"
#include "Geometry/GEMGeometry/interface/GEMGeometry.h"
#include "Geometry/GEMGeometry/interface/GEMEtaPartition.h"
#include "CLHEP/Random/RandomEngine.h"
#include "CondFormats/GEMObjects/interface/GEMStripNoise.h" 
//#include "CondFormats/DataRecord/interface/GEMStripNoiseRcd.h" 

#include <utility>
#include <set>
#include <iostream>
#include <map>
#include <vector>


namespace CLHEP
{
  class HepRandomEngine;
}

class GEMNoise
{
 public:

  virtual ~GEMNoise() {}

  void setGeometry(const GEMGeometry* geom) {geometry_ = geom;}

  const GEMGeometry * getGeometry() const {return geometry_;}

  virtual void setRandomEngine(CLHEP::HepRandomEngine& eng) = 0; 

  virtual void setUp(std::vector<GEMStripNoise::StripNoiseItem>) = 0; 

  virtual const std::set< std::pair<int, int> > simulateNoise(const GEMEtaPartition*) = 0;

  const std::map< uint32_t, std::vector<float> >& getNoiseRateMap() const {return noiseRateMap_;}
  
  const std::vector<float>& getNoiseRateVector(const uint32_t) const;
  
 protected:

  GEMNoise(const edm::ParameterSet&);

  int numberOfStripsPerPartition_;

  const GEMGeometry* geometry_; 

  std::map< uint32_t, std::vector<float> > noiseRateMap_;
};

#endif
