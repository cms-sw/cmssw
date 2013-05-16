#ifndef GEMDigitizer_GEMEfficiency_h
#define GEMDigitizer_GEMEfficiency_h

/** \class GEMEfficiency
 *
 *  Abstract base class for the GEM strip clustering simulation 
 *
 *  \author Sven Dildick
 */

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "Geometry/GEMGeometry/interface/GEMGeometry.h"
#include "CondFormats/GEMObjects/interface/GEMStripEfficiency.h" 
//#include "CondFormats/DataRecord/interface/GEMStripEfficiencyRcd.h" 

#include <iostream>
#include <map>
#include <vector>

namespace CLHEP
{
  class HepRandomEngine;
}

class GEMEfficiency
{
 public:
  
  virtual ~GEMEfficiency() {}

  void setGeometry(const GEMGeometry* geom) {geometry_ = geom;}

  const GEMGeometry * getGeometry() const {return geometry_;}

  virtual void setRandomEngine(CLHEP::HepRandomEngine& eng) = 0;

  virtual void setUp(std::vector<GEMStripEfficiency::StripEfficiencyItem>) = 0; 

  virtual const bool isGoodDetId(const uint32_t) = 0;

  const std::map< uint32_t, std::vector<float> >& getEfficiencyMap() {return efficiencyMap_;}
  
  const std::vector<float>& getEfficiencyVector(const uint32_t);

 protected:

  GEMEfficiency(const edm::ParameterSet&); 

  int numberOfStripsPerPartition_;

  const GEMGeometry* geometry_; 

  std::map< uint32_t, std::vector<float> > efficiencyMap_;
};

#endif
