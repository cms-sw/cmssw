#ifndef GEMDigitizer_GEMClustering_h
#define GEMDigitizer_GEMClustering_h

/** \class GEMClustering
 *
 *  Abstract base class for the GEM strip clustering simulation 
 *
 *  \author Sven Dildick
 */

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "SimDataFormats/TrackingHit/interface/PSimHit.h"
#include "Geometry/GEMGeometry/interface/GEMGeometry.h"
#include "Geometry/GEMGeometry/interface/GEMEtaPartitionSpecs.h"
#include "Geometry/CommonTopologies/interface/TrapezoidalStripTopology.h"
#include "CondFormats/GEMObjects/interface/GEMStripClustering.h" 
//#include "CondFormats/DataRecord/interface/GEMStripClusteringRcd.h" 

#include <iostream>
#include <map>
#include <vector>
#include <utility>

namespace CLHEP
{
  class HepRandomEngine;
}

class GEMClustering
{
 public:

  virtual ~GEMClustering() {}

  void setGeometry(const GEMGeometry* geom) {geometry_ = geom;}

  const GEMGeometry * getGeometry() const {return geometry_;}

  virtual void setRandomEngine(CLHEP::HepRandomEngine& eng) = 0;

  virtual void setUp(std::vector<GEMStripClustering::StripClusteringItem>) = 0; 

  virtual const std::vector<std::pair<int,int> > simulateClustering(const GEMEtaPartition*, const PSimHit*, const int bx) = 0;

  const std::map< uint32_t, std::vector<int> >& getClusterSizeMap() const {return clusterSizeMap_;}
  
  const std::vector<int>& getClusterSizeVector(const uint32_t) const;

 protected:

  GEMClustering(const edm::ParameterSet&);

  int numberOfStripsPerPartition_;

  const GEMGeometry* geometry_; 

  std::map< uint32_t, std::vector<int> > clusterSizeMap_;
};

#endif

