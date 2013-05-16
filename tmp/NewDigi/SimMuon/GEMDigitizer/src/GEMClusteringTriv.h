#ifndef GEMDigitizer_GEMClusteringTriv_h
#define GEMDigitizer_GEMClusteringTriv_h

/** \class GEMClusteringTriv
 *
 *  Class for the GEM strip response simulation based 
 *  on a trivial model, namely perfect timing.
 *
 *  \author Sven Dildick
 */

#include "SimMuon/GEMDigitizer/src/GEMClustering.h" 

class PSimHit;

class GEMClusteringTriv : public GEMClustering
{
 public:

  GEMClusteringTriv(const edm::ParameterSet& config);

  ~GEMClusteringTriv() {}

  void setRandomEngine(CLHEP::HepRandomEngine& eng) {}
    
  void setUp(std::vector<GEMStripClustering::StripClusteringItem>) {} 
 
  const std::vector<std::pair<int,int> >& simulateClustering(const GEMEtaPartition*, const PSimHit*, const int);
};

#endif
