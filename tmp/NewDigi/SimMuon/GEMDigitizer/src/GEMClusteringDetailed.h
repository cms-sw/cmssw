#ifndef GEMDigitizer_GEMClusteringDetailed_h
#define GEMDigitizer_GEMClusteringDetailed_h

/** \class GEMClusteringDetailed
 *
 *  Class for the GEM strip response simulation based on a trivial model
 *  Perfect timing
 *
 *  \author Sven Dildick
 */

#include "SimMuon/GEMDigitizer/src/GEMClustering.h" 
#include "CLHEP/Random/RandomEngine.h"
#include "CLHEP/Random/RandPoissonQ.h"

class GEMClusteringDetailed : public GEMClustering
{
 public:

  GEMClusteringDetailed(const edm::ParameterSet& config);

  ~GEMClusteringDetailed();

  void setRandomEngine(CLHEP::HepRandomEngine& eng);
    
  void setUp(std::vector<GEMStripClustering::StripClusteringItem>);

  const std::vector<std::pair<int,int> > simulateClustering(const GEMEtaPartition*, const PSimHit*, const int);
 
 private:
  CLHEP::RandPoissonQ* poisson_;
  double averageClusterSize_;
};

#endif
