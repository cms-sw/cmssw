#ifndef GEMDigitizer_GEMClusteringTrivial_h
#define GEMDigitizer_GEMClusteringTrivial_h

/** \class GEMClusteringTrivial
 *
 *  Class for the GEM strip response simulation based 
 *  on a trivial model, namely perfect timing.
 *
 *  \author Sven Dildick
 */

#include "SimMuon/GEMDigitizer/src/GEMClustering.h" 

class PSimHit;

class GEMClusteringTrivial : public GEMClustering
{
 public:

  GEMClusteringTrivial(const edm::ParameterSet& config);

  ~GEMClusteringTrivial() {}

  void setRandomEngine(CLHEP::HepRandomEngine& eng) {}
    
  void setUp(std::vector<GEMStripClustering::StripClusteringItem>) {} 
 
  const std::vector<std::pair<int,int> > simulateClustering(const GEMEtaPartition*, const PSimHit*, const int);
};

#endif
