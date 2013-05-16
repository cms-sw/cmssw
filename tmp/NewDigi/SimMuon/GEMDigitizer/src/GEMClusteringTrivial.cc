#include "SimMuon/GEMDigitizer/src/GEMClusteringTrivial.h"


GEMClusteringTrivial::GEMClusteringTrivial(const edm::ParameterSet& config)
  : GEMClustering(config)
{
  std::cout << ">>> Using clustering model: GEMClusteringTrivial" << std::endl;
}


const std::vector<std::pair<int,int> >
GEMClusteringTrivial::simulateClustering(const GEMEtaPartition* roll, const PSimHit* simHit, const int bx)
{
  std::vector<std::pair<int,int> > cluster;
  cluster.clear();
  const auto entry(simHit->entryPoint());
  const Topology& topology(roll->specs()->topology());
  const std::pair<int, int> digi(topology.channel(entry) + 1, bx);
  cluster.push_back(digi);
  return cluster;
}

