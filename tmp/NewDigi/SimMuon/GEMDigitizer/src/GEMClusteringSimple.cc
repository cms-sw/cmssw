#include "SimMuon/GEMDigitizer/src/GEMClusteringSimple.h"

GEMClusteringSimple::GEMClusteringSimple(const edm::ParameterSet& config)
  : GEMClustering(config)
  , poisson_(0)

{
  std::cout << ">>> Using clustering model: GEMClusteringSimple" << std::endl;

  const auto pset(config.getParameter<edm::ParameterSet>("clusteringModelConfig"));
  averageClusterSize_ = pset.getParameter<double>("averageClusterSize");
}


GEMClusteringSimple::~GEMClusteringSimple()
{
  if (poisson_) delete poisson_;
}


void 
GEMClusteringSimple::setRandomEngine(CLHEP::HepRandomEngine& eng)
{
  poisson_ = new CLHEP::RandPoissonQ(eng);
}


void 
GEMClusteringSimple::setUp(std::vector<GEMStripClustering::StripClusteringItem> vCluster)
{
  clusterSizeMap_.clear();
  // Loop over the detIds                                                                                                                                             
  for(const auto &det: getGeometry()->dets())
  {
    const GEMEtaPartition* roll(dynamic_cast<GEMEtaPartition*>(det));
    
    // check for valid rolls     
    if(!roll) continue;
    const int nStrips(roll->nstrips());
    if (numberOfStripsPerPartition_ != nStrips)
    {
      throw cms::Exception("DataCorrupt") 
	<< "GEMClusteringSimple::setUp() - number of strips per partition in configuration ("
	<< numberOfStripsPerPartition_ << ") is not the same as in geometry (" << nStrips << ")." << std::endl; 
    }
    const int clusterSize(static_cast<int>(std::round(poisson_->fire(averageClusterSize_))));
    std::vector<int> v(numberOfStripsPerPartition_);
    v.clear();
    for (int i=0; i < numberOfStripsPerPartition_; ++i)
    { 
      v.at(i) = clusterSize;
    }
    clusterSizeMap_[roll->id().rawId()] = v;  
  }
}


const std::vector<std::pair<int,int> >
GEMClusteringSimple::simulateClustering(const GEMEtaPartition* roll, const PSimHit* simHit, const int bx)
{
  const auto entry(simHit->entryPoint());
  const Topology& topology(roll->specs()->topology());
 
  const int centralStrip(topology.channel(entry)+1);  
  int fstrip(centralStrip);
  int lstrip(centralStrip);

  // Add central digi to cluster vector
  std::vector<std::pair<int,int> > cluster;
  cluster.clear();
  cluster.push_back(std::pair<int, int>(topology.channel(entry) + 1, bx));
  
  // get the cluster size
  const int clusterSize(clusterSizeMap_[roll->id().rawId()].at(0));
  if (clusterSize < 1) return cluster;
  
  // Add the other digis to the cluster
  for (int cl = 0; cl < (clusterSize-1)/2; ++cl)
  {
    if (centralStrip - cl -1 >= 1)
    {
      cluster.push_back(std::pair<int, int>(centralStrip-cl-1, bx));
    }
    if (centralStrip + cl + 1 <= roll->nstrips())
    {
      cluster.push_back(std::pair<int, int>(centralStrip+cl+1, bx));
    }
  }
  if (clusterSize%2 == 0)
  {
    // insert the last strip according to the 
    // simhit position in the central strip 
    const double deltaX(roll->centreOfStrip(centralStrip).x()-entry.x());
    if (deltaX < 0.) 
    {
      if (lstrip < roll->nstrips())
      {
	++lstrip;
	cluster.push_back(std::pair<int, int>(lstrip, bx));
      }
    }
    else
    {
      if (fstrip > 1)
      {
	--fstrip;
	cluster.push_back(std::pair<int, int>(fstrip, bx));
      }
    }
  }
  return cluster;
}

