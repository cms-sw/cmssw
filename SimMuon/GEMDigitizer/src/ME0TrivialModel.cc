#include "SimMuon/GEMDigitizer/interface/ME0TrivialModel.h"

#include "Geometry/GEMGeometry/interface/ME0EtaPartitionSpecs.h"
#include "Geometry/CommonTopologies/interface/TrapezoidalStripTopology.h"
#include "Geometry/GEMGeometry/interface/ME0Geometry.h"

#include "FWCore/ServiceRegistry/interface/Service.h"

#include <cmath>
#include <utility>
#include <map>


ME0TrivialModel::ME0TrivialModel(const edm::ParameterSet& config) 
  : ME0DigiModel(config)
{
}

void 
ME0TrivialModel::simulateSignal(const ME0EtaPartition* roll,
				const edm::PSimHitContainer& simHits)
{
  stripDigiSimLinks_.clear();
  detectorHitMap_.clear();
  stripDigiSimLinks_ = StripDigiSimLinks(roll->id().rawId());

  const Topology& topology(roll->specs()->topology());

  for (const auto & hit: simHits)
  {
    if (std::abs(hit.particleType()) != 13) continue;
    auto entry = hit.entryPoint();
     // please keep hit time always 0 for this model
    std::pair<int, int> digi(topology.channel(entry) + 1, 0);
    detectorHitMap_.insert(DetectorHitMap::value_type(digi, &hit));
    strips_.insert(digi);
  }
}


std::vector<std::pair<int,int> > 
ME0TrivialModel::simulateClustering(const ME0EtaPartition* roll, const PSimHit* simHit, const int bx)
{
  return std::vector<std::pair<int,int> >();
}

