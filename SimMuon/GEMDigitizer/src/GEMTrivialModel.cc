#include "SimMuon/GEMDigitizer/interface/GEMTrivialModel.h"

#include "Geometry/GEMGeometry/interface/GEMEtaPartitionSpecs.h"
#include "Geometry/CommonTopologies/interface/TrapezoidalStripTopology.h"
#include "Geometry/GEMGeometry/interface/GEMGeometry.h"

#include "FWCore/ServiceRegistry/interface/Service.h"

#include <cmath>
#include <utility>
#include <map>


GEMTrivialModel::GEMTrivialModel(const edm::ParameterSet& config) 
  : GEMDigiModel(config)
{
}

void 
GEMTrivialModel::simulateSignal(const GEMEtaPartition* roll,
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
GEMTrivialModel::simulateClustering(const GEMEtaPartition* roll, const PSimHit* simHit, const int bx)
{
  return std::vector<std::pair<int,int> >();
}

