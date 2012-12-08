#include "SimMuon/GEMDigitizer/src/GEMSimTriv.h"
#include "SimMuon/GEMDigitizer/src/GEMSimSetUp.h"
#include "SimMuon/GEMDigitizer/src/GEMSynchronizer.h"

#include "Geometry/GEMGeometry/interface/GEMEtaPartitionSpecs.h"
#include "Geometry/CommonTopologies/interface/TrapezoidalStripTopology.h"
#include "Geometry/GEMGeometry/interface/GEMGeometry.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/RandomNumberGenerator.h"
#include "FWCore/Utilities/interface/Exception.h"

#include "CLHEP/Random/RandomEngine.h"
#include "CLHEP/Random/RandFlat.h"
#include "CLHEP/Random/RandPoissonQ.h"

#include <cmath>
#include <utility>
#include <map>


GEMSimTriv::GEMSimTriv(const edm::ParameterSet& config) :
  GEMSim(config)
{
  rate_ = config.getParameter<double>("Rate");
  nbxing_ = config.getParameter<int>("Nbxing");
  gate_ = config.getParameter<double>("Gate");

  sync_ = new GEMSynchronizer(config);
}

void GEMSimTriv::setRandomEngine(CLHEP::HepRandomEngine& eng)
{
  flatDistr1_ = new CLHEP::RandFlat(eng);
  flatDistr2_ = new CLHEP::RandFlat(eng);
  poissonDistr_ = new CLHEP::RandPoissonQ(eng);
  sync_->setRandomEngine(eng);
}


GEMSimTriv::~GEMSimTriv()
{
  if (flatDistr1_) delete flatDistr1_;
  if (flatDistr2_) delete flatDistr2_;
  if (poissonDistr_) delete poissonDistr_;
  delete sync_;
}

void GEMSimTriv::simulate(const GEMEtaPartition* roll,
                          const edm::PSimHitContainer& simHits)
{
  //_gemSync->setGEMSimSetUp(getGEMSimSetUp());
  stripDigiSimLinks_.clear();
  detectorHitMap_.clear();
  stripDigiSimLinks_ = StripDigiSimLinks(roll->id().rawId());

  const Topology& topology = roll->specs()->topology();

  for (const auto & hit: simHits)
  {
    if (std::abs(hit.particleType()) != 13) continue;

    // Here I hould check if the RPC are up side down;
    auto entry = hit.entryPoint();

    //int time_hit = _gemSync->getSimHitBx(&(*_hit));
    // please keep hit time always 0 for this model
    int time_hit = 0;
    std::pair<int, int> digi(topology.channel(entry) + 1, time_hit);

    detectorHitMap_.insert(DetectorHitMap::value_type(digi, &hit));
    strips_.insert(digi);
  }
}


void GEMSimTriv::simulateNoise(const GEMEtaPartition* roll)
{
  // plase keep it empty for this model
  return;
}
