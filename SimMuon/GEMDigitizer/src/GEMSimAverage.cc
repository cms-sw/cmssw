#include "SimMuon/GEMDigitizer/src/GEMSimAverage.h"
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


GEMSimAverage::GEMSimAverage(const edm::ParameterSet& config) :
  GEMSim(config)
{
  averageEfficiency_ = config.getParameter<double>("averageEfficiency");
  averageShapingTime_ = config.getParameter<double>("averageShapingTime");
  averageNoiseRate_ = config.getParameter<double>("averageNoiseRate");
  bxwidth_ = config.getParameter<double>("bxwidth");
  minBunch_ = config.getParameter<int>("minBunch");
  maxBunch_ = config.getParameter<int>("maxBunch");

  sync_ = new GEMSynchronizer(config);
}

void GEMSimAverage::setRandomEngine(CLHEP::HepRandomEngine& eng)
{
  flatDistr1_ = new CLHEP::RandFlat(eng);
  flatDistr2_ = new CLHEP::RandFlat(eng);
  poissonDistr_ = new CLHEP::RandPoissonQ(eng);
  sync_->setRandomEngine(eng);
}


GEMSimAverage::~GEMSimAverage()
{
  if (flatDistr1_) delete flatDistr1_;
  if (flatDistr2_) delete flatDistr2_;
  if (poissonDistr_) delete poissonDistr_;
  delete sync_;
}

void GEMSimAverage::simulate(const GEMEtaPartition* roll,
			     const edm::PSimHitContainer& simHits)
{
  sync_->setGEMSimSetUp(getGEMSimSetUp());
  stripDigiSimLinks_.clear();
  detectorHitMap_.clear();
  stripDigiSimLinks_ = StripDigiSimLinks(roll->id().rawId());

  const Topology& topology = roll->specs()->topology();

  for (const auto & hit: simHits)
  {
    if (std::abs(hit.particleType()) != 13) continue;
    // Check GEM efficiency
    if (flatDistr1_->fire(1) > averageEfficiency_) continue;
    auto entry = hit.entryPoint();
    
    int time_hit = sync_->getSimHitBx(&hit);
    std::pair<int, int> digi(topology.channel(entry) + 1, time_hit);
      
    detectorHitMap_.insert(DetectorHitMap::value_type(digi, &hit));
    strips_.insert(digi);
  }
}


void GEMSimAverage::simulateNoise(const GEMEtaPartition* roll)
{
  GEMDetId gemId = roll->id();
  int nstrips = roll->nstrips();
  double area = 0.0;
  
  if ( gemId.region() == 0 )
    {
      throw cms::Exception("Geometry")
        << "GEMSynchronizer::simulateNoise() - this GEM id is from barrel, which cannot happen.";
    }
  else
    {
      const TrapezoidalStripTopology* top_=dynamic_cast<const TrapezoidalStripTopology*>(&(roll->topology()));
      float xmin = (top_->localPosition(0.)).x();
      float xmax = (top_->localPosition((float)roll->nstrips())).x();
      float striplength = (top_->stripLength());
      area = striplength*(xmax-xmin);
    }
  
  const int nBxing = maxBunch_ - minBunch_ + 1;
  double averageNoise = averageNoiseRate_ * nBxing * bxwidth_ * area * 1.0e-9;

  int n_hits = poissonDistr_->fire(averageNoise);

  for (int i = 0; i < n_hits; i++ ){
    int strip  = static_cast<int>(flatDistr1_->fire(1,nstrips));
    int time_hit = static_cast<int>(flatDistr2_->fire(nBxing)) + minBunch_;
    std::pair<int, int> digi(strip,time_hit);
    strips_.insert(digi);
  }
}
