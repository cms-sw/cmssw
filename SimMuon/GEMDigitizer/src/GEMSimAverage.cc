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
  rate_ = config.getParameter<double>("rate");
  nbxing_ = config.getParameter<int>("nbxing");
  gate_ = config.getParameter<double>("gate");
  averageEfficiency_ = config.getParameter<double>("averageEfficiency");
  averageTimingOffset_ = config.getParameter<double>("averageTimingOffset");
  averageNoiseRate_ = config.getParameter<double>("averageNoiseRate");

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
  //_gemSync->setGEMSimSetUp(getGEMSimSetUp());
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
    
    //int time_hit = _gemSync->getSimHitBx(&(*_hit));
    // please keep hit time always 0 for this model
    int time_hit = 0;
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

  double averageNoise = averageNoiseRate_*nbxing_*gate_*area*1.0e-9;

  int n_hits = poissonDistr_->fire(averageNoise);

  for (int i = 0; i < n_hits; i++ ){
    int strip  = static_cast<int>(flatDistr1_->fire(1,nstrips));
    int time_hit = static_cast<int>(flatDistr2_->fire((nbxing_*gate_)/gate_)) - nbxing_/2;
    std::pair<int, int> digi(strip,time_hit);
    strips_.insert(digi);
  }
}
