#include "SimMuon/GEMDigitizer/interface/GEMSimpleModel.h"

#include "Geometry/GEMGeometry/interface/GEMEtaPartitionSpecs.h"
#include "Geometry/CommonTopologies/interface/TrapezoidalStripTopology.h"
#include "Geometry/GEMGeometry/interface/GEMGeometry.h"

#include "FWCore/ServiceRegistry/interface/Service.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/RandomNumberGenerator.h"
#include "FWCore/Utilities/interface/Exception.h"

#include "CLHEP/Random/RandomEngine.h"
#include "CLHEP/Random/RandFlat.h"
#include "CLHEP/Random/RandPoissonQ.h"
#include "CLHEP/Random/RandGaussQ.h"

#include <cmath>
#include <utility>
#include <map>

namespace 
{
  // "magic" parameter for cosmics
  const double COSMIC_PAR(37.62);
}

GEMSimpleModel::GEMSimpleModel(const edm::ParameterSet& config) 
  : GEMDigiModel(config)
  , averageEfficiency_(config.getParameter<double>("averageEfficiency"))
  , averageShapingTime_(config.getParameter<double>("averageShapingTime"))
  , timeResolution_(config.getParameter<double>("timeResolution"))
  , timeJitter_(config.getParameter<double>("timeJitter"))
  , timeCalibrationOffset_(config.getParameter<double>("timeCalibrationOffset"))
  , averageNoiseRate_(config.getParameter<double>("averageNoiseRate"))
  , averageClusterSize_(config.getParameter<double>("averageClusterSize"))
  , signalPropagationSpeed_(config.getParameter<double>("signalPropagationSpeed"))
  , cosmics_(config.getParameter<bool>("cosmics"))
  , bxwidth_(config.getParameter<int>("bxwidth"))
  , minBunch_(config.getParameter<int>("minBunch"))
  , maxBunch_(config.getParameter<int>("maxBunch"))
{
}

GEMSimpleModel::~GEMSimpleModel()
{
  if (flat1_) delete flat1_;
  if (flat2_) delete flat2_;
  if (poisson_) delete poisson_;
  if (gauss1_) delete gauss1_;
  if (gauss2_) delete gauss2_;
}

void 
GEMSimpleModel::setRandomEngine(CLHEP::HepRandomEngine& eng)
{
  flat1_ = new CLHEP::RandFlat(eng);
  flat2_ = new CLHEP::RandFlat(eng);
  poisson_ = new CLHEP::RandPoissonQ(eng);
  gauss1_ = new CLHEP::RandGaussQ(eng);
  gauss2_ = new CLHEP::RandGaussQ(eng);
}

void 
GEMSimpleModel::setup()
{
  return;
}

void 
GEMSimpleModel::simulateSignal(const GEMEtaPartition* roll,
			       const edm::PSimHitContainer& simHits)
{
  stripDigiSimLinks_.clear();
  detectorHitMap_.clear();
  stripDigiSimLinks_ = StripDigiSimLinks(roll->id().rawId());

  for (const auto & hit: simHits)
  {
    if (std::abs(hit.particleType()) != 13) continue;
    // Check GEM efficiency
    if (flat1_->fire(1) > averageEfficiency_) continue;
    const int bx(getSimHitBx(&hit));
    const std::vector<std::pair<int,int> > cluster(simulateClustering(roll, &hit,bx));
    for (auto & digi : cluster){
      detectorHitMap_.insert(DetectorHitMap::value_type(digi,&hit));
      strips_.insert(digi);
    }
  }
}

int 
GEMSimpleModel::getSimHitBx(const PSimHit* simhit)
{
  int bx = -999;
  const LocalPoint simHitPos(simhit->localPosition());
  const float tof(simhit->timeOfFlight());
  // random Gaussian time correction due to electronics jitter
  const float randomJitterTime(gauss1_->fire(0., timeJitter_));
  
  const GEMDetId id(simhit->detUnitId());
  const GEMEtaPartition* roll(geometry_->etaPartition(id));

  if (!roll){
    throw cms::Exception("Geometry")
      << "GEMSimpleModel::getSimHitBx() - GEM simhit id does not match any GEM roll id: "<<id<< "\n";
    return 999;
  } 

  if(roll->id().region() == 0){
    throw cms::Exception("Geometry")
      << "GEMSimpleModel::getSimHitBx() - this GEM id is from barrel, which cannot happen: "<<roll->id()<< "\n";
  }

  const TrapezoidalStripTopology* top(dynamic_cast<const TrapezoidalStripTopology*> (&(roll->topology())));
  const float halfStripLength(0.5 * top->stripLength());
  const float distanceFromEdge(halfStripLength - simHitPos.y());
  
  // average time for the signal to propagate from the SimHit to the top of a strip
  const float averagePropagationTime(distanceFromEdge/signalPropagationSpeed_);
  // random Gaussian time correction due to the finite timing resolution of the detector
  const float randomResolutionTime(gauss2_->fire(0., timeResolution_));
  
  const float simhitTime(tof + averageShapingTime_ + randomResolutionTime + averagePropagationTime + randomJitterTime);
  const float referenceTime(timeCalibrationOffset_ + halfStripLength/signalPropagationSpeed_ + averageShapingTime_);
  const float timeDifference(cosmics_ ? (simhitTime - referenceTime)/COSMIC_PAR : simhitTime - referenceTime);
  
  // assign the bunch crossing
  bx = static_cast<int>(std::round((timeDifference)/bxwidth_));
  
  // check time
  const bool debug( false );
  if (debug){
    std::cout<<"checktime "<<bx<<" "<<timeDifference<<" "<<simhitTime<<" "<<referenceTime<<" "<<tof<<" "<<averagePropagationTime<<std::endl;
  }
  return bx;
}

void 
GEMSimpleModel::simulateNoise(const GEMEtaPartition* roll)
{
  const GEMDetId gemId(roll->id());
  const int nstrips(roll->nstrips());
  double area(0.0);
  
  if (gemId.region() == 0){
    throw cms::Exception("Geometry")
      << "GEMSynchronizer::simulateNoise() - this GEM id is from barrel, which cannot happen.";
  }
  const TrapezoidalStripTopology* top_(dynamic_cast<const TrapezoidalStripTopology*>(&(roll->topology())));
  const float xmin((top_->localPosition(0.)).x());
  const float xmax((top_->localPosition((float)roll->nstrips())).x());
  const float striplength(top_->stripLength());
  area = striplength*(xmax-xmin);
  
  const int nBxing(maxBunch_ - minBunch_ + 1);
  const double averageNoise(averageNoiseRate_ * nBxing * bxwidth_ * area * 1.0e-9);
  const int n_hits(poisson_->fire(averageNoise));

  for (int i = 0; i < n_hits; ++i){
    const int strip(static_cast<int>(flat1_->fire(1,nstrips)));
    const int time_hit(static_cast<int>(flat2_->fire(nBxing)) + minBunch_);
    std::pair<int, int> digi(strip,time_hit);
    strips_.insert(digi);
  }
  return;
}

std::vector<std::pair<int,int> >
GEMSimpleModel::simulateClustering(const GEMEtaPartition* roll, const PSimHit* simHit, const int bx)
{
  const auto entry(simHit->entryPoint());
  const Topology& topology(roll->specs()->topology());
 
  const int centralStrip(topology.channel(entry)+1);  
  int fstrip(centralStrip);
  int lstrip(centralStrip);

  // Add central digi to cluster vector
  std::vector<std::pair<int,int> > cluster_;
  cluster_.clear();
  cluster_.push_back(std::pair<int, int>(topology.channel(entry) + 1, bx));
  
  // get the cluster size
  const int clusterSize(static_cast<int>(std::round(poisson_->fire(averageClusterSize_))));
  if (clusterSize < 1) return cluster_;
  
  // Add the other digis to the cluster
  for (int cl = 0; cl < (clusterSize-1)/2; ++cl){
    if (centralStrip - cl -1 >= 1)
      cluster_.push_back(std::pair<int, int>(centralStrip-cl-1, bx));
    if (centralStrip + cl + 1 <= roll->nstrips())
      cluster_.push_back(std::pair<int, int>(centralStrip+cl+1, bx));
  }
  if (clusterSize%2 == 0){
    // insert the last strip according to the 
    // simhit position in the central strip 
    const double deltaX(roll->centreOfStrip(centralStrip).x()-entry.x());
    if (deltaX < 0.){
      if (lstrip < roll->nstrips()){
	++lstrip;
	cluster_.push_back(std::pair<int, int>(lstrip, bx));
      }
    }
    else{
      if (fstrip > 1){
	--fstrip;
	cluster_.push_back(std::pair<int, int>(fstrip, bx));
      }
    }
  }
  return cluster_;
}
