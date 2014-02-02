#include "SimMuon/GEMDigitizer/interface/ME0SimpleModel.h"
#include "Geometry/GEMGeometry/interface/ME0EtaPartitionSpecs.h"
#include "Geometry/CommonTopologies/interface/TrapezoidalStripTopology.h"
#include "Geometry/GEMGeometry/interface/ME0Geometry.h"
#include "FWCore/Utilities/interface/RandomNumberGenerator.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "CLHEP/Random/RandomEngine.h"
#include "CLHEP/Random/RandFlat.h"
#include "CLHEP/Random/RandPoissonQ.h"
#include "CLHEP/Random/RandGaussQ.h"
#include "CLHEP/Random/RandGamma.h"
#include <cmath>
#include <utility>
#include <map>

namespace
{
  // "magic" parameter for cosmics
  const double COSMIC_PAR(37.62);
}

ME0SimpleModel::ME0SimpleModel(const edm::ParameterSet& config) :
ME0DigiModel(config)
, averageEfficiency_(config.getParameter<double> ("averageEfficiency"))
, averageShapingTime_(config.getParameter<double> ("averageShapingTime"))
, timeResolution_(config.getParameter<double> ("timeResolution"))
, timeJitter_(config.getParameter<double> ("timeJitter"))
, timeCalibrationOffset1_(config.getParameter<double> ("timeCalibrationOffset1"))
, timeCalibrationOffset23_(config.getParameter<double> ("timeCalibrationOffset23"))
, averageNoiseRate_(config.getParameter<double> ("averageNoiseRate"))
, averageClusterSize_(config.getParameter<double> ("averageClusterSize"))
, signalPropagationSpeed_(config.getParameter<double> ("signalPropagationSpeed"))
, cosmics_(config.getParameter<bool> ("cosmics"))
, bxwidth_(config.getParameter<int> ("bxwidth"))
, minBunch_(config.getParameter<int> ("minBunch"))
, maxBunch_(config.getParameter<int> ("maxBunch"))
, digitizeOnlyMuons_(config.getParameter<bool> ("digitizeOnlyMuons"))
, neutronGammaRoll1_(config.getParameter<std::vector<double>>("neutronGammaRoll1"))
, neutronGammaRoll2_(config.getParameter<std::vector<double>>("neutronGammaRoll2"))
, neutronGammaRoll3_(config.getParameter<std::vector<double>>("neutronGammaRoll3"))
, doNoiseCLS_(config.getParameter<bool> ("doNoiseCLS"))
, scaleLumi_(config.getParameter<double> ("scaleLumi"))
{
}

ME0SimpleModel::~ME0SimpleModel()
{
  if (flat1_)
    delete flat1_;
  if (flat2_)
    delete flat2_;
  if (poisson_)
    delete poisson_;
  if (gauss1_)
    delete gauss1_;
  if (gauss2_)
    delete gauss2_;
  if (gamma1_)
    delete gamma1_;
}

void ME0SimpleModel::setRandomEngine(CLHEP::HepRandomEngine& eng)
{
  flat1_ = new CLHEP::RandFlat(eng);
  flat2_ = new CLHEP::RandFlat(eng);
  poisson_ = new CLHEP::RandPoissonQ(eng);
  gauss1_ = new CLHEP::RandGaussQ(eng);
  gauss2_ = new CLHEP::RandGaussQ(eng);
  gamma1_ = new CLHEP::RandGamma(eng);
}

void ME0SimpleModel::setup()
{
  return;
}

void ME0SimpleModel::simulateSignal(const ME0EtaPartition* roll, const edm::PSimHitContainer& simHits)
{
  stripDigiSimLinks_.clear();
  detectorHitMap_.clear();
  stripDigiSimLinks_ = StripDigiSimLinks(roll->id().rawId());

  for (edm::PSimHitContainer::const_iterator hit = simHits.begin(); hit != simHits.end(); ++hit)
  {
    if (std::abs(hit->particleType()) != 13 && digitizeOnlyMuons_)
      continue;

    // Check ME0 efficiency
    if (flat1_->fire(1) > averageEfficiency_)
      continue;
    const int bx(getSimHitBx(&(*hit)));
    const std::vector<std::pair<int, int> > cluster(simulateClustering(roll, &(*hit), bx));
    for  (auto & digi : cluster)
    {
      detectorHitMap_.insert(DetectorHitMap::value_type(digi,&*(hit)));
      strips_.insert(digi);
    }
  }
}

int ME0SimpleModel::getSimHitBx(const PSimHit* simhit)
{
  int bx = -999;
  const LocalPoint simHitPos(simhit->localPosition());
  const float tof(simhit->timeOfFlight());
  // random Gaussian time correction due to electronics jitter
  const float randomJitterTime(gauss1_->fire(0., timeJitter_));

  const ME0DetId id(simhit->detUnitId());
  const ME0EtaPartition* roll(geometry_->etaPartition(id));

  if (!roll)
  {
    throw cms::Exception("Geometry")
        << "ME0SimpleModel::getSimHitBx() - ME0 simhit id does not match any ME0 roll id: " << id << "\n";
    return 999;
  }

  if (roll->id().region() == 0)
  {
    throw cms::Exception("Geometry")
        << "ME0SimpleModel::getSimHitBx() - this ME0 id is from barrel, which cannot happen: " << roll->id() << "\n";
  }

  const TrapezoidalStripTopology* top(dynamic_cast<const TrapezoidalStripTopology*> (&(roll->topology())));
  const float halfStripLength(0.5 * top->stripLength());
  const float distanceFromEdge(halfStripLength - simHitPos.y());

  // signal propagation speed in vacuum in [m/s]
  const double cspeed = 299792458;
  // signal propagation speed in material in [cm/ns]
  double signalPropagationSpeedTrue = signalPropagationSpeed_ * cspeed * 1e+2 * 1e-9;

  // average time for the signal to propagate from the SimHit to the top of a strip
  const float averagePropagationTime(distanceFromEdge / signalPropagationSpeedTrue);
  // random Gaussian time correction due to the finite timing resolution of the detector
  const float randomResolutionTime(gauss2_->fire(0., timeResolution_));

  const float simhitTime(tof + averageShapingTime_ + randomResolutionTime + averagePropagationTime + randomJitterTime);

  float referenceTime = timeCalibrationOffset1_ + halfStripLength / signalPropagationSpeedTrue + averageShapingTime_;

  const float timeDifference(cosmics_ ? (simhitTime - referenceTime) / COSMIC_PAR : simhitTime - referenceTime);

  // assign the bunch crossing
  bx = static_cast<int> (std::round((timeDifference) / bxwidth_));

  // check time
  const bool debug(false);
  if (debug)
  {
    std::cout << "checktime " << "bx = " << bx << "\tdeltaT = " << timeDifference << "\tsimT =  " << simhitTime
        << "\trefT =  " << referenceTime << "\ttof = " << tof << "\tavePropT =  " << averagePropagationTime
        << "\taveRefPropT = " << halfStripLength / signalPropagationSpeedTrue << std::endl;
  }
  return bx;
}

void ME0SimpleModel::simulateNoise(const ME0EtaPartition* roll)
{
  const ME0DetId me0Id(roll->id());
  int rollNumb = me0Id.roll();
  const int nstrips(roll->nstrips());
  double trArea(0.0);
  double trStripArea(0.0);

  if (me0Id.region() == 0)
  {
    throw cms::Exception("Geometry") << "ME0Synchronizer::simulateNoise() - this ME0 id is from barrel, which cannot happen.";
  }

  const TrapezoidalStripTopology* top_(dynamic_cast<const TrapezoidalStripTopology*> (&(roll->topology())));
  const float striplength(top_->stripLength());
  trStripArea = (roll->pitch()) * striplength;
  trArea = trStripArea * nstrips;

  const int nBxing(maxBunch_ - minBunch_ + 1);
  double averageNoiseRatePerRoll = neutronGammaRoll1_[rollNumb - 1];

  //simulate intrinsic noise
  if(simulateIntrinsicNoise_)
  {
    double aveIntrinsicNoisPerStrip = averageNoiseRate_ * nBxing * bxwidth_ * trStripArea * 1.0e-9;
    for(int j = 0; j < nstrips; ++j)
    {
      const int n_intrHits = poisson_->fire(aveIntrinsicNoisPerStrip);
    
      for (int k = 0; k < n_intrHits; k++ )
      {
        const int time_hit(static_cast<int> (flat2_->fire(nBxing)) + minBunch_);
        std::pair<int, int> digi(k+1,time_hit);
        strips_.insert(digi);
      }
    }
  }//end simulate intrinsic noise

  //simulate bkg contribution
  const double averageNoise(averageNoiseRatePerRoll * nBxing * bxwidth_ * trArea * 1.0e-9 * scaleLumi_);
  const int n_hits(poisson_->fire(averageNoise));

  for (int i = 0; i < n_hits; ++i)
  {
    const int centralStrip(static_cast<int> (flat1_->fire(1, nstrips)));
    const int time_hit(static_cast<int> (flat2_->fire(nBxing)) + minBunch_);

    if (doNoiseCLS_)
    {
      std::vector<std::pair<int, int> > cluster_;
      cluster_.clear();
      cluster_.push_back(std::pair<int, int>(centralStrip, time_hit));

      //     const int clusterSize(static_cast<int>(std::round(poisson_->fire(averageClusterSize_))));
      int clusterSize(static_cast<int> (std::round(gamma1_->fire(averageClusterSize_, averageClusterSize_))));

      //keep cls between [1, 6]
      if (clusterSize < 1 || clusterSize > 5)
        clusterSize = 1;

      //odd cls
      if (clusterSize % 2 != 0)
      {
        int clsR = (clusterSize - 1) / 2;
        for (int i = 1; i <= clsR; ++i)
        {
          if (flat1_->fire(1) < averageEfficiency_ && (centralStrip - i > 0))
            cluster_.push_back(std::pair<int, int>(centralStrip - i, time_hit));
          if (flat1_->fire(1) < averageEfficiency_ && (centralStrip + i <= nstrips))
            cluster_.push_back(std::pair<int, int>(centralStrip + i, time_hit));
        }
      }
      //even cls
      if (clusterSize % 2 == 0)
      {
        int clsR = (clusterSize - 2) / 2;
        {
          if (flat1_->fire(1) < averageEfficiency_ && (centralStrip - 1 > 0))
            cluster_.push_back(std::pair<int, int>(centralStrip - 1, time_hit));
          for (int i = 1; i <= clsR; ++i)
          {
            if (flat1_->fire(1) < averageEfficiency_ && (centralStrip - 1 - i > 0))
              cluster_.push_back(std::pair<int, int>(centralStrip - 1 - i, time_hit));
            if (flat1_->fire(1) < averageEfficiency_ && (centralStrip + i <= nstrips))
              cluster_.push_back(std::pair<int, int>(centralStrip + i, time_hit));
          }
        }
      }
      for(auto & digi : cluster_)
      {
        strips_.insert(digi);
      }
    }//end doNoiseCLS_
    else
    {
      std::pair<int, int> digi(centralStrip, time_hit);
      strips_.insert(digi);
    }
  }
  return;
}

std::vector<std::pair<int, int> > ME0SimpleModel::simulateClustering(const ME0EtaPartition* roll, const PSimHit* simHit, const int bx)
{
  const StripTopology& topology = roll->specificTopology();
  const LocalPoint& hit_position(simHit->localPosition());
  const int nstrips(roll->nstrips());

  int centralStrip = 0;
  if (!(topology.channel(hit_position) + 1 > nstrips))
    centralStrip = topology.channel(hit_position) + 1;
  else
    centralStrip = topology.channel(hit_position);

  GlobalPoint pointSimHit = roll->toGlobal(hit_position);
  GlobalPoint pointDigiHit = roll->toGlobal(roll->centreOfStrip(centralStrip));
  double deltaphi = pointSimHit.phi() - pointDigiHit.phi();

  // Add central digi to cluster vector
  std::vector<std::pair<int, int> > cluster_;
  cluster_.clear();
  cluster_.push_back(std::pair<int, int>(centralStrip, bx));

  // get the cluster size
  //     const int clusterSize(static_cast<int>(std::round(poisson_->fire(averageClusterSize_))));
  const int clusterSize(static_cast<int> (std::round(gamma1_->fire(averageClusterSize_, averageClusterSize_))));

  if (abs(simHit->particleType()) != 13 && fabs(simHit->pabs()) < minPabsNoiseCLS_)
    return cluster_;

  //keep cls between [1, 6]
  if (clusterSize < 1 || clusterSize > 5)
    return cluster_;

  //odd cls
  if (clusterSize % 2 != 0)
  {
    int clsR = (clusterSize - 1) / 2;
    for (int i = 1; i <= clsR; ++i)
    {
      if (flat1_->fire(1) < averageEfficiency_ && (centralStrip - i > 0))
        cluster_.push_back(std::pair<int, int>(centralStrip - i, bx));
      if (flat1_->fire(1) < averageEfficiency_ && (centralStrip + i <= nstrips))
        cluster_.push_back(std::pair<int, int>(centralStrip + i, bx));
    }
  }
  //even cls
  if (clusterSize % 2 == 0)
  {
    int clsR = (clusterSize - 2) / 2;
    if (deltaphi <= 0)
    {
      if (flat1_->fire(1) < averageEfficiency_ && (centralStrip - 1 > 0))
        cluster_.push_back(std::pair<int, int>(centralStrip - 1, bx));
      for (int i = 1; i <= clsR; ++i)
      {
        if (flat1_->fire(1) < averageEfficiency_ && (centralStrip - 1 - i > 0))
          cluster_.push_back(std::pair<int, int>(centralStrip - 1 - i, bx));
        if (flat1_->fire(1) < averageEfficiency_ && (centralStrip + i <= nstrips))
          cluster_.push_back(std::pair<int, int>(centralStrip + i, bx));
      }
    }
    else
    {
      if (flat1_->fire(1) < averageEfficiency_ && (centralStrip + 1 <= nstrips))
        cluster_.push_back(std::pair<int, int>(centralStrip + 1, bx));
      for (int i = 1; i <= clsR; ++i)
      {
        if (flat1_->fire(1) < averageEfficiency_ && (centralStrip + 1 + i <= nstrips))
          cluster_.push_back(std::pair<int, int>(centralStrip + 1 + i, bx));
        if (flat1_->fire(1) < averageEfficiency_ && (centralStrip - i < 0))
          cluster_.push_back(std::pair<int, int>(centralStrip - i, bx));
      }
    }
  }

  return cluster_;
}
