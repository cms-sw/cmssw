#include "SimMuon/GEMDigitizer/interface/GEMSimpleModel.h"

#include "Geometry/GEMGeometry/interface/GEMEtaPartitionSpecs.h"
#include "Geometry/CommonTopologies/interface/TrapezoidalStripTopology.h"
#include "Geometry/GEMGeometry/interface/GEMGeometry.h"

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

#include "TMath.h"       /* exp */


namespace
{
  // "magic" parameter for cosmics
  const double COSMIC_PAR(37.62);
}

GEMSimpleModel::GEMSimpleModel(const edm::ParameterSet& config) :
GEMDigiModel(config)
, averageEfficiency_(config.getParameter<double> ("averageEfficiency"))
, averageShapingTime_(config.getParameter<double> ("averageShapingTime"))
, timeResolution_(config.getParameter<double> ("timeResolution"))
, timeJitter_(config.getParameter<double> ("timeJitter"))
, averageNoiseRate_(config.getParameter<double> ("averageNoiseRate"))
//, averageClusterSize_(config.getParameter<double> ("averageClusterSize"))
, clsParametrization_(config.getParameter<std::vector<double>>("clsParametrization"))
, signalPropagationSpeed_(config.getParameter<double> ("signalPropagationSpeed"))
, cosmics_(config.getParameter<bool> ("cosmics"))
, bxwidth_(config.getParameter<int> ("bxwidth"))
, minBunch_(config.getParameter<int> ("minBunch"))
, maxBunch_(config.getParameter<int> ("maxBunch"))
, digitizeOnlyMuons_(config.getParameter<bool> ("digitizeOnlyMuons"))
, doBkgNoise_(config.getParameter<bool> ("doBkgNoise"))
, doNoiseCLS_(config.getParameter<bool> ("doNoiseCLS"))
, fixedRollRadius_(config.getParameter<bool> ("fixedRollRadius"))
, scaleLumi_(config.getParameter<double> ("scaleLumi"))
, simulateElectronBkg_(config.getParameter<bool> ("simulateElectronBkg"))
, constNeuGE11_(config.getParameter<double> ("constNeuGE11"))
, slopeNeuGE11_(config.getParameter<double> ("slopeNeuGE11"))
, GE21NeuBkgParams_(config.getParameter<std::vector<double>>("GE21NeuBkgParams"))
, GE11ElecBkgParams_(config.getParameter<std::vector<double>>("GE11ElecBkgParams"))
, GE21ElecBkgParams_(config.getParameter<std::vector<double>>("GE21ElecBkgParams"))

{

}

GEMSimpleModel::~GEMSimpleModel()
{
  if (flat1_)
    delete flat1_;
  if (flat2_)
    delete flat2_;
  if (flat3_)
    delete flat3_;
  if (flat4_)
    delete flat4_;
  if (poisson_)
    delete poisson_;
  if (gauss1_)
    delete gauss1_;
  if (gauss2_)
    delete gauss2_;
  if (gamma1_)
    delete gamma1_;
}

void GEMSimpleModel::setRandomEngine(CLHEP::HepRandomEngine& eng)
{
  flat1_ = new CLHEP::RandFlat(eng);
  flat2_ = new CLHEP::RandFlat(eng);
  flat3_ = new CLHEP::RandFlat(eng);
  flat4_ = new CLHEP::RandFlat(eng);
  poisson_ = new CLHEP::RandPoissonQ(eng);
  gauss1_ = new CLHEP::RandGaussQ(eng);
  gauss2_ = new CLHEP::RandGaussQ(eng);
  gamma1_ = new CLHEP::RandGamma(eng);
}

void GEMSimpleModel::setup()
{
  return;
}

void GEMSimpleModel::simulateSignal(const GEMEtaPartition* roll, const edm::PSimHitContainer& simHits)
{
  stripDigiSimLinks_.clear();
  detectorHitMap_.clear();
  stripDigiSimLinks_ = StripDigiSimLinks(roll->id().rawId());

  for (edm::PSimHitContainer::const_iterator hit = simHits.begin(); hit != simHits.end(); ++hit)
  {
    if (std::abs(hit->particleType()) != 13 && digitizeOnlyMuons_)
      continue;

    // Check GEM efficiency
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

int GEMSimpleModel::getSimHitBx(const PSimHit* simhit)
{
  int bx = -999;
  const LocalPoint simHitPos(simhit->localPosition());
  const float tof(simhit->timeOfFlight());
  // random Gaussian time correction due to electronics jitter
  const float randomJitterTime(gauss1_->fire(0., timeJitter_));

  const GEMDetId id(simhit->detUnitId());
  const GEMEtaPartition* roll(geometry_->etaPartition(id));

  if (!roll)
  {
    throw cms::Exception("Geometry")<< "GEMSimpleModel::getSimHitBx() - GEM simhit id does not match any GEM roll id: " << id << "\n";
    return 999;
  }

  if (roll->id().region() == 0)
  {
    throw cms::Exception("Geometry") << "GEMSimpleModel::getSimHitBx() - this GEM id is from barrel, which cannot happen: " << roll->id() << "\n";
  }

  // signal propagation speed in vacuum in [m/s]
  const double cspeed = 299792458; 
  const int nstrips = roll->nstrips();
  float middleStrip = nstrips/2.;
  LocalPoint middleOfRoll = roll->centreOfStrip(middleStrip);
  GlobalPoint globMiddleRol = roll->toGlobal(middleOfRoll);
  double muRadius = sqrt(globMiddleRol.x()*globMiddleRol.x() + globMiddleRol.y()*globMiddleRol.y() +globMiddleRol.z()*globMiddleRol.z());
  double timeCalibrationOffset_ = (muRadius *1e+9)/(cspeed*1e+2); //[ns]

  const TrapezoidalStripTopology* top(dynamic_cast<const TrapezoidalStripTopology*> (&(roll->topology())));
  const float halfStripLength(0.5 * top->stripLength());
  const float distanceFromEdge(halfStripLength - simHitPos.y());

  // signal propagation speed in material in [cm/ns]
  double signalPropagationSpeedTrue = signalPropagationSpeed_ * cspeed * 1e+2 * 1e-9;

  // average time for the signal to propagate from the SimHit to the top of a strip
  const float averagePropagationTime(distanceFromEdge / signalPropagationSpeedTrue);
  // random Gaussian time correction due to the finite timing resolution of the detector
  const float randomResolutionTime(gauss2_->fire(0., timeResolution_));

  const float simhitTime(tof + averageShapingTime_ + randomResolutionTime + averagePropagationTime + randomJitterTime);

  float referenceTime = 0.;
  referenceTime = timeCalibrationOffset_ + halfStripLength / signalPropagationSpeedTrue + averageShapingTime_;
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

void GEMSimpleModel::simulateNoise(const GEMEtaPartition* roll)
{
  if (!doBkgNoise_)
  return;

  const GEMDetId gemId(roll->id());
  const int nstrips(roll->nstrips());
  double trArea(0.0);
  double trStripArea(0.0);

  if (gemId.region() == 0)
  {
    throw cms::Exception("Geometry")
        << "GEMSynchronizer::simulateNoise() - this GEM id is from barrel, which cannot happen.";
  }
  const TrapezoidalStripTopology* top_(dynamic_cast<const TrapezoidalStripTopology*> (&(roll->topology())));
  const float striplength(top_->stripLength());
  trStripArea = (roll->pitch()) * striplength;
  trArea = trStripArea * nstrips;
  const int nBxing(maxBunch_ - minBunch_ + 1);

  float rollRadius = 0;
  if(fixedRollRadius_)
  {
    rollRadius = top_->radius();
  }
  else
  {
    double varRad = flat3_->fire(-1.*top_->stripLength()/2., top_->stripLength()/2.);
    rollRadius = top_->radius() + varRad;
  }

//calculate noise from model
  double averageNeutralNoiseRatePerRoll = 0.;
  double averageNoiseElectronRatePerRoll = 0.;
  double averageNoiseRatePerRoll = 0.;

  if(gemId.station() == 1)
  {
//simulate neutral background for GE1/1
    averageNeutralNoiseRatePerRoll = constNeuGE11_ * TMath::Exp(slopeNeuGE11_*rollRadius);

//simulate eletron background for GE1/1
//the product is faster than Power or pow:
    if(simulateElectronBkg_)
    averageNoiseElectronRatePerRoll = GE11ElecBkgParams_[0]
                                    + GE11ElecBkgParams_[1]*rollRadius
                                    + GE11ElecBkgParams_[2]*rollRadius*rollRadius
                                    + GE11ElecBkgParams_[3]*rollRadius*rollRadius*rollRadius;

    averageNoiseRatePerRoll = averageNeutralNoiseRatePerRoll + averageNoiseElectronRatePerRoll;
  }

  if(gemId.station() == 2 || gemId.station() == 3)
  {
//simulate neutral background for GE2/1
    averageNeutralNoiseRatePerRoll = GE21NeuBkgParams_[0]
                                   + GE21NeuBkgParams_[1]*rollRadius
                                   + GE21NeuBkgParams_[2]*rollRadius*rollRadius
                                   + GE21NeuBkgParams_[3]*rollRadius*rollRadius*rollRadius
                                   + GE21NeuBkgParams_[4]*rollRadius*rollRadius*rollRadius*rollRadius
                                   + GE21NeuBkgParams_[5]*rollRadius*rollRadius*rollRadius*rollRadius*rollRadius;


//simulate eletron background for GE2/1
    if(simulateElectronBkg_)
    averageNoiseElectronRatePerRoll = GE21ElecBkgParams_[0]
                                    + GE21ElecBkgParams_[1]*rollRadius
                                    + GE21ElecBkgParams_[2]*rollRadius*rollRadius
                                    + GE21ElecBkgParams_[3]*rollRadius*rollRadius*rollRadius
                                    + GE21ElecBkgParams_[4]*rollRadius*rollRadius*rollRadius*rollRadius
                                    + GE21ElecBkgParams_[5]*rollRadius*rollRadius*rollRadius*rollRadius*rollRadius
                                    + GE21ElecBkgParams_[6]*rollRadius*rollRadius*rollRadius*rollRadius*rollRadius*rollRadius;

    averageNoiseRatePerRoll = averageNeutralNoiseRatePerRoll + averageNoiseElectronRatePerRoll;
  }

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

      int clusterSize = 0;
      double randForCls = flat4_->fire(1);

      if(randForCls <= clsParametrization_[0] && randForCls >= 0.)
        clusterSize = 1;
      else if(randForCls <= clsParametrization_[1] && randForCls > clsParametrization_[0])
        clusterSize = 2;
      else if(randForCls <= clsParametrization_[2] && randForCls > clsParametrization_[1])
        clusterSize = 3;
      else if(randForCls <= clsParametrization_[3] && randForCls > clsParametrization_[2])
        clusterSize = 4;
      else if(randForCls <= clsParametrization_[4] && randForCls > clsParametrization_[3])
        clusterSize = 5;
      else if(randForCls <= clsParametrization_[5] && randForCls > clsParametrization_[4])
        clusterSize = 6;
      else if(randForCls <= clsParametrization_[6] && randForCls > clsParametrization_[5])
        clusterSize = 7;
      else if(randForCls <= clsParametrization_[7] && randForCls > clsParametrization_[6])
        clusterSize = 8;
      else if(randForCls <= clsParametrization_[8] && randForCls > clsParametrization_[7])
        clusterSize = 9;

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
        if(flat3_->fire(1) < 0.5)
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

        else
        {
        if (flat1_->fire(1) < averageEfficiency_ && (centralStrip + 1 <= nstrips))
          cluster_.push_back(std::pair<int, int>(centralStrip + 1, time_hit));
          for (int i = 1; i <= clsR; ++i)
          {
            if (flat1_->fire(1) < averageEfficiency_ && (centralStrip + 1 + i <= nstrips))
            cluster_.push_back(std::pair<int, int>(centralStrip + 1 + i, time_hit));
            if (flat1_->fire(1) < averageEfficiency_ && (centralStrip - i < 0))
            cluster_.push_back(std::pair<int, int>(centralStrip - i, time_hit));
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

std::vector<std::pair<int, int> > GEMSimpleModel::simulateClustering(const GEMEtaPartition* roll,
    const PSimHit* simHit, const int bx)
{
  //  const Topology& topology(roll->specs()->topology());
  const StripTopology& topology = roll->specificTopology();
  //  const LocalPoint& entry(simHit->entryPoint());
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
      int clusterSize = 0;
      double randForCls = flat4_->fire(1);

      if(randForCls <= clsParametrization_[0] && randForCls >= 0.)
        clusterSize = 1;
      else if(randForCls <= clsParametrization_[1] && randForCls > clsParametrization_[0])
        clusterSize = 2;
      else if(randForCls <= clsParametrization_[2] && randForCls > clsParametrization_[1])
        clusterSize = 3;
      else if(randForCls <= clsParametrization_[3] && randForCls > clsParametrization_[2])
        clusterSize = 4;
      else if(randForCls <= clsParametrization_[4] && randForCls > clsParametrization_[3])
        clusterSize = 5;
      else if(randForCls <= clsParametrization_[5] && randForCls > clsParametrization_[4])
        clusterSize = 6;
      else if(randForCls <= clsParametrization_[6] && randForCls > clsParametrization_[5])
        clusterSize = 7;
      else if(randForCls <= clsParametrization_[7] && randForCls > clsParametrization_[6])
        clusterSize = 8;
      else if(randForCls <= clsParametrization_[8] && randForCls > clsParametrization_[7])
        clusterSize = 9;

  if (abs(simHit->particleType()) != 13 && fabs(simHit->pabs()) < minPabsNoiseCLS_)
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


