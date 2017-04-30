#include "SimMuon/GEMDigitizer/interface/ME0SimpleModel.h"
#include "Geometry/GEMGeometry/interface/ME0EtaPartitionSpecs.h"
#include "Geometry/CommonTopologies/interface/TrapezoidalStripTopology.h"
#include "Geometry/GEMGeometry/interface/ME0Geometry.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "CLHEP/Random/RandFlat.h"
#include "CLHEP/Random/RandPoissonQ.h"
#include "CLHEP/Random/RandGaussQ.h"
#include <cmath>
#include <utility>
#include <map>
#include "TMath.h"       /* exp */

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
, averageNoiseRate_(config.getParameter<double> ("averageNoiseRate"))
, signalPropagationSpeed_(config.getParameter<double> ("signalPropagationSpeed"))
, cosmics_(config.getParameter<bool> ("cosmics"))
, bxwidth_(config.getParameter<int> ("bxwidth"))
, minBunch_(config.getParameter<int> ("minBunch"))
, maxBunch_(config.getParameter<int> ("maxBunch"))
, digitizeOnlyMuons_(config.getParameter<bool> ("digitizeOnlyMuons"))
, doBkgNoise_(config.getParameter<bool> ("doBkgNoise"))
, doNoiseCLS_(config.getParameter<bool> ("doNoiseCLS"))
, fixedRollRadius_(config.getParameter<bool> ("fixedRollRadius"))
, simulateElectronBkg_(config.getParameter<bool> ("simulateElectronBkg"))
//, simulateLowNeutralRate_(config.getParameter<bool>("simulateLowNeutralRate"))	//nofurther use of this parameter
, instLumi_(config.getParameter<double> ("instLumi"))
, rateFact_(config.getParameter<double> ("rateFact"))
{
//initialise parameters from the fit:
//params for charged background model for ME0 at L=5x10^{34}cm^{-2}s^{-1}
  ME0ElecBkgParam0 = 0.00171409;
  ME0ElecBkgParam1 = 4900.56;
  ME0ElecBkgParam2 = 710909;
  ME0ElecBkgParam3 = -4327.25;

//params for neutral background model for ME0 at L=5x10^{34}cm^{-2}s^{-1}
  ME0NeuBkgParam0 = 0.00386257;
  ME0NeuBkgParam1 = 6344.65;
  ME0NeuBkgParam2 = 16627700;
  ME0NeuBkgParam3 = -102098;
}

ME0SimpleModel::~ME0SimpleModel()
{
}

void ME0SimpleModel::setup()
{
  return;
}

void ME0SimpleModel::simulateSignal(const ME0EtaPartition* roll, const edm::PSimHitContainer& simHits, CLHEP::HepRandomEngine* engine)
{
  stripDigiSimLinks_.clear();
  detectorHitMap_.clear();
  stripDigiSimLinks_ = StripDigiSimLinks(roll->id().rawId());

  theME0DigiSimLinks_.clear();
  theME0DigiSimLinks_ = ME0DigiSimLinks(roll->id().rawId());
  bool digiMuon = false;
  bool digiElec = false;
  for (edm::PSimHitContainer::const_iterator hit = simHits.begin(); hit != simHits.end(); ++hit)
  {
    if (std::abs(hit->particleType()) != 13 && digitizeOnlyMuons_)
      continue;
    double elecEff = 0.;
    double partMom = hit->pabs();
    double checkMuonEff = CLHEP::RandFlat::shoot(engine, 0., 1.);
    double checkElecEff = CLHEP::RandFlat::shoot(engine, 0., 1.);
    if (std::abs(hit->particleType()) == 13 && checkMuonEff < averageEfficiency_)
      digiMuon = true;
    if (std::abs(hit->particleType()) != 13) //consider all non muon particles with me0 efficiency to electrons
    {
      if (partMom <= 1.95e-03)
        elecEff = 1.7e-05 * TMath::Exp(2.1 * partMom * 1000.);
      if (partMom > 1.95e-03 && partMom < 10.e-03)
        elecEff = 1.34 * log(7.96e-01 * partMom * 1000. - 5.75e-01)
            / (1.34 + log(7.96e-01 * partMom * 1000. - 5.75e-01));
      if (partMom > 10.e-03)
        elecEff = 1.;
      if (checkElecEff < elecEff)
        digiElec = true;
    }
   if (!(digiMuon || digiElec))
      continue;
    const int bx(getSimHitBx(&(*hit), engine));
    const std::vector<std::pair<int, int> > cluster(simulateClustering(roll, &(*hit), bx, engine));
    for  (auto & digi : cluster)
    {
      detectorHitMap_.insert(DetectorHitMap::value_type(digi,&*(hit)));
      strips_.insert(digi);
    }
  }
}

int ME0SimpleModel::getSimHitBx(const PSimHit* simhit, CLHEP::HepRandomEngine* engine)
{
  int bx = -999;
  const LocalPoint simHitPos(simhit->localPosition());
  const float tof(simhit->timeOfFlight());
  // random Gaussian time correction due to electronics jitter
  float randomJitterTime = CLHEP::RandGaussQ::shoot(engine, 0., timeJitter_);
  const ME0DetId id(simhit->detUnitId());
  const ME0EtaPartition* roll(geometry_->etaPartition(id));
  if (!roll)
  {
    throw cms::Exception("Geometry")<< "ME0SimpleModel::getSimHitBx() - ME0 simhit id does not match any ME0 roll id: " << id << "\n";
    return 999;
  }
  if (roll->id().region() == 0)
  {
    throw cms::Exception("Geometry") << "ME0SimpleModel::getSimHitBx() - this ME0 id is from barrel, which cannot happen: " << roll->id() << "\n";
  }
  const double cspeed = 299792458;   // signal propagation speed in vacuum in [m/s]
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
  double signalPropagationSpeedTrue = signalPropagationSpeed_ * cspeed * 1e-7;  // 1e+2 * 1e-9;

  // average time for the signal to propagate from the SimHit to the top of a strip
  const float averagePropagationTime(distanceFromEdge / signalPropagationSpeedTrue);
  // random Gaussian time correction due to the finite timing resolution of the detector
  float randomResolutionTime = CLHEP::RandGaussQ::shoot(engine, 0., timeResolution_);
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

void ME0SimpleModel::simulateNoise(const ME0EtaPartition* roll, CLHEP::HepRandomEngine* engine)
{
  if (!doBkgNoise_)
    return;
  const ME0DetId me0Id(roll->id());
  const int nstrips(roll->nstrips());
  double trArea(0.0);
  double trStripArea(0.0);

  if (me0Id.region() == 0)
  {
    throw cms::Exception("Geometry")
        << "ME0Synchronizer::simulateNoise() - this ME0 id is from barrel, which cannot happen.";
  }
  const TrapezoidalStripTopology* top_(dynamic_cast<const TrapezoidalStripTopology*>(&(roll->topology())));
  const float striplength(top_->stripLength());
  trStripArea = (roll->pitch()) * striplength;
  trArea = trStripArea * nstrips;
  const int nBxing(maxBunch_ - minBunch_ + 1);
  const float rollRadius(fixedRollRadius_ ? top_->radius() : 
       top_->radius() + CLHEP::RandFlat::shoot(engine, -1.*top_->stripLength()/2., top_->stripLength()/2.));

  const float rSqrtR = rollRadius * sqrt(rollRadius);

//calculate noise from model
  double averageNeutralNoiseRatePerRoll = 0.;
  double averageNoiseElectronRatePerRoll = 0.;
  double averageNoiseRatePerRoll = 0.;
  if (me0Id.station() != 1)
  {
    throw cms::Exception("Geometry") << "ME0SimpleModel::simulateNoise() - this ME0 id is from station 1, which cannot happen: " <<roll->id() << "\n";
  }
  else
  {
    averageNeutralNoiseRatePerRoll = ME0NeuBkgParam0 * rollRadius* TMath::Exp(ME0NeuBkgParam1/rSqrtR) + ME0NeuBkgParam2/rSqrtR + ME0NeuBkgParam3/(sqrt(rollRadius));

//simulate electron background for ME0
    if (simulateElectronBkg_)
    averageNoiseElectronRatePerRoll = ME0ElecBkgParam0 * rSqrtR* TMath::Exp(ME0ElecBkgParam1/rSqrtR) + ME0ElecBkgParam2/rSqrtR + ME0ElecBkgParam3/(sqrt(rollRadius));

    averageNoiseRatePerRoll = averageNeutralNoiseRatePerRoll + averageNoiseElectronRatePerRoll;
    averageNoiseRatePerRoll *= instLumi_*rateFact_*1.0/5;  
  }
  
//simulate intrinsic noise
  if(simulateIntrinsicNoise_)
  {
    const double aveIntrinsicNoisePerStrip(averageNoiseRate_ * nBxing * bxwidth_ * trStripArea * 1.0e-9);
    for(int j = 0; j < nstrips; ++j)
    {
      CLHEP::RandPoissonQ randPoissonQ(*engine, aveIntrinsicNoisePerStrip);
      const int n_intrHits(randPoissonQ.fire());
    
      for (int k = 0; k < n_intrHits; k++ )
      {
        const int time_hit(static_cast<int>(CLHEP::RandFlat::shoot(engine, nBxing)) + minBunch_);
        std::pair<int, int> digi(k+1,time_hit);
        strips_.insert(digi);
      }
    }
  }//end simulate intrinsic noise
//simulate bkg contribution
  const double averageNoise(averageNoiseRatePerRoll * nBxing * bxwidth_ * trArea * 1.0e-9);
  CLHEP::RandPoissonQ randPoissonQ(*engine, averageNoise);
  const int n_hits(randPoissonQ.fire());

  for (int i = 0; i < n_hits; ++i)
  {
    const int centralStrip(static_cast<int> (CLHEP::RandFlat::shoot(engine, 1, nstrips)));
    const int time_hit(static_cast<int>(CLHEP::RandFlat::shoot(engine, nBxing)) + minBunch_);
    if (doNoiseCLS_)
    {
      std::vector < std::pair<int, int> > cluster_;
      cluster_.clear();
      cluster_.push_back(std::pair<int, int>(centralStrip, time_hit));
      int clusterSize((CLHEP::RandFlat::shoot(engine)) <= 0.53 ? 1 : 2);
      if (clusterSize == 2)
      {
        if(CLHEP::RandFlat::shoot(engine) < 0.5)
        {
          if (CLHEP::RandFlat::shoot(engine) < averageEfficiency_ && (centralStrip - 1 > 0))
            cluster_.push_back(std::pair<int, int>(centralStrip - 1, time_hit));
        }
        else
        {
          if (CLHEP::RandFlat::shoot(engine) < averageEfficiency_ && (centralStrip + 1 <= nstrips))
            cluster_.push_back(std::pair<int, int>(centralStrip + 1, time_hit));
        }
      }
      for (auto & digi : cluster_)
      {
        strips_.insert(digi);
      }
    } //end doNoiseCLS_
    else
    {
      std::pair<int, int> digi(centralStrip, time_hit);
      strips_.insert(digi);
    }
  }
  return;
}

std::vector<std::pair<int, int> > ME0SimpleModel::simulateClustering(const ME0EtaPartition* roll,
    const PSimHit* simHit, const int bx, CLHEP::HepRandomEngine* engine)
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
  double deltaX = pointSimHit.x() - pointDigiHit.x();
// Add central digi to cluster vector
  std::vector < std::pair<int, int> > cluster_;
  cluster_.clear();
  cluster_.push_back(std::pair<int, int>(centralStrip, bx));
//simulate cross talk
  int clusterSize((CLHEP::RandFlat::shoot(engine)) <= 0.53 ? 1 : 2);
  if (clusterSize == 2)
  {
    if (deltaX <= 0)
    {
      if (CLHEP::RandFlat::shoot(engine) < averageEfficiency_ && (centralStrip - 1 > 0))
        cluster_.push_back(std::pair<int, int>(centralStrip - 1, bx));
    }
    else
    {
      if (CLHEP::RandFlat::shoot(engine) < averageEfficiency_ && (centralStrip + 1 <= nstrips))
        cluster_.push_back(std::pair<int, int>(centralStrip + 1, bx));
    }
  }
  return cluster_;
}
