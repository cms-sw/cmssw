#include "SimMuon/GEMDigitizer/interface/GEMSimpleModel.h"
#include "Geometry/GEMGeometry/interface/GEMEtaPartitionSpecs.h"
#include "Geometry/CommonTopologies/interface/TrapezoidalStripTopology.h"
#include "Geometry/GEMGeometry/interface/GEMGeometry.h"
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

GEMSimpleModel::GEMSimpleModel(const edm::ParameterSet& config) :
GEMDigiModel(config)
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
, simulateLowNeutralRate_(config.getParameter<bool>("simulateLowNeutralRate"))
{
//initialise parameters from the fit:
//params for pol3 model of electron bkg for GE1/1:
  GE11ElecBkgParam0 = 3402.63;
  GE11ElecBkgParam1 = -42.9979;
  GE11ElecBkgParam2 = 0.188475;
  GE11ElecBkgParam3 = -0.0002822;
//params for expo model of electron bkg for GE2/1:
  constElecGE21 = 9.74156e+02;
  slopeElecGE21 = -1.18398e-02;
//Neutral Bkg
//Low Rate model L=10^{34}cm^{-2}s^{-1}
//const and slope for the expo model of neutral bkg for GE1/1:
  constNeuGE11 = 807.;
  slopeNeuGE11 = -0.01443;
//params for the simple pol5 model of neutral bkg for GE2/1:
  GE21NeuBkgParam0 = 2954.04;
  GE21NeuBkgParam1 = -58.7558;
  GE21NeuBkgParam2 = 0.473481;
  GE21NeuBkgParam3 = -0.00188292;
  GE21NeuBkgParam4 = 3.67041e-06;
  GE21NeuBkgParam5 = -2.80261e-09;
//High Rate model L=5x10^{34}cm^{-2}s^{-1}
//params for expo model of neutral bkg for GE1/1:
  constNeuGE11_highRate = 1.02603e+04;
  slopeNeuGE11_highRate = -1.62806e-02;
//params for pol5 model of neutral bkg for GE2/1:
  GE21ModNeuBkgParam0 = 21583.2;
  GE21ModNeuBkgParam1 = -476.59;
  GE21ModNeuBkgParam2 = 4.24037;
  GE21ModNeuBkgParam3 = -0.0185558;
  GE21ModNeuBkgParam4 = 3.97809e-05;
  GE21ModNeuBkgParam5 = -3.34575e-08;
}

GEMSimpleModel::~GEMSimpleModel()
{
}

void GEMSimpleModel::setup()
{
  return;
}

void GEMSimpleModel::simulateSignal(const GEMEtaPartition* roll, const edm::PSimHitContainer& simHits, CLHEP::HepRandomEngine* engine)
{
  stripDigiSimLinks_.clear();
  detectorHitMap_.clear();
  stripDigiSimLinks_ = StripDigiSimLinks(roll->id().rawId());
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
    if (std::abs(hit->particleType()) != 13) //consider all non muon particles with gem efficiency to electrons
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

int GEMSimpleModel::getSimHitBx(const PSimHit* simhit, CLHEP::HepRandomEngine* engine)
{
  int bx = -999;
  const LocalPoint simHitPos(simhit->localPosition());
  const float tof(simhit->timeOfFlight());
  // random Gaussian time correction due to electronics jitter
  float randomJitterTime = CLHEP::RandGaussQ::shoot(engine, 0., timeJitter_);
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
  const double cspeed = 299792458;	 // signal propagation speed in vacuum in [m/s]
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

void GEMSimpleModel::simulateNoise(const GEMEtaPartition* roll, CLHEP::HepRandomEngine* engine)
//void GEMSimpleModel::simulateNoise(const GEMEtaPartition* roll)
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
  const TrapezoidalStripTopology* top_(dynamic_cast<const TrapezoidalStripTopology*>(&(roll->topology())));
  const float striplength(top_->stripLength());
  trStripArea = (roll->pitch()) * striplength;
  trArea = trStripArea * nstrips;
  const int nBxing(maxBunch_ - minBunch_ + 1);
  const float rollRadius(fixedRollRadius_ ? top_->radius() : 
			 top_->radius() + CLHEP::RandFlat::shoot(engine, -1.*top_->stripLength()/2., top_->stripLength()/2.));

//calculate noise from model
  double averageNeutralNoiseRatePerRoll = 0.;
  double averageNoiseElectronRatePerRoll = 0.;
  double averageNoiseRatePerRoll = 0.;
  if (gemId.station() == 1)
  {
//simulate neutral background for GE1/1
    if (simulateLowNeutralRate_)
      averageNeutralNoiseRatePerRoll = constNeuGE11 * TMath::Exp(slopeNeuGE11 * rollRadius);
    else
      averageNeutralNoiseRatePerRoll = constNeuGE11_highRate * TMath::Exp(slopeNeuGE11_highRate * rollRadius);
//simulate electron background for GE1/1
    if (simulateElectronBkg_)
      averageNoiseElectronRatePerRoll = GE11ElecBkgParam0
                                      + GE11ElecBkgParam1 * rollRadius
                                      + GE11ElecBkgParam2 * rollRadius * rollRadius
                                      + GE11ElecBkgParam3 * rollRadius * rollRadius * rollRadius;
    averageNoiseRatePerRoll = averageNeutralNoiseRatePerRoll + averageNoiseElectronRatePerRoll;
  }
  if (gemId.station() == 2 || gemId.station() == 3)
  {
//simulate neutral background for GE2/1
    if (simulateLowNeutralRate_)
      averageNeutralNoiseRatePerRoll = GE21NeuBkgParam0
                                     + GE21NeuBkgParam1 * rollRadius
                                     + GE21NeuBkgParam2 * rollRadius * rollRadius
                                     + GE21NeuBkgParam3 * rollRadius * rollRadius * rollRadius
                                     + GE21NeuBkgParam4 * rollRadius * rollRadius * rollRadius * rollRadius
                                     + GE21NeuBkgParam5 * rollRadius * rollRadius * rollRadius * rollRadius * rollRadius;
    else
      averageNeutralNoiseRatePerRoll = GE21ModNeuBkgParam0
                                     + GE21ModNeuBkgParam1 * rollRadius
                                     + GE21ModNeuBkgParam2 * rollRadius * rollRadius
                                     + GE21ModNeuBkgParam3 * rollRadius * rollRadius * rollRadius
                                     + GE21ModNeuBkgParam4 * rollRadius * rollRadius * rollRadius * rollRadius
                                     + GE21ModNeuBkgParam5 * rollRadius * rollRadius * rollRadius * rollRadius * rollRadius;
//simulate electron background for GE2/1
    if (simulateElectronBkg_)
      averageNoiseElectronRatePerRoll = constElecGE21 * TMath::Exp(slopeElecGE21 * rollRadius);
    averageNoiseRatePerRoll = averageNeutralNoiseRatePerRoll + averageNoiseElectronRatePerRoll;
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

std::vector<std::pair<int, int> > GEMSimpleModel::simulateClustering(const GEMEtaPartition* roll,
    const PSimHit* simHit, const int bx, CLHEP::HepRandomEngine* engine)
{
  const StripTopology& topology = roll->specificTopology(); // const LocalPoint& entry(simHit->entryPoint());
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
