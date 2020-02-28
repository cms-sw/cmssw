#include "CLHEP/Units/defs.h"
#include "CLHEP/Units/SystemOfUnits.h"
#include "CLHEP/Units/GlobalPhysicalConstants.h"
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

ME0SimpleModel::ME0SimpleModel(const edm::ParameterSet& config)
    : ME0DigiModel(config),
      averageEfficiency_(config.getParameter<double>("averageEfficiency")),
      averageShapingTime_(config.getParameter<double>("averageShapingTime")),
      timeResolution_(config.getParameter<double>("timeResolution")),
      timeJitter_(config.getParameter<double>("timeJitter")),
      averageNoiseRate_(config.getParameter<double>("averageNoiseRate")),
      signalPropagationSpeed_(config.getParameter<double>("signalPropagationSpeed")),
      bxwidth_(config.getParameter<int>("bxwidth")),
      minBunch_(config.getParameter<int>("minBunch")),
      maxBunch_(config.getParameter<int>("maxBunch")),
      digitizeOnlyMuons_(config.getParameter<bool>("digitizeOnlyMuons")),
      doBkgNoise_(config.getParameter<bool>("doBkgNoise")),
      doNoiseCLS_(config.getParameter<bool>("doNoiseCLS")),
      fixedRollRadius_(config.getParameter<bool>("fixedRollRadius")),
      simulateElectronBkg_(config.getParameter<bool>("simulateElectronBkg")),
      instLumi_(config.getParameter<double>("instLumi")),
      rateFact_(config.getParameter<double>("rateFact")),
      referenceInstLumi_(config.getParameter<double>("referenceInstLumi")),
      ME0ElecBkgParam0_(config.getParameter<double>("ME0ElecBkgParam0")),
      ME0ElecBkgParam1_(config.getParameter<double>("ME0ElecBkgParam1")),
      ME0ElecBkgParam2_(config.getParameter<double>("ME0ElecBkgParam2")),
      ME0ElecBkgParam3_(config.getParameter<double>("ME0ElecBkgParam3")),
      ME0NeuBkgParam0_(config.getParameter<double>("ME0NeuBkgParam0")),
      ME0NeuBkgParam1_(config.getParameter<double>("ME0NeuBkgParam1")),
      ME0NeuBkgParam2_(config.getParameter<double>("ME0NeuBkgParam2")),
      ME0NeuBkgParam3_(config.getParameter<double>("ME0NeuBkgParam3")) {}

ME0SimpleModel::~ME0SimpleModel() {}

void ME0SimpleModel::setup() { return; }

void ME0SimpleModel::simulateSignal(const ME0EtaPartition* roll,
                                    const edm::PSimHitContainer& simHits,
                                    CLHEP::HepRandomEngine* engine) {
  stripDigiSimLinks_.clear();
  detectorHitMap_.clear();
  stripDigiSimLinks_ = StripDigiSimLinks(roll->id().rawId());

  theME0DigiSimLinks_.clear();
  theME0DigiSimLinks_ = ME0DigiSimLinks(roll->id().rawId());
  bool digiMuon = false;
  bool digiElec = false;
  for (const auto& hit : simHits) {
    if (std::abs(hit.particleType()) != 13 && digitizeOnlyMuons_)
      continue;
    double elecEff = 0.;
    double partMom = hit.pabs();
    double checkMuonEff = CLHEP::RandFlat::shoot(engine, 0., 1.);
    double checkElecEff = CLHEP::RandFlat::shoot(engine, 0., 1.);
    if (std::abs(hit.particleType()) == 13 && checkMuonEff < averageEfficiency_)
      digiMuon = true;
    if (std::abs(hit.particleType()) != 13)  //consider all non muon particles with me0 efficiency to electrons
    {
      if (partMom <= 1.95e-03)
        elecEff = 1.7e-05 * std::exp(2.1 * partMom * 1000.);
      if (partMom > 1.95e-03 && partMom < 10.e-03)
        elecEff =
            1.34 * log(7.96e-01 * partMom * 1000. - 5.75e-01) / (1.34 + log(7.96e-01 * partMom * 1000. - 5.75e-01));
      if (partMom > 10.e-03)
        elecEff = 1.;
      if (checkElecEff < elecEff)
        digiElec = true;
    }
    if (!(digiMuon || digiElec))
      continue;
    const int bx(getSimHitBx(&hit, engine));
    const std::vector<std::pair<int, int> >& cluster(simulateClustering(roll, &hit, bx, engine));
    for (const auto& digi : cluster) {
      detectorHitMap_.emplace(digi, &hit);
      strips_.emplace(digi);
    }
  }
}

int ME0SimpleModel::getSimHitBx(const PSimHit* simhit, CLHEP::HepRandomEngine* engine) {
  int bx = -999;
  const LocalPoint& simHitPos(simhit->localPosition());
  const float tof(simhit->timeOfFlight());
  // random Gaussian time correction due to electronics jitter
  float randomJitterTime = CLHEP::RandGaussQ::shoot(engine, 0., timeJitter_);
  const ME0DetId& id(simhit->detUnitId());
  const ME0EtaPartition* roll(geometry_->etaPartition(id));
  if (!roll) {
    throw cms::Exception("Geometry") << "ME0SimpleModel::getSimHitBx() - ME0 simhit id does not match any ME0 roll id: "
                                     << id << "\n";
    return 999;
  }
  if (roll->id().region() == 0) {
    throw cms::Exception("Geometry")
        << "ME0SimpleModel::getSimHitBx() - this ME0 id is from barrel, which cannot happen: " << roll->id() << "\n";
  }
  const int nstrips = roll->nstrips();
  float middleStrip = nstrips / 2.;
  const LocalPoint& middleOfRoll = roll->centreOfStrip(middleStrip);
  const GlobalPoint& globMiddleRol = roll->toGlobal(middleOfRoll);
  double muRadius = sqrt(globMiddleRol.x() * globMiddleRol.x() + globMiddleRol.y() * globMiddleRol.y() +
                         globMiddleRol.z() * globMiddleRol.z());
  double timeCalibrationOffset_ = (muRadius * CLHEP::ns * CLHEP::cm) / (CLHEP::c_light);  //[cm/ns]
  const TrapezoidalStripTopology* top(dynamic_cast<const TrapezoidalStripTopology*>(&(roll->topology())));
  const float halfStripLength(0.5 * top->stripLength());
  const float distanceFromEdge(halfStripLength - simHitPos.y());

  // signal propagation speed in material in [cm/ns]
  double signalPropagationSpeedTrue = signalPropagationSpeed_ * CLHEP::c_light / (CLHEP::ns * CLHEP::cm);
  // average time for the signal to propagate from the SimHit to the top of a strip
  const float averagePropagationTime(distanceFromEdge / signalPropagationSpeedTrue);
  // random Gaussian time correction due to the finite timing resolution of the detector
  float randomResolutionTime = CLHEP::RandGaussQ::shoot(engine, 0., timeResolution_);
  const float simhitTime(tof + averageShapingTime_ + randomResolutionTime + averagePropagationTime + randomJitterTime);
  float referenceTime = 0.;
  referenceTime = timeCalibrationOffset_ + halfStripLength / signalPropagationSpeedTrue + averageShapingTime_;
  const float timeDifference(simhitTime - referenceTime);
  // assign the bunch crossing
  bx = static_cast<int>(std::round((timeDifference) / bxwidth_));

  // check time
  const bool debug(false);
  if (debug) {
    LogDebug("ME0SimpleModel") << "checktime "
                               << "bx = " << bx << "\tdeltaT = " << timeDifference << "\tsimT =  " << simhitTime
                               << "\trefT =  " << referenceTime << "\ttof = " << tof
                               << "\tavePropT =  " << averagePropagationTime
                               << "\taveRefPropT = " << halfStripLength / signalPropagationSpeedTrue << std::endl;
  }
  return bx;
}

void ME0SimpleModel::simulateNoise(const ME0EtaPartition* roll, CLHEP::HepRandomEngine* engine) {
  if (!doBkgNoise_)
    return;
  const ME0DetId me0Id(roll->id());
  const int nstrips(roll->nstrips());
  double trArea(0.0);
  double trStripArea(0.0);

  if (me0Id.region() == 0) {
    throw cms::Exception("Geometry")
        << "ME0Synchronizer::simulateNoise() - this ME0 id is from barrel, which cannot happen.";
  }
  const TrapezoidalStripTopology* top_(dynamic_cast<const TrapezoidalStripTopology*>(&(roll->topology())));
  const float striplength(top_->stripLength());
  trStripArea = (roll->pitch()) * striplength;
  trArea = trStripArea * nstrips;
  const int nBxing(maxBunch_ - minBunch_ + 1);
  const float rollRadius(
      fixedRollRadius_
          ? top_->radius()
          : top_->radius() + CLHEP::RandFlat::shoot(engine, -1. * top_->stripLength() / 2., top_->stripLength() / 2.));

  const float rSqrtR = rollRadius * sqrt(rollRadius);

  //calculate noise from model
  double averageNeutralNoiseRatePerRoll = 0.;
  double averageNoiseElectronRatePerRoll = 0.;
  double averageNoiseRatePerRoll = 0.;
  if (me0Id.station() != 1) {
    throw cms::Exception("Geometry")
        << "ME0SimpleModel::simulateNoise() - this ME0 id is from station 1, which cannot happen: " << roll->id()
        << "\n";
  } else {
    averageNeutralNoiseRatePerRoll = ME0NeuBkgParam0_ * rollRadius * std::exp(ME0NeuBkgParam1_ / rSqrtR) +
                                     ME0NeuBkgParam2_ / rSqrtR + ME0NeuBkgParam3_ / (sqrt(rollRadius));
    //simulate electron background for ME0
    if (simulateElectronBkg_)
      averageNoiseElectronRatePerRoll = ME0ElecBkgParam0_ * rSqrtR * std::exp(ME0ElecBkgParam1_ / rSqrtR) +
                                        ME0ElecBkgParam2_ / rSqrtR + ME0ElecBkgParam3_ / (sqrt(rollRadius));
    averageNoiseRatePerRoll = averageNeutralNoiseRatePerRoll + averageNoiseElectronRatePerRoll;
    averageNoiseRatePerRoll *= instLumi_ * rateFact_ * 1.0 / referenceInstLumi_;
  }

  //simulate intrinsic noise
  if (simulateIntrinsicNoise_) {
    const double aveIntrinsicNoisePerStrip(averageNoiseRate_ * nBxing * bxwidth_ * trStripArea * 1.0e-9);
    for (int j = 0; j < nstrips; ++j) {
      int randPoissonQ = CLHEP::RandPoissonQ::shoot(engine, aveIntrinsicNoisePerStrip);

      for (int k = 0; k < randPoissonQ; k++) {
        const int time_hit(static_cast<int>(CLHEP::RandFlat::shoot(engine, nBxing)) + minBunch_);
        strips_.emplace(k + 1, time_hit);
      }
    }
  }  //end simulate intrinsic noise

  //simulate bkg contribution
  const double averageNoise(averageNoiseRatePerRoll * nBxing * bxwidth_ * trArea * 1.0e-9);
  int randPoissonQ = CLHEP::RandPoissonQ::shoot(engine, averageNoise);

  for (int i = 0; i < randPoissonQ; ++i) {
    const int centralStrip(static_cast<int>(CLHEP::RandFlat::shoot(engine, 1, nstrips)));
    const int time_hit(static_cast<int>(CLHEP::RandFlat::shoot(engine, nBxing)) + minBunch_);
    if (doNoiseCLS_) {
      std::vector<std::pair<int, int> > cluster;
      cluster.emplace_back(centralStrip, time_hit);
      int clusterSize((CLHEP::RandFlat::shoot(engine)) <= 0.53 ? 1 : 2);
      if (clusterSize == 2) {
        if (CLHEP::RandFlat::shoot(engine) < 0.5) {
          if (CLHEP::RandFlat::shoot(engine) < averageEfficiency_ && (centralStrip - 1 > 0))
            cluster.emplace_back(centralStrip - 1, time_hit);
        } else {
          if (CLHEP::RandFlat::shoot(engine) < averageEfficiency_ && (centralStrip + 1 <= nstrips))
            cluster.emplace_back(centralStrip + 1, time_hit);
        }
      }
      for (const auto& digi : cluster) {
        strips_.emplace(digi);
      }
    }  //end doNoiseCLS_
    else {
      strips_.emplace(centralStrip, time_hit);
    }
  }
  return;
}

std::vector<std::pair<int, int> > ME0SimpleModel::simulateClustering(const ME0EtaPartition* roll,
                                                                     const PSimHit* simHit,
                                                                     const int bx,
                                                                     CLHEP::HepRandomEngine* engine) {
  const StripTopology& topology = roll->specificTopology();
  const LocalPoint& hit_position(simHit->localPosition());
  const int nstrips(roll->nstrips());
  int centralStrip = 0;
  if (!(topology.channel(hit_position) + 1 > nstrips))
    centralStrip = topology.channel(hit_position) + 1;
  else
    centralStrip = topology.channel(hit_position);
  const GlobalPoint& pointSimHit = roll->toGlobal(hit_position);
  const GlobalPoint& pointDigiHit = roll->toGlobal(roll->centreOfStrip(centralStrip));
  double deltaX = pointSimHit.x() - pointDigiHit.x();

  // Add central digi to cluster vector
  std::vector<std::pair<int, int> > cluster;
  cluster.clear();
  cluster.emplace_back(centralStrip, bx);

  //simulate cross talk
  int clusterSize((CLHEP::RandFlat::shoot(engine)) <= 0.53 ? 1 : 2);
  if (clusterSize == 2) {
    if (deltaX <= 0) {
      if (CLHEP::RandFlat::shoot(engine) < averageEfficiency_ && (centralStrip - 1 > 0))
        cluster.emplace_back(centralStrip - 1, bx);
    } else {
      if (CLHEP::RandFlat::shoot(engine) < averageEfficiency_ && (centralStrip + 1 <= nstrips))
        cluster.emplace_back(centralStrip + 1, bx);
    }
  }
  return cluster;
}
