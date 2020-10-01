#include "SimMuon/GEMDigitizer/interface/GEMBkgModel.h"
#include "Geometry/GEMGeometry/interface/GEMEtaPartitionSpecs.h"
#include "Geometry/CommonTopologies/interface/GEMStripTopology.h"
#include "Geometry/GEMGeometry/interface/GEMGeometry.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "CLHEP/Random/RandFlat.h"
#include "CLHEP/Random/RandPoissonQ.h"
#include "CLHEP/Random/RandGaussQ.h"
#include <cmath>
#include <utility>
#include <map>

GEMBkgModel::GEMBkgModel(const edm::ParameterSet& config)
    : GEMDigiModel(config),
      clusterSizeCut(0.53),
      averageEfficiency_(config.getParameter<double>("averageEfficiency")),
      minBunch_(config.getParameter<int>("minBunch")),
      maxBunch_(config.getParameter<int>("maxBunch")),
      simulateNoiseCLS_(config.getParameter<bool>("simulateNoiseCLS")),
      fixedRollRadius_(config.getParameter<bool>("fixedRollRadius")),
      simulateElectronBkg_(config.getParameter<bool>("simulateElectronBkg")),
      instLumi_(config.getParameter<double>("instLumi")),
      rateFact_(config.getParameter<double>("rateFact")),
      bxWidth_(config.getParameter<double>("bxWidth")),
      referenceInstLumi_(config.getParameter<double>("referenceInstLumi")),
      GE11ElecBkgParam0_(config.getParameter<double>("GE11ElecBkgParam0")),
      GE11ElecBkgParam1_(config.getParameter<double>("GE11ElecBkgParam1")),
      GE11ElecBkgParam2_(config.getParameter<double>("GE11ElecBkgParam2")),
      GE21ElecBkgParam0_(config.getParameter<double>("GE21ElecBkgParam0")),
      GE21ElecBkgParam1_(config.getParameter<double>("GE21ElecBkgParam1")),
      GE21ElecBkgParam2_(config.getParameter<double>("GE21ElecBkgParam2")),
      GE11ModNeuBkgParam0_(config.getParameter<double>("GE11ModNeuBkgParam0")),
      GE11ModNeuBkgParam1_(config.getParameter<double>("GE11ModNeuBkgParam1")),
      GE11ModNeuBkgParam2_(config.getParameter<double>("GE11ModNeuBkgParam2")),
      GE21ModNeuBkgParam0_(config.getParameter<double>("GE11ModNeuBkgParam0")),
      GE21ModNeuBkgParam1_(config.getParameter<double>("GE11ModNeuBkgParam1")),
      GE21ModNeuBkgParam2_(config.getParameter<double>("GE11ModNeuBkgParam2")) {}

GEMBkgModel::~GEMBkgModel() {}

void GEMBkgModel::simulate(const GEMEtaPartition* roll,
                           const edm::PSimHitContainer&,
                           CLHEP::HepRandomEngine* engine,
                           Strips& strips_,
                           DetectorHitMap& detectorHitMap_) {
  const GEMDetId& gemId(roll->id());
  const int nstrips(roll->nstrips());
  double trArea(0.0);
  double trStripArea(0.0);
  if (gemId.region() == 0) {
    throw cms::Exception("Geometry") << "GEMBkgModel::simulate() - this GEM id is from barrel, which cannot happen.";
  }
  const GEMStripTopology* top_(dynamic_cast<const GEMStripTopology*>(&(roll->topology())));
  const float striplength(top_->stripLength());
  trStripArea = (roll->pitch()) * striplength;
  trArea = trStripArea * nstrips;
  const int nBxing(maxBunch_ - minBunch_ + 1);
  const float rollRadius(
      fixedRollRadius_
          ? top_->radius()
          : top_->radius() + CLHEP::RandFlat::shoot(engine, -1. * top_->stripLength() / 2., top_->stripLength() / 2.));

  //calculate noise from model
  double averageNeutralNoiseRatePerRoll = 0.;
  double averageNoiseElectronRatePerRoll = 0.;
  double averageNoiseRatePerRoll = 0.;
  if (gemId.station() == 1) {
    //simulate neutral background for GE1/1
    averageNeutralNoiseRatePerRoll =
        (GE11ModNeuBkgParam0_ + GE11ModNeuBkgParam1_ * rollRadius +
         GE11ModNeuBkgParam2_ * rollRadius * rollRadius);  //simulate electron background for GE1/1
    if (simulateElectronBkg_)
      averageNoiseElectronRatePerRoll =
          (GE11ElecBkgParam0_ + GE11ElecBkgParam1_ * rollRadius + GE11ElecBkgParam2_ * rollRadius * rollRadius);

    // Scale up/down for desired instantaneous lumi (reference is 5E34, double from config is in units of 1E34)
    averageNoiseRatePerRoll = averageNeutralNoiseRatePerRoll + averageNoiseElectronRatePerRoll;
    averageNoiseRatePerRoll *= instLumi_ * rateFact_ / referenceInstLumi_;
  } else if (gemId.station() == 2) {
    //simulate neutral background for GE2/1
    averageNeutralNoiseRatePerRoll =
        (GE21ModNeuBkgParam0_ + GE21ModNeuBkgParam1_ * rollRadius + GE21ModNeuBkgParam2_ * rollRadius * rollRadius);
    //simulate electron background for GE2/1
    if (simulateElectronBkg_)
      averageNoiseElectronRatePerRoll =
          (GE21ElecBkgParam0_ + GE21ElecBkgParam1_ * rollRadius + GE21ElecBkgParam2_ * rollRadius * rollRadius);

    // Scale up/down for desired instantaneous lumi (reference is 5E34, double from config is in units of 1E34)
    averageNoiseRatePerRoll = averageNeutralNoiseRatePerRoll + averageNoiseElectronRatePerRoll;
    averageNoiseRatePerRoll *= instLumi_ * rateFact_ / referenceInstLumi_;
  }

  //simulate bkg contribution
  const double averageNoise(averageNoiseRatePerRoll * nBxing * trArea * bxWidth_);
  CLHEP::RandPoissonQ randPoissonQ(*engine, averageNoise);
  const int n_hits(randPoissonQ.fire());
  for (int i = 0; i < n_hits; ++i) {
    const int centralStrip(static_cast<int>(CLHEP::RandFlat::shoot(engine, 1, nstrips)));
    const int time_hit(static_cast<int>(CLHEP::RandFlat::shoot(engine, nBxing)) + minBunch_);
    if (simulateNoiseCLS_) {
      std::vector<std::pair<int, int> > cluster_;
      cluster_.clear();
      cluster_.emplace_back(centralStrip, time_hit);
      int clusterSize((CLHEP::RandFlat::shoot(engine)) <= clusterSizeCut ? 1 : 2);
      if (clusterSize == 2) {
        if (CLHEP::RandFlat::shoot(engine) < 0.5) {
          if (CLHEP::RandFlat::shoot(engine) < averageEfficiency_ && (centralStrip - 1 > 0))
            cluster_.emplace_back(centralStrip - 1, time_hit);
        } else {
          if (CLHEP::RandFlat::shoot(engine) < averageEfficiency_ && (centralStrip + 1 <= nstrips))
            cluster_.emplace_back(centralStrip + 1, time_hit);
        }
      }
      for (const auto& digi : cluster_) {
        strips_.emplace(digi);
      }
    }  //end simulateNoiseCLS_
    else {
      strips_.emplace(centralStrip, time_hit);
    }
  }
  return;
}
