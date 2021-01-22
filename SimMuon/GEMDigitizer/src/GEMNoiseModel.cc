#include "SimMuon/GEMDigitizer/interface/GEMNoiseModel.h"
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

GEMNoiseModel::GEMNoiseModel(const edm::ParameterSet& config)
    : GEMDigiModel(config),
      averageNoiseRate_(config.getParameter<double>("averageNoiseRate")),
      bxWidth_(config.getParameter<double>("bxWidth")),
      minBunch_(config.getParameter<int>("minBunch")),
      maxBunch_(config.getParameter<int>("maxBunch")) {}

GEMNoiseModel::~GEMNoiseModel() {}

void GEMNoiseModel::simulate(const GEMEtaPartition* roll,
                             const edm::PSimHitContainer&,
                             CLHEP::HepRandomEngine* engine,
                             Strips& strips_,
                             DetectorHitMap& detectorHitMap_) {
  const GEMDetId& gemId(roll->id());
  const int nstrips(roll->nstrips());
  double trStripArea(0.0);
  if (gemId.region() == 0) {
    throw cms::Exception("Geometry") << "GEMNoiseModel::simulate() - this GEM id is from barrel, which cannot happen.";
  }
  const GEMStripTopology* top_(dynamic_cast<const GEMStripTopology*>(&(roll->topology())));
  const float striplength(top_->stripLength());
  trStripArea = (roll->pitch()) * striplength;
  float trArea(trStripArea * nstrips);
  const int nBxing(maxBunch_ - minBunch_ + 1);
  //simulate intrinsic noise
  const double aveIntrinsicNoise(averageNoiseRate_ * nBxing * trArea * bxWidth_);
  CLHEP::RandPoissonQ randPoissonQ(*engine, aveIntrinsicNoise);
  const int n_intrHits(randPoissonQ.fire());

  for (int k = 0; k < n_intrHits; k++) {
    const int centralStrip(static_cast<int>(CLHEP::RandFlat::shoot(engine, 1, nstrips)));
    const int time_hit(static_cast<int>(CLHEP::RandFlat::shoot(engine, nBxing)) + minBunch_);
    strips_.emplace(centralStrip, time_hit);
  }
  //end simulate intrinsic noise

  return;
}
