#include "SimMuon/GEMDigitizer/interface/ME0PreRecoGaussianModel.h"
#include "Geometry/GEMGeometry/interface/ME0EtaPartitionSpecs.h"
#include "Geometry/CommonTopologies/interface/TrapezoidalStripTopology.h"
#include "Geometry/GEMGeometry/interface/ME0Geometry.h"
#include "DataFormats/GEMDigi/interface/ME0DigiPreReco.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CLHEP/Random/RandomEngine.h"
#include "CLHEP/Random/RandFlat.h"
#include "CLHEP/Random/RandGaussQ.h"
#include "CLHEP/Random/RandPoissonQ.h"

#include <cmath>
#include <utility>
#include <map>

ME0PreRecoGaussianModel::ME0PreRecoGaussianModel(const edm::ParameterSet& config) :
  ME0DigiPreRecoModel(config)
, sigma_t(config.getParameter<double> ("timeResolution"))
, sigma_u(config.getParameter<double> ("phiResolution"))
, sigma_v(config.getParameter<double> ("etaResolution"))
, corr(config.getParameter<bool> ("useCorrelation"))
, etaproj(config.getParameter<bool> ("useEtaProjectiveGEO"))
, digitizeOnlyMuons_(config.getParameter<bool> ("digitizeOnlyMuons"))
, averageEfficiency_(config.getParameter<double> ("averageEfficiency"))
, doBkgNoise_(config.getParameter<bool> ("doBkgNoise"))
, simulateIntrinsicNoise_(config.getParameter<bool> ("simulateIntrinsicNoise"))
, averageNoiseRate_(config.getParameter<double> ("averageNoiseRate"))
, bxwidth_(config.getParameter<int> ("bxwidth"))
, minBunch_(config.getParameter<int> ("minBunch"))
, maxBunch_(config.getParameter<int> ("maxBunch"))

{
  //params for the simple pol6 model of neutral bkg for ME0:
  ME0ModNeuBkgParam0 = 18883;
  ME0ModNeuBkgParam1 = -553.325;
  ME0ModNeuBkgParam2 = 7.2999;
  ME0ModNeuBkgParam3 = -0.0528206;
  ME0ModNeuBkgParam4 = 0.000216248;
  ME0ModNeuBkgParam5 = -4.70012e-07;
  ME0ModNeuBkgParam6 = 4.21832e-10;
}

ME0PreRecoGaussianModel::~ME0PreRecoGaussianModel()
{
  if (flat1_)
    delete flat1_;
  if (flat2_)
    delete flat2_;
  if (gauss_)
    delete gauss_;
  if (poisson_)
    delete poisson_;
}

void ME0PreRecoGaussianModel::setRandomEngine(CLHEP::HepRandomEngine& eng)
{
  flat1_ = new CLHEP::RandFlat(eng);
  flat2_ = new CLHEP::RandFlat(eng);
  gauss_ = new CLHEP::RandGaussQ(eng);
  poisson_ = new CLHEP::RandFlat(eng);
}

void ME0PreRecoGaussianModel::simulateSignal(const ME0EtaPartition* roll, const edm::PSimHitContainer& simHits)
{
for (const auto & hit: simHits)
{
  if (std::abs(hit.particleType()) != 13 && digitizeOnlyMuons_) continue;
  // GEM efficiency
  if (flat1_->fire(1) > averageEfficiency_) continue;

  auto entry = hit.entryPoint();
  float x=gauss_->fire(entry.x(),sigma_u);
  float y=gauss_->fire(entry.y(),sigma_v);
  float ex=sigma_u;
  float ey=sigma_v;
  float corr=0.;
  float tof=gauss_->fire(hit.timeOfFlight(),sigma_t);
  int pdgid = hit.particleType();
  // please keep hit time always 0 for this model
  ME0DigiPreReco digi(x,y,ex,ey,corr,tof,pdgid);
  digi_.insert(digi);
}
}

void ME0PreRecoGaussianModel::simulateNoise(const ME0EtaPartition* roll)
{
  const double cspeed = 299792458;
  double trArea(0.0);
  const ME0DetId me0Id(roll->id());

  if (me0Id.region() == 0)
  {
    throw cms::Exception("Geometry")
        << "GEMSynchronizer::simulateNoise() - this GEM id is from barrel, which cannot happen.";
  }
  const TrapezoidalStripTopology* top_(dynamic_cast<const TrapezoidalStripTopology*> (&(roll->topology())));

  // base_bottom, base_top, height, strips, pads (all half length)
  auto& parameters(roll->specs()->parameters());
  float semiBottomEdge(parameters[0]);
  float semiTopEdge(parameters[1]);
  float semiHeight(parameters[2]);
  float myTanPhi = (semiTopEdge - semiBottomEdge) / (semiHeight * 2);
  double rollRadius = top_->radius();
  const int nBxing(maxBunch_ - minBunch_ + 1);
  trArea = 2 * semiHeight * (semiTopEdge + semiBottomEdge);

  if (simulateIntrinsicNoise_)
  {
  }

  //simulate bkg contribution
  if (!doBkgNoise_)
    return;

  double aveNeutrRateBotRoll = 0.;
  double averageNoiseElectronRatePerRoll = 0.;
  int pdgid = 0;

  double radius_bottom = rollRadius - semiHeight / 2.;
  aveNeutrRateBotRoll = ME0ModNeuBkgParam0
                      + ME0ModNeuBkgParam1 * radius_bottom
                      + ME0ModNeuBkgParam2 * radius_bottom * radius_bottom
                      + ME0ModNeuBkgParam3 * radius_bottom * radius_bottom * radius_bottom
                      + ME0ModNeuBkgParam4 * radius_bottom * radius_bottom * radius_bottom * radius_bottom
                      + ME0ModNeuBkgParam5 * radius_bottom * radius_bottom * radius_bottom * radius_bottom * radius_bottom
                      + ME0ModNeuBkgParam6 * radius_bottom * radius_bottom * radius_bottom * radius_bottom * radius_bottom * radius_bottom;

  double averageNoiseRateBottom = aveNeutrRateBotRoll + averageNoiseElectronRatePerRoll;
  const double averageNoiseBottom(averageNoiseRateBottom * nBxing * bxwidth_ * trArea * 1.0e-9);
  int n_hits(poisson_->fire(averageNoiseBottom));

  //find random x and coordinates
  for (int i = 0; i < n_hits; ++i)
  {
    float myRand = flat2_->fire(0., 1.);
    float yy_rand = 2 * semiHeight * (myRand - 0.5);

    //calculate xx_rand at a given yy_rand
    double xMax = semiTopEdge - (semiHeight - yy_rand) * myTanPhi;
    float myRandX = flat1_->fire(0., 1.);
    float xx_rand = 2 * xMax * (myRandX - 0.5);

    float ex = sigma_u;
    float ey = sigma_v;
    float corr = 0.;

    GlobalPoint pointDigiHit = roll->toGlobal(LocalPoint(xx_rand, yy_rand));

    //calc tof to the random estimated point
    double stripRadius = sqrt(pointDigiHit.x() * pointDigiHit.x() + pointDigiHit.y() * pointDigiHit.y()
        + pointDigiHit.z() * pointDigiHit.z());
    double timeCalibrationOffset_ = (stripRadius * 1e+9) / (cspeed * 1e+2); //[ns]
    float tof = gauss_->fire(timeCalibrationOffset_, sigma_t);

    //distribute bkg between neutrons and gammas
    float myrand = flat1_->fire(0., 1.);
    if (myrand <= 0.1)
      pdgid = 2112; // neutrons
    else
      pdgid = 22;

    ME0DigiPreReco digi(xx_rand, yy_rand, ex, ey, corr, tof, pdgid);
    digi_.insert(digi);
  }
}

