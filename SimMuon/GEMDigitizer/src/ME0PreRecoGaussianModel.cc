#include "SimMuon/GEMDigitizer/interface/ME0PreRecoGaussianModel.h"
#include "Geometry/GEMGeometry/interface/ME0EtaPartitionSpecs.h"
#include "Geometry/CommonTopologies/interface/TrapezoidalStripTopology.h"
#include "Geometry/GEMGeometry/interface/ME0Geometry.h"
#include "DataFormats/GEMDigi/interface/ME0DigiPreReco.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CLHEP/Random/RandomEngine.h"
#include "CLHEP/Random/RandFlat.h"
#include "CLHEP/Random/RandGaussQ.h"

#include <cmath>
#include <utility>
#include <map>


ME0PreRecoGaussianModel::ME0PreRecoGaussianModel(const edm::ParameterSet& config) : 
  ME0DigiPreRecoModel(config), 
  sigma_t(config.getParameter<double>("timeResolution")),
  sigma_u(config.getParameter<double>("phiResolution")),
  sigma_v(config.getParameter<double>("etaResolution")),
  corr(config.getParameter<bool>("useCorrelation")),
  etaproj(config.getParameter<bool>("useEtaProjectiveGEO")),
  digitizeOnlyMuons_(config.getParameter<bool> ("digitizeOnlyMuons")),
  averageEfficiency_(config.getParameter<double> ("averageEfficiency")),
  doBkgNoise_(config.getParameter<bool> ("doBkgNoise")),
  simulateIntrinsicNoise_
{
}

ME0PreRecoGaussianModel::~ME0PreRecoGaussianModel()
{
  if (flat1_)
    delete flat1_;
  if ( gauss_)
    delete gauss_;
}

void ME0PreRecoGaussianModel::setRandomEngine(CLHEP::HepRandomEngine& eng)
{
  flat1_ = new CLHEP::RandFlat(eng);
  gauss_ = new CLHEP::RandGaussQ(eng);
}

void 
ME0PreRecoGaussianModel::simulateSignal(const ME0EtaPartition* roll,
				const edm::PSimHitContainer& simHits)
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
     // please keep hit time always 0 for this model
    ME0DigiPreReco digi(x,y,ex,ey,corr,tof);
    digi_.insert(digi);
  }
}


void 
ME0PreRecoGaussianModel::simulateNoise(const GEMEtaPartition* roll)
{
  if (!doBkgNoise_)
  return;

  //calculate noise from model
  double averageNeutralNoiseRatePerRoll = 0.;
  double averageNoiseElectronRatePerRoll = 0.;
  double averageNoiseRatePerRoll = averageNeutralNoiseRatePerRoll + averageNoiseElectronRatePerRoll;

  //simulate intrinsic noise
  // fire anywhere in ME0 chamber
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
    std::pair<int, int> digi(centralStrip, time_hit);
    strips_.insert(digi);

  }
}

