#include "SimMuon/GEMDigitizer/interface/ME0PreRecoGaussianModel.h"
#include "Geometry/GEMGeometry/interface/ME0EtaPartitionSpecs.h"
#include "Geometry/CommonTopologies/interface/TrapezoidalStripTopology.h"
#include "Geometry/GEMGeometry/interface/ME0Geometry.h"
#include "DataFormats/GEMDigi/interface/ME0DigiPreReco.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CLHEP/Random/RandomEngine.h"
#include "CLHEP/Random/RandGaussQ.h"

#include <cmath>
#include <utility>
#include <map>


ME0PreRecoGaussianModel::ME0PreRecoGaussianModel(const edm::ParameterSet& config) 
  : ME0DigiPreRecoModel(config), 
    sigma_t(config.getParameter<double>("timeResolution")),
    sigma_u(config.getParameter<double>("phiResolution")),
    sigma_v(config.getParameter<double>("etaResolution")),
    corr(config.getParameter<bool>("useCorrelation")),
    etaproj(config.getParameter<bool>("useEtaProjectiveGEO"))
{
}

ME0PreRecoGaussianModel::~ME0PreRecoGaussianModel()
{
  if ( gauss_)
    delete gauss_;
}

void ME0PreRecoGaussianModel::setRandomEngine(CLHEP::HepRandomEngine& eng)
{
  gauss_ = new CLHEP::RandGaussQ(eng);
}

void 
ME0PreRecoGaussianModel::simulateSignal(const ME0EtaPartition* roll,
				const edm::PSimHitContainer& simHits)
{

  for (const auto & hit: simHits)
  {
    if (std::abs(hit.particleType()) != 13) continue;
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



