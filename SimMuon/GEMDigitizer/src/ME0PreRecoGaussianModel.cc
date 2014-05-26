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
  ME0DigiPreRecoModel(config), 
  sigma_t(config.getParameter<double>("timeResolution")),
  sigma_u(config.getParameter<double>("phiResolution")),
  sigma_v(config.getParameter<double>("etaResolution")),
  corr(config.getParameter<bool>("useCorrelation")),
  etaproj(config.getParameter<bool>("useEtaProjectiveGEO")),
  digitizeOnlyMuons_(config.getParameter<bool> ("digitizeOnlyMuons")),
  averageEfficiency_(config.getParameter<double> ("averageEfficiency")),
  doBkgNoise_(config.getParameter<bool> ("doBkgNoise")),
  simulateIntrinsicNoise_(config.getParameter<bool> ("simulateIntrinsicNoise"))

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
  if ( gauss_)
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
    int pdgid = hit.particleType();
     // please keep hit time always 0 for this model
    ME0DigiPreReco digi(x,y,ex,ey,corr,tof,pdgid);
    digi_.insert(digi);
  }
}


void 
ME0PreRecoGaussianModel::simulateNoise(const ME0EtaPartition* roll)
{
  double pi_ = 4*atan(1.);
  const double cspeed = 299792458; 
  double trArea(0.0);
  double trStripArea(0.0);
  const ME0DetId me0Id(roll->id());

  if (me0Id.region() == 0)
  {
    throw cms::Exception("Geometry")
        << "GEMSynchronizer::simulateNoise() - this GEM id is from barrel, which cannot happen.";
  }

  const TrapezoidalStripTopology* top_(dynamic_cast<const TrapezoidalStripTopology*> (&(roll->topology())));
  const int nstrips(roll->nstrips());
  const float striplength(top_->stripLength());
  double rollRadius = top_->radius();
  double xmax_half = rollRadius * tan(pi_/18.); // tan(10 o)
  const int nBxing(maxBunch_ - minBunch_ + 1);
  trArea = 2*xmax_half*striplength;

  std::cout << "-------------------" << std::endl;
  std::cout << "me0Id = " << me0Id << std::endl;

  //simulate intrinsic noise - switched Off - there are NO strips
  // fire anywhere in ME0 chamber
  if(simulateIntrinsicNoise_)
  {
    double aveIntrinsicNoisPerStrip = averageNoiseRate_ * nBxing * bxwidth_ * trStripArea * 1.0e-9;
    for(int j = 0; j < nstrips; ++j)
    {
      const int n_intrHits = poisson_->fire(aveIntrinsicNoisPerStrip);
      GlobalPoint pointDigiHit = roll->toGlobal(roll->centreOfStrip(j));
      float x=gauss_->fire(pointDigiHit.x(),sigma_u);
      float y=gauss_->fire(pointDigiHit.y(),sigma_v); 
      float ex=sigma_u;
      float ey=sigma_v;
      float corr=0.;

      double stripRadius = sqrt(pointDigiHit.x()*pointDigiHit.x() + pointDigiHit.y()*pointDigiHit.y() +pointDigiHit.z()*pointDigiHit.z());
      double timeCalibrationOffset_ = (stripRadius *1e+9)/(cspeed*1e+2); //[ns]

      for (int k = 0; k < n_intrHits; k++ )
      {
        float tof=gauss_->fire(timeCalibrationOffset_,sigma_t);
        float pdgid = 22;
        ME0DigiPreReco digi(x,y,ex,ey,corr,tof,pdgid);
        digi_.insert(digi);
      }
    }
  }//end simulate intrinsic noise

//simulate bkg contribution
  if (!doBkgNoise_)
  return;

//calculate noise from model
  double averageNeutralNoiseRatePerRoll = 0.;
  double averageNoiseElectronRatePerRoll = 0.;

//find random yy in the ME0 partition
  float halfstripLenndht = striplength/2.;
  float yy = flat2_->fire((rollRadius - halfstripLenndht), (rollRadius + halfstripLenndht));
  std::cout << "yy = " << yy << std::endl;
 
//calculate neutral bkg at yy from pol6
  averageNeutralNoiseRatePerRoll = ME0ModNeuBkgParam0
                                 + ME0ModNeuBkgParam1*yy
                                 + ME0ModNeuBkgParam2*yy*yy
                                 + ME0ModNeuBkgParam3*yy*yy*yy
                                 + ME0ModNeuBkgParam4*yy*yy*yy*yy
                                 + ME0ModNeuBkgParam5*yy*yy*yy*yy*yy
                                 + ME0ModNeuBkgParam6*yy*yy*yy*yy*yy*yy;

  double averageNoiseRatePerRoll = averageNeutralNoiseRatePerRoll + averageNoiseElectronRatePerRoll;

  std::cout << "averageNoiseRatePerRoll = " << averageNoiseRatePerRoll <<std::endl;

  const double averageNoise(averageNoiseRatePerRoll * nBxing * bxwidth_ * trArea * 1.0e-9);
  const int n_hits(poisson_->fire(averageNoise));

  std::cout << "averageNoise = " << averageNoise << std::endl;
  std::cout << "n_hits = " << n_hits << std::endl;
  float pdgid = 0;

  for (int i = 0; i < n_hits; ++i)
  {
//find random xx = yy*tg(phi_random)
//find random phi using random sin phi distribution for phi in[80, 100]
    float ktemp = flat1_->fire(0., 1.);
    float phiTemp = acos(sin(pi_/18.) - 2*ktemp*sin(pi_/18.)); // sin(10 o)
    float phiRand = phiTemp - (4*pi_/9.); // substract 80 o in order to get phiRand in [0, 20] - suppose all the rolls are 20 o

    std::cout << "phiRand = " << (phiRand * 360)/(2*pi_) << std::endl;
    float xx = yy*tan(phiRand);

    float zz = 527 + (me0Id.layer())*25./6.;
    std::cout << "layer = " << me0Id.layer() << std::endl;
    std::cout << "zz = " << zz << std::endl;
//    float zz = 539.5; // temporary fixed value, just to run the code

    GlobalPoint pointDigiHit = roll->toGlobal(LocalPoint(xx,yy,zz));

    float ex=sigma_u;
    float ey=sigma_v;
    float corr=0.;

    double stripRadius = sqrt(pointDigiHit.x()*pointDigiHit.x() + pointDigiHit.y()*pointDigiHit.y() +pointDigiHit.z()*pointDigiHit.z());
    double timeCalibrationOffset_ = (stripRadius *1e+9)/(cspeed*1e+2); //[ns]
    float tof=gauss_->fire(timeCalibrationOffset_,sigma_t);
    float myrand = flat1_->fire(0., 1.);
    if (myrand <= 0.1)
      pdgid = 2112; // neutrons
    else
      pdgid = 22;

std::cout << "bkg pdgId = " << pdgid << std::endl;

    ME0DigiPreReco digi(xx,yy,ex,ey,corr,tof,pdgid);
    digi_.insert(digi);
    std::cout << "DigiPreReco inserted" << std::endl;

  }
}

