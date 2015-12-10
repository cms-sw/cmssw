#include "SimMuon/GEMDigitizer/interface/ME0PreRecoGaussianModel.h"
#include "Geometry/GEMGeometry/interface/ME0EtaPartitionSpecs.h"
#include "Geometry/CommonTopologies/interface/TrapezoidalStripTopology.h"
#include "Geometry/GEMGeometry/interface/ME0Geometry.h"
#include "DataFormats/GEMDigi/interface/ME0DigiPreReco.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "CLHEP/Random/RandomEngine.h"
#include "CLHEP/Random/RandFlat.h"
#include "CLHEP/Random/RandGaussQ.h"
#include "CLHEP/Random/RandPoissonQ.h"
#include <cmath>
#include <utility>
#include <map>


const double cspeed = 29.9792458; // [cm/ns]
const int bxwidth   = 25;         // [ns]

ME0PreRecoGaussianModel::ME0PreRecoGaussianModel(const edm::ParameterSet& config) :
    ME0DigiPreRecoModel(config), 
    sigma_t(config.getParameter<double>("timeResolution")), 
    sigma_u(config.getParameter<double>("phiResolution")), 
    sigma_v(config.getParameter<double>("etaResolution")), 
    corr(config.getParameter<bool>("useCorrelation")), 
    etaproj(config.getParameter<bool>("useEtaProjectiveGEO")), 
    digitizeOnlyMuons_(config.getParameter<bool>("digitizeOnlyMuons")), 
    gaussianSmearing_(config.getParameter<bool>("gaussianSmearing")),
    averageEfficiency_(config.getParameter<double>("averageEfficiency")), 
    // simulateIntrinsicNoise_(config.getParameter<bool>("simulateIntrinsicNoise")),
    // averageNoiseRate_(config.getParameter<double>("averageNoiseRate")), 
    simulateElectronBkg_(config.getParameter<bool>("simulateElectronBkg")), 
    simulateNeutralBkg_(config.getParameter<bool>("simulateNeutralBkg")), 
    minBunch_(config.getParameter<int>("minBunch")), 
    maxBunch_(config.getParameter<int>("maxBunch"))
{
  // polynomial parametrisation of neutral (n+g) and electron background
  neuBkg.push_back(899644.0);     neuBkg.push_back(-30841.0);     neuBkg.push_back(441.28); 
  neuBkg.push_back(-3.3405);      neuBkg.push_back(0.0140588);    neuBkg.push_back(-3.11473e-05); neuBkg.push_back(2.83736e-08);
  eleBkg.push_back(4.68590e+05);  eleBkg.push_back(-1.63834e+04); eleBkg.push_back(2.35700e+02);
  eleBkg.push_back(-1.77706e+00); eleBkg.push_back(7.39960e-03);  eleBkg.push_back(-1.61448e-05); eleBkg.push_back(1.44368e-08);
}
ME0PreRecoGaussianModel::~ME0PreRecoGaussianModel()
{
  if (flat1_)   delete flat1_;
  if (flat2_)   delete flat2_;
  if (gauss_)   delete gauss_;
  if (poisson_) delete poisson_;
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
  for (const auto & hit : simHits)
  {
    // Digitize only Muons?
    if (std::abs(hit.particleType()) != 13 && digitizeOnlyMuons_) continue;
    // Digitize only in [minBunch,maxBunch] window
    // window is: [(2n-1)*bxw/2, (2n+1)*bxw/2], n = [minBunch, maxBunch]
    if(hit.timeOfFlight() < (2*minBunch_-1)*bxwidth*1.0/2 || hit.timeOfFlight() > (2*maxBunch_+1)*bxwidth*1.0/2) continue;
    // is GEM efficient?
    if (flat1_->fire(1) > averageEfficiency_) continue;
    // create digi
    auto entry = hit.entryPoint();
    double x=0.0, y=0.0;
    if(gaussianSmearing_) { // Gaussian Smearing
      x=gauss_->fire(entry.x(), sigma_u);
      y=gauss_->fire(entry.y(), sigma_v);
    }
    else { // Uniform Smearing ... use the sigmas as boundaries
      x=entry.x()+(flat1_->fire(0., 1.)-0.5)*sigma_u;
      y=entry.y()+(flat1_->fire(0., 1.)-0.5)*sigma_v;
    }
    double ex = sigma_u;
    double ey = sigma_v;
    double corr = 0.;
    double tof = gauss_->fire(hit.timeOfFlight(), sigma_t);
    int pdgid = hit.particleType();
    ME0DigiPreReco digi(x, y, ex, ey, corr, tof, pdgid, 1);
    digi_.insert(digi);

    edm::LogVerbatim("ME0PreRecoGaussianModel") << "[ME0PreRecoDigi :: simulateSignal] :: simhit in "<<roll->id()<<" at loc x = "<<std::setw(8)<<entry.x()<<" [cm]"
                                                << " loc y = "<<std::setw(8)<<entry.y()<<" [cm] time = "<<std::setw(8)<<hit.timeOfFlight()<<" [ns] pdgid = "<<std::showpos<<std::setw(4)<<pdgid;
    edm::LogVerbatim("ME0PreRecoGaussianModel") << "[ME0PreRecoDigi :: simulateSignal] :: digi   in "<<roll->id()<<" at loc x = "<<std::setw(8)<<x<<" [cm] loc y = "<<std::setw(8)<<y<<" [cm]"
                                                <<" time = "<<std::setw(8)<<tof<<" [ns]";
    edm::LogVerbatim("ME0PreRecoGaussianModel") << "[ME0PreRecoDigi :: simulateSignal] :: digi   in "<<roll->id()<<" with DX = "<<std::setw(8)<<(entry.x()-x)<<" [cm]"
                                                <<" DY = "<<std::setw(8)<<(entry.y()-y)<<" [cm] DT = "<<std::setw(8)<<(hit.timeOfFlight()-tof)<<" [ns]";

  }
}
void ME0PreRecoGaussianModel::simulateNoise(const ME0EtaPartition* roll)
{
  double trArea(0.0);

  // Extract detailed information from the Strip Topology:
  // base_bottom, base_top, height, strips, pads 
  // note that (0,0) is in the middle of the roll ==> all param are at all half length
  const TrapezoidalStripTopology* top_(dynamic_cast<const TrapezoidalStripTopology*>(&(roll->topology())));

  auto& parameters(roll->specs()->parameters());
  double bottomLength(parameters[0]); bottomLength = 2*bottomLength; // bottom is largest length, so furtest away from beamline
  double topLength(parameters[1]);    topLength    = 2*topLength;    // top is shortest length, so closest to beamline
  double height(parameters[2]);       height       = 2*height;
  double myTanPhi    = (topLength - bottomLength) / (height * 2);
  double rollRadius = top_->radius();
  trArea = height * (topLength + bottomLength) / 2.0;

  edm::LogVerbatim("ME0PreRecoGaussianModelNoise") << "[ME0PreRecoDigi :: simulateNoise] :: extracting parameters from the TrapezoidalStripTopology";
  edm::LogVerbatim("ME0PreRecoGaussianModelNoise") << "[ME0PreRecoDigi :: simulateNoise] :: bottom = "<<bottomLength<<" [cm] top  = "<<topLength<<" [cm] height = "<<height
						   <<" [cm] radius = "<<rollRadius<<" [cm]" ;

  // simulate intrinsic noise and background hits in all BX that are being read out
  for(int bx=minBunch_; bx<maxBunch_+1; ++bx) {

    // 1) Intrinsic Noise ... Not implemented right now
    // ------------------------------------------------
    // if (simulateIntrinsicNoise_)
    // {
    // }

    // 2) Background Noise 
    // ----------------------------
 
    // 2a) electron background
    // -----------------------
    if (simulateElectronBkg_) {

      double myRandY = flat2_->fire(0., 1.);
      double yy_rand = height * (myRandY - 0.5); // random Y coord in Local Coords
      double yy_glob = rollRadius + yy_rand;     // random Y coord in Global Coords

      // Extract / Calculate the Average Electron Rate 
      // for the given global Y coord from Parametrization
      double averageElectronRatePerRoll = 0.0;
      double yy_helper = 1.0;
      for(int j=0; j<7; ++j) { averageElectronRatePerRoll += eleBkg[j]*yy_helper; yy_helper *= yy_glob; }

      // Rate [Hz/cm^2] * 25*10^-9 [s] * Area [cm] = # hits in this roll 
      const double averageElecRate(averageElectronRatePerRoll * (bxwidth*1.0e-9) * trArea); 
      int n_elechits(poisson_->fire(averageElecRate));

      edm::LogVerbatim("ME0PreRecoGaussianModelNoise") << "[ME0PreRecoDigi :: simulateNoise :: ele bkg] :: myRandY = "<<std::setw(12)<<myRandY<<" => local y = "<<std::setw(12)<<yy_rand<<" [cm]"
                                                       <<" => global y (global R) = "<<std::setw(12)<<yy_glob<<" [cm] || Probability = "<<std::setw(12)<<averageElecRate
                                                       <<" => efficient? "<<n_elechits<<std::endl;

      // max length in x for given y coordinate (cfr trapezoidal eta partition)
      double xMax = topLength/2.0 - (height/2.0 - yy_rand) * myTanPhi;

      // loop over amount of electron hits in this roll
      for (int i = 0; i < n_elechits; ++i) {
	//calculate xx_rand at a given yy_rand
	double myRandX = flat1_->fire(0., 1.);
	double xx_rand = 2 * xMax * (myRandX - 0.5);
	double ex = sigma_u;
	double ey = sigma_v;
	double corr = 0.;
	// extract random time in this BX
	double myrandT = flat1_->fire(0., 1.);
	double minBXtime = (bx-0.5)*bxwidth;      // double maxBXtime = (bx+0.5)*bxwidth;
	double time = myrandT*bxwidth+minBXtime;
	double myrandP = flat1_->fire(0., 1.);
	int pdgid = 0;
	if (myrandP <= 0.5) pdgid = -11; // electron
	else 	            pdgid = 11;  // positron
	ME0DigiPreReco digi(xx_rand, yy_rand, ex, ey, corr, time, pdgid, 0);
	digi_.insert(digi);

	edm::LogVerbatim("ME0PreRecoGaussianModelNoise") << "[ME0PreRecoDigi :: simulateNoise :: ele bkg] :: electron hit in "<<roll->id()<<" pdgid = "<<pdgid<<" bx = "<<bx
                                                         <<" ==> digitized at loc x = "<<xx_rand<<" loc y = "<<yy_rand<<" time = "<<time<<" [ns]";

      }
    }

    // 2b) neutral (n+g) background
    // ----------------------------
    if (simulateNeutralBkg_) {

      double myRandY = flat2_->fire(0., 1.);
      double yy_rand = height * (myRandY - 0.5); // random Y coord in Local Coords
      double yy_glob = rollRadius + yy_rand;    // random Y coord in Global Coords

      // Extract / Calculate the Average Electron Rate 
      // for the given global Y coord from Parametrization
      double averageNeutralRatePerRoll = 0.0;
      double yy_helper = 1.0;
      for(int j=0; j<7; ++j) { averageNeutralRatePerRoll += neuBkg[j]*yy_helper; yy_helper *= yy_glob; }

      // Rate [Hz/cm^2] * 25*10^-9 [s] * Area [cm] = # hits in this roll
      const double averageNeutrRate(averageNeutralRatePerRoll * (bxwidth*1.0e-9) * trArea);
      int n_hits(poisson_->fire(averageNeutrRate));

      edm::LogVerbatim("ME0PreRecoGaussianModelNoise") << "[ME0PreRecoDigi :: simulateNoise :: neu bkg] :: myRandY = "<<std::setw(12)<<myRandY<<" => local y = "<<std::setw(12)<<yy_rand<<" [cm]"
                                                       <<" => global y (global R) = "<<std::setw(12)<<yy_glob<<" [cm] || Probability "<<std::setw(12)<<averageNeutrRate
                                                       <<" => efficient? "<<n_hits<<std::endl;

      // max length in x for given y coordinate (cfr trapezoidal eta partition)
      double xMax = topLength/2.0 - (height/2.0 - yy_rand) * myTanPhi;

      // loop over amount of neutral hits in this roll
      for (int i = 0; i < n_hits; ++i) {
	//calculate xx_rand at a given yy_rand
	double myRandX = flat1_->fire(0., 1.);
	double xx_rand = 2 * xMax * (myRandX - 0.5);
	double ex = sigma_u;
	double ey = sigma_v;
	double corr = 0.;
	// extract random time in this BX
        double myrandT = flat1_->fire(0., 1.);
        double minBXtime = (bx-0.5)*bxwidth;
	double time = myrandT*bxwidth+minBXtime;
	int pdgid = 0;
        double myrandP = flat1_->fire(0., 1.);
        if (myrandP <= 0.08) pdgid = 2112; // neutrons: GEM sensitivity for neutrons: 0.08%
        else                 pdgid = 22;   // photons:  GEM sensitivity for photons:  1.04% ==> neutron fraction = (0.08 / 1.04) = 0.077 = 0.08
        ME0DigiPreReco digi(xx_rand, yy_rand, ex, ey, corr, time, pdgid, 0);
        digi_.insert(digi);

	edm::LogVerbatim("ME0PreRecoGaussianModelNoise") << "[ME0PreRecoDigi :: simulateNoise :: neu bkg] :: neutral hit in "<<roll->id()<<" pdgid = "<<pdgid<<" bx = "<<bx
                                                         <<" ==> digitized at loc x = "<<xx_rand<<" loc y = "<<yy_rand<<" time = "<<time<<" [ns]";

      }
    }

  } // end loop over bx
}

