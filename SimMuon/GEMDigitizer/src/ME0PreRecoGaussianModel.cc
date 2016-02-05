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

  // Divide the detector area in different strips
  // take smearing in y-coord as height for each strip
  double heightIt = sigma_v;
  int heightbins  = height/heightIt; // round down

  edm::LogVerbatim("ME0PreRecoGaussianModelNoise") << "[ME0PreRecoDigi :: sNoise]["<<roll->id().rawId()<<"] :: roll with id = "<<roll->id();
  edm::LogVerbatim("ME0PreRecoGaussianModelNoise") << "[ME0PreRecoDigi :: sNoise]["<<roll->id().rawId()<<"] :: extracting parameters from the TrapezoidalStripTopology";
  edm::LogVerbatim("ME0PreRecoGaussianModelNoise") << "[ME0PreRecoDigi :: sNoise]["<<roll->id().rawId()<<"] :: bottom = "<<bottomLength<<" [cm] top  = "<<topLength<<" [cm] height = "<<height<<" [cm]"
						   <<" area  = "<<trArea<<" [cm^2] Rmid = "<<rollRadius<<" [cm] => Rmin = "<<rollRadius-height*1.0/2.0<<" [cm] Rmax = "<<rollRadius+height*1.0/2.0<<" [cm]";
  edm::LogVerbatim("ME0PreRecoGaussianModelNoise") << "[ME0PreRecoDigi :: sNoise]["<<roll->id().rawId()<<"] :: heightbins = "<<heightbins;


  for(int hx=0; hx<heightbins; ++hx) {
    double bottomIt = bottomLength +  hx  *2*tan(10./180*3.14)*heightIt;
    double topIt    = bottomLength + (hx+1)*2*tan(10./180*3.14)*heightIt; 
    if(hx==heightbins-1) {
      topIt = topLength; // last bin ... make strip a bit larger to cover entire roll
      heightIt = height-hx*heightIt;
    }
    double areaIt   = heightIt*(bottomIt+topIt)*1.0/2;

    edm::LogVerbatim("ME0PreRecoGaussianModelNoise") << "[ME0PreRecoDigi :: sNoise]["<<roll->id().rawId()<<"] :: height = "<<std::setw(12)<<heightIt<<" [cm] bottom = "<<std::setw(12)<<bottomIt<<" [cm]"
						     << " top = "<<std::setw(12)<<topIt<<" [cm] area = "<<std::setw(12)<<areaIt<<" [cm^2] || sin(10) = "<<sin(10./180*3.14);

    double myRandY = flat1_->fire(0., 1.);
    double y0_rand = (hx+myRandY)*heightIt;  // Y coord, measured from the bottom of the roll
    double yy_rand = (y0_rand-height*1.0/2); // Y coord, measured from the middle of the roll, which is the Y coord in Local Coords
    double yy_glob = rollRadius + yy_rand;   // R coord in Global Coords
    // max length in x for given y coordinate (cfr trapezoidal eta partition)
    double xMax = topLength/2.0 - (height/2.0 - yy_rand) * myTanPhi;

    // simulate intrinsic noise and background hits in all BX that are being read out
    // for(int bx=minBunch_; bx<maxBunch_+1; ++bx) {

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
      // Extract / Calculate the Average Electron Rate 
      // for the given global Y coord from Parametrization
      double averageElectronRatePerRoll = 0.0;
      double yy_helper = 1.0;
      for(int j=0; j<7; ++j) { averageElectronRatePerRoll += eleBkg[j]*yy_helper; yy_helper *= yy_glob; }
      
      // Rate [Hz/cm^2] * Nbx * 25*10^-9 [s] * Area [cm] = # hits in this roll in this bx
      const double averageElecRate(averageElectronRatePerRoll * (maxBunch_-minBunch_+1)*(bxwidth*1.0e-9) * areaIt); 
      
      edm::LogVerbatim("ME0PreRecoGaussianModelNoise") << "[ME0PreRecoDigi :: elebkg]["<<roll->id().rawId()<<"]" /* "] :: BX = "<<std::showpos<<bx*/
						       << " evaluation of Background Hit Rate at this coord :: "<<std::setw(12)<<averageElectronRatePerRoll<<" [Hz/cm^2]"
						       << " x 9 x 25*10^-9 [s] x Area (of strip = "<<std::setw(12)<<areaIt<<" [cm^2]) ==> "<<std::setw(12)<<averageElecRate<<" [hits]"; 
      
      // int n_elechits(poisson_->fire(averageElecRate));
      // to be fixed ... averageElecRate should be normalized ...
      // what if averageElecRate > 1?
      // what if max averageElecRate < 1 
      // ...
      bool ele_eff = (flat1_->fire(0., 1.)<averageElecRate)?1:0;      
      
      edm::LogVerbatim("ME0PreRecoGaussianModelNoise") << "[ME0PreRecoDigi :: elebkg]["<<roll->id().rawId()<<"] :: myRandY = "<<std::setw(12)<<myRandY<<" => local y = "<<std::setw(12)<<yy_rand<<" [cm]"
						       <<" => global y (global R) = "<<std::setw(12)<<yy_glob<<" [cm] || Probability = "<<std::setw(12)<<averageElecRate
						       <<" => efficient? "<<ele_eff<<std::endl;
      
      // loop over amount of electron hits in this strip (pseudo-roll) 
      // for (int i = 0; i < n_elechits; ++i) {
      if(ele_eff) {
	//calculate xx_rand at a given yy_rand
	double myRandX = flat1_->fire(0., 1.);
	double xx_rand = 2 * xMax * (myRandX - 0.5);
	double ex = sigma_u;
	double ey = sigma_v;
	double corr = 0.;
	// extract random BX
	double myrandBX = flat1_->fire(0., 1.);
	int bx = int((maxBunch_-minBunch_+1)*myrandBX)+minBunch_;
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
	edm::LogVerbatim("ME0PreRecoGaussianModelNoise") << "[ME0PreRecoDigi :: elebkg]["<<roll->id().rawId()<<"] =====> electron hit in "<<roll->id()<<" pdgid = "<<pdgid<<" bx = "<<bx
							 <<" ==> digitized"
							 <<" at loc x = "<<xx_rand<<" loc y = "<<yy_rand<<" time = "<<time<<" [ns]"; 
      }
    } // end if electron bkg
		      
    // 2b) neutral (n+g) background
    // ----------------------------
    if (simulateNeutralBkg_) {
      // Extract / Calculate the Average Electron Rate 
      // for the given global Y coord from Parametrization
      double averageNeutralRatePerRoll = 0.0;
      double yy_helper = 1.0;
      for(int j=0; j<7; ++j) { averageNeutralRatePerRoll += neuBkg[j]*yy_helper; yy_helper *= yy_glob; }
      
      // Rate [Hz/cm^2] * Nbx * 25*10^-9 [s] * Area [cm] = # hits in this roll
      const double averageNeutrRate(averageNeutralRatePerRoll * (maxBunch_-minBunch_+1)*(bxwidth*1.0e-9) * areaIt);

      edm::LogVerbatim("ME0PreRecoGaussianModelNoise") << "[ME0PreRecoDigi :: neubkg]["<<roll->id().rawId()<<"]" /* "] :: BX = "<<std::showpos<<bx*/
						       << " evaluation of Background Hit Rate at this coord :: "<<std::setw(12)<<averageNeutralRatePerRoll<<" [Hz/cm^2]"
						       << " x 9 x 25*10^-9 [s] x Area (of strip = "<<std::setw(12)<<areaIt<<" [cm^2]) ==> "<<std::setw(12)<<averageNeutrRate<<" [hits]"; 

      // int n_hits(poisson_->fire(averageNeutrRate));
      bool neu_eff = (flat1_->fire(0., 1.)<averageNeutrRate)?1:0;

      edm::LogVerbatim("ME0PreRecoGaussianModelNoise") << "[ME0PreRecoDigi :: neubkg]["<<roll->id().rawId()<<"] :: myRandY = "<<std::setw(12)<<myRandY<<" => local y = "<<std::setw(12)<<yy_rand<<" [cm]"
						       <<" => global y (global R) = "<<std::setw(12)<<yy_glob<<" [cm] || Probability = "<<std::setw(12)<<averageNeutrRate
      <<" => efficient? "<</*n_hits*/neu_eff<<std::endl;
      
      // loop over amount of neutral hits in this roll
      // for (int i = 0; i < n_hits; ++i) {
      if(neu_eff) {
	//calculate xx_rand at a given yy_rand
	double myRandX = flat1_->fire(0., 1.);
	double xx_rand = 2 * xMax * (myRandX - 0.5);
	double ex = sigma_u;
	double ey = sigma_v;
	double corr = 0.;
	// extract random BX
        double myrandBX= flat1_->fire(0., 1.);
	int bx = int((maxBunch_-minBunch_+1)*myrandBX)+minBunch_;
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
	edm::LogVerbatim("ME0PreRecoGaussianModelNoise") << "[ME0PreRecoDigi :: neubkg]["<<roll->id().rawId()<<"] ======> neutral hit in "<<roll->id()<<" pdgid = "<<pdgid<<" bx = "<<bx
							 <<" ==> digitized"
							 <<" at loc x = "<<xx_rand<<" loc y = "<<yy_rand<<" time = "<<time<<" [ns]"; 
      }
      
    } // end if neutral bkg
    
  // } // end loop over bx
  } // end loop over strips (= pseudo rolls)
}

