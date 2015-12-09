#include "SimMuon/GEMDigitizer/interface/ME0PreRecoGaussianModel.h"
#include "Geometry/GEMGeometry/interface/ME0EtaPartitionSpecs.h"
#include "Geometry/CommonTopologies/interface/TrapezoidalStripTopology.h"
#include "Geometry/GEMGeometry/interface/ME0Geometry.h"

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
}

void ME0PreRecoGaussianModel::simulateSignal(const ME0EtaPartition* roll, const edm::PSimHitContainer& simHits, CLHEP::HepRandomEngine* engine)
{
for (const auto & hit: simHits)
{
  // Digitize only Muons?
  if (std::abs(hit.particleType()) != 13 && digitizeOnlyMuons_) continue;
  // is GEM efficient?
  if (CLHEP::RandFlat::shoot(engine) > averageEfficiency_) continue;
  // create digi
  auto entry = hit.entryPoint();
  double x=0.0, y=0.0;
  if(gaussianSmearing_) { // Gaussian Smearing
    x=CLHEP::RandGaussQ::shoot(engine, entry.x(), sigma_u);
    y=CLHEP::RandGaussQ::shoot(engine, entry.y(), sigma_v);
  }
  else { // Uniform Smearing ... use the sigmas as boundaries
    x=entry.x()+(CLHEP::RandFlat::shoot(engine)-0.5)*sigma_u;
    y=entry.y()+(CLHEP::RandFlat::shoot(engine)-0.5)*sigma_v;
  }
  double ex=sigma_u;
  double ey=sigma_v;
  double corr=0.;
  double tof=CLHEP::RandGaussQ::shoot(engine, hit.timeOfFlight(), sigma_t);
  int pdgid = hit.particleType();
  ME0DigiPreReco digi(x,y,ex,ey,corr,tof,pdgid);
  digi_.insert(digi);
}
}

void ME0PreRecoGaussianModel::simulateNoise(const ME0EtaPartition* roll, CLHEP::HepRandomEngine* engine)
{
  double trArea(0.0);
  const ME0DetId me0Id(roll->id());

  // Extract detailed information from the Strip Topology:
  // base_bottom, base_top, height, strips, pads
  // note that (0,0) is in the middle of the roll ==> all param are at all half length
  if (me0Id.region() == 0) { throw cms::Exception("Geometry") << "Asking TrapezoidalStripTopology from a ME0 will fail"; } // not sure we really need this
  const TrapezoidalStripTopology* top_(dynamic_cast<const TrapezoidalStripTopology*>(&(roll->topology())));

  auto& parameters(roll->specs()->parameters());
  double bottomLength(parameters[0]); bottomLength = 2*bottomLength;
  double topLength(parameters[1]);    topLength    = 2*topLength;
  double height(parameters[2]);       height       = 2*height;
  double myTanPhi    = (topLength - bottomLength) / (height * 2);
  double rollRadius  = top_->radius();
  trArea = height * (topLength + bottomLength) / 2.0;

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

      double myRandY = CLHEP::RandFlat::shoot(engine);
      double yy_rand = height * (myRandY - 0.5); // random Y coord in Local Coords
      double yy_glob = rollRadius + yy_rand;     // random Y coord in Global Coords

      // Extract / Calculate the Average Electron Rate
      // for the given global Y coord from Parametrization
      double averageElectronRatePerRoll = 0.0;
      double yy_helper = 1.0;
      for(int j=0; j<7; ++j) { averageElectronRatePerRoll += eleBkg[j]*yy_helper; yy_helper *= yy_glob; }

      // Rate [Hz/cm^2] * 25*10^-9 [s] * Area [cm] = # hits in this roll
      const double averageElecRate(averageElectronRatePerRoll * (bxwidth*1.0e-9) * trArea);
      int n_elechits(CLHEP::RandPoissonQ::shoot(engine, averageElecRate));

      // max length in x for given y coordinate (cfr trapezoidal eta partition)
      double xMax = topLength/2.0 - (height/2.0 - yy_rand) * myTanPhi;

      // loop over amount of electron hits in this roll
      for (int i = 0; i < n_elechits; ++i) {
        //calculate xx_rand at a given yy_rand
        double myRandX = CLHEP::RandFlat::shoot(engine);
        double xx_rand = 2 * xMax * (myRandX - 0.5);
        double ex = sigma_u;
        double ey = sigma_v;
        double corr = 0.;
        // extract random time in this BX
        double myrandT = CLHEP::RandFlat::shoot(engine);
	double minBXtime = (bx-0.5)*bxwidth;      // double maxBXtime = (bx+0.5)*bxwidth;
        double time = myrandT*bxwidth+minBXtime;
        double myrandP = CLHEP::RandFlat::shoot(engine);
        int pdgid = 0;
        if (myrandP <= 0.5) pdgid = -11; // electron                                   
        else                pdgid = 11;  // positron                                   
        ME0DigiPreReco digi(xx_rand, yy_rand, ex, ey, corr, time, pdgid);
        digi_.insert(digi);
      }
    }

    // 2b) neutral (n+g) background                                                    
    // ----------------------------                                                    
    if (simulateNeutralBkg_) {

      double myRandY = CLHEP::RandFlat::shoot(engine);
      double yy_rand = height * (myRandY - 0.5); // random Y coord in Local Coords
      double yy_glob = rollRadius + yy_rand;    // random Y coord in Global Coords
      // Extract / Calculate the Average Electron Rate                            
      // for the given global Y coord from Parametrization                        
      double averageNeutralRatePerRoll = 0.0;
      double yy_helper = 1.0;
      for(int j=0; j<7; ++j) { averageNeutralRatePerRoll += neuBkg[j]*yy_helper; yy_helper *= yy_glob; }
      // Rate [Hz/cm^2] * 25*10^-9 [s] * Area [cm] = # hits in this roll          
      const double averageNeutrRate(averageNeutralRatePerRoll * (bxwidth*1.0e-9) * trArea);
      int n_hits(CLHEP::RandPoissonQ::shoot(engine, averageNeutrRate));

      // max length in x for given y coordinate (cfr trapezoidal eta partition)   
      double xMax = topLength/2.0 - (height/2.0 - yy_rand) * myTanPhi;

      // loop over amount of neutral hits in this roll                            
      for (int i = 0; i < n_hits; ++i) {
        //calculate xx_rand at a given yy_rand                                    
        double myRandX = CLHEP::RandFlat::shoot(engine);
        double xx_rand = 2 * xMax * (myRandX - 0.5);
        double ex = sigma_u;
        double ey = sigma_v;
        double corr = 0.;
        // extract random time in this BX                                         
        double myrandT = CLHEP::RandFlat::shoot(engine);
        double minBXtime = (bx-0.5)*bxwidth;
        double time = myrandT*bxwidth+minBXtime;
        int pdgid = 0;
        double myrandP = CLHEP::RandFlat::shoot(engine);
        if (myrandP <= 0.08) pdgid = 2112; // neutrons: GEM sensitivity for neutrons: 0.08%
        else                 pdgid = 22;   // photons:  GEM sensitivity for photons:  1.04% ==> neutron fraction = (0.08 / 1.04) = 0.077 = 0.08
        ME0DigiPreReco digi(xx_rand, yy_rand, ex, ey, corr, time, pdgid);
        digi_.insert(digi);
      }
    }

  } // end loop over bx                                  
}

