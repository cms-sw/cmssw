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
  averageEfficiency_(config.getParameter<double>("averageEfficiency")), 
  // simulateIntrinsicNoise_(config.getParameter<bool>("simulateIntrinsicNoise")),
  // averageNoiseRate_(config.getParameter<double>("averageNoiseRate")), 
  simulateElectronBkg_(config.getParameter<bool>("simulateElectronBkg")), 
  simulateNeutralBkg_(config.getParameter<bool>("simulateNeutralBkg")), 
  minBunch_(config.getParameter<int>("minBunch")), 
  maxBunch_(config.getParameter<int>("maxBunch"))
{
  // polynomial parametrisation of neutral (n+g) and electron background
  neuBkg.push_back(5.69e+06);     neuBkg.push_back(-293334);     neuBkg.push_back(6279.6); 
  neuBkg.push_back(-71.2928);     neuBkg.push_back(0.452244);    neuBkg.push_back(-0.0015191);  neuBkg.push_back(2.1106e-06);
  eleBkg.push_back(3.77712e+06);  eleBkg.push_back(-199280);     eleBkg.push_back(4340.69);
  eleBkg.push_back(-49.922);      eleBkg.push_back(0.319699);    eleBkg.push_back(-0.00108113); eleBkg.push_back(1.50889e-06);
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
  float x=CLHEP::RandGaussQ::shoot(engine, entry.x(), sigma_u);
  float y=CLHEP::RandGaussQ::shoot(engine, entry.y(), sigma_v);
  float ex=sigma_u;
  float ey=sigma_v;
  float corr=0.;
  float tof=CLHEP::RandGaussQ::shoot(engine, hit.timeOfFlight(), sigma_t);
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
  float bottomLength(parameters[0]); bottomLength = 2*bottomLength;
  float topLength(parameters[1]);    topLength    = 2*topLength;
  float height(parameters[2]);       height       = 2*height;
  float myTanPhi    = (topLength - bottomLength) / (height * 2);
  double rollRadius = top_->radius();
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

      float myRandY = CLHEP::RandFlat::shoot(engine);
      float yy_rand = height * (myRandY - 0.5); // random Y coord in Local Coords
      double yy_glob = rollRadius + yy_rand;    // random Y coord in Global Coords

      // Extract / Calculate the Average Electron Rate
      // for the given global Y coord from Parametrization
      double averageElectronRatePerRoll = 0.0;
      for(int j=0; j<7; ++j) { averageElectronRatePerRoll += eleBkg[j]*pow(yy_glob,j); }

      // Rate [Hz/cm^2] * 25*10^-9 [s] * Area [cm] = # hits in this roll
      const double averageElecRate(averageElectronRatePerRoll * (bxwidth*1.0e-9) * trArea);
      int n_elechits(CLHEP::RandPoissonQ::shoot(engine, averageElecRate));

      // max length in x for given y coordinate (cfr trapezoidal eta partition)
      double xMax = topLength/2.0 - (height/2.0 - yy_rand) * myTanPhi;

      // loop over amount of electron hits in this roll
      for (int i = 0; i < n_elechits; ++i) {
        //calculate xx_rand at a given yy_rand
        float myRandX = CLHEP::RandFlat::shoot(engine);
        float xx_rand = 2 * xMax * (myRandX - 0.5);
        float ex = sigma_u;
        float ey = sigma_v;
        float corr = 0.;
        // extract random time in this BX
        float myrandT = CLHEP::RandFlat::shoot(engine);
	float minBXtime = (bx-0.5)*bxwidth;      // float maxBXtime = (bx+0.5)*bxwidth;
        float time = myrandT*bxwidth+minBXtime;
        float myrandP = CLHEP::RandFlat::shoot(engine);
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

      float myRandY = CLHEP::RandFlat::shoot(engine);
      float yy_rand = height * (myRandY - 0.5); // random Y coord in Local Coords
      double yy_glob = rollRadius + yy_rand;    // random Y coord in Global Coords
      // Extract / Calculate the Average Electron Rate                            
      // for the given global Y coord from Parametrization                        
      double averageNeutralRatePerRoll = 0.0;
      for(int j=0; j<7; ++j) { averageNeutralRatePerRoll += neuBkg[j]*pow(yy_glob,j); }
      // Rate [Hz/cm^2] * 25*10^-9 [s] * Area [cm] = # hits in this roll          
      const double averageNeutrRate(averageNeutralRatePerRoll * (bxwidth*1.0e-9) * trArea);
      int n_hits(CLHEP::RandPoissonQ::shoot(engine, averageNeutrRate));

      // max length in x for given y coordinate (cfr trapezoidal eta partition)   
      double xMax = topLength/2.0 - (height/2.0 - yy_rand) * myTanPhi;

      // loop over amount of neutral hits in this roll                            
      for (int i = 0; i < n_hits; ++i) {
        //calculate xx_rand at a given yy_rand                                    
        float myRandX = CLHEP::RandFlat::shoot(engine);
        float xx_rand = 2 * xMax * (myRandX - 0.5);
        float ex = sigma_u;
        float ey = sigma_v;
        float corr = 0.;
        // extract random time in this BX                                         
        float myrandT = CLHEP::RandFlat::shoot(engine);
        float minBXtime = (bx-0.5)*bxwidth;
        float time = myrandT*bxwidth+minBXtime;
        int pdgid = 0;
        float myrandP = CLHEP::RandFlat::shoot(engine);
        if (myrandP <= 0.08) pdgid = 2112; // neutrons: GEM sensitivity for neutrons: 0.08%
        else                 pdgid = 22;   // photons:  GEM sensitivity for photons:  1.04% ==> neutron fraction = (0.08 / 1.04) = 0.077 = 0.08
        ME0DigiPreReco digi(xx_rand, yy_rand, ex, ey, corr, time, pdgid);
        digi_.insert(digi);
      }
    }

  } // end loop over bx                                  
}

