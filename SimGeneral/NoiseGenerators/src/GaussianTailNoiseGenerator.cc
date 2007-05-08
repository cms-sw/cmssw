#include "SimGeneral/NoiseGenerators/interface/GaussianTailNoiseGenerator.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/RandomNumberGenerator.h"
#include "CLHEP/Random/RandPoisson.h"
#include "CLHEP/Random/RandFlat.h"
#include "FWCore/Utilities/interface/Exception.h"
#include <gsl/gsl_sf_erf.h>
#include <gsl/gsl_sf_result.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_rng.h>

extern "C"   float freq_(const float& x);   
extern "C"   float gausin_(const float& x);

GaussianTailNoiseGenerator::GaussianTailNoiseGenerator(CLHEP::HepRandomEngine& eng,int NumbChannels,float thr ):
  poissonDistribution_(0),flatDistribution_(0),engine(eng),numberOfChannels(NumbChannels)
{
  // Gaussian tail probability
  gsl_sf_result result;
  int status = gsl_sf_erf_Q_e(thr, &result);
  if (status != 0) throw cms::Exception("")
    <<"GaussianTailNoiseGenerator::could not compute gaussian\n" 
		     "tail probability for the threshold chosen";
  
  float probabilityLeft = result.val;
  
  float meanNumberOfNoisyChannels = probabilityLeft * numberOfChannels;
  
  poissonDistribution_ = new CLHEP::RandPoisson(engine, meanNumberOfNoisyChannels);

  flatDistribution_ = new CLHEP::RandFlat(engine, numberOfChannels); 
 
}

GaussianTailNoiseGenerator::~GaussianTailNoiseGenerator()
{
  delete poissonDistribution_;
  delete flatDistribution_;
}

void GaussianTailNoiseGenerator::generate(float threshold, 
					  float noiseRMS, 
					  std::map<int,float, std::less<int> >& theMap )
{

  int numberOfNoisyChannels = poissonDistribution_->fire();
  
  // initialise default gsl uniform generator engine
  static gsl_rng * mt19937 = gsl_rng_alloc (gsl_rng_mt19937);
  
  float lowLimit = threshold * noiseRMS;
  for (int i = 0; i < numberOfNoisyChannels; i++) {
    
    // Find a random channel number    
    int theChannelNumber = (int)flatDistribution_->fire();
    
    // Find random noise value
    float noise = gsl_ran_gaussian_tail(mt19937, lowLimit, noiseRMS);
    
    // Fill in map
    theMap[theChannelNumber] = noise;
    
  }
}
