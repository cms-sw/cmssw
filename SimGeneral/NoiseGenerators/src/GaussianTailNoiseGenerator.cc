//#include "CommonDet/DetUtilities/interface/GaussianTailNoiseGenerator.h"
#include "SimGeneral/NoiseGenerators/interface/GaussianTailNoiseGenerator.h"
#include "CLHEP/Random/RandPoisson.h"
#include "CLHEP/Random/RandFlat.h"
#include <gsl/gsl_sf_erf.h>
#include <gsl/gsl_sf_result.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_rng.h>

extern "C"   float freq_(const float& x);   
extern "C"   float gausin_(const float& x);

void GaussianTailNoiseGenerator::generate(int NumberOfchannels, 
					  float threshold, 
					  float noiseRMS, 
					  std::map<int,float, std::less<int> >& theMap )
{

  // Compute number of channels with noise above threshold

  // Gaussian tail probability
  gsl_sf_result result;
  int status = gsl_sf_erf_Q_e(threshold, &result);
  //MP 
  //  if (status != 0) throw DetLogicError("GaussianTailNoiseGenerator::could not compute gaussian tail probability for the threshold chosen");
  if (status != 0) std::cerr<<"GaussianTailNoiseGenerator::could not compute gaussian tail probability for the threshold chosen"<<std::endl;
  float probabilityLeft = result.val;
  
  float meanNumberOfNoisyChannels = probabilityLeft * NumberOfchannels;
  int numberOfNoisyChannels = RandPoisson::shoot(meanNumberOfNoisyChannels);

  // draw noise at random according to Gaussian tail

  // initialise default gsl uniform generator engine
  static gsl_rng * mt19937 = gsl_rng_alloc (gsl_rng_mt19937);

  float lowLimit = threshold * noiseRMS;
  for (int i = 0; i < numberOfNoisyChannels; i++) {

    // Find a random channel number    
    int theChannelNumber = (int) RandFlat::shootInt(NumberOfchannels);
    
    // Find random noise value
    float noise = gsl_ran_gaussian_tail(mt19937, lowLimit, noiseRMS);
						          
    // Fill in map
    theMap[theChannelNumber] = noise;
    
  }
}
