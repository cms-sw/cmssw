#include "SimGeneral/NoiseGenerators/interface/GaussianTailNoiseGenerator.h"
#include "CLHEP/Random/RandPoisson.h"
#include "CLHEP/Random/RandGauss.h"
#include "CLHEP/Random/RandFlat.h"

//extern "C"   float freq_(const float& x);   
//extern "C"   float gausin_(const float& x);


GaussianTailNoiseGenerator::GaussianTailNoiseGenerator(CLHEP::HepRandomEngine& eng ) :
  gaussDistribution_(0),poissonDistribution_(0),flatDistribution_(0),rndEngine(eng),mt19937(0) {
  
  gaussDistribution_ = new CLHEP::RandGauss(rndEngine);
  poissonDistribution_ = new CLHEP::RandPoisson(rndEngine);
  flatDistribution_ = new CLHEP::RandFlat(rndEngine); 
 
}

GaussianTailNoiseGenerator::~GaussianTailNoiseGenerator() {
  delete gaussDistribution_;
  delete poissonDistribution_;
  delete flatDistribution_;
  gsl_rng_free(mt19937);
}

void GaussianTailNoiseGenerator::generate(int NumberOfchannels, 
					  float threshold, 
					  float noiseRMS, 
					  std::map<int,float, std::less<int> >& theMap ) {

   // Gaussian tail probability
  gsl_sf_result result;
  int status = gsl_sf_erf_Q_e(threshold, &result);

  if (status != 0) std::cerr<<"GaussianTailNoiseGenerator::could not compute gaussian tail probability for the threshold chosen"<<std::endl;

  float probabilityLeft = result.val;  
  float meanNumberOfNoisyChannels = probabilityLeft * NumberOfchannels;
   int numberOfNoisyChannels = poissonDistribution_->fire(meanNumberOfNoisyChannels);

  // draw noise at random according to Gaussian tail
  // initialise default gsl uniform generator engine
  if(mt19937 == 0) 
    mt19937 = gsl_rng_alloc (gsl_rng_mt19937);

  float lowLimit = threshold * noiseRMS;
  for (int i = 0; i < numberOfNoisyChannels; i++) {

    // Find a random channel number    
    int theChannelNumber = (int)flatDistribution_->fire(NumberOfchannels);

    // Find random noise value
    float noise = gsl_ran_gaussian_tail(mt19937, lowLimit, noiseRMS);
              
    // Fill in map
    theMap[theChannelNumber] = noise;
    
  }
}

void GaussianTailNoiseGenerator::generate(int NumberOfchannels, 
					  float threshold, 
					  float noiseRMS, 
					  std::vector<std::pair<int,float> > &theVector ) {

  // Compute number of channels with noise above threshold
  // Gaussian tail probability
  gsl_sf_result result;
  int status = gsl_sf_erf_Q_e(threshold, &result);
  
  if (status != 0) std::cerr<<"GaussianTailNoiseGenerator::could not compute gaussian tail probability for the threshold chosen"<<std::endl;
  
  double probabilityLeft = result.val;  
  double meanNumberOfNoisyChannels = probabilityLeft * NumberOfchannels;
  int numberOfNoisyChannels = poissonDistribution_->fire(meanNumberOfNoisyChannels);

  // initialise default gsl uniform generator engine
  if(mt19937 == 0) 
    mt19937 = gsl_rng_alloc (gsl_rng_mt19937);

  theVector.reserve(numberOfNoisyChannels);
  float lowLimit = threshold * noiseRMS;
  for (int i = 0; i < numberOfNoisyChannels; i++) {

    // Find a random channel number    
    int theChannelNumber = (int)flatDistribution_->fire(NumberOfchannels);
    
    // Find random noise value
    float noise = gsl_ran_gaussian_tail(mt19937, lowLimit, noiseRMS);
              
    // Fill in the vector
    theVector.push_back(std::pair<int, float>(theChannelNumber, noise));
  }
}

void GaussianTailNoiseGenerator::generateRaw(int NumberOfchannels, 
					     float noiseRMS, 
					     std::vector<std::pair<int,float> > &theVector ) {
  theVector.reserve(NumberOfchannels);
  for (int i = 0; i < NumberOfchannels; i++) {
    // Find random noise value
    float noise = gaussDistribution_->fire(0.,noiseRMS);
    // Fill in the vector
    theVector.push_back(std::pair<int, float>(i,noise));
  }
}
