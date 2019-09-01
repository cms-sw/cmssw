///////////////////////////////////////////////////////////////////////////////
// File: GaussNoiseProducerFP420.cc
// Date: 12.2006
// Description: GaussNoiseFP420 for FP420
// Modifications:
///////////////////////////////////////////////////////////////////////////////
#include "CLHEP/Random/RandFlat.h"
#include "CLHEP/Random/RandPoisson.h"
#include "SimRomanPot/SimFP420/interface/GaussNoiseProducerFP420.h"
#include <gsl/gsl_randist.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_sf_erf.h>
#include <gsl/gsl_sf_result.h>

extern "C" float freq_(const float &x);
extern "C" float gausin_(const float &x);

void GaussNoiseProducerFP420::generate(int NumberOfchannels,
                                       float threshold,
                                       float noiseRMS,
                                       std::map<int, float, std::less<int>> &theMap) {
  // estimale mean number of noisy channels with amplidudes above $AdcThreshold$

  // Gauss is centered at 0 with sigma=1
  // Gaussian tail probability higher threshold(=5sigma for instance):
  gsl_sf_result result;
  int status = gsl_sf_erf_Q_e(threshold, &result);
  // MP
  //  if (status != 0) throw DetLogicError("GaussNoiseProducerFP420::could not
  //  compute gaussian tail probability for the threshold chosen");
  if (status != 0)
    std::cerr << "GaussNoiseProducerFP420::could not compute gaussian tail "
                 "probability for the threshold chosen"
              << std::endl;
  float probabilityLeft = result.val;

  // with known probability higher threshold compute number of noisy channels
  // distributed in Poisson:
  float meanNumberOfNoisyChannels = probabilityLeft * NumberOfchannels;
  int numberOfNoisyChannels = CLHEP::RandPoisson::shoot(meanNumberOfNoisyChannels);

  // draw noise at random according to Gaussian tail

  // initialise default gsl uniform generator engine
  static gsl_rng const *const mt19937 = gsl_rng_alloc(gsl_rng_mt19937);

  float lowLimit = threshold * noiseRMS;
  for (int i = 0; i < numberOfNoisyChannels; i++) {
    // Find a random channel number
    int theChannelNumber = (int)CLHEP::RandFlat::shootInt(NumberOfchannels);

    // Find random noise value: random mt19937 over Gaussian tail above
    // threshold:
    float noise = gsl_ran_gaussian_tail(mt19937, lowLimit, noiseRMS);

    // Fill in map
    theMap[theChannelNumber] = noise;
  }
}
