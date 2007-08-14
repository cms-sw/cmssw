/** \class GaussianTailNoiseGenerator
 * Generation of random noise
 * on "numberOfChannels" channels with a given threshold.
 * The generated noise : <BR>
 * - corresponds to a Gaussian distribution of RMS = "noiseRMS" <BR>
 * - is larger than threshold*noiseRMS. <BR>
 *
 * Initial author : Veronique Lefebure 08.10.98 <BR>
 *                  according to the FORTRAN code tgreset.F from Pascal Vanlaer <BR>
 *
 * Fills in a map \< channel number, generated noise \>
 */
#ifndef GaussianTailNoiseGenerator_h
#define GaussianTailNoiseGenerator_h

#include <vector>
#include <map>

namespace CLHEP {
  class HepRandomEngine;
}

#include "CLHEP/Random/RandPoisson.h"
#include "CLHEP/Random/RandGauss.h"
#include "CLHEP/Random/RandFlat.h"


class GaussianTailNoiseGenerator {

public:

  GaussianTailNoiseGenerator( CLHEP::HepRandomEngine& eng);

  // Compiler-generated destructor, copy c'tor, and assignment are all
  // correct.

  void generate(int NumberOfchannels, 
		float threshold,
		float noiseRMS, 
		std::map<int,float>& theMap );

  void generate(int NumberOfchannels, 
		float threshold,
		float noiseRMS, 
		std::vector<std::pair<int,float> >&);

  void generateRaw(int NumberOfchannels, 
		   float noiseRMS, 
		   std::vector<std::pair<int,float> >&);

  double generate_gaussian_tail(const double,const double);

private:
  CLHEP::RandGauss        gaussDistribution_;
  CLHEP::RandPoisson      poissonDistribution_;
  CLHEP::RandFlat         flatDistribution_;
};

#endif
