/** \class GaussianTailNoiseGenerator
 * Generation of random noise
 * on "numberOfChannels" channels with a given threshold.
 * The generated noise : <BR>
 * - corresponds to a Gaussian distribution of RMS = "noiseRMS" <BR>
 * - is larger than threshold*noiseRMS. <BR>
 *
 * Initial author : Veronique Lefebure 08.10.98 <BR>
 *                  according to the FORTRAN code tgreset.F from Pascal Vanlaer
 * <BR> Modified by C. Delaere 01.10.09 <BR>
 *
 * Fills in a map \< channel number, generated noise \>
 */
#ifndef GaussianTailNoiseGenerator_h
#define GaussianTailNoiseGenerator_h

#include <map>
#include <vector>

namespace CLHEP {
  class HepRandomEngine;
}

class GaussianTailNoiseGenerator {
public:
  GaussianTailNoiseGenerator();

  // Compiler-generated destructor, copy c'tor, and assignment are all
  // correct.

  void generate(
      int NumberOfchannels, float threshold, float noiseRMS, std::map<int, float> &theMap, CLHEP::HepRandomEngine *);

  void generate(int NumberOfchannels,
                float threshold,
                float noiseRMS,
                std::vector<std::pair<int, float>> &,
                CLHEP::HepRandomEngine *);
  /*
    void generateRaw(int NumberOfchannels,
                     float noiseRMS,
                     std::vector<std::pair<int,float> >&,
                     CLHEP::HepRandomEngine*);
  */
  void generateRaw(float noiseRMS, std::vector<double> &, CLHEP::HepRandomEngine *);

protected:
  int *getRandomChannels(int, int, CLHEP::HepRandomEngine *);

  double generate_gaussian_tail(const double, const double, CLHEP::HepRandomEngine *);

private:
  int channel512_[512];
  int channel768_[768];
};

#endif
