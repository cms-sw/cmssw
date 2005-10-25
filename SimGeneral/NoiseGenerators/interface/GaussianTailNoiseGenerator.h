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
#define GaussianTailNoiseGenerator_h _ 1

#include <map>

class GaussianTailNoiseGenerator {

  public:
     
      GaussianTailNoiseGenerator(){}
     ~GaussianTailNoiseGenerator(){}

      void generate(int NumberOfchannels, 
		    float threshold,
		    float noiseRMS, 
		    std::map<int,float, std::less<int> >& theMap );

};

#endif
