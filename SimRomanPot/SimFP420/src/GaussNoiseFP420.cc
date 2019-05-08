///////////////////////////////////////////////////////////////////////////////
// File: GaussNoiseFP420.cc
// Date: 12.2006
// Description: GaussNoiseFP420 for FP420
// Modifications:
///////////////////////////////////////////////////////////////////////////////
#include "CLHEP/Random/RandGauss.h"
#include "SimRomanPot/SimFP420/interface/GaussNoiseFP420.h"
#include "SimRomanPot/SimFP420/interface/GaussNoiseProducerFP420.h"

GaussNoiseFP420::GaussNoiseFP420(int ns, float nrms, float th, bool aNpixel, int verbosity)
    : numPixels(ns), noiseRMS(nrms), threshold(th), addNoisyPixels(aNpixel), verbosi(verbosity) {}

PileUpFP420::signal_map_type GaussNoiseFP420::addNoise(const PileUpFP420::signal_map_type &in) {
  PileUpFP420::signal_map_type _signal;

  // Add noise on non-hit pixels
  std::map<int, float, std::less<int>> generatedNoise;

  //  int numberOfPixels = (numRows * numColumns);// numPixels=numberOfPixels

  GaussNoiseProducerFP420 gen;
  gen.generate(numPixels, threshold, noiseRMS,
               generatedNoise);  // threshold is thePixelThreshold

  // noise for channels with signal:
  // ----------------------------

  for (PileUpFP420::signal_map_type::const_iterator si = in.begin(); si != in.end(); si++) {
    if (verbosi > 0) {
      std::cout << " ***GaussNoiseFP420:  before noise:" << std::endl;
      std::cout << " for si->first=  " << si->first << "    _signal[si->first]=  " << _signal[si->first]
                << "        si->second=      " << si->second << std::endl;
    }
    // define Gaussian noise centered at 0. with sigma = noiseRMS:
    float noise(CLHEP::RandGauss::shoot(0., noiseRMS));
    //    float noise  = CLHEP::RandGaussQ::shoot(0.,theNoiseInElectrons) ;
    // add noise to each channel with signal:
    _signal[si->first] = si->second + noise;

    if (verbosi > 0) {
      std::cout << " ***GaussNoiseFP420: after noise added  = " << noise << std::endl;
      std::cout << "after noise added the _signal[si->first]=  " << _signal[si->first] << std::endl;
    }
  }

  //                                                                                                    //
  if (addNoisyPixels) {  // Option to skip noise in non-hit pixels
    // Noise on the other channels:
    typedef std::map<int, float, std::less<int>>::iterator MI;
    for (MI p = generatedNoise.begin(); p != generatedNoise.end(); p++) {
      if (_signal[(*p).first] == 0) {
        _signal[(*p).first] += (*p).second;
      }
    }  // for(MI
  }

  // or:
  //                                                                        //
  /*
    if(addNoisyPixels){  // Option to skip noise in non-hit pixels
    // Noise on the other channels:
    typedef std::map<int,float,std::less<int> >::iterator MI;
    for(MI p = generatedNoise.begin(); p != generatedNoise.end(); p++){
    int iy = ((*p).first) / numRows;
    int ix = ((*p).first) - (iy*numRows);
    // Keep for a while for testing.
    if( iy < 0 || iy > (numColumns-1) )
    LogWarning ("Pixel Geometry") << " error in iy " << iy ;
    if( ix < 0 || ix > (numRows-1) )
    LogWarning ("Pixel Geometry")  << " error in ix " << ix ;
    int chan = PixelDigi::pixelToChannel(ix, iy);
    LogDebug ("Pixel Digitizer")<<" Storing noise = " << (*mapI).first << " " <<
    (*mapI).second
    << " " << ix << " " << iy << " " << chan ;
    if(_signal[chan] == 0){
    _signal[(*p).first] += (*p).second;
    }//if
    }//for(MI
    }
  */
  //                                                                        //
  //                                                                                                    //

  //
  return _signal;
  //
}
//
