///////////////////////////////////////////////////////////////////////////////
// File: GaussNoiseFP420.cc
// Date: 12.2006
// Description: GaussNoiseFP420 for FP420
// Modifications: 
///////////////////////////////////////////////////////////////////////////////
#include "SimRomanPot/SimFP420/interface/GaussNoiseFP420.h"
#include "SimRomanPot/SimFP420/interface/GaussNoiseProducerFP420.h"
#include "CLHEP/Random/RandGauss.h"
//#define mydigidebug7

GaussNoiseFP420::GaussNoiseFP420(int ns, float nrms, float th):
  numStrips(ns), noiseRMS(nrms), threshold(th){}


PileUpFP420::signal_map_type 
GaussNoiseFP420::addNoise(PileUpFP420::signal_map_type in){
  
  PileUpFP420::signal_map_type _signal;  
  
  map<int,float,less<int> > generatedNoise;
  
  GaussNoiseProducerFP420 gen;
  gen.generate(numStrips,threshold,noiseRMS,generatedNoise);
  
  // noise for channels with signal:
  // ----------------------------
  
  for (PileUpFP420::signal_map_type::const_iterator si  = in.begin();
       si != in.end()  ; si++){
    
#ifdef mydigidebug7
    std::cout << " ***GaussNoiseFP420:  before noise:" << std::endl;
    std::cout << " for si->first=  " << si->first  << "    _signal[si->first]=  " << _signal[si->first] << "        si->second=      " << si->second  << std::endl;
#endif
    // define Gaussian noise centered at 0. with sigma = noiseRMS:
    float noise( RandGauss::shoot(0.,noiseRMS) );           
    // add noise to each channel with signal:
    _signal[si->first] = si->second + noise;
    
#ifdef mydigidebug7
    std::cout << " ***GaussNoiseFP420: after noise added  = " << noise  << std::endl;
    std::cout << "after noise added the _signal[si->first]=  " << _signal[si->first] << std::endl;
#endif
  }
  
  //
  // Noise on the other channels:
  
  typedef map<int,float,less<int> >::iterator MI;
  
  for(MI p = generatedNoise.begin(); p != generatedNoise.end(); p++){
    if(_signal[(*p).first] == 0) {
      _signal[(*p).first] += (*p).second;
    }
  }
  return _signal;
}
