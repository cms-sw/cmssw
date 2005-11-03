#include "SimTracker/SiStripDigitizer/interface/SiGaussianTailNoiseAdder.h"
#include "SimGeneral/NoiseGenerators/interface/GaussianTailNoiseGenerator.h"
#include "CLHEP/Random/RandGauss.h"

SiGaussianTailNoiseAdder::SiGaussianTailNoiseAdder(int ns, float nrms, float th):
  numStrips(ns), noiseRMS(nrms), threshold(th){}
   
  
SiPileUpSignals::signal_map_type 
SiGaussianTailNoiseAdder::addNoise(SiPileUpSignals::signal_map_type in){

  SiPileUpSignals::signal_map_type _signal;  
  
  map<int,float,less<int> > generatedNoise;
  
  GaussianTailNoiseGenerator gen;
  gen.generate(numStrips,threshold,noiseRMS,generatedNoise);

  // noise on strips with signal:
  // ----------------------------
  
  for (SiPileUpSignals::signal_map_type::const_iterator si  = in.begin();
       si != in.end()  ; si++){

    float noise( RandGauss::shoot(0.,noiseRMS) );           
    _signal[si->first] = si->second + noise;
    
  }
    
  //
  // Noise on the other strips
  
  typedef map<int,float,less<int> >::iterator MI;
  
  for(MI p = generatedNoise.begin(); p != generatedNoise.end(); p++){
    if(_signal[(*p).first] == 0) {
      _signal[(*p).first] += (*p).second;
    }
  }
  return _signal;
}
