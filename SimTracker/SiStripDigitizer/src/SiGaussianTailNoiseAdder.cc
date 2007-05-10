#include "SimTracker/SiStripDigitizer/interface/SiGaussianTailNoiseAdder.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/RandomNumberGenerator.h"
#include "CLHEP/Random/RandGauss.h"
#include "FWCore/Utilities/interface/Exception.h"

SiGaussianTailNoiseAdder::SiGaussianTailNoiseAdder(int ns, float nrms, float th, CLHEP::HepRandomEngine& eng):
  numStrips(ns), 
  noiseRMS(nrms), 
  threshold(th),
  rndEngine(eng),
  gaussDistribution(0)
{
  genNoise = new GaussianTailNoiseGenerator(rndEngine);
  gaussDistribution = new CLHEP::RandGauss(rndEngine,0.,noiseRMS);

}

SiGaussianTailNoiseAdder::~SiGaussianTailNoiseAdder(){
  delete genNoise;
  delete gaussDistribution;
}

SiPileUpSignals::signal_map_type 
SiGaussianTailNoiseAdder::addNoise(SiPileUpSignals::signal_map_type in){
  
  SiPileUpSignals::signal_map_type _signal;  
  
  std::map<int,float,std::less<int> > generatedNoise;
  
  genNoise->generate(numStrips,threshold,noiseRMS,generatedNoise);

  // noise on strips with signal:
  // ----------------------------
  
  for (SiPileUpSignals::signal_map_type::const_iterator si  = in.begin();
       si != in.end()  ; si++){

    float noise = gaussDistribution->fire();           
    _signal[si->first] = si->second + noise;
    
  }
    
  //
  // Noise on the other strips
  
  typedef std::map<int,float,std::less<int> >::iterator MI;
  
  for(MI p = generatedNoise.begin(); p != generatedNoise.end(); p++){
    if(_signal[(*p).first] == 0) {
      _signal[(*p).first] += (*p).second;
    }
  }
  return _signal;
}
