#include "SimTracker/SiStripDigitizer/interface/SiGaussianTailNoiseAdder.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/RandomNumberGenerator.h"
#include "CLHEP/Random/RandGaussQ.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "Utilities/Timing/interface/TimingReport.h"

SiGaussianTailNoiseAdder::SiGaussianTailNoiseAdder(float th,CLHEP::HepRandomEngine& eng):
  threshold(th),
  rndEngine(eng),
  gaussDistribution(0)
{
  genNoise = new GaussianTailNoiseGenerator(rndEngine);
  gaussDistribution = new CLHEP::RandGaussQ(rndEngine);
}

SiGaussianTailNoiseAdder::~SiGaussianTailNoiseAdder(){
  delete genNoise;
  delete gaussDistribution;
}

void SiGaussianTailNoiseAdder::addNoise(std::vector<double> &in,
					unsigned int& minChannel, unsigned int& maxChannel,
					int ns, float nrms){
  
  numStrips = ns; 
  noiseRMS = nrms; 
  
  std::vector<std::pair<int,float> > generatedNoise;

  genNoise->generate(numStrips,threshold,noiseRMS,generatedNoise);
  
  // noise on strips with signal:
  // ----------------------------
  
  for (unsigned int iChannel=minChannel; iChannel<=maxChannel; iChannel++) {
    if(in[iChannel] != 0) {
      float noise = gaussDistribution->fire(0.,noiseRMS);           
      in[iChannel] += noise;
    }
  }

  //
  // Noise on the other strips
  
  typedef std::vector<std::pair<int,float> >::const_iterator VI;  
  
  for(VI p = generatedNoise.begin(); p != generatedNoise.end(); p++){
    if(in[(*p).first] == 0) {
      in[(*p).first] += (*p).second;
    }
  }
  
}

void SiGaussianTailNoiseAdder::createRaw(std::vector<double> &in,
					 unsigned int& minChannel, unsigned int& maxChannel,
					 int ns, float nrms){
  
  numStrips = ns; 
  noiseRMS = nrms; 
  
  std::vector<std::pair<int,float> > generatedNoise;
  
  genNoise->generateRaw(numStrips,noiseRMS,generatedNoise);
  
  // noise on strips with signal:
  // ----------------------------
  
  for (unsigned int iChannel=minChannel; iChannel<=maxChannel; iChannel++) {
    if(in[iChannel] != 0) {
      float noise = gaussDistribution->fire(0.,noiseRMS);           
      in[iChannel] += noise;
    }
  }
  
  //
  // Noise on the other strips
  
  typedef std::vector<std::pair<int,float> >::const_iterator VI;  
  
  for(VI p = generatedNoise.begin(); p != generatedNoise.end(); p++){
    if(in[(*p).first] == 0) {
      in[(*p).first] += (*p).second;
    }
  }
  
}
