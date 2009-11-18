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
					size_t& minChannel, size_t& maxChannel,
					int ns, float nrms){
  numStrips = ns; 
  noiseRMS = nrms; 
  std::vector<std::pair<int,float> > generatedNoise;
  genNoise->generate(numStrips,threshold,noiseRMS,generatedNoise);
  
  // noise on strips with signal:
  for (size_t iChannel=minChannel; iChannel<=maxChannel; iChannel++) {
    if(in[iChannel] != 0) {
      in[iChannel] += gaussDistribution->fire(0.,noiseRMS);
    }
  }

  // Noise on the other strips
  typedef std::vector<std::pair<int,float> >::const_iterator VI;  
  for(VI p = generatedNoise.begin(); p != generatedNoise.end(); p++){
    if(in[(*p).first] == 0) {
      in[(*p).first] += (*p).second;
    }
  }
}

void SiGaussianTailNoiseAdder::createRaw(std::vector<double> &in,
					 size_t& minChannel, size_t& maxChannel,
					 int ns, float nrms, float ped){
  
  // Add noise
  // Full Gaussian noise is only added to signal strips, and only 
  // tails are added elsewhere. 
  // This is clearly wrong, but it's the best we can do.
  // Generating a Gaussian noise everywhere would lead to huge time
  // (some minutes)
  // Still, we must generate both + and - tails not to bias CMN algos.
  numStrips = ns; 
  noiseRMS = nrms; 
  std::vector<std::pair<int,float> > generatedNoiseP;
  genNoise->generate(numStrips,threshold,noiseRMS,generatedNoiseP);
  std::vector<std::pair<int,float> > generatedNoiseN;
  genNoise->generate(numStrips,threshold,noiseRMS,generatedNoiseN);
  
  // noise on strips with signal:
  for (size_t iChannel=minChannel; iChannel<=maxChannel; iChannel++) {
    if(in[iChannel] != 0) {
      in[iChannel] += gaussDistribution->fire(0.,noiseRMS);
    }
  }

  // Noise on the other strips
  typedef std::vector<std::pair<int,float> >::const_iterator VI;  
  for(VI p = generatedNoiseP.begin(); p != generatedNoiseP.end(); p++){
    if(in[(*p).first] == 0) {
      in[(*p).first] += (*p).second;
    }
  }
  for(VI p = generatedNoiseN.begin(); p != generatedNoiseN.end(); p++){
    if(in[(*p).first] == 0) {
      in[(*p).first] -= (*p).second;
    }
  }
  
  // Add pedestals
  for (size_t iChannel=0; iChannel!=in.size(); iChannel++) {
    in[iChannel] += ped;           
  }
}
