#include "SimPPS/RPDigiProducer/interface/RPGaussianTailNoiseAdder.h"
#include "CLHEP/Random/RandGauss.h"
#include <iostream>
#include "TMath.h"
#include "TRandom.h"

using namespace std;

RPGaussianTailNoiseAdder::RPGaussianTailNoiseAdder(int numStrips, 
    double theNoiseInElectrons, double theStripThresholdInE, int verbosity)
     : numStrips_(numStrips), theNoiseInElectrons_(theNoiseInElectrons), 
     theStripThresholdInE_(theStripThresholdInE)
{
  verbosity_ = verbosity;
  strips_above_threshold_prob_ = 
      TMath::Erfc(theStripThresholdInE_/sqrt(2.0)/theNoiseInElectrons_)/2;
}

SimRP::strip_charge_map RPGaussianTailNoiseAdder::addNoise(
    const SimRP::strip_charge_map &theSignal)
{
  the_strip_charge_map_.clear();
  
  // noise on strips with signal:
  for (SimRP::strip_charge_map::const_iterator i=theSignal.begin();
       i!=theSignal.end(); ++i)
  {
    double noise = CLHEP::RandGauss::shoot(0.0, theNoiseInElectrons_);
    the_strip_charge_map_[i->first] = i->second + noise;
    if(verbosity_)
      cout<<"noise added to signal strips: "<<noise<<endl;
  }
  
  // noise on the other strips
  int strips_no_above_threshold = gRandom->Binomial(numStrips_, 
      strips_above_threshold_prob_);
  
  for(int j=0; j<strips_no_above_threshold; j++)
  {
    int strip = gRandom->Integer(numStrips_);
    if(the_strip_charge_map_[strip] == 0)
    {
      the_strip_charge_map_[strip] = 2*theStripThresholdInE_;  
      //only binary decision later, no need to simulate the noise precisely, 
      //enough to know that it exceeds the threshold
      if(verbosity_)
        cout<<"nonsignal strips noise :"<<strip<<" "<<the_strip_charge_map_[strip]<<endl;
    }
  }
  
  return the_strip_charge_map_;
}
