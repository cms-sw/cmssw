/*#include "SimCalorimetry/CaloSimAlgos/interface/CaloGaussianiNoisifier.h"
#include "CLHEP/Random/Random.h"
#include "CLHEP/Random/JamesRandom.h" 
#include "CLHEP/Random/RandGaussQ.h"


CaloGaussianNoisifier::CaloGaussianNoisifier(double sigma) :
  theRandomGaussian(*HepRandom::getTheEngine()),
  theSigma(sigma)
{
}

  
void HcalNoisifier::noisify(CaloSamples & frame) const
{
  int tsize = frame.size();
  for(int isample=0;isample<tsize;isample++){
    frame[isample] += theRandomGaussian.fire(0., theSigma);
  }
}
i*/
