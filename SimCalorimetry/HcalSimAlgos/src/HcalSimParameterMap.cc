#include "SimCalorimetry/HcalSimAlgos/interface/HcalSimParameterMap.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/HcalDetId/interface/HcalSubdetector.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"
  

// the second value, amplifierGain, maybe should be 1/(4 pe/fC) in barrel,
// and 1/(0.3 pe/fC) in the endcap.  I'll change them to make them
// consistent with the current default calibrations.
HcalSimParameterMap::HcalSimParameterMap() :
  theHBParameters(2000., 0.3305,
                   117, 5, 
                   10, 5, true),
  theHEParameters(2000., 0.3305,
                   178, 5,
                   10, 5, true),
  theHOParameters( 4000., 0.3065, 217., 5, 10, 5, true),
  theHFParameters1(1., 18.93,
                 2.84 , -4,
                6, 4, false),
  theHFParameters2(1., 13.93,
                 2.09 , -4,
                6, 4, false),
  theSamplingFactors(29),
  theSamplingFactorHF1(2.84),
  theSamplingFactorHF2(2.09)
{
  for(unsigned i = 0; i <16; ++i)
  {
    theSamplingFactors[i] = 117.;
  }
  for(unsigned i = 16; i <29; ++i)
  {
    theSamplingFactors[i] = 178.;
  }
}
/*
  CaloSimParameters(double photomultiplierGain, double amplifierGain,
                 double samplingFactor, double timePhase,
                 int readoutFrameSize, int binOfMaximum,
                 bool doPhotostatistics)
*/

HcalSimParameterMap::HcalSimParameterMap(const edm::ParameterSet & p)
: theHBParameters(  p.getParameter<edm::ParameterSet>("hb") ),
  theHEParameters(  p.getParameter<edm::ParameterSet>("he") ),
  theHOParameters(  p.getParameter<edm::ParameterSet>("ho") ),
  theHFParameters1( p.getParameter<edm::ParameterSet>("hf1") ),
  theHFParameters2( p.getParameter<edm::ParameterSet>("hf2") ),
  theSamplingFactors( p.getParameter<std::vector<double> >("samplingFactors") ),
  theSamplingFactorHF1( p.getParameter<double>("samplingFactorHF1") ),
  theSamplingFactorHF2( p.getParameter<double>("samplingFactorHF2") )
{
}



const CaloSimParameters & HcalSimParameterMap::simParameters(const DetId & detId) const {
  HcalDetId hcalDetId(detId);
  if(hcalDetId.subdet() == HcalBarrel) {
     return theHBParameters;
  } else if(hcalDetId.subdet() == HcalEndcap) {
     return theHEParameters;
  } else if(hcalDetId.subdet() == HcalOuter) {
     return theHOParameters;
  } else { // HF
    if(hcalDetId.depth() == 1) {
      return theHFParameters1;
    } else {
      return theHFParameters2;
    }
  }
}


double HcalSimParameterMap::samplingFactor(const HcalDetId & detId) const
{
  double result;
  if(detId.subdet() == HcalForward)
  {
    result = (detId.depth() == 1) ? theSamplingFactorHF1 : theSamplingFactorHF2;
  }
  else
  {
    result = theSamplingFactors[detId.ietaAbs()-1];
  }
  return result;
}

