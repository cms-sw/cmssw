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
                6, 4, false)
{
}
/*
  CaloSimParameters(double photomultiplierGain, double amplifierGain,
                 double samplingFactor, double timePhase,
                 int readoutFrameSize, int binOfMaximum,
                 bool doPhotostatistics)
*/

HcalSimParameterMap::HcalSimParameterMap(const edm::ParameterSet & p)
: theHBParameters(p.getParameter<double>("photomultiplierGainHB"), 
                  p.getParameter<double>("amplifierGainHB"),
                  p.getParameter<double>("samplingFactorHB"),
                  p.getParameter<double>("timePhaseHB"),
                  p.getParameter<int>("readoutFrameSizeHB"),
                  p.getParameter<int>("binOfMaximumHB"),
                  p.getParameter<bool>("doPhotostatisticsHB")),
  theHEParameters(p.getParameter<double>("photomultiplierGainHE"),
                  p.getParameter<double>("amplifierGainHE"),
                  p.getParameter<double>("samplingFactorHE"),
                  p.getParameter<double>("timePhaseHE"),
                  p.getParameter<int>("readoutFrameSizeHE"),
                  p.getParameter<int>("binOfMaximumHE"),
                  p.getParameter<bool>("doPhotostatisticsHE")),
  theHOParameters(p.getParameter<double>("photomultiplierGainHO"),
                  p.getParameter<double>("amplifierGainHO"),
                  p.getParameter<double>("samplingFactorHO"),
                  p.getParameter<double>("timePhaseHO"),
                  p.getParameter<int>("readoutFrameSizeHO"),
                  p.getParameter<int>("binOfMaximumHO"),
                  p.getParameter<bool>("doPhotostatisticsHO")),
  theHFParameters1(p.getParameter<double>("photomultiplierGainHF1"),
                   p.getParameter<double>("amplifierGainHF1"),
                   p.getParameter<double>("samplingFactorHF1"),
                   p.getParameter<double>("timePhaseHF1"),
                   p.getParameter<int>("readoutFrameSizeHF1"),
                   p.getParameter<int>("binOfMaximumHF1"),
                   p.getParameter<bool>("doPhotostatisticsHF1")),
  theHFParameters2(p.getParameter<double>("photomultiplierGainHF2"),
                   p.getParameter<double>("amplifierGainHF2"),
                   p.getParameter<double>("samplingFactorHF2"),
                   p.getParameter<double>("timePhaseHF2"),
                   p.getParameter<int>("readoutFrameSizeHF2"),
                   p.getParameter<int>("binOfMaximumHF2"),
                   p.getParameter<bool>("doPhotostatisticsHF2"))
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

