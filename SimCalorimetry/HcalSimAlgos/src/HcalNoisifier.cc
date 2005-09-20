#include "SimCalorimetry/HcalSimAlgos/interface/HcalNoisifier.h"
#include "CalibFormats/HcalObjects/interface/HcalDbServiceHandle.h"
#include "CalibFormats/CaloObjects/interface/CaloSamples.h"
#include "CLHEP/Random/RandGaussQ.h"

#include<iostream>
namespace cms {

  HcalNoisifier::HcalNoisifier(HcalDbServiceHandle * calibrator) :
    theStartingCapId(0),
    theCalibrationHandle(calibrator)
  {
  }


  void HcalNoisifier::noisify(::CaloSamples & frame) const {
    HcalDetId hcalDetId(frame.id());
    const HcalCalibrations * calibrations = theCalibrationHandle->getHcalCalibrations(hcalDetId);
    const HcalCalibrationWidths * widths  = theCalibrationHandle->getHcalCalibrationWidths(hcalDetId);
    for(int tbin = 0; tbin < frame.size(); ++tbin) {
       //@@ assumes the capIDs start at zero
       int capId = (theStartingCapId + tbin)%4;
      //@@ replace by a real random number generator
      double pedestalJitter = theRandGaussian->shoot(0., widths->pedestal(capId));
      double gainJitter = theRandGaussian->shoot(1., widths->gain(capId));
      frame[tbin] *= gainJitter;
      // pedestals come in units of GeV.  Use gain to convert
      frame[tbin] += (calibrations->pedestal(capId) + pedestalJitter) / calibrations->gain(capId);
    }
    std::cout << frame << std::endl;
  }
}


