#include "SimCalorimetry/HcalSimAlgos/interface/HcalNoisifier.h"
#include "CalibFormats/HcalObjects/interface/HcalDbService.h"
#include "CalibFormats/CaloObjects/interface/CaloSamples.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "CLHEP/Random/RandGaussQ.h"

#include<iostream>
namespace cms {

  HcalNoisifier::HcalNoisifier(HcalDbService * calibrator) :
    theStartingCapId(0),
    theCalibrationHandle(calibrator)
  {
  }


  void HcalNoisifier::noisify(CaloSamples & frame) const {
    HcalDetId hcalDetId(frame.id());
    HcalCalibrations calibrations = *(theCalibrationHandle->getHcalCalibrations(hcalDetId));
    HcalCalibrationWidths widths = *(theCalibrationHandle->getHcalCalibrationWidths(hcalDetId));
    for(int tbin = 0; tbin < frame.size(); ++tbin) {

      int capId = (theStartingCapId + tbin)%4;
      double pedestal = theRandGaussian->shoot(calibrations.pedestal(capId), widths.pedestal(capId));
      double gain = theRandGaussian->shoot(calibrations.gain(capId), widths.gain(capId));
      // pedestals come in units of GeV.  Use gain to convert
      frame[tbin] = (frame[tbin]+pedestal) / gain;
    }
std::cout << "AFTER HCAL NOISE  " << std::endl;
    std::cout << frame << std::endl;
  }
}


