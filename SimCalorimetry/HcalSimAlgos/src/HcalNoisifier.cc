#include "SimCalorimetry/HcalSimAlgos/interface/HcalNoisifier.h"
#include "CalibFormats/HcalObjects/interface/HcalDbService.h"
#include "CalibFormats/HcalObjects/interface/HcalCalibrations.h"
#include "CalibFormats/HcalObjects/interface/HcalCalibrationWidths.h"
#include "CalibFormats/CaloObjects/interface/CaloSamples.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "CLHEP/Random/RandGaussQ.h"

#include<iostream>
namespace cms {

  HcalNoisifier::HcalNoisifier() :
    theStartingCapId(0),
    theDbService(0)
  {
  }


  void HcalNoisifier::noisify(CaloSamples & frame) const {
    assert(theDbService != 0);
    HcalDetId hcalDetId(frame.id());
    HcalCalibrations calibrations;
    HcalCalibrationWidths widths;
    if (!theDbService->makeHcalCalibration (hcalDetId, &calibrations) ||
	!theDbService->makeHcalCalibrationWidth (hcalDetId, &widths)) {
      //TODO handle this gracefully
      assert (0);
    }

    for(int tbin = 0; tbin < frame.size(); ++tbin) {
      int capId = (theStartingCapId + tbin)%4;
//std::cout << "PEDS " << capId << " " << calibrations->pedestal(capId) << " " << widths->pedestal(capId) << " " << calibrations->gain(capId) <<" " << widths->gain(capId) << std::endl;
      double pedestal = theRandGaussian->shoot(calibrations.pedestal(capId), widths.pedestal(capId));
      double gain0 = calibrations.gain(capId);
      double gain = theRandGaussian->shoot(gain0, widths.gain(capId));
      // pedestals come in units of fC for now. Could be in ADC's eventually
      frame[tbin] += pedestal; // noisify pedestal
      frame[tbin] = frame[tbin] / gain * gain0; // nosify gain
    }
    //std::cout << frame << std::endl;
  }
}


