#include "SimCalorimetry/HcalSimAlgos/interface/HcalElectronicsSim.h"
#include "SimCalorimetry/HcalSimAlgos/interface/HcalNoisifier.h"
#include "CalibFormats/HcalObjects/interface/HcalCoderDb.h"
#include "DataFormats/HcalDigi/interface/HBHEDataFrame.h"
#include "DataFormats/HcalDigi/interface/HODataFrame.h"
#include "DataFormats/HcalDigi/interface/HFDataFrame.h"
#include "CLHEP/Random/RandFlat.h"


namespace cms {

  HcalElectronicsSim::HcalElectronicsSim(HcalNoisifier * noisifier)
    : theNoisifier(noisifier),
      theDbService(0)
  {
  }


  template<class Digi> 
  void HcalElectronicsSim::convert(CaloSamples & frame, Digi & result, bool addNoise) {
    // make a coder first
    assert(theDbService != 0);
    const HcalQIECoder * qieCoder = theDbService->getHcalCoder( HcalDetId(frame.id()) );
    const HcalQIEShape * qieShape = theDbService->getHcalShape();
    HcalCoderDb coder(*qieCoder, *qieShape);

    result.setSize(frame.size());
    if(addNoise) theNoisifier->noisify(frame);
    coder.fC2adc(frame, result, theStartingCapId);

  }

  void HcalElectronicsSim::analogToDigital(CaloSamples & lf, HBHEDataFrame & result, bool addNoise) {
    convert<HBHEDataFrame>(lf, result, addNoise);
  }


  void HcalElectronicsSim::analogToDigital(CaloSamples & lf, HODataFrame & result, bool addNoise) {
    convert<HODataFrame>(lf, result, addNoise);
  }


  void HcalElectronicsSim::analogToDigital(CaloSamples & lf, HFDataFrame & result, bool addNoise) {
    convert<HFDataFrame>(lf, result, addNoise);
  }

  
  void HcalElectronicsSim::newEvent() {
    // pick a new starting Capacitor ID
    theStartingCapId = RandFlat::shootInt(4);
    theNoisifier->setStartingCapId(theStartingCapId);
  }


  /// the Producer will probably update this every event
  void HcalElectronicsSim::setDbService(const HcalDbService * service) {
    theDbService = service;
    theNoisifier->setDbService(service);
  }


}

