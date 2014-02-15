#ifndef HcalSimAlgos_HcalElectronicsSim_h
#define HcalSimAlgos_HcalElectronicsSim_h
  
  /** This class turns a CaloSamples, representing the analog
      signal input to the readout electronics, into a
      digitized data frame
   */
#include "CalibFormats/CaloObjects/interface/CaloSamples.h"
#include "SimCalorimetry/HcalSimAlgos/interface/HcalTDC.h"

class HBHEDataFrame;
class HODataFrame;
class HFDataFrame;
class ZDCDataFrame;
class HcalUpgradeDataFrame;

class HcalAmplifier;
class HcalCoderFactory;

namespace CLHEP {
  class HepRandomEngine;
}

class HcalElectronicsSim {
public:
  HcalElectronicsSim(HcalAmplifier * amplifier, 
                     const HcalCoderFactory * coderFactory);
  ~HcalElectronicsSim();

  void setDbService(const HcalDbService * service);

  void analogToDigital(CLHEP::HepRandomEngine*, CaloSamples & linearFrame, HBHEDataFrame & result);
  void analogToDigital(CLHEP::HepRandomEngine*, CaloSamples & linearFrame, HODataFrame & result);
  void analogToDigital(CLHEP::HepRandomEngine*, CaloSamples & linearFrame, HFDataFrame & result);
  void analogToDigital(CLHEP::HepRandomEngine*, CaloSamples & linearFrame, ZDCDataFrame & result);
  void analogToDigital(CLHEP::HepRandomEngine*, CaloSamples & linearFrame, HcalUpgradeDataFrame& result);
  /// Things that need to be initialized every event
  /// sets starting CapID randomly
  void newEvent(CLHEP::HepRandomEngine*);
  void setStartingCapId(int startingCapId);

private:
  template<class Digi> void convert(CaloSamples & frame, Digi & result, CLHEP::HepRandomEngine*);

  HcalAmplifier * theAmplifier;
  const HcalCoderFactory * theCoderFactory;
  HcalTDC theTDC;

  int theStartingCapId;
  bool theStartingCapIdIsRandom;
};
#endif
