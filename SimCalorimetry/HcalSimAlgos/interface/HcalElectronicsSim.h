#ifndef HcalSimAlgos_HcalElectronicsSim_h
#define HcalSimAlgos_HcalElectronicsSim_h

/** This class turns a CaloSamples, representing the analog
      signal input to the readout electronics, into a
      digitized data frame
   */
#include "CalibFormats/CaloObjects/interface/CaloSamples.h"
#include "SimCalorimetry/HcalSimAlgos/interface/HcalTDC.h"
#include "SimCalorimetry/HcalSimAlgos/interface/HcalAmplifier.h"
#include "SimCalorimetry/HcalSimAlgos/interface/HcalCoderFactory.h"
#include "SimCalorimetry/HcalSimAlgos/interface/HcalSimParameterMap.h"

class HBHEDataFrame;
class HODataFrame;
class HFDataFrame;
class ZDCDataFrame;
class QIE10DataFrame;
class QIE11DataFrame;

namespace CLHEP {
  class HepRandomEngine;
}

class HcalElectronicsSim {
public:
  HcalElectronicsSim(const HcalSimParameterMap* parameterMap,
                     HcalAmplifier* amplifier,
                     const HcalCoderFactory* coderFactory,
                     bool PreMix);
  ~HcalElectronicsSim();

  void setDbService(const HcalDbService* service);

  //these need to be overloads instead of templates to avoid linking issues when calling private member function templates
  void analogToDigital(CLHEP::HepRandomEngine*,
                       CaloSamples& linearFrame,
                       HBHEDataFrame& result,
                       double preMixFactor = 10.0,
                       unsigned preMixBits = 126);
  void analogToDigital(CLHEP::HepRandomEngine*,
                       CaloSamples& linearFrame,
                       HODataFrame& result,
                       double preMixFactor = 10.0,
                       unsigned preMixBits = 126);
  void analogToDigital(CLHEP::HepRandomEngine*,
                       CaloSamples& linearFrame,
                       HFDataFrame& result,
                       double preMixFactor = 10.0,
                       unsigned preMixBits = 126);
  void analogToDigital(CLHEP::HepRandomEngine*,
                       CaloSamples& linearFrame,
                       ZDCDataFrame& result,
                       double preMixFactor = 10.0,
                       unsigned preMixBits = 126);
  void analogToDigital(CLHEP::HepRandomEngine*,
                       CaloSamples& linearFrame,
                       QIE10DataFrame& result,
                       double preMixFactor = 10.0,
                       unsigned preMixBits = 126);
  void analogToDigital(CLHEP::HepRandomEngine*,
                       CaloSamples& linearFrame,
                       QIE11DataFrame& result,
                       double preMixFactor = 10.0,
                       unsigned preMixBits = 126);
  /// Things that need to be initialized every event
  /// sets starting CapID randomly
  void newEvent(CLHEP::HepRandomEngine*);
  void setStartingCapId(int startingCapId);

private:
  template <class Digi>
  void analogToDigitalImpl(
      CLHEP::HepRandomEngine*, CaloSamples& linearFrame, Digi& result, double preMixFactor, unsigned preMixBits);
  template <class Digi>
  void convert(CaloSamples& frame, Digi& result, CLHEP::HepRandomEngine*);
  template <class Digi>
  void premix(CaloSamples& frame, Digi& result, double preMixFactor, unsigned preMixBits);

  const HcalSimParameterMap* theParameterMap;
  HcalAmplifier* theAmplifier;
  const HcalCoderFactory* theCoderFactory;
  HcalTDC theTDC;

  int theStartingCapId;
  bool theStartingCapIdIsRandom;
  bool PreMixDigis;
};

#endif
