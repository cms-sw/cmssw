
#ifndef EcalSimAlgos_EcalElectronicsSim_h
#define EcalSimAlgos_EcalElectronicsSim_h 1


#include "CalibFormats/CaloObjects/interface/CaloSamples.h"


class EcalCoder;
class EBDataFrame;
class EEDataFrame;
class EcalSimParameterMap;


//! Converts CaloDataFrame in CaloTimeSample and vice versa.

class EcalElectronicsSim
{
 public:
  EcalElectronicsSim(const EcalSimParameterMap * parameterMap, EcalCoder * coder) ;

  void amplify(CaloSamples & clf) const;

  void analogToDigital(CaloSamples& clf, EBDataFrame& df) const;
  void analogToDigital(CaloSamples& clf, EEDataFrame& df) const;
 
  ///  anything that needs to be done once per event
  void newEvent() {}

 private:

  const EcalSimParameterMap * theParameterMap;
  EcalCoder * theCoder;
} ;


#endif
