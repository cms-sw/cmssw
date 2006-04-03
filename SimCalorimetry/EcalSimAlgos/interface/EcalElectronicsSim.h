
#ifndef EcalSimAlgos_EcalElectronicsSim_h
#define EcalSimAlgos_EcalElectronicsSim_h 1


#include "CalibFormats/CaloObjects/interface/CaloSamples.h"


class EcalCoder;
class EBDataFrame;
class EEDataFrame;
class EcalSimParameterMap;


/* \class EcalElectronicsSim
 * \brief Converts CaloDataFrame in CaloTimeSample and vice versa.
 * 
 */                                                                                            
class EcalElectronicsSim
{
 public:
  /// ctor
  EcalElectronicsSim(const EcalSimParameterMap * parameterMap, EcalCoder * coder) ;

  /// input signal is in pe.  Converted in GeV
  void amplify(CaloSamples & clf) const;

  /// from CaloSamples to EBDataFrame
  void analogToDigital(CaloSamples& clf, EBDataFrame& df) const;
  /// from CaloSamples to EEDataFrame
  void analogToDigital(CaloSamples& clf, EEDataFrame& df) const;
 
  ///  anything that needs to be done once per event
  void newEvent() {}

 private:

  /// map of parameters
  const EcalSimParameterMap * theParameterMap;
  /// Converts CaloDataFrame in CaloTimeSample and vice versa
  EcalCoder * theCoder;
} ;


#endif
