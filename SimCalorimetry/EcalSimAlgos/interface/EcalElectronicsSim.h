
#ifndef EcalSimAlgos_EcalElectronicsSim_h
#define EcalSimAlgos_EcalElectronicsSim_h 1


#include "CalibFormats/CaloObjects/interface/CaloSamples.h"


class EcalCoder;
class EcalDataFrame;
class EcalSimParameterMap;


/* \class EcalElectronicsSim
 * \brief Converts CaloDataFrame in CaloTimeSample and vice versa.
 * 
 */                                                                                            
class EcalElectronicsSim
{
 public:
  /// ctor
  EcalElectronicsSim(const EcalSimParameterMap * parameterMap, EcalCoder * coder, bool applyConstantTerm, double rmsConstantTerm) ;

  /// input signal is in pe.  Converted in GeV
  void amplify(CaloSamples & clf) const;

  /// from CaloSamples to EcalDataFrame
  void analogToDigital(CaloSamples& clf, EcalDataFrame& df) const;
  /// compute the event random constant term
  double constantTerm() const;

  ///  anything that needs to be done once per event
  void newEvent() {}

 private:

  /// map of parameters
  const EcalSimParameterMap * theParameterMap;
  /// Converts CaloDataFrame in CaloTimeSample and vice versa
  EcalCoder * theCoder;
  const bool applyConstantTerm_;
  const double rmsConstantTerm_;
} ;


#endif
