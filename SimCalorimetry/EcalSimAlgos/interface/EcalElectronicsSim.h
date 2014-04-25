#ifndef EcalSimAlgos_EcalElectronicsSim_h
#define EcalSimAlgos_EcalElectronicsSim_h 1


#include "CalibFormats/CaloObjects/interface/CaloTSamples.h"


class EcalCoder           ;
class EcalDataFrame       ;
class EcalSimParameterMap ;

namespace CLHEP {
  class HepRandomEngine;
}

/* \class EcalElectronicsSim
 * \brief Converts CaloDataFrame in CaloTimeSample and vice versa.
 * 
 */

class EcalElectronicsSim
{
   public:

      typedef CaloTSamples<float,10> EcalSamples ;

      EcalElectronicsSim( const EcalSimParameterMap* parameterMap      , 
			  EcalCoder*                 coder             , 
			  bool                       applyConstantTerm , 
			  double                     rmsConstantTerm     ) ;

      ~EcalElectronicsSim() ;

      /// from EcalSamples to EcalDataFrame
      void analogToDigital( CLHEP::HepRandomEngine*, EcalSamples& clf, EcalDataFrame& df ) const ;

      void newEvent() {}

   private:

      /// input signal is in pe.  Converted in GeV
      void amplify( EcalSamples& clf, CLHEP::HepRandomEngine* ) const ;

      /// map of parameters

      const EcalSimParameterMap* m_simMap ;

      EcalCoder*                 m_theCoder ;

      const double               m_thisCT;
      const bool                 m_applyConstantTerm;
} ;


#endif
