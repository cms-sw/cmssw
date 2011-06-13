
#ifndef EcalSimAlgos_EcalElectronicsSim_h
#define EcalSimAlgos_EcalElectronicsSim_h 1


#include "CalibFormats/CaloObjects/interface/CaloSamples.h"
#include "CLHEP/Random/RandGaussQ.h"


class EcalCoder           ;
class EcalDataFrame       ;
class EcalSimParameterMap ;


/* \class EcalElectronicsSim
 * \brief Converts CaloDataFrame in CaloTimeSample and vice versa.
 * 
 */

class EcalElectronicsSim
{
   public:

      EcalElectronicsSim( const EcalSimParameterMap* parameterMap      , 
			  EcalCoder*                 coder             , 
			  bool                       applyConstantTerm , 
			  double                     rmsConstantTerm     ) ;

      ~EcalElectronicsSim() ;

      /// from CaloSamples to EcalDataFrame
      void analogToDigital( CaloSamples& clf, EcalDataFrame& df ) const ;

      void newEvent() {}

   private:

      /// input signal is in pe.  Converted in GeV
      void amplify( CaloSamples & clf ) const ;

      /// map of parameters

      const EcalSimParameterMap* m_simMap ;

      EcalCoder*                 m_theCoder ;

      CLHEP::RandGaussQ*         m_gaussQDistribution ;
} ;


#endif
