
#ifndef EcalSimAlgos_EcalElectronicsSim_h
#define EcalSimAlgos_EcalElectronicsSim_h 1


#include "CalibFormats/CaloObjects/interface/CaloTSamples.h"
#include "SimCalorimetry/CaloSimAlgos/interface/CaloVNoiseSignalGenerator.h"
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

      typedef CaloTSamples<float,10> EcalSamples ;

      EcalElectronicsSim( const EcalSimParameterMap* parameterMap      , 
			  EcalCoder*                 coder             , 
			  bool                       applyConstantTerm , 
			  double                     rmsConstantTerm     ) ;

      ~EcalElectronicsSim() ;

      /// from EcalSamples to EcalDataFrame
      void analogToDigital( EcalSamples& clf, EcalDataFrame& df ) const ;

      void newEvent() {}

      void setNoiseSignalGenerator(const CaloVNoiseSignalGenerator * noiseSignalGenerator) {
	theNoiseSignalGenerator = noiseSignalGenerator;
      }

   private:

      /// input signal is in pe.  Converted in GeV
      void amplify( EcalSamples& clf ) const ;

      /// map of parameters

      const EcalSimParameterMap* m_simMap ;

      const CaloVNoiseSignalGenerator * theNoiseSignalGenerator;

      EcalCoder*                 m_theCoder ;

      CLHEP::RandGaussQ*         m_gaussQDistribution ;
} ;


#endif
