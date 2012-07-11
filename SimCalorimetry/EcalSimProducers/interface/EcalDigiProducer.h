#ifndef ECALDDIGIPRODUCER_H
#define ECALDDIGIPRODUCER_H

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "SimCalorimetry/EcalSimAlgos/interface/APDShape.h"
#include "SimCalorimetry/EcalSimAlgos/interface/EBShape.h"
#include "SimCalorimetry/EcalSimAlgos/interface/EEShape.h"
#include "SimCalorimetry/EcalSimAlgos/interface/ESShape.h"
#include "DataFormats/Math/interface/Error.h"
#include "SimGeneral/NoiseGenerators/interface/CorrelatedNoisifier.h"
#include "SimCalorimetry/EcalSimAlgos/interface/EcalCorrelatedNoiseMatrix.h"
#include "SimCalorimetry/EcalSimAlgos/interface/ESElectronicsSim.h"

#include "SimCalorimetry/CaloSimAlgos/interface/CaloTDigitizer.h"
#include "SimCalorimetry/EcalSimAlgos/interface/EcalTDigitizer.h"
#include "SimCalorimetry/EcalSimAlgos/interface/EcalDigitizerTraits.h"


typedef EcalTDigitizer<EBDigitizerTraits> EBDigitizer  ;
typedef EcalTDigitizer<EEDigitizerTraits> EEDigitizer  ;
typedef CaloTDigitizer<ESOldDigitizerTraits> ESOldDigitizer  ;

class ESDigitizer ;

class APDSimParameters ;
class EBHitResponse ;
class EEHitResponse ;
class ESHitResponse ;
class CaloHitResponse ;
class EcalSimParameterMap ;
class EcalCoder ;
class EcalElectronicsSim ;
class ESElectronicsSim ;
class ESElectronicsSimFast ;
class CaloGeometry ;
class EBDigiCollection ;
class EEDigiCollection ;
class ESDigiCollection ;

class EcalDigiProducer : public edm::EDProducer
{

   public:

      EcalDigiProducer( const edm::ParameterSet& params ) ;
      virtual ~EcalDigiProducer() ;

      /**Produces the EDM products,*/
      virtual void produce( edm::Event&            event ,
			    const edm::EventSetup& eventSetup ) ;

      virtual void cacheEBDigis( const EBDigiCollection* ebDigiPtr ) const { }
      virtual void cacheEEDigis( const EEDigiCollection* eeDigiPtr ) const { }

   protected:

      void checkGeometry(const edm::EventSetup& eventSetup) ;

      void updateGeometry() ;

      void checkCalibrations(const edm::EventSetup& eventSetup) ;

      const APDShape m_APDShape ;
      const EBShape  m_EBShape  ;
      const EEShape  m_EEShape  ;
      ESShape        m_ESShape  ; // no const because gain must be set

      const std::string m_EBdigiCollection ;
      const std::string m_EEdigiCollection ;
      const std::string m_ESdigiCollection ;
      const std::string m_hitsProducerTag  ;

      const bool m_apdSeparateDigi ;

      const double m_EBs25notCont ;
      const double m_EEs25notCont ;

      const unsigned int         m_readoutFrameSize ;
      const EcalSimParameterMap* m_ParameterMap  ;
      const std::string          m_apdDigiTag    ;
      const APDSimParameters*    m_apdParameters ;

      EBHitResponse* m_APDResponse ;
      EBHitResponse* m_EBResponse ;
      EEHitResponse* m_EEResponse ;
      ESHitResponse* m_ESResponse ;
      CaloHitResponse* m_ESOldResponse ;

      const bool m_addESNoise ;

      const bool m_doFastES   ;

      ESElectronicsSim*     m_ESElectronicsSim     ;
      ESOldDigitizer*       m_ESOldDigitizer       ;
      ESElectronicsSimFast* m_ESElectronicsSimFast ;
      ESDigitizer*          m_ESDigitizer          ;

      EBDigitizer*          m_APDDigitizer ;
      EBDigitizer*          m_BarrelDigitizer ;
      EEDigitizer*          m_EndcapDigitizer ;

      EcalElectronicsSim*   m_ElectronicsSim ;
      EcalCoder*            m_Coder ;

      EcalElectronicsSim*   m_APDElectronicsSim ;
      EcalCoder*            m_APDCoder ;

      const CaloGeometry*   m_Geometry ;

      CorrelatedNoisifier<EcalCorrMatrix>* m_EBCorrNoise[3] ;
      CorrelatedNoisifier<EcalCorrMatrix>* m_EECorrNoise[3] ;
};

#endif 
