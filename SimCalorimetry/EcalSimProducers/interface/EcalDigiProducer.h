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

#include "SimCalorimetry/CaloSimAlgos/interface/CaloTDigitizer.h"
#include "SimCalorimetry/EcalSimAlgos/interface/EcalTDigitizer.h"
#include "SimCalorimetry/EcalSimAlgos/interface/EcalDigitizerTraits.h"

typedef EcalTDigitizer<EBDigitizerTraits> EBDigitizer  ;
typedef EcalTDigitizer<EEDigitizerTraits> EEDigitizer  ;
typedef CaloTDigitizer<ESDigitizerTraits> ESDigitizer  ;


class APDSimParameters ;
class CaloHitRespoNew ;
class CaloHitResponse ;
class EBHitResponse ;
class EcalSimParameterMap ;
class EcalCoder ;
class EcalElectronicsSim ;
class ESElectronicsSim ;
class ESElectronicsSimFast ;
class ESFastTDigitizer ;
class CaloGeometry ;

class EcalDigiProducer : public edm::EDProducer
{

   public:

      EcalDigiProducer( const edm::ParameterSet& params ) ;
      virtual ~EcalDigiProducer() ;

      /**Produces the EDM products,*/
      virtual void produce( edm::Event&            event ,
			    const edm::EventSetup& eventSetup ) ;

   private:

      void checkGeometry(const edm::EventSetup& eventSetup) ;

      void updateGeometry() ;

      void checkCalibrations(const edm::EventSetup& eventSetup) ;

      EBDigitizer*      m_APDDigitizer ;
      EBDigitizer*      m_BarrelDigitizer ;
      EEDigitizer*      m_EndcapDigitizer ;
      ESDigitizer*      m_ESDigitizer ;
      ESFastTDigitizer* m_ESDigitizerFast ;

      const EcalSimParameterMap* m_ParameterMap ;
      const APDShape             m_APDShape ;
      const EBShape              m_EBShape ;
      const EEShape              m_EEShape ;
      ESShape*                   m_ESShape ;

      EBHitResponse*   m_APDResponse ;
      EBHitResponse*   m_EBResponse ;
      CaloHitRespoNew* m_EEResponse ;
      CaloHitResponse* m_ESResponse ;

      EcalElectronicsSim*   m_ElectronicsSim ;
      ESElectronicsSim*     m_ESElectronicsSim ;
      ESElectronicsSimFast* m_ESElectronicsSimFast ;
      EcalCoder*            m_Coder ;

      EcalElectronicsSim*   m_APDElectronicsSim ;
      EcalCoder*            m_APDCoder ;

      const CaloGeometry* m_Geometry ;

      std::string m_EBdigiCollection ;
      std::string m_EEdigiCollection ;
      std::string m_ESdigiCollection ;

      std::string m_hitsProducerTag ;

      double m_EBs25notCont ;
      double m_EEs25notCont ;
      bool   m_doFast       ; 

      APDSimParameters* m_apdParameters ;

      CorrelatedNoisifier<EcalCorrMatrix>* m_EBCorrNoise[3] ;
      CorrelatedNoisifier<EcalCorrMatrix>* m_EECorrNoise[3] ;
};

#endif 
