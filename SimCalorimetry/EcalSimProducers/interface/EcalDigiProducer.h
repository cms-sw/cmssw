#ifndef ECALDDIGIPRODUCER_H
#define ECALDDIGIPRODUCER_H

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "DataFormats/Common/interface/Handle.h"
#include "SimCalorimetry/EcalSimAlgos/interface/EBShape.h"
#include "SimCalorimetry/EcalSimAlgos/interface/EEShape.h"
#include "SimCalorimetry/EcalSimAlgos/interface/ESShape.h"
#include "DataFormats/Math/interface/Error.h"
#include "SimCalorimetry/EcalSimAlgos/interface/EcalTDigitizer.h"
#include "SimCalorimetry/EcalSimAlgos/interface/EcalDigitizerTraits.h"
#include "SimCalorimetry/CaloSimAlgos/interface/CaloTDigitizer.h"
#include "SimGeneral/NoiseGenerators/interface/CorrelatedNoisifier.h"
#include "SimCalorimetry/EcalSimAlgos/interface/EcalCorrelatedNoiseMatrix.h"
#include "CondFormats/ESObjects/interface/ESIntercalibConstants.h"
#include "CondFormats/DataRecord/interface/ESIntercalibConstantsRcd.h"
#include "CondFormats/ESObjects/interface/ESMIPToGeVConstant.h"
#include "CondFormats/DataRecord/interface/ESMIPToGeVConstantRcd.h"
#include "CondFormats/ESObjects/interface/ESGain.h"
#include "CondFormats/DataRecord/interface/ESGainRcd.h"
#include "CondFormats/ESObjects/interface/ESPedestals.h"
#include "CondFormats/DataRecord/interface/ESPedestalsRcd.h"

class CaloHitResponse ;
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

      void checkGeometry(const edm::EventSetup & eventSetup) ;

      void updateGeometry() ;

      void checkCalibrations(const edm::EventSetup & eventSetup) ;

      /** Reconstruction algorithm*/
      typedef EcalTDigitizer<EBDigitizerTraits> EBDigitizer ;
      typedef EcalTDigitizer<EEDigitizerTraits> EEDigitizer ;
      typedef CaloTDigitizer<ESDigitizerTraits> ESDigitizer ;

      EBDigitizer*      m_BarrelDigitizer ;
      EEDigitizer*      m_EndcapDigitizer ;
      ESDigitizer*      m_ESDigitizer ;
      ESFastTDigitizer* m_ESDigitizerFast ;

      const EcalSimParameterMap* m_ParameterMap ;
      const EBShape              m_EBShape ;
      const EEShape              m_EEShape ;
      ESShape*         m_ESShape ;

      CaloHitResponse* m_EBResponse ;
      CaloHitResponse* m_EEResponse ;
      CaloHitResponse* m_ESResponse ;

      CorrelatedNoisifier<EcalCorrMatrix>* m_EBCorrNoise ;
      CorrelatedNoisifier<EcalCorrMatrix>* m_EECorrNoise ;

      EcalElectronicsSim*   m_ElectronicsSim ;
      ESElectronicsSim*     m_ESElectronicsSim ;
      ESElectronicsSimFast* m_ESElectronicsSimFast ;
      EcalCoder*            m_Coder ;

      const CaloGeometry* m_Geometry ;

      std::string m_EBdigiCollection ;
      std::string m_EEdigiCollection ;
      std::string m_ESdigiCollection ;

      std::string m_hitsProducerTag ;

      double m_EBs25notCont ;
      double m_EEs25notCont ;

      bool   m_doFast ; 

};

#endif 
