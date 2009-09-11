#ifndef ECALDDIGIPRODUCER_H
#define ECALDDIGIPRODUCER_H

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "SimCalorimetry/CaloSimAlgos/interface/CaloHitResponse.h"
#include "SimCalorimetry/CaloSimAlgos/interface/CaloTDigitizer.h"
#include "SimCalorimetry/EcalSimAlgos/interface/EcalTDigitizer.h"
#include "SimGeneral/NoiseGenerators/interface/CorrelatedNoisifier.h"
#include "SimCalorimetry/EcalSimAlgos/interface/EcalSimParameterMap.h"
#include "SimCalorimetry/EcalSimAlgos/interface/EBShape.h"
#include "SimCalorimetry/EcalSimAlgos/interface/EEShape.h"
#include "SimCalorimetry/EcalSimAlgos/interface/EcalCorrelatedNoiseMatrix.h"
#include "SimCalorimetry/EcalSimAlgos/interface/ESShape.h"
#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"
#include "SimCalorimetry/EcalSimAlgos/interface/EcalCoder.h"
#include "SimCalorimetry/EcalSimAlgos/interface/EcalElectronicsSim.h"
#include "SimCalorimetry/EcalSimAlgos/interface/ESElectronicsSimFast.h"
#include "SimCalorimetry/EcalSimAlgos/interface/ESElectronicsSim.h"
#include "SimCalorimetry/EcalSimAlgos/interface/EcalDigitizerTraits.h"
#include "SimCalorimetry/EcalSimAlgos/interface/ESFastTDigitizer.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"

#include "DataFormats/Math/interface/Error.h"
#include "CalibFormats/CaloObjects/interface/CaloSamples.h"

class EcalDigiProducer : public edm::EDProducer
{

   public:

      // The following is not yet used, but will be the primary
      // constructor when the parameter set system is available.
      //
      explicit EcalDigiProducer(const edm::ParameterSet& params);
      virtual ~EcalDigiProducer();

      /**Produces the EDM products,*/
      virtual void produce( edm::Event&            event,
			    const edm::EventSetup& eventSetup);

   private:

      void checkGeometry(const edm::EventSetup & eventSetup);

      void updateGeometry();

      void checkCalibrations(const edm::EventSetup & eventSetup);

      /** Reconstruction algorithm*/
      typedef EcalTDigitizer<EBDigitizerTraits> EBDigitizer;
      typedef EcalTDigitizer<EEDigitizerTraits> EEDigitizer;
      typedef CaloTDigitizer<ESDigitizerTraits> ESDigitizer;

      EBDigitizer*      m_BarrelDigitizer;
      EEDigitizer*      m_EndcapDigitizer;
      ESDigitizer*      m_ESDigitizer;
      ESFastTDigitizer* m_ESDigitizerFast;

      const EcalSimParameterMap* m_ParameterMap;
      const EBShape              m_EBShape;
      const EEShape              m_EEShape;
      const ESShape*             m_ESShape;

      CaloHitResponse* m_EBResponse ;
      CaloHitResponse* m_EEResponse ;
      CaloHitResponse* m_ESResponse ;

      CorrelatedNoisifier<EcalCorrMatrix>* m_CorrNoise;
      EcalCorrelatedNoiseMatrix* m_NoiseMatrix;

      EcalElectronicsSim*   m_ElectronicsSim;
      ESElectronicsSim*     m_ESElectronicsSim;
      ESElectronicsSimFast* m_ESElectronicsSimFast;
      EcalCoder*            m_Coder;

      const CaloGeometry* m_Geometry;

      std::string m_EBdigiCollection ;
      std::string m_EEdigiCollection ;
      std::string m_ESdigiCollection ;

      std::string m_hitsProducerTag ;

      double m_EBs25notCont ;
      double m_EEs25notCont ;

      bool   m_doFast ; 
};

#endif 
