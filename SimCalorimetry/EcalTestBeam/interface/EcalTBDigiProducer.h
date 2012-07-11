#ifndef ECALTBDIGIPRODUCER_H
#define ECALTBDIGIPRODUCER_H

#include "SimCalorimetry/EcalSimProducers/interface/EcalDigiProducer.h"
#include "Geometry/CaloTopology/interface/EcalTrigTowerConstituentsMap.h"
#include "SimCalorimetry/EcalTestBeamAlgos/interface/EcalTBReadout.h"
#include "RecoTBCalo/EcalTBTDCReconstructor/interface/EcalTBTDCRecInfoAlgo.h"
#include "TBDataFormats/EcalTBObjects/interface/EcalTBTDCRawInfo.h"

class EcalTBDigiProducer : public EcalDigiProducer
{
   public:

      EcalTBDigiProducer( const edm::ParameterSet& params ) ;
      virtual ~EcalTBDigiProducer() ;

      /**Produces the EDM products,*/
      virtual void produce( edm::Event&            event ,
			    const edm::EventSetup& eventSetup ) ;

      virtual void cacheEBDigis( const EBDigiCollection* ebDigiPtr ) const ;
      virtual void cacheEEDigis( const EEDigiCollection* eeDigiPtr ) const ; 

   private:

      void setPhaseShift( const DetId& detId ) ;

      void fillTBTDCRawInfo( EcalTBTDCRawInfo& theTBTDCRawInfo ) ;

      const EcalTrigTowerConstituentsMap m_theTTmap        ;
      EcalTBReadout*                     m_theTBReadout    ;

      std::string m_ecalTBInfoLabel ;
      std::string m_EBdigiFinalTag  ;
      std::string m_EBdigiTempTag   ;

      bool   m_doPhaseShift   ;
      double m_thisPhaseShift ;

      bool   m_doReadout      ;

      std::vector<EcalTBTDCRecInfoAlgo::EcalTBTDCRanges> m_tdcRanges ;
      bool   m_use2004OffsetConvention ;
      
      double m_tunePhaseShift ;

      mutable std::auto_ptr<EBDigiCollection> m_ebDigis ;
      mutable std::auto_ptr<EEDigiCollection> m_eeDigis ;
};

#endif 
