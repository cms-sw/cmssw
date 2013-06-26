#ifndef SimCalorimetry_EcalTestBeam_EcalTBDigiProducer_h
#define SimCalorimetry_EcalTestBeam_EcalTBDigiProducer_h

#include "SimCalorimetry/EcalSimProducers/interface/EcalDigiProducer.h"
#include "Geometry/CaloTopology/interface/EcalTrigTowerConstituentsMap.h"
#include "SimCalorimetry/EcalTestBeamAlgos/interface/EcalTBReadout.h"
#include "RecoTBCalo/EcalTBTDCReconstructor/interface/EcalTBTDCRecInfoAlgo.h"
#include "TBDataFormats/EcalTBObjects/interface/EcalTBTDCRawInfo.h"

namespace edm {
  class EDProducer;
  class Event;
  class EventSetup;
  class ParameterSet;
}
class PileUpEventPrincipal;

class EcalTBDigiProducer : public EcalDigiProducer
{
   public:

      EcalTBDigiProducer( const edm::ParameterSet& params, edm::EDProducer& mixMod ) ;
      virtual ~EcalTBDigiProducer() ;


      virtual void initializeEvent(edm::Event const&, edm::EventSetup const&);
      virtual void finalizeEvent(edm::Event&, edm::EventSetup const&);

   private:

      virtual void cacheEBDigis( const EBDigiCollection* ebDigiPtr ) const ;
      virtual void cacheEEDigis( const EEDigiCollection* eeDigiPtr ) const ; 

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
      mutable std::auto_ptr<EcalTBTDCRawInfo> m_TDCproduct ;
};

#endif 
