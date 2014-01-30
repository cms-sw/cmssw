#ifndef SimCalorimetry_EcalSimProducers_EcalTimeDigiProducer_h
#define SimCalorimetry_EcalSimProducers_EcalTimeDigiProducer_h


#include "DataFormats/Math/interface/Error.h"

#include "SimCalorimetry/EcalSimAlgos/interface/EcalDigitizerTraits.h"
#include "SimGeneral/MixingModule/interface/DigiAccumulatorMixMod.h"
#include "SimDataFormats/CaloHit/interface/PCaloHit.h"
#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"

#include <vector>

class ESDigitizer ;

class CaloGeometry ;
class EcalSimParameterMap ;
class PileUpEventPrincipal ;
class EcalTimeMapDigitizer;

namespace edm {
  class EDProducer;
  class Event;
  class EventSetup;
  template<typename T> class Handle;
  class ParameterSet;
}

class EcalTimeDigiProducer : public DigiAccumulatorMixMod {
   public:

      EcalTimeDigiProducer( const edm::ParameterSet& params , edm::EDProducer& mixMod);
      virtual ~EcalTimeDigiProducer();

      virtual void initializeEvent(edm::Event const& e, edm::EventSetup const& c);
      virtual void accumulate(edm::Event const& e, edm::EventSetup const& c);
      virtual void accumulate(PileUpEventPrincipal const& e, edm::EventSetup const& c);
      virtual void finalizeEvent(edm::Event& e, edm::EventSetup const& c);

   private:

      typedef edm::Handle<std::vector<PCaloHit> > HitsHandle;
      void accumulateCaloHits(HitsHandle const& ebHandle, HitsHandle const& eeHandle,  int bunchCrossing);

      void checkGeometry(const edm::EventSetup& eventSetup) ;

      void updateGeometry() ;

      const std::string m_EBdigiCollection ;
      const std::string m_EEdigiCollection ;
      const std::string m_hitsProducerTag  ;

   private:
      int m_timeLayerEB;
      int m_timeLayerEE;
      const CaloGeometry*   m_Geometry ;

      EcalTimeMapDigitizer* m_BarrelDigitizer;
      EcalTimeMapDigitizer* m_EndcapDigitizer;
};

#endif 
