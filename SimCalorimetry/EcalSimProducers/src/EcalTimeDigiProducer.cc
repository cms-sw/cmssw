#include "FWCore/Framework/interface/Event.h"
#include "SimCalorimetry/EcalSimProducers/interface/EcalTimeDigiProducer.h"
#include "SimCalorimetry/EcalSimAlgos/interface/EcalTimeMapDigitizer.h"

#include "CalibFormats/CaloObjects/interface/CaloSamples.h"

#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "Geometry/EcalAlgo/interface/EcalEndcapGeometry.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "SimGeneral/MixingModule/interface/PileUpEventPrincipal.h"


//#define ecal_time_debug 1

EcalTimeDigiProducer::EcalTimeDigiProducer( const edm::ParameterSet& params, edm::EDProducer& mixMod ) :
   DigiAccumulatorMixMod(),
   m_EBdigiCollection ( params.getParameter<std::string>("EBtimeDigiCollection") ) ,
   m_EEdigiCollection ( params.getParameter<std::string>("EEtimeDigiCollection") ) ,
   m_hitsProducerTag  ( params.getParameter<std::string>("hitsProducer"    ) ) ,
   m_timeLayerEB     (  params.getParameter<int> ("timeLayerBarrel") ),
   m_timeLayerEE     (  params.getParameter<int> ("timeLayerEndcap") ),
   m_Geometry          ( 0 ) 
{
   mixMod.produces<EcalTimeDigiCollection>(m_EBdigiCollection);
   mixMod.produces<EcalTimeDigiCollection>(m_EEdigiCollection);

   m_BarrelDigitizer = new EcalTimeMapDigitizer(EcalBarrel);
   m_EndcapDigitizer = new EcalTimeMapDigitizer(EcalEndcap);

#ifdef ecal_time_debug
   std::cout << "[EcalTimeDigiProducer]::Create EB " << m_EBdigiCollection << " and EE " << m_EEdigiCollection << " collections and digitizers" << std::endl;
#endif

   m_BarrelDigitizer->setTimeLayerId(m_timeLayerEB);
   m_EndcapDigitizer->setTimeLayerId(m_timeLayerEE);
}

EcalTimeDigiProducer::~EcalTimeDigiProducer() 
{
}

void
EcalTimeDigiProducer::initializeEvent(edm::Event const& event, edm::EventSetup const& eventSetup) {
   checkGeometry( eventSetup );
   //    checkCalibrations( event, eventSetup );
   // here the methods to clean the maps
   m_BarrelDigitizer->initializeMap();
   m_EndcapDigitizer->initializeMap();
}

void
EcalTimeDigiProducer::accumulateCaloHits(HitsHandle const& ebHandle, HitsHandle const& eeHandle, int bunchCrossing) {
  // accumulate the simHits and do the averages in a given layer per bunch crossing
  if(ebHandle.isValid()) {
    m_BarrelDigitizer->add(*ebHandle.product(), bunchCrossing);
  }

  if(eeHandle.isValid()) {
    m_EndcapDigitizer->add(*eeHandle.product(), bunchCrossing);
  }

}

void
EcalTimeDigiProducer::accumulate(edm::Event const& e, edm::EventSetup const& eventSetup) {
  // Step A: Get Inputs
  edm::InputTag ebTag(m_hitsProducerTag, "EcalHitsEB");
  edm::Handle<std::vector<PCaloHit> > ebHandle;
  e.getByLabel(ebTag, ebHandle);

  edm::InputTag eeTag(m_hitsProducerTag, "EcalHitsEE");
  edm::Handle<std::vector<PCaloHit> > eeHandle;
  e.getByLabel(eeTag, eeHandle);

#ifdef ecal_time_debug
  std::cout << "[EcalTimeDigiProducer]::Accumulate Hits HS  event" << std::endl;
#endif

  accumulateCaloHits(ebHandle, eeHandle, 0);
}

void
EcalTimeDigiProducer::accumulate(PileUpEventPrincipal const& e, edm::EventSetup const& eventSetup) {
  edm::InputTag ebTag(m_hitsProducerTag, "EcalHitsEB");
  edm::Handle<std::vector<PCaloHit> > ebHandle;
  e.getByLabel(ebTag, ebHandle);

  edm::InputTag eeTag(m_hitsProducerTag, "EcalHitsEE");
  edm::Handle<std::vector<PCaloHit> > eeHandle;
  e.getByLabel(eeTag, eeHandle);

#ifdef ecal_time_debug
  std::cout << "[EcalTimeDigiProducer]::Accumulate Hits for BC " << e.bunchCrossing() << std::endl;
#endif
  accumulateCaloHits(ebHandle, eeHandle, e.bunchCrossing());
}

void 
EcalTimeDigiProducer::finalizeEvent(edm::Event& event, edm::EventSetup const& eventSetup) {
   std::auto_ptr<EcalTimeDigiCollection> barrelResult   ( new EcalTimeDigiCollection() ) ;
   std::auto_ptr<EcalTimeDigiCollection> endcapResult   ( new EcalTimeDigiCollection() ) ;

#ifdef ecal_time_debug
   std::cout << "[EcalTimeDigiProducer]::finalizeEvent" << std::endl;
#endif

   // here basically just put everything in the final collections
   m_BarrelDigitizer->run( *barrelResult ) ;

#ifdef ecal_time_debug
   std::cout << "[EcalTimeDigiProducer]::EB Digi size " <<  barrelResult->size() << std::endl;
#endif

   edm::LogInfo("TimeDigiInfo") << "EB time Digis: " << barrelResult->size() ;

   m_EndcapDigitizer->run( *endcapResult ) ;

#ifdef ecal_time_debug
   std::cout << "[EcalTimeDigiProducer]::EE Digi size " <<  endcapResult->size() << std::endl;
#endif

    edm::LogInfo("TimeDigiInfo") << "EE Digis: " << endcapResult->size() ;

#ifdef ecal_time_debug
    std::cout << "[EcalTimeDigiProducer]::putting collections into the event " << std::endl;
#endif

   event.put( barrelResult,    m_EBdigiCollection ) ;
   event.put( endcapResult,    m_EEdigiCollection ) ;
}

void 
EcalTimeDigiProducer::checkGeometry( const edm::EventSetup & eventSetup ) 
{
   // TODO find a way to avoid doing this every event
   edm::ESHandle<CaloGeometry>               hGeometry   ;
   eventSetup.get<CaloGeometryRecord>().get( hGeometry ) ;

   const CaloGeometry* pGeometry = &*hGeometry;

   if( pGeometry != m_Geometry )
   {
      m_Geometry = pGeometry;
      updateGeometry();
   }
}

void
EcalTimeDigiProducer::updateGeometry() 
{
   m_BarrelDigitizer->setGeometry(
      m_Geometry->getSubdetectorGeometry( DetId::Ecal, EcalBarrel    ) ) ;
   m_EndcapDigitizer->setGeometry(
      m_Geometry->getSubdetectorGeometry( DetId::Ecal, EcalEndcap    ) ) ;
}
