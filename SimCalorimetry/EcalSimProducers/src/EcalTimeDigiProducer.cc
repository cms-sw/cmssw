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


// #define ecal_time_debug 1

EcalTimeDigiProducer::EcalTimeDigiProducer( const edm::ParameterSet& params, edm::EDProducer& mixMod ) :
   DigiAccumulatorMixMod(),
   m_EBdigiCollection ( params.getParameter<std::string>("EBtimeDigiCollection") ) ,
   m_EEdigiCollection ( params.getParameter<std::string>("EEtimeDigiCollection") ) ,
   m_EKdigiCollection ( params.getParameter<std::string>("EKtimeDigiCollection") ) ,
   m_hitsProducerTag  ( params.getParameter<std::string>("hitsProducer"    ) ) ,
   m_timeLayerEB     (  params.getParameter<int> ("timeLayerBarrel") ),
   m_timeLayerEE     (  params.getParameter<int> ("timeLayerEndcap") ),
   m_timeLayerEK     (  params.getParameter<int> ("timeLayerShashlik") ),
   m_Geometry          ( 0 ) 
{
   mixMod.produces<EcalTimeDigiCollection>(m_EBdigiCollection);
   mixMod.produces<EcalTimeDigiCollection>(m_EEdigiCollection);
   mixMod.produces<EcalTimeDigiCollection>(m_EKdigiCollection);

   m_BarrelDigitizer = new EcalTimeMapDigitizer(EcalBarrel);
   m_EndcapDigitizer = new EcalTimeMapDigitizer(EcalEndcap);
   m_ShashlikDigitizer = new EcalTimeMapDigitizer(EcalShashlik);

#ifdef ecal_time_debug
   std::cout << "[EcalTimeDigiProducer]::Create EB " << m_EBdigiCollection;
   std::cout << " and EE " << m_EEdigiCollection;
   std::cout << " and EK " << m_EKdigiCollection;
   std::cout << " collections and digitizers" << std::endl;
#endif

   m_BarrelDigitizer->setTimeLayerId(m_timeLayerEB);
   m_EndcapDigitizer->setTimeLayerId(m_timeLayerEE);
   m_ShashlikDigitizer->setTimeLayerId(m_timeLayerEK);
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
   m_ShashlikDigitizer->initializeMap();
}

void
EcalTimeDigiProducer::accumulateCaloHits(HitsHandle const& ebHandle, HitsHandle const& eeHandle, HitsHandle const& ekHandle, int bunchCrossing) {
  // accumulate the simHits and do the averages in a given layer per bunch crossing
  if(ebHandle.isValid()) {
    m_BarrelDigitizer->add(*ebHandle.product(), bunchCrossing);
  }

  if(eeHandle.isValid()) {
    m_EndcapDigitizer->add(*eeHandle.product(), bunchCrossing);
  }

  if(ekHandle.isValid()) {
    m_ShashlikDigitizer->add(*ekHandle.product(), bunchCrossing);
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

  edm::InputTag ekTag(m_hitsProducerTag, "EcalHitsEK");
  edm::Handle<std::vector<PCaloHit> > ekHandle;
  e.getByLabel(ekTag, ekHandle);

#ifdef ecal_time_debug
  std::cout << "[EcalTimeDigiProducer]::Accumulate Hits HS  event" << std::endl;
#endif

  accumulateCaloHits(ebHandle, eeHandle, ekHandle, 0);
}

void
EcalTimeDigiProducer::accumulate(PileUpEventPrincipal const& e, edm::EventSetup const& eventSetup) {
  edm::InputTag ebTag(m_hitsProducerTag, "EcalHitsEB");
  edm::Handle<std::vector<PCaloHit> > ebHandle;
  e.getByLabel(ebTag, ebHandle);

  edm::InputTag eeTag(m_hitsProducerTag, "EcalHitsEE");
  edm::Handle<std::vector<PCaloHit> > eeHandle;
  e.getByLabel(eeTag, eeHandle);

  edm::InputTag ekTag(m_hitsProducerTag, "EcalHitsEK");
  edm::Handle<std::vector<PCaloHit> > ekHandle;
  e.getByLabel(ekTag, ekHandle);

#ifdef ecal_time_debug
  std::cout << "[EcalTimeDigiProducer]::Accumulate Hits for BC " << e.bunchCrossing() << std::endl;
#endif
  accumulateCaloHits(ebHandle, eeHandle, ekHandle, e.bunchCrossing());
}

void 
EcalTimeDigiProducer::finalizeEvent(edm::Event& event, edm::EventSetup const& eventSetup) {
   std::auto_ptr<EcalTimeDigiCollection> barrelResult   ( new EcalTimeDigiCollection() ) ;
   std::auto_ptr<EcalTimeDigiCollection> endcapResult   ( new EcalTimeDigiCollection() ) ;
   std::auto_ptr<EcalTimeDigiCollection> shashlikResult ( new EcalTimeDigiCollection() ) ;

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

   m_ShashlikDigitizer->run( *shashlikResult ) ;

#ifdef ecal_time_debug
   std::cout << "[EcalTimeDigiProducer]::EK Digi size " <<  shashlikResult->size() << std::endl;
#endif

    edm::LogInfo("TimeDigiInfo") << "EK Digis: " << shashlikResult->size() ;

#ifdef ecal_time_debug
    std::cout << "[EcalTimeDigiProducer]::putting collections into the event " << std::endl;
#endif

   event.put( barrelResult,    m_EBdigiCollection ) ;
   event.put( endcapResult,    m_EEdigiCollection ) ;
   event.put( shashlikResult,  m_EKdigiCollection ) ;
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
   m_ShashlikDigitizer->setGeometry(
      m_Geometry->getSubdetectorGeometry( DetId::Ecal, EcalShashlik  ) ) ;
}
