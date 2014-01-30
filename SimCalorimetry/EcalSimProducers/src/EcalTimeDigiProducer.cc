#include "FWCore/Framework/interface/Event.h"
#include "SimCalorimetry/EcalSimProducers/interface/EcalTimeDigiProducer.h"
#include "SimCalorimetry/EcalSimAlgos/interface/EcalTimeMapDigitizer.h"

#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "CalibFormats/CaloObjects/interface/CaloSamples.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "SimGeneral/MixingModule/interface/PileUpEventPrincipal.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "Geometry/EcalAlgo/interface/EcalEndcapGeometry.h"

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
}

EcalTimeDigiProducer::~EcalTimeDigiProducer() 
{
}

void
EcalTimeDigiProducer::initializeEvent(edm::Event const& event, edm::EventSetup const& eventSetup) {
   checkGeometry( eventSetup );
//    checkCalibrations( event, eventSetup );
   m_BarrelDigitizer->initializeMap();
   m_EndcapDigitizer->initializeMap();
}

void
EcalTimeDigiProducer::accumulateCaloHits(HitsHandle const& ebHandle, HitsHandle const& eeHandle, int bunchCrossing) {

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

  accumulateCaloHits(ebHandle, eeHandle, 0);
}

void
EcalTimeDigiProducer::accumulate(PileUpEventPrincipal const& e, edm::EventSetup const& eventSetup) {
  // Step A: Get Inputs
  edm::InputTag ebTag(m_hitsProducerTag, "EcalHitsEB");
  edm::Handle<std::vector<PCaloHit> > ebHandle;
  e.getByLabel(ebTag, ebHandle);

  edm::InputTag eeTag(m_hitsProducerTag, "EcalHitsEE");
  edm::Handle<std::vector<PCaloHit> > eeHandle;
  e.getByLabel(eeTag, eeHandle);

  accumulateCaloHits(ebHandle, eeHandle, e.bunchCrossing());
}

void 
EcalTimeDigiProducer::finalizeEvent(edm::Event& event, edm::EventSetup const& eventSetup) {
   // Step B: Create empty output

   std::auto_ptr<EcalTimeDigiCollection> barrelResult   ( new EcalTimeDigiCollection() ) ;
   std::auto_ptr<EcalTimeDigiCollection> endcapResult   ( new EcalTimeDigiCollection() ) ;
   
   // run the algorithm

   m_BarrelDigitizer->run( *barrelResult ) ;

   edm::LogInfo("TimeDigiInfo") << "EB time Digis: " << barrelResult->size() ;

   m_EndcapDigitizer->run( *endcapResult ) ;
   edm::LogInfo("TimeDigiInfo") << "EE Digis: " << endcapResult->size() ;

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
