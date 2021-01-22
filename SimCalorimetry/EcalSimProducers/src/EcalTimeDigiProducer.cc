#include "FWCore/Framework/interface/Event.h"
#include "SimCalorimetry/EcalSimAlgos/interface/EcalTimeMapDigitizer.h"
#include "SimCalorimetry/EcalSimProducers/interface/EcalTimeDigiProducer.h"
#include "CalibFormats/CaloObjects/interface/CaloSamples.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "SimGeneral/MixingModule/interface/PileUpEventPrincipal.h"

//#define ecal_time_debug 1

EcalTimeDigiProducer::EcalTimeDigiProducer(const edm::ParameterSet &params,
                                           edm::ProducesCollector producesCollector,
                                           edm::ConsumesCollector &sumes)
    : DigiAccumulatorMixMod(),
      m_EBdigiCollection(params.getParameter<std::string>("EBtimeDigiCollection")),
      m_hitsProducerTagEB(params.getParameter<edm::InputTag>("hitsProducerEB")),
      m_hitsProducerTokenEB(sumes.consumes<std::vector<PCaloHit>>(m_hitsProducerTagEB)),
      m_timeLayerEB(params.getParameter<int>("timeLayerBarrel")),
      m_Geometry(nullptr) {
  producesCollector.produces<EcalTimeDigiCollection>(m_EBdigiCollection);

  m_BarrelDigitizer = new EcalTimeMapDigitizer(EcalBarrel);

#ifdef ecal_time_debug
  std::cout << "[EcalTimeDigiProducer]::Create EB " << m_EBdigiCollection << " "
            << " collection and digitizer" << std::endl;
#endif

  m_BarrelDigitizer->setTimeLayerId(m_timeLayerEB);
}

EcalTimeDigiProducer::~EcalTimeDigiProducer() {}

void EcalTimeDigiProducer::initializeEvent(edm::Event const &event, edm::EventSetup const &eventSetup) {
  checkGeometry(eventSetup);
  //    checkCalibrations( event, eventSetup );
  // here the methods to clean the maps
  m_BarrelDigitizer->initializeMap();
}

void EcalTimeDigiProducer::accumulateCaloHits(HitsHandle const &ebHandle, int bunchCrossing) {
  // accumulate the simHits and do the averages in a given layer per bunch
  // crossing
  if (ebHandle.isValid()) {
    m_BarrelDigitizer->add(*ebHandle.product(), bunchCrossing);
  }
}

void EcalTimeDigiProducer::accumulate(edm::Event const &e, edm::EventSetup const &eventSetup) {
  // Step A: Get Inputs
  edm::Handle<std::vector<PCaloHit>> ebHandle;
  e.getByToken(m_hitsProducerTokenEB, ebHandle);

#ifdef ecal_time_debug
  std::cout << "[EcalTimeDigiProducer]::Accumulate Hits HS  event" << std::endl;
#endif

  accumulateCaloHits(ebHandle, 0);
}

void EcalTimeDigiProducer::accumulate(PileUpEventPrincipal const &e,
                                      edm::EventSetup const &eventSetup,
                                      edm::StreamID const &) {
  edm::Handle<std::vector<PCaloHit>> ebHandle;
  e.getByLabel(m_hitsProducerTagEB, ebHandle);

#ifdef ecal_time_debug
  std::cout << "[EcalTimeDigiProducer]::Accumulate Hits for BC " << e.bunchCrossing() << std::endl;
#endif
  accumulateCaloHits(ebHandle, e.bunchCrossing());
}

void EcalTimeDigiProducer::finalizeEvent(edm::Event &event, edm::EventSetup const &eventSetup) {
  std::unique_ptr<EcalTimeDigiCollection> barrelResult = std::make_unique<EcalTimeDigiCollection>();

#ifdef ecal_time_debug
  std::cout << "[EcalTimeDigiProducer]::finalizeEvent" << std::endl;
#endif

  // here basically just put everything in the final collections
  m_BarrelDigitizer->run(*barrelResult);

#ifdef ecal_time_debug
  std::cout << "[EcalTimeDigiProducer]::EB Digi size " << barrelResult->size() << std::endl;
#endif

  edm::LogInfo("TimeDigiInfo") << "EB time Digis: " << barrelResult->size();

#ifdef ecal_time_debug
  std::cout << "[EcalTimeDigiProducer]::putting EcalTimeDigiCollection into the event " << std::endl;
#endif

  event.put(std::move(barrelResult), m_EBdigiCollection);
}

void EcalTimeDigiProducer::checkGeometry(const edm::EventSetup &eventSetup) {
  // TODO find a way to avoid doing this every event
  edm::ESHandle<CaloGeometry> hGeometry;
  eventSetup.get<CaloGeometryRecord>().get(hGeometry);

  const CaloGeometry *pGeometry = &*hGeometry;

  if (pGeometry != m_Geometry) {
    m_Geometry = pGeometry;
    updateGeometry();
  }
}

void EcalTimeDigiProducer::updateGeometry() {
  m_BarrelDigitizer->setGeometry(m_Geometry->getSubdetectorGeometry(DetId::Ecal, EcalBarrel));
}
