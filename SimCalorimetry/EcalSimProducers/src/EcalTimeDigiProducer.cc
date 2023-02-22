#include "FWCore/Framework/interface/Event.h"
#include "SimCalorimetry/EcalSimAlgos/interface/EcalTimeMapDigitizer.h"
#include "SimCalorimetry/EcalSimProducers/interface/EcalTimeDigiProducer.h"
#include "CalibFormats/CaloObjects/interface/CaloSamples.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "SimGeneral/MixingModule/interface/PileUpEventPrincipal.h"

//#define EDM_ML_DEBUG

EcalTimeDigiProducer::EcalTimeDigiProducer(const edm::ParameterSet &params,
                                           edm::ProducesCollector producesCollector,
                                           edm::ConsumesCollector &sumes)
    : DigiAccumulatorMixMod(),
      m_EBdigiCollection(params.getParameter<std::string>("EBtimeDigiCollection")),
      m_hitsProducerTagEB(params.getParameter<edm::InputTag>("hitsProducerEB")),
      m_hitsProducerTokenEB(sumes.consumes<std::vector<PCaloHit>>(m_hitsProducerTagEB)),
      m_geometryToken(sumes.esConsumes()),
      m_timeLayerEB(params.getParameter<int>("timeLayerBarrel")),
      m_Geometry(nullptr),
      m_componentWaveform(params.getParameter<bool>("componentWaveform")) {
  producesCollector.produces<EcalTimeDigiCollection>(m_EBdigiCollection);

  if (m_componentWaveform)
    m_ComponentShapes = new ComponentShapeCollection(sumes);
  m_BarrelDigitizer = new EcalTimeMapDigitizer(EcalBarrel, m_ComponentShapes);

#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("TimeDigiInfo") << "[EcalTimeDigiProducer]::Create EB " << m_EBdigiCollection
                                   << "  collection and digitizer";
#endif

  m_BarrelDigitizer->setTimeLayerId(m_timeLayerEB);
}

EcalTimeDigiProducer::~EcalTimeDigiProducer() {}

void EcalTimeDigiProducer::initializeEvent(edm::Event const &event, edm::EventSetup const &eventSetup) {
  checkGeometry(eventSetup);
  //    checkCalibrations( event, eventSetup );
  // here the methods to clean the maps
  m_BarrelDigitizer->initializeMap();
  if (m_componentWaveform) {
    m_ComponentShapes->setEventSetup(eventSetup);
    m_BarrelDigitizer->setEventSetup(eventSetup);
  }
#ifdef EDM_ML_DEBUG
  if (m_componentWaveform)
    m_ComponentShapes->test();
#endif
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
  const edm::Handle<std::vector<PCaloHit>> &ebHandle = e.getHandle(m_hitsProducerTokenEB);

#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("TimeDigiInfo") << "[EcalTimeDigiProducer]::Accumulate Hits HS  event";
#endif

  if (m_componentWaveform) {
    m_BarrelDigitizer->setEventSetup(eventSetup);
  }
  accumulateCaloHits(ebHandle, 0);
}

void EcalTimeDigiProducer::accumulate(PileUpEventPrincipal const &e,
                                      edm::EventSetup const &eventSetup,
                                      edm::StreamID const &) {
  edm::Handle<std::vector<PCaloHit>> ebHandle;
  e.getByLabel(m_hitsProducerTagEB, ebHandle);

#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("TimeDigiInfo") << "[EcalTimeDigiProducer]::Accumulate Hits for BC " << e.bunchCrossing();
#endif
  if (m_componentWaveform) {
    m_BarrelDigitizer->setEventSetup(eventSetup);
  }
  accumulateCaloHits(ebHandle, e.bunchCrossing());
}

void EcalTimeDigiProducer::finalizeEvent(edm::Event &event, edm::EventSetup const &eventSetup) {
  std::unique_ptr<EcalTimeDigiCollection> barrelResult = std::make_unique<EcalTimeDigiCollection>();

#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("TimeDigiInfo") << "[EcalTimeDigiProducer]::finalizeEvent";
#endif

  // here basically just put everything in the final collections
  m_BarrelDigitizer->run(*barrelResult);

#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("TimeDigiInfo") << "[EcalTimeDigiProducer]::EB Digi size " << barrelResult->size();
#endif

  edm::LogInfo("TimeDigiInfo") << "EB time Digis: " << barrelResult->size();

#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("TimeDigiInfo") << "[EcalTimeDigiProducer]::putting EcalTimeDigiCollection into the event ";
#endif

  event.put(std::move(barrelResult), m_EBdigiCollection);
}

void EcalTimeDigiProducer::checkGeometry(const edm::EventSetup &eventSetup) {
  if (m_geometryWatcher.check(eventSetup)) {
    m_Geometry = &eventSetup.getData(m_geometryToken);
    updateGeometry();
  }
}

void EcalTimeDigiProducer::updateGeometry() {
  m_BarrelDigitizer->setGeometry(m_Geometry->getSubdetectorGeometry(DetId::Ecal, EcalBarrel));
}
