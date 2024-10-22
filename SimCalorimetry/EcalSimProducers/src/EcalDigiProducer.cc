#include <memory>

#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"
#include "FWCore/Framework/interface/Event.h"
#include "SimCalorimetry/CaloSimAlgos/interface/CaloHitResponse.h"
#include "SimCalorimetry/EcalSimAlgos/interface/APDSimParameters.h"
#include "SimCalorimetry/EcalSimAlgos/interface/ComponentSimParameterMap.h"
#include "SimCalorimetry/EcalSimAlgos/interface/EBHitResponse.h"
#include "SimCalorimetry/EcalSimAlgos/interface/EEHitResponse.h"
#include "SimCalorimetry/EcalSimAlgos/interface/ESElectronicsSim.h"
#include "SimCalorimetry/EcalSimAlgos/interface/ESElectronicsSimFast.h"
#include "SimCalorimetry/EcalSimAlgos/interface/ESHitResponse.h"
#include "SimCalorimetry/EcalSimAlgos/interface/EcalCoder.h"
#include "SimCalorimetry/EcalSimAlgos/interface/EcalElectronicsSim.h"
#include "SimCalorimetry/EcalSimAlgos/interface/EcalSimParameterMap.h"
#include "SimCalorimetry/EcalSimProducers/interface/EcalDigiProducer.h"
//#include "SimCalorimetry/EcalSimAlgos/interface/ESFastTDigitizer.h"
#include "CalibFormats/CaloObjects/interface/CaloSamples.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Utilities/interface/RandomNumberGenerator.h"
#include "FWCore/Utilities/interface/StreamID.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "SimCalorimetry/EcalSimAlgos/interface/ESDigitizer.h"
#include "SimGeneral/MixingModule/interface/PileUpEventPrincipal.h"

#include "Geometry/EcalAlgo/interface/EcalEndcapGeometry.h"

EcalDigiProducer::EcalDigiProducer(const edm::ParameterSet &params,
                                   edm::ProducesCollector producesCollector,
                                   edm::ConsumesCollector &iC)
    : EcalDigiProducer(params, iC) {
  if (m_apdSeparateDigi)
    producesCollector.produces<EBDigiCollection>(m_apdDigiTag);

  if (m_componentSeparateDigi)
    producesCollector.produces<EBDigiCollection>(m_componentDigiTag);

  producesCollector.produces<EBDigiCollection>(m_EBdigiCollection);
  producesCollector.produces<EEDigiCollection>(m_EEdigiCollection);
  producesCollector.produces<ESDigiCollection>(m_ESdigiCollection);
}

// version for Pre-Mixing, for use outside of MixingModule
EcalDigiProducer::EcalDigiProducer(const edm::ParameterSet &params, edm::ConsumesCollector &iC)
    : DigiAccumulatorMixMod(),
      m_APDShape(iC),
      m_ComponentShapes(iC),
      m_EBShape(iC),
      m_EEShape(iC),
      m_ESShape(),
      m_EBdigiCollection(params.getParameter<std::string>("EBdigiCollection")),
      m_EEdigiCollection(params.getParameter<std::string>("EEdigiCollection")),
      m_ESdigiCollection(params.getParameter<std::string>("ESdigiCollection")),
      m_hitsProducerTag(params.getParameter<std::string>("hitsProducer")),
      m_HitsEBToken_(iC.consumes<std::vector<PCaloHit>>(edm::InputTag(m_hitsProducerTag, "EcalHitsEB"))),
      m_HitsEEToken_(iC.consumes<std::vector<PCaloHit>>(edm::InputTag(m_hitsProducerTag, "EcalHitsEE"))),
      m_HitsESToken_(iC.consumes<std::vector<PCaloHit>>(edm::InputTag(m_hitsProducerTag, "EcalHitsES"))),
      m_pedestalsToken(iC.esConsumes()),
      m_icalToken(iC.esConsumes()),
      m_laserToken(iC.esConsumes()),
      m_agcToken(iC.esConsumes()),
      m_grToken(iC.esConsumes()),
      m_geometryToken(iC.esConsumes()),
      m_useLCcorrection(params.getUntrackedParameter<bool>("UseLCcorrection")),
      m_apdSeparateDigi(params.getParameter<bool>("apdSeparateDigi")),
      m_componentSeparateDigi(params.getParameter<bool>("componentSeparateDigi")),

      m_EBs25notCont(params.getParameter<double>("EBs25notContainment")),
      m_EEs25notCont(params.getParameter<double>("EEs25notContainment")),

      m_readoutFrameSize(ecalPh1::sampleSize),
      m_ParameterMap(new EcalSimParameterMap(params.getParameter<double>("simHitToPhotoelectronsBarrel"),
                                             params.getParameter<double>("simHitToPhotoelectronsEndcap"),
                                             params.getParameter<double>("photoelectronsToAnalogBarrel"),
                                             params.getParameter<double>("photoelectronsToAnalogEndcap"),
                                             params.getParameter<double>("samplingFactor"),
                                             params.getParameter<double>("timePhase"),
                                             m_readoutFrameSize,
                                             params.getParameter<int>("binOfMaximum"),
                                             params.getParameter<bool>("doPhotostatistics"),
                                             params.getParameter<bool>("syncPhase"))),

      m_apdDigiTag(params.getParameter<std::string>("apdDigiTag")),
      m_apdParameters(new APDSimParameters(params.getParameter<bool>("apdAddToBarrel"),
                                           m_apdSeparateDigi,
                                           params.getParameter<double>("apdSimToPELow"),
                                           params.getParameter<double>("apdSimToPEHigh"),
                                           params.getParameter<double>("apdTimeOffset"),
                                           params.getParameter<double>("apdTimeOffWidth"),
                                           params.getParameter<bool>("apdDoPEStats"),
                                           m_apdDigiTag,
                                           params.getParameter<std::vector<double>>("apdNonlParms"))),

      m_componentDigiTag(params.getParameter<std::string>("componentDigiTag")),
      m_componentParameters(
          std::make_unique<ComponentSimParameterMap>(params.getParameter<bool>("componentAddToBarrel"),
                                                     m_componentSeparateDigi,
                                                     params.getParameter<double>("simHitToPhotoelectronsBarrel"),
                                                     0,  // endcap parameters not needed
                                                     params.getParameter<double>("photoelectronsToAnalogBarrel"),
                                                     0,
                                                     params.getParameter<double>("samplingFactor"),
                                                     params.getParameter<double>("componentTimePhase"),
                                                     m_readoutFrameSize,
                                                     params.getParameter<int>("binOfMaximum"),
                                                     params.getParameter<bool>("doPhotostatistics"),
                                                     params.getParameter<bool>("syncPhase"))),

      m_APDResponse(!m_apdSeparateDigi ? nullptr
                                       : new EBHitResponse(m_ParameterMap.get(),
                                                           &m_EBShape,
                                                           true,
                                                           false,
                                                           m_apdParameters.get(),
                                                           &m_APDShape,
                                                           m_componentParameters.get(),
                                                           &m_ComponentShapes)),

      m_ComponentResponse(!m_componentSeparateDigi
                              ? nullptr
                              : std::make_unique<EBHitResponse>(
                                    m_ParameterMap.get(),
                                    &m_EBShape,
                                    false,
                                    true,
                                    m_apdParameters.get(),
                                    &m_APDShape,
                                    m_componentParameters.get(),
                                    &m_ComponentShapes)),  // check if that false is correct // TODO HERE JCH

      m_EBResponse(new EBHitResponse(m_ParameterMap.get(),
                                     &m_EBShape,
                                     false,  // barrel
                                     false,  // normal non-component shape based
                                     m_apdParameters.get(),
                                     &m_APDShape,
                                     m_componentParameters.get(),
                                     &m_ComponentShapes)),

      m_EEResponse(new EEHitResponse(m_ParameterMap.get(), &m_EEShape)),
      m_ESResponse(new ESHitResponse(m_ParameterMap.get(), &m_ESShape)),
      m_ESOldResponse(new CaloHitResponse(m_ParameterMap.get(), &m_ESShape)),

      m_addESNoise(params.getParameter<bool>("doESNoise")),
      m_PreMix1(params.getParameter<bool>("EcalPreMixStage1")),
      m_PreMix2(params.getParameter<bool>("EcalPreMixStage2")),

      m_doFastES(params.getParameter<bool>("doFast")),

      m_doEB(params.getParameter<bool>("doEB")),
      m_doEE(params.getParameter<bool>("doEE")),
      m_doES(params.getParameter<bool>("doES")),

      m_ESElectronicsSim(m_doFastES ? nullptr : new ESElectronicsSim(m_addESNoise)),

      m_ESOldDigitizer(m_doFastES ? nullptr
                                  : new ESOldDigitizer(m_ESOldResponse.get(), m_ESElectronicsSim.get(), m_addESNoise)),

      m_ESElectronicsSimFast(!m_doFastES ? nullptr : new ESElectronicsSimFast(m_addESNoise, m_PreMix1)),

      m_ESDigitizer(!m_doFastES ? nullptr
                                : new ESDigitizer(m_ESResponse.get(), m_ESElectronicsSimFast.get(), m_addESNoise)),

      m_APDDigitizer(nullptr),
      m_ComponentDigitizer(nullptr),
      m_BarrelDigitizer(nullptr),
      m_EndcapDigitizer(nullptr),
      m_ElectronicsSim(nullptr),
      m_Coder(nullptr),
      m_APDElectronicsSim(nullptr),
      m_APDCoder(nullptr),
      m_Geometry(nullptr),
      m_EBCorrNoise({{nullptr, nullptr, nullptr}}),
      m_EECorrNoise({{nullptr, nullptr, nullptr}}) {
  // "produces" statements taken care of elsewhere.
  //   if(m_apdSeparateDigi) mixMod.produces<EBDigiCollection>(m_apdDigiTag);
  // mixMod.produces<EBDigiCollection>(m_EBdigiCollection);
  // mixMod.produces<EEDigiCollection>(m_EEdigiCollection);
  // mixMod.produces<ESDigiCollection>(m_ESdigiCollection);
  if (m_doEB)
    iC.consumes<std::vector<PCaloHit>>(edm::InputTag(m_hitsProducerTag, "EcalHitsEB"));
  if (m_doEE)
    iC.consumes<std::vector<PCaloHit>>(edm::InputTag(m_hitsProducerTag, "EcalHitsEE"));
  if (m_doES) {
    iC.consumes<std::vector<PCaloHit>>(edm::InputTag(m_hitsProducerTag, "EcalHitsES"));
    m_esGainToken = iC.esConsumes();
    m_esMIPToGeVToken = iC.esConsumes();
    m_esPedestalsToken = iC.esConsumes();
    m_esMIPsToken = iC.esConsumes();
  }

  const std::vector<double> ebCorMatG12 = params.getParameter<std::vector<double>>("EBCorrNoiseMatrixG12");
  const std::vector<double> eeCorMatG12 = params.getParameter<std::vector<double>>("EECorrNoiseMatrixG12");
  const std::vector<double> ebCorMatG06 = params.getParameter<std::vector<double>>("EBCorrNoiseMatrixG06");
  const std::vector<double> eeCorMatG06 = params.getParameter<std::vector<double>>("EECorrNoiseMatrixG06");
  const std::vector<double> ebCorMatG01 = params.getParameter<std::vector<double>>("EBCorrNoiseMatrixG01");
  const std::vector<double> eeCorMatG01 = params.getParameter<std::vector<double>>("EECorrNoiseMatrixG01");

  const bool applyConstantTerm = params.getParameter<bool>("applyConstantTerm");
  const double rmsConstantTerm = params.getParameter<double>("ConstantTerm");

  const bool addNoise = params.getParameter<bool>("doENoise");
  const bool cosmicsPhase = params.getParameter<bool>("cosmicsPhase");
  const double cosmicsShift = params.getParameter<double>("cosmicsShift");

  //++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

  // further phase for cosmics studies
  if (cosmicsPhase) {
    if (m_doEB)
      m_EBResponse->setPhaseShift(1. + cosmicsShift);
    if (m_doEE)
      m_EEResponse->setPhaseShift(1. + cosmicsShift);
  }

  EcalCorrMatrix ebMatrix[3];
  EcalCorrMatrix eeMatrix[3];

  assert(ebCorMatG12.size() == m_readoutFrameSize);
  assert(eeCorMatG12.size() == m_readoutFrameSize);
  assert(ebCorMatG06.size() == m_readoutFrameSize);
  assert(eeCorMatG06.size() == m_readoutFrameSize);
  assert(ebCorMatG01.size() == m_readoutFrameSize);
  assert(eeCorMatG01.size() == m_readoutFrameSize);

  assert(1.e-7 > fabs(ebCorMatG12[0] - 1.0));
  assert(1.e-7 > fabs(ebCorMatG06[0] - 1.0));
  assert(1.e-7 > fabs(ebCorMatG01[0] - 1.0));
  assert(1.e-7 > fabs(eeCorMatG12[0] - 1.0));
  assert(1.e-7 > fabs(eeCorMatG06[0] - 1.0));
  assert(1.e-7 > fabs(eeCorMatG01[0] - 1.0));

  for (unsigned int row(0); row != m_readoutFrameSize; ++row) {
    assert(0 == row || 1. >= ebCorMatG12[row]);
    assert(0 == row || 1. >= ebCorMatG06[row]);
    assert(0 == row || 1. >= ebCorMatG01[row]);
    assert(0 == row || 1. >= eeCorMatG12[row]);
    assert(0 == row || 1. >= eeCorMatG06[row]);
    assert(0 == row || 1. >= eeCorMatG01[row]);
    for (unsigned int column(0); column <= row; ++column) {
      const unsigned int index(row - column);
      ebMatrix[0](row, column) = ebCorMatG12[index];
      eeMatrix[0](row, column) = eeCorMatG12[index];
      ebMatrix[1](row, column) = ebCorMatG06[index];
      eeMatrix[1](row, column) = eeCorMatG06[index];
      ebMatrix[2](row, column) = ebCorMatG01[index];
      eeMatrix[2](row, column) = eeCorMatG01[index];
    }
  }

  m_EBCorrNoise[0] = std::make_unique<CorrelatedNoisifier<EcalCorrMatrix>>(ebMatrix[0]);
  m_EECorrNoise[0] = std::make_unique<CorrelatedNoisifier<EcalCorrMatrix>>(eeMatrix[0]);
  m_EBCorrNoise[1] = std::make_unique<CorrelatedNoisifier<EcalCorrMatrix>>(ebMatrix[1]);
  m_EECorrNoise[1] = std::make_unique<CorrelatedNoisifier<EcalCorrMatrix>>(eeMatrix[1]);
  m_EBCorrNoise[2] = std::make_unique<CorrelatedNoisifier<EcalCorrMatrix>>(ebMatrix[2]);
  m_EECorrNoise[2] = std::make_unique<CorrelatedNoisifier<EcalCorrMatrix>>(eeMatrix[2]);

  m_Coder = std::make_unique<EcalCoder>(addNoise,
                                        m_PreMix1,
                                        m_EBCorrNoise[0].get(),
                                        m_EECorrNoise[0].get(),
                                        m_EBCorrNoise[1].get(),
                                        m_EECorrNoise[1].get(),
                                        m_EBCorrNoise[2].get(),
                                        m_EECorrNoise[2].get());

  m_ElectronicsSim =
      std::make_unique<EcalElectronicsSim_Ph1>(m_ParameterMap.get(), m_Coder.get(), applyConstantTerm, rmsConstantTerm);

  if (m_apdSeparateDigi) {
    m_APDCoder = std::make_unique<EcalCoder>(false,
                                             m_PreMix1,
                                             m_EBCorrNoise[0].get(),
                                             m_EECorrNoise[0].get(),
                                             m_EBCorrNoise[1].get(),
                                             m_EECorrNoise[1].get(),
                                             m_EBCorrNoise[2].get(),
                                             m_EECorrNoise[2].get());

    m_APDElectronicsSim = std::make_unique<EcalElectronicsSim_Ph1>(
        m_ParameterMap.get(), m_APDCoder.get(), applyConstantTerm, rmsConstantTerm);

    m_APDDigitizer = std::make_unique<EBDigitizer>(m_APDResponse.get(), m_APDElectronicsSim.get(), false);
  }
  if (m_componentSeparateDigi) {
    m_ComponentCoder = std::make_unique<EcalCoder>(addNoise,
                                                   m_PreMix1,
                                                   m_EBCorrNoise[0].get(),
                                                   m_EECorrNoise[0].get(),
                                                   m_EBCorrNoise[1].get(),
                                                   m_EECorrNoise[1].get(),
                                                   m_EBCorrNoise[2].get(),
                                                   m_EECorrNoise[2].get());
    m_ComponentElectronicsSim = std::make_unique<EcalElectronicsSim_Ph1>(
        m_ParameterMap.get(), m_ComponentCoder.get(), applyConstantTerm, rmsConstantTerm);
    m_ComponentDigitizer =
        std::make_unique<EBDigitizer>(m_ComponentResponse.get(), m_ComponentElectronicsSim.get(), addNoise);
  }

  if (m_doEB) {
    m_BarrelDigitizer = std::make_unique<EBDigitizer>(m_EBResponse.get(), m_ElectronicsSim.get(), addNoise);
  }

  if (m_doEE) {
    m_EndcapDigitizer = std::make_unique<EEDigitizer>(m_EEResponse.get(), m_ElectronicsSim.get(), addNoise);
  }
}

EcalDigiProducer::~EcalDigiProducer() {}

void EcalDigiProducer::initializeEvent(edm::Event const &event, edm::EventSetup const &eventSetup) {
  edm::Service<edm::RandomNumberGenerator> rng;
  randomEngine_ = &rng->getEngine(event.streamID());

  checkGeometry(eventSetup);
  checkCalibrations(event, eventSetup);
  if (m_doEB) {
    m_BarrelDigitizer->initializeHits();
    if (m_apdSeparateDigi) {
      m_APDDigitizer->initializeHits();
    }
    if (m_componentSeparateDigi) {
      m_ComponentDigitizer->initializeHits();
    }
  }
  if (m_doEE) {
    m_EndcapDigitizer->initializeHits();
  }
  if (m_doES) {
    if (m_doFastES) {
      m_ESDigitizer->initializeHits();
    } else {
      m_ESOldDigitizer->initializeHits();
    }
  }
}

void EcalDigiProducer::accumulateCaloHits(HitsHandle const &ebHandle,
                                          HitsHandle const &eeHandle,
                                          HitsHandle const &esHandle,
                                          int bunchCrossing) {
  if (m_doEB && ebHandle.isValid()) {
    m_BarrelDigitizer->add(*ebHandle.product(), bunchCrossing, randomEngine_);

    if (m_apdSeparateDigi) {
      m_APDDigitizer->add(*ebHandle.product(), bunchCrossing, randomEngine_);
    }
    if (m_componentSeparateDigi) {
      m_ComponentDigitizer->add(*ebHandle.product(), bunchCrossing, randomEngine_);
    }
  }

  if (m_doEE && eeHandle.isValid()) {
    m_EndcapDigitizer->add(*eeHandle.product(), bunchCrossing, randomEngine_);
  }

  if (m_doES && esHandle.isValid()) {
    if (m_doFastES) {
      m_ESDigitizer->add(*esHandle.product(), bunchCrossing, randomEngine_);
    } else {
      m_ESOldDigitizer->add(*esHandle.product(), bunchCrossing, randomEngine_);
    }
  }
}

void EcalDigiProducer::accumulate(edm::Event const &e, edm::EventSetup const &eventSetup) {
  // Step A: Get Inputs
  const edm::Handle<std::vector<PCaloHit>> &ebHandle = e.getHandle(m_HitsEBToken_);
  if (m_doEB) {
    m_EBShape.setEventSetup(eventSetup);
    m_APDShape.setEventSetup(eventSetup);
    m_ComponentShapes.setEventSetup(eventSetup);
  }

  const edm::Handle<std::vector<PCaloHit>> &eeHandle = e.getHandle(m_HitsEEToken_);
  if (m_doEE) {
    m_EEShape.setEventSetup(eventSetup);
  }

  const edm::Handle<std::vector<PCaloHit>> &esHandle = e.getHandle(m_HitsESToken_);

  accumulateCaloHits(ebHandle, eeHandle, esHandle, 0);
}

void EcalDigiProducer::accumulate(PileUpEventPrincipal const &e,
                                  edm::EventSetup const &eventSetup,
                                  edm::StreamID const &streamID) {
  // Step A: Get Inputs
  edm::Handle<std::vector<PCaloHit>> ebHandle;
  if (m_doEB) {
    edm::InputTag ebTag(m_hitsProducerTag, "EcalHitsEB");
    e.getByLabel(ebTag, ebHandle);
  }

  edm::Handle<std::vector<PCaloHit>> eeHandle;
  if (m_doEE) {
    edm::InputTag eeTag(m_hitsProducerTag, "EcalHitsEE");
    e.getByLabel(eeTag, eeHandle);
  }

  edm::Handle<std::vector<PCaloHit>> esHandle;
  if (m_doES) {
    edm::InputTag esTag(m_hitsProducerTag, "EcalHitsES");
    e.getByLabel(esTag, esHandle);
  }

  accumulateCaloHits(ebHandle, eeHandle, esHandle, e.bunchCrossing());
}

void EcalDigiProducer::finalizeEvent(edm::Event &event, edm::EventSetup const &eventSetup) {
  // Step B: Create empty output
  std::unique_ptr<EBDigiCollection> apdResult(!m_apdSeparateDigi || !m_doEB ? nullptr : new EBDigiCollection());
  std::unique_ptr<EBDigiCollection> componentResult(!m_componentSeparateDigi || !m_doEB ? nullptr
                                                                                        : new EBDigiCollection());
  std::unique_ptr<EBDigiCollection> barrelResult(new EBDigiCollection());
  std::unique_ptr<EEDigiCollection> endcapResult(new EEDigiCollection());
  std::unique_ptr<ESDigiCollection> preshowerResult(new ESDigiCollection());

  // run the algorithm

  if (m_doEB) {
    m_BarrelDigitizer->run(*barrelResult, randomEngine_);
    cacheEBDigis(&*barrelResult);

    edm::LogInfo("DigiInfo") << "EB Digis: " << barrelResult->size();

    if (m_apdSeparateDigi) {
      m_APDDigitizer->run(*apdResult, randomEngine_);
      edm::LogInfo("DigiInfo") << "APD Digis: " << apdResult->size();
    }
    if (m_componentSeparateDigi) {
      m_ComponentDigitizer->run(*componentResult, randomEngine_);
      edm::LogInfo("DigiInfo") << "Component Digis: " << componentResult->size();
    }
  }

  if (m_doEE) {
    m_EndcapDigitizer->run(*endcapResult, randomEngine_);
    edm::LogInfo("EcalDigi") << "EE Digis: " << endcapResult->size();
    cacheEEDigis(&*endcapResult);
  }
  if (m_doES) {
    if (m_doFastES) {
      m_ESDigitizer->run(*preshowerResult, randomEngine_);
    } else {
      m_ESOldDigitizer->run(*preshowerResult, randomEngine_);
    }
    edm::LogInfo("EcalDigi") << "ES Digis: " << preshowerResult->size();
  }

  // Step D: Put outputs into event
  if (m_apdSeparateDigi) {
    // event.put(std::move(apdResult),    m_apdDigiTag         ) ;
  }

  event.put(std::move(barrelResult), m_EBdigiCollection);
  if (m_componentSeparateDigi) {
    event.put(std::move(componentResult), m_componentDigiTag);
  }
  event.put(std::move(endcapResult), m_EEdigiCollection);
  event.put(std::move(preshowerResult), m_ESdigiCollection);

  randomEngine_ = nullptr;  // to prevent access outside event
}

void EcalDigiProducer::beginLuminosityBlock(edm::LuminosityBlock const &lumi, edm::EventSetup const &setup) {
  edm::Service<edm::RandomNumberGenerator> rng;
  if (!rng.isAvailable()) {
    throw cms::Exception("Configuration") << "RandomNumberGenerator service is not available.\n"
                                             "You must add the service in the configuration file\n"
                                             "or remove the module that requires it.";
  }
  CLHEP::HepRandomEngine *engine = &rng->getEngine(lumi.index());

  if (m_doEB) {
    if (nullptr != m_APDResponse)
      m_APDResponse->initialize(engine);
    if (nullptr != m_ComponentResponse)
      m_ComponentResponse->initialize(engine);
    m_EBResponse->initialize(engine);
  }
}

void EcalDigiProducer::checkCalibrations(const edm::Event &event, const edm::EventSetup &eventSetup) {
  // Pedestals from event setup

  const EcalPedestals *pedestals = &eventSetup.getData(m_pedestalsToken);

  m_Coder->setPedestals(pedestals);
  if (nullptr != m_APDCoder)
    m_APDCoder->setPedestals(pedestals);
  if (nullptr != m_ComponentCoder)
    m_ComponentCoder->setPedestals(pedestals);

  // Ecal Intercalibration Constants
  const EcalIntercalibConstantsMC *ical = &eventSetup.getData(m_icalToken);

  m_Coder->setIntercalibConstants(ical);
  if (nullptr != m_APDCoder)
    m_APDCoder->setIntercalibConstants(ical);
  if (nullptr != m_ComponentCoder)
    m_ComponentCoder->setIntercalibConstants(ical);

  m_EBResponse->setIntercal(ical);
  if (nullptr != m_APDResponse)
    m_APDResponse->setIntercal(ical);
  if (nullptr != m_ComponentResponse)
    m_ComponentResponse->setIntercal(ical);

  // Ecal LaserCorrection Constants
  const EcalLaserDbService *laser = &eventSetup.getData(m_laserToken);
  const edm::TimeValue_t eventTimeValue = event.time().value();

  m_EBResponse->setEventTime(eventTimeValue);
  m_EBResponse->setLaserConstants(laser, m_useLCcorrection);

  m_EEResponse->setEventTime(eventTimeValue);
  m_EEResponse->setLaserConstants(laser, m_useLCcorrection);

  // ADC -> GeV Scale
  const EcalADCToGeVConstant *agc = &eventSetup.getData(m_agcToken);

  // Gain Ratios
  const EcalGainRatios *gr = &eventSetup.getData(m_grToken);

  m_Coder->setGainRatios(gr);
  if (nullptr != m_APDCoder)
    m_APDCoder->setGainRatios(gr);
  if (nullptr != m_ComponentCoder)
    m_ComponentCoder->setGainRatios(gr);

  EcalMGPAGainRatio *defaultRatios = new EcalMGPAGainRatio();

  double theGains[m_Coder->NGAINS + 1];
  theGains[0] = 0.;
  theGains[3] = 1.;
  theGains[2] = defaultRatios->gain6Over1();
  theGains[1] = theGains[2] * (defaultRatios->gain12Over6());

  LogDebug("EcalDigi") << " Gains: "
                       << "\n"
                       << " g1 = " << theGains[1] << "\n"
                       << " g2 = " << theGains[2] << "\n"
                       << " g3 = " << theGains[3];

  delete defaultRatios;

  const double EBscale((agc->getEBValue()) * theGains[1] * (m_Coder->MAXADC) * m_EBs25notCont);

  LogDebug("EcalDigi") << " GeV/ADC = " << agc->getEBValue() << "\n"
                       << " notCont = " << m_EBs25notCont << "\n"
                       << " saturation for EB = " << EBscale << ", " << m_EBs25notCont;

  const double EEscale((agc->getEEValue()) * theGains[1] * (m_Coder->MAXADC) * m_EEs25notCont);

  LogDebug("EcalDigi") << " GeV/ADC = " << agc->getEEValue() << "\n"
                       << " notCont = " << m_EEs25notCont << "\n"
                       << " saturation for EB = " << EEscale << ", " << m_EEs25notCont;

  m_Coder->setFullScaleEnergy(EBscale, EEscale);
  if (nullptr != m_APDCoder)
    m_APDCoder->setFullScaleEnergy(EBscale, EEscale);
  if (nullptr != m_ComponentCoder)
    m_ComponentCoder->setFullScaleEnergy(EBscale, EEscale);

  if (m_doES) {
    // ES condition objects
    const ESGain *esgain = &eventSetup.getData(m_esGainToken);
    const ESPedestals *espeds = &eventSetup.getData(m_esPedestalsToken);
    const ESIntercalibConstants *esmips = &eventSetup.getData(m_esMIPsToken);
    const ESMIPToGeVConstant *esMipToGeV = &eventSetup.getData(m_esMIPToGeVToken);
    const int ESGain(1.1 > esgain->getESGain() ? 1 : 2);
    const double ESMIPToGeV((1 == ESGain) ? esMipToGeV->getESValueLow() : esMipToGeV->getESValueHigh());

    m_ESShape.setGain(ESGain);
    if (!m_doFastES) {
      m_ESElectronicsSim->setGain(ESGain);
      m_ESElectronicsSim->setPedestals(espeds);
      m_ESElectronicsSim->setMIPs(esmips);
      m_ESElectronicsSim->setMIPToGeV(ESMIPToGeV);
    } else {
      m_ESDigitizer->setGain(ESGain);
      m_ESElectronicsSimFast->setPedestals(espeds);
      m_ESElectronicsSimFast->setMIPs(esmips);
      m_ESElectronicsSimFast->setMIPToGeV(ESMIPToGeV);
    }
  }
}

void EcalDigiProducer::checkGeometry(const edm::EventSetup &eventSetup) {
  if (m_geometryWatcher.check(eventSetup)) {
    m_Geometry = &eventSetup.getData(m_geometryToken);
    updateGeometry();
  }
}

void EcalDigiProducer::updateGeometry() {
  if (m_doEB) {
    if (nullptr != m_APDResponse)
      m_APDResponse->setGeometry(m_Geometry->getSubdetectorGeometry(DetId::Ecal, EcalBarrel));
    if (nullptr != m_ComponentResponse)
      m_ComponentResponse->setGeometry(m_Geometry->getSubdetectorGeometry(DetId::Ecal, EcalBarrel));
    m_EBResponse->setGeometry(m_Geometry->getSubdetectorGeometry(DetId::Ecal, EcalBarrel));
  }
  if (m_doEE) {
    m_EEResponse->setGeometry(m_Geometry->getSubdetectorGeometry(DetId::Ecal, EcalEndcap));
  }
  if (m_doES) {
    m_ESResponse->setGeometry(m_Geometry->getSubdetectorGeometry(DetId::Ecal, EcalPreshower));
    m_ESOldResponse->setGeometry(m_Geometry);

    const std::vector<DetId> *theESDets(
        nullptr != m_Geometry->getSubdetectorGeometry(DetId::Ecal, EcalPreshower)
            ? &m_Geometry->getSubdetectorGeometry(DetId::Ecal, EcalPreshower)->getValidDetIds()
            : nullptr);

    if (!m_doFastES) {
      if (nullptr != m_ESOldDigitizer && nullptr != theESDets)
        m_ESOldDigitizer->setDetIds(*theESDets);
    } else {
      if (nullptr != m_ESDigitizer && nullptr != theESDets)
        m_ESDigitizer->setDetIds(*theESDets);
    }
  }
}

void EcalDigiProducer::setEBNoiseSignalGenerator(EcalBaseSignalGenerator *noiseGenerator) {
  // noiseGenerator->setParameterMap(theParameterMap);
  if (nullptr != m_BarrelDigitizer)
    m_BarrelDigitizer->setNoiseSignalGenerator(noiseGenerator);
}

void EcalDigiProducer::setEENoiseSignalGenerator(EcalBaseSignalGenerator *noiseGenerator) {
  // noiseGenerator->setParameterMap(theParameterMap);
  if (nullptr != m_EndcapDigitizer)
    m_EndcapDigitizer->setNoiseSignalGenerator(noiseGenerator);
}

void EcalDigiProducer::setESNoiseSignalGenerator(EcalBaseSignalGenerator *noiseGenerator) {
  // noiseGenerator->setParameterMap(theParameterMap);
  if (nullptr != m_ESDigitizer)
    m_ESDigitizer->setNoiseSignalGenerator(noiseGenerator);
}
