#include "FWCore/Framework/interface/Event.h"
#include "SimCalorimetry/EcalSimProducers/interface/EcalDigiProducer_Ph2.h"
#include "SimCalorimetry/EcalSimAlgos/interface/EBHitResponse.h"
#include "SimCalorimetry/CaloSimAlgos/interface/CaloHitResponse.h"
#include "SimCalorimetry/EcalSimAlgos/interface/EcalSimParameterMap.h"
#include "SimCalorimetry/EcalSimAlgos/interface/APDSimParameters.h"
#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"
#include "SimCalorimetry/EcalSimAlgos/interface/EcalLiteDTUCoder.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "CalibFormats/CaloObjects/interface/CaloSamples.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Framework/interface/ProducerBase.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Utilities/interface/RandomNumberGenerator.h"
#include "FWCore/Utilities/interface/StreamID.h"
#include "SimGeneral/MixingModule/interface/PileUpEventPrincipal.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "CondFormats/EcalObjects/interface/EcalLiteDTUPedestals.h"
#include "CondFormats/DataRecord/interface/EcalLiteDTUPedestalsRcd.h"
#include "CondFormats/EcalObjects/interface/EcalIntercalibConstantsMC.h"
#include "CondFormats/DataRecord/interface/EcalIntercalibConstantsMCRcd.h"
#include "CalibCalorimetry/EcalLaserCorrection/interface/EcalLaserDbService.h"
#include "CalibCalorimetry/EcalLaserCorrection/interface/EcalLaserDbRecord.h"
#include "CondFormats/EcalObjects/interface/EcalADCToGeVConstant.h"
#include "CondFormats/DataRecord/interface/EcalADCToGeVConstantRcd.h"

//*****************************************//
//Ecal Digi Producer for PhaseII data format
//Removed EE and ES
//Moved to EBDigiCollectionPh2
//Moved to 2 Gains instead of 3, and from 10 to 16 ecal digi samples
//This producer calls the EcalLiteDTUCoder, the PhaseII noise matrices and the EcalLiteDTUPedestals
//*****************************************//
EcalDigiProducer_Ph2::EcalDigiProducer_Ph2(const edm::ParameterSet& params,
                                           edm::ProducesCollector producesCollector,
                                           edm::ConsumesCollector& iC)
    : EcalDigiProducer_Ph2(params, iC) {
  if (m_apdSeparateDigi)
    producesCollector.produces<EBDigiCollectionPh2>(m_apdDigiTag);

  producesCollector.produces<EBDigiCollectionPh2>(m_EBdigiCollection);
}

// version for Pre-Mixing, for use outside of MixingModule
EcalDigiProducer_Ph2::EcalDigiProducer_Ph2(const edm::ParameterSet& params, edm::ConsumesCollector& iC)
    : DigiAccumulatorMixMod(),
      m_APDShape(iC),
      m_EBShape(iC),

      m_EBdigiCollection(params.getParameter<std::string>("EBdigiCollectionPh2")),

      m_hitsProducerTag(params.getParameter<std::string>("hitsProducer")),
      m_useLCcorrection(params.getUntrackedParameter<bool>("UseLCcorrection")),
      m_apdSeparateDigi(params.getParameter<bool>("apdSeparateDigi")),

      m_EBs25notCont(params.getParameter<double>("EBs25notContainment")),

      m_readoutFrameSize(ecalPh2::sampleSize),

      m_ParameterMap(std::make_unique<EcalSimParameterMap>(params.getParameter<double>("simHitToPhotoelectronsBarrel"),
                                                           0,  // endcap parameters not needed
                                                           params.getParameter<double>("photoelectronsToAnalogBarrel"),
                                                           0,
                                                           params.getParameter<double>("samplingFactor"),
                                                           params.getParameter<double>("timePhase"),
                                                           m_readoutFrameSize,
                                                           params.getParameter<int>("binOfMaximum"),
                                                           params.getParameter<bool>("doPhotostatistics"),
                                                           params.getParameter<bool>("syncPhase"))),

      m_apdDigiTag(params.getParameter<std::string>("apdDigiTag")),
      m_apdParameters(std::make_unique<APDSimParameters>(params.getParameter<bool>("apdAddToBarrel"),
                                                         m_apdSeparateDigi,
                                                         params.getParameter<double>("apdSimToPELow"),
                                                         params.getParameter<double>("apdSimToPEHigh"),
                                                         params.getParameter<double>("apdTimeOffset"),
                                                         params.getParameter<double>("apdTimeOffWidth"),
                                                         params.getParameter<bool>("apdDoPEStats"),
                                                         m_apdDigiTag,
                                                         params.getParameter<std::vector<double>>("apdNonlParms"))),

      m_APDResponse(!m_apdSeparateDigi
                        ? nullptr
                        : std::make_unique<EBHitResponse_Ph2>(
                              m_ParameterMap.get(), &m_EBShape, true, m_apdParameters.get(), &m_APDShape)),

      m_EBResponse(std::make_unique<EBHitResponse_Ph2>(m_ParameterMap.get(),
                                                       &m_EBShape,
                                                       false,  // barrel
                                                       m_apdParameters.get(),
                                                       &m_APDShape)),

      m_PreMix1(params.getParameter<bool>("EcalPreMixStage1")),
      m_PreMix2(params.getParameter<bool>("EcalPreMixStage2")),
      m_HitsEBToken(iC.consumes<std::vector<PCaloHit>>(edm::InputTag(m_hitsProducerTag, "EcalHitsEB"))),

      m_APDDigitizer(nullptr),
      m_BarrelDigitizer(nullptr),
      m_ElectronicsSim(nullptr),
      m_Coder(nullptr),
      m_APDElectronicsSim(nullptr),
      m_APDCoder(nullptr),
      m_Geometry(nullptr),
      m_EBCorrNoise({{nullptr, nullptr}})

{
  iC.consumes<std::vector<PCaloHit>>(edm::InputTag(m_hitsProducerTag, "EcalHitsEB"));
  pedestalToken_ = iC.esConsumes();
  laserToken_ = iC.esConsumes<EcalLaserDbService, EcalLaserDbRecord>();
  agcToken_ = iC.esConsumes<EcalADCToGeVConstant, EcalADCToGeVConstantRcd>();
  icalToken_ = iC.esConsumes<EcalIntercalibConstants, EcalIntercalibConstantsRcd>();
  geom_token_ = iC.esConsumes<CaloGeometry, CaloGeometryRecord>();

  const std::vector<double> ebCorMatG10Ph2 = params.getParameter<std::vector<double>>("EBCorrNoiseMatrixG10Ph2");
  const std::vector<double> ebCorMatG01Ph2 = params.getParameter<std::vector<double>>("EBCorrNoiseMatrixG01Ph2");

  const bool applyConstantTerm = params.getParameter<bool>("applyConstantTerm");
  const double rmsConstantTerm = params.getParameter<double>("ConstantTerm");

  const bool addNoise = params.getParameter<bool>("doENoise");
  const bool cosmicsPhase = params.getParameter<bool>("cosmicsPhase");
  const double cosmicsShift = params.getParameter<double>("cosmicsShift");

  // further phase for cosmics studies
  if (cosmicsPhase) {
    m_EBResponse->setPhaseShift(1. + cosmicsShift);
  }

  EcalCorrMatrix_Ph2 ebMatrix[2];
  const double errorCorrelation = 1.e-7;
  assert(ebCorMatG10Ph2.size() == m_readoutFrameSize);
  assert(ebCorMatG01Ph2.size() == m_readoutFrameSize);

  assert(errorCorrelation > std::abs(ebCorMatG10Ph2[0] - 1.0));
  assert(errorCorrelation > std::abs(ebCorMatG01Ph2[0] - 1.0));

  for (unsigned int row(0); row != m_readoutFrameSize; ++row) {
    assert(0 == row || 1. >= ebCorMatG10Ph2[row]);
    assert(0 == row || 1. >= ebCorMatG01Ph2[row]);

    for (unsigned int column(0); column <= row; ++column) {
      const unsigned int index(row - column);
      ebMatrix[0](row, column) = ebCorMatG10Ph2[index];
      ebMatrix[1](row, column) = ebCorMatG01Ph2[index];
    }
  }
  m_EBCorrNoise[0] = std::make_unique<CorrelatedNoisifier<EcalCorrMatrix_Ph2>>(ebMatrix[0]);
  m_EBCorrNoise[1] = std::make_unique<CorrelatedNoisifier<EcalCorrMatrix_Ph2>>(ebMatrix[1]);
  m_Coder = std::make_unique<EcalLiteDTUCoder>(addNoise, m_PreMix1, m_EBCorrNoise[0].get(), m_EBCorrNoise[1].get());
  m_ElectronicsSim =
      std::make_unique<EcalElectronicsSim_Ph2>(m_ParameterMap.get(), m_Coder.get(), applyConstantTerm, rmsConstantTerm);

  if (m_apdSeparateDigi) {
    m_APDCoder = std::make_unique<EcalLiteDTUCoder>(false, m_PreMix1, m_EBCorrNoise[0].get(), m_EBCorrNoise[1].get());

    m_APDElectronicsSim = std::make_unique<EcalElectronicsSim_Ph2>(
        m_ParameterMap.get(), m_APDCoder.get(), applyConstantTerm, rmsConstantTerm);

    m_APDDigitizer = std::make_unique<EBDigitizer_Ph2>(m_APDResponse.get(), m_APDElectronicsSim.get(), false);
  }

  m_BarrelDigitizer = std::make_unique<EBDigitizer_Ph2>(m_EBResponse.get(), m_ElectronicsSim.get(), addNoise);
}

EcalDigiProducer_Ph2::~EcalDigiProducer_Ph2() {}

void EcalDigiProducer_Ph2::initializeEvent(edm::Event const& event, edm::EventSetup const& eventSetup) {
  edm::Service<edm::RandomNumberGenerator> rng;
  randomEngine_ = &rng->getEngine(event.streamID());

  checkGeometry(eventSetup);
  checkCalibrations(event, eventSetup);

  m_BarrelDigitizer->initializeHits();
  if (m_apdSeparateDigi) {
    m_APDDigitizer->initializeHits();
  }
}

void EcalDigiProducer_Ph2::accumulateCaloHits(HitsHandle const& ebHandle, int bunchCrossing) {
  if (ebHandle.isValid()) {
    m_BarrelDigitizer->add(*ebHandle.product(), bunchCrossing, randomEngine_);

    if (m_apdSeparateDigi) {
      m_APDDigitizer->add(*ebHandle.product(), bunchCrossing, randomEngine_);
    }
  }
}

void EcalDigiProducer_Ph2::accumulate(edm::Event const& e, edm::EventSetup const& eventSetup) {
  // Step A: Get Inputs

  m_EBShape.setEventSetup(eventSetup);
  m_APDShape.setEventSetup(eventSetup);
  const edm::Handle<std::vector<PCaloHit>>& ebHandle = e.getHandle(m_HitsEBToken);

  accumulateCaloHits(ebHandle, 0);
}

void EcalDigiProducer_Ph2::accumulate(PileUpEventPrincipal const& e,
                                      edm::EventSetup const& eventSetup,
                                      edm::StreamID const& streamID) {
  // Step A: Get Inputs
  edm::Handle<std::vector<PCaloHit>> ebHandle;

  edm::InputTag ebTag(m_hitsProducerTag, "EcalHitsEB");
  e.getByLabel(ebTag, ebHandle);

  accumulateCaloHits(ebHandle, e.bunchCrossing());
}

void EcalDigiProducer_Ph2::finalizeEvent(edm::Event& event, edm::EventSetup const& eventSetup) {
  // Step B: Create empty output
  std::unique_ptr<EBDigiCollectionPh2> apdResult(nullptr);
  std::unique_ptr<EBDigiCollectionPh2> barrelResult = std::make_unique<EBDigiCollectionPh2>();
  if (m_apdSeparateDigi) {
    apdResult = std::make_unique<EBDigiCollectionPh2>();
  }
  // run the algorithm

  m_BarrelDigitizer->run(*barrelResult, randomEngine_);
  cacheEBDigis(&*barrelResult);

  edm::LogInfo("DigiInfo") << "EB Digis: " << barrelResult->size();

  if (m_apdSeparateDigi) {
    m_APDDigitizer->run(*apdResult, randomEngine_);
    edm::LogInfo("DigiInfo") << "APD Digis: " << apdResult->size();
  }

  // Step D: Put outputs into event

  event.put(std::move(barrelResult), m_EBdigiCollection);

  randomEngine_ = nullptr;  // to prevent access outside event
}

void EcalDigiProducer_Ph2::beginLuminosityBlock(edm::LuminosityBlock const& lumi, edm::EventSetup const& setup) {
  edm::Service<edm::RandomNumberGenerator> rng;
  if (!rng.isAvailable()) {
    throw cms::Exception("Configuration") << "RandomNumberGenerator service is not available.\n"
                                             "You must add the service in the configuration file\n"
                                             "or remove the module that requires it.";
  }
  CLHEP::HepRandomEngine* engine = &rng->getEngine(lumi.index());

  if (nullptr != m_APDResponse)
    m_APDResponse->initialize(engine);
  m_EBResponse->initialize(engine);
}

void EcalDigiProducer_Ph2::checkCalibrations(const edm::Event& event, const edm::EventSetup& eventSetup) {
  // Pedestals from event setup
  auto pedestals = &eventSetup.getData(pedestalToken_);

  m_Coder->setPedestals(pedestals);
  if (nullptr != m_APDCoder)
    m_APDCoder->setPedestals(pedestals);

  // Ecal Intercalibration Constants
  auto ical = &eventSetup.getData(icalToken_);

  m_Coder->setIntercalibConstants(ical);
  if (nullptr != m_APDCoder)
    m_APDCoder->setIntercalibConstants(ical);

  m_EBResponse->setIntercal(ical);
  if (nullptr != m_APDResponse)
    m_APDResponse->setIntercal(ical);

  // Ecal LaserCorrection Constants
  auto laser = &eventSetup.getData(laserToken_);

  const edm::TimeValue_t eventTimeValue = event.time().value();

  m_EBResponse->setEventTime(eventTimeValue);
  m_EBResponse->setLaserConstants(laser, m_useLCcorrection);

  // ADC -> GeV Scale
  auto agc = &eventSetup.getData(agcToken_);

  m_Coder->setGainRatios(ecalPh2::gains[0] / ecalPh2::gains[1]);
  if (nullptr != m_APDCoder)
    m_APDCoder->setGainRatios(ecalPh2::gains[0] / ecalPh2::gains[1]);

  const double EBscale((agc->getEBValue()) * ecalPh2::gains[1] * (ecalPh2::MAXADC)*m_EBs25notCont);

  LogDebug("EcalDigi") << " GeV/ADC = " << agc->getEBValue() << "\n"
                       << " notCont = " << m_EBs25notCont << "\n"
                       << " saturation for EB = " << EBscale << ", " << m_EBs25notCont;

  m_Coder->setFullScaleEnergy(EBscale);
  if (nullptr != m_APDCoder)
    m_APDCoder->setFullScaleEnergy(EBscale);
}

void EcalDigiProducer_Ph2::checkGeometry(const edm::EventSetup& eventSetup) {
  // TODO find a way to avoid doing this every event
  edm::ESHandle<CaloGeometry> hGeometry = eventSetup.getHandle(geom_token_);
  const CaloGeometry* pGeometry = &*hGeometry;

  if (pGeometry != m_Geometry) {
    m_Geometry = pGeometry;
    updateGeometry();
  }
}

void EcalDigiProducer_Ph2::updateGeometry() {
  if (nullptr != m_APDResponse)
    m_APDResponse->setGeometry(m_Geometry->getSubdetectorGeometry(DetId::Ecal, EcalBarrel));
  m_EBResponse->setGeometry(m_Geometry->getSubdetectorGeometry(DetId::Ecal, EcalBarrel));
}

void EcalDigiProducer_Ph2::setEBNoiseSignalGenerator(EcalBaseSignalGenerator* noiseGenerator) {
  m_BarrelDigitizer->setNoiseSignalGenerator(noiseGenerator);
}
