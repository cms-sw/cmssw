#include "SimCalorimetry/HcalSimProducers/interface/HcalDigitizer.h"
#include "CalibFormats/HcalObjects/interface/HcalDbService.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/Wrapper.h"
#include "DataFormats/HcalDetId/interface/HcalSubdetector.h"
#include "DataFormats/HcalDetId/interface/HcalTestNumbering.h"
#include "DataFormats/HcalDetId/interface/HcalZDCDetId.h"
#include "DataFormats/HcalDigi/interface/HcalDigiCollections.h"
#include "DataFormats/HcalDigi/interface/HcalQIENum.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "Geometry/CaloTopology/interface/HcalTopology.h"
#include "SimCalorimetry/CaloSimAlgos/interface/CaloHitResponse.h"
#include "SimCalorimetry/CaloSimAlgos/interface/CaloTDigitizer.h"
#include "SimCalorimetry/HcalSimAlgos/interface/HPDIonFeedbackSim.h"
#include "SimCalorimetry/HcalSimAlgos/interface/HcalAmplifier.h"
#include "SimCalorimetry/HcalSimAlgos/interface/HcalBaseSignalGenerator.h"
#include "SimCalorimetry/HcalSimAlgos/interface/HcalCoderFactory.h"
#include "SimCalorimetry/HcalSimAlgos/interface/HcalElectronicsSim.h"
#include "SimCalorimetry/HcalSimAlgos/interface/HcalSiPMHitResponse.h"
#include "SimCalorimetry/HcalSimAlgos/interface/HcalSimParameterMap.h"
#include "SimCalorimetry/HcalSimAlgos/interface/HcalTimeSlewSim.h"
#include "SimDataFormats/CrossingFrame/interface/CrossingFrame.h"
#include "SimDataFormats/CrossingFrame/interface/MixCollection.h"
#include "SimGeneral/MixingModule/interface/PileUpEventPrincipal.h"
#include <boost/foreach.hpp>
#include <memory>

//#define EDM_ML_DEBUG

HcalDigitizer::HcalDigitizer(const edm::ParameterSet &ps, edm::ConsumesCollector &iC)
    : conditionsToken_(iC.esConsumes()),
      topoToken_(iC.esConsumes()),
      hcalTimeSlew_delay_token_(iC.esConsumes(edm::ESInputTag("", "HBHE"))),
      theGeometryToken(iC.esConsumes()),
      theRecNumberToken(iC.esConsumes()),
      qieTypesToken_(iC.esConsumes()),
      theGeometry(nullptr),
      theRecNumber(nullptr),
      theParameterMap(ps),
      theShapes(),
      theHBHEResponse(std::make_unique<CaloHitResponse>(&theParameterMap, &theShapes)),
      theHBHESiPMResponse(std::make_unique<HcalSiPMHitResponse>(
          &theParameterMap, &theShapes, ps.getParameter<bool>("HcalPreMixStage1"), true)),
      theHOResponse(std::make_unique<CaloHitResponse>(&theParameterMap, &theShapes)),
      theHOSiPMResponse(std::make_unique<HcalSiPMHitResponse>(
          &theParameterMap, &theShapes, ps.getParameter<bool>("HcalPreMixStage1"), false)),
      theHFResponse(std::make_unique<CaloHitResponse>(&theParameterMap, &theShapes)),
      theHFQIE10Response(std::make_unique<CaloHitResponse>(
          &theParameterMap, &theShapes, ps.getParameter<bool>("HcalPreMixStage1"), true)),
      theZDCResponse(std::make_unique<CaloHitResponse>(
          &theParameterMap, &theShapes, ps.getParameter<bool>("HcalPreMixStage1"), false)),
      theHBHEAmplifier(nullptr),
      theHFAmplifier(nullptr),
      theHOAmplifier(nullptr),
      theZDCAmplifier(nullptr),
      theHFQIE10Amplifier(nullptr),
      theHBHEQIE11Amplifier(nullptr),
      theIonFeedback(nullptr),
      theCoderFactory(nullptr),
      theHBHEElectronicsSim(nullptr),
      theHFElectronicsSim(nullptr),
      theHOElectronicsSim(nullptr),
      theZDCElectronicsSim(nullptr),
      theHFQIE10ElectronicsSim(nullptr),
      theHBHEQIE11ElectronicsSim(nullptr),
      theHBHEHitFilter(),
      theHBHEQIE11HitFilter(),
      theHFHitFilter(),
      theHFQIE10HitFilter(),
      theHOHitFilter(),
      theHOSiPMHitFilter(),
      theZDCHitFilter(),
      theHBHEDigitizer(nullptr),
      theHODigitizer(nullptr),
      theHOSiPMDigitizer(nullptr),
      theHFDigitizer(nullptr),
      theZDCDigitizer(nullptr),
      theHFQIE10Digitizer(nullptr),
      theHBHEQIE11Digitizer(nullptr),
      theRelabeller(nullptr),
      isZDC(true),
      isHCAL(true),
      zdcgeo(true),
      hbhegeo(true),
      hogeo(true),
      hfgeo(true),
      doHFWindow_(ps.getParameter<bool>("doHFWindow")),
      killHE_(ps.getParameter<bool>("killHE")),
      debugCS_(ps.getParameter<bool>("debugCaloSamples")),
      ignoreTime_(ps.getParameter<bool>("ignoreGeantTime")),
      injectTestHits_(ps.getParameter<bool>("injectTestHits")),
      hitsProducer_(ps.getParameter<std::string>("hitsProducer")),
      theHOSiPMCode(ps.getParameter<edm::ParameterSet>("ho").getParameter<int>("siPMCode")),
      deliveredLumi(0.),
      agingFlagHB(ps.getParameter<bool>("HBDarkening")),
      agingFlagHE(ps.getParameter<bool>("HEDarkening")),
      zdcToken_(iC.consumes(edm::InputTag(hitsProducer_, "ZDCHITS"))),
      hcalToken_(iC.consumes(edm::InputTag(hitsProducer_, "HcalHits"))),
      m_HBDarkening(nullptr),
      m_HEDarkening(nullptr),
      m_HFRecalibration(nullptr),
      injectedHitsEnergy_(ps.getParameter<std::vector<double>>("injectTestHitsEnergy")),
      injectedHitsTime_(ps.getParameter<std::vector<double>>("injectTestHitsTime")),
      injectedHitsCells_(ps.getParameter<std::vector<int>>("injectTestHitsCells")) {
  if (agingFlagHB) {
    m_HBDarkeningToken = iC.esConsumes(edm::ESInputTag("", "HB"));
  }
  if (agingFlagHE) {
    m_HEDarkeningToken = iC.esConsumes(edm::ESInputTag("", "HE"));
  }
  if (theHOSiPMCode == 2) {
    mcParamsToken_ = iC.esConsumes();
  }

  bool doNoise = ps.getParameter<bool>("doNoise");

  bool PreMix1 = ps.getParameter<bool>("HcalPreMixStage1");  // special threshold/pedestal treatment
  bool PreMix2 = ps.getParameter<bool>("HcalPreMixStage2");  // special threshold/pedestal treatment
  bool doEmpty = ps.getParameter<bool>("doEmpty");
  deliveredLumi = ps.getParameter<double>("DelivLuminosity");
  bool agingFlagHF = ps.getParameter<bool>("HFDarkening");
  double minFCToDelay = ps.getParameter<double>("minFCToDelay");

  if (PreMix1 && PreMix2) {
    throw cms::Exception("Configuration") << "HcalDigitizer cannot operate in PreMixing digitization and "
                                             "PreMixing\n"
                                             "digi combination modes at the same time.  Please set one mode to "
                                             "False\n"
                                             "in the configuration file.";
  }

  // need to make copies, because they might get different noise generators
  theHBHEAmplifier = std::make_unique<HcalAmplifier>(&theParameterMap, doNoise, PreMix1, PreMix2);
  theHFAmplifier = std::make_unique<HcalAmplifier>(&theParameterMap, doNoise, PreMix1, PreMix2);
  theHOAmplifier = std::make_unique<HcalAmplifier>(&theParameterMap, doNoise, PreMix1, PreMix2);
  theZDCAmplifier = std::make_unique<HcalAmplifier>(&theParameterMap, doNoise, PreMix1, PreMix2);
  theHFQIE10Amplifier = std::make_unique<HcalAmplifier>(&theParameterMap, doNoise, PreMix1, PreMix2);
  theHBHEQIE11Amplifier = std::make_unique<HcalAmplifier>(&theParameterMap, doNoise, PreMix1, PreMix2);

  theCoderFactory = std::make_unique<HcalCoderFactory>(HcalCoderFactory::DB);

  theHBHEElectronicsSim =
      std::make_unique<HcalElectronicsSim>(&theParameterMap, theHBHEAmplifier.get(), theCoderFactory.get(), PreMix1);
  theHFElectronicsSim =
      std::make_unique<HcalElectronicsSim>(&theParameterMap, theHFAmplifier.get(), theCoderFactory.get(), PreMix1);
  theHOElectronicsSim =
      std::make_unique<HcalElectronicsSim>(&theParameterMap, theHOAmplifier.get(), theCoderFactory.get(), PreMix1);
  theZDCElectronicsSim =
      std::make_unique<HcalElectronicsSim>(&theParameterMap, theZDCAmplifier.get(), theCoderFactory.get(), PreMix1);
  theHFQIE10ElectronicsSim =
      std::make_unique<HcalElectronicsSim>(&theParameterMap,
                                           theHFQIE10Amplifier.get(),
                                           theCoderFactory.get(),
                                           PreMix1);  // should this use a different coder factory?
  theHBHEQIE11ElectronicsSim =
      std::make_unique<HcalElectronicsSim>(&theParameterMap,
                                           theHBHEQIE11Amplifier.get(),
                                           theCoderFactory.get(),
                                           PreMix1);  // should this use a different coder factory?

  bool doHOHPD = (theHOSiPMCode != 1);
  bool doHOSiPM = (theHOSiPMCode != 0);
  if (doHOHPD) {
    theHOResponse = std::make_unique<CaloHitResponse>(&theParameterMap, &theShapes);
    theHOResponse->setHitFilter(&theHOHitFilter);
    theHODigitizer = std::make_unique<HODigitizer>(theHOResponse.get(), theHOElectronicsSim.get(), doEmpty);
  }
  if (doHOSiPM) {
    theHOSiPMResponse->setHitFilter(&theHOSiPMHitFilter);
    theHOSiPMDigitizer = std::make_unique<HODigitizer>(theHOSiPMResponse.get(), theHOElectronicsSim.get(), doEmpty);
  }

  theHBHEResponse->setHitFilter(&theHBHEHitFilter);
  theHBHESiPMResponse->setHitFilter(&theHBHEQIE11HitFilter);

  // QIE8 and QIE11 can coexist in HBHE
  theHBHEQIE11Digitizer =
      std::make_unique<QIE11Digitizer>(theHBHESiPMResponse.get(), theHBHEQIE11ElectronicsSim.get(), doEmpty);
  theHBHEDigitizer = std::make_unique<HBHEDigitizer>(theHBHEResponse.get(), theHBHEElectronicsSim.get(), doEmpty);

  bool doTimeSlew = ps.getParameter<bool>("doTimeSlew");
  // initialize: they won't be called later if flag is set
  hcalTimeSlew_delay_ = nullptr;
  theTimeSlewSim.reset(nullptr);
  if (doTimeSlew) {
    // no time slewing for HF
    theTimeSlewSim = std::make_unique<HcalTimeSlewSim>(&theParameterMap, minFCToDelay);
    theHBHEAmplifier->setTimeSlewSim(theTimeSlewSim.get());
    theHBHEQIE11Amplifier->setTimeSlewSim(theTimeSlewSim.get());
    theHOAmplifier->setTimeSlewSim(theTimeSlewSim.get());
    theZDCAmplifier->setTimeSlewSim(theTimeSlewSim.get());
  }

  theHFResponse->setHitFilter(&theHFHitFilter);
  theHFQIE10Response->setHitFilter(&theHFQIE10HitFilter);
  theZDCResponse->setHitFilter(&theZDCHitFilter);

  // QIE8 and QIE10 can coexist in HF
  theHFQIE10Digitizer =
      std::make_unique<QIE10Digitizer>(theHFQIE10Response.get(), theHFQIE10ElectronicsSim.get(), doEmpty);
  theHFDigitizer = std::make_unique<HFDigitizer>(theHFResponse.get(), theHFElectronicsSim.get(), doEmpty);

  theZDCDigitizer = std::make_unique<ZDCDigitizer>(theZDCResponse.get(), theZDCElectronicsSim.get(), doEmpty);

  testNumbering_ = ps.getParameter<bool>("TestNumbering");
  //  edm::LogVerbatim("HcalSim") << "Flag to see if Hit Relabeller to be initiated " << testNumbering_;
  if (testNumbering_)
    theRelabeller = std::make_unique<HcalHitRelabeller>(ps.getParameter<bool>("doNeutralDensityFilter"));

  if (ps.getParameter<bool>("doIonFeedback") && theHBHEResponse) {
    theIonFeedback = std::make_unique<HPDIonFeedbackSim>(ps, &theShapes);
    theHBHEResponse->setPECorrection(theIonFeedback.get());
    if (ps.getParameter<bool>("doThermalNoise")) {
      theHBHEAmplifier->setIonFeedbackSim(theIonFeedback.get());
    }
  }

  // option to save CaloSamples as event product for debugging
  if (debugCS_) {
    if (theHBHEDigitizer)
      theHBHEDigitizer->setDebugCaloSamples(true);
    if (theHBHEQIE11Digitizer)
      theHBHEQIE11Digitizer->setDebugCaloSamples(true);
    if (theHODigitizer)
      theHODigitizer->setDebugCaloSamples(true);
    if (theHOSiPMDigitizer)
      theHOSiPMDigitizer->setDebugCaloSamples(true);
    if (theHFDigitizer)
      theHFDigitizer->setDebugCaloSamples(true);
    if (theHFQIE10Digitizer)
      theHFQIE10Digitizer->setDebugCaloSamples(true);
    theZDCDigitizer->setDebugCaloSamples(true);
  }

  // option to ignore Geant time distribution in SimHits, for debugging
  if (ignoreTime_) {
    theHBHEResponse->setIgnoreGeantTime(ignoreTime_);
    theHBHESiPMResponse->setIgnoreGeantTime(ignoreTime_);
    theHOResponse->setIgnoreGeantTime(ignoreTime_);
    theHOSiPMResponse->setIgnoreGeantTime(ignoreTime_);
    theHFResponse->setIgnoreGeantTime(ignoreTime_);
    theHFQIE10Response->setIgnoreGeantTime(ignoreTime_);
    theZDCResponse->setIgnoreGeantTime(ignoreTime_);
  }

  if (agingFlagHF)
    m_HFRecalibration = std::make_unique<HFRecalibration>(ps.getParameter<edm::ParameterSet>("HFRecalParameterBlock"));
}

HcalDigitizer::~HcalDigitizer() {}

void HcalDigitizer::setHBHENoiseSignalGenerator(HcalBaseSignalGenerator *noiseGenerator) {
  noiseGenerator->setParameterMap(&theParameterMap);
  noiseGenerator->setElectronicsSim(theHBHEElectronicsSim.get());
  if (theHBHEDigitizer)
    theHBHEDigitizer->setNoiseSignalGenerator(noiseGenerator);
  theHBHEAmplifier->setNoiseSignalGenerator(noiseGenerator);
}

void HcalDigitizer::setQIE11NoiseSignalGenerator(HcalBaseSignalGenerator *noiseGenerator) {
  noiseGenerator->setParameterMap(&theParameterMap);
  noiseGenerator->setElectronicsSim(theHBHEQIE11ElectronicsSim.get());
  if (theHBHEQIE11Digitizer)
    theHBHEQIE11Digitizer->setNoiseSignalGenerator(noiseGenerator);
  theHBHEQIE11Amplifier->setNoiseSignalGenerator(noiseGenerator);
}

void HcalDigitizer::setHFNoiseSignalGenerator(HcalBaseSignalGenerator *noiseGenerator) {
  noiseGenerator->setParameterMap(&theParameterMap);
  noiseGenerator->setElectronicsSim(theHFElectronicsSim.get());
  if (theHFDigitizer)
    theHFDigitizer->setNoiseSignalGenerator(noiseGenerator);
  theHFAmplifier->setNoiseSignalGenerator(noiseGenerator);
}

void HcalDigitizer::setQIE10NoiseSignalGenerator(HcalBaseSignalGenerator *noiseGenerator) {
  noiseGenerator->setParameterMap(&theParameterMap);
  noiseGenerator->setElectronicsSim(theHFQIE10ElectronicsSim.get());
  if (theHFQIE10Digitizer)
    theHFQIE10Digitizer->setNoiseSignalGenerator(noiseGenerator);
  theHFQIE10Amplifier->setNoiseSignalGenerator(noiseGenerator);
}

void HcalDigitizer::setHONoiseSignalGenerator(HcalBaseSignalGenerator *noiseGenerator) {
  noiseGenerator->setParameterMap(&theParameterMap);
  noiseGenerator->setElectronicsSim(theHOElectronicsSim.get());
  if (theHODigitizer)
    theHODigitizer->setNoiseSignalGenerator(noiseGenerator);
  if (theHOSiPMDigitizer)
    theHOSiPMDigitizer->setNoiseSignalGenerator(noiseGenerator);
  theHOAmplifier->setNoiseSignalGenerator(noiseGenerator);
}

void HcalDigitizer::setZDCNoiseSignalGenerator(HcalBaseSignalGenerator *noiseGenerator) {
  noiseGenerator->setParameterMap(&theParameterMap);
  noiseGenerator->setElectronicsSim(theZDCElectronicsSim.get());
  theZDCDigitizer->setNoiseSignalGenerator(noiseGenerator);
  theZDCAmplifier->setNoiseSignalGenerator(noiseGenerator);
}

void HcalDigitizer::initializeEvent(edm::Event const &e, edm::EventSetup const &eventSetup) {
  setup(eventSetup);

  // get the appropriate gains, noises, & widths for this event
  const HcalDbService *conditions = &eventSetup.getData(conditionsToken_);

  theShapes.setDbService(conditions);

  theHBHEAmplifier->setDbService(conditions);
  theHFAmplifier->setDbService(conditions);
  theHOAmplifier->setDbService(conditions);
  theZDCAmplifier->setDbService(conditions);
  theHFQIE10Amplifier->setDbService(conditions);
  theHBHEQIE11Amplifier->setDbService(conditions);

  theHFQIE10ElectronicsSim->setDbService(conditions);
  theHBHEQIE11ElectronicsSim->setDbService(conditions);

  theCoderFactory->setDbService(conditions);
  theParameterMap.setDbService(conditions);

  // initialize hits
  if (theHBHEDigitizer)
    theHBHEDigitizer->initializeHits();
  if (theHBHEQIE11Digitizer)
    theHBHEQIE11Digitizer->initializeHits();
  if (theHODigitizer)
    theHODigitizer->initializeHits();
  if (theHOSiPMDigitizer)
    theHOSiPMDigitizer->initializeHits();
  if (theHFQIE10Digitizer)
    theHFQIE10Digitizer->initializeHits();
  if (theHFDigitizer)
    theHFDigitizer->initializeHits();
  theZDCDigitizer->initializeHits();
}

void HcalDigitizer::accumulateCaloHits(edm::Handle<std::vector<PCaloHit>> const &hcalHandle,
                                       edm::Handle<std::vector<PCaloHit>> const &zdcHandle,
                                       int bunchCrossing,
                                       CLHEP::HepRandomEngine *engine,
                                       const HcalTopology *htopoP) {
  // Step A: pass in inputs, and accumulate digis
  if (isHCAL) {
    std::vector<PCaloHit> hcalHitsOrig = *hcalHandle.product();
    if (injectTestHits_)
      hcalHitsOrig = injectedHits_;
    std::vector<PCaloHit> hcalHits;
    hcalHits.reserve(hcalHitsOrig.size());

    // evaluate darkening before relabeling
    if (testNumbering_) {
      if (m_HBDarkening || m_HEDarkening || m_HFRecalibration) {
        darkening(hcalHitsOrig);
      }
      // Relabel PCaloHits if necessary
      edm::LogInfo("HcalDigitizer") << "Calling Relabeller";
      theRelabeller->process(hcalHitsOrig);
    }

    // eliminate bad hits
    for (unsigned int i = 0; i < hcalHitsOrig.size(); i++) {
      DetId id(hcalHitsOrig[i].id());
      HcalDetId hid(id);
      if (!htopoP->validHcal(hid)) {
        edm::LogError("HcalDigitizer") << "bad hcal id found in digitizer. Skipping " << id.rawId() << " " << hid;
        continue;
      } else if (hid.subdet() == HcalForward && !doHFWindow_ && hcalHitsOrig[i].depth() != 0) {
        // skip HF window hits unless desired
        continue;
      } else if (killHE_ && hid.subdet() == HcalEndcap) {
        // remove HE hits if asked for (phase 2)
        continue;
      } else {
#ifdef EDM_ML_DEBUG
        edm::LogVerbatim("HcalSim") << "HcalDigitizer format " << hid.oldFormat() << " for " << hid;
#endif
        DetId newid = DetId(hid.newForm());
#ifdef EDM_ML_DEBUG
        edm::LogVerbatim("HcalSim") << "Hit " << i << " out of " << hcalHits.size() << " " << std::hex << id.rawId()
                                    << " --> " << newid.rawId() << std::dec << " " << HcalDetId(newid.rawId()) << '\n';
#endif
        hcalHitsOrig[i].setID(newid.rawId());
        hcalHits.push_back(hcalHitsOrig[i]);
      }
    }

    if (hbhegeo) {
      if (theHBHEDigitizer)
        theHBHEDigitizer->add(hcalHits, bunchCrossing, engine);
      if (theHBHEQIE11Digitizer)
        theHBHEQIE11Digitizer->add(hcalHits, bunchCrossing, engine);
    }

    if (hogeo) {
      if (theHODigitizer)
        theHODigitizer->add(hcalHits, bunchCrossing, engine);
      if (theHOSiPMDigitizer)
        theHOSiPMDigitizer->add(hcalHits, bunchCrossing, engine);
    }

    if (hfgeo) {
      if (theHFDigitizer)
        theHFDigitizer->add(hcalHits, bunchCrossing, engine);
      if (theHFQIE10Digitizer)
        theHFQIE10Digitizer->add(hcalHits, bunchCrossing, engine);
    }
  } else {
    edm::LogInfo("HcalDigitizer") << "We don't have HCAL hit collection available ";
  }

  if (isZDC) {
    if (zdcgeo) {
      theZDCDigitizer->add(*zdcHandle.product(), bunchCrossing, engine);
    }
  } else {
    edm::LogInfo("HcalDigitizer") << "We don't have ZDC hit collection available ";
  }
}

void HcalDigitizer::accumulate(edm::Event const &e, edm::EventSetup const &eventSetup, CLHEP::HepRandomEngine *engine) {
  // Step A: Get Inputs
  const edm::Handle<std::vector<PCaloHit>> &zdcHandle = e.getHandle(zdcToken_);
  isZDC = zdcHandle.isValid();

  const edm::Handle<std::vector<PCaloHit>> &hcalHandle = e.getHandle(hcalToken_);
  isHCAL = hcalHandle.isValid() or injectTestHits_;

  const HcalTopology *htopoP = &eventSetup.getData(topoToken_);

  accumulateCaloHits(hcalHandle, zdcHandle, 0, engine, htopoP);
}

void HcalDigitizer::accumulate(PileUpEventPrincipal const &e,
                               edm::EventSetup const &eventSetup,
                               CLHEP::HepRandomEngine *engine) {
  // Step A: Get Inputs
  edm::InputTag zdcTag(hitsProducer_, "ZDCHITS");
  edm::Handle<std::vector<PCaloHit>> zdcHandle;
  e.getByLabel(zdcTag, zdcHandle);
  isZDC = zdcHandle.isValid();

  edm::InputTag hcalTag(hitsProducer_, "HcalHits");
  edm::Handle<std::vector<PCaloHit>> hcalHandle;
  e.getByLabel(hcalTag, hcalHandle);
  isHCAL = hcalHandle.isValid();

  const HcalTopology *htopoP = &eventSetup.getData(topoToken_);

  accumulateCaloHits(hcalHandle, zdcHandle, e.bunchCrossing(), engine, htopoP);
}

void HcalDigitizer::finalizeEvent(edm::Event &e, const edm::EventSetup &eventSetup, CLHEP::HepRandomEngine *engine) {
  // Step B: Create empty output
  std::unique_ptr<HBHEDigiCollection> hbheResult(new HBHEDigiCollection());
  std::unique_ptr<HODigiCollection> hoResult(new HODigiCollection());
  std::unique_ptr<HFDigiCollection> hfResult(new HFDigiCollection());
  std::unique_ptr<ZDCDigiCollection> zdcResult(new ZDCDigiCollection());
  std::unique_ptr<QIE10DigiCollection> hfQIE10Result(new QIE10DigiCollection(
      !theHFQIE10DetIds.empty() ? theHFQIE10Response.get()->getReadoutFrameSize(theHFQIE10DetIds[0])
                                : QIE10DigiCollection::MAXSAMPLES));
  std::unique_ptr<QIE11DigiCollection> hbheQIE11Result(new QIE11DigiCollection(
      !theHBHEQIE11DetIds.empty() ? theHBHESiPMResponse.get()->getReadoutFrameSize(theHBHEQIE11DetIds[0]) :
                                  //      theParameterMap->simParameters(theHBHEQIE11DetIds[0]).readoutFrameSize()
          //      :
          QIE11DigiCollection::MAXSAMPLES));

  // Step C: Invoke the algorithm, getting back outputs.
  if (isHCAL && hbhegeo) {
    if (theHBHEDigitizer)
      theHBHEDigitizer->run(*hbheResult, engine);
    if (theHBHEQIE11Digitizer)
      theHBHEQIE11Digitizer->run(*hbheQIE11Result, engine);
  }
  if (isHCAL && hogeo) {
    if (theHODigitizer)
      theHODigitizer->run(*hoResult, engine);
    if (theHOSiPMDigitizer)
      theHOSiPMDigitizer->run(*hoResult, engine);
  }
  if (isHCAL && hfgeo) {
    if (theHFDigitizer)
      theHFDigitizer->run(*hfResult, engine);
    if (theHFQIE10Digitizer)
      theHFQIE10Digitizer->run(*hfQIE10Result, engine);
  }
  if (isZDC && zdcgeo) {
    theZDCDigitizer->run(*zdcResult, engine);
  }

  edm::LogInfo("HcalDigitizer") << "HCAL HBHE digis : " << hbheResult->size();
  edm::LogInfo("HcalDigitizer") << "HCAL HO digis   : " << hoResult->size();
  edm::LogInfo("HcalDigitizer") << "HCAL HF digis   : " << hfResult->size();
  edm::LogInfo("HcalDigitizer") << "HCAL ZDC digis  : " << zdcResult->size();
  edm::LogInfo("HcalDigitizer") << "HCAL HF QIE10 digis : " << hfQIE10Result->size();
  edm::LogInfo("HcalDigitizer") << "HCAL HBHE QIE11 digis : " << hbheQIE11Result->size();

#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HcalSim") << "\nHCAL HBHE digis : " << hbheResult->size();
  edm::LogVerbatim("HcalSim") << "HCAL HO   digis : " << hoResult->size();
  edm::LogVerbatim("HcalSim") << "HCAL HF   digis : " << hfResult->size();
  edm::LogVerbatim("HcalSim") << "HCAL ZDC  digis : " << zdcResult->size();
  edm::LogVerbatim("HcalSim") << "HCAL HF QIE10 digis : " << hfQIE10Result->size();
  edm::LogVerbatim("HcalSim") << "HCAL HBHE QIE11 digis : " << hbheQIE11Result->size();
#endif

  // Step D: Put outputs into event
  e.put(std::move(hbheResult));
  e.put(std::move(hoResult));
  e.put(std::move(hfResult));
  e.put(std::move(zdcResult));
  e.put(std::move(hfQIE10Result), "HFQIE10DigiCollection");
  e.put(std::move(hbheQIE11Result), "HBHEQIE11DigiCollection");

  if (debugCS_) {
    std::unique_ptr<CaloSamplesCollection> csResult(new CaloSamplesCollection());
    // smush together all the results
    if (theHBHEDigitizer)
      csResult->insert(
          csResult->end(), theHBHEDigitizer->getCaloSamples().begin(), theHBHEDigitizer->getCaloSamples().end());
    if (theHBHEQIE11Digitizer)
      csResult->insert(csResult->end(),
                       theHBHEQIE11Digitizer->getCaloSamples().begin(),
                       theHBHEQIE11Digitizer->getCaloSamples().end());
    if (theHODigitizer)
      csResult->insert(
          csResult->end(), theHODigitizer->getCaloSamples().begin(), theHODigitizer->getCaloSamples().end());
    if (theHOSiPMDigitizer)
      csResult->insert(
          csResult->end(), theHOSiPMDigitizer->getCaloSamples().begin(), theHOSiPMDigitizer->getCaloSamples().end());
    if (theHFDigitizer)
      csResult->insert(
          csResult->end(), theHFDigitizer->getCaloSamples().begin(), theHFDigitizer->getCaloSamples().end());
    if (theHFQIE10Digitizer)
      csResult->insert(
          csResult->end(), theHFQIE10Digitizer->getCaloSamples().begin(), theHFQIE10Digitizer->getCaloSamples().end());
    csResult->insert(
        csResult->end(), theZDCDigitizer->getCaloSamples().begin(), theZDCDigitizer->getCaloSamples().end());
    e.put(std::move(csResult), "HcalSamples");
  }

  if (injectTestHits_) {
    std::unique_ptr<edm::PCaloHitContainer> pcResult(new edm::PCaloHitContainer());
    pcResult->insert(pcResult->end(), injectedHits_.begin(), injectedHits_.end());
    e.put(std::move(pcResult), "HcalHits");
  }

#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HcalSim") << "\n========>  HcalDigitizer e.put\n";
#endif
}

void HcalDigitizer::setup(const edm::EventSetup &es) {
  checkGeometry(es);

  if (agingFlagHB) {
    m_HBDarkening = &es.getData(m_HBDarkeningToken);
  }
  if (agingFlagHE) {
    m_HEDarkening = &es.getData(m_HEDarkeningToken);
  }

  hcalTimeSlew_delay_ = &es.getData(hcalTimeSlew_delay_token_);

  theHBHEAmplifier->setTimeSlew(hcalTimeSlew_delay_);
  theHBHEQIE11Amplifier->setTimeSlew(hcalTimeSlew_delay_);
  theHOAmplifier->setTimeSlew(hcalTimeSlew_delay_);
  theZDCAmplifier->setTimeSlew(hcalTimeSlew_delay_);
}

void HcalDigitizer::checkGeometry(const edm::EventSetup &eventSetup) {
  theGeometry = &eventSetup.getData(theGeometryToken);
  theRecNumber = &eventSetup.getData(theRecNumberToken);

  if (theHBHEResponse)
    theHBHEResponse->setGeometry(theGeometry);
  if (theHBHESiPMResponse)
    theHBHESiPMResponse->setGeometry(theGeometry);
  if (theHOResponse)
    theHOResponse->setGeometry(theGeometry);
  if (theHOSiPMResponse)
    theHOSiPMResponse->setGeometry(theGeometry);
  theHFResponse->setGeometry(theGeometry);
  theHFQIE10Response->setGeometry(theGeometry);
  theZDCResponse->setGeometry(theGeometry);
  if (theRelabeller)
    theRelabeller->setGeometry(theRecNumber);

  // See if it's been updated
  bool check1 = theGeometryWatcher_.check(eventSetup);
  bool check2 = theRecNumberWatcher_.check(eventSetup);
  if (check1 or check2) {
    updateGeometry(eventSetup);
  }
}

void HcalDigitizer::updateGeometry(const edm::EventSetup &eventSetup) {
  const std::vector<DetId> &hbCells = theGeometry->getValidDetIds(DetId::Hcal, HcalBarrel);
  const std::vector<DetId> &heCells = theGeometry->getValidDetIds(DetId::Hcal, HcalEndcap);
  const std::vector<DetId> &hoCells = theGeometry->getValidDetIds(DetId::Hcal, HcalOuter);
  const std::vector<DetId> &hfCells = theGeometry->getValidDetIds(DetId::Hcal, HcalForward);
  const std::vector<DetId> &zdcCells = theGeometry->getValidDetIds(DetId::Calo, HcalZDCDetId::SubdetectorId);
  // const std::vector<DetId>& hcalTrigCells =
  // geometry->getValidDetIds(DetId::Hcal, HcalTriggerTower); const
  // std::vector<DetId>& hcalCalib = geometry->getValidDetIds(DetId::Calo,
  // HcalCastorDetId::SubdetectorId);
  //  edm::LogVerbatim("HcalSim") <<"HcalDigitizer::CheckGeometry number of cells: << zdcCells.size();
  if (zdcCells.empty())
    zdcgeo = false;
  if (hbCells.empty() && heCells.empty())
    hbhegeo = false;
  if (hoCells.empty())
    hogeo = false;
  if (hfCells.empty())
    hfgeo = false;
  // combine HB & HE

  hbheCells = hbCells;
  if (!killHE_) {
    hbheCells.insert(hbheCells.end(), heCells.begin(), heCells.end());
  }
  // handle mixed QIE8/11 scenario in HBHE
  buildHBHEQIECells(hbheCells, eventSetup);
  if (theHBHESiPMResponse)
    theHBHESiPMResponse->setDetIds(theHBHEQIE11DetIds);

  if (theHOSiPMDigitizer) {
    buildHOSiPMCells(hoCells, eventSetup);
    if (theHOSiPMResponse)
      theHOSiPMResponse->setDetIds(hoCells);
  }

  // handle mixed QIE8/10 scenario in HF
  buildHFQIECells(hfCells, eventSetup);

  theZDCDigitizer->setDetIds(zdcCells);

  // fill test hits collection if desired and empty
  if (injectTestHits_ && injectedHits_.empty() && !injectedHitsCells_.empty() && !injectedHitsEnergy_.empty()) {
    // make list of specified cells if desired
    std::vector<DetId> testCells;
    if (injectedHitsCells_.size() >= 4) {
      testCells.reserve(injectedHitsCells_.size() / 4);
      for (unsigned ic = 0; ic < injectedHitsCells_.size(); ic += 4) {
        if (ic + 4 > injectedHitsCells_.size())
          break;
        testCells.push_back(HcalDetId((HcalSubdetector)injectedHitsCells_[ic],
                                      injectedHitsCells_[ic + 1],
                                      injectedHitsCells_[ic + 2],
                                      injectedHitsCells_[ic + 3]));
      }
    } else {
      int testSubdet = injectedHitsCells_[0];
      if (testSubdet == HcalBarrel)
        testCells = hbCells;
      else if (testSubdet == HcalEndcap)
        testCells = heCells;
      else if (testSubdet == HcalForward)
        testCells = hfCells;
      else if (testSubdet == HcalOuter)
        testCells = hoCells;
      else
        throw cms::Exception("Configuration") << "Unknown subdet " << testSubdet << " for HCAL test hit injection";
    }
    bool useHitTimes = (injectedHitsTime_.size() == injectedHitsEnergy_.size());
    injectedHits_.reserve(testCells.size() * injectedHitsEnergy_.size());
    for (unsigned ih = 0; ih < injectedHitsEnergy_.size(); ++ih) {
      double tmp = useHitTimes ? injectedHitsTime_[ih] : 0.;
      for (auto &aCell : testCells) {
        injectedHits_.emplace_back(aCell, injectedHitsEnergy_[ih], tmp);
      }
    }
  }
}

void HcalDigitizer::buildHFQIECells(const std::vector<DetId> &allCells, const edm::EventSetup &eventSetup) {
  // if results are already cached, no need to look again
  if (!theHFQIE8DetIds.empty() || !theHFQIE10DetIds.empty())
    return;

  // get the QIETypes
  // intentional copy
  HcalQIETypes qieTypes = eventSetup.getData(qieTypesToken_);
  if (qieTypes.topo() == nullptr) {
    qieTypes.setTopo(&eventSetup.getData(topoToken_));
  }

  for (std::vector<DetId>::const_iterator detItr = allCells.begin(); detItr != allCells.end(); ++detItr) {
    HcalQIENum qieType = HcalQIENum(qieTypes.getValues(*detItr)->getValue());
    if (qieType == QIE8) {
      theHFQIE8DetIds.push_back(*detItr);
    } else if (qieType == QIE10) {
      theHFQIE10DetIds.push_back(*detItr);
    } else {  // default is QIE8
      theHFQIE8DetIds.push_back(*detItr);
    }
  }

  if (!theHFQIE8DetIds.empty())
    theHFDigitizer->setDetIds(theHFQIE8DetIds);
  else {
    theHFDigitizer.reset();
  }

  if (!theHFQIE10DetIds.empty())
    theHFQIE10Digitizer->setDetIds(theHFQIE10DetIds);
  else {
    theHFQIE10Digitizer.reset();
  }
}

void HcalDigitizer::buildHBHEQIECells(const std::vector<DetId> &allCells, const edm::EventSetup &eventSetup) {
  // if results are already cached, no need to look again
  if (!theHBHEQIE8DetIds.empty() || !theHBHEQIE11DetIds.empty())
    return;

  // get the QIETypes
  // intentional copy
  HcalQIETypes qieTypes = eventSetup.getData(qieTypesToken_);
  if (qieTypes.topo() == nullptr) {
    qieTypes.setTopo(&eventSetup.getData(topoToken_));
  }

  for (std::vector<DetId>::const_iterator detItr = allCells.begin(); detItr != allCells.end(); ++detItr) {
    HcalQIENum qieType = HcalQIENum(qieTypes.getValues(*detItr)->getValue());
    if (qieType == QIE8) {
      theHBHEQIE8DetIds.push_back(*detItr);
    } else if (qieType == QIE11) {
      theHBHEQIE11DetIds.push_back(*detItr);
    } else {  // default is QIE8
      theHBHEQIE8DetIds.push_back(*detItr);
    }
  }

  if (!theHBHEQIE8DetIds.empty())
    theHBHEDigitizer->setDetIds(theHBHEQIE8DetIds);
  else {
    theHBHEDigitizer.reset();
  }

  if (!theHBHEQIE11DetIds.empty())
    theHBHEQIE11Digitizer->setDetIds(theHBHEQIE11DetIds);
  else {
    theHBHEQIE11Digitizer.reset();
  }

  if (!theHBHEQIE8DetIds.empty() && !theHBHEQIE11DetIds.empty()) {
    theHBHEHitFilter.setDetIds(theHBHEQIE8DetIds);
    theHBHEQIE11HitFilter.setDetIds(theHBHEQIE11DetIds);
  }
}

void HcalDigitizer::buildHOSiPMCells(const std::vector<DetId> &allCells, const edm::EventSetup &eventSetup) {
  // all HPD

  if (theHOSiPMCode == 0) {
    theHODigitizer->setDetIds(allCells);
  } else if (theHOSiPMCode == 1) {
    theHOSiPMDigitizer->setDetIds(allCells);
    // FIXME pick Zecotek or hamamatsu?
  } else if (theHOSiPMCode == 2) {
    std::vector<HcalDetId> zecotekDetIds, hamamatsuDetIds;

    // intentional copy
    HcalMCParams mcParams = eventSetup.getData(mcParamsToken_);
    if (mcParams.topo() == nullptr) {
      mcParams.setTopo(&eventSetup.getData(topoToken_));
    }

    for (std::vector<DetId>::const_iterator detItr = allCells.begin(); detItr != allCells.end(); ++detItr) {
      int shapeType = mcParams.getValues(*detItr)->signalShape();
      if (shapeType == HcalShapes::ZECOTEK) {
        zecotekDetIds.emplace_back(*detItr);
        theHOSiPMDetIds.push_back(*detItr);
      } else if (shapeType == HcalShapes::HAMAMATSU) {
        hamamatsuDetIds.emplace_back(*detItr);
        theHOSiPMDetIds.push_back(*detItr);
      } else {
        theHOHPDDetIds.push_back(*detItr);
      }
    }

    if (!theHOHPDDetIds.empty())
      theHODigitizer->setDetIds(theHOHPDDetIds);
    else {
      theHODigitizer.reset();
    }

    if (!theHOSiPMDetIds.empty())
      theHOSiPMDigitizer->setDetIds(theHOSiPMDetIds);
    else {
      theHOSiPMDigitizer.reset();
    }

    if (!theHOHPDDetIds.empty() && !theHOSiPMDetIds.empty()) {
      theHOSiPMHitFilter.setDetIds(theHOSiPMDetIds);
      theHOHitFilter.setDetIds(theHOHPDDetIds);
    }

    theParameterMap.setHOZecotekDetIds(zecotekDetIds);
    theParameterMap.setHOHamamatsuDetIds(hamamatsuDetIds);

    // make sure we don't got through this exercise again
    theHOSiPMCode = -2;
  }
}

void HcalDigitizer::darkening(std::vector<PCaloHit> &hcalHits) {
  for (unsigned int ii = 0; ii < hcalHits.size(); ++ii) {
    uint32_t tmpId = hcalHits[ii].id();
    int det, z, depth, ieta, phi, lay;
    HcalTestNumbering::unpackHcalIndex(tmpId, det, z, depth, ieta, phi, lay);

    bool darkened = false;
    float dweight = 1.;

    if (det == int(HcalBarrel) && m_HBDarkening) {
      // HB darkening
      dweight = m_HBDarkening->degradation(deliveredLumi, ieta, lay);
      darkened = true;
    } else if (det == int(HcalEndcap) && m_HEDarkening) {
      // HE darkening
      dweight = m_HEDarkening->degradation(deliveredLumi, ieta, lay);
      darkened = true;
    } else if (det == int(HcalForward) && m_HFRecalibration) {
      // HF darkening - approximate: invert recalibration factor
      dweight = 1.0 / m_HFRecalibration->getCorr(ieta, depth, deliveredLumi);
      darkened = true;
    }

    // reset hit energy
    if (darkened)
      hcalHits[ii].setEnergy(hcalHits[ii].energy() * dweight);
  }
}
