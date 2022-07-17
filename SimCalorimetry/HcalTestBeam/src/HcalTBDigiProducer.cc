#include <memory>

#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/RandomNumberGenerator.h"
#include "FWCore/Utilities/interface/StreamID.h"
#include "SimCalorimetry/HcalTestBeam/interface/HcalTBDigiProducer.h"

#include "CalibFormats/HcalObjects/interface/HcalDbService.h"
#include "DataFormats/HcalDigi/interface/HcalDigiCollections.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "SimCalorimetry/CaloSimAlgos/interface/CaloShapeIntegrator.h"
#include "SimCalorimetry/CaloSimAlgos/interface/CaloTDigitizer.h"
#include "SimDataFormats/EcalTestBeam/interface/PEcalTBInfo.h"
#include "SimGeneral/MixingModule/interface/PileUpEventPrincipal.h"

#include "CalibCalorimetry/HcalAlgos/interface/HcalTimeSlew.h"

HcalTBDigiProducer::HcalTBDigiProducer(const edm::ParameterSet &ps,
                                       edm::ProducesCollector producesCollector,
                                       edm::ConsumesCollector &iC)
    : tunePhaseShift(ps.getUntrackedParameter<double>("tunePhaseShiftTB", 1.)),
      ecalTBInfoLabel(ps.getUntrackedParameter<std::string>("EcalTBInfoLabel", "SimEcalTBG4Object")),
      theParameterMap(new HcalTBSimParameterMap(ps)),
      paraMap(new HcalSimParameterMap(ps)),
      theHcalShape(new HcalShape()),
      theHcalIntegratedShape(new CaloShapeIntegrator(theHcalShape)),
      theHBHEResponse(new CaloHitResponse(theParameterMap, theHcalIntegratedShape)),
      theHOResponse(new CaloHitResponse(theParameterMap, theHcalIntegratedShape)),
      theAmplifier(nullptr),
      theCoderFactory(nullptr),
      theElectronicsSim(nullptr),
      theTimeSlewSim(nullptr),
      geometryToken_(iC.esConsumes()),
      conditionsToken_(iC.esConsumes()),
      hcalTimeSlew_delay_token_(iC.esConsumes(edm::ESInputTag("", "HBHE"))),
      hcalToken_(iC.consumes<std::vector<PCaloHit>>(edm::InputTag("g4SimHits", "HcalHits"))),
      theHBHEHits(),
      theHOHits(),
      thisPhaseShift(0) {
  std::string const instance("simHcalDigis");
  producesCollector.produces<HBHEDigiCollection>(instance);
  producesCollector.produces<HODigiCollection>(instance);

  DetId detId(DetId::Hcal, 1);
  bool syncPhase = (theParameterMap->simParameters(detId)).syncPhase();
  doPhaseShift = !syncPhase;

  theHBHEResponse->setHitFilter(&theHBHEHitFilter);
  theHOResponse->setHitFilter(&theHOHitFilter);

  bool doNoise = ps.getParameter<bool>("doNoise");
  bool dummy1 = false;
  bool dummy2 = false;  // extra arguments for premixing
  theAmplifier = new HcalAmplifier(theParameterMap, doNoise, dummy1, dummy2);
  theCoderFactory = new HcalCoderFactory(HcalCoderFactory::DB);
  theElectronicsSim = new HcalElectronicsSim(paraMap, theAmplifier, theCoderFactory, dummy1);

  double minFCToDelay = ps.getParameter<double>("minFCToDelay");
  bool doTimeSlew = ps.getParameter<bool>("doTimeSlew");

  hcalTimeSlew_delay_ = nullptr;
  if (doTimeSlew) {
    // no time slewing for HF
    theTimeSlewSim = new HcalTimeSlewSim(theParameterMap, minFCToDelay);
    theAmplifier->setTimeSlewSim(theTimeSlewSim);
  }

  theHBHEDigitizer = std::make_unique<HBHEDigitizer>(theHBHEResponse, theElectronicsSim, doNoise);
  theHODigitizer = std::make_unique<HODigitizer>(theHOResponse, theElectronicsSim, doNoise);

  edm::LogVerbatim("HcalSim") << "HcalTBDigiProducer initialized with doNoise = " << doNoise
                              << ", doTimeSlew = " << doTimeSlew << " and doPhaseShift = " << doPhaseShift
                              << " tunePhasShift = " << tunePhaseShift;

  if (doPhaseShift)
    theEcalTBToken_ = iC.consumes<PEcalTBInfo>(edm::InputTag(ecalTBInfoLabel, ""));
}

HcalTBDigiProducer::~HcalTBDigiProducer() {
  if (theParameterMap)
    delete theParameterMap;
  if (paraMap)
    delete paraMap;
  if (theHcalShape)
    delete theHcalShape;
  if (theHcalIntegratedShape)
    delete theHcalIntegratedShape;
  if (theHBHEResponse)
    delete theHBHEResponse;
  if (theHOResponse)
    delete theHOResponse;
  if (theElectronicsSim)
    delete theElectronicsSim;
  if (theAmplifier)
    delete theAmplifier;
  if (theCoderFactory)
    delete theCoderFactory;
  if (theTimeSlewSim)
    delete theTimeSlewSim;
}

void HcalTBDigiProducer::initializeEvent(edm::Event const &e, edm::EventSetup const &eventSetup) {
  // get the appropriate gains, noises, & widths for this event
  const HcalDbService *conditions = &eventSetup.getData(conditionsToken_);
  theAmplifier->setDbService(conditions);
  theCoderFactory->setDbService(conditions);

  // get the correct geometry
  checkGeometry(eventSetup);

  // Cache random number engine
  edm::Service<edm::RandomNumberGenerator> rng;
  randomEngine_ = &rng->getEngine(e.streamID());

  theHBHEHits.clear();
  theHOHits.clear();
  if (doPhaseShift) {
    const edm::Handle<PEcalTBInfo> &theEcalTBInfo = e.getHandle(theEcalTBToken_);
    thisPhaseShift = theEcalTBInfo->phaseShift();

    DetId detIdHB(DetId::Hcal, 1);
    setPhaseShift(detIdHB);
    DetId detIdHO(DetId::Hcal, 3);
    setPhaseShift(detIdHO);
  }

  hcalTimeSlew_delay_ = &eventSetup.getData(hcalTimeSlew_delay_token_);

  theAmplifier->setTimeSlew(hcalTimeSlew_delay_);

  theHBHEDigitizer->initializeHits();
  theHODigitizer->initializeHits();
}

void HcalTBDigiProducer::accumulateCaloHits(edm::Handle<std::vector<PCaloHit>> const &hcalHandle, int bunchCrossing) {
  LogDebug("HcalSim") << "HcalTBDigiProducer::accumulate trying to get SimHit";

  if (hcalHandle.isValid()) {
    std::vector<PCaloHit> hits = *hcalHandle.product();
    LogDebug("HcalSim") << "HcalTBDigiProducer::accumulate Hits corrected";
    theHBHEDigitizer->add(hits, bunchCrossing, randomEngine_);
    theHODigitizer->add(hits, bunchCrossing, randomEngine_);
  }
}

void HcalTBDigiProducer::accumulate(edm::Event const &e, edm::EventSetup const &) {
  // Step A: Get Inputs, and accumulate digis

  edm::InputTag hcalTag("g4SimHits", "HcalHits");
  const edm::Handle<std::vector<PCaloHit>> &hcalHandle = e.getHandle(hcalToken_);

  accumulateCaloHits(hcalHandle, 0);
}

void HcalTBDigiProducer::accumulate(PileUpEventPrincipal const &e,
                                    edm::EventSetup const &,
                                    edm::StreamID const &streamID) {
  // Step A: Get Inputs, and accumulate digis

  edm::InputTag hcalTag("g4SimHits", "HcalHits");
  edm::Handle<std::vector<PCaloHit>> hcalHandle;
  e.getByLabel(hcalTag, hcalHandle);

  accumulateCaloHits(hcalHandle, e.bunchCrossing());
}

void HcalTBDigiProducer::finalizeEvent(edm::Event &e, const edm::EventSetup &eventSetup) {
  // Step B: Create empty output
  std::unique_ptr<HBHEDigiCollection> hbheResult(new HBHEDigiCollection());
  std::unique_ptr<HODigiCollection> hoResult(new HODigiCollection());
  LogDebug("HcalSim") << "HcalTBDigiProducer::produce Empty collection created";
  // Step C: Invoke the algorithm, getting back outputs.
  theHBHEDigitizer->run(*hbheResult, randomEngine_);
  edm::LogVerbatim("HcalSim") << "HcalTBDigiProducer: HBHE digis : " << hbheResult->size();
  theHODigitizer->run(*hoResult, randomEngine_);
  edm::LogVerbatim("HcalSim") << "HcalTBDigiProducer: HO digis   : " << hoResult->size();

  // Step D: Put outputs into event
  std::string const instance("simHcalDigis");
  e.put(std::move(hbheResult), instance);
  e.put(std::move(hoResult), instance);

  randomEngine_ = nullptr;  // to prevent access outside event
}

void HcalTBDigiProducer::sortHits(const edm::PCaloHitContainer &hits) {
  for (edm::PCaloHitContainer::const_iterator hitItr = hits.begin(); hitItr != hits.end(); ++hitItr) {
    HcalSubdetector subdet = HcalDetId(hitItr->id()).subdet();
    if (subdet == HcalBarrel || subdet == HcalEndcap) {
      theHBHEHits.push_back(*hitItr);
    } else if (subdet == HcalOuter) {
      theHOHits.push_back(*hitItr);
    } else {
      edm::LogError("HcalSim") << "Bad HcalHit subdetector " << subdet;
    }
  }
}

void HcalTBDigiProducer::checkGeometry(const edm::EventSetup &eventSetup) {
  // see if we need to update
  if (geometryWatcher_.check(eventSetup)) {
    theGeometry = &eventSetup.getData(geometryToken_);
    updateGeometry();
  }
}

void HcalTBDigiProducer::updateGeometry() {
  theHBHEResponse->setGeometry(theGeometry);
  theHOResponse->setGeometry(theGeometry);

  // Get cells for HB and HE
  hbheCells.clear();
  hbheCells = theGeometry->getValidDetIds(DetId::Hcal, HcalBarrel);
  std::vector<DetId> heCells = theGeometry->getValidDetIds(DetId::Hcal, HcalEndcap);
  // combine HB & HE
  hbheCells.insert(hbheCells.end(), heCells.begin(), heCells.end());

  // Get cells for HO
  hoCells.clear();
  hoCells = theGeometry->getValidDetIds(DetId::Hcal, HcalOuter);

  edm::LogVerbatim("HcalSim") << "HcalTBDigiProducer update Geometry with " << hbheCells.size()
                              << " cells in HB/HE and " << hoCells.size() << " cells in HO";

  theHBHEDigitizer->setDetIds(hbheCells);
  LogDebug("HcalSim") << "HcalTBDigiProducer: Set DetID's for HB/HE";
  theHODigitizer->setDetIds(hoCells);
  LogDebug("HcalSim") << "HcalTBDigiProducer: Set DetID's for HO";
}

void HcalTBDigiProducer::setPhaseShift(const DetId &detId) {
  const CaloSimParameters &parameters = theParameterMap->simParameters(detId);
  if (!parameters.syncPhase()) {
    int myDet = detId.subdetId();
    double passPhaseShift = thisPhaseShift + tunePhaseShift;
    if (myDet <= 2) {
      theHBHEResponse->setPhaseShift(passPhaseShift);
    } else {
      theHOResponse->setPhaseShift(passPhaseShift);
    }
  }
}
