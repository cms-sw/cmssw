#include "SimCalorimetry/HcalTestBeam/interface/HcalTBDigiProducer.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "SimCalorimetry/CaloSimAlgos/interface/CaloTDigitizer.h"
#include "SimCalorimetry/CaloSimAlgos/interface/CaloShapeIntegrator.h"
#include "DataFormats/HcalDigi/interface/HcalDigiCollections.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "CalibFormats/HcalObjects/interface/HcalDbService.h"
#include "CalibFormats/HcalObjects/interface/HcalDbRecord.h"
#include "SimDataFormats/EcalTestBeam/interface/PEcalTBInfo.h"
#include "SimGeneral/MixingModule/interface/PileUpEventPrincipal.h"

HcalTBDigiProducer::HcalTBDigiProducer(const edm::ParameterSet& ps, edm::one::EDProducerBase& mixMod, edm::ConsumesCollector& iC) :
  theParameterMap(new HcalTBSimParameterMap(ps)), 
  theHcalShape(new HcalShape()),
  theHcalIntegratedShape(new CaloShapeIntegrator(theHcalShape)),
  theHBHEResponse(new CaloHitResponse(theParameterMap, theHcalIntegratedShape)),
  theHOResponse(new CaloHitResponse(theParameterMap, theHcalIntegratedShape)),
  theAmplifier(0), theCoderFactory(0), theElectronicsSim(0), 
  theHitCorrection(0), theHBHEDigitizer(0), theHODigitizer(0), theHBHEHits(),
  theHOHits(), thisPhaseShift(0) {
  std::string const instance("simHcalDigis");
  mixMod.produces<HBHEDigiCollection>(instance);
  mixMod.produces<HODigiCollection>(instance);
  iC.consumes<std::vector<PCaloHit> >(edm::InputTag("g4SimHits", "HcalHits"));

  DetId detId(DetId::Hcal, 1);
  bool syncPhase = (theParameterMap->simParameters(detId)).syncPhase();
  doPhaseShift   = !syncPhase;

  theHBHEResponse->setHitFilter(&theHBHEHitFilter);
  theHOResponse->setHitFilter(&theHOHitFilter);

  bool doTimeSlew = ps.getParameter<bool>("doTimeSlew");
  if(doTimeSlew) {
    // no time slewing for HF
    theHitCorrection = new HcalHitCorrection(theParameterMap);
    theHBHEResponse->setHitCorrection(theHitCorrection);
    theHOResponse->setHitCorrection(theHitCorrection);
  }

  bool doNoise = ps.getParameter<bool>("doNoise");
  bool dummy1 = false; 
  bool dummy2 = false;  // extra arguments for premixing
  theAmplifier = new HcalAmplifier(theParameterMap, doNoise, dummy1, dummy2);
  theCoderFactory = new HcalCoderFactory(HcalCoderFactory::DB);
  theElectronicsSim = new HcalElectronicsSim(theAmplifier, theCoderFactory);

  theHBHEDigitizer = new HBHEDigitizer(theHBHEResponse, theElectronicsSim, doNoise);
  theHODigitizer = new HODigitizer(theHOResponse, theElectronicsSim, doNoise);

  tunePhaseShift =  ps.getUntrackedParameter<double>("tunePhaseShiftTB", 1.);
  ecalTBInfoLabel = ps.getUntrackedParameter<std::string>("EcalTBInfoLabel","SimEcalTBG4Object");
  edm::LogInfo("HcalSim") << "HcalTBDigiProducer initialized with doNoise = "
			  << doNoise << ", doTimeSlew = " << doTimeSlew
			  << " and doPhaseShift = " << doPhaseShift
			  << " tunePhasShift = " << tunePhaseShift;

  if (doPhaseShift) {
    iC.consumes<PEcalTBInfo>(edm::InputTag(ecalTBInfoLabel, ""));
  }
}

HcalTBDigiProducer::~HcalTBDigiProducer() {

  if (theHBHEDigitizer)       delete theHBHEDigitizer;
  if (theHODigitizer)         delete theHODigitizer;
  if (theParameterMap)        delete theParameterMap;
  if (theHcalShape)           delete theHcalShape;
  if (theHcalIntegratedShape) delete theHcalIntegratedShape;
  if (theHBHEResponse)        delete theHBHEResponse;
  if (theHOResponse)          delete theHOResponse;
  if (theElectronicsSim)      delete theElectronicsSim;
  if (theAmplifier)           delete theAmplifier;
  if (theCoderFactory)        delete theCoderFactory;
  if (theHitCorrection)       delete theHitCorrection;
}


void HcalTBDigiProducer::initializeEvent(edm::Event const& e, edm::EventSetup const& eventSetup) {
  // get the appropriate gains, noises, & widths for this event
  edm::ESHandle<HcalDbService> conditions;
  eventSetup.get<HcalDbRecord>().get(conditions);
  theAmplifier->setDbService(conditions.product());
  theCoderFactory->setDbService(conditions.product());

  // get the correct geometry
  checkGeometry(eventSetup);

  theHBHEHits.clear();
  theHOHits.clear();
  if (doPhaseShift) {
    
    edm::Handle<PEcalTBInfo> theEcalTBInfo;
    e.getByLabel(ecalTBInfoLabel,theEcalTBInfo);
    thisPhaseShift = theEcalTBInfo->phaseShift();

    DetId detIdHB(DetId::Hcal, 1);
    setPhaseShift(detIdHB);
    DetId detIdHO(DetId::Hcal, 3);
    setPhaseShift(detIdHO);
  }

  theHBHEDigitizer->initializeHits();
  theHODigitizer->initializeHits();
}

void HcalTBDigiProducer::accumulateCaloHits(edm::Handle<std::vector<PCaloHit> > const& hcalHandle, int bunchCrossing) {

  LogDebug("HcalSim") << "HcalTBDigiProducer::accumulate trying to get SimHit";

  if(hcalHandle.isValid()) {
    std::vector<PCaloHit> hits = *hcalHandle.product();
    if(theHitCorrection != 0) {
      theHitCorrection->fillChargeSums(hits);
    }
    LogDebug("HcalSim") << "HcalTBDigiProducer::accumulate Hits corrected";
    theHBHEDigitizer->add(hits, bunchCrossing);
    theHODigitizer->add(hits, bunchCrossing);
  }
}

void HcalTBDigiProducer::accumulate(edm::Event const& e, edm::EventSetup const&) {
  // Step A: Get Inputs, and accumulate digis

  edm::InputTag hcalTag("g4SimHits", "HcalHits");
  edm::Handle<std::vector<PCaloHit> > hcalHandle;
  e.getByLabel(hcalTag, hcalHandle);

  accumulateCaloHits(hcalHandle, 0);
}

void HcalTBDigiProducer::accumulate(PileUpEventPrincipal const& e, edm::EventSetup const&) {
  // Step A: Get Inputs, and accumulate digis

  edm::InputTag hcalTag("g4SimHits", "HcalHits");
  edm::Handle<std::vector<PCaloHit> > hcalHandle;
  e.getByLabel(hcalTag, hcalHandle);

  accumulateCaloHits(hcalHandle, e.bunchCrossing());
}

void HcalTBDigiProducer::finalizeEvent(edm::Event& e, const edm::EventSetup& eventSetup) {
  // Step B: Create empty output
  std::auto_ptr<HBHEDigiCollection> hbheResult(new HBHEDigiCollection());
  std::auto_ptr<HODigiCollection> hoResult(new HODigiCollection());
  LogDebug("HcalSim") << "HcalTBDigiProducer::produce Empty collection created";
  // Step C: Invoke the algorithm, getting back outputs.
  theHBHEDigitizer->run(*hbheResult);
  edm::LogInfo("HcalSim") << "HcalTBDigiProducer: HBHE digis : " << hbheResult->size();
  theHODigitizer->run(*hoResult);
  edm::LogInfo("HcalSim") << "HcalTBDigiProducer: HO digis   : " << hoResult->size();

  // Step D: Put outputs into event
  std::string const instance("simHcalDigis");
  e.put(hbheResult, instance);
  e.put(hoResult, instance);

}

void HcalTBDigiProducer::sortHits(const edm::PCaloHitContainer & hits) {

  for (edm::PCaloHitContainer::const_iterator hitItr = hits.begin();
       hitItr != hits.end(); ++hitItr) {
    HcalSubdetector subdet = HcalDetId(hitItr->id()).subdet();
    if(subdet == HcalBarrel || subdet == HcalEndcap) {
      theHBHEHits.push_back(*hitItr);
    } else if(subdet == HcalOuter) {
      theHOHits.push_back(*hitItr);
    } else {
      edm::LogError("HcalSim") << "Bad HcalHit subdetector " << subdet;
    }
  }
}

void HcalTBDigiProducer::checkGeometry(const edm::EventSetup & eventSetup) {

  // TODO find a way to avoid doing this every event
  edm::ESHandle<CaloGeometry> geometry;
  eventSetup.get<CaloGeometryRecord>().get(geometry);
 
  const CaloGeometry * pGeometry = &*geometry;

  // see if we need to update
  if(pGeometry != theGeometry) {
    theGeometry = pGeometry;
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

  edm::LogInfo("HcalSim") << "HcalTBDigiProducer update Geometry with "
			  << hbheCells.size() << " cells in HB/HE and "
			  << hoCells.size() << " cells in HO";

  theHBHEDigitizer->setDetIds(hbheCells);
  LogDebug("HcalSim") << "HcalTBDigiProducer: Set DetID's for HB/HE";
  theHODigitizer->setDetIds(hoCells);
  LogDebug("HcalSim") << "HcalTBDigiProducer: Set DetID's for HO";
}

void HcalTBDigiProducer::setPhaseShift(const DetId & detId) {

  const CaloSimParameters & parameters = theParameterMap->simParameters(detId);
  if ( !parameters.syncPhase() ) {
    int    myDet          = detId.subdetId();
    double passPhaseShift = thisPhaseShift + tunePhaseShift;
    if (myDet <= 2) {
      theHBHEResponse->setPhaseShift(passPhaseShift);
    } else {
      theHOResponse->setPhaseShift(passPhaseShift);
    }
  }
}
