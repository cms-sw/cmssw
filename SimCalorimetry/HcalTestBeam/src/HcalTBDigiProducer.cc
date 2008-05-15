#include "FWCore/PluginManager/interface/PluginManager.h"

#include "SimCalorimetry/HcalTestBeam/interface/HcalTBDigiProducer.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "SimCalorimetry/CaloSimAlgos/interface/CaloTDigitizer.h"
#include "SimCalorimetry/CaloSimAlgos/interface/CaloShapeIntegrator.h"
#include "DataFormats/HcalDigi/interface/HcalDigiCollections.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "CalibFormats/HcalObjects/interface/HcalDbService.h"
#include "CalibFormats/HcalObjects/interface/HcalDbRecord.h"
#include "SimDataFormats/CrossingFrame/interface/CrossingFrame.h"
#include "SimDataFormats/CrossingFrame/interface/MixCollection.h"
#include "SimDataFormats/EcalTestBeam/interface/PEcalTBInfo.h"

HcalTBDigiProducer::HcalTBDigiProducer(const edm::ParameterSet& ps) :
  theParameterMap(new HcalTBSimParameterMap(ps)), 
  theHcalShape(new HcalShape()),
  theHcalIntegratedShape(new CaloShapeIntegrator(theHcalShape)),
  theHBHEResponse(new CaloHitResponse(theParameterMap, theHcalIntegratedShape)),
  theHOResponse(new CaloHitResponse(theParameterMap, theHcalIntegratedShape)),
  theAmplifier(0), theCoderFactory(0), theElectronicsSim(0), 
  theHitCorrection(0), theHBHEDigitizer(0), theHODigitizer(0), theHBHEHits(),
  theHOHits(), thisPhaseShift(0) {
  
  produces<HBHEDigiCollection>();
  produces<HODigiCollection>();

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
  theAmplifier = new HcalAmplifier(theParameterMap, doNoise);
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


void HcalTBDigiProducer::produce(edm::Event& e, const edm::EventSetup& eventSetup) {

  // get the appropriate gains, noises, & widths for this event
  edm::ESHandle<HcalDbService> conditions;
  eventSetup.get<HcalDbRecord>().get(conditions);
  theAmplifier->setDbService(conditions.product());
  theCoderFactory->setDbService(conditions.product());

  // get the correct geometry
  checkGeometry(eventSetup);

  theHBHEHits.clear();
  theHOHits.clear();

  // Step A: Get Inputs
  edm::Handle<CrossingFrame<PCaloHit> > cf;
  // e.getByType(cf);

  LogDebug("HcalSim") << "HcalTBDigiProducer::produce trying to ger SimHit";
  // test access to SimHits
  const std::string subdet("HcalHits");
  // std::auto_ptr<MixCollection<PCaloHit> > col(new MixCollection<PCaloHit>(cf.product(), subdet));
  e.getByLabel("mix", subdet, cf);
  std::auto_ptr<MixCollection<PCaloHit> > col(new MixCollection<PCaloHit>(cf.product() ));

  LogDebug("HcalSim") << "HcalTBDigiProducer::produce Collection of SimHit found";
  if(theHitCorrection != 0) {
    theHitCorrection->fillChargeSums(*col);
  }
  LogDebug("HcalSim") << "HcalTBDigiProducer::produce Hits corrected";

  // Step B: Create empty output
  std::auto_ptr<HBHEDigiCollection> hbheResult(new HBHEDigiCollection());
  std::auto_ptr<HODigiCollection> hoResult(new HODigiCollection());
  LogDebug("HcalSim") << "HcalTBDigiProducer::produce Empty collection created";
  if (doPhaseShift) {
    
    edm::Handle<PEcalTBInfo> theEcalTBInfo;
    e.getByLabel(ecalTBInfoLabel,theEcalTBInfo);
    thisPhaseShift = theEcalTBInfo->phaseShift();

    DetId detIdHB(DetId::Hcal, 1);
    setPhaseShift(detIdHB);
    DetId detIdHO(DetId::Hcal, 3);
    setPhaseShift(detIdHO);
  }

  // Step C: Invoke the algorithm, passing in inputs and getting back outputs.
  theHBHEDigitizer->run(*col, *hbheResult);
  edm::LogInfo("HcalSim") << "HcalTBDigiProducer: HBHE digis : " << hbheResult->size();
  theHODigitizer->run(*col, *hoResult);
  edm::LogInfo("HcalSim") << "HcalTBDigiProducer: HO digis   : " << hoResult->size();

  // Step D: Put outputs into event
  e.put(hbheResult);
  e.put(hoResult);

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
