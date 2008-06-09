#include "SimCalorimetry/CastorSim/plugins/CastorDigiProducer.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "SimCalorimetry/CaloSimAlgos/interface/CaloTDigitizer.h"
#include "SimCalorimetry/CaloSimAlgos/interface/CaloShapeIntegrator.h"
#include "DataFormats/HcalDigi/interface/HcalDigiCollections.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "CalibFormats/HcalObjects/interface/HcalDbService.h"
#include "CalibFormats/HcalObjects/interface/HcalDbRecord.h"
#include "SimDataFormats/CrossingFrame/interface/CrossingFrame.h"
#include "SimDataFormats/CrossingFrame/interface/MixCollection.h"
#include "FWCore/Utilities/interface/RandomNumberGenerator.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "DataFormats/HcalDetId/interface/HcalCastorDetId.h"

using namespace std;


CastorDigiProducer::CastorDigiProducer(const edm::ParameterSet& ps) 
: theParameterMap(new CastorSimParameterMap(ps)),
  theCastorShape(new CastorShape()),
  theCastorIntegratedShape(new CaloShapeIntegrator(theCastorShape)),
  theCastorResponse(new CaloHitResponse(theParameterMap, theCastorIntegratedShape)),
  theAmplifier(0),
  theCoderFactory(0),
  theElectronicsSim(0),
  theHitCorrection(0),
  theCastorDigitizer(0),
  theCastorHits()
{
  
  produces<CastorDigiCollection>();

  theCastorResponse->setHitFilter(&theCastorHitFilter);

  bool doTimeSlew = ps.getParameter<bool>("doTimeSlew");
  if(doTimeSlew) {
    // no time slewing for HF
    theCastorResponse->setHitCorrection(theHitCorrection);
  }

  bool doNoise = ps.getParameter<bool>("doNoise");
  theAmplifier = new CastorAmplifier(theParameterMap, doNoise);
  theCoderFactory = new CastorCoderFactory(CastorCoderFactory::DB);
  theElectronicsSim = new CastorElectronicsSim(theAmplifier, theCoderFactory);

  theCastorDigitizer = new CastorDigitizer(theCastorResponse, theElectronicsSim, doNoise);

  edm::Service<edm::RandomNumberGenerator> rng;
  if ( ! rng.isAvailable()) {
    throw cms::Exception("Configuration")
      << "CastorDigiProducer requires the RandomNumberGeneratorService\n"
         "which is not present in the configuration file.  You must add the service\n"
         "in the configuration file or remove the modules that require it.";
  }

  CLHEP::HepRandomEngine& engine = rng->getEngine();
  theAmplifier->setRandomEngine(engine);
  theElectronicsSim->setRandomEngine(engine);
}


CastorDigiProducer::~CastorDigiProducer() {
  delete theCastorDigitizer;
  delete theParameterMap;
  delete theCastorShape;
  delete theCastorIntegratedShape;
  delete theCastorResponse;
  delete theElectronicsSim;
  delete theAmplifier;
  delete theCoderFactory;
  delete theHitCorrection;
}


void CastorDigiProducer::produce(edm::Event& e, const edm::EventSetup& eventSetup) {
  // get the appropriate gains, noises, & widths for this event
  edm::ESHandle<HcalDbService> conditions;
  eventSetup.get<HcalDbRecord>().get(conditions);
  theAmplifier->setDbService(conditions.product());
  theCoderFactory->setDbService(conditions.product());
  theParameterMap->setDbService(conditions.product());

  // get the correct geometry
  checkGeometry(eventSetup);
  
  theCastorHits.clear();

  // Step A: Get Inputs
  edm::Handle<CrossingFrame<PCaloHit> > castorcf;
  e.getByLabel("mix", "CastorHits", castorcf);

  // test access to SimHits for HcalHits and ZDC hits
  std::auto_ptr<MixCollection<PCaloHit> > colcastor(new MixCollection<PCaloHit>(castorcf.product()));

  //fillFakeHits();

  if(theHitCorrection != 0)
  {
//    theHitCorrection->fillChargeSums(*col);
//    theHitCorrection->fillChargeSums(*colzdc);
    theHitCorrection->fillChargeSums(*colcastor);
  }
  // Step B: Create empty output

//  std::auto_ptr<HBHEDigiCollection> hbheResult(new HBHEDigiCollection());
//  std::auto_ptr<HODigiCollection> hoResult(new HODigiCollection());
//  std::auto_ptr<HFDigiCollection> hfResult(new HFDigiCollection());
//  std::auto_ptr<ZDCDigiCollection> zdcResult(new ZDCDigiCollection());
  std::auto_ptr<CastorDigiCollection> castorResult(new CastorDigiCollection());

  // Step C: Invoke the algorithm, passing in inputs and getting back outputs.
//  theHBHEDigitizer->run(*col, *hbheResult);
//  theHODigitizer->run(*col, *hoResult);
//  theHFDigitizer->run(*col, *hfResult);
//  theZDCDigitizer->run(*colzdc, *zdcResult);
  theCastorDigitizer->run(*colcastor, *castorResult);

//  edm::LogInfo("HcalDigiProducer") << "HCAL HBHE digis : " << hbheResult->size();
//  edm::LogInfo("HcalDigiProducer") << "HCAL HO digis   : " << hoResult->size();
//  edm::LogInfo("HcalDigiProducer") << "HCAL HF digis   : " << hfResult->size();
//  edm::LogInfo("HcalDigiProducer") << "HCAL ZDC digis   : " << zdcResult->size();
  edm::LogInfo("HcalDigiProducer") << "HCAL Castor digis   : " << castorResult->size();

  // Step D: Put outputs into event
//  e.put(hbheResult);
//  e.put(hoResult);
//  e.put(hfResult);
//  e.put(zdcResult);
  e.put(castorResult);
}


void CastorDigiProducer::sortHits(const edm::PCaloHitContainer & hits){
  for(edm::PCaloHitContainer::const_iterator hitItr = hits.begin();
      hitItr != hits.end(); ++hitItr){
    DetId detId = hitItr->id();
    if (detId.det()==DetId::Calo && detId.subdetId()==HcalCastorDetId::SubdetectorId){
      theCastorHits.push_back(*hitItr);
    }
      else {
	edm::LogError("CastorDigiProducer") << "Bad Hit subdetector " << detId.subdetId();
      }
  }
}

void CastorDigiProducer::fillFakeHits() {
 /* 
  HcalDetId barrelDetId(HcalBarrel, 1, 1, 1);
  PCaloHit barrelHit(barrelDetId.rawId(), 0.855, 0., 0., 0);

  HcalDetId endcapDetId(HcalEndcap, 17, 1, 1);
  PCaloHit endcapHit(endcapDetId.rawId(), 0.9, 0., 0., 0);

  HcalDetId outerDetId(HcalOuter, 1, 1, 4);
  PCaloHit outerHit(outerDetId.rawId(), 0.45, 0., 0., 0);

  HcalDetId forwardDetId1(HcalForward, 30, 1, 1);
  PCaloHit forwardHit1(forwardDetId1.rawId(), 35., 0., 0., 0);

  HcalDetId forwardDetId2(HcalForward, 30, 1, 2);
  PCaloHit forwardHit2(forwardDetId2.rawId(), 48., 0., 0., 0);

  HcalZDCDetId zdcDetId(HcalZDCDetId::Section(2),true,1);
  PCaloHit zdcHit(zdcDetId.rawId(), 50.0, 0.);
*/
  HcalCastorDetId castorDetId(HcalCastorDetId::Section(2),true,1,1);
  PCaloHit castorHit(castorDetId.rawId(), 50.0, 0.);

//  theHBHEHits.push_back(barrelHit);
//  theHBHEHits.push_back(endcapHit);
//  theHOHits.push_back(outerHit);
//  theHFHits.push_back(forwardHit1);
//  theHFHits.push_back(forwardHit2);
//  theZDCHits.push_back(zdcHit);
  theCastorHits.push_back(castorHit);
}


void CastorDigiProducer::checkGeometry(const edm::EventSetup & eventSetup) {
  // TODO find a way to avoid doing this every event
  edm::ESHandle<CaloGeometry> geometry;
  eventSetup.get<IdealGeometryRecord>().get(geometry);
  //theHBHEResponse->setGeometry(&*geometry);
  //theHOResponse->setGeometry(&*geometry);
  //theHFResponse->setGeometry(&*geometry);
  //theZDCResponse->setGeometry(&*geometry);
  theCastorResponse->setGeometry(&*geometry);

 // vector<DetId> hbCells =  geometry->getValidDetIds(DetId::Hcal, HcalBarrel);
//  vector<DetId> heCells =  geometry->getValidDetIds(DetId::Hcal, HcalEndcap);
//  vector<DetId> hoCells =  geometry->getValidDetIds(DetId::Hcal, HcalOuter);
//  vector<DetId> hfCells =  geometry->getValidDetIds(DetId::Hcal, HcalForward);
//  vector<DetId> zdcCells = geometry->getValidDetIds(DetId::Calo, HcalZDCDetId::SubdetectorId);
  vector<DetId> castorCells = geometry->getValidDetIds(DetId::Calo, HcalCastorDetId::SubdetectorId);

  std::cout<<"HcalDigiProducer::CheckGeometry number of cells: "<<castorCells.size()<<std::endl;
  // combine HB & HE

//  vector<DetId> hbheCells = hbCells;
//  hbheCells.insert(hbheCells.end(), heCells.begin(), heCells.end());

//  theHBHEDigitizer->setDetIds(hbheCells);
//  theHODigitizer->setDetIds(hoCells);
//  theHFDigitizer->setDetIds(hfCells);
//  theZDCDigitizer->setDetIds(zdcCells); 
  theCastorDigitizer->setDetIds(castorCells);
}


