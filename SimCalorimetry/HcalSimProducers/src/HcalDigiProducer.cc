#include "SimCalorimetry/HcalSimProducers/src/HcalDigiProducer.h"
#include "FWCore/Framework/interface/EDProducer.h"
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
#include "FWCore/Utilities/interface/RandomNumberGenerator.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "DataFormats/HcalDetId/interface/HcalZDCDetId.h"
#include "SimCalorimetry/HcalSimAlgos/interface/HPDNoiseGenerator.h"
using namespace std;


HcalDigiProducer::HcalDigiProducer(const edm::ParameterSet& ps) 
: theParameterMap(new HcalSimParameterMap(ps)),
  theHcalShape(new HcalShape()),
  theHFShape(new HFShape()),
  theZDCShape(new ZDCShape()),
  theHcalIntegratedShape(new CaloShapeIntegrator(theHcalShape)),
  theHFIntegratedShape(new CaloShapeIntegrator(theHFShape)),
  theZDCIntegratedShape(new CaloShapeIntegrator(theZDCShape)),
  theHBHEResponse(new CaloHitResponse(theParameterMap, theHcalIntegratedShape)),
  theHOResponse(new CaloHitResponse(theParameterMap, theHcalIntegratedShape)),   
  theHFResponse(new CaloHitResponse(theParameterMap, theHFIntegratedShape)),
  theZDCResponse(new CaloHitResponse(theParameterMap, theZDCIntegratedShape)),
  theAmplifier(0),
  theCoderFactory(0),
  theElectronicsSim(0),
  theHitCorrection(0),
  theHPDNoiseGenerator(0),
  theHBHEDigitizer(0),
  theHODigitizer(0),
  theHFDigitizer(0),
  theZDCDigitizer(0),
  doZDC(true)
{
  
  produces<HBHEDigiCollection>();
  produces<HODigiCollection>();
  produces<HFDigiCollection>();
  produces<ZDCDigiCollection>();

  theHBHEResponse->setHitFilter(&theHBHEHitFilter);
  theHOResponse->setHitFilter(&theHOHitFilter);
  theHFResponse->setHitFilter(&theHFHitFilter);
  theZDCResponse->setHitFilter(&theZDCHitFilter);

  bool doTimeSlew = ps.getParameter<bool>("doTimeSlew");
  if(doTimeSlew) {
    // no time slewing for HF
    theHitCorrection = new HcalHitCorrection(theParameterMap);
    theHBHEResponse->setHitCorrection(theHitCorrection);
    theHOResponse->setHitCorrection(theHitCorrection);
    theZDCResponse->setHitCorrection(theHitCorrection);
  }

  bool doNoise = ps.getParameter<bool>("doNoise");
  theAmplifier = new HcalAmplifier(theParameterMap, doNoise);
  theCoderFactory = new HcalCoderFactory(HcalCoderFactory::DB);
  theElectronicsSim = new HcalElectronicsSim(theAmplifier, theCoderFactory);

  theHBHEDigitizer = new HBHEDigitizer(theHBHEResponse, theElectronicsSim, doNoise);
  theHODigitizer = new HODigitizer(theHOResponse, theElectronicsSim, doNoise);
  theHFDigitizer = new HFDigitizer(theHFResponse, theElectronicsSim, doNoise);
  theZDCDigitizer = new ZDCDigitizer(theZDCResponse, theElectronicsSim, doNoise);

  bool doHPDNoise = ps.getParameter<bool>("doHPDNoise");
  if(doHPDNoise) {
    theHPDNoiseGenerator = new HPDNoiseGenerator(theParameterMap); 
    theHBHEDigitizer->setNoiseSignalGenerator(theHPDNoiseGenerator);
  }

  edm::Service<edm::RandomNumberGenerator> rng;
  if ( ! rng.isAvailable()) {
    throw cms::Exception("Configuration")
      << "HcalDigiProducer requires the RandomNumberGeneratorService\n"
         "which is not present in the configuration file.  You must add the service\n"
         "in the configuration file or remove the modules that require it.";
  }

  CLHEP::HepRandomEngine& engine = rng->getEngine();
  theAmplifier->setRandomEngine(engine);
  theElectronicsSim->setRandomEngine(engine);

  hitsProducer_ = ps.getParameter<std::string>("hitsProducer");

}


HcalDigiProducer::~HcalDigiProducer() {
  delete theHBHEDigitizer;
  delete theHODigitizer;
  delete theHFDigitizer;
  delete theZDCDigitizer;
  delete theParameterMap;
  delete theHcalShape;
  delete theHFShape;
  delete theZDCShape;
  delete theHcalIntegratedShape;
  delete theHFIntegratedShape;
  delete theZDCIntegratedShape;
  delete theHBHEResponse;
  delete theHOResponse;
  delete theHFResponse;
  delete theZDCResponse;
  delete theElectronicsSim;
  delete theAmplifier;
  delete theCoderFactory;
  delete theHitCorrection;
  delete theHPDNoiseGenerator;
}


void HcalDigiProducer::produce(edm::Event& e, const edm::EventSetup& eventSetup) {
  // get the appropriate gains, noises, & widths for this event
  edm::ESHandle<HcalDbService> conditions;
  eventSetup.get<HcalDbRecord>().get(conditions);
  theAmplifier->setDbService(conditions.product());
  theCoderFactory->setDbService(conditions.product());
  theParameterMap->setDbService(conditions.product());

  // get the correct geometry
  checkGeometry(eventSetup);
  
  // Step A: Get Inputs
  edm::Handle<CrossingFrame<PCaloHit> > cf, zdccf;

  const std::string hcalHitsName(hitsProducer_+"HcalHits");
  const std::string zdcHitsName(hitsProducer_+"ZDCHITS");

  e.getByLabel("mix", hcalHitsName ,cf);
  e.getByLabel("mix", zdcHitsName , zdccf);

  // test access to SimHits for HcalHits and ZDC hits
  std::auto_ptr<MixCollection<PCaloHit> > col(new MixCollection<PCaloHit>(cf.product()));
  std::auto_ptr<MixCollection<PCaloHit> > colzdc(new MixCollection<PCaloHit>(zdccf.product()));

  if(theHitCorrection != 0)
  {
    theHitCorrection->clear();
    theHitCorrection->fillChargeSums(*col);
    theHitCorrection->fillChargeSums(*colzdc);
  }
  // Step B: Create empty output

  std::auto_ptr<HBHEDigiCollection> hbheResult(new HBHEDigiCollection());
  std::auto_ptr<HODigiCollection> hoResult(new HODigiCollection());
  std::auto_ptr<HFDigiCollection> hfResult(new HFDigiCollection());
  std::auto_ptr<ZDCDigiCollection> zdcResult(new ZDCDigiCollection());

  // Step C: Invoke the algorithm, passing in inputs and getting back outputs.
  theHBHEDigitizer->run(*col, *hbheResult);
  theHODigitizer->run(*col, *hoResult);
  theHFDigitizer->run(*col, *hfResult);
  if(doZDC) {
    //theZDCDigitizer->run(*colzdc, *zdcResult);
  }
  edm::LogInfo("HcalDigiProducer") << "HCAL HBHE digis : " << hbheResult->size();
  edm::LogInfo("HcalDigiProducer") << "HCAL HO digis   : " << hoResult->size();
  edm::LogInfo("HcalDigiProducer") << "HCAL HF digis   : " << hfResult->size();
  edm::LogInfo("HcalDigiProducer") << "HCAL ZDC digis   : " << zdcResult->size();

  // Step D: Put outputs into event
  e.put(hbheResult);
  e.put(hoResult);
  e.put(hfResult);
  e.put(zdcResult);
}


void HcalDigiProducer::checkGeometry(const edm::EventSetup & eventSetup) {
  // TODO find a way to avoid doing this every event
  edm::ESHandle<CaloGeometry> geometry;
  eventSetup.get<CaloGeometryRecord>().get(geometry);
  theHBHEResponse->setGeometry(&*geometry);
  theHOResponse->setGeometry(&*geometry);
  theHFResponse->setGeometry(&*geometry);
  theZDCResponse->setGeometry(&*geometry);

  vector<DetId> hbCells =  geometry->getValidDetIds(DetId::Hcal, HcalBarrel);
  vector<DetId> heCells =  geometry->getValidDetIds(DetId::Hcal, HcalEndcap);
  vector<DetId> hoCells =  geometry->getValidDetIds(DetId::Hcal, HcalOuter);
  vector<DetId> hfCells =  geometry->getValidDetIds(DetId::Hcal, HcalForward);
  vector<DetId> zdcCells = geometry->getValidDetIds(DetId::Calo, HcalZDCDetId::SubdetectorId);

  //std::cout<<"HcalDigiProducer::CheckGeometry number of cells: "<<zdcCells.size()<<std::endl;
  if(zdcCells.size()==0) doZDC = false;
  // combine HB & HE

  vector<DetId> hbheCells = hbCells;
  hbheCells.insert(hbheCells.end(), heCells.begin(), heCells.end());

  theHBHEDigitizer->setDetIds(hbheCells);
  theHODigitizer->setDetIds(hoCells);
  theHFDigitizer->setDetIds(hfCells);
  theZDCDigitizer->setDetIds(zdcCells); 
}


