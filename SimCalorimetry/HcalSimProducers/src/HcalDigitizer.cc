#include "SimCalorimetry/HcalSimProducers/interface/HcalDigitizer.h"
#include "SimCalorimetry/HcalSimProducers/src/HcalTestHitGenerator.h"
#include "SimDataFormats/CaloHit/interface/PCaloHitContainer.h"
#include "SimCalorimetry/HcalSimAlgos/interface/HcalSimParameterMap.h"
#include "SimCalorimetry/HcalSimAlgos/interface/HcalShape.h"
#include "SimCalorimetry/HcalSimAlgos/interface/HFShape.h"
#include "SimCalorimetry/HcalSimAlgos/interface/ZDCShape.h"
#include "SimCalorimetry/HcalSimAlgos/interface/HcalElectronicsSim.h"
#include "SimCalorimetry/CaloSimAlgos/interface/CaloHitResponse.h"
#include "SimCalorimetry/HcalSimAlgos/interface/HcalAmplifier.h"
#include "SimCalorimetry/HcalSimAlgos/interface/HcalCoderFactory.h"
#include "SimCalorimetry/HcalSimAlgos/interface/HcalHitCorrection.h"
#include "SimCalorimetry/HcalSimAlgos/interface/HcalSimParameterMap.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
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
#include <boost/foreach.hpp>
using namespace std;


HcalDigitizer::HcalDigitizer(const edm::ParameterSet& ps) 
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
  theHBHEAmplifier(0),
  theHFAmplifier(0),
  theHOAmplifier(0),
  theZDCAmplifier(0),
  theCoderFactory(0),
  theHBHEElectronicsSim(0),
  theHFElectronicsSim(0),
  theHOElectronicsSim(0),
  theZDCElectronicsSim(0),
  theHBHEHitFilter(),
  theHFHitFilter(ps.getParameter<bool>("doHFWindow")),
  theHOHitFilter(),
  theZDCHitFilter(),
  theHitCorrection(0),
  theNoiseGenerator(0),
  theNoiseHitGenerator(0),
  theHBHEDigitizer(0),
  theHODigitizer(0),
  theHFDigitizer(0),
  theZDCDigitizer(0),
  isZDC(true),
  isHCAL(true),
  zdcgeo(true),
  hbhegeo(true),
  hogeo(true),
  hfgeo(true)
{
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
  bool doEmpty = ps.getParameter<bool>("doEmpty");
  // need to make copies, because they might get different noise generators
  theHBHEAmplifier = new HcalAmplifier(theParameterMap, doNoise);
  theHFAmplifier = new HcalAmplifier(theParameterMap, doNoise);
  theHOAmplifier = new HcalAmplifier(theParameterMap, doNoise);
  theZDCAmplifier = new HcalAmplifier(theParameterMap, doNoise);
  theCoderFactory = new HcalCoderFactory(HcalCoderFactory::DB);
  theHBHEElectronicsSim = new HcalElectronicsSim(theHBHEAmplifier, theCoderFactory);
  theHFElectronicsSim = new HcalElectronicsSim(theHFAmplifier, theCoderFactory);
  theHOElectronicsSim = new HcalElectronicsSim(theHOAmplifier, theCoderFactory);
  theZDCElectronicsSim = new HcalElectronicsSim(theZDCAmplifier, theCoderFactory);

  theHBHEDigitizer = new HBHEDigitizer(theHBHEResponse, theHBHEElectronicsSim, doEmpty);
  theHODigitizer = new HODigitizer(theHOResponse, theHOElectronicsSim, doEmpty);
  theHFDigitizer = new HFDigitizer(theHFResponse, theHFElectronicsSim, doEmpty);
  theZDCDigitizer = new ZDCDigitizer(theZDCResponse, theZDCElectronicsSim, doEmpty);

  bool doHPDNoise = ps.getParameter<bool>("doHPDNoise");
  if(doHPDNoise) {
    //edm::ParameterSet hpdNoisePset = ps.getParameter<edm::ParameterSet>("HPDNoiseLibrary");
    theNoiseGenerator = new HPDNoiseGenerator(ps, theParameterMap); 
    theHBHEDigitizer->setNoiseSignalGenerator(theNoiseGenerator);
  }

  if(ps.getParameter<bool>("injectTestHits") ){
    theNoiseHitGenerator = new HcalTestHitGenerator(ps);
    theHBHEDigitizer->setNoiseHitGenerator(theNoiseHitGenerator);
    theHODigitizer->setNoiseHitGenerator(theNoiseHitGenerator);
    theHFDigitizer->setNoiseHitGenerator(theNoiseHitGenerator);
    theZDCDigitizer->setNoiseHitGenerator(theNoiseHitGenerator);
  }

  edm::Service<edm::RandomNumberGenerator> rng;
  if ( ! rng.isAvailable()) {
    throw cms::Exception("Configuration")
      << "HcalDigitizer requires the RandomNumberGeneratorService\n"
         "which is not present in the configuration file.  You must add the service\n"
         "in the configuration file or remove the modules that require it.";
  }

  CLHEP::HepRandomEngine& engine = rng->getEngine();
  theHBHEAmplifier->setRandomEngine(engine);
  theHFAmplifier->setRandomEngine(engine);
  theHOAmplifier->setRandomEngine(engine);
  theZDCAmplifier->setRandomEngine(engine);

  theHBHEElectronicsSim->setRandomEngine(engine);
  theHFElectronicsSim->setRandomEngine(engine);
  theHOElectronicsSim->setRandomEngine(engine);
  theZDCElectronicsSim->setRandomEngine(engine);

  if (theHitCorrection!=0) theHitCorrection->setRandomEngine(engine);

  hitsProducer_ = ps.getParameter<std::string>("hitsProducer");

}


HcalDigitizer::~HcalDigitizer() {
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
  delete theHBHEElectronicsSim;
  delete theHFElectronicsSim;
  delete theHOElectronicsSim;
  delete theZDCElectronicsSim;
  delete theHBHEAmplifier;
  delete theHFAmplifier;
  delete theHOAmplifier;
  delete theZDCAmplifier;
  delete theCoderFactory;
  delete theHitCorrection;
  delete theNoiseGenerator;
}


void HcalDigitizer::setHBHENoiseSignalGenerator(CaloVNoiseSignalGenerator * noiseGenerator)
{
  noiseGenerator->setParameterMap(theParameterMap);
  theHBHEDigitizer->setNoiseSignalGenerator(noiseGenerator);
  theHBHEAmplifier->setNoiseSignalGenerator(noiseGenerator);
}

void HcalDigitizer::setHFNoiseSignalGenerator(CaloVNoiseSignalGenerator * noiseGenerator)
{
  noiseGenerator->setParameterMap(theParameterMap);
  theHFDigitizer->setNoiseSignalGenerator(noiseGenerator);
  theHFAmplifier->setNoiseSignalGenerator(noiseGenerator);
}

void HcalDigitizer::setHONoiseSignalGenerator(CaloVNoiseSignalGenerator * noiseGenerator)
{
  noiseGenerator->setParameterMap(theParameterMap);
  theHODigitizer->setNoiseSignalGenerator(noiseGenerator);
  theHOAmplifier->setNoiseSignalGenerator(noiseGenerator);
}

void HcalDigitizer::setZDCNoiseSignalGenerator(CaloVNoiseSignalGenerator * noiseGenerator)
{
  noiseGenerator->setParameterMap(theParameterMap);
  theZDCAmplifier->setNoiseSignalGenerator(noiseGenerator);
  theZDCAmplifier->setNoiseSignalGenerator(noiseGenerator);
}


void HcalDigitizer::produce(edm::Event& e, const edm::EventSetup& eventSetup) {
  // get the appropriate gains, noises, & widths for this event
  edm::ESHandle<HcalDbService> conditions;
  eventSetup.get<HcalDbRecord>().get(conditions);
  theHBHEAmplifier->setDbService(conditions.product());
  theHFAmplifier->setDbService(conditions.product());
  theHOAmplifier->setDbService(conditions.product());
  theZDCAmplifier->setDbService(conditions.product());

  theCoderFactory->setDbService(conditions.product());
  theParameterMap->setDbService(conditions.product());

  // get the correct geometry
  checkGeometry(eventSetup);
  
  // Step A: Get Inputs
  edm::Handle<CrossingFrame<PCaloHit> > cf, zdccf;

  // test access to SimHits for HcalHits and ZDC hits
  const std::string zdcHitsName(hitsProducer_+"ZDCHITS");
  e.getByLabel("mix", zdcHitsName , zdccf);
  MixCollection<PCaloHit> * colzdc = 0 ;
  if(zdccf.isValid()){
    colzdc = new MixCollection<PCaloHit>(zdccf.product());
  }else{
    edm::LogInfo("HcalDigitizer") << "We don't have ZDC hit collection available ";
    isZDC = false;
  }

  const std::string hcalHitsName(hitsProducer_+"HcalHits");
  e.getByLabel("mix", hcalHitsName ,cf);
  MixCollection<PCaloHit> * col = 0 ;
  if(cf.isValid()){
    col = new MixCollection<PCaloHit>(cf.product());
  }else{
    edm::LogInfo("HcalDigitizer") << "We don't have HCAL hit collection available ";
    isHCAL = false;
  }

  if(theHitCorrection != 0)
  {
    theHitCorrection->clear();
    if(isHCAL)
      theHitCorrection->fillChargeSums(*col);
    if(isZDC)
      theHitCorrection->fillChargeSums(*colzdc);
  }
  // Step B: Create empty output

  std::auto_ptr<HBHEDigiCollection> hbheResult(new HBHEDigiCollection());
  std::auto_ptr<HODigiCollection> hoResult(new HODigiCollection());
  std::auto_ptr<HFDigiCollection> hfResult(new HFDigiCollection());
  std::auto_ptr<ZDCDigiCollection> zdcResult(new ZDCDigiCollection());

  // Step C: Invoke the algorithm, passing in inputs and getting back outputs.
  if(isHCAL&&hbhegeo)
    theHBHEDigitizer->run(*col, *hbheResult);
  if(isHCAL&&hogeo)
    theHODigitizer->run(*col, *hoResult);
  if(isHCAL&&hfgeo)
    theHFDigitizer->run(*col, *hfResult);  
  if(isZDC&&zdcgeo) 
    theZDCDigitizer->run(*colzdc, *zdcResult);
  
  edm::LogInfo("HcalDigitizer") << "HCAL HBHE digis : " << hbheResult->size();
  edm::LogInfo("HcalDigitizer") << "HCAL HO digis   : " << hoResult->size();
  edm::LogInfo("HcalDigitizer") << "HCAL HF digis   : " << hfResult->size();
  edm::LogInfo("HcalDigitizer") << "HCAL ZDC digis   : " << zdcResult->size();

  // Step D: Put outputs into event
  e.put(hbheResult);
  e.put(hoResult);
  e.put(hfResult);
  e.put(zdcResult);
}


void HcalDigitizer::checkGeometry(const edm::EventSetup & eventSetup) {
  // TODO find a way to avoid doing this every event
  edm::ESHandle<CaloGeometry> geometry;
  eventSetup.get<CaloGeometryRecord>().get(geometry);
  theHBHEResponse->setGeometry(&*geometry);
  theHOResponse->setGeometry(&*geometry);
  theHFResponse->setGeometry(&*geometry);
  theZDCResponse->setGeometry(&*geometry);

  const vector<DetId>& hbCells =  geometry->getValidDetIds(DetId::Hcal, HcalBarrel);
  const vector<DetId>& heCells =  geometry->getValidDetIds(DetId::Hcal, HcalEndcap);
  const vector<DetId>& hoCells =  geometry->getValidDetIds(DetId::Hcal, HcalOuter);
  const vector<DetId>& hfCells =  geometry->getValidDetIds(DetId::Hcal, HcalForward);
  const vector<DetId>& zdcCells = geometry->getValidDetIds(DetId::Calo, HcalZDCDetId::SubdetectorId);
  //const vector<DetId>& hcalTrigCells = geometry->getValidDetIds(DetId::Hcal, HcalTriggerTower);
  //const vector<DetId>& hcalCalib = geometry->getValidDetIds(DetId::Calo, HcalCastorDetId::SubdetectorId);
  //std::cout<<"HcalDigitizer::CheckGeometry number of cells: "<<zdcCells.size()<<std::endl;
  if(zdcCells.size()==0) zdcgeo = false;
  if(hbCells.size()==0&&heCells.size()==0) hbhegeo = false;
  if(hoCells.size()==0) hogeo = false;
  if(hfCells.size()==0) hfgeo = false;
  // combine HB & HE



  vector<DetId> hbheCells = hbCells;
  hbheCells.insert(hbheCells.end(), heCells.begin(), heCells.end());

  theHBHEDigitizer->setDetIds(hbheCells);
  theHODigitizer->setDetIds(hoCells);
  theHFDigitizer->setDetIds(hfCells);
  theZDCDigitizer->setDetIds(zdcCells); 
}


