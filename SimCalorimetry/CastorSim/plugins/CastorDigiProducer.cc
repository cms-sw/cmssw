#include "SimCalorimetry/CastorSim/plugins/CastorDigiProducer.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "SimCalorimetry/CaloSimAlgos/interface/CaloTDigitizer.h"
#include "SimCalorimetry/CaloSimAlgos/interface/CaloShapeIntegrator.h"
#include "DataFormats/HcalDigi/interface/HcalDigiCollections.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "CalibFormats/CastorObjects/interface/CastorDbService.h"
#include "CalibFormats/CastorObjects/interface/CastorDbRecord.h"
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
  edm::ESHandle<CastorDbService> conditions;
  eventSetup.get<CastorDbRecord>().get(conditions);
  theAmplifier->setDbService(conditions.product());
  theCoderFactory->setDbService(conditions.product());
  theParameterMap->setDbService(conditions.product());

edm::LogInfo("CastorDigiProducer") << "checking the geometry...";

  // get the correct geometry
checkGeometry(eventSetup);
  
theCastorHits.clear();

  // Step A: Get Inputs
//edm::Handle<edm::PCaloHitContainer> castorcf;
edm::Handle<CrossingFrame<PCaloHit> > castorcf;
e.getByLabel("mix", "g4SimHitsCastorFI", castorcf);

  // test access to SimHits for HcalHits and ZDC hits
std::auto_ptr<MixCollection<PCaloHit> > colcastor(new MixCollection<PCaloHit>(castorcf.product()));

  //fillFakeHits();

if(theHitCorrection != 0)
  {
theHitCorrection->fillChargeSums(*colcastor);
  }
  // Step B: Create empty output

std::auto_ptr<CastorDigiCollection> castorResult(new CastorDigiCollection());

  // Step C: Invoke the algorithm, passing in inputs and getting back outputs.
theCastorDigitizer->run(*colcastor, *castorResult);

edm::LogInfo("CastorDigiProducer") << "HCAL/Castor digis   : " << castorResult->size();

  // Step D: Put outputs into event
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
HcalCastorDetId castorDetId(HcalCastorDetId::Section(2),true,1,1);
PCaloHit castorHit(castorDetId.rawId(), 50.0, 0.);

theCastorHits.push_back(castorHit);
}


void CastorDigiProducer::checkGeometry(const edm::EventSetup & eventSetup) {
  // TODO find a way to avoid doing this every event
edm::ESHandle<CaloGeometry> geometry;
eventSetup.get<CaloGeometryRecord>().get(geometry);
theCastorResponse->setGeometry(&*geometry);

const vector<DetId>& castorCells = geometry->getValidDetIds(DetId::Calo, HcalCastorDetId::SubdetectorId);

//std::cout<<"CastorDigiProducer::CheckGeometry number of cells: "<<castorCells.size()<<std::endl;
theCastorDigitizer->setDetIds(castorCells);
}


