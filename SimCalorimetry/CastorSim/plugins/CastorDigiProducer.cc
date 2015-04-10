#include "SimCalorimetry/CastorSim/plugins/CastorDigiProducer.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Framework/interface/EDProducer.h"
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
#include "FWCore/Utilities/interface/StreamID.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "DataFormats/HcalDetId/interface/HcalCastorDetId.h"
#include "SimGeneral/MixingModule/interface/PileUpEventPrincipal.h"

CastorDigiProducer::CastorDigiProducer(const edm::ParameterSet& ps, edm::one::EDProducerBase& mixMod, edm::ConsumesCollector& iC) 
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
  theHitsProducerTag = ps.getParameter<edm::InputTag>("hitsProducer");
  iC.consumes<std::vector<PCaloHit> >(theHitsProducerTag);
  
  mixMod.produces<CastorDigiCollection>();

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

void CastorDigiProducer::initializeEvent(edm::Event const&, edm::EventSetup const& eventSetup) {
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

  theCastorDigitizer->initializeHits();
}

void CastorDigiProducer::accumulateCaloHits(std::vector<PCaloHit> const& hcalHits, int bunchCrossing, CLHEP::HepRandomEngine* engine) {
  //fillFakeHits();

  if(theHitCorrection != 0) {
    theHitCorrection->fillChargeSums(hcalHits);
  }
  theCastorDigitizer->add(hcalHits, bunchCrossing, engine);
}

void CastorDigiProducer::accumulate(edm::Event const& e, edm::EventSetup const&) {
  // Step A: Get and accumulate digitized hits 
  edm::Handle<std::vector<PCaloHit> > castorHandle;
  e.getByLabel(theHitsProducerTag, castorHandle);

  accumulateCaloHits(*castorHandle.product(), 0, randomEngine(e.streamID()));
}

void CastorDigiProducer::accumulate(PileUpEventPrincipal const& e, edm::EventSetup const&, edm::StreamID const& streamID) {
  // Step A: Get and accumulate digitized hits 
  edm::Handle<std::vector<PCaloHit> > castorHandle;
  e.getByLabel(theHitsProducerTag, castorHandle);

  accumulateCaloHits(*castorHandle.product(), e.bunchCrossing(), randomEngine(streamID));
}

void CastorDigiProducer::finalizeEvent(edm::Event& e, const edm::EventSetup& eventSetup) {
  // Step B: Create empty output

  std::auto_ptr<CastorDigiCollection> castorResult(new CastorDigiCollection());

  // Step C: Invoke the algorithm, getting back outputs.
  theCastorDigitizer->run(*castorResult, randomEngine(e.streamID()));

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

  theCastorHits.emplace_back(castorDetId.rawId(), 50.0, 0.);
}


void CastorDigiProducer::checkGeometry(const edm::EventSetup & eventSetup) {
  // TODO find a way to avoid doing this every event
  edm::ESHandle<CaloGeometry> geometry;
  eventSetup.get<CaloGeometryRecord>().get(geometry);
  theCastorResponse->setGeometry(&*geometry);

  const std::vector<DetId>& castorCells = geometry->getValidDetIds(DetId::Calo, HcalCastorDetId::SubdetectorId);

  //std::cout<<"CastorDigiProducer::CheckGeometry number of cells: "<<castorCells.size()<<std::endl;
  theCastorDigitizer->setDetIds(castorCells);
}

CLHEP::HepRandomEngine* CastorDigiProducer::randomEngine(edm::StreamID const& streamID) {
  unsigned int index = streamID.value();
  if(index >= randomEngines_.size()) {
    randomEngines_.resize(index + 1, nullptr);
  }
  CLHEP::HepRandomEngine* ptr = randomEngines_[index];
  if(!ptr) {
    edm::Service<edm::RandomNumberGenerator> rng;
    ptr = &rng->getEngine(streamID);
    randomEngines_[index] = ptr;
  }
  return ptr;
}

