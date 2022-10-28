#include "CalibFormats/CastorObjects/interface/CastorDbRecord.h"
#include "CalibFormats/CastorObjects/interface/CastorDbService.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/HcalDetId/interface/HcalCastorDetId.h"
#include "DataFormats/HcalDigi/interface/HcalDigiCollections.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESWatcher.h"
#include "FWCore/Framework/interface/FrameworkfwdMostUsed.h"
#include "FWCore/Framework/interface/ProducesCollector.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/RandomNumberGenerator.h"
#include "FWCore/Utilities/interface/StreamID.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "SimCalorimetry/CaloSimAlgos/interface/CaloShapeIntegrator.h"
#include "SimCalorimetry/CaloSimAlgos/interface/CaloTDigitizer.h"
#include "SimCalorimetry/CaloSimAlgos/interface/CaloHitResponse.h"
#include "SimCalorimetry/CaloSimAlgos/interface/CaloTDigitizer.h"
#include "SimCalorimetry/CastorSim/interface/CastorAmplifier.h"
#include "SimCalorimetry/CastorSim/interface/CastorCoderFactory.h"
#include "SimCalorimetry/CastorSim/interface/CastorDigitizerTraits.h"
#include "SimCalorimetry/CastorSim/interface/CastorElectronicsSim.h"
#include "SimCalorimetry/CastorSim/interface/CastorHitCorrection.h"
#include "SimCalorimetry/CastorSim/interface/CastorHitFilter.h"
#include "SimCalorimetry/CastorSim/interface/CastorShape.h"
#include "SimCalorimetry/CastorSim/interface/CastorSimParameterMap.h"
#include "SimDataFormats/CrossingFrame/interface/CrossingFrame.h"
#include "SimDataFormats/CaloHit/interface/PCaloHitContainer.h"
#include "SimGeneral/MixingModule/interface/DigiAccumulatorMixMod.h"
#include "SimDataFormats/CrossingFrame/interface/MixCollection.h"
#include "SimGeneral/MixingModule/interface/PileUpEventPrincipal.h"

#include <vector>

namespace CLHEP {
  class HepRandomEngine;
}

class CastorDigiProducer : public DigiAccumulatorMixMod {
public:
  explicit CastorDigiProducer(const edm::ParameterSet &ps, edm::ProducesCollector, edm::ConsumesCollector &iC);
  ~CastorDigiProducer() override;

  void initializeEvent(edm::Event const &e, edm::EventSetup const &c) override;
  void accumulate(edm::Event const &e, edm::EventSetup const &c) override;
  void accumulate(PileUpEventPrincipal const &e, edm::EventSetup const &c, edm::StreamID const &) override;
  void finalizeEvent(edm::Event &e, edm::EventSetup const &c) override;

private:
  void accumulateCaloHits(std::vector<PCaloHit> const &, int bunchCrossing);

  /// fills the vectors for each subdetector
  void sortHits(const edm::PCaloHitContainer &hits);
  /// some hits in each subdetector, just for testing purposes
  void fillFakeHits();
  /// make sure the digitizer has the correct list of all cells that
  /// exist in the geometry
  void checkGeometry(const edm::EventSetup &eventSetup);

  const edm::ESGetToken<CastorDbService, CastorDbRecord> theConditionsToken;
  const edm::ESGetToken<CaloGeometry, CaloGeometryRecord> theGeometryToken;
  edm::ESWatcher<CaloGeometryRecord> theGeometryWatcher;
  const edm::InputTag theHitsProducerTag;
  const edm::EDGetTokenT<std::vector<PCaloHit>> hitToken_;

  /** Reconstruction algorithm*/
  typedef CaloTDigitizer<CastorDigitizerTraits> CastorDigitizer;

  CastorSimParameterMap *theParameterMap;
  CaloVShape *theCastorShape;
  CaloVShape *theCastorIntegratedShape;

  CaloHitResponse *theCastorResponse;

  CastorAmplifier *theAmplifier;
  CastorCoderFactory *theCoderFactory;
  CastorElectronicsSim *theElectronicsSim;

  CastorHitFilter theCastorHitFilter;

  CastorHitCorrection *theHitCorrection;

  CastorDigitizer *theCastorDigitizer;

  std::vector<PCaloHit> theCastorHits;

  CLHEP::HepRandomEngine *randomEngine_ = nullptr;
};

CastorDigiProducer::CastorDigiProducer(const edm::ParameterSet &ps,
                                       edm::ProducesCollector producesCollector,
                                       edm::ConsumesCollector &iC)
    : theConditionsToken(iC.esConsumes()),
      theGeometryToken(iC.esConsumes()),
      theHitsProducerTag(ps.getParameter<edm::InputTag>("hitsProducer")),
      hitToken_(iC.consumes<std::vector<PCaloHit>>(theHitsProducerTag)),
      theParameterMap(new CastorSimParameterMap(ps)),
      theCastorShape(new CastorShape()),
      theCastorIntegratedShape(new CaloShapeIntegrator(theCastorShape)),
      theCastorResponse(new CaloHitResponse(theParameterMap, theCastorIntegratedShape)),
      theAmplifier(nullptr),
      theCoderFactory(nullptr),
      theElectronicsSim(nullptr),
      theHitCorrection(nullptr),
      theCastorDigitizer(nullptr),
      theCastorHits() {
  producesCollector.produces<CastorDigiCollection>();

  theCastorResponse->setHitFilter(&theCastorHitFilter);

  bool doTimeSlew = ps.getParameter<bool>("doTimeSlew");
  if (doTimeSlew) {
    // no time slewing for HF
    theCastorResponse->setHitCorrection(theHitCorrection);
  }

  bool doNoise = ps.getParameter<bool>("doNoise");
  theAmplifier = new CastorAmplifier(theParameterMap, doNoise);
  theCoderFactory = new CastorCoderFactory(CastorCoderFactory::DB);
  theElectronicsSim = new CastorElectronicsSim(theAmplifier, theCoderFactory);

  theCastorDigitizer = new CastorDigitizer(theCastorResponse, theElectronicsSim, doNoise);

  edm::Service<edm::RandomNumberGenerator> rng;
  if (!rng.isAvailable()) {
    throw cms::Exception("Configuration") << "CastorDigiProducer requires the RandomNumberGeneratorService\n"
                                             "which is not present in the configuration file.  You must add the "
                                             "service\n"
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

void CastorDigiProducer::initializeEvent(edm::Event const &event, edm::EventSetup const &eventSetup) {
  // get the appropriate gains, noises, & widths for this event
  const CastorDbService *conditions = &eventSetup.getData(theConditionsToken);
  theAmplifier->setDbService(conditions);
  theCoderFactory->setDbService(conditions);
  theParameterMap->setDbService(conditions);

  // Cache random number engine
  edm::Service<edm::RandomNumberGenerator> rng;
  randomEngine_ = &rng->getEngine(event.streamID());

  edm::LogInfo("CastorDigiProducer") << "checking the geometry...";

  // get the correct geometry
  checkGeometry(eventSetup);

  theCastorHits.clear();

  theCastorDigitizer->initializeHits();
}

void CastorDigiProducer::accumulateCaloHits(std::vector<PCaloHit> const &hcalHits, int bunchCrossing) {
  // fillFakeHits();

  if (theHitCorrection != nullptr) {
    theHitCorrection->fillChargeSums(hcalHits);
  }
  theCastorDigitizer->add(hcalHits, bunchCrossing, randomEngine_);
}

void CastorDigiProducer::accumulate(edm::Event const &e, edm::EventSetup const &) {
  // Step A: Get and accumulate digitized hits
  const edm::Handle<std::vector<PCaloHit>> &castorHandle = e.getHandle(hitToken_);

  accumulateCaloHits(*castorHandle.product(), 0);
}

void CastorDigiProducer::accumulate(PileUpEventPrincipal const &e,
                                    edm::EventSetup const &,
                                    edm::StreamID const &streamID) {
  // Step A: Get and accumulate digitized hits
  edm::Handle<std::vector<PCaloHit>> castorHandle;
  e.getByLabel(theHitsProducerTag, castorHandle);

  accumulateCaloHits(*castorHandle.product(), e.bunchCrossing());
}

void CastorDigiProducer::finalizeEvent(edm::Event &e, const edm::EventSetup &eventSetup) {
  // Step B: Create empty output

  std::unique_ptr<CastorDigiCollection> castorResult(new CastorDigiCollection());

  // Step C: Invoke the algorithm, getting back outputs.
  theCastorDigitizer->run(*castorResult, randomEngine_);

  edm::LogInfo("CastorDigiProducer") << "HCAL/Castor digis   : " << castorResult->size();

  // Step D: Put outputs into event
  e.put(std::move(castorResult));

  randomEngine_ = nullptr;  // to prevent access outside event
}

void CastorDigiProducer::sortHits(const edm::PCaloHitContainer &hits) {
  for (edm::PCaloHitContainer::const_iterator hitItr = hits.begin(); hitItr != hits.end(); ++hitItr) {
    DetId detId = hitItr->id();
    if (detId.det() == DetId::Calo && detId.subdetId() == HcalCastorDetId::SubdetectorId) {
      theCastorHits.push_back(*hitItr);
    } else {
      edm::LogError("CastorDigiProducer") << "Bad Hit subdetector " << detId.subdetId();
    }
  }
}

void CastorDigiProducer::fillFakeHits() {
  HcalCastorDetId castorDetId(HcalCastorDetId::Section(2), true, 1, 1);

  theCastorHits.emplace_back(castorDetId.rawId(), 50.0, 0.);
}

void CastorDigiProducer::checkGeometry(const edm::EventSetup &eventSetup) {
  if (theGeometryWatcher.check(eventSetup)) {
    const CaloGeometry *geometry = &eventSetup.getData(theGeometryToken);
    theCastorResponse->setGeometry(geometry);

    const std::vector<DetId> &castorCells = geometry->getValidDetIds(DetId::Calo, HcalCastorDetId::SubdetectorId);

    // // edm::LogInfo("CastorDigiProducer") << "CastorDigiProducer::CheckGeometry number of cells:" << castorCells.size()
    ;
    theCastorDigitizer->setDetIds(castorCells);
  }
}

#include "FWCore/Framework/interface/MakerMacros.h"
#include "SimGeneral/MixingModule/interface/DigiAccumulatorMixModFactory.h"

DEFINE_DIGI_ACCUMULATOR(CastorDigiProducer);
