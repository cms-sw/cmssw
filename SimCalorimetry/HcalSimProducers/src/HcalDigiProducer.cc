using namespace std;
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/EDProduct/interface/EDProduct.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Handle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "SimCalorimetry/CaloSimAlgos/interface/CaloHitResponse.h"
#include "SimCalorimetry/CaloSimAlgos/interface/CaloTDigitizer.h"
#include "SimCalorimetry/HcalSimAlgos/interface/HcalSimParameterMap.h"
#include "SimCalorimetry/HcalSimAlgos/interface/HcalShape.h"
#include "SimCalorimetry/HcalSimAlgos/interface/HFShape.h"
#include "SimCalorimetry/HcalSimAlgos/interface/HcalElectronicsSim.h"
#include "DataFormats/HcalDigi/interface/HcalDigiCollections.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "CalibFormats/HcalObjects/interface/HcalDbService.h"
#include "CalibFormats/HcalObjects/interface/HcalNominalCoder.h"
#include "CalibFormats/HcalObjects/interface/HcalDbRecord.h"
#include "SimCalorimetry/HcalSimAlgos/interface/HcalNoisifier.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "SimCalorimetry/HcalSimAlgos/interface/HcalDigitizerTraits.h"

using namespace cms;


class HcalDigiProducer : public edm::EDProducer
{
public:

  explicit HcalDigiProducer(const edm::ParameterSet& ps);
  virtual ~HcalDigiProducer();

  /**Produces the EDM products,*/
  virtual void produce(edm::Event& e, const edm::EventSetup& c);

private:
  /// some hits in each subdetector, just for testing purposes
  void fillFakeHits();
  /// make sure the digitizer has the correct list of all cells that
  /// exist in the geometry
  void checkGeometry(const edm::EventSetup& eventSetup);

  /** Reconstruction algorithm*/
  typedef CaloTDigitizer<HBHEDigitizerTraits> HBHEDigitizer;
  typedef CaloTDigitizer<HODigitizerTraits> HODigitizer;
  typedef CaloTDigitizer<HFDigitizerTraits> HFDigitizer;

  HBHEDigitizer * theHBHEDigitizer;
  HODigitizer* theHODigitizer;
  HFDigitizer* theHFDigitizer;

  CaloVSimParameterMap * theParameterMap;
  CaloVShape * theHcalShape;
  CaloVShape * theHFShape;

  CaloHitResponse * theHcalResponse;
  CaloHitResponse * theHFResponse;

  HcalNoisifier * theNoisifier;
  HcalCoder * theCoder;

  HcalElectronicsSim * theElectronicsSim;

  std::vector<CaloHit> theHBHEHits, theHOHits, theHFHits;
};


HcalDigiProducer::HcalDigiProducer(const edm::ParameterSet& ps) {

  produces<HBHEDigiCollection>();
  produces<HODigiCollection>();
  produces<HFDigiCollection>();

  theParameterMap = new HcalSimParameterMap();
  theHcalShape = new HcalShape();
  theHFShape = new HFShape();

  theHcalResponse = new CaloHitResponse(theParameterMap, theHcalShape);
  theHFResponse = new CaloHitResponse(theParameterMap, theHFShape);


  theNoisifier = new HcalNoisifier();
  theCoder = new HcalNominalCoder();
  theElectronicsSim = new HcalElectronicsSim(theNoisifier, theCoder);

  theHBHEDigitizer = new HBHEDigitizer(theHcalResponse, theElectronicsSim);
  theHODigitizer = new HODigitizer(theHcalResponse, theElectronicsSim);
  theHFDigitizer = new HFDigitizer(theHFResponse, theElectronicsSim);

}


HcalDigiProducer::~HcalDigiProducer() {
  delete theParameterMap;
  delete theHcalShape;
  delete theHFShape;
  delete theHcalResponse;
  delete theHFResponse;
  delete theElectronicsSim;
  delete theNoisifier;
  delete theCoder;
}


void HcalDigiProducer::produce(edm::Event& e, const edm::EventSetup& eventSetup) {
std::cout << "DIGIPRODUCER" << std::endl;
  // get the appropriate gains, noises, & widths for this event
  edm::ESHandle<HcalDbService> conditions;
  eventSetup.get<HcalDbRecord>().get(conditions);
  theNoisifier->setDbService(conditions.product());

  // get the correct geometry
  checkGeometry(eventSetup);

  theHBHEHits.clear();
  theHOHits.clear();
  theHFHits.clear();
  // Step A: Get Inputs
  //edm::Handle<vector<CaloHit> > allHits;  //Fancy Event Pointer to CaloTowers
  //e.getByLabel("CaloHits", allHits);           //Set pointer to CaloTowers
  fillFakeHits();


  // Step B: Create empty output

  auto_ptr<HBHEDigiCollection> hbheResult(new HBHEDigiCollection());
  auto_ptr<HODigiCollection> hoResult(new HODigiCollection());
  auto_ptr<HFDigiCollection> hfResult(new HFDigiCollection());


  // Step C: Invoke the algorithm, passing in inputs and getting back outputs.
  theHBHEDigitizer->run(theHBHEHits, *hbheResult);
  theHODigitizer->run(theHOHits, *hoResult);
  theHFDigitizer->run(theHFHits, *hfResult);


  // Step D: Put outputs into event
  e.put(hbheResult);
  e.put(hoResult);
  e.put(hfResult);

}


void HcalDigiProducer::fillFakeHits() {
  HcalDetId barrelDetId(HcalBarrel, 1, 1, 1);
  CaloHit barrelHit(barrelDetId, 0.855, 0., 0);

  HcalDetId endcapDetId(HcalEndcap, 17, 1, 1);
  CaloHit endcapHit(endcapDetId, 0.9, 0., 0);

  HcalDetId outerDetId(HcalOuter, 1, 1, 4);
  CaloHit outerHit(outerDetId, 0.45, 0., 0);

  HcalDetId forwardDetId1(HcalForward, 30, 1, 1);
  CaloHit forwardHit1(forwardDetId1, 35., 0., 0);

  HcalDetId forwardDetId2(HcalForward, 30, 1, 2);
  CaloHit forwardHit2(forwardDetId2, 48., 0., 0);

  theHBHEHits.push_back(barrelHit);
  theHBHEHits.push_back(endcapHit);
  theHOHits.push_back(outerHit);
  theHFHits.push_back(forwardHit1);
  theHFHits.push_back(forwardHit2);
}


void HcalDigiProducer::checkGeometry(const edm::EventSetup & eventSetup) {
  // TODO find a way to avoid doing this every event
  edm::ESHandle<CaloGeometry> geometry;
  eventSetup.get<IdealGeometryRecord>().get(geometry);

  vector<DetId> hbCells =  geometry->getValidDetIds(DetId::Hcal, HcalBarrel);
  vector<DetId> heCells =  geometry->getValidDetIds(DetId::Hcal, HcalEndcap);
  vector<DetId> hoCells =  geometry->getValidDetIds(DetId::Hcal, HcalOuter);
  vector<DetId> hfCells =  geometry->getValidDetIds(DetId::Hcal, HcalForward);

  // combine HB & HE
  vector<DetId> hbheCells = hbCells;
  hbheCells.insert(hbheCells.end(), heCells.begin(), heCells.end());

  theHBHEDigitizer->setDetIds(hbheCells);
  theHODigitizer->setDetIds(hoCells);
  theHFDigitizer->setDetIds(hfCells);
}


#include "PluginManager/ModuleDef.h"

DEFINE_SEAL_MODULE();
DEFINE_ANOTHER_FWK_MODULE(HcalDigiProducer)

