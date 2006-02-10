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
#include "SimCalorimetry/EcalSimAlgos/interface/EcalSimParameterMap.h"
#include "SimCalorimetry/EcalSimAlgos/interface/EcalShape.h"
#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"
#include "SimCalorimetry/EcalSimAlgos/interface/EcalElectronicsSim.h"
#include "SimCalorimetry/EcalSimAlgos/interface/EcalCoder.h"
#include "CondFormats/DataRecord/interface/EcalPedestalsRcd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "SimCalorimetry/EcalSimAlgos/interface/EcalDigitizerTraits.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "CondFormats/EcalObjects/interface/EcalPedestals.h"
#include "SimDataFormats/CaloHit/interface/PCaloHitContainer.h"
#include "SimDataFormats/CrossingFrame/interface/CrossingFrame.h"
#include "SimDataFormats/CrossingFrame/interface/MixCollection.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/Provenance.h"

#include "SimCalorimetry/CaloSimAlgos/interface/CaloDigiCollectionSorter.h"

class EcalDigiProducer : public edm::EDProducer
{
public:

  // The following is not yet used, but will be the primary
  // constructor when the parameter set system is available.
  //
  explicit EcalDigiProducer(const edm::ParameterSet& params);
  virtual ~EcalDigiProducer();

  /**Produces the EDM products,*/
  virtual void produce(edm::Event& event, const edm::EventSetup& eventSetup);

private:
  // some hits in each subdetector, just for testing purposes
  void fillFakeHits();

  void checkGeometry(const edm::EventSetup & eventSetup);
  void checkCalibrations(const edm::EventSetup & eventSetup);

  /** Reconstruction algorithm*/
  typedef CaloTDigitizer<EBDigitizerTraits> EBDigitizer;
  typedef CaloTDigitizer<EEDigitizerTraits> EEDigitizer;

  EBDigitizer * theBarrelDigitizer;
  EEDigitizer * theEndcapDigitizer;

  const EcalSimParameterMap * theParameterMap;
  const CaloVShape * theEcalShape;

  CaloHitResponse * theEcalResponse;

  EcalElectronicsSim * theElectronicsSim;
  EcalCoder * theCoder;

  std::vector<DetId> theBarrelDets;
  std::vector<DetId> theEndcapDets;
  EcalPedestals thePedestals;
  void setupFakePedestals();

};


EcalDigiProducer::EcalDigiProducer(const edm::ParameterSet& params) 
{
  produces<EBDigiCollection>();
  produces<EEDigiCollection>();

  theParameterMap = new EcalSimParameterMap();
  theEcalShape = new EcalShape();

  theEcalResponse = new CaloHitResponse(theParameterMap, theEcalShape);
  
  bool addNoise = params.getUntrackedParameter<bool>("doNoise" , false);
  theCoder = new EcalCoder(addNoise);
  theElectronicsSim = new EcalElectronicsSim(theParameterMap, theCoder);

  theBarrelDigitizer = new EBDigitizer(theEcalResponse, theElectronicsSim, addNoise);
  theEndcapDigitizer = new EEDigitizer(theEcalResponse, theElectronicsSim, addNoise);

  // temporary hacks for missing pieces
  setupFakePedestals();
}



void EcalDigiProducer::setupFakePedestals() 
{
  thePedestals.m_pedestals.clear();
  // make pedestals for each of these
  EcalPedestals::Item item;
  item.mean_x1 = 0.;
  item.rms_x1 = 0.;
  item.mean_x6 = 0.;
  item.rms_x6 = 0.;
  item.mean_x12 = 0.;
  item.rms_x12 = 0.;

  // make one vector of all det ids
  vector<DetId> detIds(theBarrelDets.begin(), theBarrelDets.end());
  detIds.insert(detIds.end(), theEndcapDets.begin(), theEndcapDets.end());

  // make a pedestal entry for each one 
  for(std::vector<DetId>::const_iterator detItr = detIds.begin();
       detItr != detIds.end(); ++detItr)
  {
    pair<int, EcalPedestals::Item> entry(detItr->rawId(), item);
    thePedestals.m_pedestals.insert(entry);
  }

  theCoder->setPedestals(&thePedestals);
}


EcalDigiProducer::~EcalDigiProducer() 
{
  delete theParameterMap;
  delete theEcalShape;
  delete theEcalResponse;
  delete theCoder;
}


void EcalDigiProducer::produce(edm::Event& event, const edm::EventSetup& eventSetup) 
{
//  edm::ESHandle<EcalPedestals> pedHandle;
//  eventSetup.get<EcalPedestalsRcd>().get( pedHandle );
//  theCoder->setPedestals(pedHandle.product());

  // Step A: Get Inputs

  checkCalibrations(eventSetup);
  checkGeometry(eventSetup);

  // Get input
  edm::Handle<CrossingFrame> crossingFrame;
  event.getByType(crossingFrame);

  // test access to SimHits
  const std::string barrelHitsName("EcalHitsEB");
  const std::string endcapHitsName("EcalHitsEE");

  std::auto_ptr<MixCollection<PCaloHit> > 
    barrelHits( new MixCollection<PCaloHit>(crossingFrame.product(), barrelHitsName) );
//  std::auto_ptr<MixCollection<PCaloHit> > 
//    endcapHits( new MixCollection<PCaloHit>(crossingFrame.product(),endcapHitsName) );

  // Step B: Create empty output
  auto_ptr<EBDigiCollection> barrelResult(new EBDigiCollection());
//  auto_ptr<EEDigiCollection> endcapResult(new EEDigiCollection());

  // run the algorithm
  theBarrelDigitizer->run(*barrelHits, *barrelResult);

  edm::LogInfo("EcalDigiProducer") << "EB Digis: " << barrelResult->size();

//  theEndcapDigitizer->run(*endcapHits, *endcapResult);

  CaloDigiCollectionSorter sorter(5);
  std::vector<EBDataFrame> sortedDigis = sorter.sortedVector(*barrelResult);
  std::cout << "Top 10 EB digis" << std::endl;
  for(int i = 0; i < std::min(10,(int) sortedDigis.size()); ++i) 
   {
    std::cout << sortedDigis[i];
   }
  // Step D: Put outputs into event
  event.put(barrelResult);
//  event.put(endcapResult);

}



void  EcalDigiProducer::checkCalibrations(const edm::EventSetup & eventSetup) 
{}


void EcalDigiProducer::checkGeometry(const edm::EventSetup & eventSetup) 
{
  // TODO find a way to avoid doing this every event
  edm::ESHandle<CaloGeometry> geometry;
  eventSetup.get<IdealGeometryRecord>().get(geometry);

  theEcalResponse->setGeometry(&*geometry);

  theBarrelDets.clear();
  theEndcapDets.clear();

  theBarrelDets =  geometry->getValidDetIds(DetId::Ecal, EcalBarrel);
  theEndcapDets =  geometry->getValidDetIds(DetId::Ecal, EcalEndcap);

  //PG FIXME
  std::cout << "deb geometry: "
            << "\t barrel: " << theBarrelDets.size () 
            << "\t endcap: " << theEndcapDets.size () 
            << std::endl ;

  theBarrelDigitizer->setDetIds(theBarrelDets);
  theEndcapDigitizer->setDetIds(theEndcapDets);

  setupFakePedestals();
}



#include "PluginManager/ModuleDef.h"

DEFINE_SEAL_MODULE () ;
DEFINE_ANOTHER_FWK_MODULE (EcalDigiProducer) ;

