#include "SimCalorimetry/EcalSimProducers/interface/EcalDigiProducer.h"

EcalDigiProducer::EcalDigiProducer(const edm::ParameterSet& params) 
:  theGeometry(0)
{
  produces<EBDigiCollection>();
  produces<EEDigiCollection>();
  produces<ESDigiCollection>();

  theParameterMap = new EcalSimParameterMap();
  theEcalShape = new EcalShape();

  theESShape = new ESShape();

  theEcalResponse = new CaloHitResponse(theParameterMap, theEcalShape);
  theESResponse = new CaloHitResponse(theParameterMap, theESShape);
  
  bool addNoise = params.getUntrackedParameter<bool>("doNoise" , false);
  theCoder = new EcalCoder(addNoise);
  theElectronicsSim = new EcalElectronicsSim(theParameterMap, theCoder);
  theESElectronicsSim = new ESElectronicsSim(15);

  theBarrelDigitizer = new EBDigitizer(theEcalResponse, theElectronicsSim, addNoise);
  theEndcapDigitizer = new EEDigitizer(theEcalResponse, theElectronicsSim, addNoise);
  theESDigitizer = new ESDigitizer(theESResponse, theESElectronicsSim, false);

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
    thePedestals.m_pedestals[detItr->rawId()] = item;
  }

  theCoder->setPedestals(&thePedestals);
}


EcalDigiProducer::~EcalDigiProducer() 
{
  delete theParameterMap;
  delete theEcalShape;
  delete theESShape;
  delete theEcalResponse;
  delete theESResponse;
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
  const std::string ESHitsName("EcalHitsES");

  std::auto_ptr<MixCollection<PCaloHit> > 
    barrelHits( new MixCollection<PCaloHit>(crossingFrame.product(), barrelHitsName) );
  std::auto_ptr<MixCollection<PCaloHit> > 
    endcapHits( new MixCollection<PCaloHit>(crossingFrame.product(),endcapHitsName) );
  std::auto_ptr<MixCollection<PCaloHit> >
    ESHits( new MixCollection<PCaloHit>(crossingFrame.product(), ESHitsName) ); 

  // Step B: Create empty output
  auto_ptr<EBDigiCollection> barrelResult(new EBDigiCollection());
  auto_ptr<EEDigiCollection> endcapResult(new EEDigiCollection());
  auto_ptr<ESDigiCollection> ESResult(new ESDigiCollection());

  // run the algorithm
  theBarrelDigitizer->run(*barrelHits, *barrelResult);

  edm::LogInfo("EcalDigiProducer") << "EB Digis: " << barrelResult->size();

  theEndcapDigitizer->run(*endcapHits, *endcapResult);

  theESDigitizer->run(*ESHits, *ESResult);
  edm::LogInfo("EcalDigiProducer") << "ES Digis: " << ESResult->size();

  CaloDigiCollectionSorter sorter(5);
  std::vector<EBDataFrame> sortedDigis = sorter.sortedVector(*barrelResult);
  std::cout << "Top 10 EB digis" << std::endl;
  for(int i = 0; i < std::min(10,(int) sortedDigis.size()); ++i) 
   {
    std::cout << sortedDigis[i];
   }

  // Step D: Put outputs into event
  event.put(barrelResult);
  event.put(endcapResult);
  event.put(ESResult);

}



void  EcalDigiProducer::checkCalibrations(const edm::EventSetup & eventSetup) 
{}


void EcalDigiProducer::checkGeometry(const edm::EventSetup & eventSetup) 
{
  // TODO find a way to avoid doing this every event
  edm::ESHandle<CaloGeometry> hGeometry;
  eventSetup.get<IdealGeometryRecord>().get(hGeometry);

  const CaloGeometry * pGeometry = &*hGeometry;
  
  // see if we need to update
  if(pGeometry != theGeometry) {
    theGeometry = pGeometry;
    updateGeometry();
    setupFakePedestals();
  }
}


void EcalDigiProducer::updateGeometry() {
  theEcalResponse->setGeometry(theGeometry);

  theBarrelDets.clear();
  theEndcapDets.clear();
  theESDets.clear();

  theBarrelDets =  theGeometry->getValidDetIds(DetId::Ecal, EcalBarrel);
  theEndcapDets =  theGeometry->getValidDetIds(DetId::Ecal, EcalEndcap);
  theESDets     =  theGeometry->getValidDetIds(DetId::Ecal, EcalPreshower);

  //PG FIXME
  std::cout << "deb geometry: "
            << "\t barrel: " << theBarrelDets.size ()
            << "\t endcap: " << theEndcapDets.size ()
	    << "\t preshower: " << theESDets.size()
            << std::endl ;

  theBarrelDigitizer->setDetIds(theBarrelDets);
  theEndcapDigitizer->setDetIds(theEndcapDets);
  theESDigitizer->setDetIds(theESDets);
}

