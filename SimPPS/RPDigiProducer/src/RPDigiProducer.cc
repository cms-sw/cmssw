#include "SimPPS/RPDigiProducer/interface/RPDigiProducer.h"

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"
#include "SimDataFormats/TrackingHit/interface/PSimHit.h"
#include "SimDataFormats/CrossingFrame/interface/CrossingFrame.h"
#include "SimDataFormats/CrossingFrame/interface/MixCollection.h"
#include <cstdlib> // I need it for random numbers
//needed for the geometry:
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "DataFormats/CTPPSDigi/interface/TotemRPDigi.h"
#include "DataFormats/CTPPSDigi/interface/TotemRPDigi.h"
#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/CTPPSDetId/interface/TotemRPDetId.h"

//Random Number
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/RandomNumberGenerator.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "CLHEP/Random/RandomEngine.h"
#include "CondFormats/CTPPSReadoutObjects/interface/TotemAnalysisMask.h"
#include "CondFormats/DataRecord/interface/TotemReadoutRcd.h"
#include <iostream>

RPDigiProducer::RPDigiProducer(const edm::ParameterSet& conf) :
  conf_(conf) {
    //register your products
    /* Examples
       produces<ExampleData2>();

    //if do put with a label
    produces<ExampleData2>("label");
    */
    //now do what ever other initialization is needed
    produces<edm::DetSetVector<TotemRPDigi> > ();

    // register data to consume
    tokenCrossingFrameTotemRP = consumes<CrossingFrame<PSimHit>>(edm::InputTag("mix","g4SimHitsTotemHitsRP", ""));

    RP_hit_containers_.clear();
    RP_hit_containers_ = conf.getParameter<std::vector<std::string> > ("ROUList");
    verbosity_ = conf.getParameter<int> ("RPVerbosity");

    simulateDeadChannels = false;
    if (conf.exists("simulateDeadChannels")) { //check if "simulateDeadChannels" variable is defined in configuration file
      simulateDeadChannels = conf.getParameter<bool> ("simulateDeadChannels");
    }
  }

RPDigiProducer::~RPDigiProducer() {

  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)

}

//
// member functions
//

// ------------ method called to produce the data  ------------
void RPDigiProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  using namespace edm;

  // initialize random engine
  if(!rndEngine) {
    Service<RandomNumberGenerator> rng;
    if (!rng.isAvailable()) {
      throw cms::Exception("Configuration") << "This class requires the RandomNumberGeneratorService\n"
	"which is not present in the configuration file.  You must add the service\n"
	"in the configuration file or remove the modules that require it.";
    }
    rndEngine = &(rng->getEngine(iEvent.streamID()));
  }

  // Step A: Get Inputs
  edm::Handle<CrossingFrame<PSimHit> > cf;
  //	iEvent.getByToken(tokenCrossingFrameTotemRP, cf);
  iEvent.getByLabel("mix", "g4SimHitsTotemHitsRP", cf);

  if (verbosity_) {
    std::cout << "\n\n=================== Starting SimHit access" << "  ===================" << std::endl;

    std::auto_ptr<MixCollection<PSimHit> > col(
	new MixCollection<PSimHit> (cf.product(), std::pair<int, int>(-0, 0)));
    std::cout << *(col.get()) << std::endl;
    MixCollection<PSimHit>::iterator cfi;
    int count = 0;
    for (cfi = col->begin(); cfi != col->end(); cfi++) {
      std::cout << " Hit " << count << " has tof " << cfi->timeOfFlight() << " trackid "
	<< cfi->trackId() << " bunchcr " << cfi.bunch() << " trigger " << cfi.getTrigger()
	<< ", from EncodedEventId: " << cfi->eventId().bunchCrossing() << " "
	<< cfi->eventId().event() << " bcr from MixCol " << cfi.bunch() << std::endl;
      std::cout << " Hit: " << (*cfi) << std::endl;
      count++;
    }
  }

  std::auto_ptr<MixCollection<PSimHit> > allRPHits(
      new MixCollection<PSimHit> (cf.product(), std::pair<int, int>(0, 0)));

  if (verbosity_)
    std::cout << "Input MixCollection size = " << allRPHits->size() << std::endl;

  //Loop on PSimHit
  SimHitMap.clear();

  MixCollection<PSimHit>::iterator isim;
  for (isim = allRPHits->begin(); isim != allRPHits->end(); ++isim) {
    SimHitMap[(*isim).detUnitId()].push_back((*isim));

  }

  // Step B: LOOP on hits in event
  theDigiVector.reserve(400);
  theDigiVector.clear();

  for (simhit_map_iterator it = SimHitMap.begin(); it != SimHitMap.end(); ++it) {
    edm::DetSet<TotemRPDigi> digi_collector(it->first);

    if (theAlgoMap.find(it->first) == theAlgoMap.end()) {
      theAlgoMap[it->first] = boost::shared_ptr<RPDetDigitizer>(
	  new RPDetDigitizer(conf_, *rndEngine, it->first, iSetup));
    }

    std::vector<int> input_links;
    SimRP::DigiPrimaryMapType output_digi_links;

    (theAlgoMap.find(it->first)->second)->run(SimHitMap[it->first], input_links, digi_collector.data,
	output_digi_links);

    if (!digi_collector.data.empty()) {
      theDigiVector.push_back(convertRPStripDetSet(digi_collector));
    }
  }

  // Step C: create empty output collection
  std::unique_ptr<edm::DetSetVector<TotemRPDigi> > digi_output(
      new edm::DetSetVector<TotemRPDigi>(theDigiVector));

  if (verbosity_) {
    std::cout << "digi_output->size()=" << digi_output->size() << std::endl;
  }
  // Step D: write output to file
  iEvent.put(std::move(digi_output));
}

// ------------ method called once each job just before starting event loop  ------------
void RPDigiProducer::beginRun(edm::Run&, edm::EventSetup const& es){
  // get analysis mask to mask channels
  if (simulateDeadChannels) {
    edm::ESHandle<TotemAnalysisMask> analysisMask;
    es.get<TotemReadoutRcd> ().get(analysisMask);
    deadChannelsManager = DeadChannelsManager(analysisMask); //set analysisMask in deadChannelsManager
  }
}

// ------------ method called once each job just after ending the event loop  ------------
void RPDigiProducer::endJob() {
}

edm::DetSet<TotemRPDigi> RPDigiProducer::convertRPStripDetSet(const edm::DetSet<TotemRPDigi>& rpstrip_detset){
  edm::DetSet<TotemRPDigi> rpdigi_detset(rpstrip_detset.detId());
  rpdigi_detset.reserve(rpstrip_detset.size());

  for(std::vector<TotemRPDigi>::const_iterator stripIterator = rpstrip_detset.data.begin(); stripIterator < rpstrip_detset.data.end(); ++stripIterator){
    rpdigi_detset.push_back(TotemRPDigi(stripIterator->getStripNumber()));
  }

  return rpdigi_detset;
}

DEFINE_FWK_MODULE( RPDigiProducer);
