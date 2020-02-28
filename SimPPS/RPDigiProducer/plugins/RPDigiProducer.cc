// user include files
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"
#include "SimDataFormats/TrackingHit/interface/PSimHit.h"
#include "SimDataFormats/CrossingFrame/interface/CrossingFrame.h"
#include "SimDataFormats/CrossingFrame/interface/MixCollection.h"

#include "DataFormats/CTPPSDigi/interface/TotemRPDigi.h"
#include "DataFormats/CTPPSDigi/interface/TotemRPDigi.h"
#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/CTPPSDetId/interface/TotemRPDetId.h"
#include "DataFormats/Common/interface/DetSet.h"

//Random Number
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/RandomNumberGenerator.h"

#include "CondFormats/CTPPSReadoutObjects/interface/TotemAnalysisMask.h"
#include "CondFormats/DataRecord/interface/TotemReadoutRcd.h"
#include "SimPPS/RPDigiProducer/interface/RPSimTypes.h"
#include "SimPPS/RPDigiProducer/plugins/RPDetDigitizer.h"
#include "SimPPS/RPDigiProducer/plugins/DeadChannelsManager.h"

// system include files
#include <memory>
#include <vector>
#include <map>
#include <string>
#include <iostream>
#include <cstdlib>  // I need it for random numbers

// user include files

//
// class decleration
//

namespace CLHEP {
  class HepRandomEngine;
}

class RPDigiProducer : public edm::EDProducer {
public:
  explicit RPDigiProducer(const edm::ParameterSet&);
  ~RPDigiProducer() override;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void beginRun(const edm::Run&, const edm::EventSetup&) override;
  void produce(edm::Event&, const edm::EventSetup&) override;

  edm::DetSet<TotemRPDigi> convertRPStripDetSet(const edm::DetSet<TotemRPDigi>&);

  // ----------member data ---------------------------
  std::vector<std::string> RP_hit_containers_;
  typedef std::map<unsigned int, std::vector<PSimHit>> simhit_map;
  typedef simhit_map::iterator simhit_map_iterator;

  edm::ParameterSet conf_;
  std::map<RPDetId, std::unique_ptr<RPDetDigitizer>> theAlgoMap;

  CLHEP::HepRandomEngine* rndEngine_ = nullptr;
  int verbosity_;

  /**
       * this variable answers the question whether given channel is dead or not
       */
  DeadChannelsManager deadChannelsManager;
  /**
       * this variable indicates whether we take into account dead channels or simulate as if all
       * channels work ok (by default we do not simulate dead channels)
       */
  bool simulateDeadChannels;

  edm::EDGetTokenT<CrossingFrame<PSimHit>> tokenCrossingFrameTotemRP;
};

RPDigiProducer::RPDigiProducer(const edm::ParameterSet& conf) : conf_(conf) {
  //now do what ever other initialization is needed
  produces<edm::DetSetVector<TotemRPDigi>>();

  // register data to consume
  tokenCrossingFrameTotemRP = consumes<CrossingFrame<PSimHit>>(edm::InputTag("mix", "g4SimHitsTotemHitsRP", ""));

  RP_hit_containers_ = conf.getParameter<std::vector<std::string>>("ROUList");
  verbosity_ = conf.getParameter<int>("RPVerbosity");

  simulateDeadChannels = false;
  if (conf.exists(
          "simulateDeadChannels")) {  //check if "simulateDeadChannels" variable is defined in configuration file
    simulateDeadChannels = conf.getParameter<bool>("simulateDeadChannels");
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
  if (!rndEngine_) {
    Service<RandomNumberGenerator> rng;
    if (!rng.isAvailable()) {
      throw cms::Exception("Configuration")
          << "This class requires the RandomNumberGeneratorService\n"
             "which is not present in the configuration file.  You must add the service\n"
             "in the configuration file or remove the modules that require it.";
    }
    rndEngine_ = &(rng->getEngine(iEvent.streamID()));
  }

  // Step A: Get Inputs
  edm::Handle<CrossingFrame<PSimHit>> cf;
  iEvent.getByLabel("mix", "g4SimHitsTotemHitsRP", cf);

  if (verbosity_) {
    edm::LogInfo("RPDigiProducer") << "\n\n=================== Starting SimHit access"
                                   << "  ==================="
                                   << "\n";

    MixCollection<PSimHit> col{cf.product(), std::pair(-0, 0)};
    MixCollection<PSimHit>::iterator cfi;
    int count = 0;
    for (cfi = col.begin(); cfi != col.end(); cfi++) {
      edm::LogInfo("RPDigiProducer") << " Hit " << count << " has tof " << cfi->timeOfFlight() << " trackid "
                                     << cfi->trackId() << " bunchcr " << cfi.bunch() << " trigger " << cfi.getTrigger()
                                     << ", from EncodedEventId: " << cfi->eventId().bunchCrossing() << " "
                                     << cfi->eventId().event() << " bcr from MixCol " << cfi.bunch() << "\n";
      edm::LogInfo("RPDigiProducer") << " Hit: " << (*cfi) << "\n";
      count++;
    }
  }

  MixCollection<PSimHit> allRPHits{cf.product(), std::pair(0, 0)};

  if (verbosity_)
    edm::LogInfo("RPDigiProducer") << "Input MixCollection size = " << allRPHits.size() << "\n";

  //Loop on PSimHit
  simhit_map simHitMap_;
  simHitMap_.clear();

  MixCollection<PSimHit>::iterator isim;
  for (isim = allRPHits.begin(); isim != allRPHits.end(); ++isim) {
    simHitMap_[(*isim).detUnitId()].push_back((*isim));
  }

  // Step B: LOOP on hits in event
  std::vector<edm::DetSet<TotemRPDigi>> DigiVector;
  DigiVector.reserve(400);
  DigiVector.clear();

  for (simhit_map_iterator it = simHitMap_.begin(); it != simHitMap_.end(); ++it) {
    edm::DetSet<TotemRPDigi> digi_collector(it->first);

    if (theAlgoMap.find(it->first) == theAlgoMap.end()) {
      theAlgoMap[it->first] =
          std::unique_ptr<RPDetDigitizer>(new RPDetDigitizer(conf_, *rndEngine_, it->first, iSetup));
    }

    std::vector<int> input_links;
    simromanpot::DigiPrimaryMapType output_digi_links;

    (theAlgoMap.find(it->first)->second)
        ->run(simHitMap_[it->first], input_links, digi_collector.data, output_digi_links);

    if (!digi_collector.data.empty()) {
      DigiVector.push_back(convertRPStripDetSet(digi_collector));
    }
  }

  // Step C: create empty output collection
  std::unique_ptr<edm::DetSetVector<TotemRPDigi>> digi_output(new edm::DetSetVector<TotemRPDigi>(DigiVector));

  if (verbosity_) {
    edm::LogInfo("RPDigiProducer") << "digi_output->size()=" << digi_output->size() << "\n";
  }
  // Step D: write output to file
  iEvent.put(std::move(digi_output));
}

// ------------ method called once each job just before starting event loop  ------------
void RPDigiProducer::beginRun(const edm::Run& beginrun, const edm::EventSetup& es) {
  // get analysis mask to mask channels
  if (simulateDeadChannels) {
    edm::ESHandle<TotemAnalysisMask> analysisMask;
    es.get<TotemReadoutRcd>().get(analysisMask);
    deadChannelsManager = DeadChannelsManager(analysisMask);  //set analysisMask in deadChannelsManager
  }
}

edm::DetSet<TotemRPDigi> RPDigiProducer::convertRPStripDetSet(const edm::DetSet<TotemRPDigi>& rpstrip_detset) {
  edm::DetSet<TotemRPDigi> rpdigi_detset(rpstrip_detset.detId());
  rpdigi_detset.reserve(rpstrip_detset.size());

  for (std::vector<TotemRPDigi>::const_iterator stripIterator = rpstrip_detset.data.begin();
       stripIterator < rpstrip_detset.data.end();
       ++stripIterator) {
    rpdigi_detset.push_back(TotemRPDigi(stripIterator->stripNumber()));
  }

  return rpdigi_detset;
}

void RPDigiProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //RPSiDetDigitizer
  //all distances in [mm]
  edm::ParameterSetDescription desc;
  desc.add<bool>("RPLandauFluctuations", true);
  desc.add<bool>("RPDisplacementOn", false);
  desc.add<int>("RPVerbosity", 0);
  desc.add<double>("RPVFATThreshold", 9000.0);
  desc.add<double>("RPTopEdgePosition", 1.5);
  desc.add<double>("RPActiveEdgeSmearing", 0.013);
  desc.add<double>("RPEquivalentNoiseCharge300um", 1000.0);
  desc.add<int>("RPVFATTriggerMode", 2);
  desc.add<std::vector<double>>("RPInterStripSmearing",
                                {
                                    0.011,
                                });
  desc.add<double>("RPSharingSigmas", 5.0);  //how many sigmas taken into account for the edges and inter strips
  desc.add<double>("RPGeVPerElectron", 3.61e-09);
  desc.add<double>("RPActiveEdgePosition", 0.034);  //from the physical edge
  desc.add<bool>("RPDeadStripSimulationOn", false);
  desc.add<std::vector<std::string>>("ROUList",
                                     {
                                         "TotemHitsRP",
                                     });
  desc.add<bool>("RPNoNoise", false);
  desc.add<bool>("RPDigiSimHitRelationsPresistence", false);  //save links betweend digi, clusters and OSCAR/Geant4 hits
  desc.add<std::string>("mixLabel", "mix");
  desc.add<int>("RPChargeDivisionsPerThickness", 5);
  desc.add<double>("RPDeltaProductionCut", 0.120425);  //[MeV]
  desc.add<double>("RPBottomEdgePosition", 1.5);
  desc.add<double>("RPBottomEdgeSmearing", 0.011);
  desc.add<double>("RPTopEdgeSmearing", 0.011);
  desc.add<std::string>("InputCollection", "g4SimHitsTotemHitsRP");
  desc.add<double>("RPInterStripCoupling",
                   1.0);  //fraction of charge going to the strip, the missing part is taken by its neighbours
  desc.add<double>("RPDeadStripProbability", 0.001);
  desc.add<int>("RPChargeDivisionsPerStrip", 15);
  descriptions.add("RPSiDetDigitizer", desc);
}

DEFINE_FWK_MODULE(RPDigiProducer);
