#ifndef SimPPS_PPSPixelDigiProducer_PPSPixelDigiProducer_h
#define SimPPS_PPSPixelDigiProducer_PPSPixelDigiProducer_h

// -*- C++ -*-
//
// Package:    PPSPixelDigiProducer
// Class:      CTPPSPixelDigiProducer
//
/**\class CTPPSPixelDigiProducer PPSPixelDigiProducer.cc SimPPS/PPSPixelDigiProducer/plugins/PPSPixelDigiProducer.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  F.Ferro
//

// system include files
#include <memory>
#include <vector>
#include <map>
#include <string>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"

//  ****  CTPPS
#include "DataFormats/CTPPSDigi/interface/CTPPSPixelDigi.h"
#include "DataFormats/CTPPSDigi/interface/CTPPSPixelDigiCollection.h"

#include "SimPPS/PPSPixelDigiProducer/interface/RPixDetDigitizer.h"
#include "SimDataFormats/CrossingFrame/interface/MixCollection.h"

#include "DataFormats/Common/interface/DetSet.h"

// DB
#include "CondFormats/DataRecord/interface/CTPPSPixelDAQMappingRcd.h"
#include "CondFormats/DataRecord/interface/CTPPSPixelAnalysisMaskRcd.h"
#include "CondFormats/DataRecord/interface/CTPPSPixelGainCalibrationsRcd.h"
#include "CondFormats/PPSObjects/interface/CTPPSPixelDAQMapping.h"
#include "CondFormats/PPSObjects/interface/CTPPSPixelAnalysisMask.h"
#include "CondFormats/PPSObjects/interface/CTPPSPixelGainCalibrations.h"

// user include files
#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"
#include "SimDataFormats/TrackingHit/interface/PSimHit.h"
#include "SimDataFormats/CrossingFrame/interface/CrossingFrame.h"
#include "SimDataFormats/CrossingFrame/interface/MixCollection.h"
#include <cstdlib>  // I need it for random numbers
//needed for the geometry:
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "DataFormats/Common/interface/DetSetVector.h"

//Random Number
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/RandomNumberGenerator.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "CLHEP/Random/RandomEngine.h"

namespace CLHEP {
  class HepRandomEngine;
}

class CTPPSPixelDigiProducer : public edm::stream::EDProducer<> {
public:
  explicit CTPPSPixelDigiProducer(const edm::ParameterSet&);
  ~CTPPSPixelDigiProducer() override;
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void produce(edm::Event&, const edm::EventSetup&) override;

  // ----------member data ---------------------------
  std::vector<std::string> RPix_hit_containers_;
  typedef std::map<unsigned int, std::vector<PSimHit>> simhit_map;
  typedef simhit_map::iterator simhit_map_iterator;

  edm::ParameterSet conf_;

  std::map<uint32_t, std::unique_ptr<RPixDetDigitizer>> theAlgoMap;  //DetId = uint32_t

  CLHEP::HepRandomEngine* rndEngine_ = nullptr;
  int verbosity_;

  edm::EDGetTokenT<CrossingFrame<PSimHit>> tokenCrossingFramePPSPixel;
  edm::ESGetToken<CTPPSPixelGainCalibrations, CTPPSPixelGainCalibrationsRcd> gainCalibESToken_;
};

CTPPSPixelDigiProducer::CTPPSPixelDigiProducer(const edm::ParameterSet& conf)
    : conf_(conf), gainCalibESToken_(esConsumes()) {
  produces<edm::DetSetVector<CTPPSPixelDigi>>();

  // register data to consume
  tokenCrossingFramePPSPixel = consumes<CrossingFrame<PSimHit>>(edm::InputTag("mix", "g4SimHitsCTPPSPixelHits"));

  RPix_hit_containers_.clear();
  RPix_hit_containers_ = conf.getParameter<std::vector<std::string>>("ROUList");
  verbosity_ = conf.getParameter<int>("RPixVerbosity");
}

CTPPSPixelDigiProducer::~CTPPSPixelDigiProducer() {}

void CTPPSPixelDigiProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  // all distances in [mm]
  // RPDigiProducer
  desc.add<std::vector<std::string>>("ROUList", {"CTPPSPixelHits"});
  desc.add<int>("RPixVerbosity", 0);
  desc.add<bool>("CTPPSPixelDigiSimHitRelationsPersistence",
                 false);  // save links betweend digi, clusters and OSCAR/Geant4 hits

  // RPDetDigitizer
  desc.add<double>("RPixEquivalentNoiseCharge", 1000.0);
  desc.add<bool>("RPixNoNoise", false);

  // RPDisplacementGenerator
  desc.add<double>("RPixGeVPerElectron", 3.61e-09);
  desc.add<std::vector<double>>("RPixInterSmearing", {0.011});
  desc.add<bool>("RPixLandauFluctuations", true);
  desc.add<int>("RPixChargeDivisions", 20);
  desc.add<double>("RPixDeltaProductionCut", 0.120425);  // [MeV]

  // RPixChargeShare
  desc.add<std::string>("ChargeMapFile2E", "SimPPS/PPSPixelDigiProducer/data/PixelChargeMap.txt");
  desc.add<std::string>("ChargeMapFile2E_2X", "SimPPS/PPSPixelDigiProducer/data/PixelChargeMap_2X.txt");
  desc.add<std::string>("ChargeMapFile2E_2Y", "SimPPS/PPSPixelDigiProducer/data/PixelChargeMap_2Y.txt");
  desc.add<std::string>("ChargeMapFile2E_2X2Y", "SimPPS/PPSPixelDigiProducer/data/PixelChargeMap_2X2Y.txt");
  desc.add<double>(
      "RPixCoupling",
      0.250);  // fraction of the remaining charge going to the closer neighbour pixel. Value = 0.135, Value = 0.0 bypass the charge map and the charge sharing approach

  // RPixDummyROCSimulator
  desc.add<double>("RPixDummyROCThreshold", 1900.0);
  desc.add<double>("RPixDummyROCElectronPerADC", 135.0);  // 210.0 to be verified
  desc.add<int>("VCaltoElectronGain", 50);                // same values as in RPixDetClusterizer
  desc.add<int>("VCaltoElectronOffset", -411);            //
  desc.add<bool>("doSingleCalibration", false);           //
  desc.add<double>("RPixDeadPixelProbability", 0.001);
  desc.add<bool>("RPixDeadPixelSimulationOn", true);

  // CTPPSPixelSimTopology
  desc.add<double>("RPixActiveEdgeSmearing", 0.020);
  desc.add<double>("RPixActiveEdgePosition", 0.150);

  desc.add<std::string>("mixLabel", "mix");
  desc.add<std::string>("InputCollection", "g4SimHitsCTPPSPixelHits");
  descriptions.add("RPixDetDigitizer", desc);
}

//
// member functions
//

// ------------ method called to produce the data  ------------
void CTPPSPixelDigiProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  using namespace edm;
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

  // get calibration DB
  const auto& gainCalibration = iSetup.getData(gainCalibESToken_);

  // Step A: Get Inputs
  edm::Handle<CrossingFrame<PSimHit>> cf;

  iEvent.getByToken(tokenCrossingFramePPSPixel, cf);

  if (verbosity_) {
    edm::LogInfo("PPSPixelDigiProducer") << "\n\n=================== Starting SimHit access"
                                         << "  ===================";

    MixCollection<PSimHit> col{cf.product(), std::pair(-0, 0)};
    edm::LogInfo("PPSPixelDigiProducer") << col;
    MixCollection<PSimHit>::iterator cfi;
    int count = 0;
    for (cfi = col.begin(); cfi != col.end(); cfi++) {
      edm::LogInfo("PPSPixelDigiProducer")
          << " Hit " << count << " has tof " << cfi->timeOfFlight() << " trackid " << cfi->trackId() << " bunchcr "
          << cfi.bunch() << " trigger " << cfi.getTrigger()
          << ", from EncodedEventId: " << cfi->eventId().bunchCrossing() << " " << cfi->eventId().event()
          << " bcr from MixCol " << cfi.bunch();
      edm::LogInfo("PPSPixelDigiProducer") << " Hit: " << (*cfi) << "  " << cfi->exitPoint();
      count++;
    }
  }

  MixCollection<PSimHit> allRPixHits{cf.product(), std::pair(0, 0)};

  if (verbosity_)
    edm::LogInfo("PPSPixelDigiProducer") << "Input MixCollection size = " << allRPixHits.size();

  //Loop on PSimHit
  simhit_map SimHitMap;
  SimHitMap.clear();

  MixCollection<PSimHit>::iterator isim;
  for (isim = allRPixHits.begin(); isim != allRPixHits.end(); ++isim) {
    SimHitMap[(*isim).detUnitId()].push_back((*isim));
  }

  // Step B: LOOP on hits in event
  std::vector<edm::DetSet<CTPPSPixelDigi>> theDigiVector;
  theDigiVector.reserve(400);
  theDigiVector.clear();

  for (simhit_map_iterator it = SimHitMap.begin(); it != SimHitMap.end(); ++it) {
    edm::DetSet<CTPPSPixelDigi> digi_collector(it->first);

    if (theAlgoMap.find(it->first) == theAlgoMap.end()) {
      theAlgoMap[it->first] =
          std::make_unique<RPixDetDigitizer>(conf_, *rndEngine_, it->first, iSetup);  //a digitizer for eny detector
    }

    std::vector<int> input_links;
    std::vector<std::vector<std::pair<int, double>>> output_digi_links;  // links to simhits

    theAlgoMap.at(it->first)->run(
        SimHitMap[it->first], input_links, digi_collector.data, output_digi_links, &gainCalibration);

    if (!digi_collector.data.empty()) {
      theDigiVector.push_back(digi_collector);
    }
  }

  std::unique_ptr<edm::DetSetVector<CTPPSPixelDigi>> digi_output(new edm::DetSetVector<CTPPSPixelDigi>(theDigiVector));

  if (verbosity_) {
    edm::LogInfo("PPSPixelDigiProducer") << "digi_output->size()=" << digi_output->size();
  }

  iEvent.put(std::move(digi_output));
}

DEFINE_FWK_MODULE(CTPPSPixelDigiProducer);

#endif
