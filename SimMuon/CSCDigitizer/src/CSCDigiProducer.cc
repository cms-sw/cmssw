#include "SimMuon/CSCDigitizer/src/CSCDigiProducer.h"

#include "DataFormats/Common/interface/Handle.h"

#include "SimMuon/CSCDigitizer/src/CSCConfigurableStripConditions.h"
#include "SimMuon/CSCDigitizer/src/CSCDbStripConditions.h"

#include "FWCore/AbstractServices/interface/RandomNumberGenerator.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include <FWCore/MessageLogger/interface/MessageLogger.h>

#include <string>

CSCDigiProducer::CSCDigiProducer(const edm::ParameterSet &ps) : theDigitizer(ps), theStripConditions(nullptr) {
  geom_Token =
      esConsumes<CSCGeometry, MuonGeometryRecord>(edm::ESInputTag("", ps.getParameter<std::string>("GeometryType")));
  magfield_Token = esConsumes<MagneticField, IdealMagneticFieldRecord>();
  pdt_Token = esConsumes<ParticleDataTable, edm::DefaultRecord>();
  produces<CSCWireDigiCollection>("MuonCSCWireDigi");
  produces<CSCStripDigiCollection>("MuonCSCStripDigi");
  produces<CSCComparatorDigiCollection>("MuonCSCComparatorDigi");
  produces<DigiSimLinks>("MuonCSCWireDigiSimLinks");
  produces<DigiSimLinks>("MuonCSCStripDigiSimLinks");
  std::string stripConditions(ps.getParameter<std::string>("stripConditions"));

  edm::ParameterSet stripPSet = ps.getParameter<edm::ParameterSet>("strips");
  if (stripConditions == "Configurable") {
    theStripConditions = new CSCConfigurableStripConditions(stripPSet);
  } else if (stripConditions == "Database") {
    theStripConditions = new CSCDbStripConditions(stripPSet, consumesCollector());
  } else {
    throw cms::Exception("CSCDigiProducer") << "Bad option for strip conditions: " << stripConditions;
  }
  theDigitizer.setStripConditions(theStripConditions);

  edm::Service<edm::RandomNumberGenerator> rng;
  if (!rng.isAvailable()) {
    throw cms::Exception("Configuration") << "CSCDigitizer requires the RandomNumberGeneratorService\n"
                                             "which is not present in the configuration file.  You must add the "
                                             "service\n"
                                             "in the configuration file or remove the modules that require it.";
  }

  const std::string &mix = ps.getParameter<std::string>("mixLabel");
  for (const auto &cname :
       {ps.getParameter<std::string>("InputCollection"), ps.getParameter<std::string>("InputCollectionPU")}) {
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("CSCDigiProducer") << "Creating CrossingFrame Consumers for InputTag " << mix << ":" << cname;
#endif
    cf_tokens.push_back(consumes<CrossingFrame<PSimHit>>(edm::InputTag(mix, cname)));
  }
}

CSCDigiProducer::~CSCDigiProducer() { delete theStripConditions; }

void CSCDigiProducer::produce(edm::Event &ev, const edm::EventSetup &eventSetup) {
  edm::LogVerbatim("CSCDigitizer") << "[CSCDigiProducer::produce] starting event " << ev.id().event() << " of run "
                                   << ev.id().run();
  edm::Service<edm::RandomNumberGenerator> rng;
  CLHEP::HepRandomEngine *engine = &rng->getEngine(ev.streamID());

  std::vector<const CrossingFrame<PSimHit> *> cf_list;
  for (const auto &token : cf_tokens) {
    const auto &handle = ev.getHandle(token);
    if (handle.isValid()) {
      cf_list.emplace_back(handle.product());
    } else
      edm::LogWarning("CSCDigitizer") << "Input Source not Valid !!";
  }

  auto hits = std::make_unique<MixCollection<PSimHit>>(cf_list);

  // Create empty output
  auto pWireDigis = std::make_unique<CSCWireDigiCollection>();
  auto pStripDigis = std::make_unique<CSCStripDigiCollection>();
  auto pComparatorDigis = std::make_unique<CSCComparatorDigiCollection>();
  auto pWireDigiSimLinks = std::make_unique<DigiSimLinks>();
  auto pStripDigiSimLinks = std::make_unique<DigiSimLinks>();

  //@@ DOES NOTHING IF NO HITS.  Remove this for when there's real neutrons
  if (hits->size() > 0) {
    // find the geometry & conditions for this event
    edm::ESHandle<CSCGeometry> hGeom = eventSetup.getHandle(geom_Token);
    const CSCGeometry *pGeom = &*hGeom;

    theDigitizer.setGeometry(pGeom);

    // find the magnetic field
    edm::ESHandle<MagneticField> magfield = eventSetup.getHandle(magfield_Token);

    theDigitizer.setMagneticField(&*magfield);

    // set the particle table
    edm::ESHandle<ParticleDataTable> pdt = eventSetup.getHandle(pdt_Token);
    theDigitizer.setParticleDataTable(&*pdt);

    theStripConditions->initializeEvent(eventSetup);

    // run the digitizer
    theDigitizer.doAction(
        *hits, *pWireDigis, *pStripDigis, *pComparatorDigis, *pWireDigiSimLinks, *pStripDigiSimLinks, engine);
  }

  // store them in the event
  ev.put(std::move(pWireDigis), "MuonCSCWireDigi");
  ev.put(std::move(pStripDigis), "MuonCSCStripDigi");
  ev.put(std::move(pComparatorDigis), "MuonCSCComparatorDigi");
  ev.put(std::move(pWireDigiSimLinks), "MuonCSCWireDigiSimLinks");
  ev.put(std::move(pStripDigiSimLinks), "MuonCSCStripDigiSimLinks");
}
