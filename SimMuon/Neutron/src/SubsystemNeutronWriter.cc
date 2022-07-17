#include "SimMuon/Neutron/interface/SubsystemNeutronWriter.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "SimMuon/Neutron/src/AsciiNeutronWriter.h"
#include "SimMuon/Neutron/src/RootNeutronWriter.h"
#include "SimMuon/Neutron/src/NeutronWriter.h"
#include "SimMuon/Neutron/src/EDMNeutronWriter.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/RandomNumberGenerator.h"

#include "CLHEP/Random/RandFlat.h"

#include <algorithm>

using namespace std;

class SortByTime {
public:
  bool operator()(const PSimHit& h1, const PSimHit& h2) { return (h1.tof() < h2.tof()); }
};

SubsystemNeutronWriter::SubsystemNeutronWriter(edm::ParameterSet const& pset)
    : theHitWriter(nullptr),
      useRandFlat(false),
      theInputTag(pset.getParameter<edm::InputTag>("input")),
      theNeutronTimeCut(pset.getParameter<double>("neutronTimeCut")),
      theTimeWindow(pset.getParameter<double>("timeWindow")),
      theT0(pset.getParameter<double>("t0")),
      hitToken_(consumes<edm::PSimHitContainer>(theInputTag)),
      theNEvents(0),
      initialized(false),
      useLocalDetId_(true) {
  string writer = pset.getParameter<string>("writer");
  string output = pset.getParameter<string>("output");
  if (writer == "ASCII") {
    theHitWriter = new AsciiNeutronWriter(output);
  } else if (writer == "ROOT") {
    theHitWriter = new RootNeutronWriter(output);
  } else if (writer == "EDM") {
    produces<edm::PSimHitContainer>();
    theHitWriter = new EDMNeutronWriter();
    // write out the real DetId, not the local one
    useLocalDetId_ = false;
    // smear the times
    edm::Service<edm::RandomNumberGenerator> rng;
    if (!rng.isAvailable()) {
      throw cms::Exception("Configuration")
          << "SubsystemNeutronWriter requires the RandomNumberGeneratorService\n"
             "which is not present in the configuration file.  You must add the service\n"
             "in the configuration file or remove the modules that require it.";
    }
    useRandFlat = true;
  } else {
    throw cms::Exception("NeutronWriter") << "Bad writer: " << writer;
  }
}

SubsystemNeutronWriter::~SubsystemNeutronWriter() {
  printStats();
  delete theHitWriter;
}

void SubsystemNeutronWriter::printStats() {
  edm::LogInfo("SubsystemNeutronWriter") << "SubsystemNeutronWriter Statistics:\n";
  for (map<int, int>::iterator mapItr = theCountPerChamberType.begin(); mapItr != theCountPerChamberType.end();
       ++mapItr) {
    edm::LogInfo("SubsystemNeutronWriter")
        << "   theEventOccupancy[" << mapItr->first << "] = " << mapItr->second << " / " << theNEvents << " / NCT \n";
  }
}

void SubsystemNeutronWriter::produce(edm::Event& e, edm::EventSetup const& c) {
  CLHEP::HepRandomEngine* engine = nullptr;
  if (useRandFlat) {
    edm::Service<edm::RandomNumberGenerator> rng;
    engine = &rng->getEngine(e.streamID());
  }
  theHitWriter->beginEvent(e, c);
  ++theNEvents;
  const edm::Handle<edm::PSimHitContainer>& hits = e.getHandle(hitToken_);

  // sort hits by chamber
  std::map<int, edm::PSimHitContainer> hitsByChamber;
  for (edm::PSimHitContainer::const_iterator hitItr = hits->begin(); hitItr != hits->end(); ++hitItr) {
    int chamberIndex = chamberId(hitItr->detUnitId());
    hitsByChamber[chamberIndex].push_back(*hitItr);
  }

  // now write out each chamber's contents
  for (std::map<int, edm::PSimHitContainer>::iterator hitsByChamberItr = hitsByChamber.begin();
       hitsByChamberItr != hitsByChamber.end();
       ++hitsByChamberItr) {
    int chambertype = chamberType(hitsByChamberItr->first);
    writeHits(chambertype, hitsByChamberItr->second, engine);
  }
  theHitWriter->endEvent();
}

void SubsystemNeutronWriter::initialize(int chamberType) {
  // should instantiate one of every chamber type, just so
  // ROOT knows what file to associate them with
  theHitWriter->initialize(chamberType);
}

void SubsystemNeutronWriter::writeHits(int chamberType,
                                       edm::PSimHitContainer& chamberHits,
                                       CLHEP::HepRandomEngine* engine) {
  sort(chamberHits.begin(), chamberHits.end(), SortByTime());
  edm::PSimHitContainer cluster;
  float startTime = -1000.;
  float smearing = 0.;
  for (size_t i = 0; i < chamberHits.size(); ++i) {
    PSimHit hit = chamberHits[i];
    float tof = hit.tof();
    LogDebug("SubsystemNeutronWriter") << "found hit from part type " << hit.particleType() << " at tof " << tof
                                       << " p " << hit.pabs() << " on det " << hit.detUnitId() << " chamber type "
                                       << chamberType;
    if (tof > theNeutronTimeCut) {
      if (tof > (startTime + theTimeWindow)) {  // 1st in cluster
        startTime = tof;
        // set the time to be [t0, t0+25] at start of event
        smearing = theT0;
        if (useRandFlat) {
          smearing += CLHEP::RandFlat::shoot(engine, 25.);
        }
        if (!cluster.empty()) {
          LogDebug("SubsystemNeutronWriter") << "filling old cluster";
          writeCluster(chamberType, cluster);
          cluster.clear();
        }
        LogDebug("SubsystemNeutronWriter")
            << "starting neutron cluster at time " << startTime << " on detType " << chamberType;
      }
      adjust(hit, -1. * startTime, smearing);
      cluster.push_back(hit);
    }
  }
  // get any leftover cluster
  if (!cluster.empty()) {
    writeCluster(chamberType, cluster);
  }
}

void SubsystemNeutronWriter::writeCluster(int chamberType, const edm::PSimHitContainer& cluster) {
  if (accept(cluster)) {
    theHitWriter->writeCluster(chamberType, cluster);
    updateCount(chamberType);
  }
}

void SubsystemNeutronWriter::adjust(PSimHit& h, float timeOffset, float smearing) {
  unsigned int detId = useLocalDetId_ ? localDetId(h.detUnitId()) : h.detUnitId();
  float htime = h.timeOfFlight() + timeOffset + smearing;
  // prevent float precision loss
  if (h.timeOfFlight() > 1.E+6) {
    htime = smearing;
  }
  h = PSimHit(h.entryPoint(),
              h.exitPoint(),
              h.pabs(),
              htime,
              h.energyLoss(),
              h.particleType(),
              detId,
              h.trackId(),
              h.momentumAtEntry().theta(),
              h.momentumAtEntry().phi());
}

void SubsystemNeutronWriter::updateCount(int chamberType) {
  map<int, int>::iterator entry = theCountPerChamberType.find(chamberType);
  if (entry == theCountPerChamberType.end()) {
    theCountPerChamberType.insert(pair<int, int>(chamberType, 1));
  } else {
    ++(entry->second);
  }
}
