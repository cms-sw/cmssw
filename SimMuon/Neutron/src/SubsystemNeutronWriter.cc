#include "SimMuon/Neutron/interface/SubsystemNeutronWriter.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "SimMuon/Neutron/src/AsciiNeutronWriter.h"
#include "SimMuon/Neutron/src/RootNeutronWriter.h"
#include <algorithm>

using namespace std;

class SortByTime {
public:
  bool operator()(const PSimHit & h1, const PSimHit & h2) {
   return (h1.tof() < h2.tof());
  }
};


SubsystemNeutronWriter::SubsystemNeutronWriter(edm::ParameterSet const& pset) 
: theInputTag(pset.getParameter<edm::InputTag>("input")),
  theNeutronTimeCut(pset.getParameter<double>("neutronTimeCut")),
  theTimeWindow(pset.getParameter<double>("timeWindow")),
  initialized(false)
{
  string writer = pset.getParameter<string>("writer");
  string output = pset.getParameter<string>("output");
  int nChamberTypes = pset.getParameter<int>("nChamberTypes");
  if(writer == "ASCII")
  {
    theHitWriter = new AsciiNeutronWriter(output);
  }
  else if (writer == "ROOT")
  {
    theHitWriter = new RootNeutronWriter(output);
  }
  else 
  {
    throw cms::Exception("NeutronWriter") << "Bad writer: "
      << writer;
  } 
}


SubsystemNeutronWriter::~SubsystemNeutronWriter()
{
  printStats();
  delete theHitWriter;
}


void SubsystemNeutronWriter::printStats() {
  edm::LogInfo("SubsystemNeutronWriter") << "SubsystemNeutronWriter Statistics:\n";
  for(map<int,int>::iterator mapItr = theCountPerChamberType.begin();
      mapItr != theCountPerChamberType.end();  ++mapItr) {
     edm::LogInfo("SubsystemNeutronWriter") << "   theEventOccupancy[" << mapItr->first << "] = "
         << mapItr->second << " / NEV / NCT \n";
  }
}


void SubsystemNeutronWriter::analyze(edm::Event const& e, edm::EventSetup const& c)
{
  edm::Handle<edm::PSimHitContainer> hits;
  e.getByLabel(theInputTag, hits);

  // sort hits by chamber
  map<int, edm::PSimHitContainer> hitsByChamber;
  for(edm::PSimHitContainer::const_iterator hitItr = hits->begin();
      hitItr != hits->end(); ++hitItr)
  {
    int chamberIndex = chamberId(hitItr->detUnitId());
    hitsByChamber[chamberIndex].push_back(*hitItr);
  }

  // now write out each chamber's contents
  for(map<int, edm::PSimHitContainer>::iterator hitsByChamberItr = hitsByChamber.begin();
      hitsByChamberItr != hitsByChamber.end(); ++hitsByChamberItr)
  {
    int chambertype = chamberType(hitsByChamberItr->first);
    writeHits(chambertype, hitsByChamberItr->second);
  }
}


void SubsystemNeutronWriter::writeHits(int chamberType, edm::PSimHitContainer & input)
{

  sort(input.begin(), input.end(), SortByTime());
  edm::PSimHitContainer current;
  float startTime = -1000.;
  for(size_t i = 0; i < input.size(); ++i) {
    PSimHit hit = input[i];
std::cout << hit << std::endl;
    float tof = hit.tof();
    LogDebug("SubsystemNeutronWriter") << "found hit from part type " << hit.particleType()
                   << " at tof " << tof << " p " << hit.pabs() 
                   << " on det " << hit.detUnitId() 
                   << " chamber type " << chamberType;
    if(tof > theNeutronTimeCut) {
      if(tof > (startTime + theTimeWindow) ) { // 1st in cluster
        startTime = tof;
        if(!current.empty()) {
          LogDebug("SubsystemNeutronWriter") << "filling old cluster";
          theHitWriter->writeEvent(chamberType, current);
          updateCount(chamberType);
          current.clear();
        }
        LogDebug("SubsystemNeutronWriter") << "starting neutron cluster at time " << startTime 
          << " on detType " << chamberType;
      }
      // set the time to be 0 at start of event
std::cout << "ADJUST " << startTime << std::endl;
      adjust(hit, -1.*startTime);
      current.push_back( hit );
std::cout << "NEXT HIT" << std::endl;
    }
  }
std::cout << "LOOPED OVER HITS " << theHitWriter << std::endl;
  if(!current.empty()) {
    theHitWriter->writeEvent(chamberType, current);
    updateCount(chamberType);
  }
}


void SubsystemNeutronWriter::adjust(PSimHit & h, float timeOffset) {
  
  h = PSimHit( h.entryPoint(), h.exitPoint(), h.pabs(), 
               h.timeOfFlight() + timeOffset,
               h.energyLoss(), h.particleType(), 
               localDetId(h.detUnitId()), h.trackId(),
               h.momentumAtEntry().theta(),
               h.momentumAtEntry().phi() );
}


void SubsystemNeutronWriter::updateCount(int chamberType) {
  map<int,int>::iterator entry = theCountPerChamberType.find(chamberType);
  if(entry == theCountPerChamberType.end()) {
    theCountPerChamberType.insert( pair<int,int>(chamberType, 1) );
  } else {
    ++(entry->second);
  }
}

