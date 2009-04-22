#include "SimMuon/Neutron/src/NeutronProducer.h"
#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"
#include <map>


NeutronProducer::NeutronProducer(const edm::ParameterSet & pset)
: theInputTag(pset.getParameter<edm::InputTag>("input")),
  theNeutronTimeCut(pset.getParameter<double>("neutronTimeCut")),
  theTimeWindow(pset.getParameter<double>("timeWindow"))
{
  produces<edm::PSimHitContainer>();
}



NeutronProducer::~NeutronProducer() {}


class SortByTime {
public:
  bool operator()(const PSimHit & h1, const PSimHit & h2) {
   return (h1.tof() < h2.tof());
  }
};


void NeutronProducer::produce(edm::Event& e, const edm::EventSetup& eventSetup)
{

  edm::Handle<edm::PSimHitContainer> hits;
  e.getByLabel(theInputTag, hits);

  // sort hits by chamber
  std::map<int, edm::PSimHitContainer> hitsByChamber;
  for(edm::PSimHitContainer::const_iterator hitItr = hits->begin();
      hitItr != hits->end(); ++hitItr)
  {
    int chamberIndex = chamberId(hitItr->detUnitId());
    hitsByChamber[chamberIndex].push_back(*hitItr);
  }

  // output
  std::auto_ptr<edm::PSimHitContainer> output(new edm::PSimHitContainer());

  // now write out each chamber's contents
  for(std::map<int, edm::PSimHitContainer>::iterator hitsByChamberItr = hitsByChamber.begin();
      hitsByChamberItr != hitsByChamber.end(); ++hitsByChamberItr)
  {
    edm::PSimHitContainer chamberHits = hitsByChamberItr->second;
    sort(chamberHits.begin(), chamberHits.end(), SortByTime());
    float startTime = -1000.;
    for(size_t i = 0; i < chamberHits.size(); ++i) {
      PSimHit hit = chamberHits[i];
      float tof = hit.tof();
      LogDebug("SubsystemNeutronWriter") << "found hit from part type " << hit.particleType()
                   << " at tof " << tof << " p " << hit.pabs()
                   << " on det " << hit.detUnitId();
      if(tof > theNeutronTimeCut) {
        if(tof > (startTime + theTimeWindow) ) { // 1st in cluster
          startTime = tof;
          LogDebug("SubsystemNeutronWriter") << "starting neutron cluster at time " << startTime;
        }
        // set the time to be 0 at start of event
        adjust(hit, -1.*startTime);
        output->push_back( hit );
      }
    }
  } // loop over chambers
  e.put(output);
}


void NeutronProducer::adjust(PSimHit & h, float timeOffset) {

  h = PSimHit( h.entryPoint(), h.exitPoint(), h.pabs(),
               h.timeOfFlight() + timeOffset,
               h.energyLoss(), h.particleType(),
               h.detUnitId(), h.trackId(),
               h.momentumAtEntry().theta(),
               h.momentumAtEntry().phi() );
}

