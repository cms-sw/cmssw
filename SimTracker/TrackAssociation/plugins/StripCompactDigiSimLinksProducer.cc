// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "SimDataFormats/TrackerDigiSimLink/interface/StripDigiSimLink.h"
#include "DataFormats/Common/interface/DetSetVector.h"
#include "SimDataFormats/TrackerDigiSimLink/interface/StripCompactDigiSimLinks.h"

#ifdef SCDSL_DEBUG
  #define DEBUG(X) X
#else
  #define DEBUG(X)
#endif

class StripCompactDigiSimLinksProducer : public edm::one::EDProducer<> {
    public:
        StripCompactDigiSimLinksProducer(const edm::ParameterSet &iConfig) ;
        ~StripCompactDigiSimLinksProducer() override;

        void produce(edm::Event&, const edm::EventSetup&) override;

    private:
        edm::InputTag src_;
        uint32_t      maxHoleSize_;
};

StripCompactDigiSimLinksProducer::StripCompactDigiSimLinksProducer(const edm::ParameterSet &iConfig) :
    src_(iConfig.getParameter<edm::InputTag>("src")),
    maxHoleSize_(iConfig.getParameter<uint32_t>("maxHoleSize"))
{
    produces<StripCompactDigiSimLinks>();
}

StripCompactDigiSimLinksProducer::~StripCompactDigiSimLinksProducer()
{
}

void
StripCompactDigiSimLinksProducer::produce(edm::Event & iEvent, const edm::EventSetup&) 
{
    using namespace edm;
    Handle<DetSetVector<StripDigiSimLink> > src;
    iEvent.getByLabel(src_, src);

    StripCompactDigiSimLinks::Filler output;

    int previousStrip;     // previous strip with at least one link (might not be the strip of the previous link there are overlapping clusters)
    int previousLinkStrip; // strip of the previous link (can be the same as the one of this link if there are overlapping clusters)
    std::vector<StripCompactDigiSimLinks::key_type> thisStripSignals;      // particles on this strip
    std::vector<StripCompactDigiSimLinks::key_type> previousStripSignals;  // particles on the previous strip

    for(auto const& det : *src) {
        DEBUG(std::cerr << "\n\nProcessing detset " << det.detId() << ", size = " << det.size() << std::endl;)
        previousStrip     = -2; // previous strip with at least one link (might not be the strip of the previous link there are overlapping clusters)
        previousLinkStrip = -2; // strip of the previous link (can be the same as the one of this link if there are overlapping clusters)
        thisStripSignals.clear();
        previousStripSignals.clear();
        for (DetSet<StripDigiSimLink>::const_iterator it = det.begin(), ed = det.end(); it != ed; ++it) {
            DEBUG(std::cerr << "  processing digiSimLink on strip " << it->channel() << " left by particle " << it->SimTrackId() << ", event " << it->eventId().rawId() << std::endl;)
            if (int(it->channel()) != previousLinkStrip) { 
                previousStrip = previousLinkStrip;
                DEBUG(std::cerr << "   strip changed!" << std::endl;)
                swap(thisStripSignals, previousStripSignals); 
                thisStripSignals.clear();
            }
            DEBUG(std::cerr << "    previous strip " << previousStrip << ", previous link strip " << previousLinkStrip << std::endl;)
            //DEBUG(std::cerr << "    on previous strip: "; for(auto const& k : previousStripSignals) { std::cerr << "(ev " << k.first.rawId() << ", id " << k.second << ") "; } std::cerr << std::endl;)
            //DEBUG(std::cerr << "    on this strip: ";     for(auto const& k :     thisStripSignals) { std::cerr << "(ev " << k.first.rawId() << ", id " << k.second << ") "; } std::cerr << std::endl;)
            StripCompactDigiSimLinks::key_type key(it->eventId(), it->SimTrackId());
            bool alreadyClusterized = false;
            if (int(it->channel()) == previousStrip+1) {
                DEBUG(std::cerr << "  on next strip" << std::endl;)
                if (std::find(previousStripSignals.begin(), previousStripSignals.end(), key) != previousStripSignals.end()) {
                    alreadyClusterized = true;
                    DEBUG(std::cerr << "    and part of previous cluster" << std::endl;)
                }
            }
            if (!alreadyClusterized) {
                DEBUG(std::cerr << "   clusterize!" << std::endl;)
                unsigned int size = 1;
                int myLastStrip = it->channel(); // last known strip with signals from this particle
                for (DetSet<StripDigiSimLink>::const_iterator it2 = it+1; it2 < ed; ++it2) {
                    DEBUG(std::cerr << "     digiSimLink on strip " << it2->channel() << " left by particle " << it2->SimTrackId() << ", event " << it2->eventId().rawId() << std::endl;)
                    if ((it2->channel() - myLastStrip) > maxHoleSize_+1) {
                        DEBUG(std::cerr << "        found hole of size " << (it2->channel() - myLastStrip) << ", stopping." << std::endl;)
                        break;
                    }
                    if ((it2->eventId() == key.first) && (it2->SimTrackId() == key.second)) {
                        size++;
                        DEBUG(std::cerr << "        extending cluster, now size = " << size << std::endl;)
                        myLastStrip = it2->channel();
                    }
                }
                output.insert(key, StripCompactDigiSimLinks::HitRecord(det.detId(), it->channel(), size));
            }
            if (int(it->channel()) != previousLinkStrip) {
                previousLinkStrip = it->channel();
            }
            thisStripSignals.push_back(key);
            DEBUG(std::cerr << "    ending state " << previousStrip << ", previous link strip " << previousLinkStrip << std::endl;)
            //DEBUG(std::cerr << "    on previous strip: "; for(auto const& k : previousStripSignals) { std::cerr << "(ev " << k.first.rawId() << ", id " << k.second << ") "; } std::cerr << std::endl;)
            //DEBUG(std::cerr << "    on this strip: ";     for(auto const& k :     thisStripSignals) { std::cerr << "(ev " << k.first.rawId() << ", id " << k.second << ") "; } std::cerr << std::endl;)
            DEBUG(std::cerr << std::endl;)
        }
    }
   
    std::unique_ptr< StripCompactDigiSimLinks > ptr(new StripCompactDigiSimLinks(output));
    iEvent.put(std::move(ptr));
}

DEFINE_FWK_MODULE(StripCompactDigiSimLinksProducer);
