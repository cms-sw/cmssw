#ifndef TrackingAnalysis_TrackingTruthProducer_h
#define TrackingAnalysis_TrackingTruthProducer_h

#include <map>

#include "DataFormats/Common/interface/Handle.h"

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "CommonTools/RecoAlgos/interface/TrackingParticleSelector.h"

#include "SimDataFormats/CrossingFrame/interface/CrossingFrame.h"
#include "SimDataFormats/CrossingFrame/interface/MixCollection.h"
#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"
#include "SimDataFormats/Track/interface/SimTrack.h"
#include "SimDataFormats/Vertex/interface/SimVertex.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticleFwd.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingVertexContainer.h"
#include "SimDataFormats/TrackingHit/interface/PSimHit.h"

#include "SimGeneral/TrackingAnalysis/interface/EncodedTruthId.h"

#include "Utilities/Timing/interface/TimingReport.h"
#include "Utilities/Timing/interface/TimerStack.h"

using namespace edm;
using namespace std;

class TrackingTruthProducer : public edm::EDProducer
{

public:
    explicit TrackingTruthProducer( const edm::ParameterSet & );
//  ~TrackingTruthProducer() { TimingReport::current()->dump(std::cout); }

private:

    void produce( edm::Event &, const edm::EventSetup & );

    int LayerFromDetid(const unsigned int&);

    edm::ParameterSet conf_;

    double                   distanceCut_;
    std::vector<std::string> dataLabels_;
    std::vector<std::string> hitLabelsVector_;
    double                   volumeRadius_;
    double                   volumeZ_;
    //    bool                     discardOutVolume_;
    bool					   mergedBremsstrahlung_;
    bool                     discardHitsFromDeltas_;
    std::string              simHitLabel_;

    std::string MessageCategory_;

    // Related to production

    edm::Handle<edm::HepMCProduct>            hepmc_;

    std::auto_ptr<MixCollection<PSimHit> >    pSimHits_;
    std::auto_ptr<MixCollection<SimTrack> >   simTracks_;
    std::auto_ptr<MixCollection<SimVertex> >  simVertexes_;

    std::auto_ptr<TrackingParticleCollection> trackingParticles_;
    std::auto_ptr<TrackingVertexCollection>   trackingVertexes_;
    TrackingParticleRefProd refTrackingParticles_;
    TrackingVertexRefProd   refTrackingVertexes_;

    std::auto_ptr<TrackingParticleCollection> mergedTrackingParticles_;
    std::auto_ptr<TrackingVertexCollection>   mergedTrackingVertexes_;
    TrackingParticleRefProd refMergedTrackingParticles_;
    TrackingVertexRefProd   refMergedTrackingVertexes_;

    typedef map<EncodedEventId, unsigned int> EncodedEventIdToIndex;
    typedef map<EncodedTruthId, unsigned int> EncodedTruthIdToIndex;
    typedef multimap<EncodedTruthId, unsigned int> EncodedTruthIdToIndexes;

    EncodedEventIdToIndex   eventIdCounter_;
    EncodedTruthIdToIndexes trackIdToHits_;
    EncodedTruthIdToIndex   trackIdToIndex_;

    bool selectorFlag_;
    TrackingParticleSelector selector_;

    template<typename Object, typename Associator>
    void associator(
        std::auto_ptr<MixCollection<Object> > const &,
        Associator &
    );

    void mergeBremsstrahlung();

    bool isBremsstrahlungVertex(
      TrackingVertex const & vertex,
      auto_ptr<TrackingParticleCollection> & tPC
    );
    
    void createTrackingTruth();

    bool setTrackingParticle(
        SimTrack const &,
        TrackingParticle &
    );

    int setTrackingVertex(
        SimVertex const &,
        TrackingVertex &
    );

    void addCloseGenVertexes(TrackingVertex &);
};


template<typename Object, typename Associator>
void TrackingTruthProducer::associator(
    std::auto_ptr<MixCollection<Object> > const & mixCollection,
    Associator & association
)
{
	int index = 0;
    // Clear the association map
    association.clear();
    // Create a association from simtracks to overall index in the mix collection
    for (typename MixCollection<Object>::MixItr iterator = mixCollection->begin(); iterator != mixCollection->end(); ++iterator, ++index)
    {
        typename Associator::key_type objectId = typename Associator::key_type(iterator->eventId(), iterator->trackId());
        association.insert( make_pair(objectId, index) );
    }
}


#endif
