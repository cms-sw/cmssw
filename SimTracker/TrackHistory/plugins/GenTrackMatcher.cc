/** \class GenTrackMatcher
 *
 * \author Luca Lista, INFN
 * \author Victor E. Bazterra, UIC
 *
 *
 */


#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "DataFormats/Common/interface/Association.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "SimTracker/TrackHistory/interface/TrackHistory.h"

namespace edm
{
class ParameterSet;
}

using namespace edm;
using namespace std;
using namespace reco;

class GenTrackMatcher : public edm::stream::EDProducer<>
{
public:
    /// constructor
    GenTrackMatcher( const edm::ParameterSet & );

private:
    void produce( edm::Event& evt, const edm::EventSetup& es ) override;
    TrackHistory tracer_;
    edm::EDGetTokenT<View<Track>> tracks_;
    edm::EDGetTokenT<GenParticleCollection> genParticles_;
    edm::EDGetTokenT<vector<int>> genParticleInts_;
    typedef edm::Association<reco::GenParticleCollection> GenParticleMatch;
};

#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

GenTrackMatcher::GenTrackMatcher(const ParameterSet & p) :
        tracer_(p,consumesCollector()),
        tracks_(consumes<View<Track>>(p.getUntrackedParameter<edm::InputTag>("trackProducer"))),
        genParticles_(consumes<GenParticleCollection>(p.getUntrackedParameter<edm::InputTag>("genParticles"))),
        genParticleInts_(consumes<vector<int>>(p.getUntrackedParameter<edm::InputTag>("genParticles")))
{
    produces<GenParticleMatch>();
}

void GenTrackMatcher::produce(Event& evt, const EventSetup& es)
{
    Handle<View<Track> > tracks;
    evt.getByToken(tracks_, tracks);
    Handle<vector<int> > barCodes;
    evt.getByToken(genParticles_, barCodes);
    Handle<GenParticleCollection> genParticles;
    evt.getByToken(genParticles_, genParticles);
    auto_ptr<GenParticleMatch> match(new GenParticleMatch(GenParticleRefProd(genParticles)));
    GenParticleMatch::Filler filler(*match);
    size_t n = tracks->size();
    vector<int> indices(n,-1);
    tracer_.newEvent(evt, es);
    for (size_t i = 0; i < n; ++ i )
    {
        RefToBase<Track> track(tracks, i);
        if (tracer_.evaluate(track))
        {
            const HepMC::GenParticle * particle = tracer_.genParticle();
            if (particle)
            {
                int barCode = particle->barcode();
                vector<int>::const_iterator b = barCodes->begin(), e = barCodes->end(), f = find( b, e, barCode );
                if (f == e) {
                  edm::EDConsumerBase::Labels labels;
                  labelsForToken(genParticles_, labels);
                  throw edm::Exception(errors::InvalidReference)
                    << "found matching particle with barcode" << *f
                    << " which has not been found in " << labels.module;
                }
                indices[i] = *f;
            }
        }
    }
    filler.insert(tracks, indices.begin(), indices.end());
    filler.fill();
    evt.put(match);
}

#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE( GenTrackMatcher );

