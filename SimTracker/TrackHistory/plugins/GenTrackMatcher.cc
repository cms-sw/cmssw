/** \class GenTrackMatcher
 *
 * \author Luca Lista, INFN
 * \author Victor E. Bazterra, UIC
 *
 * \version $Id: GenTrackMatcher.cc,v 1.6 2008/07/23 01:40:41 bazterra Exp $
 *
 */


#include "FWCore/Framework/interface/EDProducer.h"
#include "DataFormats/Common/interface/Association.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "SimTracker/TrackHistory/interface/TrackHistory.h"

namespace edm
{
class ParameterSet;
}

class GenTrackMatcher : public edm::EDProducer
{
public:
    /// constructor
    GenTrackMatcher( const edm::ParameterSet & );

private:
    void produce( edm::Event& evt, const edm::EventSetup& es );
    TrackHistory tracer_;
    edm::InputTag tracks_, genParticles_;
    typedef edm::Association<reco::GenParticleCollection> GenParticleMatch;
};

#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

using namespace edm;
using namespace std;
using namespace reco;

GenTrackMatcher::GenTrackMatcher(const ParameterSet & p) :
        tracer_(p),
        tracks_(p.getUntrackedParameter<edm::InputTag>("trackProducer")),
        genParticles_(p.getUntrackedParameter<edm::InputTag>("genParticles"))
{
    produces<GenParticleMatch>();
}

void GenTrackMatcher::produce(Event& evt, const EventSetup& es)
{
    Handle<View<Track> > tracks;
    evt.getByLabel(tracks_, tracks);
    Handle<vector<int> > barCodes;
    evt.getByLabel(genParticles_, barCodes);
    Handle<GenParticleCollection> genParticles;
    evt.getByLabel(genParticles_, genParticles);
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
                if (f == e) throw edm::Exception(errors::InvalidReference)
                    << "found matching particle with barcode" << *f
                    << " which has not been found in " << genParticles_;
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

