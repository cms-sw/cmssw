#include "SimDataFormats/TrackingAnalysis/interface/TrackingVertex.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticle.h"

#include "DataFormats/HepMCCandidate/interface/GenParticle.h"

#include <FWCore/MessageLogger/interface/MessageLogger.h>

const unsigned int TrackingParticle::longLivedTag = 65536;

TrackingParticle::TrackingParticle()
{
	// No operation
}

TrackingParticle::TrackingParticle( const SimTrack& simtrk, const TrackingVertexRef& parentVertex )
{
	addG4Track( simtrk );
	setParentVertex( parentVertex );
}

TrackingParticle::~TrackingParticle()
{
}

void TrackingParticle::addGenParticle( const reco::GenParticleRef& ref )
{
	genParticles_.push_back( ref );
}

void TrackingParticle::addG4Track( const SimTrack& t )
{
	g4Tracks_.push_back( t );
}

TrackingParticle::genp_iterator TrackingParticle::genParticle_begin() const
{
	return genParticles_.begin();
}

TrackingParticle::genp_iterator TrackingParticle::genParticle_end() const
{
	return genParticles_.end();
}

TrackingParticle::g4t_iterator TrackingParticle::g4Track_begin() const
{
	return g4Tracks_.begin();
}

TrackingParticle::g4t_iterator TrackingParticle::g4Track_end() const
{
	return g4Tracks_.end();
}

void TrackingParticle::setParentVertex( const TrackingVertexRef& ref )
{
	parentVertex_=ref;
}

void TrackingParticle::addDecayVertex( const TrackingVertexRef& ref )
{
	decayVertices_.push_back( ref );
}

void TrackingParticle::clearParentVertex()
{
	parentVertex_=TrackingVertexRef();
}

void TrackingParticle::clearDecayVertices()
{
	decayVertices_.clear();
}

int TrackingParticle::matchedHit() const
{
	edm::LogWarning("TrackingParticle") << "The method matchedHit() has been deprecated. Use numberOfTrackerLayers() instead.";
	return numberOfTrackerLayers_;
}


void TrackingParticle::setNumberOfHits( int numberOfHits )
{
    numberOfHits_=numberOfHits;
}

void TrackingParticle::setNumberOfTrackerHits( int numberOfTrackerHits )
{
    numberOfTrackerHits_=numberOfTrackerHits;
}

void TrackingParticle::setNumberOfTrackerLayers( const int numberOfTrackerLayers )
{
	numberOfTrackerLayers_=numberOfTrackerLayers;
}

std::ostream& operator<< (std::ostream& s, TrackingParticle const & tp)
{
    s << "TP momentum, q, ID, & Event #: "
    << tp.p4()                      << " " << tp.charge() << " "   << tp.pdgId() << " "
    << tp.eventId().bunchCrossing() << "." << tp.eventId().event() << std::endl;

    for (TrackingParticle::genp_iterator hepT = tp.genParticle_begin(); hepT !=  tp.genParticle_end(); ++hepT)
    {
        s << " HepMC Track Momentum " << (*hepT)->momentum().rho() << std::endl;
    }

    for (TrackingParticle::g4t_iterator g4T = tp.g4Track_begin(); g4T !=  tp.g4Track_end(); ++g4T)
    {
        s << " Geant Track Momentum  " << g4T->momentum() << std::endl;
        s << " Geant Track ID & type " << g4T->trackId() << " " << g4T->type() << std::endl;
        if (g4T->type() !=  tp.pdgId())
        {
            s << " Mismatch b/t TrackingParticle and Geant types" << std::endl;
        }
    }
    // Loop over decay vertices
    s << " TP Vertex " << tp.vertex() << std::endl;
    s << " Source vertex: " << tp.parentVertex()->position() << std::endl;
    s << " " << tp.decayVertices().size() << " Decay vertices" << std::endl;
    for (tv_iterator iTV = tp.decayVertices_begin(); iTV != tp.decayVertices_end(); ++iTV)
    {
        s << " Decay vertices:      " << (**iTV).position() << std::endl;
    }

    return s;
}
