#include "SimDataFormats/TrackingAnalysis/interface/TrackingVertexContainer.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingVertex.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticle.h"

typedef edm::RefVector<edm::HepMCProduct, HepMC::GenVertex > GenVertexRefVector;
typedef edm::Ref<edm::HepMCProduct, HepMC::GenVertex >       GenVertexRef;

// Constructors

TrackingVertex::TrackingVertex() : position_(LorentzVector(0,0,0,0)), eId_(0)
{
//  daughterTracks_.clear();
}

TrackingVertex::TrackingVertex(const LorentzVector &p, const bool inVolume, const EncodedEventId eId) :
        position_(p), inVolume_(inVolume), eId_(eId)
{
//  daughterTracks_.clear();
}

// Add a reference to vertex vectors

void TrackingVertex::addG4Vertex(const SimVertex& v)
{
    g4Vertices_.push_back(v);
}

void TrackingVertex::addGenVertex(const GenVertexRef &ref)
{
    genVertices_.push_back(ref);
}

// Add a reference to track vectors

void TrackingVertex::addDaughterTrack(const TrackingParticleRef &ref)
{
    daughterTracks_.push_back(ref);
}

void TrackingVertex::addParentTrack(const TrackingParticleRef &ref)
{
    sourceTracks_.push_back(ref);
}

void TrackingVertex::clearDaughterTracks()
{
    daughterTracks_.clear();
}

void TrackingVertex::clearParentTracks()
{
    sourceTracks_.clear();
}


// Iterators over vertices and tracks

TrackingVertex::genv_iterator TrackingVertex::genVertices_begin() const
{
    return genVertices_.begin();
}
TrackingVertex::genv_iterator TrackingVertex::genVertices_end()   const
{
    return genVertices_.end();
}
TrackingVertex::g4v_iterator  TrackingVertex::g4Vertices_begin()  const
{
    return  g4Vertices_.begin();
}
TrackingVertex::g4v_iterator  TrackingVertex::g4Vertices_end()    const
{
    return  g4Vertices_.end();
}

TrackingVertex::tp_iterator TrackingVertex::daughterTracks_begin() const
{
    return daughterTracks_.begin();
}
TrackingVertex::tp_iterator TrackingVertex::daughterTracks_end()   const
{
    return daughterTracks_.end();
}
TrackingVertex::tp_iterator TrackingVertex::sourceTracks_begin()   const
{
    return sourceTracks_.begin();
}
TrackingVertex::tp_iterator TrackingVertex::sourceTracks_end()     const
{
    return sourceTracks_.end();
}

// Accessors for whole vectors

const std::vector<SimVertex>&    TrackingVertex::g4Vertices()     const
{
    return  g4Vertices_;
}
const GenVertexRefVector&        TrackingVertex::genVertices()    const
{
    return  genVertices_;
}
const TrackingParticleRefVector& TrackingVertex::sourceTracks()   const
{
    return  sourceTracks_;
}
const TrackingParticleRefVector& TrackingVertex::daughterTracks() const
{
    return  daughterTracks_;
}

std::ostream& operator<< (std::ostream& s, const TrackingVertex & v)
{

    using std::endl;
    typedef        GenVertexRefVector::iterator                  genv_iterator;
    typedef    std::vector<SimVertex>::const_iterator            g4v_iterator;
    typedef TrackingParticleRefVector::iterator                  tp_iterator;
    typedef       std::vector<SimTrack>::const_iterator             g4t_iterator;

    s << "Vertex Position & Event #" << v.position() << " " << v.eventId().bunchCrossing() << "." << v.eventId().event() << endl;
    s << " Associated with " << v.daughterTracks().size() << " tracks" << endl;
    for (genv_iterator genV = v.genVertices_begin(); genV != v.genVertices_end(); ++genV)
    {
        s << " HepMC vertex position " << (*(*genV)).position().x() << ","<< (*(*genV)).position().y() << (*(*genV)).position().z()  << endl;
    }

    for (g4v_iterator g4V = v.g4Vertices_begin(); g4V != v.g4Vertices_end(); ++g4V)
    {
        s << " Geant vertex position " << (*g4V).position() << endl;
        // Probably empty all the time, currently
    }

    // Loop over daughter track(s)
    for (tp_iterator iTP = v.daughterTracks_begin(); iTP != v.daughterTracks_end(); ++iTP)
    {
        s << " Daughter starts:      " << (*(*iTP)).vertex();
        for (g4t_iterator g4T  = (*(*iTP)).g4Track_begin(); g4T != (*(*iTP)).g4Track_end(); ++g4T)
        {
            s << " p " << g4T->momentum();
        }
        s << endl;
    }

    // Loop over source track(s) (can be multiple since vertices are collapsed)
    for (tp_iterator iTP = v.sourceTracks_begin(); iTP != v.sourceTracks_end(); ++iTP)
    {
        s << " Source   starts: " << (*(*iTP)).vertex();
        for (g4t_iterator g4T  = (*iTP)->g4Track_begin(); g4T != (*iTP)->g4Track_end(); ++g4T)
        {
            s << ", p " <<  g4T ->momentum();
        }
        s << endl;
    }
    return s;
}

