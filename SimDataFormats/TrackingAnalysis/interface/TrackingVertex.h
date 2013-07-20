#ifndef SimDataFormats_TrackingVertex_h
#define SimDataFormats_TrackingVertex_h

/** \class TrackingVertex
 *
 * A simulated Vertex with links to TrackingParticles
 * for analysis of track and vertex reconstruction
 *
 * \version $Id: TrackingVertex.h,v 1.27 2013/06/24 13:07:32 speer Exp $
 *
 */

#include "DataFormats/Common/interface/RefVector.h"
#include "DataFormats/Math/interface/LorentzVector.h"
#include "DataFormats/Math/interface/Point3D.h"

#include "SimDataFormats/EncodedEventId/interface/EncodedEventId.h"
#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticleFwd.h"
#include "SimDataFormats/Track/interface/SimTrackContainer.h"
#include "SimDataFormats/Vertex/interface/SimVertexContainer.h"

class TrackingVertex
{

    friend std::ostream& operator<< (std::ostream& s, const TrackingVertex & tv);

public:

    typedef edm::RefVector<edm::HepMCProduct, HepMC::GenVertex > GenVertexRefVector;
    typedef edm::Ref<edm::HepMCProduct, HepMC::GenVertex >       GenVertexRef;
    typedef math::XYZTLorentzVectorD                             LorentzVector;
    typedef        GenVertexRefVector::iterator                  genv_iterator;
    typedef    std::vector<SimVertex>::const_iterator            g4v_iterator;
    typedef TrackingParticleRefVector::iterator                  tp_iterator;

// Default constructor and constructor from values
    TrackingVertex();
    TrackingVertex(const LorentzVector &position, const bool inVolume,
                   const EncodedEventId e = EncodedEventId(0));

// Setters
    void setEventId(EncodedEventId e)
    {
        eId_=e;
    };

// Track and vertex iterators
    genv_iterator genVertices_begin() const; // Ref's to HepMC and Geant4
    genv_iterator genVertices_end()   const; // vertices associated with
    g4v_iterator   g4Vertices_begin() const; // this vertex, respectively
    g4v_iterator   g4Vertices_end()   const; // ....

    tp_iterator    daughterTracks_begin() const; // Ref's to daughter and source
    tp_iterator    daughterTracks_end()   const; // tracks associated with
    tp_iterator      sourceTracks_begin() const; // this vertex, respectively
    tp_iterator      sourceTracks_end()   const; // ....

    unsigned int nG4Vertices()     const
    {
        return     g4Vertices_.size();
    };
    unsigned int nGenVertices()    const
    {
        return    genVertices_.size();
    };
    unsigned int nDaughterTracks() const
    {
        return daughterTracks_.size();
    };
    unsigned int nSourceTracks()   const
    {
        return   sourceTracks_.size();
    };

// Add references to TrackingParticles, Geant4, and HepMC vertices to containers
    void addG4Vertex(     const SimVertex&       );
    void addGenVertex(    const GenVertexRef&       );
    void addDaughterTrack(const TrackingParticleRef&);
    void addParentTrack(  const TrackingParticleRef&);
    void clearDaughterTracks();
    void clearParentTracks();

// Getters for RefVectors
    const std::vector<SimVertex>&        g4Vertices() const;
    const GenVertexRefVector&           genVertices() const;
    const TrackingParticleRefVector&   sourceTracks() const;
    const TrackingParticleRefVector& daughterTracks() const;

// Getters for other info
    const LorentzVector& position() const
    {
        return position_;
    };
    const EncodedEventId& eventId() const
    {
        return eId_;
    };
    const bool            inVolume() const
    {
        return inVolume_;
    };

private:

    LorentzVector  position_; // Vertex position and time
    bool           inVolume_; // Is it inside tracker volume?
    EncodedEventId eId_;

// References to G4 and generator vertices and TrackingParticles

    std::vector<SimVertex>        g4Vertices_;
    GenVertexRefVector           genVertices_;
    TrackingParticleRefVector daughterTracks_;
    TrackingParticleRefVector   sourceTracks_;
};

#endif
