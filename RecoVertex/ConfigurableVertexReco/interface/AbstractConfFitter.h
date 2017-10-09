#ifndef _AbstractConfFitter_H_
#define _AbstractConfFitter_H_

#include "RecoVertex/VertexPrimitives/interface/VertexFitter.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

/**
 *  An abstract configurable reconstructor.
 *  must be configurable via ::configure
 */

class AbstractConfFitter : public VertexFitter<5>
{
  public:

    typedef CachingVertex<5>::RefCountedVertexTrack RefCountedVertexTrack;
    /** The configure method configures the vertex reconstructor.
     *  It also should also write all its applied defaults back into the map,
     */
    AbstractConfFitter ( const VertexFitter<5> & f );
    AbstractConfFitter ();
    AbstractConfFitter ( const AbstractConfFitter & );

    virtual void configure ( const edm::ParameterSet & ) = 0;
    virtual edm::ParameterSet defaults() const = 0;
    virtual ~AbstractConfFitter();
    AbstractConfFitter * clone() const = 0;

    CachingVertex<5> vertex ( const std::vector < reco::TransientTrack > & t ) const;
    CachingVertex<5> vertex( const std::vector<RefCountedVertexTrack> & tracks) const;
    CachingVertex<5> vertex( const std::vector<RefCountedVertexTrack> & tracks,
        const reco::BeamSpot & spot ) const;
    CachingVertex<5> vertex( const std::vector<reco::TransientTrack> & tracks, 
        const GlobalPoint& linPoint) const;
    CachingVertex<5> vertex( const std::vector<reco::TransientTrack> & tracks, 
        const GlobalPoint& priorPos, const GlobalError& priorError) const;
    CachingVertex<5> vertex( const std::vector<reco::TransientTrack> & tracks, 
                          const reco::BeamSpot& beamSpot) const;
    CachingVertex<5> vertex(const std::vector<RefCountedVertexTrack> & tracks, 
      const GlobalPoint& priorPos, const GlobalError& priorError) const;
 public:
    const VertexFitter<5> * theFitter;
};

#endif
