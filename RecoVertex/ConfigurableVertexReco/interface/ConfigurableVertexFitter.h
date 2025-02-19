#ifndef _ConfigurableVertexFitter_H_
#define _ConfigurableVertexFitter_H_

#include "RecoVertex/VertexPrimitives/interface/VertexFitter.h"
#include "RecoVertex/ConfigurableVertexReco/interface/AbstractConfFitter.h"
#include <string>
#include <map>

/**
 *  A VertexFitter whose concrete implementation
 *  (kalman filter, adaptive method, etc.) 
 *  is completely definable at runtime via
 *  a ParameterSet.
 *  Note that every fitter registers as a finder, also.
 */

class ConfigurableVertexFitter : public VertexFitter<5>
{
  public:

    typedef CachingVertex<5>::RefCountedVertexTrack RefCountedVertexTrack;

    ConfigurableVertexFitter ( const edm::ParameterSet & );
    ConfigurableVertexFitter ( const ConfigurableVertexFitter & o );
    ~ConfigurableVertexFitter();

    ConfigurableVertexFitter * clone () const;

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

  private:
    AbstractConfFitter * theFitter;
};

#endif
