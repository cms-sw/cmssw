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
    ~ConfigurableVertexFitter() override;

    ConfigurableVertexFitter * clone () const override;

    CachingVertex<5> vertex ( const std::vector < reco::TransientTrack > & t ) const override;
    CachingVertex<5> vertex( const std::vector<RefCountedVertexTrack> & tracks) const override;
    CachingVertex<5> vertex( const std::vector<RefCountedVertexTrack> & tracks,
        const reco::BeamSpot & spot ) const override;
    CachingVertex<5> vertex( const std::vector<reco::TransientTrack> & tracks, 
        const GlobalPoint& linPoint) const override;
    CachingVertex<5> vertex( const std::vector<reco::TransientTrack> & tracks, 
        const GlobalPoint& priorPos, const GlobalError& priorError) const override;
    CachingVertex<5> vertex( const std::vector<reco::TransientTrack> & tracks, 
                          const reco::BeamSpot& beamSpot) const override;
    CachingVertex<5> vertex(const std::vector<RefCountedVertexTrack> & tracks, 
      const GlobalPoint& priorPos, const GlobalError& priorError) const override;

  private:
    AbstractConfFitter * theFitter;
};

#endif
