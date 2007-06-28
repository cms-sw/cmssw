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

class ConfigurableVertexFitter : public VertexFitter
{
  public:
    ConfigurableVertexFitter ( const edm::ParameterSet & );
    ConfigurableVertexFitter ( const ConfigurableVertexFitter & o );
    ~ConfigurableVertexFitter();

    ConfigurableVertexFitter * clone () const;

    CachingVertex vertex ( const std::vector < reco::TransientTrack > & t ) const;
    CachingVertex vertex( const vector<RefCountedVertexTrack> & tracks) const;
    CachingVertex vertex( const vector<reco::TransientTrack> & tracks, 
        const GlobalPoint& linPoint) const;
    CachingVertex vertex( const vector<reco::TransientTrack> & tracks, 
        const GlobalPoint& priorPos, const GlobalError& priorError) const;
    CachingVertex vertex( const vector<reco::TransientTrack> & tracks, 
                          const reco::BeamSpot& beamSpot) const;
    CachingVertex vertex(const vector<RefCountedVertexTrack> & tracks, 
      const GlobalPoint& priorPos, const GlobalError& priorError) const;

  private:
    AbstractConfFitter * theFitter;
};

#endif
