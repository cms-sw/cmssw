#include "RecoVertex/ConfigurableVertexReco/interface/AbstractConfFitter.h"
    
AbstractConfFitter::AbstractConfFitter ( const VertexFitter & f ) :
  theFitter ( f.clone() )
{}

AbstractConfFitter::AbstractConfFitter() : theFitter ( 0 )
{}

AbstractConfFitter::AbstractConfFitter ( const AbstractConfFitter & o ) :
  theFitter ( o.theFitter->clone() )
{}

AbstractConfFitter::~AbstractConfFitter()
{
  if ( theFitter ) delete theFitter;
}

CachingVertex AbstractConfFitter::vertex ( 
    const std::vector < reco::TransientTrack > & t ) const
{
  return theFitter->vertex ( t );
}

CachingVertex AbstractConfFitter::vertex(
  const vector<RefCountedVertexTrack> & tracks) const
{
  return theFitter->vertex ( tracks );
}

CachingVertex AbstractConfFitter::vertex(
  const vector<reco::TransientTrack> & tracks, const GlobalPoint& linPoint) const
{
  return theFitter->vertex ( tracks, linPoint );
}

CachingVertex AbstractConfFitter::vertex(
  const vector<reco::TransientTrack> & tracks, const GlobalPoint& priorPos,
  const GlobalError& priorError) const
{
  return theFitter->vertex ( tracks, priorPos, priorError );
}

CachingVertex AbstractConfFitter::vertex(
  const vector<reco::TransientTrack> & tracks, const reco::BeamSpot& beamSpot) const
{
  return theFitter->vertex ( tracks, beamSpot );
}

CachingVertex AbstractConfFitter::vertex(const vector<RefCountedVertexTrack> & tracks, 
  const GlobalPoint& priorPos, const GlobalError& priorError) const
{
  return theFitter->vertex ( tracks, priorPos, priorError );
}
