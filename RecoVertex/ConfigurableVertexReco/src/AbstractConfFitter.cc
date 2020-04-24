#include "RecoVertex/ConfigurableVertexReco/interface/AbstractConfFitter.h"
    
AbstractConfFitter::AbstractConfFitter ( const VertexFitter<5> & f ) :
  theFitter ( f.clone() )
{}

AbstractConfFitter::AbstractConfFitter() : theFitter ( nullptr )
{}

AbstractConfFitter::AbstractConfFitter ( const AbstractConfFitter & o ) :
  theFitter ( o.theFitter->clone() )
{}

AbstractConfFitter::~AbstractConfFitter()
{
  if ( theFitter ) delete theFitter;
}

CachingVertex<5> AbstractConfFitter::vertex ( 
    const std::vector < reco::TransientTrack > & t ) const
{
  return theFitter->vertex ( t );
}

CachingVertex<5> AbstractConfFitter::vertex(
  const std::vector<RefCountedVertexTrack> & tracks) const
{
  return theFitter->vertex ( tracks );
}

CachingVertex<5> AbstractConfFitter::vertex(
  const std::vector<RefCountedVertexTrack> & tracks,
  const reco::BeamSpot & spot ) const
{
  return theFitter->vertex ( tracks, spot );
}


CachingVertex<5> AbstractConfFitter::vertex(
  const std::vector<reco::TransientTrack> & tracks, const GlobalPoint& linPoint) const
{
  return theFitter->vertex ( tracks, linPoint );
}

CachingVertex<5> AbstractConfFitter::vertex(
  const std::vector<reco::TransientTrack> & tracks, const GlobalPoint& priorPos,
  const GlobalError& priorError) const
{
  return theFitter->vertex ( tracks, priorPos, priorError );
}

CachingVertex<5> AbstractConfFitter::vertex(
  const std::vector<reco::TransientTrack> & tracks, const reco::BeamSpot& beamSpot) const
{
  return theFitter->vertex ( tracks, beamSpot );
}

CachingVertex<5> AbstractConfFitter::vertex(const std::vector<RefCountedVertexTrack> & tracks, 
  const GlobalPoint& priorPos, const GlobalError& priorError) const
{
  return theFitter->vertex ( tracks, priorPos, priorError );
}
