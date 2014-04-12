#include "RecoVertex/ConfigurableVertexReco/interface/ConfigurableVertexFitter.h"
#include "RecoVertex/ConfigurableVertexReco/interface/VertexFitterManager.h"

using namespace std;

namespace {
  void errorNoFitter( const string & finder )
  {
    cout << "[ConfigurableVertexFitter] got no fitter for \""
         << finder << "\"" << endl;
    map < string, AbstractConfFitter * > valid = 
      VertexFitterManager::Instance().get();
    cout << "  Valid fitters are:";
    for ( map < string, AbstractConfFitter * >::const_iterator i=valid.begin(); 
          i!=valid.end() ; ++i )
    {
      if ( i->second ) cout << "  " << i->first;
    }
    cout << endl;
    throw std::string ( finder + " not available!" );
  }
}

ConfigurableVertexFitter::ConfigurableVertexFitter ( 
    const edm::ParameterSet & p ) : theFitter ( 0 )
{
  string fitter=p.getParameter<string>("fitter");
  theFitter = VertexFitterManager::Instance().get ( fitter );
  if (!theFitter)
  {
    errorNoFitter ( fitter );
  }
  theFitter->configure ( p );
}

ConfigurableVertexFitter::~ConfigurableVertexFitter()
{
}

ConfigurableVertexFitter::ConfigurableVertexFitter 
    ( const ConfigurableVertexFitter & o ) :
  theFitter ( o.theFitter->clone() )
{}


ConfigurableVertexFitter * ConfigurableVertexFitter::clone() const
{
  return new ConfigurableVertexFitter ( *this );
}


CachingVertex<5> ConfigurableVertexFitter::vertex ( 
    const std::vector < reco::TransientTrack > & t ) const
{
  return theFitter->vertex ( t );
}

CachingVertex<5> ConfigurableVertexFitter::vertex(
  const vector<RefCountedVertexTrack> & tracks) const
{
  return theFitter->vertex ( tracks );
}

CachingVertex<5> ConfigurableVertexFitter::vertex(
  const vector<RefCountedVertexTrack> & tracks,
  const reco::BeamSpot & spot ) const
{
  return theFitter->vertex ( tracks, spot );
}


CachingVertex<5> ConfigurableVertexFitter::vertex(
  const vector<reco::TransientTrack> & tracks, const GlobalPoint& linPoint) const
{
  return theFitter->vertex ( tracks, linPoint );
}

CachingVertex<5> ConfigurableVertexFitter::vertex(
  const vector<reco::TransientTrack> & tracks, const GlobalPoint& priorPos,
  const GlobalError& priorError) const
{
  return theFitter->vertex ( tracks, priorPos, priorError );
}

CachingVertex<5> ConfigurableVertexFitter::vertex(
  const vector<reco::TransientTrack> & tracks, const reco::BeamSpot& beamSpot) const
{
  return theFitter->vertex ( tracks, beamSpot );
}

CachingVertex<5> ConfigurableVertexFitter::vertex(const vector<RefCountedVertexTrack> & tracks, 
  const GlobalPoint& priorPos, const GlobalError& priorError) const
{
  return theFitter->vertex ( tracks, priorPos, priorError );
}

