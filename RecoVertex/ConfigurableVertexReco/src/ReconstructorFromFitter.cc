#include "RecoVertex/ConfigurableVertexReco/interface/ReconstructorFromFitter.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

using namespace std;

ReconstructorFromFitter::ReconstructorFromFitter ( const AbstractConfFitter & f ) :
  theFitter ( f.clone() )
{}

vector < TransientVertex > ReconstructorFromFitter::vertices
  ( const vector < reco::TransientTrack > & t )  const
{
  int verbose=1;
  vector < TransientVertex > ret;
  // cout << "[ReconstructorFromFitter] debug: fitting without bs!" << endl; 
  try {
    CachingVertex<5> tmp = theFitter->vertex ( t );
    ret.push_back ( tmp );
  } catch ( exception & e ) {
    if ( verbose )
    {
      edm::LogWarning("ReconstructorFromFitter") << "exception caught: " << e.what();
    }
  }
  return ret;
}

vector < TransientVertex > ReconstructorFromFitter::vertices
  ( const vector < reco::TransientTrack > & t, const reco::BeamSpot & s )  const
{
  int verbose=1;
  vector < TransientVertex > ret;
  try {
    /*
    cout << "[ReconstructorFromFitter] debug: fitting with s: " << s.BeamWidth() 
         << " sz=" << s.sigmaZ() << endl;
         */
    CachingVertex<5> tmp = theFitter->vertex ( t, s );
    ret.push_back ( tmp );
  } catch ( exception & e ) {
    if ( verbose )
    {
      edm::LogWarning("ReconstructorFromFitter") << "exception caught: " << e.what();
    }
  }
  return ret;
}

ReconstructorFromFitter::~ReconstructorFromFitter()
{
  delete theFitter;
}

ReconstructorFromFitter::ReconstructorFromFitter ( const ReconstructorFromFitter & o ) :
  theFitter ( o.theFitter->clone() )
{}

edm::ParameterSet ReconstructorFromFitter::defaults() const
{
  return theFitter->defaults();
}

void ReconstructorFromFitter::configure ( const edm::ParameterSet & s )
{
  const_cast < AbstractConfFitter *> (theFitter)->configure (s );
}

ReconstructorFromFitter * ReconstructorFromFitter::clone () const
{
  return new ReconstructorFromFitter ( *this );
}
