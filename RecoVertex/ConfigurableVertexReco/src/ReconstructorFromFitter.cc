#include "RecoVertex/ConfigurableVertexReco/interface/ReconstructorFromFitter.h"

using namespace std;

ReconstructorFromFitter::ReconstructorFromFitter ( const AbstractConfFitter & f ) :
  theFitter ( f.clone() )
{}

vector < TransientVertex > ReconstructorFromFitter::vertices
  ( const vector < reco::TransientTrack > & t )  const
{
  int verbose=1;
  vector < TransientVertex > ret;
  try {
    CachingVertex tmp = theFitter->vertex ( t );
    ret.push_back ( tmp );
  } catch ( exception & e ) {
    if ( verbose )
    {
      cout << "[ReconstructorFromFitter] exception caught: " << e.what() << endl;
    }
  } catch ( ... ) {
    if ( verbose )
    {
      cout << "[ReconstructorFromFitter] unidentified exception caught." << endl;
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
