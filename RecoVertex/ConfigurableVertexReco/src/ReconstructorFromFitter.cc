#include "RecoVertex/ConfigurableVertexReco/interface/ReconstructorFromFitter.h"

using namespace std;

ReconstructorFromFitter::ReconstructorFromFitter ( const VertexFitter & f, int verbose ) :
  theFitter ( f.clone() ), theVerbosity ( verbose )
{}

vector < TransientVertex > ReconstructorFromFitter::vertices
  ( const vector < reco::TransientTrack > & t )  const
{
  vector < TransientVertex > ret;
  try {
    CachingVertex tmp = theFitter->vertex ( t );
    ret.push_back ( tmp );
  } catch ( exception & e ) {
    if ( theVerbosity )
    {
      cout << "[ReconstructorFromFitter] exception caught: " << e.what() << endl;
    }
  } catch ( ... ) {
    if ( theVerbosity )
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
  theFitter ( o.theFitter->clone() ), theVerbosity ( o.theVerbosity )
{}

ReconstructorFromFitter * ReconstructorFromFitter::clone () const
{
  return new ReconstructorFromFitter ( *this );
}
