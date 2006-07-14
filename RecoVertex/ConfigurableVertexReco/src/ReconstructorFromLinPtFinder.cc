#include "RecoVertex/ConfigurableVertexReco/interface/ReconstructorFromLinPtFinder.h"

using namespace std;

ReconstructorFromLinPtFinder::ReconstructorFromLinPtFinder (
    const LinearizationPointFinder & f, int verbose ) :
  theLinPtFinder ( f.clone() ), theVerbosity ( verbose )
{}

vector < TransientVertex > ReconstructorFromLinPtFinder::vertices
  ( const vector < reco::TransientTrack > & t )  const
{
  vector < TransientVertex > ret;
  try {
    GlobalPoint pt = theLinPtFinder->getLinearizationPoint ( t );
    TransientVertex tmp ( pt, GlobalError(), vector<reco::TransientTrack>(), -1.0 );
    ret.push_back ( tmp );
  } catch ( exception & e ) {
    if ( theVerbosity )
    {
      cout << "[ReconstructorFromLinPtFinder] exception caught: " << e.what() << endl;
    }
  } catch ( ... ) {
    if ( theVerbosity )
    {
      cout << "[ReconstructorFromLinPtFinder] unidentified exception caught." << endl;
    }
  }
  return ret;
}

ReconstructorFromLinPtFinder::~ReconstructorFromLinPtFinder()
{
  delete theLinPtFinder;
}

ReconstructorFromLinPtFinder::ReconstructorFromLinPtFinder ( const ReconstructorFromLinPtFinder & o ) :
  theLinPtFinder ( o.theLinPtFinder->clone() ), theVerbosity ( o.theVerbosity )
{}

ReconstructorFromLinPtFinder * ReconstructorFromLinPtFinder::clone () const
{
  return new ReconstructorFromLinPtFinder ( *this );
}
