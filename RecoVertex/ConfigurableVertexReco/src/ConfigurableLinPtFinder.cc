#include "RecoVertex/ConfigurableVertexReco/interface/ConfigurableLinPtFinder.h"
#include "RecoVertex/ConfigurableVertexReco/interface/ReconstructorFromLinPtFinder.h"
#include "RecoVertex/LinearizationPointFinders/interface/DefaultLinearizationPointFinder.h"

using namespace std;

namespace {
  edm::ParameterSet mydefaults()
  {
    edm::ParameterSet ret;
//    ret.addUntrackedParameter<string>("name","default");
    return ret;
  }
}
    

ConfigurableLinPtFinder::ConfigurableLinPtFinder() :
    theRector( new ReconstructorFromLinPtFinder ( DefaultLinearizationPointFinder()  ) )
{}

void ConfigurableLinPtFinder::configure(
    const edm::ParameterSet & n )
{
  edm::ParameterSet m = mydefaults();
  m.augment ( n );
  if ( theRector ) delete theRector;
  theRector = new ReconstructorFromLinPtFinder ( DefaultLinearizationPointFinder() );
}

ConfigurableLinPtFinder::~ConfigurableLinPtFinder()
{
  if ( theRector ) delete theRector;
}

ConfigurableLinPtFinder::ConfigurableLinPtFinder 
    ( const ConfigurableLinPtFinder & o ) :
  theRector ( o.theRector->clone() )
{}


ConfigurableLinPtFinder * ConfigurableLinPtFinder::clone() const
{
  return new ConfigurableLinPtFinder ( *this );
}

vector < TransientVertex > ConfigurableLinPtFinder::vertices ( 
    const std::vector < reco::TransientTrack > & t ) const
{
  return theRector->vertices ( t );
}

edm::ParameterSet ConfigurableLinPtFinder::defaults() const
{
  return mydefaults();
}

#include "RecoVertex/ConfigurableVertexReco/interface/ConfRecoBuilder.h"

namespace {
  ConfRecoBuilder < ConfigurableLinPtFinder > t ( "linpt", "LinearizationPointFinder" );
}
