#include "RecoVertex/ConfigurableVertexReco/interface/ConfigurableVertexReconstructor.h"
#include "RecoVertex/ConfigurableVertexReco/interface/VertexRecoManager.h"

using namespace std;

ConfigurableVertexReconstructor::ConfigurableVertexReconstructor ( 
    const edm::ParameterSet & p ) : theRector ( 0 )
{
  string finder=p.getParameter<string>("finder");
  theRector = VertexRecoManager::Instance().get ( finder );
  if (!theRector)
  {
    cout << "[ConfigurableVertexReconstructor] got no reconstructor for \""
         << finder << "\"" << endl;
    exit(0);
  }
  theRector->configure ( p );
  // theRector = theRector->clone();
  // theRector = new ReconstructorFromFitter ( KalmanVertexFitter() );
}

ConfigurableVertexReconstructor::~ConfigurableVertexReconstructor()
{
//  delete theRector;
}

ConfigurableVertexReconstructor::ConfigurableVertexReconstructor 
    ( const ConfigurableVertexReconstructor & o ) :
  theRector ( o.theRector->clone() )
{}


ConfigurableVertexReconstructor * ConfigurableVertexReconstructor::clone() const
{
  return new ConfigurableVertexReconstructor ( *this );
}


vector < TransientVertex > ConfigurableVertexReconstructor::vertices ( 
    const std::vector < reco::TransientTrack > & t ) const
{
  return theRector->vertices ( t );
}
