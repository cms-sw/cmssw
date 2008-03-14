#include "RecoVertex/ConfigurableVertexReco/interface/ConfigurableVertexReconstructor.h"
#include "RecoVertex/ConfigurableVertexReco/interface/VertexRecoManager.h"

using namespace std;

namespace {
  void errorNoReconstructor( const string & finder )
  {
    cout << "[ConfigurableVertexReconstructor] got no reconstructor for \""
         << finder << "\"" << endl;
    map < string, AbstractConfReconstructor * > valid = 
      VertexRecoManager::Instance().get();
    cout << "  Valid reconstructors are:";
    for ( map < string, AbstractConfReconstructor * >::const_iterator i=valid.begin(); 
          i!=valid.end() ; ++i )
    {
      if ( i->second ) cout << "  " << i->first;
    }
    cout << endl;
    throw string ( finder + " not available!" );
  }
}

ConfigurableVertexReconstructor::ConfigurableVertexReconstructor ( 
    const edm::ParameterSet & p ) : theRector ( 0 )
{
  string finder=p.getParameter<string>("finder");
  theRector = VertexRecoManager::Instance().get ( finder );
  if (!theRector)
  {
    errorNoReconstructor ( finder );
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
    const vector < reco::TransientTrack > & t ) const
{
  return theRector->vertices ( t );
}

vector < TransientVertex > ConfigurableVertexReconstructor::vertices ( 
    const vector < reco::TransientTrack > & t, const reco::BeamSpot & b ) const
{
  return theRector->vertices ( t, b );
}
