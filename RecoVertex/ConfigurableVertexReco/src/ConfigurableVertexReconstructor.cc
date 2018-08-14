#include "RecoVertex/ConfigurableVertexReco/interface/ConfigurableVertexReconstructor.h"
#include "RecoVertex/ConfigurableVertexReco/interface/VertexRecoManager.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

using namespace std;

namespace {
  void errorNoReconstructor( const string & finder )
  {
    edm::LogError ( "ConfigurableVertexReconstructor") << "got no reconstructor for \""
         << finder << "\"";
    vector <string> valid = 
      VertexRecoManager::Instance().getNames();
    cout << "  Valid reconstructors are:";
    for (const auto& i : valid)
    {
      cout <<" "<<i;
    }
    cout << endl;
    throw std::string ( finder + " not available!" );
  }
}

ConfigurableVertexReconstructor::ConfigurableVertexReconstructor ( 
    const edm::ParameterSet & p ) : theRector ( nullptr )
{
  string finder=p.getParameter<string>("finder");
  theRector = VertexRecoManager::Instance().get ( finder ).release();
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
  delete theRector;
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
    const std::vector < reco::TransientTrack > & prims,
    const std::vector < reco::TransientTrack > & secs,
    const reco::BeamSpot & s ) const
{
  return theRector->vertices ( prims, secs, s );
}

vector < TransientVertex > ConfigurableVertexReconstructor::vertices ( 
    const std::vector < reco::TransientTrack > & t,
    const reco::BeamSpot & s ) const
{
  return theRector->vertices ( t, s );
}

vector < TransientVertex > ConfigurableVertexReconstructor::vertices ( 
    const std::vector < reco::TransientTrack > & t ) const
{
  return theRector->vertices ( t );
}
