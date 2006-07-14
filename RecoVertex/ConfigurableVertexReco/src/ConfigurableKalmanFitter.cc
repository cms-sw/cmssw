#include "RecoVertex/ConfigurableVertexReco/interface/ConfigurableKalmanFitter.h"
#include "RecoVertex/ConfigurableVertexReco/interface/ReconstructorFromFitter.h"
#include "RecoVertex/KalmanVertexFit/interface/KalmanVertexFitter.h"

namespace {
  edm::ParameterSet mydefaults()
  {
    edm::ParameterSet ret;
    return ret;
  }
}

ConfigurableKalmanFitter::ConfigurableKalmanFitter() :
    theRector( new ReconstructorFromFitter ( KalmanVertexFitter()  ) )
{}

void ConfigurableKalmanFitter::configure(
    const edm::ParameterSet & n )
{
  edm::ParameterSet m = mydefaults();
  m.augment ( n );
  if ( theRector ) delete theRector;
  theRector = new ReconstructorFromFitter ( KalmanVertexFitter() );
}

ConfigurableKalmanFitter::~ConfigurableKalmanFitter()
{
  if ( theRector ) delete theRector;
}

ConfigurableKalmanFitter::ConfigurableKalmanFitter 
    ( const ConfigurableKalmanFitter & o ) :
  theRector ( o.theRector->clone() )
{}


ConfigurableKalmanFitter * ConfigurableKalmanFitter::clone() const
{
  return new ConfigurableKalmanFitter ( *this );
}

vector < TransientVertex > ConfigurableKalmanFitter::vertices ( 
    const std::vector < reco::TransientTrack > & t ) const
{
  return theRector->vertices ( t );
}

edm::ParameterSet ConfigurableKalmanFitter::defaults() const
{
  return mydefaults();
}

#include "RecoVertex/ConfigurableVertexReco/interface/ConfRecoBuilder.h"

namespace {
  ConfRecoBuilder < ConfigurableKalmanFitter > t ( "kalman", "Standard Kalman Filter" );
  ConfRecoBuilder < ConfigurableKalmanFitter > s ( "default", "Standard Kalman Filter" );
}
