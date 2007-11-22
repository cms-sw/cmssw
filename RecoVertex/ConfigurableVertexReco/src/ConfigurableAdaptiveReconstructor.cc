#include "RecoVertex/ConfigurableVertexReco/interface/ConfigurableAdaptiveReconstructor.h"
#include "RecoVertex/AdaptiveVertexFinder/interface/AdaptiveVertexReconstructor.h"

using namespace std;

namespace {
  edm::ParameterSet mydefaults()
  {
    edm::ParameterSet ret;
    ret.addParameter<double>("primcut",2.0);
    ret.addParameter<double>("seccut",6.0);
    ret.addParameter<double>("minweight",0.5);
    ret.addParameter<bool>("smoothing",0);
    ret.addParameter<double>("weightthreshold",0.001 );
    return ret;
  }
}

ConfigurableAdaptiveReconstructor::ConfigurableAdaptiveReconstructor() :
    theRector( new AdaptiveVertexReconstructor() )
{}

void ConfigurableAdaptiveReconstructor::configure(
    const edm::ParameterSet & n )
{
  edm::ParameterSet m=n;
  m.augment ( mydefaults() );
  if ( theRector ) delete theRector;
  theRector = new AdaptiveVertexReconstructor( m );
}

ConfigurableAdaptiveReconstructor::~ConfigurableAdaptiveReconstructor()
{
  if ( theRector ) delete theRector;
}

ConfigurableAdaptiveReconstructor::ConfigurableAdaptiveReconstructor 
    ( const ConfigurableAdaptiveReconstructor & o ) :
  theRector ( o.theRector->clone() )
{}


ConfigurableAdaptiveReconstructor * ConfigurableAdaptiveReconstructor::clone() const
{
  return new ConfigurableAdaptiveReconstructor ( *this );
}

vector < TransientVertex > ConfigurableAdaptiveReconstructor::vertices ( 
    const std::vector < reco::TransientTrack > & t ) const
{
  return theRector->vertices ( t );
}

vector < TransientVertex > ConfigurableAdaptiveReconstructor::vertices ( 
    const std::vector < reco::TransientTrack > & t, const reco::BeamSpot & s ) const
{
  return theRector->vertices ( t, s );
}


edm::ParameterSet ConfigurableAdaptiveReconstructor::defaults() const
{
  return mydefaults();
}

#include "RecoVertex/ConfigurableVertexReco/interface/ConfRecoBuilder.h"

namespace {
  ConfRecoBuilder < ConfigurableAdaptiveReconstructor > t
    ( "avr", "Adaptive Vertex Reconstructor [ = Iterative avf]" );
  // ConfRecoBuilder < ConfigurableAdaptiveReconstructor > s ( "default", "Adaptive Vertex Reconstructor [ = Iterative avf]" );
}
