#include "RecoVertex/ConfigurableVertexReco/interface/ConfigurableTrimmedKalmanFinder.h"
#include "RecoVertex/TrimmedKalmanVertexFinder/interface/KalmanTrimmedVertexFinder.h"

namespace {
  edm::ParameterSet mydefaults ()
  {
    edm::ParameterSet ret;
    ret.addParameter<double>("ptcut",0.);
    ret.addParameter<double>("trkcutpv",0.05);
    ret.addParameter<double>("trkcutsv",0.01);
    ret.addParameter<double>("vtxcut",0.01);
    return ret;
  }
}

ConfigurableTrimmedKalmanFinder::ConfigurableTrimmedKalmanFinder() :
    theRector( new KalmanTrimmedVertexFinder() )
{}

void ConfigurableTrimmedKalmanFinder::configure(
    const edm::ParameterSet & n )
{
  if ( theRector ) delete theRector;
  edm::ParameterSet m=n;
  m.augment ( mydefaults() );
  KalmanTrimmedVertexFinder * tmp = new KalmanTrimmedVertexFinder();
  tmp->setPtCut ( m.getParameter<double>("ptcut") );
  tmp->setTrackCompatibilityCut ( m.getParameter<double>("trkcutpv") );
  tmp->setTrackCompatibilityToSV ( m.getParameter<double>("trkcutsv") );
  tmp->setVertexFitProbabilityCut ( m.getParameter<double>( "vtxcut" ) );
  theRector = tmp;
}

ConfigurableTrimmedKalmanFinder::~ConfigurableTrimmedKalmanFinder()
{
  if ( theRector ) delete theRector;
}

ConfigurableTrimmedKalmanFinder::ConfigurableTrimmedKalmanFinder 
    ( const ConfigurableTrimmedKalmanFinder & o ) :
  theRector ( o.theRector->clone() )
{}


ConfigurableTrimmedKalmanFinder * ConfigurableTrimmedKalmanFinder::clone() const
{
  return new ConfigurableTrimmedKalmanFinder ( *this );
}

std::vector < TransientVertex > ConfigurableTrimmedKalmanFinder::vertices ( 
    const std::vector < reco::TransientTrack > & t,
    const reco::BeamSpot & s ) const
{
  return theRector->vertices ( t, s );
}

std::vector < TransientVertex > ConfigurableTrimmedKalmanFinder::vertices ( 
    const std::vector < reco::TransientTrack > & prims,
    const std::vector < reco::TransientTrack > & secs,
    const reco::BeamSpot & s ) const
{
  return theRector->vertices ( prims, secs, s );
}

std::vector < TransientVertex > ConfigurableTrimmedKalmanFinder::vertices ( 
    const std::vector < reco::TransientTrack > & t ) const
{
  return theRector->vertices ( t );
}

edm::ParameterSet ConfigurableTrimmedKalmanFinder::defaults() const
{
  return mydefaults();
}

#include "RecoVertex/ConfigurableVertexReco/interface/ConfRecoBuilder.h"

namespace {
  ConfRecoBuilder < ConfigurableTrimmedKalmanFinder > t ( "tkf", "Trimmed Kalman Vertex Finder" );
}
