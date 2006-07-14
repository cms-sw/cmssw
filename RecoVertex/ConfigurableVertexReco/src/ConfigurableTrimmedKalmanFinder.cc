#include "RecoVertex/ConfigurableVertexReco/interface/ConfigurableTrimmedKalmanFinder.h"
#include "RecoVertex/TrimmedKalmanVertexFinder/interface/KalmanTrimmedVertexFinder.h"

namespace {
  edm::ParameterSet mydefaults ()
  {
    edm::ParameterSet ret;
    ret.addUntrackedParameter<double>("ptcut",0.);
    ret.addUntrackedParameter<double>("trkcutpv",0.05);
    ret.addUntrackedParameter<double>("trkcutsv",0.01);
    ret.addUntrackedParameter<double>("vtxcut",0.01);
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
  edm::ParameterSet m = mydefaults();
  m.augment ( n );
  KalmanTrimmedVertexFinder * tmp = new KalmanTrimmedVertexFinder();
  tmp->setPtCut ( m.getUntrackedParameter<double>("ptcut") );
  tmp->setTrackCompatibilityCut ( m.getUntrackedParameter<double>("trkcutpv") );
  tmp->setTrackCompatibilityToSV ( m.getUntrackedParameter<double>("trkcutsv") );
  tmp->setVertexFitProbabilityCut ( m.getUntrackedParameter<double>( "vtxcut" ) );
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

vector < TransientVertex > ConfigurableTrimmedKalmanFinder::vertices ( 
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
