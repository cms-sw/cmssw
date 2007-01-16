#include "RecoVertex/ConfigurableVertexReco/interface/ConfigurableAdaptiveFitter.h"
#include "RecoVertex/ConfigurableVertexReco/interface/ReconstructorFromFitter.h"
#include "RecoVertex/AdaptiveVertexFit/interface/AdaptiveVertexFitter.h"
#include "RecoVertex/VertexTools/interface/GeometricAnnealing.h"
#include "RecoVertex/LinearizationPointFinders/interface/DefaultLinearizationPointFinder.h"
#include "RecoVertex/KalmanVertexFit/interface/KalmanVertexUpdator.h"
#include "RecoVertex/KalmanVertexFit/interface/KalmanVertexTrackCompatibilityEstimator.h"
#include "RecoVertex/ConfigurableVertexReco/interface/ConfigurableAnnealing.h"

using namespace std;

namespace {
  edm::ParameterSet mydefaults()
  {
    edm::ParameterSet ret;
    ret.addParameter<string>("annealing", "geom" );
    ret.addParameter<double>("sigmacut",3.0);
    ret.addParameter<double>("Tini",256.0);
    ret.addParameter<double>("ratio",0.25);
    return ret;
  }
}

ConfigurableAdaptiveFitter::ConfigurableAdaptiveFitter() :
    theRector( new ReconstructorFromFitter ( AdaptiveVertexFitter()  ) )
{}

void ConfigurableAdaptiveFitter::configure(
    const edm::ParameterSet & n )
{
  edm::ParameterSet m = mydefaults();
  m.augment ( n );
  ConfigurableAnnealing ann ( m );
  DefaultLinearizationPointFinder linpt;
  KalmanVertexUpdator updator;
  DummyVertexSmoother smoother;
  KalmanVertexTrackCompatibilityEstimator estimator;

  AdaptiveVertexFitter fitter ( ann, linpt, updator, estimator, smoother );

  if ( theRector ) delete theRector;
  theRector = new ReconstructorFromFitter ( fitter );
}

ConfigurableAdaptiveFitter::~ConfigurableAdaptiveFitter()
{
  delete theRector;
}

ConfigurableAdaptiveFitter::ConfigurableAdaptiveFitter 
    ( const ConfigurableAdaptiveFitter & o ) :
  theRector ( o.theRector->clone() )
{}

ConfigurableAdaptiveFitter * ConfigurableAdaptiveFitter::clone() const
{
  return new ConfigurableAdaptiveFitter ( *this );
}

vector < TransientVertex > ConfigurableAdaptiveFitter::vertices ( 
    const std::vector < reco::TransientTrack > & t ) const
{
  return theRector->vertices ( t );
}

edm::ParameterSet ConfigurableAdaptiveFitter::defaults() const
{
  return mydefaults();
}

#include "RecoVertex/ConfigurableVertexReco/interface/ConfRecoBuilder.h"

namespace {
  ConfRecoBuilder < ConfigurableAdaptiveFitter > t ( "avf", "Adaptive Vertex Filter" );
}
