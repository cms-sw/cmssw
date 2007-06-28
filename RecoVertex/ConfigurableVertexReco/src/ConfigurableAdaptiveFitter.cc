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

    ret.addParameter<double>("maxshift",0.0001);
    ret.addParameter<double>("maxlpshift",0.1);
    ret.addParameter<int>("maxstep",30);
    ret.addParameter<double>("weightthreshhold",0.001);
    return ret;
  }
}

ConfigurableAdaptiveFitter::ConfigurableAdaptiveFitter() :
    AbstractConfFitter ( AdaptiveVertexFitter() )
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

  if (theFitter) delete theFitter;
  AdaptiveVertexFitter * fitter = new AdaptiveVertexFitter ( ann, linpt, updator, estimator, smoother );
  fitter->setParameters ( m );
  theFitter=fitter;
}

ConfigurableAdaptiveFitter::~ConfigurableAdaptiveFitter()
{
  /*
  if (theFitter) delete theFitter;
  theFitter=0;
  */
}

ConfigurableAdaptiveFitter::ConfigurableAdaptiveFitter 
    ( const ConfigurableAdaptiveFitter & o ) :
  AbstractConfFitter ( o )
{}

ConfigurableAdaptiveFitter * ConfigurableAdaptiveFitter::clone() const
{
  return new ConfigurableAdaptiveFitter ( *this );
}

edm::ParameterSet ConfigurableAdaptiveFitter::defaults() const
{
  return mydefaults();
}

#include "RecoVertex/ConfigurableVertexReco/interface/ConfFitterBuilder.h"

namespace {
  ConfFitterBuilder < ConfigurableAdaptiveFitter > t ( "avf", "AdaptiveVertexFitter" );
}
