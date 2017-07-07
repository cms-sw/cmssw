#include "RecoVertex/ConfigurableVertexReco/interface/ConfigurableAdaptiveFitter.h"
#include "RecoVertex/ConfigurableVertexReco/interface/ReconstructorFromFitter.h"
#include "RecoVertex/AdaptiveVertexFit/interface/AdaptiveVertexFitter.h"
#include "RecoVertex/VertexTools/interface/GeometricAnnealing.h"
#include "RecoVertex/LinearizationPointFinders/interface/DefaultLinearizationPointFinder.h"
// #include "RecoVertex/LinearizationPointFinders/interface/GenericLinearizationPointFinder.h"
// #include "RecoVertex/KalmanVertexFit/interface/KalmanVertexFitter.h"
// #include "RecoVertex/LinearizationPointFinders/interface/ZeroLinearizationPointFinder.h"
#include "RecoVertex/KalmanVertexFit/interface/KalmanVertexUpdator.h"
#include "RecoVertex/KalmanVertexFit/interface/KalmanVertexTrackCompatibilityEstimator.h"
#include "RecoVertex/ConfigurableVertexReco/interface/ConfigurableAnnealing.h"
#include "RecoVertex/VertexTools/interface/DummyVertexSmoother.h"
#include "RecoVertex/KalmanVertexFit/interface/KalmanVertexSmoother.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

using namespace std;

namespace {
  edm::ParameterSet mydefaults()
  {
    edm::ParameterSet ret;
    ret.addParameter<string>("annealing", "geom" );
    ret.addParameter<bool>("smoothing", true );
    ret.addParameter<double>("sigmacut",3.0);
    ret.addParameter<double>("Tini",256.0);
    ret.addParameter<double>("ratio",0.25);

    ret.addParameter<double>("maxshift",0.0001);
    ret.addParameter<double>("maxlpshift",0.1);
    ret.addParameter<int>("maxstep",30);
    ret.addParameter<double>("weightthreshold",0.001);
    return ret;
  }
}

ConfigurableAdaptiveFitter::ConfigurableAdaptiveFitter() :
    AbstractConfFitter ( AdaptiveVertexFitter() )
{}

void ConfigurableAdaptiveFitter::configure(
    const edm::ParameterSet & n )
{
  edm::ParameterSet m=n;
  m.augment ( mydefaults() );
  ConfigurableAnnealing ann ( m );
  DefaultLinearizationPointFinder linpt;
  // ZeroLinearizationPointFinder linpt;
  // KalmanVertexFitter kvf;
  // GenericLinearizationPointFinder linpt ( kvf );
  KalmanVertexUpdator<5> updator;
  bool s=m.getParameter< bool >("smoothing");
  VertexSmoother<5> * smoother=nullptr;
  if ( s )
  {
    smoother = new KalmanVertexSmoother ();
  } else {
    smoother = new DummyVertexSmoother<5> ();
  }
  KalmanVertexTrackCompatibilityEstimator<5> estimator;

  if (theFitter) delete theFitter;
  AdaptiveVertexFitter * fitter = new AdaptiveVertexFitter ( ann, linpt, updator, estimator, *smoother );
  delete smoother;
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
