#include "RecoVertex/ConfigurableVertexReco/interface/ConfigurableKalmanFitter.h"
#include "RecoVertex/KalmanVertexFit/interface/KalmanVertexFitter.h"

namespace {
  edm::ParameterSet mydefaults() {
    edm::ParameterSet ret;
    ret.addParameter<double>("maxDistance", 0.01);
    ret.addParameter<int>("maxNbrOfIterations", 10);
    return ret;
  }
}  // namespace

ConfigurableKalmanFitter::ConfigurableKalmanFitter() : AbstractConfFitter(KalmanVertexFitter()) {}

void ConfigurableKalmanFitter::configure(const edm::ParameterSet& n) {
  edm::ParameterSet m = n;
  m.augment(mydefaults());
  if (theFitter)
    delete theFitter;
  theFitter = new KalmanVertexFitter(m);
}

ConfigurableKalmanFitter::~ConfigurableKalmanFitter() {
  // if (theFitter) delete theFitter;
}

ConfigurableKalmanFitter::ConfigurableKalmanFitter(const ConfigurableKalmanFitter& o) : AbstractConfFitter(o) {}

ConfigurableKalmanFitter* ConfigurableKalmanFitter::clone() const { return new ConfigurableKalmanFitter(*this); }

edm::ParameterSet ConfigurableKalmanFitter::defaults() const { return mydefaults(); }

#include "RecoVertex/ConfigurableVertexReco/interface/ConfFitterBuilder.h"

namespace {
  const ConfFitterBuilder<ConfigurableKalmanFitter> t("kalman", "Standard Kalman Filter");
  const ConfFitterBuilder<ConfigurableKalmanFitter> s("default", "Standard Kalman Filter");
}  // namespace
