#include "RecoVertex/ConfigurableVertexReco/interface/ConfigurableAnnealing.h"
#include "RecoVertex/VertexTools/interface/GeometricAnnealing.h"
#include "RecoVertex/VertexTools/interface/DeterministicAnnealing.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <string>

using namespace std;

ConfigurableAnnealing::ConfigurableAnnealing(const edm::ParameterSet& m) : theImpl(nullptr) {
  string type = m.getParameter<string>("annealing");
  // edm::LogWarning("ConfigurableAnnealing") << "below one code ist still here.";
  if (type == "below") {
    edm::LogError("ConfigurableAnnealing") << "below one annealing employed!";
    vector<float> sched;
    double final = m.getParameter<double>("Tfinal");
    sched.push_back(256.);
    sched.push_back(64.);
    sched.push_back(16.);
    sched.push_back(4.);
    sched.push_back(1.);
    sched.push_back(final);
    theImpl = new DeterministicAnnealing(sched, m.getParameter<double>("sigmacut"));
  } else if (type == "geom") {
    theImpl = new GeometricAnnealing(
        m.getParameter<double>("sigmacut"), m.getParameter<double>("Tini"), m.getParameter<double>("ratio"));
  } else {
    edm::LogError("ConfigurableAnnealing") << "annealing type " << type << " is not known.";
    exit(-1);
  }
}

ConfigurableAnnealing::ConfigurableAnnealing(const ConfigurableAnnealing& o) : theImpl(o.theImpl->clone()) {}

ConfigurableAnnealing* ConfigurableAnnealing::clone() const { return new ConfigurableAnnealing(*this); }

ConfigurableAnnealing::~ConfigurableAnnealing() { delete theImpl; }

void ConfigurableAnnealing::debug() const { theImpl->debug(); }

void ConfigurableAnnealing::anneal() { theImpl->anneal(); }

double ConfigurableAnnealing::weight(double chi2) const { return theImpl->weight(chi2); }

void ConfigurableAnnealing::resetAnnealing() { theImpl->resetAnnealing(); }

inline double ConfigurableAnnealing::phi(double chi2) const { return theImpl->phi(chi2); }

double ConfigurableAnnealing::cutoff() const { return theImpl->cutoff(); }

double ConfigurableAnnealing::currentTemp() const { return theImpl->currentTemp(); }

double ConfigurableAnnealing::initialTemp() const { return theImpl->initialTemp(); }

bool ConfigurableAnnealing::isAnnealed() const { return theImpl->isAnnealed(); }
