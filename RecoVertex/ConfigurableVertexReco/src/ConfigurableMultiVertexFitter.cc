#include "RecoVertex/ConfigurableVertexReco/interface/ConfigurableMultiVertexFitter.h"
#include "RecoVertex/MultiVertexFit/interface/MultiVertexReconstructor.h"
#include "RecoVertex/ConfigurableVertexReco/interface/ConfigurableVertexReconstructor.h"
// #include "RecoVertex/ConfigurableVertexReco/interface/ConfigurableTrimmedKalmanFinder.h"
#include "RecoVertex/MultiVertexFit/interface/MultiVertexBSeeder.h"

namespace {
  edm::ParameterSet mydefaults() {
    edm::ParameterSet ret;
    ret.addParameter<double>("sigmacut", 9.0);
    ret.addParameter<double>("Tini", 8.0);
    ret.addParameter<double>("ratio", 0.25);
    ret.addParameter<int>("cheat", 0);
    edm::ParameterSet nest;
    nest.addParameter<std::string>("finder", "mbs");
    ret.addParameter<edm::ParameterSet>("ini", nest);
    return ret;
  }

  const AnnealingSchedule* schedule(const edm::ParameterSet& m) {
    return new GeometricAnnealing(
        m.getParameter<double>("sigmacut"), m.getParameter<double>("Tini"), m.getParameter<double>("ratio"));
  }

  const VertexReconstructor* initialiser(const edm::ParameterSet& p) {
    // std::cout << "[ConfigurableMultiVertexFitter] ini: " << p << std::endl;
    return new ConfigurableVertexReconstructor(p);
  }
}  // namespace

ConfigurableMultiVertexFitter::ConfigurableMultiVertexFitter()
    : theRector(new MultiVertexReconstructor(MultiVertexBSeeder())), theCheater(0) {}

void ConfigurableMultiVertexFitter::configure(const edm::ParameterSet& n) {
  edm::ParameterSet m = n;
  m.augment(mydefaults());
  // print ( m );
  const AnnealingSchedule* ann = schedule(m);
  const VertexReconstructor* ini = initialiser(m.getParameter<edm::ParameterSet>("ini"));
  if (theRector)
    delete theRector;
  theRector = new MultiVertexReconstructor(*ini, *ann);
  theCheater = m.getParameter<int>("cheat");
  delete ann;
  delete ini;
}

ConfigurableMultiVertexFitter::~ConfigurableMultiVertexFitter() {
  if (theRector)
    delete theRector;
}

ConfigurableMultiVertexFitter::ConfigurableMultiVertexFitter(const ConfigurableMultiVertexFitter& o)
    : theRector(o.theRector->clone()), theCheater(o.theCheater) {}

ConfigurableMultiVertexFitter* ConfigurableMultiVertexFitter::clone() const {
  return new ConfigurableMultiVertexFitter(*this);
}

std::vector<TransientVertex> ConfigurableMultiVertexFitter::vertices(const std::vector<reco::TransientTrack>& t,
                                                                     const reco::BeamSpot& s) const {
  return theRector->vertices(t, s);
}

std::vector<TransientVertex> ConfigurableMultiVertexFitter::vertices(const std::vector<reco::TransientTrack>& prims,
                                                                     const std::vector<reco::TransientTrack>& secs,
                                                                     const reco::BeamSpot& s) const {
  return theRector->vertices(prims, secs, s);
}

std::vector<TransientVertex> ConfigurableMultiVertexFitter::vertices(const std::vector<reco::TransientTrack>& t) const {
  return theRector->vertices(t);
}

edm::ParameterSet ConfigurableMultiVertexFitter::defaults() const { return mydefaults(); }

#include "RecoVertex/ConfigurableVertexReco/interface/ConfRecoBuilder.h"

namespace {
  const ConfRecoBuilder<ConfigurableMultiVertexFitter> t("mvf", "Multi Vertex Fitter");
}
