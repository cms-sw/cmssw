#include "SimG4Core/CustomPhysics/interface/RHStopDump.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

RHStopDump::RHStopDump(edm::ParameterSet const& parameters)
    : mStream(parameters.getParameter<std::string>("stoppedFile").c_str()),
      mProducer(parameters.getUntrackedParameter<std::string>("producer", "g4SimHits")),
      tokNames_(consumes<std::vector<std::string> >(edm::InputTag(mProducer, "StoppedParticlesName"))),
      tokenXs_(consumes<std::vector<float> >(edm::InputTag(mProducer, "StoppedParticlesX"))),
      tokenYs_(consumes<std::vector<float> >(edm::InputTag(mProducer, "StoppedParticlesY"))),
      tokenZs_(consumes<std::vector<float> >(edm::InputTag(mProducer, "StoppedParticlesZ"))),
      tokenTs_(consumes<std::vector<float> >(edm::InputTag(mProducer, "StoppedParticlesTime"))),
      tokenIds_(consumes<std::vector<int> >(edm::InputTag(mProducer, "StoppedParticlesPdgId"))),
      tokenMasses_(consumes<std::vector<float> >(edm::InputTag(mProducer, "StoppedParticlesMass"))),
      tokenCharges_(consumes<std::vector<float> >(edm::InputTag(mProducer, "StoppedParticlesCharge"))) {}

void RHStopDump::analyze(const edm::Event& fEvent, const edm::EventSetup&) {
  const edm::Handle<std::vector<std::string> >& names = fEvent.getHandle(tokNames_);
  const edm::Handle<std::vector<float> >& xs = fEvent.getHandle(tokenXs_);
  const edm::Handle<std::vector<float> >& ys = fEvent.getHandle(tokenYs_);
  const edm::Handle<std::vector<float> >& zs = fEvent.getHandle(tokenZs_);
  const edm::Handle<std::vector<float> >& ts = fEvent.getHandle(tokenTs_);
  const edm::Handle<std::vector<int> >& ids = fEvent.getHandle(tokenIds_);
  const edm::Handle<std::vector<float> >& masses = fEvent.getHandle(tokenMasses_);
  const edm::Handle<std::vector<float> >& charges = fEvent.getHandle(tokenCharges_);

  if (names->size() != xs->size() || xs->size() != ys->size() || ys->size() != zs->size()) {
    edm::LogError("RHStopDump") << "mismatch array sizes name/x/y/z:" << names->size() << '/' << xs->size() << '/'
                                << ys->size() << '/' << zs->size() << std::endl;
  } else {
    for (size_t i = 0; i < names->size(); ++i) {
      mStream << (*names)[i] << ' ' << (*xs)[i] << ' ' << (*ys)[i] << ' ' << (*zs)[i] << ' ' << (*ts)[i] << std::endl;
      mStream << (*ids)[i] << ' ' << (*masses)[i] << ' ' << (*charges)[i] << std::endl;
    }
  }
}
