/*
 *  TrackHistoryAnalyzer.C
 *
 *  Created by Victor Eduardo Bazterra on 5/31/07.
 *
 */

// system include files
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

// user include files

#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/VertexReco/interface/Vertex.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "SimGeneral/HepPDTRecord/interface/ParticleDataTable.h"
#include "SimTracker/TrackHistory/interface/TrackClassifier.h"

//
// class decleration
//

class TrackHistoryAnalyzer : public edm::one::EDAnalyzer<edm::one::WatchRuns> {
public:
  explicit TrackHistoryAnalyzer(const edm::ParameterSet &);
  ~TrackHistoryAnalyzer() override = default;

private:
  void beginJob() override;
  void beginRun(const edm::Run &, const edm::EventSetup &) override;
  void analyze(const edm::Event &, const edm::EventSetup &) override;
  void endRun(const edm::Run &, const edm::EventSetup &) override{};

  // Member data
  const edm::ESGetToken<ParticleDataTable, PDTRecord> pdtToken_;
  const edm::EDGetTokenT<edm::View<reco::Track>> trkToken_;

  std::size_t totalTracks_;

  edm::ESHandle<ParticleDataTable> pdt_;

  std::string particleString(int) const;

  TrackClassifier classifier_;

  std::string vertexString(const TrackingParticleRefVector &, const TrackingParticleRefVector &) const;

  std::string vertexString(HepMC::GenVertex::particles_in_const_iterator,
                           HepMC::GenVertex::particles_in_const_iterator,
                           HepMC::GenVertex::particles_out_const_iterator,
                           HepMC::GenVertex::particles_out_const_iterator) const;
};

TrackHistoryAnalyzer::TrackHistoryAnalyzer(const edm::ParameterSet &config)
    : pdtToken_(esConsumes<edm::Transition::BeginRun>()),
      trkToken_(consumes<edm::View<reco::Track>>(config.getUntrackedParameter<edm::InputTag>("trackProducer"))),
      classifier_(config, consumesCollector()) {}

void TrackHistoryAnalyzer::analyze(const edm::Event &event, const edm::EventSetup &setup) {
  // Track collection
  edm::Handle<edm::View<reco::Track>> trackCollection;
  event.getByToken(trkToken_, trackCollection);

  // Set the classifier for a new event
  classifier_.newEvent(event, setup);

  // Get a constant reference to the track history associated to the classifier
  TrackHistory const &tracer = classifier_.history();

  // Loop over the track collection.
  for (std::size_t index = 0; index < trackCollection->size(); index++) {
    edm::LogPrint("TrackHistoryAnalyzer") << std::endl << "History for track #" << index << " : ";

    // Classify the track and detect for fakes
    if (!classifier_.evaluate(reco::TrackBaseRef(trackCollection, index)).is(TrackClassifier::Fake)) {
      // Get the list of TrackingParticles associated to
      TrackHistory::SimParticleTrail simParticles(tracer.simParticleTrail());

      // Loop over all simParticles
      for (std::size_t hindex = 0; hindex < simParticles.size(); hindex++) {
        edm::LogPrint("TrackHistoryAnalyzer")
            << "  simParticles [" << hindex << "] : " << particleString(simParticles[hindex]->pdgId());
      }

      // Get the list of TrackingVertexes associated to
      TrackHistory::SimVertexTrail simVertexes(tracer.simVertexTrail());

      // Loop over all simVertexes
      if (!simVertexes.empty()) {
        for (std::size_t hindex = 0; hindex < simVertexes.size(); hindex++) {
          edm::LogPrint("TrackHistoryAnalyzer")
              << "  simVertex    [" << hindex
              << "] : " << vertexString(simVertexes[hindex]->sourceTracks(), simVertexes[hindex]->daughterTracks());
        }
      } else
        edm::LogPrint("TrackHistoryAnalyzer") << "  simVertex no found";

      // Get the list of GenParticles associated to
      TrackHistory::GenParticleTrail genParticles(tracer.genParticleTrail());

      // Loop over all genParticles
      for (std::size_t hindex = 0; hindex < genParticles.size(); hindex++) {
        edm::LogPrint("TrackHistoryAnalyzer")
            << "  genParticles [" << hindex << "] : " << particleString(genParticles[hindex]->pdg_id());
      }

      // Get the list of TrackingVertexes associated to
      TrackHistory::GenVertexTrail genVertexes(tracer.genVertexTrail());

      // Loop over all simVertexes
      if (!genVertexes.empty()) {
        for (std::size_t hindex = 0; hindex < genVertexes.size(); hindex++) {
          edm::LogPrint("TrackHistoryAnalyzer") << "  genVertex    [" << hindex << "] : "
                                                << vertexString(genVertexes[hindex]->particles_in_const_begin(),
                                                                genVertexes[hindex]->particles_in_const_end(),
                                                                genVertexes[hindex]->particles_out_const_begin(),
                                                                genVertexes[hindex]->particles_out_const_end());
        }
      } else
        edm::LogPrint("TrackHistoryAnalyzer") << "  genVertex no found";
    } else
      edm::LogPrint("TrackHistoryAnalyzer") << "  fake track";

    edm::LogPrint("TrackHistoryAnalyzer") << "  track categories : " << classifier_;
  }
}

void TrackHistoryAnalyzer::beginRun(const edm::Run &run, const edm::EventSetup &setup) {
  // Get the particles table.
  pdt_ = setup.getHandle(pdtToken_);
}

void TrackHistoryAnalyzer::beginJob() { totalTracks_ = 0; }

std::string TrackHistoryAnalyzer::particleString(int pdgId) const {
  ParticleData const *pid;

  std::ostringstream vDescription;

  HepPDT::ParticleID particleType(pdgId);

  if (particleType.isValid()) {
    pid = pdt_->particle(particleType);
    if (pid)
      vDescription << pid->name();
    else
      vDescription << pdgId;
  } else
    vDescription << pdgId;

  return vDescription.str();
}

std::string TrackHistoryAnalyzer::vertexString(const TrackingParticleRefVector &in,
                                               const TrackingParticleRefVector &out) const {
  ParticleData const *pid;

  std::ostringstream vDescription;

  for (std::size_t j = 0; j < in.size(); j++) {
    if (!j)
      vDescription << "(";

    HepPDT::ParticleID particleType(in[j]->pdgId());

    if (particleType.isValid()) {
      pid = pdt_->particle(particleType);
      if (pid)
        vDescription << pid->name();
      else
        vDescription << in[j]->pdgId();
    } else
      vDescription << in[j]->pdgId();

    if (j == in.size() - 1)
      vDescription << ")";
    else
      vDescription << ",";
  }

  vDescription << "->";

  for (std::size_t j = 0; j < out.size(); j++) {
    if (!j)
      vDescription << "(";

    HepPDT::ParticleID particleType(out[j]->pdgId());

    if (particleType.isValid()) {
      pid = pdt_->particle(particleType);
      if (pid)
        vDescription << pid->name();
      else
        vDescription << out[j]->pdgId();
    } else
      vDescription << out[j]->pdgId();

    if (j == out.size() - 1)
      vDescription << ")";
    else
      vDescription << ",";
  }

  return vDescription.str();
}

std::string TrackHistoryAnalyzer::vertexString(HepMC::GenVertex::particles_in_const_iterator in_begin,
                                               HepMC::GenVertex::particles_in_const_iterator in_end,
                                               HepMC::GenVertex::particles_out_const_iterator out_begin,
                                               HepMC::GenVertex::particles_out_const_iterator out_end) const {
  ParticleData const *pid;

  std::ostringstream vDescription;

  std::size_t j = 0;

  HepMC::GenVertex::particles_in_const_iterator in, itmp;

  for (in = in_begin; in != in_end; in++, j++) {
    if (!j)
      vDescription << "(";

    HepPDT::ParticleID particleType((*in)->pdg_id());

    if (particleType.isValid()) {
      pid = pdt_->particle(particleType);
      if (pid)
        vDescription << pid->name();
      else
        vDescription << (*in)->pdg_id();
    } else
      vDescription << (*in)->pdg_id();

    itmp = in;

    if (++itmp == in_end)
      vDescription << ")";
    else
      vDescription << ",";
  }

  vDescription << "->";
  j = 0;

  HepMC::GenVertex::particles_out_const_iterator out, otmp;

  for (out = out_begin; out != out_end; out++, j++) {
    if (!j)
      vDescription << "(";

    HepPDT::ParticleID particleType((*out)->pdg_id());

    if (particleType.isValid()) {
      pid = pdt_->particle(particleType);
      if (pid)
        vDescription << pid->name();
      else
        vDescription << (*out)->pdg_id();
    } else
      vDescription << (*out)->pdg_id();

    otmp = out;

    if (++otmp == out_end)
      vDescription << ")";
    else
      vDescription << ",";
  }

  return vDescription.str();
}

DEFINE_FWK_MODULE(TrackHistoryAnalyzer);
