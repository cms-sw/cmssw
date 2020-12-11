#include "SimDataFormats/PFAnalysis/interface/PFParticle.h"

#include "DataFormats/HepMCCandidate/interface/GenParticle.h"

#include <FWCore/MessageLogger/interface/MessageLogger.h>

PFParticle::PFParticle() {
  // No operation
}

PFParticle::~PFParticle() {}

PFParticle::PFParticle(const TrackingParticleRefVector& trackingParticles, const SimClusterRefVector& simClusters) {
    setTrackingParticles(trackingParticles);
    setSimClusters(simClusters);
}

void PFParticle::setTrackingParticles(const TrackingParticleRefVector& refs) { trackingParticles_ = refs; }

void PFParticle::setSimClusters(const SimClusterRefVector& refs) { simClusters_ = refs; }

void PFParticle::addSimCluster(const SimClusterRef& sc) { simClusters_.push_back(sc); }

void PFParticle::addTrackingParticle(const TrackingParticleRef& tp) { trackingParticles_.push_back(tp); }

void PFParticle::addGenParticle(const reco::GenParticleRef& gp) { genParticles_.push_back(gp); }

void PFParticle::addG4Track(const SimTrack& t) { g4Tracks_.push_back(t); }

PFParticle::genp_iterator PFParticle::genParticle_begin() const { return genParticles_.begin(); }

PFParticle::genp_iterator PFParticle::genParticle_end() const { return genParticles_.end(); }

PFParticle::g4t_iterator PFParticle::g4Track_begin() const { return g4Tracks_.begin(); }

PFParticle::g4t_iterator PFParticle::g4Track_end() const { return g4Tracks_.end(); }
