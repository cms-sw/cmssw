#include "SimDataFormats/CaloAnalysis/interface/MtdSimLayerCluster.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <numeric>

MtdSimLayerCluster::MtdSimLayerCluster() {
  // No operation
}

MtdSimLayerCluster::MtdSimLayerCluster(const SimTrack &simtrk) {
  addG4Track(simtrk);
  event_ = simtrk.eventId();
  particleId_ = simtrk.trackId();

  theMomentum_.SetPxPyPzE(
      simtrk.momentum().px(), simtrk.momentum().py(), simtrk.momentum().pz(), simtrk.momentum().E());
}

MtdSimLayerCluster::MtdSimLayerCluster(EncodedEventId eventID, uint32_t particleID) {
  event_ = eventID;
  particleId_ = particleID;
}

MtdSimLayerCluster::~MtdSimLayerCluster() {}

std::ostream &operator<<(std::ostream &s, MtdSimLayerCluster const &tp) {
  s << "CP momentum, q, ID, & Event #: " << tp.p4() << " " << tp.charge() << " " << tp.pdgId() << " "
    << tp.eventId().bunchCrossing() << "." << tp.eventId().event() << std::endl;

  for (MtdSimLayerCluster::genp_iterator hepT = tp.genParticle_begin(); hepT != tp.genParticle_end(); ++hepT) {
    s << " HepMC Track Momentum " << (*hepT)->momentum().rho() << std::endl;
  }

  for (MtdSimLayerCluster::g4t_iterator g4T = tp.g4Track_begin(); g4T != tp.g4Track_end(); ++g4T) {
    s << " Geant Track Momentum  " << g4T->momentum() << std::endl;
    s << " Geant Track ID & type " << g4T->trackId() << " " << g4T->type() << std::endl;
    if (g4T->type() != tp.pdgId()) {
      s << " Mismatch b/t MtdSimLayerCluster and Geant types" << std::endl;
    }
  }
  s << " # of cells = " << tp.hits_.size()
    << ", effective cells = " << std::accumulate(tp.fractions_.begin(), tp.fractions_.end(), 0.f) << std::endl;
  return s;
}
