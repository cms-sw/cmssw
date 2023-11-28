#include "SimDataFormats/CaloAnalysis/interface/MtdSimCluster.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <numeric>

MtdSimCluster::MtdSimCluster() {
  // No operation
}

MtdSimCluster::MtdSimCluster(const SimTrack &simtrk) {
  addG4Track(simtrk);
  event_ = simtrk.eventId();
  particleId_ = simtrk.trackId();

  theMomentum_.SetPxPyPzE(
      simtrk.momentum().px(), simtrk.momentum().py(), simtrk.momentum().pz(), simtrk.momentum().E());
}

MtdSimCluster::MtdSimCluster(EncodedEventId eventID, uint32_t particleID) {
  event_ = eventID;
  particleId_ = particleID;
}

MtdSimCluster::~MtdSimCluster() {}

std::ostream &operator<<(std::ostream &s, MtdSimCluster const &tp) {
  s << "CP momentum, q, ID, & Event #: " << tp.p4() << " " << tp.charge() << " " << tp.pdgId() << " "
    << tp.eventId().bunchCrossing() << "." << tp.eventId().event() << std::endl;

  for (MtdSimCluster::genp_iterator hepT = tp.genParticle_begin(); hepT != tp.genParticle_end(); ++hepT) {
    s << " HepMC Track Momentum " << (*hepT)->momentum().rho() << std::endl;
  }

  for (MtdSimCluster::g4t_iterator g4T = tp.g4Track_begin(); g4T != tp.g4Track_end(); ++g4T) {
    s << " Geant Track Momentum  " << g4T->momentum() << std::endl;
    s << " Geant Track ID & type " << g4T->trackId() << " " << g4T->type() << std::endl;
    if (g4T->type() != tp.pdgId()) {
      s << " Mismatch b/t MtdSimCluster and Geant types" << std::endl;
    }
  }
  s << " # of cells = " << tp.hits_.size()
    << ", effective cells = " << std::accumulate(tp.fractions_.begin(), tp.fractions_.end(), 0.f) << std::endl;
  return s;
}
