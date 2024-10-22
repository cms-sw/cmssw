#include "SimDataFormats/CaloAnalysis/interface/MtdCaloParticle.h"

#include "DataFormats/HepMCCandidate/interface/GenParticle.h"

#include <FWCore/MessageLogger/interface/MessageLogger.h>

#include "SimDataFormats/CaloAnalysis/interface/SimCluster.h"

MtdCaloParticle::MtdCaloParticle() {
  // No operation
}

MtdCaloParticle::MtdCaloParticle(const SimTrack &simtrk) {
  addG4Track(simtrk);
  event_ = simtrk.eventId();
  particleId_ = simtrk.trackId();
  theMomentum_.SetPxPyPzE(
      simtrk.momentum().px(), simtrk.momentum().py(), simtrk.momentum().pz(), simtrk.momentum().E());
}

MtdCaloParticle::MtdCaloParticle(EncodedEventId eventID, uint32_t particleID) {
  event_ = eventID;
  particleId_ = particleID;
}

MtdCaloParticle::~MtdCaloParticle() {}

std::ostream &operator<<(std::ostream &s, MtdCaloParticle const &tp) {
  s << "Calo Particle:" << std::endl;
  s << "CP momentum, q, ID, & Event #: " << tp.p4() << " " << tp.charge() << " " << tp.pdgId() << " "
    << tp.eventId().bunchCrossing() << "." << tp.eventId().event() << std::endl;

  for (MtdCaloParticle::genp_iterator hepT = tp.genParticle_begin(); hepT != tp.genParticle_end(); ++hepT) {
    s << " HepMC Track Momentum " << (*hepT)->momentum().rho() << std::endl;
  }

  for (MtdCaloParticle::g4t_iterator g4T = tp.g4Track_begin(); g4T != tp.g4Track_end(); ++g4T) {
    s << " Geant Track Momentum  " << g4T->momentum() << std::endl;
    s << " Geant Track ID & type " << g4T->trackId() << " " << g4T->type() << std::endl;
    if (g4T->type() != tp.pdgId()) {
      s << " Mismatch b/t MtdCaloParticle and Geant types" << std::endl;
    }
  }
  s << "SimClusters in this MtdCaloParticle: " << std::endl;
  for (auto itr = tp.simClusters_.begin(); itr != tp.simClusters_.end(); ++itr) {
    s << **itr;
  }
  s << std::endl;
  return s;
}
