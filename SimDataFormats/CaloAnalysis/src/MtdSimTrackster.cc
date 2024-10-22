#include "SimDataFormats/CaloAnalysis/interface/MtdSimTrackster.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

MtdSimTrackster::MtdSimTrackster() {
  // No operation
}

MtdSimTrackster::MtdSimTrackster(const SimCluster &sc) {
  auto simtrk = sc.g4Tracks()[0];
  addG4Track(simtrk);
  event_ = simtrk.eventId();
  particleId_ = simtrk.trackId();

  theMomentum_.SetPxPyPzE(
      simtrk.momentum().px(), simtrk.momentum().py(), simtrk.momentum().pz(), simtrk.momentum().E());
}

MtdSimTrackster::MtdSimTrackster(EncodedEventId eventID, uint32_t particleID) {
  event_ = eventID;
  particleId_ = particleID;
}

MtdSimTrackster::MtdSimTrackster(const SimCluster &sc,
                                 const std::vector<uint32_t> SCs,
                                 const float time,
                                 const GlobalPoint pos) {
  auto simtrk = sc.g4Tracks()[0];
  addG4Track(simtrk);
  event_ = simtrk.eventId();
  particleId_ = simtrk.trackId();

  theMomentum_.SetPxPyPzE(
      simtrk.momentum().px(), simtrk.momentum().py(), simtrk.momentum().pz(), simtrk.momentum().E());

  clusters_ = SCs;
  timeAtEntrance_ = time;
  posAtEntrance_ = pos;
}

MtdSimTrackster::~MtdSimTrackster() {}

std::ostream &operator<<(std::ostream &s, MtdSimTrackster const &tp) {
  s << "CP momentum, q, ID, & Event #: " << tp.p4() << " " << tp.charge() << " " << tp.pdgId() << " "
    << tp.eventId().bunchCrossing() << "." << tp.eventId().event() << std::endl;

  for (MtdSimTrackster::genp_iterator hepT = tp.genParticle_begin(); hepT != tp.genParticle_end(); ++hepT) {
    s << " HepMC Track Momentum " << (*hepT)->momentum().rho() << std::endl;
  }

  for (MtdSimTrackster::g4t_iterator g4T = tp.g4Track_begin(); g4T != tp.g4Track_end(); ++g4T) {
    s << " Geant Track Momentum  " << g4T->momentum() << std::endl;
    s << " Geant Track ID & type " << g4T->trackId() << " " << g4T->type() << std::endl;
    if (g4T->type() != tp.pdgId()) {
      s << " Mismatch b/t MtdSimTrackster and Geant types" << std::endl;
    }
  }
  s << " # of clusters = " << tp.clusters_.size() << std::endl;
  return s;
}
