#ifndef SimDataFormats_SimTauCPLink_h
#define SimDataFormats_SimTauCPLink_h

#include "DataFormats/HGCalReco/interface/TICLCandidate.h"
#include "SimDataFormats/CaloAnalysis/interface/CaloParticleFwd.h"
#include "SimDataFormats/CaloAnalysis/interface/CaloParticle.h"
#include "SimDataFormats/CaloAnalysis/interface/SimCluster.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

class SimTauCPLink {
public:
  SimTauCPLink() {}
  ~SimTauCPLink() {}
  struct DecayNav {
    int pdgId_;
    int resonance_idx_;
    int calo_particle_idx_;
    int gen_particle_idx_;
    int pdgId() const { return pdgId_; }
    int resonance_idx() const { return resonance_idx_; }
    int calo_particle_idx() const { return calo_particle_idx_; }
    int gen_particle_idx() const { return gen_particle_idx_; }

  private:
  };

  std::vector<std::pair<int, int>> resonances;
  std::vector<DecayNav> leaves;
  CaloParticleRefVector calo_particle_leaves;
  int decayMode;

  enum decayModes {
    kNull = -1,
    kOneProng0PiZero,
    kOneProng1PiZero,
    kOneProng2PiZero,
    kOneProng3PiZero,
    kOneProngNPiZero,
    kTwoProng0PiZero,
    kTwoProng1PiZero,
    kTwoProng2PiZero,
    kTwoProng3PiZero,
    kTwoProngNPiZero,
    kThreeProng0PiZero,
    kThreeProng1PiZero,
    kThreeProng2PiZero,
    kThreeProng3PiZero,
    kThreeProngNPiZero,
    kRareDecayMode,
    kElectron,
    kMuon
  };

  void dump(void) const {
    for (auto const &l : leaves) {
      LogDebug("SimTauProducer")
          .format(
              "L {} {} CP: {} GenP idx: {}", l.pdgId(), l.resonance_idx(), l.calo_particle_idx(), l.gen_particle_idx());
    }
    for (auto const &r : resonances) {
      LogDebug("SimTauProducer").format("R {} {}", r.first, r.second);
    }
  }

  void dumpDecay(const std::pair<int, int> &entry) const {
    if (entry.second == -1) {  // No intermediate mother.
      LogDebug("SimTauProducer").format("{} {}", entry.first, entry.second);
    } else {
      LogDebug("SimTauProducer").format("{} {} coming from: ", entry.first, entry.second);
      auto const &mother = resonances[entry.second];
      dumpDecay(mother);
    }
  }

  void dumpFullDecay(void) const {
    for (auto const &leaf : leaves) {
      dumpDecay({leaf.pdgId(), leaf.resonance_idx()});
    }
  }

  int buildDecayModes() {
    int numElectrons = 0;
    int numMuons = 0;
    int numHadrons = 0;
    int numPhotons = 0;
    for (auto leaf : leaves) {
      int pdg_id = abs(leaf.pdgId());
      switch (pdg_id) {
        case 22:
          numPhotons++;
          break;
        case 11:
          numElectrons++;
          break;
        case 13:
          numMuons++;
          break;
        case 16:
          break;
        default:
          numHadrons++;
      }
    }

    if (numElectrons == 1)
      return kElectron;
    else if (numMuons == 1)
      return kMuon;
    switch (numHadrons) {
      case 1:
        switch (numPhotons) {
          case 0:
            return kOneProng0PiZero;
          case 2:
            return kOneProng1PiZero;
          case 4:
            return kOneProng2PiZero;
          case 6:
            return kOneProng3PiZero;
          default:
            return kOneProngNPiZero;
        }
      case 2:
        switch (numPhotons) {
          case 0:
            return kTwoProng0PiZero;
          case 2:
            return kTwoProng1PiZero;
          case 4:
            return kTwoProng2PiZero;
          case 6:
            return kTwoProng3PiZero;
          default:
            return kTwoProngNPiZero;
        }
      case 3:
        switch (numPhotons) {
          case 0:
            return kThreeProng0PiZero;
          case 2:
            return kThreeProng1PiZero;
          case 4:
            return kThreeProng2PiZero;
          case 6:
            return kThreeProng3PiZero;
          default:
            return kThreeProngNPiZero;
        }
      default:
        return kRareDecayMode;
    }
  }

private:
};

#endif  //SimTauCPLink
