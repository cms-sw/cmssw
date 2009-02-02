#ifndef TtSemiLepEvtPartons_h
#define TtSemiLepEvtPartons_h

#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "AnalysisDataFormats/TopObjects/interface/TtGenEvent.h"

#include <vector>

class TtSemiLepEvtPartons {
  // common wrapper class to fill partons in a well
  // defined order for semileptonic ttbar decays
 public:

  enum { LightQ, LightQBar, HadB, LepB, Lepton };

  TtSemiLepEvtPartons(){};
  ~TtSemiLepEvtPartons(){};
  std::vector<const reco::Candidate*> vec(const TtGenEvent& genEvt)
  {
    std::vector<const reco::Candidate*> vec;
    vec.push_back( (genEvt.isSemiLeptonic() && genEvt.hadronicDecayQuark()   ) ? genEvt.hadronicDecayQuark()    : new reco::GenParticle(0, reco::Particle::LorentzVector(), reco::Particle::Point(), 0, 0, false) );
    vec.push_back( (genEvt.isSemiLeptonic() && genEvt.hadronicDecayQuarkBar()) ? genEvt.hadronicDecayQuarkBar() : new reco::GenParticle(0, reco::Particle::LorentzVector(), reco::Particle::Point(), 0, 0, false));
    vec.push_back( (genEvt.isSemiLeptonic() && genEvt.hadronicDecayB()       ) ? genEvt.hadronicDecayB()        : new reco::GenParticle(0, reco::Particle::LorentzVector(), reco::Particle::Point(), 0, 0, false) );
    vec.push_back( (genEvt.isSemiLeptonic() && genEvt.leptonicDecayB()       ) ? genEvt.leptonicDecayB()        : new reco::GenParticle(0, reco::Particle::LorentzVector(), reco::Particle::Point(), 0, 0, false) );
    return vec;
  }
};

#endif
