#ifndef TtHadEvtPartons_h
#define TtHadEvtPartons_h

#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "AnalysisDataFormats/TopObjects/interface/TtGenEvent.h"

#include <vector>

class TtHadEvtPartons {
  // common wrapper class to fill partons in a well
  // defined order for fully hadronic ttbar decays
 public:

  enum { LightQTop, LightQBarTop, B, LightQTopBar, LightQBarTopBar, BBar};

  TtHadEvtPartons(){};
  ~TtHadEvtPartons(){};
  std::vector<const reco::Candidate*> vec(const TtGenEvent& genEvt)
  {
    std::vector<const reco::Candidate*> vec;
    vec.push_back( (genEvt.isFullHadronic() && genEvt.quarkFromTop()       ) ? genEvt.quarkFromTop()        : new reco::GenParticle(0, reco::Particle::LorentzVector(), reco::Particle::Point(), 0, 0, false) );
    vec.push_back( (genEvt.isFullHadronic() && genEvt.quarkFromTopBar()    ) ? genEvt.quarkFromTopBar()     : new reco::GenParticle(0, reco::Particle::LorentzVector(), reco::Particle::Point(), 0, 0, false) );
    vec.push_back( (genEvt.isFullHadronic() && genEvt.b()                  ) ? genEvt.b()                   : new reco::GenParticle(0, reco::Particle::LorentzVector(), reco::Particle::Point(), 0, 0, false) );
    vec.push_back( (genEvt.isFullHadronic() && genEvt.quarkFromAntiTop()   ) ? genEvt.quarkFromAntiTop()    : new reco::GenParticle(0, reco::Particle::LorentzVector(), reco::Particle::Point(), 0, 0, false) );
    vec.push_back( (genEvt.isFullHadronic() && genEvt.quarkFromAntiTopBar()) ? genEvt.quarkFromAntiTopBar() : new reco::GenParticle(0, reco::Particle::LorentzVector(), reco::Particle::Point(), 0, 0, false) );
    vec.push_back( (genEvt.isFullHadronic() && genEvt.bBar()               ) ? genEvt.bBar()                : new reco::GenParticle(0, reco::Particle::LorentzVector(), reco::Particle::Point(), 0, 0, false) );
    return vec;
  }
};

#endif
