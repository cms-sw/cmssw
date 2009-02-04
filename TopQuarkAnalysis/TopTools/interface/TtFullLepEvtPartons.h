#ifndef TtFullLepEvtPartons_h
#define TtFullLepEvtPartons_h

#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "AnalysisDataFormats/TopObjects/interface/TtGenEvent.h"

#include <vector>

class TtFullLepEvtPartons {
  // common wrapper class to fill partons in a well
  // defined order for fully leptonic ttbar decays
 public:

  enum { B, BBar };

  TtFullLepEvtPartons(){};
  ~TtFullLepEvtPartons(){};
  std::vector<const reco::Candidate*> vec(const TtGenEvent& genEvt)
  {
    std::vector<const reco::Candidate*> vec;
    vec.push_back( (genEvt.isFullLeptonic() && genEvt.b()   ) ? genEvt.b()    : new reco::GenParticle(0, reco::Particle::LorentzVector(), reco::Particle::Point(), 0, 0, false) );
    vec.push_back( (genEvt.isFullLeptonic() && genEvt.bBar()) ? genEvt.bBar() : new reco::GenParticle(0, reco::Particle::LorentzVector(), reco::Particle::Point(), 0, 0, false) );
    return vec;
  }
};

#endif
