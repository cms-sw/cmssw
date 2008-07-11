#ifndef TtSemiEvtPartons_h
#define TtSemiEvtPartons_h

#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "AnalysisDataFormats/TopObjects/interface/TtGenEvent.h"

#include <vector>

class TtSemiEvtPartons {
  // common wrapper class to fill partons in a well
  // defined order for semileptonic ttbar decays
 public:

  enum { LightQ, LightQBar, HadB, LepB};

  TtSemiEvtPartons(){};
  ~TtSemiEvtPartons(){};
  std::vector<const reco::Candidate*> vec(const TtGenEvent& genEvt)
  {
    std::vector<const reco::Candidate*> vec;
    vec.push_back( genEvt.hadronicDecayQuark()    ? genEvt.hadronicDecayQuark()    : new reco::GenParticle() );
    vec.push_back( genEvt.hadronicDecayQuarkBar() ? genEvt.hadronicDecayQuarkBar() : new reco::GenParticle() );
    vec.push_back( genEvt.hadronicDecayB()        ? genEvt.hadronicDecayB()        : new reco::GenParticle() );
    vec.push_back( genEvt.leptonicDecayB()        ? genEvt.leptonicDecayB()        : new reco::GenParticle() );
    return vec;
  }
};

#endif
