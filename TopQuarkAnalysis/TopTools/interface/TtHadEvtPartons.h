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
    vec.push_back( genEvt.quarkFromTop()        ? genEvt.quarkFromTop()        : new reco::GenParticle() );
    vec.push_back( genEvt.quarkFromTopBar()     ? genEvt.quarkFromTopBar()     : new reco::GenParticle() );
    vec.push_back( genEvt.b()                   ? genEvt.b()                   : new reco::GenParticle() );
    vec.push_back( genEvt.quarkFromAntiTop()    ? genEvt.quarkFromAntiTop()    : new reco::GenParticle() );
    vec.push_back( genEvt.quarkFromAntiTopBar() ? genEvt.quarkFromAntiTopBar() : new reco::GenParticle() );
    vec.push_back( genEvt.bBar()                ? genEvt.bBar()                : new reco::GenParticle() );
    return vec;
  }
};

#endif
