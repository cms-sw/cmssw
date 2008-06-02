#ifndef JetPartonMatching_h
#define JetPartonMatching_h

#include "DataFormats/JetReco/interface/GenJet.h"
#include "DataFormats/JetReco/interface/CaloJet.h"
#include "DataFormats/PatCandidates/interface/Jet.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "AnalysisDataFormats/TopObjects/interface/TtGenEvent.h"

#include <vector>
#include <Math/VectorUtil.h>

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

class TtHadEvtPartons {
  // common wrapper class to fill partons in a well
  // defined order for fully hadronic ttbar decays
 public:

  enum { LightQTop, LightQTopBar, B, LightQAntiTop, LightQBarAntiTop, BBar};

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

class JetPartonMatching {
  // common class for jet parton matching in ttbar
  // decays. Several matching algorithms are provided
 public:

  typedef std::vector< std::pair<unsigned int, int> > MatchingCollection;
  enum algorithms { totalMinDist, minSumDist, ptOrderedMinDist, unambiguousOnly };

  JetPartonMatching(){};
  JetPartonMatching(const std::vector<const reco::Candidate*>&, const std::vector<reco::GenJet>&,
		    const int, const bool, const bool, const double);
  JetPartonMatching(const std::vector<const reco::Candidate*>&, const std::vector<reco::CaloJet>&,
		    const int, const bool, const bool, const double);
  JetPartonMatching(const std::vector<const reco::Candidate*>&, const std::vector<pat::JetType>&,
		    const int, const bool, const bool, const double);
  JetPartonMatching(const std::vector<const reco::Candidate*>&, const std::vector<const reco::Candidate*>&,
		    const int, const bool, const bool, const double);
  ~JetPartonMatching(){};	
  
  //matching meta information
  unsigned int getNumberOfUnmatchedPartons(){ return numberOfUnmatchedPartons; }
  std::vector< std::pair<unsigned int, int> > getMatching() { return matching; }
  int getMatchForParton(const unsigned int ip) { return matching[ip].second; }
  double getAngleForParton(const unsigned int);
  double getSumAngles();

  //matching quantities
  double getSumDeltaE()   { return sumDeltaE;  }
  double getSumDeltaPt()  { return sumDeltaPt; }
  double getSumDeltaR()   { return sumDeltaR;  }
  
 private:
  
  void calculate();
  double distance(const math::XYZTLorentzVector&, const math::XYZTLorentzVector&);
  void matchingTotalMinDist();
  void minSumDist_recursion(const unsigned int, std::vector<unsigned int>&, std::vector<bool>&, std::vector<int>&, double&);
  void matchingMinSumDist();
  void matchingPtOrderedMinDist();
  void matchingUnambiguousOnly();
  
 private:
 
  std::vector<const reco::Candidate*> partons;
  std::vector<const reco::Candidate*> jets;
  MatchingCollection matching;
  
  unsigned int numberOfUnmatchedPartons;
  double sumDeltaE;
  double sumDeltaPt;
  double sumDeltaR;
  
  int algorithm_;
  bool useMaxDist_;
  bool useDeltaR_;
  double maxDist_;
};

#endif
