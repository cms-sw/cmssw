#ifndef JetPartonMatching_h
#define JetPartonMatching_h

#include "DataFormats/JetReco/interface/GenJet.h"
#include "DataFormats/JetReco/interface/CaloJet.h"
#include "DataFormats/PatCandidates/interface/Jet.h"
#include "DataFormats/Candidate/interface/Candidate.h"

#include <vector>

class JetPartonMatching {
  // common class for jet parton matching in ttbar
  // decays. Several matching algorithms are provided
 public:

  typedef std::vector< std::pair<unsigned int, int> > MatchingCollection;
  enum algorithms { totalMinDist, minSumDist, ptOrderedMinDist, unambiguousOnly };

  JetPartonMatching(){};
  JetPartonMatching(const std::vector<const reco::Candidate*>&, const std::vector<reco::GenJet>&,
		    int, bool, bool, double);
  JetPartonMatching(const std::vector<const reco::Candidate*>&, const std::vector<reco::CaloJet>&,
		    int, bool, bool, double);
  JetPartonMatching(const std::vector<const reco::Candidate*>&, const std::vector<pat::Jet>&,
		    int, bool, bool, double);
  JetPartonMatching(const std::vector<const reco::Candidate*>&, const std::vector<const reco::Candidate*>&,
		    int, bool, bool, double);
  ~JetPartonMatching(){};	
  
  //matching meta information
  unsigned int getNumberOfAvailableCombinations() { return matching.size(); }
  int getNumberOfUnmatchedPartons(const unsigned int comb=0){ return (comb<numberOfUnmatchedPartons.size() ? (int)numberOfUnmatchedPartons[comb] : -1 ); }
  int getMatchForParton(unsigned int part, unsigned int comb=0);
  std::vector<int> getMatchesForPartons(unsigned int comb=0);
  double getDistanceForParton(unsigned int part, unsigned int comb=0);
  double getSumDistances(unsigned int comb=0);

  //matching quantities
  double getSumDeltaE (const unsigned int comb=0) { return (comb<sumDeltaE .size() ? sumDeltaE [comb] : -999.); }
  double getSumDeltaPt(const unsigned int comb=0) { return (comb<sumDeltaPt.size() ? sumDeltaPt[comb] : -999.); }
  double getSumDeltaR (const unsigned int comb=0) { return (comb<sumDeltaR .size() ? sumDeltaR [comb] : -999.); }

  void print();
  
 private:
  
  void calculate();
  double distance(const math::XYZTLorentzVector&, const math::XYZTLorentzVector&);
  void matchingTotalMinDist();
  void minSumDist_recursion(unsigned int, std::vector<unsigned int>&, std::vector<bool>&, std::vector<std::pair<double, MatchingCollection> >&);
  void matchingMinSumDist();
  void matchingPtOrderedMinDist();
  void matchingUnambiguousOnly();
  
 private:
 
  std::vector<const reco::Candidate*> partons;
  std::vector<const reco::Candidate*> jets;
  std::vector<MatchingCollection> matching;
  
  std::vector<unsigned int> numberOfUnmatchedPartons;
  std::vector<double> sumDeltaE;
  std::vector<double> sumDeltaPt;
  std::vector<double> sumDeltaR;
  
  int algorithm_;
  bool useMaxDist_;
  bool useDeltaR_;
  double maxDist_;
};

#endif
