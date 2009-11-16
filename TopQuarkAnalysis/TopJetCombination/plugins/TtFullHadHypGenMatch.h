#ifndef TtFullHadHypGenMatch_h
#define TtFullHadHypGenMatch_h

#include "TopQuarkAnalysis/TopJetCombination/interface/TtFullHadHypothesis.h"
#include "DataFormats/Candidate/interface/LeafCandidate.h"

class TtFullHadHypGenMatch : public TtFullHadHypothesis  {

 public:

  explicit TtFullHadHypGenMatch(const edm::ParameterSet& cfg);
  ~TtFullHadHypGenMatch();

 private:

  /// build the event hypothesis key
  virtual void buildKey() { key_= TtFullHadronicEvent::kGenMatch; };  
  /// build event hypothesis from the reco objects of a semi-leptonic event 
  virtual void buildHypo(edm::Event& evt,
			 const edm::Handle<std::vector<pat::Jet> >& jets, 
			 std::vector<int>& match,
			 const unsigned int iComb);
};

#endif
