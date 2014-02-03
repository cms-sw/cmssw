#ifndef TtFullLepHypGenMatch_h
#define TtFullLepHypGenMatch_h

#include "TopQuarkAnalysis/TopJetCombination/interface/TtFullLepHypothesis.h"
#include "DataFormats/Candidate/interface/LeafCandidate.h"

class TtFullLepHypGenMatch : public TtFullLepHypothesis  {

 public:

  explicit TtFullLepHypGenMatch(const edm::ParameterSet&);
  ~TtFullLepHypGenMatch();

 private:

  /// build the event hypothesis key
  virtual void buildKey() { key_= TtFullLeptonicEvent::kGenMatch; };  
  /// build event hypothesis from the reco objects of a semi-leptonic event 
  virtual void buildHypo(edm::Event& evt,
			 const edm::Handle<std::vector<pat::Electron > >& elecs, 
			 const edm::Handle<std::vector<pat::Muon> >& mus, 
			 const edm::Handle<std::vector<pat::Jet> >& jets, 
			 const edm::Handle<std::vector<pat::MET> >& mets, 
			 std::vector<int>& match,
			 const unsigned int iComb);
			 
  template <typename O>			 
  int findMatchingLepton(const reco::GenParticle*, 
			 const edm::Handle<std::vector<O> >&);
  void buildMatchingNeutrinos(edm::Event&,
                              const edm::Handle<std::vector<pat::MET> >&);			 	 
};

#endif
