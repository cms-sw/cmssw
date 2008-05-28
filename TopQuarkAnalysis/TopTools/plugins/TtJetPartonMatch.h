#ifndef TtJetPartonMatch_h
#define TtJetPartonMatch_h

#include <memory>
#include <vector>

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/PatCandidates/interface/Jet.h"
#include "AnalysisDataFormats/TopObjects/interface/TtGenEvent.h"
#include "TopQuarkAnalysis/TopTools/interface/JetPartonMatching.h"

template <typename C>
class TtJetPartonMatch : public edm::EDProducer {
  
 public:
  
  explicit TtJetPartonMatch(const edm::ParameterSet&);
  ~TtJetPartonMatch();
  
 private:
  
  typedef std::vector<pat::Jet> TopJetCollection;
  
 private:

  virtual void produce(edm::Event&, const edm::EventSetup&);

 private:

  edm::InputTag jets_;

  unsigned int nJets_;
  int algorithm_;
  bool useDeltaR_;
  bool useMaxDist_;
  double maxDist_;
};

template<typename C>
TtJetPartonMatch<C>::TtJetPartonMatch(const edm::ParameterSet& cfg):
  jets_(cfg.getParameter<edm::InputTag>("jets")),
  nJets_(cfg.getParameter<unsigned int>("nJets")),
  algorithm_(cfg.getParameter<int>("algorithm")),
  useDeltaR_(cfg.getParameter<bool>("useDeltaR")),
  useMaxDist_(cfg.getParameter<bool>("useMaxDist")),
  maxDist_(cfg.getParameter<double>("maxDist"))
{
  // produces a vector of jet indices in the order
  // of TtSemiEvtPartons or TtHadEvtPartons
  produces< std::vector<int> >();
  produces< double >("SumPt");
  produces< double >("SumDR");
}

template<typename C>
TtJetPartonMatch<C>::~TtJetPartonMatch()
{
}

template<typename C>
void
TtJetPartonMatch<C>::produce(edm::Event& evt, const edm::EventSetup& setup)
{
  edm::Handle<TtGenEvent> genEvt;
  evt.getByLabel("genEvt", genEvt);
  
  edm::Handle<TopJetCollection> topJets;
  evt.getByLabel(jets_, topJets);

  // fill vector of partons in the order of
  // of TtSemiEvtPartons or TtHadEvtPartons
  C parts;
  std::vector<const reco::Candidate*> partons = parts.vec(*genEvt);

  // prepare vector of jets
  std::vector<pat::JetType> jets;
  for(unsigned int ij=0; ij<topJets->size(); ++ij) {
    if(nJets_>=partons.size()){ if(ij==nJets_) break; }
    else{ if(ij==partons.size()) break; }
    pat::JetType jet = (*topJets)[ij].recJet();
    jets.push_back( jet );
  }

  // do the matching with specified parameters
  JetPartonMatching jetPartonMatch(partons, jets, algorithm_, useMaxDist_, useDeltaR_, maxDist_);

  // feed out parton match
  std::auto_ptr< std::vector<int> > pOut(new std::vector<int>);
  for(unsigned int i=0; i<partons.size(); ++i){
    pOut->push_back( jetPartonMatch.getMatchForParton(i) );
  }
  evt.put(pOut);

  // feed out sum of delta pt
  std::auto_ptr< double > sumPt( new double);
  *sumPt=jetPartonMatch.getSumDeltaPt();
  evt.put(sumPt, "SumPt");

  // feed out sum of delta r
  std::auto_ptr< double > sumDR( new double);
  *sumDR=jetPartonMatch.getSumDeltaR();
  evt.put(sumDR, "SumDR");
}

#endif
