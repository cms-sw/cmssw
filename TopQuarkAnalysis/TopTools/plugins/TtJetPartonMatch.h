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

  int maxNJets_;
  int maxNComb_;
  int algorithm_;
  bool useDeltaR_;
  bool useMaxDist_;
  double maxDist_;
  int verbosity_;
};

template<typename C>
TtJetPartonMatch<C>::TtJetPartonMatch(const edm::ParameterSet& cfg):
  jets_      (cfg.getParameter<edm::InputTag>("jets"      )),
  maxNJets_  (cfg.getParameter<int>          ("maxNJets"  )),
  maxNComb_  (cfg.getParameter<int>          ("maxNComb"  )),
  algorithm_ (cfg.getParameter<int>          ("algorithm" )),
  useDeltaR_ (cfg.getParameter<bool>         ("useDeltaR" )),
  useMaxDist_(cfg.getParameter<bool>         ("useMaxDist")),
  maxDist_   (cfg.getParameter<double>       ("maxDist"   )),
  verbosity_ (cfg.getParameter<int>          ("verbosity" ))
{
  // produces a vector of jet indices in the order
  // of TtSemiLepEvtPartons, TtFullHadEvtPartons or
  // TtFullLepEvtPartons
  produces< std::vector<std::vector<int> > >();
  produces< std::vector<double> >("SumPt");
  produces< std::vector<double> >("SumDR");
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
  // TtSemiLepEvtPartons or TtFullHadEvtPartons
  C parts;
  std::vector<const reco::Candidate*> partons = parts.vec(*genEvt);

  // prepare vector of jets
  std::vector<pat::Jet> jets;
  for(unsigned int ij=0; ij<topJets->size(); ++ij) {
    // take all jets if maxNJets_ == -1; otherwise
    // use maxNJets_ if maxNJets_ is big enough or
    // use same number of jets as partons if
    // maxNJets_ < number of partons
    if(maxNJets_!=-1) {
      if(maxNJets_>=(int)partons.size()) {
	if((int)ij==maxNJets_) break;
      }
      else {
	if(ij==partons.size()) break;
      }
    }
    jets.push_back( (*topJets)[ij] );
  }

  // do the matching with specified parameters
  JetPartonMatching jetPartonMatch(partons, jets, algorithm_, useMaxDist_, useDeltaR_, maxDist_);

  // print some info for each event
  // if corresponding verbosity level set
  if(verbosity_>0) jetPartonMatch.print();

  // feed out parton match, sumPt and sumDR
  std::auto_ptr< std::vector<std::vector<int> > > match(new std::vector<std::vector<int> >);
  std::auto_ptr< std::vector<double> > sumPt(new std::vector<double>);
  std::auto_ptr< std::vector<double> > sumDR(new std::vector<double>);
  for(unsigned int ic=0; ic<jetPartonMatch.getNumberOfAvailableCombinations(); ++ic) {
    if((int)ic>=maxNComb_ && maxNComb_>=0) break;
    match->push_back( jetPartonMatch.getMatchesForPartons(ic) );
    sumPt->push_back( jetPartonMatch.getSumDeltaPt       (ic) );
    sumDR->push_back( jetPartonMatch.getSumDeltaR        (ic) );
  }
  evt.put(match);
  evt.put(sumPt, "SumPt");
  evt.put(sumDR, "SumDR");
}

#endif
