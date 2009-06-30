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

// ----------------------------------------------------------------------
// template class for: 
//
//  * TtFullHadJetPartonMatch
//  * TtSemiLepJetPartonMatch
//  * TtFullLepJetPartonMatch
//
//  the class provides plugins for jet parton matching corresponding 
//  to the JetPartonMatching class; expected input is one of the 
//  classes in:
//
//  AnalysisDataFormats/TopObjects/interface/TtEventPartons.h
//
//  output is:
//  * a vector of vectors in the orders defined in TtEventPartons 
//    containing the index of the coresponding jet/lepton in the 
//    input collection of jets/leptons
//  * and a set of vectors with quality parameters of the matching 
//
//  the matching can be performed on an arbitary number of jet 
//  combinations; per default the combination which matches best 
//  according to the quality parameters will be stored; the length
//  of the vectors will be 1 then 
// ----------------------------------------------------------------------

template <typename C>
class TtJetPartonMatch : public edm::EDProducer {

 private:
  
  /// typedef for simplified reading
  typedef std::vector<pat::Jet> TopJetCollection;
  
 public:

  /// default conructor  
  explicit TtJetPartonMatch(const edm::ParameterSet&);
  /// default destructor
  ~TtJetPartonMatch();
  /// write jet parton match objects into the event
  virtual void produce(edm::Event&, const edm::EventSetup&);

 private:

  /// event counter for internal use with the verbosity 
  /// level
  int event_;
  /// jet collection input
  edm::InputTag jets_;
  /// maximal number of jets to be considered for the 
  /// matching
  int maxNJets_;
  /// maximal number of combinationws for which the 
  /// matching should be stored
  int maxNComb_;
  /// choice of algorithm (defined in JetPartonMatching)
  int algorithm_;
  /// switch to choose between deltaR/deltaTheta matching
  bool useDeltaR_;
  /// switch to choose whether an outlier rejection 
  /// should be applied or not
  bool useMaxDist_;
  /// threshold for outliers in the case that useMaxDist_
  /// =true
  double maxDist_;
  /// verbolity level
  int verbosity_;
};

template<typename C>
TtJetPartonMatch<C>::TtJetPartonMatch(const edm::ParameterSet& cfg): event_(0),
  jets_      (cfg.getParameter<edm::InputTag>("jets"      )),
  maxNJets_  (cfg.getParameter<int>          ("maxNJets"  )),
  maxNComb_  (cfg.getParameter<int>          ("maxNComb"  )),
  algorithm_ (cfg.getParameter<int>          ("algorithm" )),
  useDeltaR_ (cfg.getParameter<bool>         ("useDeltaR" )),
  useMaxDist_(cfg.getParameter<bool>         ("useMaxDist")),
  maxDist_   (cfg.getParameter<double>       ("maxDist"   )),
  verbosity_ (cfg.getParameter<int>          ("verbosity" ))
{
  // produces a vector of jet/lepton indices in the order of
  //  * TtSemiLepEventPartons
  //  * TtFullHadEventPartons
  //  * TtFullLepEventPartons
  // and vectors of the corresponding quality parameters
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
  //  * TtFullLepEventPartons
  //  * TtSemiLepEventPartons 
  //  * TtFullHadEventPartons
  C parts;
  std::vector<const reco::Candidate*> partons = parts.vec(*genEvt);

  // prepare vector of jets
  std::vector<pat::Jet> jets;
  for(unsigned int ij=0; ij<topJets->size(); ++ij) {
    // take all jets if maxNJets_ == -1; otherwise use
    // maxNJets_ if maxNJets_ is big enough or use same
    // number of jets as partons if maxNJets_ < number 
    // of partons
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
  if(verbosity_>0 && event_<verbosity_){
    ++event_;
    jetPartonMatch.print();
  }

  // write 
  // * parton match 
  // * sumPt 
  // * sumDR
  // to the event
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
