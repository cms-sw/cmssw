#ifndef TtJetPartonMatch_h
#define TtJetPartonMatch_h

#include <memory>
#include <vector>

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/JetReco/interface/Jet.h"
#include "AnalysisDataFormats/TopObjects/interface/TtGenEvent.h"
#include "TopQuarkAnalysis/TopTools/interface/JetPartonMatching.h"

// ----------------------------------------------------------------------
// template class for:
//
//  * TtFullHadJetPartonMatch
//  * TtSemiLepJetPartonMatch
//  * TtFullLepJetPartonMatch
//
//  the class provides plugins for jet-parton matching corresponding
//  to the JetPartonMatching class; expected input is one of the
//  classes in:
//
//  AnalysisDataFormats/TopObjects/interface/TtFullHadEvtPartons.h
//  AnalysisDataFormats/TopObjects/interface/TtSemiLepEvtPartons.h
//  AnalysisDataFormats/TopObjects/interface/TtFullLepEvtPartons.h
//
//  output is:
//  * a vector of vectors containing the indices of the jets in the
//    input collection matched to the partons in the order defined in
//    the corresponding Tt*EvtPartons class
//  * a set of vectors with quality parameters of the matching
//
//  the matching can be performed on an arbitrary number of jet
//  combinations; per default the combination which matches best
//  according to the quality parameters will be stored; the length
//  of the vectors will be 1 then
// ----------------------------------------------------------------------

template <typename C>
class TtJetPartonMatch : public edm::EDProducer {

 public:

  /// default conructor
  explicit TtJetPartonMatch(const edm::ParameterSet&);
  /// default destructor
  ~TtJetPartonMatch();
  /// write jet parton match objects into the event
  virtual void produce(edm::Event&, const edm::EventSetup&);

 private:

  /// convert string for algorithm into corresponding enumerator type
  JetPartonMatching::algorithms readAlgorithm(const std::string& str);

  /// partons
  C partons_;
  /// TtGenEvent collection input
  edm::EDGetTokenT<TtGenEvent> genEvt_;
  /// jet collection input
  edm::EDGetTokenT<edm::View<reco::Jet> > jets_;
  /// maximal number of jets to be considered for the
  /// matching
  int maxNJets_;
  /// maximal number of combinations for which the
  /// matching should be stored
  int maxNComb_;
  /// choice of algorithm
  JetPartonMatching::algorithms algorithm_;
  /// switch to choose between deltaR/deltaTheta matching
  bool useDeltaR_;
  /// switch to choose whether an outlier rejection
  /// should be applied or not
  bool useMaxDist_;
  /// threshold for outliers in the case that useMaxDist_
  /// =true
  double maxDist_;
  /// verbosity level
  int verbosity_;
};

template<typename C>
TtJetPartonMatch<C>::TtJetPartonMatch(const edm::ParameterSet& cfg):
  partons_   (cfg.getParameter<std::vector<std::string> >("partonsToIgnore")),
  genEvt_    (consumes<TtGenEvent>(edm::InputTag("genEvt"))),
  jets_      (consumes<edm::View<reco::Jet> >(cfg.getParameter<edm::InputTag>("jets"))),
  maxNJets_  (cfg.getParameter<int>                      ("maxNJets"       )),
  maxNComb_  (cfg.getParameter<int>                      ("maxNComb"       )),
  algorithm_ (readAlgorithm(cfg.getParameter<std::string>("algorithm"      ))),
  useDeltaR_ (cfg.getParameter<bool>                     ("useDeltaR"      )),
  useMaxDist_(cfg.getParameter<bool>                     ("useMaxDist"     )),
  maxDist_   (cfg.getParameter<double>                   ("maxDist"        )),
  verbosity_ (cfg.getParameter<int>                      ("verbosity"      ))
{
  // produces a vector of jet/lepton indices in the order of
  //  * TtSemiLepEvtPartons
  //  * TtFullHadEvtPartons
  //  * TtFullLepEvtPartons
  // and vectors of the corresponding quality parameters
  produces<std::vector<std::vector<int> > >();
  produces<std::vector<double> >("SumPt");
  produces<std::vector<double> >("SumDR");
  produces<int>("NumberOfConsideredJets");
}

template<typename C>
TtJetPartonMatch<C>::~TtJetPartonMatch()
{
}

template<typename C>
void
TtJetPartonMatch<C>::produce(edm::Event& evt, const edm::EventSetup& setup)
{
  // will write
  // * parton match
  // * sumPt
  // * sumDR
  // to the event
  std::auto_ptr<std::vector<std::vector<int> > > match(new std::vector<std::vector<int> >);
  std::auto_ptr<std::vector<double> > sumPt(new std::vector<double>);
  std::auto_ptr<std::vector<double> > sumDR(new std::vector<double>);
  std::auto_ptr<int> pJetsConsidered(new int);

  // get TtGenEvent and jet collection from the event
  edm::Handle<TtGenEvent> genEvt;
  evt.getByToken(genEvt_, genEvt);

  edm::Handle<edm::View<reco::Jet> > topJets;
  evt.getByToken(jets_, topJets);

  // fill vector of partons in the order of
  //  * TtFullLepEvtPartons
  //  * TtSemiLepEvtPartons
  //  * TtFullHadEvtPartons
  std::vector<const reco::Candidate*> partons = partons_.vec(*genEvt);

  // prepare vector of jets
  std::vector<const reco::Candidate*> jets;
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
    jets.push_back( (const reco::Candidate*) &(*topJets)[ij] );
  }
  *pJetsConsidered = jets.size();

  // do the matching with specified parameters
  JetPartonMatching jetPartonMatch(partons, jets, algorithm_, useMaxDist_, useDeltaR_, maxDist_);

  // print some info for each event
  // if corresponding verbosity level set
  if(verbosity_>0)
    jetPartonMatch.print();

  for(unsigned int ic=0; ic<jetPartonMatch.getNumberOfAvailableCombinations(); ++ic) {
    if((int)ic>=maxNComb_ && maxNComb_>=0) break;
    std::vector<int> matches = jetPartonMatch.getMatchesForPartons(ic);
    partons_.expand(matches); // insert dummy indices for partons that were chosen to be ignored
    match->push_back( matches );
    sumPt->push_back( jetPartonMatch.getSumDeltaPt(ic) );
    sumDR->push_back( jetPartonMatch.getSumDeltaR (ic) );
  }
  evt.put(match);
  evt.put(sumPt, "SumPt");
  evt.put(sumDR, "SumDR");
  evt.put(pJetsConsidered, "NumberOfConsideredJets");
}

template<typename C>
JetPartonMatching::algorithms
TtJetPartonMatch<C>::readAlgorithm(const std::string& str)
{
  if     (str == "totalMinDist"    ) return JetPartonMatching::totalMinDist;
  else if(str == "minSumDist"      ) return JetPartonMatching::minSumDist;
  else if(str == "ptOrderedMinDist") return JetPartonMatching::ptOrderedMinDist;
  else if(str == "unambiguousOnly" ) return JetPartonMatching::unambiguousOnly;
  else throw cms::Exception("Configuration")
    << "Chosen algorithm is not supported: " << str << "\n";
}

#endif
