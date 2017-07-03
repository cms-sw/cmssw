#ifndef TtFullLepHypothesis_h
#define TtFullLepHypothesis_h

#include <memory>
#include <vector>

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/PatCandidates/interface/Electron.h"
#include "DataFormats/PatCandidates/interface/Muon.h"
#include "DataFormats/PatCandidates/interface/Jet.h"
#include "DataFormats/PatCandidates/interface/MET.h"
#include "DataFormats/RecoCandidate/interface/RecoCandidate.h"
#include "DataFormats/Candidate/interface/ShallowClonePtrCandidate.h"

#include "AnalysisDataFormats/TopObjects/interface/TtFullLeptonicEvent.h"

/*
   \class   TtFullLepHypothesis TtFullLepHypothesis.h "TopQuarkAnalysis/TopJetCombination/interface/TtFullLepHypothesis.h"

   \brief   Interface class for the creation of full leptonic ttbar event hypotheses

   The class provides an interface for the creation of full leptonic ttbar event hypotheses. Input information is read
   from the event content and the proper candidate creation is taken care of. Hypotheses are characterized by the
   CompositeCandidate made of a ttbar pair (including all its decay products in a parton level interpretation) and an
   enumerator type key to specify the algorithm to determine the candidate (hypothesis class). The buildKey and the
   buildHypo class have to implemented by derived classes.
**/

class TtFullLepHypothesis : public edm::EDProducer {

 public:
  /// default constructor
  explicit TtFullLepHypothesis(const edm::ParameterSet&);
  /// default destructor
  ~TtFullLepHypothesis() override;

 protected:
  /// produce the event hypothesis as CompositeCandidate and Key
  void produce(edm::Event&, const edm::EventSetup&) override;
  /// reset candidate pointers before hypo build process
  void resetCandidates();
  /// use one object in a collection to set a ShallowClonePtrCandidate
  template <typename C>
  void setCandidate(const edm::Handle<C>& handle, const int& idx, reco::ShallowClonePtrCandidate*& clone);
  /// use one object in a jet collection to set a ShallowClonePtrCandidate with proper jet corrections
  void setCandidate(const edm::Handle<std::vector<pat::Jet> >& handle, const int& idx, reco::ShallowClonePtrCandidate*& clone, const std::string& correctionLevel);
  /// return key
  int key() const { return key_; };
  /// return event hypothesis
  reco::CompositeCandidate hypo();
  /// check if index is in valid range of selected jets
  bool isValid(const int& idx, const edm::Handle<std::vector<pat::Jet> >& jets){ return (0<=idx && idx<(int)jets->size()); };

  // -----------------------------------------
  // implemet the following two functions
  // for a concrete event hypothesis
  // -----------------------------------------

  /// build the event hypothesis key
  virtual void buildKey() = 0;
  /// build event hypothesis from the reco objects of a semi-leptonic event
  virtual void buildHypo(edm::Event& evt,
			 const edm::Handle<std::vector<pat::Electron > >& elecs,
			 const edm::Handle<std::vector<pat::Muon> >& mus,
			 const edm::Handle<std::vector<pat::Jet> >& jets,
			 const edm::Handle<std::vector<pat::MET> >& mets,
			 std::vector<int>& match,
			 const unsigned int iComb) = 0;

 protected:
  /// internal check whether the match information exists or not,
  /// if false a blind dummy match vector will be used internally
  bool getMatch_;
  /// input label for all necessary collections
  edm::EDGetTokenT<std::vector<std::vector<int> > > matchToken_;
  edm::EDGetTokenT<std::vector<pat::Electron> > elecsToken_;
  edm::EDGetTokenT<std::vector<pat::Muon> > musToken_;
  edm::EDGetTokenT<std::vector<pat::Jet> > jetsToken_;
  edm::EDGetTokenT<std::vector<pat::MET> > metsToken_;
  /// specify the desired jet correction level (the default should
  /// be L3Absolute-'abs')
  std::string jetCorrectionLevel_;
  /// hypothesis key (to be set by the buildKey function)
  int key_;
  /// candidates for internal use for the creation of the hypothesis
  /// candidate
  reco::ShallowClonePtrCandidate *lepton_;
  reco::ShallowClonePtrCandidate *leptonBar_;
  reco::ShallowClonePtrCandidate *b_;
  reco::ShallowClonePtrCandidate *bBar_;
  reco::ShallowClonePtrCandidate *neutrino_;
  reco::ShallowClonePtrCandidate *neutrinoBar_;
  //reco::ShallowClonePtrCandidate *met_;

  /// candidates needed for the genmatch hypothesis
  reco::LeafCandidate* recNu;
  reco::LeafCandidate* recNuBar;
};

// unfortunately this has to be placed in the header since otherwise the function template
// would cause unresolved references in classes derived from this base class
template<typename C>
void
TtFullLepHypothesis::setCandidate(const edm::Handle<C>& handle, const int& idx, reco::ShallowClonePtrCandidate* &clone) {
  typedef typename C::value_type O;
  edm::Ptr<O> ptr = edm::Ptr<O>(handle, idx);
  clone = new reco::ShallowClonePtrCandidate( ptr, ptr->charge(), ptr->p4(), ptr->vertex() );
}

#endif
