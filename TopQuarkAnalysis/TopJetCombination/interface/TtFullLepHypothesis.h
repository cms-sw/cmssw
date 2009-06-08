#ifndef TtFullLepHypothesis_h
#define TtFullLepHypothesis_h

#include <memory>
#include <vector>

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/Candidate/interface/ShallowClonePtrCandidate.h"
#include "DataFormats/PatCandidates/interface/Electron.h"
#include "DataFormats/PatCandidates/interface/Muon.h"
#include "DataFormats/PatCandidates/interface/Jet.h"
#include "DataFormats/PatCandidates/interface/MET.h"
#include "DataFormats/RecoCandidate/interface/RecoCandidate.h"

#include "AnalysisDataFormats/TopObjects/interface/TtFullLeptonicEvent.h"

class TtFullLepHypothesis : public edm::EDProducer {

 public:

  explicit TtFullLepHypothesis(const edm::ParameterSet&);
  ~TtFullLepHypothesis();

 protected:
  
  /// produce the event hypothesis as CompositeCandidate and Key
  virtual void produce(edm::Event&, const edm::EventSetup&);
  /// reset candidate pointers before hypo build process
  void resetCandidates();
  /// use one object in a collection to set a ShallowClonePtrCandidate
  template <typename O, template<typename> class C>
  void setCandidate(const edm::Handle<C<O> >&, const int&, reco::ShallowClonePtrCandidate*&);
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
  
  edm::InputTag match_;
  edm::InputTag elecs_;
  edm::InputTag mus_;
  edm::InputTag jets_;
  edm::InputTag mets_;

  int key_;

  reco::ShallowClonePtrCandidate *lepton_;
  reco::ShallowClonePtrCandidate *leptonBar_;
  reco::ShallowClonePtrCandidate *b_;
  reco::ShallowClonePtrCandidate *bBar_;
  reco::ShallowClonePtrCandidate *neutrino_;
  reco::ShallowClonePtrCandidate *neutrinoBar_;
  //reco::ShallowClonePtrCandidate *met_;
  
  reco::LeafCandidate* recNu;
  reco::LeafCandidate* recNuBar;	//these are needed for genmatch hypothesis	
};

// unfortunately this has to be placed in the header since otherwise the function template
// would cause unresolved references in classes derived from this base class
template <typename O, template<typename> class C>
void
TtFullLepHypothesis::setCandidate(const edm::Handle<C<O> >& handle, const int& idx, reco::ShallowClonePtrCandidate* &clone) {
  edm::Ptr<O> ptr = edm::Ptr<O>(handle, idx);
  clone = new reco::ShallowClonePtrCandidate( ptr, ptr->charge(), ptr->p4(), ptr->vertex() );
}

#endif
