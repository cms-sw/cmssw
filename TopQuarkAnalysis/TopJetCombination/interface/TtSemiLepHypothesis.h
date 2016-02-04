#ifndef TtSemiLepHypothesis_h
#define TtSemiLepHypothesis_h

#include <memory>
#include <vector>

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/PatCandidates/interface/Jet.h"
#include "DataFormats/PatCandidates/interface/MET.h"
#include "DataFormats/RecoCandidate/interface/RecoCandidate.h"
#include "DataFormats/Candidate/interface/ShallowClonePtrCandidate.h"

#include "AnalysisDataFormats/TopObjects/interface/TtSemiLeptonicEvent.h"

/*
   \class   TtSemiLepHypothesis TtSemiLepHypothesis.h "TopQuarkAnalysis/TopJetCombination/interface/TtSemiLepHypothesis.h"

   \brief   Interface class for the creation of semi-leptonic ttbar event hypotheses

   The class provides an interface for the creation of semi-leptonic ttbar event hypotheses. Input information is read 
   from the event content and the proper candidate creation is taken care of. Hypotheses are characterized by the 
   CompositeCandidate made of a ttbar pair (including all its decay products in a parton level interpretation) and an 
   enumerator type key to specify the algorithm to determine the candidate (hypothesis cklass). The buildKey and the 
   buildHypo class have to implemented by derived classes.
**/

class TtSemiLepHypothesis : public edm::EDProducer {

 public:
  /// default constructor
  explicit TtSemiLepHypothesis(const edm::ParameterSet&);
  /// default destructor
  ~TtSemiLepHypothesis();

 protected:
  /// produce the event hypothesis as CompositeCandidate and Key
  virtual void produce(edm::Event&, const edm::EventSetup&);
  /// reset candidate pointers before hypo build process
  void resetCandidates();
  /// helper function to construct the proper correction level string for corresponding quarkType,
  /// for unknown quarkTypes an empty string is returned
  std::string jetCorrectionLevel(const std::string& quarkType);
  /// use one object in a collection to set a ShallowClonePtrCandidate
  template <typename C>
  void setCandidate(const edm::Handle<C>& handle, const int& idx, reco::ShallowClonePtrCandidate*& clone);
  /// use one object in a jet collection to set a ShallowClonePtrCandidate with proper jet corrections
  void setCandidate(const edm::Handle<std::vector<pat::Jet> >& handle, const int& idx, reco::ShallowClonePtrCandidate*& clone, const std::string& correctionLevel);
  /// set neutrino, using mW = 80.4 to calculate the neutrino pz
  void setNeutrino(const edm::Handle<std::vector<pat::MET> >& met, const edm::Handle<edm::View<reco::RecoCandidate> >& leps, const int& idx, const int& type);
  /// return key
  int key() const { return key_; };
  /// return event hypothesis
  reco::CompositeCandidate hypo();
  /// check if index is in valid range of selected jets
  bool isValid(const int& idx, const edm::Handle<std::vector<pat::Jet> >& jets){ return (0<=idx && idx<(int)jets->size()); };
  /// determine lepton type of reco candidate and return a corresponding WDecay::LeptonType; the type is kNone if it is whether a muon nor an electron 
  WDecay::LeptonType leptonType(const reco::RecoCandidate* cand);

  // -----------------------------------------
  // implemet the following two functions
  // for a concrete event hypothesis
  // -----------------------------------------

  /// build the event hypothesis key
  virtual void buildKey() = 0;
  /// build event hypothesis from the reco objects of a semi-leptonic event 
  virtual void buildHypo(edm::Event& event,
			 const edm::Handle<edm::View<reco::RecoCandidate> >& lepton, 
			 const edm::Handle<std::vector<pat::MET> >& neutrino, 
			 const edm::Handle<std::vector<pat::Jet> >& jets, 
			 std::vector<int>& jetPartonAssociation,
			 const unsigned int iComb) = 0;

 protected:
  /// internal check whether the match information exists or not, 
  /// if false a blind dummy match vector will be used internally
  bool getMatch_;
  /// input label for all necessary collections
  edm::InputTag jets_;
  edm::InputTag leps_;
  edm::InputTag mets_;
  edm::InputTag match_;
  /// specify the desired jet correction level (the default should 
  /// be L3Absolute-'abs')
  std::string jetCorrectionLevel_;
  /// hypothesis key (to be set by the buildKey function)
  int key_;
  /// number of real neutrino solutions:
  /// -1 if not determined, 0 if only complex, 2 if two real solutions
  int numberOfRealNeutrinoSolutions_;
  /// candidates for internal use for the creation of the hypothesis 
  /// candidate
  reco::ShallowClonePtrCandidate *lightQ_;
  reco::ShallowClonePtrCandidate *lightQBar_;
  reco::ShallowClonePtrCandidate *hadronicB_;
  reco::ShallowClonePtrCandidate *leptonicB_;
  reco::ShallowClonePtrCandidate *neutrino_;
  reco::ShallowClonePtrCandidate *lepton_;
};

// has to be placed in the header since otherwise the function template
// would cause unresolved references in classes derived from this base class
template<typename C>
void
TtSemiLepHypothesis::setCandidate(const edm::Handle<C>& handle, const int& idx, reco::ShallowClonePtrCandidate* &clone) {
  typedef typename C::value_type O;
  edm::Ptr<O> ptr = edm::Ptr<O>(handle, idx);
  clone = new reco::ShallowClonePtrCandidate( ptr, ptr->charge(), ptr->p4(), ptr->vertex() );
}

#endif
