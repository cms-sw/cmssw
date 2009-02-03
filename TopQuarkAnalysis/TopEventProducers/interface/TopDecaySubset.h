#include <memory>
#include <string>
#include <vector>
#include <map>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/InputTag.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"

class TopDecaySubset : public edm::EDProducer {
 public:
  explicit TopDecaySubset(const edm::ParameterSet&);
  ~TopDecaySubset();
  
  virtual void produce(edm::Event&, const edm::EventSetup&);
  /// check whether w is in original gen particle listing or not
    bool wInDecayChain(const reco::GenParticleCollection&, const int& partId);
  /// fill output vector with full decay chain with intermediate w's
  void fillFromFullListing(const reco::GenParticleCollection&, reco::GenParticleCollection&, const int& partId);
  /// fill output vector with full decay chain w/o  intermediate w's
  void fillFromTruncatedListing(const reco::GenParticleCollection&, reco::GenParticleCollection&, const int& partId);
  /// clear references
  void clearReferences();
  /// fill references for output vector
  void fillReferences(const reco::GenParticleRefProd&, reco::GenParticleCollection&);
  /// calculate lorentz vector from input with additional mass constraint
  reco::Particle::LorentzVector getP4(const reco::GenParticle::const_iterator, 
				      const reco::GenParticle::const_iterator, int pdgId, double mass);
  /// calculate lorentz vector from input
  reco::Particle::LorentzVector getP4(const reco::GenParticle::const_iterator, 
				      const reco::GenParticle::const_iterator, int pdgId);
 protected:
  /// fill vector recursively for all further decay particles of a tau
  void fillTree(int& index, const reco::GenParticle::const_iterator, reco::GenParticleCollection&);
  /// print the whole decay chain if particle with pdgId is contained in the top decay chain
  void printTarget(reco::GenParticleCollection&, const int& pdgId);
  /// print the whole listing if particle with pdgId is contained in the top decay chain
  void printSource(const reco::GenParticleCollection&, const int& pdgId);

 private:

  unsigned int pdg_;                     // pdgId for special selection for printout; if set to 0 the listing 
                                         // is printed starting from the top quark (pdgId==6)
  int motherPartIdx_;                    // index in new evt listing of parts with daughters; has to be set 
                                         // to -1 in produce to deliver consistent results!
  std::map<int,std::vector<int> > refs_; // management of daughter indices for fillRefs
  edm::InputTag src_;                    // input tag for genParticle source
};
