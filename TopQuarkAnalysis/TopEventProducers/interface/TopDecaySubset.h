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

namespace TopDecayID{
  static const int stable = 2;
  static const int unfrag = 3;
  static const int tID    = 6;
  static const int bID    = 5;
  static const int glueID = 21;
  static const int photID = 22;
  static const int ZID    = 23;
  static const int WID    = 24;
  static const int tauID  = 15;
}

class TopDecaySubset : public edm::EDProducer {
 public:
  explicit TopDecaySubset(const edm::ParameterSet&);
  ~TopDecaySubset();
  
  virtual void produce(edm::Event&, const edm::EventSetup&);
  /// fill output vector with full decay chain (for pythia like generator listing)
  void fillPythiaOutput(const reco::GenParticleCollection&, reco::GenParticleCollection&);
  /// fill output vector with full decay chain (for madgraph like generator listing)
  void fillMadgraphOutput(const reco::GenParticleCollection&, reco::GenParticleCollection&);
  /// fill references for output vector
  void fillRefs(const reco::GenParticleRefProd&, reco::GenParticleCollection&);
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
  void print(reco::GenParticleCollection&, int pdgId);
  /// print the whole listing if particle with pdgId is contained in the top decay chain
  void printSource(const reco::GenParticleCollection&, int pdgId);

 private:

  unsigned int pdg_;                     // pdgId for special selection
                                         // for printout
  edm::InputTag src_;  
  unsigned int genType_;                 // switch for generator listing
  std::map<int,std::vector<int> > refs_; // management of daughter
                                         // indices for fillRefs
};
