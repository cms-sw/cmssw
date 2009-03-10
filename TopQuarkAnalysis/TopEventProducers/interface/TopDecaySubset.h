#include <memory>
#include <string>
#include <vector>
#include <map>

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"

#include "FWCore/ParameterSet/interface/InputTag.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"


class TopDecaySubset : public edm::EDProducer {

 public:

  /// supported modes to fill the new vectors 
  /// of gen particles
  enum  FillMode {kStable, kME, kBeforePS, kAfterPS};

  /// default constructor
  explicit TopDecaySubset(const edm::ParameterSet&);
  /// default destructor
  ~TopDecaySubset();

  /// write TopDecaySubset into the event
  virtual void produce(edm::Event&, const edm::EventSetup&);

 private:

  /// fill output collection depending on whether the W
  /// boson is contained in the generator listing or not
  void fillOutput(const reco::GenParticleCollection& src, reco::GenParticleCollection& target, const reco::GenParticleRefProd& ref, FillMode mode);
  /// check whether the W boson is contained in the 
  /// original gen particle listing or not
  bool wInDecayChain(const reco::GenParticleCollection& src, const int& partId);
  /// fill output vector with full decay chain with 
  /// intermediate W bosons
  void fromFullListing(const reco::GenParticleCollection& src, reco::GenParticleCollection& target, const int& partId, FillMode mode);
  /// fill output vector with full decay chain w/o  
  /// intermediate W bosons
  void fromTruncListing(const reco::GenParticleCollection& src, reco::GenParticleCollection& target, const int& partId, FillMode mode);
  /// clear references
  void clearReferences();
  /// fill references for output vector
  void fillReferences(const reco::GenParticleRefProd& refProd, reco::GenParticleCollection& target);
  /// calculate lorentz vector from input 
  /// (dedicated to top reconstruction)
  reco::Particle::LorentzVector p4(const std::vector<reco::GenParticle>::const_iterator top, int statusFlag);
  /// calculate lorentz vector from input
    reco::Particle::LorentzVector p4(const reco::GenParticle::const_iterator part, int statusFlag);
  /// recursively fill vector for all further 
  /// dacay particles of a given particle
  void addDaughters(int& idx, const reco::GenParticle::const_iterator part, reco::GenParticleCollection& target, bool recursive=true);
  /// fill vector including all radiations from quarks 
  /// originating from W/top
  void addRadiation(int& idx, const reco::GenParticle::const_iterator part, reco::GenParticleCollection& target);
  /// print the whole decay chain if particle with pdgId is 
  /// contained in the top decay chain
  void printTarget(reco::GenParticleCollection& target);
  /// print the whole listing if particle with pdgId is 
  /// contained in the top decay chain
  void printSource(const reco::GenParticleCollection& src);

 private:

  // index in new evt listing of parts with daughters; 
  // has to be set to -1 in produce to deliver consistent 
  // results!
  int motherPartIdx_;                    
  // management of daughter indices for fillRefs
  std::map<int,std::vector<int> > refs_; 
  // input tag for the genParticle source
  edm::InputTag src_;                    
};
