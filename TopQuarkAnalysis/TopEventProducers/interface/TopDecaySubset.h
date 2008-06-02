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
  void fillOutput(const reco::GenParticleCollection&, reco::GenParticleCollection&);
  void fillRefs(const reco::GenParticleRefProd&, reco::GenParticleCollection&);

  reco::Particle::LorentzVector getP4(const reco::GenParticle::const_iterator, 
				      const reco::GenParticle::const_iterator, int, double);
  
  reco::Particle::LorentzVector getP4(const reco::GenParticle::const_iterator, 
				      const reco::GenParticle::const_iterator, int);
 protected:
  void fillTree(int& index, const reco::GenParticle&, reco::GenParticleCollection&);
 private:
  edm::InputTag src_;  
  std::map<int,std::vector<int> > refs_; //management of daughter
                                         //indices for fillRefs
};
