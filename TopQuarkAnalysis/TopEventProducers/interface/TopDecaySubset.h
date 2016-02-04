#include <memory>
#include <string>
#include <vector>
#include <map>

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"

#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"

/**
   \class   TopDecaySubset TopDecaySubset.h "TopQuarkAnalysis/TopEventProducers/interface/TopDecaySubset.h"

   \brief   Module to produce the subset of generator particles directly contained in top decay chains

   The module produces the subset of generator particles directly contained in top decay chains. The 
   particles are saved as a collection of reco::GenParticles. Depending on the configuration of the module
   the 4-vector kinematics can be taken from the status 3 particles (ME before parton showering) of from 
   the status 2 particles (after parton showering), additioanlly radiated gluons may be considered during
   the creation if the subset or not. 
*/


class TopDecaySubset : public edm::EDProducer {

 public:
  /// supported modes to fill the new vectors 
  /// of gen particles
  enum  FillMode {kStable, kME};
  /// classification of potential shower types
  enum ShowerModel{kStart=-1, kNone, kPythia, kHerwig};

  /// default constructor
  explicit TopDecaySubset(const edm::ParameterSet& cfg);
  /// default destructor
  ~TopDecaySubset();
  /// write output into the event
  virtual void produce(edm::Event& event, const edm::EventSetup& setup);

 private:
  /// find top quarks in list of input particles
  std::vector<const reco::GenParticle*> findTops(const reco::GenParticleCollection& parts);
  /// check the decay chain for the used shower model
  ShowerModel checkShowerModel(const std::vector<const reco::GenParticle*>& tops) const;
  /// check whether W bosons are contained in the original gen particle listing
  void checkWBosons(std::vector<const reco::GenParticle*>& tops) const;
  /// fill output vector for full decay chain 
  void fillListing(const std::vector<const reco::GenParticle*>& tops, reco::GenParticleCollection& target);

  /// clear references
  void clearReferences();
  /// fill references for output vector
  void fillReferences(const reco::GenParticleRefProd& refProd, reco::GenParticleCollection& target);
  /// calculate lorentz vector from input (dedicated to top reconstruction)
  reco::Particle::LorentzVector p4(const std::vector<const reco::GenParticle*>::const_iterator top, int statusFlag);
  /// calculate lorentz vector from input
  reco::Particle::LorentzVector p4(const reco::GenParticle::const_iterator part, int statusFlag);
  /// recursively fill vector for all further decay particles of a given particle
  void addDaughters(int& idx, const reco::GenParticle::const_iterator part, reco::GenParticleCollection& target, bool recursive=true);
  /// fill vector including all radiations from quarks originating from W/top
  void addRadiation(int& idx, const reco::GenParticle::const_iterator part, reco::GenParticleCollection& target);

 private:
  /// input tag for the genParticle source
  edm::InputTag src_;
  /// add radiation or not?
  bool addRadiation_;
  /// print the whole list of input particles or not?
  /// mode of decaySubset creation 
  FillMode fillMode_;
  /// parton shower mode (filled in checkShowerModel)
  ShowerModel showerModel_;

  /// index in new evt listing of parts with daughters; 
  /// has to be set to -1 in produce to deliver consistent 
  /// results!
  int motherPartIdx_;                    
  /// management of daughter indices for fillRefs
  std::map<int,std::vector<int> > refs_; 
};
