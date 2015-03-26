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

#include "SimDataFormats/GeneratorProducts/interface/GenEventInfoProduct.h"

/**
   \class   TopDecaySubset TopDecaySubset.h "TopQuarkAnalysis/TopEventProducers/interface/TopDecaySubset.h"

   \brief   Module to produce the subset of generator particles directly contained in top decay chains

   The module produces the subset of generator particles directly contained in top decay chains. The
   particles are saved as a collection of reco::GenParticles. Depending on the configuration of the module,
   the 4-vector kinematics can be taken from the status-3 particles (ME before parton showering) or from
   the status-2 particles (after parton showering), additionally radiated gluons may be considered during
   the creation of the subset or not.
*/


class TopDecaySubset : public edm::EDProducer {

 public:
  /// supported modes to fill the new vectors
  /// of gen particles
  enum  FillMode {kStable, kME};
  /// classification of potential shower types
  enum ShowerModel{kStart=-1, kNone, kPythia, kHerwig, kPythia8, kSherpa};
  /// supported modes to run the code
  enum RunMode {kRun1, kRun2};

  /// default constructor
  explicit TopDecaySubset(const edm::ParameterSet& cfg);
  /// default destructor
  ~TopDecaySubset();
  /// write output into the event
  virtual void produce(edm::Event& event, const edm::EventSetup& setup);

 private:
  /// find top quarks in list of input particles
  std::vector<const reco::GenParticle*> findTops(const reco::GenParticleCollection& parts);
  /// find primal top quarks (top quarks from the hard interaction)
  /// for Pythia6 this is identical to findDecayingTops
  std::vector<const reco::GenParticle*> findPrimalTops(const reco::GenParticleCollection& parts);
  /// find decaying top quarks (quarks that decay to qW)
  /// for Pythia6 this is identical to findPrimalTops
  std::vector<const reco::GenParticle*> findDecayingTops(const reco::GenParticleCollection& parts);
  /// find W bosons that come from top quark decays
  /// for Pythia6 this is identical to findDecayingW
  const reco::GenParticle* findPrimalW(const reco::GenParticle* top);
  /// find W bosons that come from top quark decays and decay themselves (end of the MC chain)
  /// for Pythia6 this is identical to findPrimalW
//  const reco::GenParticle* findDecayingW(const reco::GenParticle* top);
  /// find the last particle in a (potentially) long chain of state transitions
  /// e.g. top[status==22]-> top[status==44 -> top[status==44] ->
  /// top[status==44] -> top[status==62]
  /// this function would pick the top with status 62
  const reco::GenParticle* findLastParticleInChain(const reco::GenParticle* p);
  /// check the decay chain for the used shower model
  ShowerModel checkShowerModel(const std::vector<const reco::GenParticle*>& tops) const;
  ShowerModel checkShowerModel(edm::Event& event);
  /// check whether W bosons are contained in the original gen particle listing
  void checkWBosons(std::vector<const reco::GenParticle*>& tops) const;
  /// fill output vector for full decay chain
  void fillListing(const std::vector<const reco::GenParticle*>& tops, reco::GenParticleCollection& target);
  /// fill output vector for full decay chain
  void fillListing(const std::vector<const reco::GenParticle*>& primalTops,
				   const std::vector<const reco::GenParticle*>& decayingTops,
				   reco::GenParticleCollection& target);

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
  void addRadiation(int& idx, const reco::GenParticle* part, reco::GenParticleCollection& target);

 private:
  /// input tag for the genParticle source
  edm::EDGetTokenT<reco::GenParticleCollection> srcToken_;
  /// input tag for the genEventInfo source
  edm::EDGetTokenT<GenEventInfoProduct> genEventInfo_srcToken_;
  /// add radiation or not?
  bool addRadiation_;
  /// print the whole list of input particles or not?
  /// mode of decaySubset creation
  FillMode fillMode_;
  /// parton shower mode (filled in checkShowerModel)
  ShowerModel showerModel_;
  /// run mode (Run1 || Run2)
  RunMode runMode_;

  /// index in new evt listing of parts with daughters;
  /// has to be set to -1 in produce to deliver consistent
  /// results!
  int motherPartIdx_;
  /// management of daughter indices for fillRefs
  std::map<int,std::vector<int> > refs_;
};
