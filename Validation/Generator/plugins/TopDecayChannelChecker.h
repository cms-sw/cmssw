#include <memory>
#include <string>
#include <vector>
#include <fstream>
#include <iostream>

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"

/**
   \class   TopDecayChannelChecker TopDecayChannelChecker.h "Validation/Generator/plugins/TopDecayChannelChecker.h"

   \brief   Plugin to monitor the properties of top Monte Carlo samples on generator level

   The module monitors the properties of top Monte Carlo samples from different generators 
   on generator level. The validity ot the decay chain and the decay branches are checked.
*/


class TopDecayChannelChecker : public edm::EDAnalyzer {

 public:
  /// classification of potential shower types: 
  /// *kPythia* is characterized by: ME (status 3) ->after showering (status 2).
  /// *kHerwig* is characterized by: (status 2)-> ME (status 3)->after showering 
  /// (status 2).
  enum ShowerType{kNone, kPythia, kHerwig};
  
 public:
  /// constructor
  explicit TopDecayChannelChecker(const edm::ParameterSet& cfg);
  /// destructor
  ~TopDecayChannelChecker();
     
 private:
  /// all that needs to be done at the beginning of a run
  virtual void beginJob();
  /// all that needs to done during the event loop
  virtual void analyze(const edm::Event& event, const edm::EventSetup& setup);
  /// all that needs to be done at the end of a run
  virtual void endJob() ;
  /// check the decay chain for different shower types
  ShowerType showerType(const edm::View<reco::GenParticle>& parts) const;
  /// search for particle with pdgId (for top)
  bool search(edm::View<reco::GenParticle>::const_iterator& part, int pdgId, const std::string& inputType) const;
  /// search for particle with pdgId (overloaded for top daughters)
  bool search(reco::GenParticle::const_iterator& part, int pdgId, const std::string& inputType) const;
  /// check tau decay to be leptonic, 1-prong or 3-prong
  int tauDecay(const reco::Candidate& part) const;
  /// count the number of charged particles for tau decays
  unsigned int countProngs(const reco::Candidate& part) const;
  /// dump the characteristics of Candidate part 
  void dumpDecayChain(const edm::View<reco::GenParticle>& src) const;

 private:
  /// this will be the name of the output file 
  std::string outputFileName_;
  /// number of events for which to print the 
  /// full decay chain to the log output
  unsigned int log_;
  /// generated particle collection src
  edm::InputTag src_;
  /// DQM service
  bool saveDQMMEs_ ;

  /// event counter for decay chain 
  /// logging
  unsigned int evts_;
  /// DQM service
  DQMStore* dqmStore_;
  /// histogram container
  std::map<std::string, MonitorElement*> hists_;
};

