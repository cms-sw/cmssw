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
   \class   TopDecayChannelDQM TopDecayChannelDQM.h "Validation/Generator/plugins/TopDecayChannelDQM.h"

   \brief   Plugin to monitor the properties of top monte carlo samples on generator level

   The module monitors the properties of top monte carlo samples from different generators 
   on generator level. The following generators are explicitely supported:

   * Pythia (lund fragmentation model)
   * MC@NLO (herwig cluster hadronization model)
   * Madgraph (lund fragmentation model)

   while other generators (using the same fragmentation models) might be implicitely supported.
   The validity of the decay chain and the decay branches are checked. And the following histo-
   grams are written to file: 

   * TopDecayChannel  : No of events per decay channel (full hadronic, single leptonic, dilep-
                        tonic).

   * TopDecayWBosons  : No of W bosons in the decay chain.

   * SemiLeptonType   : No of semi-leptonic events with different lepton types in the final 
                        state.
   * FullLeptonType   : No of dileptonic events with different lepton types in the final 
                        state.
   * TauDecayMode     : No of tau candidates with different decay modes in the final state.

   * TopDecayQuark    : No of different quark types, which were produced via the top quark 
                        decay.

   For more details on the filling have a look for the comments given in the implementation of
   the module or have a look to the TopValidationTutorialGEN TWiki page. The module exploits 
   the DQM infrastructure to be compatible with the DQM GUI.
*/


class TopDecayChannelDQM : public edm::EDAnalyzer {

 public:
  /// classification of potential shower types: 
  /// *kPythia* is characterized by: ME (status 3) ->after showering (status 2).
  /// *kHerwig* is characterized by: (status 2)-> ME (status 3)->after showering 
  /// (status 2).
  enum ShowerType{kNone, kPythia, kHerwig};
  
 public:
  /// constructor
  explicit TopDecayChannelDQM(const edm::ParameterSet& cfg);
  /// destructor
  ~TopDecayChannelDQM();
     
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
  /// number of events for which to print the 
  /// full decay chain to the log output
  unsigned int log_;
  /// generated particle collection src
  edm::InputTag src_;

  /// event counter for decay chain 
  /// logging
  unsigned int evts_;
  /// DQM service
  DQMStore* store_;
  /// histogram container
  std::map<std::string, MonitorElement*> hists_;
};

