#ifndef Validation_RecoMuon_RecoMuonValidator_H
#define Validation_RecoMuon_RecoMuonValidator_H

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/InputTag.h"

#include <map>

class DQMStore;
class MonitorElement;
class MuonServiceProxy;
class TrackAssociatorBase;

class RecoMuonValidator : public edm::EDAnalyzer
{
 public:
  RecoMuonValidator(const edm::ParameterSet& pset);
  ~RecoMuonValidator();
  
  virtual void beginJob(const edm::EventSetup& eventSetup);
  virtual void endJob();
  virtual void analyze(const edm::Event& event, const edm::EventSetup& eventSetup);

 protected:
  edm::InputTag simLabel_;
  edm::InputTag recoLabel_;

  edm::InputTag assocLabel_;
  
  std::string outputFileName_;
  std::string subDir_;

  MuonServiceProxy * theMuonService;
  DQMStore * theDQM;
  
  std::map<std::string, MonitorElement*> meMap_;

  bool doAbsEta_;
  bool doAssoc_;

  TrackAssociatorBase* theAssociator;
};

#endif

