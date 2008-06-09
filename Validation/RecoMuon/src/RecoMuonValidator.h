#ifndef Validation_RecoMuon_RecoMuonValidator_H
#define Validation_RecoMuon_RecoMuonValidator_H

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/InputTag.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "RecoMuon/TrackingTools/interface/MuonServiceProxy.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

class MuonHisto;
class DQMStore;
class TFileService;
class TFileDirectory;

class RecoMuonValidator : public edm::EDAnalyzer
{
 public:
  RecoMuonValidator(const edm::ParameterSet& pset);
  ~RecoMuonValidator();
  
  virtual void beginJob(const edm::EventSetup& eventSetup);
  virtual void endJob();
  virtual void analyze(const edm::Event& event, const edm::EventSetup& eventSetup);
  
 protected:
  edm::InputTag simPtclLabel_;
  edm::InputTag recoMuonLabel_;
  //edm::InputTag staMuonLabel_;
  //edm::InputTag glbMuonLabel_;
  
  std::string histoManager_;
  std::string outputFileName_;

  DQMStore * theDQMService;
  TFileService * theTFileService;
  TFileDirectory * theTFileDirectory;
  
  MuonServiceProxy * theMuonService;
  std::string seedPropagatorName_;

  std::map<std::string, MuonHisto> fillHisto_;
};

#endif

