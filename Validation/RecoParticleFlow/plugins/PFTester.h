

#ifndef PFTESTER_H
#define PFTESTER_H

// author: Mike Schmitt (The University of Florida)
// date: 11/7/2007

#include "DQMServices/Core/interface/DQMStore.h"

#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"

#include <map>
#include <string>

class PFTester : public edm::EDAnalyzer {
public:
  explicit PFTester(const edm::ParameterSet &);
  ~PFTester() override;

  void analyze(const edm::Event &, const edm::EventSetup &) override;
  void beginJob() override;
  void endJob() override;

private:
  // DAQ Tools
  DQMStore *dbe_;
  std::map<std::string, MonitorElement *> me;

  // Inputs from Configuration File
  std::string outputFile_;
  edm::EDGetTokenT<reco::PFCandidateCollection> inputPFlowLabel_tok_;
};

#endif  // PFTESTER_H
