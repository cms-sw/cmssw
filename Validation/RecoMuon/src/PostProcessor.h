#ifndef Validation_RecoMuon_PostProcessor_H
#define Validation_RecoMuon_PostProcessor_H

/*
 *  Class:PostProcessor 
 *
 *  DQM histogram post processor
 *
 *  $Date: 2008/05/27 23:07:22 $
 *  $Revision: 1.1 $
 *
 *  \author Junghwan Goh - SungKyunKwan University
 */

#include "Validation/RecoMuon/src/PostProcessor.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include <string>
#include <vector>

class DQMStore;

class PostProcessor : public edm::EDAnalyzer
{
 public:
  PostProcessor(const edm::ParameterSet& pset);
  ~PostProcessor() {};

  void analyze(const edm::Event& event, const edm::EventSetup& eventSetup) {};
  void endJob();

  void computeEfficiency(const std::string& efficMEName, const std::string& efficMETitle,
                         const std::string& recoMEName, const std::string& simMEName);
  void computeResolution(const std::string& fitMEPrefix, const std::string& fitMETItlePrefix, 
                         const std::string& srcMEName);

 private:
  DQMStore* theDQM;
  std::string subDir_;
  std::string outputFileName_;
  std::vector<std::string> commands_;
};

#endif
