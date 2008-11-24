#ifndef Validation_RecoMuon_PostProcessor_H
#define Validation_RecoMuon_PostProcessor_H

/*
 *  Class:PostProcessor 
 *
 *  DQM histogram post processor
 *
 *  $Date: 2008/11/05 17:28:44 $
 *  $Revision: 1.5 $
 *
 *  \author Junghwan Goh - SungKyunKwan University
 */

#include "Validation/RecoMuon/src/PostProcessor.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include <string>
#include <vector>
#include <boost/tokenizer.hpp>

class DQMStore;

typedef boost::escaped_list_separator<char> elsc;

class PostProcessor : public edm::EDAnalyzer
{
 public:
  PostProcessor(const edm::ParameterSet& pset);
  ~PostProcessor() {};

  void analyze(const edm::Event& event, const edm::EventSetup& eventSetup) {};
  void endJob();

  void computeEfficiency(const std::string& startDir, 
                         const std::string& efficMEName, const std::string& efficMETitle,
                         const std::string& recoMEName, const std::string& simMEName);
  void computeResolution(const std::string &, 
                         const std::string& fitMEPrefix, const std::string& fitMETItlePrefix, 
                         const std::string& srcMEName);

 private:
  unsigned int verbose_;

  DQMStore* theDQM;
  std::string subDir_;
  std::string outputFileName_;
  std::vector<std::string> effCmds_, resCmds_;
};

#endif

/* vim:set ts=2 sts=2 sw=2 expandtab: */
