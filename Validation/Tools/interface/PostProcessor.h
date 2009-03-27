#ifndef Validation_Tools_PostProcessor_H
#define Validation_Tools_PostProcessor_H

/*
 *  Class:PostProcessor 
 *
 *  DQM histogram post processor
 *
 *  $Date: 2008/12/19 17:14:39 $
 *  $Revision: 1.3 $
 *
 *  \author Junghwan Goh - SungKyunKwan University
 */

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include <string>
#include <vector>
#include <boost/tokenizer.hpp>

class DQMStore;
class MonitorElement;

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
                         const std::string& recoMEName, const std::string& simMEName,const std::string& type="eff");
  void computeResolution(const std::string &, 
                         const std::string& fitMEPrefix, const std::string& fitMETitlePrefix, 
                         const std::string& srcMEName);

  void limitedFit(MonitorElement * srcME, MonitorElement * meanME, MonitorElement * sigmaME);

 private:
  unsigned int verbose_;
  bool isWildcardUsed_;

  DQMStore* theDQM;
  std::vector<std::string> subDirs_;
  std::string outputFileName_;
  std::vector<std::string> effCmds_, resCmds_;
  bool resLimitedFit_;
};

#endif

/* vim:set ts=2 sts=2 sw=2 expandtab: */
