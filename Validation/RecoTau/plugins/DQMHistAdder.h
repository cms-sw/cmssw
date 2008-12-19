#ifndef ElectroWeakAnalysis_EWKTau_DQMHistAdder_h
#define ElectroWeakAnalysis_EWKTau_DQMHistAdder_h

/** \class DQMHistAdder
 *  
 *  Class to add DQM monitoring elements stored in separate directories in DQMStore
 *  (in order to e.g. compute the total Standard Model expectation)
 *
 *  $Date: 2008/11/18 16:41:39 $
 *  $Revision: 1.1 $
 *  \author Christian Veelken, UC Davis
 */

// framework & common header files
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DQMServices/Core/interface/DQMDefinitions.h"

#include <TH1.h>

#include <vector>
#include <string>

class DQMHistAdder : public edm::EDAnalyzer
{
  typedef std::vector<std::string> vstring;

  struct cfgEntryAddJob
  {
    cfgEntryAddJob(const std::string&, const edm::ParameterSet&);
    void print() const;
    std::string name_;
    vstring dqmDirectories_input_;
    std::string dqmDirectory_output_;
  };

 public:
  explicit DQMHistAdder(const edm::ParameterSet&);
  virtual ~DQMHistAdder();
  virtual void analyze(const edm::Event&, const edm::EventSetup&);
  virtual void endJob();  

private:
  std::map<std::string, cfgEntryAddJob> addJobs_;
  int cfgError_;
};

#endif


