#ifndef TauDQMFileLoader_h
#define TauDQMFileLoader_h

/** \class TauDQMFileLoader
 *  
 *  Class to load DQM monitoring elements from ROOT files into DQMStore --> hanged name to avoid conflict with TauAnalysis package
 *
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

class TauDQMFileLoader : public edm::EDAnalyzer
{
  typedef std::vector<std::string> vstring;
  typedef std::set<std::string> sstring;

  struct cfgEntryFileSet
  {
    cfgEntryFileSet(const std::string&, const edm::ParameterSet&);
    void print() const;
    std::string name_;
    vstring inputFileNames_;
    double scaleFactor_;
    std::string dqmDirectory_store_;
  };

 public:
  explicit TauDQMFileLoader(const edm::ParameterSet&);
  virtual ~TauDQMFileLoader();
  virtual void analyze(const edm::Event&, const edm::EventSetup&);
  virtual void endJob(){}
  virtual void endRun(const edm::Run& r, const edm::EventSetup& c);  

private:
  std::map<std::string, cfgEntryFileSet> fileSets_;
  std::map<std::string, sstring> subDirectoryMap_;
  int cfgError_;
};

#endif


