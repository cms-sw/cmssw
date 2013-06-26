#ifndef JETFILESAVER_H
#define JETFILESAVER_H

#include "DQMServices/Core/interface/DQMStore.h"

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include <string>
#include <map>
#include "DQMServices/Core/interface/MonitorElement.h"

class JetFileSaver: public edm::EDAnalyzer {
public:

  explicit JetFileSaver(const edm::ParameterSet&);

  virtual void analyze(const edm::Event&, const edm::EventSetup&);
  virtual void beginJob() ;
  virtual void endJob() ;

 private:

  // DAQ Tools
  DQMStore* dbe_;
  std::map<std::string, MonitorElement*> me;

  // Inputs from Configuration File
  std::string outputFile_;

};

#endif // JETFILESAVER_H
