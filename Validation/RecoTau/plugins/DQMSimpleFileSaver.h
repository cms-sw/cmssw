#ifndef TauDQMSimpleFileSaver_h
#define TauDQMSimpleFileSaver_h

/** \class TauDQMSimpleFileSaver
 *  
 *  Class to write all monitor elements registered in DQMStore into ROOT file
 *  (without any naming restrictions imposed by "regular" DQMFileSaver)
 *
 *  \author Christian Veelken, UC Davis
 */

// framework & common header files
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DQMServices/Core/interface/DQMStore.h"

#include <string>

class TauDQMSimpleFileSaver : public edm::one::EDAnalyzer<edm::one::SharedResources> {
public:
  typedef dqm::legacy::DQMStore DQMStore;
  typedef dqm::legacy::MonitorElement MonitorElement;

  explicit TauDQMSimpleFileSaver(const edm::ParameterSet&);
  ~TauDQMSimpleFileSaver() override;
  void analyze(const edm::Event&, const edm::EventSetup&) override;
  void endJob() override;

private:
  std::string outputFileName_;
  int cfgError_;
};

#endif
