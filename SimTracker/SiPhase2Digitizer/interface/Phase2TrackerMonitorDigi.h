#ifndef Phase2TrackerMonitorDigi_h
#define Phase2TrackerMonitorDigi_h

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"

// DQM Histograming
class DQMStore;
class MonitorElement;

class Phase2TrackerMonitorDigi : public edm::EDAnalyzer {

public:

  explicit Phase2TrackerMonitorDigi(const edm::ParameterSet&);
  ~Phase2TrackerMonitorDigi();
  virtual void beginJob();
  virtual void beginRun(const edm::Run& iRun, const edm::EventSetup& iSetup);
  virtual void analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup);

  virtual void endJob(); 

  struct DigiMEs{
    MonitorElement* NumberOfDigis;
    MonitorElement* PositionOfDigis;
    MonitorElement* DigiCharge;
    MonitorElement* NumberOfClusters;
    MonitorElement* ClusterCharge;
    MonitorElement* ClusterWidth;
    MonitorElement* ClusterPosition;
  };

private:
  void bookHistos();
  void bookLayerHistos(unsigned int ilayer); 

  DQMStore* dqmStore_;
  edm::ParameterSet config_;
  std::map<unsigned int, DigiMEs> layerMEs;
  edm::InputTag digiSrc_;
};
#endif
