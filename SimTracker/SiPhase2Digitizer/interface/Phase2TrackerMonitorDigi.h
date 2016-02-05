#ifndef Phase2TrackerMonitorDigi_h
#define Phase2TrackerMonitorDigi_h

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "DataFormats/Common/interface/DetSetVector.h"

// DQM Histograming
class DQMStore;
class MonitorElement;
class PixelDigi;
class Phase2TrackerDigi;

class Phase2TrackerMonitorDigi : public edm::one::EDAnalyzer<> {

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
  edm::InputTag pixDigiSrc_;
  edm::InputTag otDigiSrc_;
  const edm::EDGetTokenT< edm::DetSetVector<PixelDigi> > pixDigiToken_;
  const edm::EDGetTokenT< edm::DetSetVector<Phase2TrackerDigi> > otDigiToken_;
};
#endif
