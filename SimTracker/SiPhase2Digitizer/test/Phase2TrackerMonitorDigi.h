#ifndef Phase2TrackerMonitorDigi_h
#define Phase2TrackerMonitorDigi_h

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/Common/interface/DetSetVector.h"
#include "DQMServices/Core/interface/DQMEDAnalyzer.h"

class MonitorElement;
class PixelDigi;
class Phase2TrackerDigi;
class TrackerTopology;

class Phase2TrackerMonitorDigi : public DQMEDAnalyzer{

public:
  
  explicit Phase2TrackerMonitorDigi(const edm::ParameterSet&);
  ~Phase2TrackerMonitorDigi();
  void bookHistograms(DQMStore::IBooker & ibooker,
		      edm::Run const &  iRun ,
		      edm::EventSetup const &  iSetup );
  void dqmBeginRun(const edm::Run& iRun, const edm::EventSetup& iSetup); 
  virtual void analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup);
  virtual void endLuminosityBlock(edm::LuminosityBlock const& lumiBlock, edm::EventSetup const& iSetup);
  
  
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
  void bookLayerHistos(DQMStore::IBooker & ibooker, unsigned int det_id, const TrackerTopology* tTopo); 
  
  edm::ParameterSet config_;
  std::map<unsigned int, DigiMEs> layerMEs;
  edm::InputTag pixDigiSrc_;
  edm::InputTag otDigiSrc_;
  const edm::EDGetTokenT< edm::DetSetVector<PixelDigi> > pixDigiToken_;
  const edm::EDGetTokenT< edm::DetSetVector<Phase2TrackerDigi> > otDigiToken_;
};
#endif
