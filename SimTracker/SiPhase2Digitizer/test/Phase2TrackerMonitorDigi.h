#ifndef Phase2TrackerMonitorDigi_h
#define Phase2TrackerMonitorDigi_h

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/Common/interface/DetSetVector.h"
#include "DQMServices/Core/interface/DQMEDAnalyzer.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"

class MonitorElement;
class PixelDigi;
class Phase2TrackerDigi;
class TrackerGeometry;

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
    MonitorElement* DigiOccupancyP;
    MonitorElement* DigiOccupancyS;
    MonitorElement* PositionOfDigis;
    MonitorElement* ChargeOfDigis;
    MonitorElement* TotalNumberOfDigis;
    MonitorElement* NumberOfHitDetectors;
    MonitorElement* NumberOfClusters;
    MonitorElement* ClusterWidth;
    MonitorElement* ClusterPosition;
    MonitorElement* FractionOfOvTBits;
    MonitorElement* FractionOfOvTBitsVsEta;
    MonitorElement* EtaOccupancyProfP;
    MonitorElement* EtaOccupancyProfS;
    unsigned int nDigiPerLayer; 
    unsigned int nHitDetsPerLayer; 
  };

  MonitorElement* XYPositionMap;
  MonitorElement* RZPositionMap;
  MonitorElement* XYOccupancyMap;
  MonitorElement* RZOccupancyMap;

private:
  void bookLayerHistos(DQMStore::IBooker & ibooker, unsigned int det_id, const TrackerTopology* tTopo); 
  void fillITPixelDigiHistos(const edm::Handle<edm::DetSetVector<PixelDigi>>  handle, const edm::ESHandle<TrackerGeometry> gHandle);
  void fillOTDigiHistos(const edm::Handle<edm::DetSetVector<Phase2TrackerDigi>>  handle, const edm::ESHandle<TrackerGeometry> gHandle);

  edm::ParameterSet config_;
  std::map<unsigned int, DigiMEs> layerMEs;
  bool pixelFlag_;
  std::string geomType_; 
  edm::InputTag otDigiSrc_; 
  edm::InputTag itPixelDigiSrc_; 
  const edm::EDGetTokenT< edm::DetSetVector<Phase2TrackerDigi> > otDigiToken_;
  const edm::EDGetTokenT< edm::DetSetVector<PixelDigi> > itPixelDigiToken_;
  edm::ESHandle<TrackerTopology> tTopoHandle_;

};
#endif
