#ifndef Phase2TrackerValidateDigi_h
#define Phase2TrackerValidateDigi_h

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"

#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "SimDataFormats/Track/interface/SimTrackContainer.h"
#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"
#include "SimDataFormats/Vertex/interface/SimVertexContainer.h"
#include "DQMServices/Core/interface/DQMEDAnalyzer.h"

// DQM Histograming
class MonitorElement;
class PixelDigiSimLink;
class SimTrack;
class TrackerTopology;
class PixelDigi;
class Phase2TrackerDigi;
class TrackerGeometry;
 
class Phase2TrackerValidateDigi : public DQMEDAnalyzer{

public:

  explicit Phase2TrackerValidateDigi(const edm::ParameterSet&);
  ~Phase2TrackerValidateDigi();
  void bookHistograms(DQMStore::IBooker & ibooker, edm::Run const &  iRun ,
		      edm::EventSetup const &  iSetup );
  void dqmBeginRun(const edm::Run& iRun, const edm::EventSetup& iSetup);
  virtual void analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup);
  

  struct DigiMEs{
    MonitorElement* SimTrackPt;  
    MonitorElement* SimTrackEta;  
    MonitorElement* SimTrackPhi;  
    MonitorElement* SimTrackPtP;  
    MonitorElement* SimTrackEtaP;  
    MonitorElement* SimTrackPhiP;  
    MonitorElement* SimTrackPtS;  
    MonitorElement* SimTrackEtaS;  
    MonitorElement* SimTrackPhiS;  
    MonitorElement* MatchedTrackPt;
    MonitorElement* MatchedTrackPhi;
    MonitorElement* MatchedTrackEta;
    MonitorElement* MatchedTrackPtP;
    MonitorElement* MatchedTrackPhiP;
    MonitorElement* MatchedTrackEtaP;
    MonitorElement* MatchedTrackPtS;
    MonitorElement* MatchedTrackPhiS;
    MonitorElement* MatchedTrackEtaS;
    MonitorElement* SimHitElossP;  
    MonitorElement* SimHitElossS;  
    MonitorElement* SimHitDx;
    MonitorElement* SimHitDy;
    MonitorElement* SimHitDz;
    MonitorElement* BunchXTimeBin;
    MonitorElement* FractionOfOOTDigis;
  };

private:

  MonitorElement* nSimulatedTracks;  
  MonitorElement* nSimulatedTracksP;  
  MonitorElement* nSimulatedTracksS;  
  MonitorElement* SimulatedTrackPt;  
  MonitorElement* SimulatedTrackEta;  
  MonitorElement* SimulatedTrackPhi;  
  MonitorElement* SimulatedTrackPtP;  
  MonitorElement* SimulatedTrackEtaP;  
  MonitorElement* SimulatedTrackPhiP;  
  MonitorElement* SimulatedTrackPtS;  
  MonitorElement* SimulatedTrackEtaS;  
  MonitorElement* SimulatedTrackPhiS;  
 
  MonitorElement* SimulatedXYPositionMap;
  MonitorElement* SimulatedRZPositionMap;
   
  MonitorElement* MatchedXYPositionMap;
  MonitorElement* MatchedRZPositionMap;

  MonitorElement* SimulatedTOFEtaMap;
  MonitorElement* SimulatedTOFPhiMap;
  MonitorElement* SimulatedTOFRMap;
  MonitorElement* SimulatedTOFZMap;

  float etaCut_;
  float ptCut_;

  void bookLayerHistos(DQMStore::IBooker & ibooker, unsigned int det_id, const TrackerTopology* tTopo, bool flag); 
  unsigned int getSimTrackId(const edm::DetSetVector<PixelDigiSimLink>* simLinks, const DetId& detId, unsigned int& channel);
  int matchedSimTrack(edm::Handle<edm::SimTrackContainer>& SimTk, unsigned int simTrkId);
  int isPrimary(const SimTrack& simTrk, edm::Handle<edm::PSimHitContainer>& simHitHandle);

  void fillHistogram(MonitorElement* th1, MonitorElement* th2, MonitorElement* th3, float val, int primary);
  int fillSimHitInfo(const edm::Event& iEvent, unsigned int id, float pt, float eta, float phi, int type, const edm::ESHandle<TrackerGeometry> gHandle);
  bool findOTDigi(unsigned int detid, unsigned int id);
  bool findITPixelDigi(unsigned int detid, unsigned int id);
  void fillOTBXInfo();
  void fillITPixelBXInfo();

  edm::ParameterSet config_;
  std::map<unsigned int, DigiMEs> layerMEs;

  bool pixelFlag_;
  std::string geomType_;

  edm::InputTag otDigiSrc_; 
  edm::InputTag otDigiSimLinkSrc_; 
  edm::InputTag itPixelDigiSrc_; 
  edm::InputTag itPixelDigiSimLinkSrc_; 
  std::vector<edm::InputTag> pSimHitSrc_;
  edm::InputTag simTrackSrc_;
  edm::InputTag simVertexSrc_;

  const edm::EDGetTokenT< edm::DetSetVector<Phase2TrackerDigi> > otDigiToken_;
  const edm::EDGetTokenT< edm::DetSetVector<PixelDigiSimLink> > otDigiSimLinkToken_;
  const edm::EDGetTokenT< edm::DetSetVector<PixelDigi> > itPixelDigiToken_;
  const edm::EDGetTokenT< edm::DetSetVector<PixelDigiSimLink> > itPixelDigiSimLinkToken_;
  //  const edm::EDGetTokenT< edm::PSimHitContainer > psimHitToken_;
  const edm::EDGetTokenT< edm::SimTrackContainer > simTrackToken_;
  const edm::EDGetTokenT< edm::SimVertexContainer > simVertexToken_;
  std::vector< edm::EDGetTokenT< edm::PSimHitContainer > > simHitTokens_;

  edm::Handle< edm::DetSetVector<PixelDigi> > itPixelDigiHandle_;
  edm::Handle< edm::DetSetVector<Phase2TrackerDigi> > otDigiHandle_;
  edm::Handle< edm::DetSetVector<PixelDigiSimLink> > itPixelSimLinkHandle_;
  edm::Handle< edm::DetSetVector<PixelDigiSimLink> > otSimLinkHandle_;
  edm::Handle<edm::PSimHitContainer> simHits;
  edm::Handle<edm::SimTrackContainer> simTracks;
  edm::Handle<edm::SimVertexContainer> simVertices;
  edm::ESHandle<TrackerTopology> tTopoHandle_;

  const float GeVperElectron; // 3.7E-09 
};
#endif
