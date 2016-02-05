#ifndef Phase2TrackerValidateDigi_h
#define Phase2TrackerValidateDigi_h

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"

#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "SimDataFormats/Track/interface/SimTrackContainer.h"
#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"
#include "SimDataFormats/Vertex/interface/SimVertexContainer.h"

// DQM Histograming
class DQMStore;
class MonitorElement;
class PixelDigiSimLink;
class SimTrack;
class TrackerTopology;
class PixelDigi;
class Phase2TrackerDigi;

 
class Phase2TrackerValidateDigi : public edm::one::EDAnalyzer<> {

public:

  explicit Phase2TrackerValidateDigi(const edm::ParameterSet&);
  ~Phase2TrackerValidateDigi();
  virtual void beginJob();
  virtual void beginRun(const edm::Run& iRun, const edm::EventSetup& iSetup);
  virtual void analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup);

  virtual void endJob(); 

  
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
  };

private:

  MonitorElement* SimulatedTrackPt;  
  MonitorElement* SimulatedTrackEta;  
  MonitorElement* SimulatedTrackPhi;  
  MonitorElement* SimulatedTrackPtP;  
  MonitorElement* SimulatedTrackEtaP;  
  MonitorElement* SimulatedTrackPhiP;  
  MonitorElement* SimulatedTrackPtS;  
  MonitorElement* SimulatedTrackEtaS;  
  MonitorElement* SimulatedTrackPhiS;  

  float etaCut_;
  float ptCut_;

  void bookHistos();
  void bookLayerHistos(unsigned int ilayer); 
  unsigned int getSimTrackId(edm::Handle<edm::DetSetVector<PixelDigiSimLink> >&, const DetId& detId, unsigned int& channel);
  int matchedSimTrack(edm::Handle<edm::SimTrackContainer>& SimTk, unsigned int simTrkId);
  int isPrimary(const SimTrack& simTrk, edm::Handle<edm::PSimHitContainer>& simHits);

  void fillHistogram(MonitorElement* th1, MonitorElement* th2, MonitorElement* th3, float val, int primary);

  DQMStore* dqmStore_;
  edm::ParameterSet config_;
  std::map<unsigned int, DigiMEs> layerMEs;
  edm::InputTag pixDigiSrc_; 
  edm::InputTag otDigiSrc_; 
  edm::InputTag digiSimLinkSrc_; 
  edm::InputTag pSimHitSrc_;
  edm::InputTag simTrackSrc_;
  edm::InputTag simVertexSrc_;

  const edm::EDGetTokenT< edm::DetSetVector<PixelDigi> > pixDigiToken_;
  const edm::EDGetTokenT< edm::DetSetVector<Phase2TrackerDigi> > otDigiToken_;
  const edm::EDGetTokenT< edm::DetSetVector<PixelDigiSimLink> > digiSimLinkToken_;
  const edm::EDGetTokenT< edm::PSimHitContainer > psimHitToken_;
  const edm::EDGetTokenT< edm::SimTrackContainer > simTrackToken_;
  const edm::EDGetTokenT< edm::SimVertexContainer > simVertexToken_;
};
#endif
