#ifndef Validation_RPCRecHits_RPCRecHitValid_h
#define Validaiton_RPCRecHits_RPCRecHitValid_h

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"

#include "Validation/RPCRecHits/interface/RPCValidHistograms.h"

#include <string>

class RPCRecHitValid : public edm::EDAnalyzer
{
public:
  RPCRecHitValid(const edm::ParameterSet& pset);
  ~RPCRecHitValid();

  void analyze(const edm::Event& event, const edm::EventSetup& eventSetup);
  void beginRun(const edm::Run& run, const edm::EventSetup& eventSetup);
  void endRun(const edm::Run& run, const edm::EventSetup& eventSetup);
  void beginJob();
  void endJob();

private:
  std::string subDir_;
  edm::InputTag simHitLabel_, recHitLabel_;
  edm::InputTag simParticleLabel_;
  edm::InputTag simHitAssocLabel_;
  edm::InputTag muonLabel_;

  DQMStore* dbe_;

  typedef MonitorElement* MEP;
  RPCValidHistograms h_;

  MEP h_eventCount;

  MEP h_nRPCHitPerSimMuon, h_nRPCHitPerSimMuonBarrel, h_nRPCHitPerSimMuonOverlap, h_nRPCHitPerSimMuonEndcap;
  MEP h_nRPCHitPerRecoMuon, h_nRPCHitPerRecoMuonBarrel, h_nRPCHitPerRecoMuonOverlap, h_nRPCHitPerRecoMuonEndcap;
  MEP h_simMuonBarrel_pt, h_simMuonOverlap_pt, h_simMuonEndcap_pt, h_simMuonNoRPC_pt;
  MEP h_simMuonBarrel_eta, h_simMuonOverlap_eta, h_simMuonEndcap_eta, h_simMuonNoRPC_eta;
  MEP h_simMuonBarrel_phi, h_simMuonOverlap_phi, h_simMuonEndcap_phi, h_simMuonNoRPC_phi;
  MEP h_recoMuonBarrel_pt, h_recoMuonOverlap_pt, h_recoMuonEndcap_pt, h_recoMuonNoRPC_pt;
  MEP h_recoMuonBarrel_eta, h_recoMuonOverlap_eta, h_recoMuonEndcap_eta, h_recoMuonNoRPC_eta;
  MEP h_recoMuonBarrel_phi, h_recoMuonOverlap_phi, h_recoMuonEndcap_phi, h_recoMuonNoRPC_phi;
  MEP h_simParticleType, h_simParticleTypeBarrel, h_simParticleTypeEndcap;

  MEP h_refPunchOccupancyBarrel_wheel, h_refPunchOccupancyEndcap_disk, h_refPunchOccupancyBarrel_station;
  MEP h_refPunchOccupancyBarrel_wheel_station, h_refPunchOccupancyEndcap_disk_ring;
  MEP h_recPunchOccupancyBarrel_wheel, h_recPunchOccupancyEndcap_disk, h_recPunchOccupancyBarrel_station;
  MEP h_recPunchOccupancyBarrel_wheel_station, h_recPunchOccupancyEndcap_disk_ring;

  MEP h_matchOccupancyBarrel_detId;
  MEP h_matchOccupancyEndcap_detId;
  MEP h_refOccupancyBarrel_detId;
  MEP h_refOccupancyEndcap_detId;
  MEP h_noiseOccupancyBarrel_detId;
  MEP h_noiseOccupancyEndcap_detId;
  MEP h_rollAreaBarrel_detId;
  MEP h_rollAreaEndcap_detId;

  std::map<int, int> detIdToIndexMapBarrel_, detIdToIndexMapEndcap_;
};

#endif
