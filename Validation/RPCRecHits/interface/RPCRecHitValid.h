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
  void beginJob();
  void endJob();

private:
  edm::InputTag simHitLabel_, recHitLabel_;
  edm::InputTag simTrackLabel_;

  DQMStore* dbe_;
  std::string rootFileName_;
  bool isStandAloneMode_;

  typedef MonitorElement* MEP;
  RPCValidHistograms h_;
  MEP h_nRPCHitPerSimMuon, h_nRPCHitPerSimMuonBarrel, h_nRPCHitPerSimMuonOverlap, h_nRPCHitPerSimMuonEndcap;
  MEP h_simMuonBarrel_pt, h_simMuonOverlap_pt, h_simMuonEndcap_pt, h_simMuonNoRPC_pt;
  MEP h_simMuonBarrel_eta, h_simMuonOverlap_eta, h_simMuonEndcap_eta, h_simMuonNoRPC_eta;
  MEP h_simHitPType, h_simHitPTypeBarrel, h_simHitPTypeEndcap;
  MEP h_refBkgBarrelOccupancy_wheel, h_refBkgEndcapOccupancy_disk, h_refBkgBarrelOccupancy_station;
  MEP h_refBkgBarrelOccupancy_wheel_station, h_refBkgEndcapOccupancy_disk_ring;
  MEP h_refPunchBarrelOccupancy_wheel, h_refPunchEndcapOccupancy_disk, h_refPunchBarrelOccupancy_station;
  MEP h_refPunchBarrelOccupancy_wheel_station, h_refPunchEndcapOccupancy_disk_ring;
  MEP h_recPunchBarrelOccupancy_wheel, h_recPunchEndcapOccupancy_disk, h_recPunchBarrelOccupancy_station;
  MEP h_recPunchBarrelOccupancy_wheel_station, h_recPunchEndcapOccupancy_disk_ring;
  MEP h_noiseBarrelOccupancy_wheel, h_noiseEndcapOccupancy_disk, h_noiseBarrelOccupancy_station;
  MEP h_noiseBarrelOccupancy_wheel_station, h_noiseEndcapOccupancy_disk_ring;
  
};

#endif
