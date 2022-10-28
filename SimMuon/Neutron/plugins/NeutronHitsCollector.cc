// -*- C++ -*-
//
// Package:    SimMuon/Neutron
// Class:      NeutronHitsCollector
//
/**\class NeutronHitsCollector NeutronHitsCollector.cc SimMuon/Neutron/plugins/NeutronHitsCollector.cc

 Description:
 Utility for neutron SimHits produced by CSC/RPC/DTNeutronWriter modules.
   * Re-packs neutron simhits in muon detectors into new collections that have a single module label.
   * Creates a bunch of empty collections with the same module label to make MixingModule happy


*/
//
// Original Author:  Vadim Khotilovich
//         Created:  Mon Aug 09 19:11:42 CST 2010
//
//

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"
#include "SimDataFormats/CaloHit/interface/PCaloHitContainer.h"

#include "SimDataFormats/Track/interface/SimTrackContainer.h"
#include "SimDataFormats/Vertex/interface/SimVertexContainer.h"

class NeutronHitsCollector : public edm::stream::EDProducer<> {
public:
  explicit NeutronHitsCollector(const edm::ParameterSet&);
  ~NeutronHitsCollector() override = default;

private:
  virtual void beginJob();
  void produce(edm::Event&, const edm::EventSetup&) override;
  virtual void endJob();

  const std::string neutron_label_csc;
  const std::string neutron_label_dt;
  const std::string neutron_label_rpc;
  const edm::EDGetTokenT<edm::PSimHitContainer> tokenCSC_;
  const edm::EDGetTokenT<edm::PSimHitContainer> tokenDT_;
  const edm::EDGetTokenT<edm::PSimHitContainer> tokenRPC_;
};

NeutronHitsCollector::NeutronHitsCollector(const edm::ParameterSet& iConfig)
    : neutron_label_csc(iConfig.getUntrackedParameter<std::string>("neutronLabelCSC", "")),
      neutron_label_dt(iConfig.getUntrackedParameter<std::string>("neutronLabelDT", "")),
      neutron_label_rpc(iConfig.getUntrackedParameter<std::string>("neutronLabelRPC", "")),
      tokenCSC_(consumes<edm::PSimHitContainer>(neutron_label_csc)),
      tokenDT_(consumes<edm::PSimHitContainer>(neutron_label_dt)),
      tokenRPC_(consumes<edm::PSimHitContainer>(neutron_label_rpc)) {
  // The following list duplicates
  // http://cmslxr.fnal.gov/lxr/source/SimG4Core/Application/plugins/OscarProducer.cc

  produces<edm::PSimHitContainer>("MuonDTHits");
  produces<edm::PSimHitContainer>("MuonCSCHits");
  produces<edm::PSimHitContainer>("MuonRPCHits");

  produces<edm::SimTrackContainer>().setBranchAlias("SimTracks");
  produces<edm::SimVertexContainer>().setBranchAlias("SimVertices");

  produces<edm::PSimHitContainer>("TrackerHitsPixelBarrelLowTof");
  produces<edm::PSimHitContainer>("TrackerHitsPixelBarrelHighTof");
  produces<edm::PSimHitContainer>("TrackerHitsTIBLowTof");
  produces<edm::PSimHitContainer>("TrackerHitsTIBHighTof");
  produces<edm::PSimHitContainer>("TrackerHitsTIDLowTof");
  produces<edm::PSimHitContainer>("TrackerHitsTIDHighTof");
  produces<edm::PSimHitContainer>("TrackerHitsPixelEndcapLowTof");
  produces<edm::PSimHitContainer>("TrackerHitsPixelEndcapHighTof");
  produces<edm::PSimHitContainer>("TrackerHitsTOBLowTof");
  produces<edm::PSimHitContainer>("TrackerHitsTOBHighTof");
  produces<edm::PSimHitContainer>("TrackerHitsTECLowTof");
  produces<edm::PSimHitContainer>("TrackerHitsTECHighTof");

  produces<edm::PSimHitContainer>("TotemHitsT1");
  produces<edm::PSimHitContainer>("TotemHitsT2Gem");
  produces<edm::PSimHitContainer>("TotemHitsRP");
  produces<edm::PSimHitContainer>("FP420SI");
  produces<edm::PSimHitContainer>("BSCHits");

  produces<edm::PCaloHitContainer>("EcalHitsEB");
  produces<edm::PCaloHitContainer>("EcalHitsEE");
  produces<edm::PCaloHitContainer>("EcalHitsES");
  produces<edm::PCaloHitContainer>("HcalHits");
  produces<edm::PCaloHitContainer>("CaloHitsTk");
  produces<edm::PCaloHitContainer>("CastorPL");
  produces<edm::PCaloHitContainer>("CastorFI");
  produces<edm::PCaloHitContainer>("CastorBU");
  produces<edm::PCaloHitContainer>("CastorTU");
  produces<edm::PCaloHitContainer>("EcalTBH4BeamHits");
  produces<edm::PCaloHitContainer>("HcalTB06BeamHits");
  produces<edm::PCaloHitContainer>("ZDCHITS");
  //produces<edm::PCaloHitContainer>("ChamberHits");
  //produces<edm::PCaloHitContainer>("FibreHits");
  //produces<edm::PCaloHitContainer>("WedgeHits");
}

void NeutronHitsCollector::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  edm::PSimHitContainer::const_iterator hit;

  // ----- MuonCSCHits -----
  //
  std::unique_ptr<edm::PSimHitContainer> simCSC(new edm::PSimHitContainer);
  if (neutron_label_csc.length() > 0) {
    const edm::Handle<edm::PSimHitContainer>& MuonCSCHits = iEvent.getHandle(tokenCSC_);
    for (hit = MuonCSCHits->begin(); hit != MuonCSCHits->end(); ++hit)
      simCSC->push_back(*hit);
  }
  iEvent.put(std::move(simCSC), "MuonCSCHits");

  // ----- MuonDTHits -----
  //
  std::unique_ptr<edm::PSimHitContainer> simDT(new edm::PSimHitContainer);
  if (neutron_label_dt.length() > 0) {
    const edm::Handle<edm::PSimHitContainer>& MuonDTHits = iEvent.getHandle(tokenDT_);
    for (hit = MuonDTHits->begin(); hit != MuonDTHits->end(); ++hit)
      simDT->push_back(*hit);
  }
  iEvent.put(std::move(simDT), "MuonDTHits");

  // ----- MuonRPCHits -----
  //
  std::unique_ptr<edm::PSimHitContainer> simRPC(new edm::PSimHitContainer);
  if (neutron_label_rpc.length() > 0) {
    const edm::Handle<edm::PSimHitContainer>& MuonRPCHits = iEvent.getHandle(tokenRPC_);
    for (hit = MuonRPCHits->begin(); hit != MuonRPCHits->end(); ++hit)
      simRPC->push_back(*hit);
  }
  iEvent.put(std::move(simRPC), "MuonRPCHits");

  // Further, produce a bunch of empty collections

  // ----- calorimetry -----
  //
  std::unique_ptr<edm::PCaloHitContainer> calout1(new edm::PCaloHitContainer);
  iEvent.put(std::move(calout1), "EcalHitsEB");
  std::unique_ptr<edm::PCaloHitContainer> calout2(new edm::PCaloHitContainer);
  iEvent.put(std::move(calout2), "EcalHitsEE");
  std::unique_ptr<edm::PCaloHitContainer> calout3(new edm::PCaloHitContainer);
  iEvent.put(std::move(calout3), "EcalHitsES");
  std::unique_ptr<edm::PCaloHitContainer> calout4(new edm::PCaloHitContainer);
  iEvent.put(std::move(calout4), "HcalHits");
  std::unique_ptr<edm::PCaloHitContainer> calout5(new edm::PCaloHitContainer);
  iEvent.put(std::move(calout5), "CaloHitsTk");
  std::unique_ptr<edm::PCaloHitContainer> calout6(new edm::PCaloHitContainer);
  iEvent.put(std::move(calout6), "CastorPL");
  std::unique_ptr<edm::PCaloHitContainer> calout7(new edm::PCaloHitContainer);
  iEvent.put(std::move(calout7), "CastorFI");
  std::unique_ptr<edm::PCaloHitContainer> calout8(new edm::PCaloHitContainer);
  iEvent.put(std::move(calout8), "CastorBU");
  std::unique_ptr<edm::PCaloHitContainer> calout9(new edm::PCaloHitContainer);
  iEvent.put(std::move(calout9), "CastorTU");
  std::unique_ptr<edm::PCaloHitContainer> calout10(new edm::PCaloHitContainer);
  iEvent.put(std::move(calout10), "EcalTBH4BeamHits");
  std::unique_ptr<edm::PCaloHitContainer> calout11(new edm::PCaloHitContainer);
  iEvent.put(std::move(calout11), "HcalTB06BeamHits");
  std::unique_ptr<edm::PCaloHitContainer> calout12(new edm::PCaloHitContainer);
  iEvent.put(std::move(calout12), "ZDCHITS");
  //std::unique_ptr<edm::PCaloHitContainer> calout13(new edm::PCaloHitContainer);
  //iEvent.put(std::move(calout13), "ChamberHits");
  //std::unique_ptr<edm::PCaloHitContainer> calout14(new edm::PCaloHitContainer);
  //iEvent.put(std::move(calout14), "FibreHits");
  //std::unique_ptr<edm::PCaloHitContainer> calout15(new edm::PCaloHitContainer);
  //iEvent.put(std::move(calout15), "WedgeHits");

  // ----- Tracker -----
  //
  std::unique_ptr<edm::PSimHitContainer> trout1(new edm::PSimHitContainer);
  iEvent.put(std::move(trout1), "TrackerHitsPixelBarrelLowTof");
  std::unique_ptr<edm::PSimHitContainer> trout2(new edm::PSimHitContainer);
  iEvent.put(std::move(trout2), "TrackerHitsPixelBarrelHighTof");
  std::unique_ptr<edm::PSimHitContainer> trout3(new edm::PSimHitContainer);
  iEvent.put(std::move(trout3), "TrackerHitsTIBLowTof");
  std::unique_ptr<edm::PSimHitContainer> trout4(new edm::PSimHitContainer);
  iEvent.put(std::move(trout4), "TrackerHitsTIBHighTof");
  std::unique_ptr<edm::PSimHitContainer> trout5(new edm::PSimHitContainer);
  iEvent.put(std::move(trout5), "TrackerHitsTIDLowTof");
  std::unique_ptr<edm::PSimHitContainer> trout6(new edm::PSimHitContainer);
  iEvent.put(std::move(trout6), "TrackerHitsTIDHighTof");
  std::unique_ptr<edm::PSimHitContainer> trout7(new edm::PSimHitContainer);
  iEvent.put(std::move(trout7), "TrackerHitsPixelEndcapLowTof");
  std::unique_ptr<edm::PSimHitContainer> trout8(new edm::PSimHitContainer);
  iEvent.put(std::move(trout8), "TrackerHitsPixelEndcapHighTof");
  std::unique_ptr<edm::PSimHitContainer> trout9(new edm::PSimHitContainer);
  iEvent.put(std::move(trout9), "TrackerHitsTOBLowTof");
  std::unique_ptr<edm::PSimHitContainer> trout10(new edm::PSimHitContainer);
  iEvent.put(std::move(trout10), "TrackerHitsTOBHighTof");
  std::unique_ptr<edm::PSimHitContainer> trout11(new edm::PSimHitContainer);
  iEvent.put(std::move(trout11), "TrackerHitsTECLowTof");
  std::unique_ptr<edm::PSimHitContainer> trout12(new edm::PSimHitContainer);
  iEvent.put(std::move(trout12), "TrackerHitsTECHighTof");

  // ----- Forward stuff -----
  //
  std::unique_ptr<edm::PSimHitContainer> fwout1(new edm::PSimHitContainer);
  iEvent.put(std::move(fwout1), "TotemHitsT1");
  std::unique_ptr<edm::PSimHitContainer> fwout2(new edm::PSimHitContainer);
  iEvent.put(std::move(fwout2), "TotemHitsT2Gem");
  std::unique_ptr<edm::PSimHitContainer> fwout3(new edm::PSimHitContainer);
  iEvent.put(std::move(fwout3), "TotemHitsRP");
  std::unique_ptr<edm::PSimHitContainer> fwout4(new edm::PSimHitContainer);
  iEvent.put(std::move(fwout4), "FP420SI");
  std::unique_ptr<edm::PSimHitContainer> fwout5(new edm::PSimHitContainer);
  iEvent.put(std::move(fwout5), "BSCHits");

  // ----- SimTracks & SimVertices -----
  //
  std::unique_ptr<edm::SimTrackContainer> simTr(new edm::SimTrackContainer);
  iEvent.put(std::move(simTr));
  std::unique_ptr<edm::SimVertexContainer> simVe(new edm::SimVertexContainer);
  iEvent.put(std::move(simVe));
}

void NeutronHitsCollector::beginJob() {}

void NeutronHitsCollector::endJob() {}

//define this as a plug-in
DEFINE_FWK_MODULE(NeutronHitsCollector);
