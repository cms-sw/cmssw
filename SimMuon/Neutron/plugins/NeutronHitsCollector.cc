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
// $Id: NeutronHitsCollector.cc,v 1.1 2010/08/20 00:28:13 khotilov Exp $
//
//

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"
#include "SimDataFormats/CaloHit/interface/PCaloHitContainer.h"

#include "SimDataFormats/Track/interface/SimTrackContainer.h"
#include "SimDataFormats/Vertex/interface/SimVertexContainer.h"



class NeutronHitsCollector : public edm::EDProducer
{
public:
  explicit NeutronHitsCollector(const edm::ParameterSet&);
  ~NeutronHitsCollector() {};

private:
  virtual void beginJob();
  virtual void produce(edm::Event&, const edm::EventSetup&);
  virtual void endJob();

  std::string neutron_label_csc;
  std::string neutron_label_dt;
  std::string neutron_label_rpc;
};



NeutronHitsCollector::NeutronHitsCollector(const edm::ParameterSet& iConfig)
{
  neutron_label_csc = iConfig.getUntrackedParameter<std::string>("neutronLabelCSC","");
  neutron_label_dt  = iConfig.getUntrackedParameter<std::string>("neutronLabelDT","");
  neutron_label_rpc = iConfig.getUntrackedParameter<std::string>("neutronLabelRPC","");

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


void NeutronHitsCollector::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  edm::PSimHitContainer::const_iterator hit;

  // ----- MuonCSCHits -----
  //
  std::auto_ptr<edm::PSimHitContainer> simCSC(new edm::PSimHitContainer);
  if (neutron_label_csc.length()>0)
  {
    edm::Handle<edm::PSimHitContainer> MuonCSCHits;
    iEvent.getByLabel(neutron_label_csc,MuonCSCHits);
    for (hit = MuonCSCHits->begin();  hit != MuonCSCHits->end();  ++hit) simCSC->push_back(*hit);
  }
  iEvent.put(simCSC, "MuonCSCHits");

  // ----- MuonDTHits -----
  //
  std::auto_ptr<edm::PSimHitContainer> simDT(new edm::PSimHitContainer);
  if (neutron_label_dt.length()>0)
  {
    edm::Handle<edm::PSimHitContainer> MuonDTHits;
    iEvent.getByLabel(neutron_label_dt,MuonDTHits);
    for (hit = MuonDTHits->begin();  hit != MuonDTHits->end();  ++hit) simDT->push_back(*hit);
  }
  iEvent.put(simDT, "MuonDTHits");

  // ----- MuonRPCHits -----
  //
  std::auto_ptr<edm::PSimHitContainer> simRPC(new edm::PSimHitContainer);
  if (neutron_label_rpc.length()>0)
  {
    edm::Handle<edm::PSimHitContainer> MuonRPCHits;
    iEvent.getByLabel(neutron_label_rpc,MuonRPCHits);
    for (hit = MuonRPCHits->begin();  hit != MuonRPCHits->end();  ++hit) simRPC->push_back(*hit);
  }
  iEvent.put(simRPC, "MuonRPCHits");

  // Further, produce a bunch of empty collections

  // ----- calorimetry -----
  //
  std::auto_ptr<edm::PCaloHitContainer> calout1(new edm::PCaloHitContainer);
  iEvent.put(calout1, "EcalHitsEB");
  std::auto_ptr<edm::PCaloHitContainer> calout2(new edm::PCaloHitContainer);
  iEvent.put(calout2, "EcalHitsEE");
  std::auto_ptr<edm::PCaloHitContainer> calout3(new edm::PCaloHitContainer);
  iEvent.put(calout3, "EcalHitsES");
  std::auto_ptr<edm::PCaloHitContainer> calout4(new edm::PCaloHitContainer);
  iEvent.put(calout4, "HcalHits");
  std::auto_ptr<edm::PCaloHitContainer> calout5(new edm::PCaloHitContainer);
  iEvent.put(calout5, "CaloHitsTk");
  std::auto_ptr<edm::PCaloHitContainer> calout6(new edm::PCaloHitContainer);
  iEvent.put(calout6, "CastorPL");
  std::auto_ptr<edm::PCaloHitContainer> calout7(new edm::PCaloHitContainer);
  iEvent.put(calout7, "CastorFI");
  std::auto_ptr<edm::PCaloHitContainer> calout8(new edm::PCaloHitContainer);
  iEvent.put(calout8, "CastorBU");
  std::auto_ptr<edm::PCaloHitContainer> calout9(new edm::PCaloHitContainer);
  iEvent.put(calout9, "CastorTU");
  std::auto_ptr<edm::PCaloHitContainer> calout10(new edm::PCaloHitContainer);
  iEvent.put(calout10, "EcalTBH4BeamHits");
  std::auto_ptr<edm::PCaloHitContainer> calout11(new edm::PCaloHitContainer);
  iEvent.put(calout11, "HcalTB06BeamHits");
  std::auto_ptr<edm::PCaloHitContainer> calout12(new edm::PCaloHitContainer);
  iEvent.put(calout12, "ZDCHITS");
  //std::auto_ptr<edm::PCaloHitContainer> calout13(new edm::PCaloHitContainer);
  //iEvent.put(calout13, "ChamberHits");
  //std::auto_ptr<edm::PCaloHitContainer> calout14(new edm::PCaloHitContainer);
  //iEvent.put(calout14, "FibreHits");
  //std::auto_ptr<edm::PCaloHitContainer> calout15(new edm::PCaloHitContainer);
  //iEvent.put(calout15, "WedgeHits");

  // ----- Tracker -----
  //
  std::auto_ptr<edm::PSimHitContainer> trout1(new edm::PSimHitContainer);
  iEvent.put(trout1, "TrackerHitsPixelBarrelLowTof");
  std::auto_ptr<edm::PSimHitContainer> trout2(new edm::PSimHitContainer);
  iEvent.put(trout2, "TrackerHitsPixelBarrelHighTof");
  std::auto_ptr<edm::PSimHitContainer> trout3(new edm::PSimHitContainer);
  iEvent.put(trout3, "TrackerHitsTIBLowTof");
  std::auto_ptr<edm::PSimHitContainer> trout4(new edm::PSimHitContainer);
  iEvent.put(trout4, "TrackerHitsTIBHighTof");
  std::auto_ptr<edm::PSimHitContainer> trout5(new edm::PSimHitContainer);
  iEvent.put(trout5, "TrackerHitsTIDLowTof");
  std::auto_ptr<edm::PSimHitContainer> trout6(new edm::PSimHitContainer);
  iEvent.put(trout6, "TrackerHitsTIDHighTof");
  std::auto_ptr<edm::PSimHitContainer> trout7(new edm::PSimHitContainer);
  iEvent.put(trout7, "TrackerHitsPixelEndcapLowTof");
  std::auto_ptr<edm::PSimHitContainer> trout8(new edm::PSimHitContainer);
  iEvent.put(trout8, "TrackerHitsPixelEndcapHighTof");
  std::auto_ptr<edm::PSimHitContainer> trout9(new edm::PSimHitContainer);
  iEvent.put(trout9, "TrackerHitsTOBLowTof");
  std::auto_ptr<edm::PSimHitContainer> trout10(new edm::PSimHitContainer);
  iEvent.put(trout10, "TrackerHitsTOBHighTof");
  std::auto_ptr<edm::PSimHitContainer> trout11(new edm::PSimHitContainer);
  iEvent.put(trout11, "TrackerHitsTECLowTof");
  std::auto_ptr<edm::PSimHitContainer> trout12(new edm::PSimHitContainer);
  iEvent.put(trout12, "TrackerHitsTECHighTof");

  // ----- Forward stuff -----
  //
  std::auto_ptr<edm::PSimHitContainer> fwout1(new edm::PSimHitContainer);
  iEvent.put(fwout1, "TotemHitsT1");
  std::auto_ptr<edm::PSimHitContainer> fwout2(new edm::PSimHitContainer);
  iEvent.put(fwout2, "TotemHitsT2Gem");
  std::auto_ptr<edm::PSimHitContainer> fwout3(new edm::PSimHitContainer);
  iEvent.put(fwout3, "TotemHitsRP");
  std::auto_ptr<edm::PSimHitContainer> fwout4(new edm::PSimHitContainer);
  iEvent.put(fwout4, "FP420SI");
  std::auto_ptr<edm::PSimHitContainer> fwout5(new edm::PSimHitContainer);
  iEvent.put(fwout5, "BSCHits");

  // ----- SimTracks & SimVertices -----
  //
  std::auto_ptr<edm::SimTrackContainer> simTr(new edm::SimTrackContainer);
  iEvent.put(simTr);
  std::auto_ptr<edm::SimVertexContainer> simVe(new edm::SimVertexContainer);
  iEvent.put(simVe);


}


void NeutronHitsCollector::beginJob() {}


void NeutronHitsCollector::endJob() {}


//define this as a plug-in
DEFINE_FWK_MODULE(NeutronHitsCollector);
