// -*- C++ -*-
//
// Class:      EcalTrigPrimAnalyzer
//
/**\class EcalTrigPrimAnalyzer

 Description: test of the output of EcalTrigPrimProducer

*/
//
//
// Original Author:  Ursula Berthon, Stephanie Baffioni, Pascal Paganini
//         Created:  Thu Jul 4 11:38:38 CEST 2005
//
//

// system include files
#include <memory>
#include <utility>

// user include files
#include "FWCore/Framework/interface/MakerMacros.h"
#include "DataFormats/EcalDigi/interface/EcalTriggerPrimitiveDigi.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHit.h"
#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"
#include "CalibCalorimetry/EcalTPGTools/interface/EcalTPGScale.h"
#include "EcalTrigPrimAnalyzer.h"

#include <TMath.h>

class CaloSubdetectorGeometry;

EcalTrigPrimAnalyzer::EcalTrigPrimAnalyzer(const edm::ParameterSet &iConfig)
    : recHits_(iConfig.getParameter<bool>("AnalyzeRecHits")),
      label_(iConfig.getParameter<edm::InputTag>("inputTP")),
      rechits_labelEB_(iConfig.getParameter<edm::InputTag>("inputRecHitsEB")),
      rechits_labelEE_(iConfig.getParameter<edm::InputTag>("inputRecHitsEE")),
      tpToken_(consumes<EcalTrigPrimDigiCollection>(label_)),
      ebToken_(consumes<EcalRecHitCollection>(rechits_labelEB_)),
      eeToken_(consumes<EcalRecHitCollection>(rechits_labelEE_)),
      tokens_(consumesCollector()) {
  usesResource(TFileService::kSharedResource);

  ecal_parts_.push_back("Barrel");
  ecal_parts_.push_back("Endcap");

  edm::Service<TFileService> fs;
  tree_ = fs->make<TTree>("TPGtree", "TPGtree");
  tree_->Branch("iphi", &iphi_, "iphi/I");
  tree_->Branch("ieta", &ieta_, "ieta/I");
  tree_->Branch("eRec", &eRec_, "eRec/F");
  tree_->Branch("tpgADC", &tpgADC_, "tpgADC/I");
  tree_->Branch("tpgGeV", &tpgGeV_, "tpgGeV/F");
  tree_->Branch("ttf", &ttf_, "ttf/I");
  tree_->Branch("fg", &fg_, "fg/I");
  for (unsigned int i = 0; i < 2; ++i) {
    ecal_et_[i] = fs->make<TH1I>(ecal_parts_[i].c_str(), "Et", 255, 0, 255);
    char title[30];
    sprintf(title, "%s_ttf", ecal_parts_[i].c_str());
    ecal_tt_[i] = fs->make<TH1I>(title, "TTF", 10, 0, 10);
    sprintf(title, "%s_fgvb", ecal_parts_[i].c_str());
    ecal_fgvb_[i] = fs->make<TH1I>(title, "FGVB", 10, 0, 10);
  }

  if (recHits_) {
    hTPvsRechit_ = fs->make<TH2F>("TP_vs_RecHit", "TP vs  rechit", 256, -1, 255, 255, 0, 255);
    hTPoverRechit_ = fs->make<TH1F>("TP_over_RecHit", "TP over rechit", 500, 0, 4);
    endcapGeomToken_ = esConsumes<CaloSubdetectorGeometry, EcalEndcapGeometryRecord>(edm::ESInputTag("", "EcalEndcap"));
    barrelGeomToken_ = esConsumes<CaloSubdetectorGeometry, EcalBarrelGeometryRecord>(edm::ESInputTag("", "EcalBarrel"));
    eTTmapToken_ = esConsumes<EcalTrigTowerConstituentsMap, IdealGeometryRecord>();
  }
}

//
// member functions
//

// ------------ method called to analyze the data  ------------
void EcalTrigPrimAnalyzer::analyze(const edm::Event &iEvent, const edm::EventSetup &iSetup) {
  // Get input
  const auto &tp = iEvent.get(tpToken_);
  for (unsigned int i = 0; i < tp.size(); i++) {
    const EcalTriggerPrimitiveDigi &d = tp[i];
    int subdet = d.id().subDet() - 1;
    if (subdet == 0) {
      ecal_et_[subdet]->Fill(d.compressedEt());
    } else {
      if (d.id().ietaAbs() == 27 || d.id().ietaAbs() == 28) {
        if (i % 2)
          ecal_et_[subdet]->Fill(d.compressedEt() * 2.);
      } else
        ecal_et_[subdet]->Fill(d.compressedEt());
    }
    ecal_tt_[subdet]->Fill(d.ttFlag());
    ecal_fgvb_[subdet]->Fill(d.fineGrain());
  }
  if (!recHits_)
    return;

  // comparison with RecHits
  const EcalRecHitCollection &rechit_EB_col = iEvent.get(ebToken_);
  const EcalRecHitCollection &rechit_EE_col = iEvent.get(eeToken_);

  const auto &theEndcapGeometry = iSetup.getData(endcapGeomToken_);
  const auto &theBarrelGeometry = iSetup.getData(barrelGeomToken_);
  const auto &eTTmap = iSetup.getData(eTTmapToken_);

  std::map<EcalTrigTowerDetId, float> mapTow_Et;

  for (unsigned int i = 0; i < rechit_EB_col.size(); i++) {
    const EBDetId &myid1 = rechit_EB_col[i].id();
    EcalTrigTowerDetId towid1 = myid1.tower();
    float theta = theBarrelGeometry.getGeometry(myid1)->getPosition().theta();
    float Etsum = rechit_EB_col[i].energy() * sin(theta);
    bool test_alreadyin = false;
    std::map<EcalTrigTowerDetId, float>::iterator ittest = mapTow_Et.find(towid1);
    if (ittest != mapTow_Et.end())
      test_alreadyin = true;
    if (test_alreadyin)
      continue;
    unsigned int j = i + 1;
    bool loopend = false;
    unsigned int count = 0;
    while (j < rechit_EB_col.size() && !loopend) {
      count++;
      const EBDetId &myid2 = rechit_EB_col[j].id();
      EcalTrigTowerDetId towid2 = myid2.tower();
      if (towid1 == towid2) {
        float theta = theBarrelGeometry.getGeometry(myid2)->getPosition().theta();
        Etsum += rechit_EB_col[j].energy() * sin(theta);
      }
      j++;
      if (count > 1800)
        loopend = true;
    }
    mapTow_Et.insert(std::pair<EcalTrigTowerDetId, float>(towid1, Etsum));
  }

  for (unsigned int i = 0; i < rechit_EE_col.size(); i++) {
    const EEDetId &myid1 = rechit_EE_col[i].id();
    EcalTrigTowerDetId towid1 = eTTmap.towerOf(myid1);
    float theta = theEndcapGeometry.getGeometry(myid1)->getPosition().theta();
    float Etsum = rechit_EE_col[i].energy() * sin(theta);
    bool test_alreadyin = false;
    std::map<EcalTrigTowerDetId, float>::iterator ittest = mapTow_Et.find(towid1);
    if (ittest != mapTow_Et.end())
      test_alreadyin = true;
    if (test_alreadyin)
      continue;
    unsigned int j = i + 1;
    bool loopend = false;
    unsigned int count = 0;
    while (j < rechit_EE_col.size() && !loopend) {
      const EEDetId &myid2 = rechit_EE_col[j].id();
      EcalTrigTowerDetId towid2 = eTTmap.towerOf(myid2);
      if (towid1 == towid2) {
        float theta = theEndcapGeometry.getGeometry(myid2)->getPosition().theta();
        Etsum += rechit_EE_col[j].energy() * sin(theta);
      }
      //  else loopend=true;
      j++;
      if (count > 500)
        loopend = true;
    }
    //    alreadyin_EE.push_back(towid1);
    mapTow_Et.insert(std::pair<EcalTrigTowerDetId, float>(towid1, Etsum));
  }

  EcalTPGScale ecalScale(tokens_, iSetup);
  for (unsigned int i = 0; i < tp.size(); i++) {
    const EcalTriggerPrimitiveDigi &d = tp[i];
    const EcalTrigTowerDetId TPtowid = d.id();
    std::map<EcalTrigTowerDetId, float>::iterator it = mapTow_Et.find(TPtowid);
    float Et = ecalScale.getTPGInGeV(d.compressedEt(), TPtowid);
    if (d.id().ietaAbs() == 27 || d.id().ietaAbs() == 28)
      Et *= 2;
    iphi_ = TPtowid.iphi();
    ieta_ = TPtowid.ieta();
    tpgADC_ = d.compressedEt();
    tpgGeV_ = Et;
    ttf_ = d.ttFlag();
    fg_ = d.fineGrain();
    if (it != mapTow_Et.end()) {
      hTPvsRechit_->Fill(it->second, Et);
      hTPoverRechit_->Fill(Et / it->second);
      eRec_ = it->second;
    }
    tree_->Fill();
  }
}
