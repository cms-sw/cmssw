// -*- C++ -*-
//
// Class:      EcalTPDBAnalyzer
//
/**\class EcalTPDBAnalyzer

 Description: test of the output of EcalTPDBProducer

*/
//
//
// Original Author:  Ursula Berthon
//         Created:  Wed Oct 15  11:38:38 CEST 2008
//
//
//

// system include files
#include <memory>
#include <utility>
#include <vector>

// user include files
#include "DataFormats/EcalDetId/interface/EcalTrigTowerDetId.h"
#include "DataFormats/EcalDetId/interface/EcalTriggerElectronicsId.h"

#include "SimCalorimetry/EcalTrigPrimProducers/plugins/EcalTPCondAnalyzer.h"

EcalTPCondAnalyzer::EcalTPCondAnalyzer(const edm::ParameterSet &iConfig)
    : tokenEndcapGeom_(esConsumes<edm::Transition::BeginRun>(edm::ESInputTag("", "EcalEndcap"))),
      tokenBarrelGeom_(esConsumes<edm::Transition::BeginRun>(edm::ESInputTag("", "EcalBarrel"))),
      tokenEcalTPGPhysics_(esConsumes<edm::Transition::BeginRun>()),
      tokenEcalTPGLinearization_(esConsumes<edm::Transition::BeginRun>()),
      tokenEcalTPGPedestals_(esConsumes<edm::Transition::BeginRun>()),
      tokenEcalTPGWeightIdMap_(esConsumes<edm::Transition::BeginRun>()),
      tokenEcalTPGFineGrainEBIdMap_(esConsumes<edm::Transition::BeginRun>()),
      tokenEcalTPGLutIdMap_(esConsumes<edm::Transition::BeginRun>()),
      tokenEcalTPGSlidingWindow_(esConsumes<edm::Transition::BeginRun>()),
      tokenEcalTPGFineGrainStripEE_(esConsumes<edm::Transition::BeginRun>()),
      tokenEcalTPGWeightGroup_(esConsumes<edm::Transition::BeginRun>()),
      tokenEcalTPGLutGroup_(esConsumes<edm::Transition::BeginRun>()),
      tokenEcalTPGFineGrainEBGroup_(esConsumes<edm::Transition::BeginRun>()),
      tokenEcalTPGSpike_(esConsumes<edm::Transition::BeginRun>()),
      tokenEcalTPGFineGrainTowerEE_(esConsumes<edm::Transition::BeginRun>()) {}

void EcalTPCondAnalyzer::beginRun(const edm::Run &run, edm::EventSetup const &evtSetup) {
  // get geometry
  theEndcapGeometry_ = &evtSetup.getData(tokenEndcapGeom_);
  theBarrelGeometry_ = &evtSetup.getData(tokenBarrelGeom_);

  cacheID_ = this->getRecords(evtSetup);
}

void EcalTPCondAnalyzer::beginJob() {}

EcalTPCondAnalyzer::~EcalTPCondAnalyzer() {}

unsigned long long EcalTPCondAnalyzer::getRecords(edm::EventSetup const &setup) {
  //
  // get Eventsetup records and print them
  //
  printComment();

  const auto ecaltpPhysConst = &setup.getData(tokenEcalTPGPhysics_);
  printEcalTPGPhysicsConst(ecaltpPhysConst);

  // for EcalFenixStrip...
  // get parameter records for xtals
  const auto *ecaltpLin = &setup.getData(tokenEcalTPGLinearization_);
  const auto *ecaltpPed = &setup.getData(tokenEcalTPGPedestals_);
  printCRYSTAL(ecaltpPed, ecaltpLin);

  // weight
  const auto *ecaltpgWeightMap = &setup.getData(tokenEcalTPGWeightIdMap_);
  printWEIGHT(ecaltpgWeightMap);

  // .. and for EcalFenixTcp
  const auto *ecaltpgFineGrainEB = &setup.getData(tokenEcalTPGFineGrainEBIdMap_);
  printEcalTPGFineGrainEBIdMap(ecaltpgFineGrainEB);

  const auto *ecaltpgLut = &setup.getData(tokenEcalTPGLutIdMap_);
  printEcalTPGLutIdMap(ecaltpgLut);

  // for strips
  const auto *ecaltpgSlidW = &setup.getData(tokenEcalTPGSlidingWindow_);
  const auto *ecaltpgFgStripEE = &setup.getData(tokenEcalTPGFineGrainStripEE_);
  const auto *ecaltpgWeightGroup = &setup.getData(tokenEcalTPGWeightGroup_);
  printSTRIP(ecaltpgSlidW, ecaltpgWeightGroup, ecaltpgFgStripEE);

  // get parameter records for towers
  const auto *ecaltpgLutGroup = &setup.getData(tokenEcalTPGLutGroup_);
  const auto *ecaltpgFgEBGroup = &setup.getData(tokenEcalTPGFineGrainEBGroup_);
  const auto *ecaltpgSpikeTh = &setup.getData(tokenEcalTPGSpike_);
  const auto *ecaltpgFineGrainTowerEE = &setup.getData(tokenEcalTPGFineGrainTowerEE_);

  printTOWEREB(ecaltpgSpikeTh, ecaltpgFgEBGroup, ecaltpgLutGroup);
  printTOWEREE(ecaltpgFineGrainTowerEE, ecaltpgLutGroup);

  edm::LogVerbatim("EcalTPCondAnalyzer") << "EOF";

  return setup.get<EcalTPGFineGrainTowerEERcd>().cacheIdentifier();
}

// ------------ method called to analyze the data  ------------
void EcalTPCondAnalyzer::analyze(const edm::Event &iEvent, const edm::EventSetup &iSetup) {}

void EcalTPCondAnalyzer::endJob() {}

void EcalTPCondAnalyzer::endRun(edm::Run const &, edm::EventSetup const &) {}

void EcalTPCondAnalyzer::printEcalTPGPhysicsConst(const EcalTPGPhysicsConst *ecaltpgPhysConst) const {
  EcalTPGPhysicsConstMapIterator it;
  const EcalTPGPhysicsConstMap &mymap = ecaltpgPhysConst->getMap();
  for (it = mymap.begin(); it != mymap.end(); ++it) {
    if (it == mymap.begin()) {
      edm::LogVerbatim("EcalTPCondAnalyzer") << "\nPHYSICS_EB " << (*it).first;
    } else {
      edm::LogVerbatim("EcalTPCondAnalyzer") << "\nPHYSICS_EE " << (*it).first;
    }
    edm::LogVerbatim("EcalTPCondAnalyzer")
        << (*it).second.EtSat << " " << (*it).second.ttf_threshold_Low << " " << (*it).second.ttf_threshold_High;
    edm::LogVerbatim("EcalTPCondAnalyzer") << (*it).second.FG_lowThreshold << " " << (*it).second.FG_highThreshold
                                           << " " << (*it).second.FG_lowRatio << " " << (*it).second.FG_highRatio;
  }
}

void EcalTPCondAnalyzer::printSTRIP(const EcalTPGSlidingWindow *slWin,
                                    const EcalTPGWeightGroup *ecaltpgWeightGroup,
                                    const EcalTPGFineGrainStripEE *ecaltpgFgStripEE) const {
  // print STRIP information
  const EcalTPGSlidingWindowMap &slwinmap = slWin->getMap();
  const EcalTPGFineGrainStripEEMap &fgstripEEmap = ecaltpgFgStripEE->getMap();
  EcalTPGSlidingWindowMapIterator it;
  const EcalTPGGroups::EcalTPGGroupsMap &gMap = ecaltpgWeightGroup->getMap();
  EcalTPGGroups::EcalTPGGroupsMapItr groupId;

  edm::LogVerbatim("EcalTPCondAnalyzer");
  for (int mysub = 1; mysub <= 2; ++mysub) {
    edm::LogVerbatim("EcalTPCondAnalyzer");
    for (it = slwinmap.begin(); it != slwinmap.end(); ++it) {
      EcalTriggerElectronicsId elid((*it).first);
      groupId = gMap.find((*it).first);
      int subdet = elid.subdet();
      if (subdet == mysub) {
        if (subdet == 1) {
          edm::LogVerbatim("EcalTPCondAnalyzer") << "STRIP_EB " << std::dec << (*it).first << "\n"
                                                 << std::hex << "0x" << (*it).second << "\n"
                                                 << "" << (*groupId).second;  // weightgroupid
          EcalTPGFineGrainStripEEMapIterator it2 = fgstripEEmap.find((*it).first);
          if (it2 == fgstripEEmap.end()) {
            edm::LogWarning("EcalTPGCondAnalyzer") << " could not find strip Id " << (*it).first
                                                   << ", given in sliding window, inside the "
                                                      "EcalTPGFineGranStripEEMap!!!";
          } else {
            EcalTPGFineGrainStripEE::Item item = (*it2).second;
            edm::LogVerbatim("EcalTPCondAnalyzer") << std::hex << "0x" << item.threshold << " 0x" << item.lut;
          }
        } else if (subdet == 2) {
          edm::LogVerbatim("EcalTPCondAnalyzer") << "STRIP_EE " << std::dec << (*it).first << "\n"
                                                 << std::hex << "0x" << (*it).second << "\n"
                                                 << " " << (*groupId).second;  // weightgroupid
          EcalTPGFineGrainStripEEMapIterator it2 = fgstripEEmap.find((*it).first);
          if (it2 == fgstripEEmap.end()) {
            edm::LogWarning("EcalTPGCondAnalyzer") << " could not find strip Id " << (*it).first
                                                   << ", given in sliding window, inside the "
                                                      "EcalTPGFineGranStripEEMap!!!";
          } else {
            EcalTPGFineGrainStripEE::Item item = (*it2).second;
            edm::LogVerbatim("EcalTPCondAnalyzer") << std::hex << "0x" << item.threshold << " 0x" << item.lut;
          }
        }
      }
    }
  }
}

void EcalTPCondAnalyzer::printWEIGHT(const EcalTPGWeightIdMap *ecaltpgWeightIdMap) const {
  edm::LogVerbatim("EcalTPCondAnalyzer");
  EcalTPGWeightIdMap::EcalTPGWeightMapItr it;
  uint32_t w0, w1, w2, w3, w4;
  const EcalTPGWeightIdMap::EcalTPGWeightMap &map = ecaltpgWeightIdMap->getMap();
  for (it = map.begin(); it != map.end(); ++it) {
    edm::LogVerbatim("EcalTPCondAnalyzer") << "WEIGHT " << (*it).first;
    (*it).second.getValues(w0, w1, w2, w3, w4);
    edm::LogVerbatim("EcalTPCondAnalyzer")
        << std::hex << "0x" << w0 << " 0x" << w1 << " 0x" << w2 << " 0x" << w3 << " 0x" << w4 << " \n\n\n";
  }
}

void EcalTPCondAnalyzer::printEcalTPGFineGrainEBIdMap(const EcalTPGFineGrainEBIdMap *ecaltpgFineGrainEB) const {
  EcalTPGFineGrainEBIdMap::EcalTPGFineGrainEBMapItr it;
  const EcalTPGFineGrainEBIdMap::EcalTPGFineGrainEBMap &map = ecaltpgFineGrainEB->getMap();
  uint32_t ThresholdETLow, ThresholdETHigh, RatioLow, RatioHigh, LUT;

  for (it = map.begin(); it != map.end(); ++it) {
    edm::LogVerbatim("EcalTPCondAnalyzer") << "FG " << (*it).first;
    (*it).second.getValues(ThresholdETLow, ThresholdETHigh, RatioLow, RatioHigh, LUT);
    edm::LogVerbatim("EcalTPCondAnalyzer") << std::hex << "0x" << ThresholdETLow << " 0x" << ThresholdETHigh << " 0x"
                                           << RatioLow << " 0x" << RatioHigh << " 0x" << LUT;
  }
}

void EcalTPCondAnalyzer::printEcalTPGLutIdMap(const EcalTPGLutIdMap *ecaltpgLut) const {
  EcalTPGLutIdMap::EcalTPGLutMapItr it;
  const EcalTPGLutIdMap::EcalTPGLutMap &map = ecaltpgLut->getMap();

  edm::LogVerbatim("EcalTPCondAnalyzer");
  for (it = map.begin(); it != map.end(); ++it) {
    edm::LogVerbatim("EcalTPCondAnalyzer") << "LUT " << (*it).first;
    const unsigned int *lut = (*it).second.getLut();
    for (unsigned int i = 0; i < 1024; ++i)
      edm::LogVerbatim("EcalTPCondAnalyzer") << std::hex << "0x" << *lut++;
  }
}

void EcalTPCondAnalyzer::printCRYSTAL(const EcalTPGPedestals *ecaltpPed, const EcalTPGLinearizationConst *ecaltpLin) {
  edm::LogVerbatim("EcalTPCondAnalyzer");
  const EcalTPGPedestalsMap &pedMap = ecaltpPed->getMap();
  const EcalTPGLinearizationConstMap &linMap = ecaltpLin->getMap();

  const std::vector<DetId> &ebCells = theBarrelGeometry_->getValidDetIds(DetId::Ecal, EcalBarrel);

  edm::LogVerbatim("EcalTPCondAnalyzer") << "COMMENT ====== barrel crystals ====== ";
  for (std::vector<DetId>::const_iterator it = ebCells.begin(); it != ebCells.end(); ++it) {
    EBDetId id(*it);
    edm::LogVerbatim("EcalTPCondAnalyzer") << "CRYSTAL " << std::dec << id.rawId();
    const EcalTPGPedestal &ped = pedMap[id.rawId()];
    const EcalTPGLinearizationConstant &lin = linMap[id.rawId()];
    edm::LogVerbatim("EcalTPCondAnalyzer")
        << std::hex << " 0x" << ped.mean_x12 << " 0x" << lin.mult_x12 << " 0x" << lin.shift_x12 << "\n"
        << std::hex << " 0x" << ped.mean_x6 << " 0x" << lin.mult_x6 << " 0x" << lin.shift_x6 << "\n"
        << std::hex << " 0x" << ped.mean_x1 << " 0x" << lin.mult_x1 << " 0x" << lin.shift_x1;
  }

  const std::vector<DetId> &eeCells = theEndcapGeometry_->getValidDetIds(DetId::Ecal, EcalEndcap);
  edm::LogVerbatim("EcalTPCondAnalyzer") << "COMMENT ====== endcap crystals ====== ";
  for (std::vector<DetId>::const_iterator it = eeCells.begin(); it != eeCells.end(); ++it) {
    EEDetId id(*it);
    edm::LogVerbatim("EcalTPCondAnalyzer") << "CRYSTAL " << std::dec << id.rawId();
    const EcalTPGPedestal &ped = pedMap[id.rawId()];
    const EcalTPGLinearizationConstant &lin = linMap[id.rawId()];
    edm::LogVerbatim("EcalTPCondAnalyzer")
        << std::hex << " 0x" << ped.mean_x12 << " 0x" << lin.mult_x12 << " 0x" << lin.shift_x12 << "\n"
        << std::hex << " 0x" << ped.mean_x6 << " 0x" << lin.mult_x6 << " 0x" << lin.shift_x6 << "\n"
        << std::hex << " 0x" << ped.mean_x1 << " 0x" << lin.mult_x1 << " 0x" << lin.shift_x1;
  }
}
void EcalTPCondAnalyzer::printComment() const {
  edm::LogVerbatim("EcalTPCondAnalyzer") << "COMMENT put your comments here\n"
                                         << "COMMENT =================================\n"
                                         << "COMMENT           physics EB structure\n"
                                         << "COMMENT\n"
                                         << "COMMENT  EtSaturation (GeV), ttf_threshold_Low (GeV), "
                                            "ttf_threshold_High (GeV)\n"
                                         << "COMMENT  FG_lowThreshold (GeV), FG_highThreshold (GeV), "
                                            "FG_lowRatio, FG_highRatio\n"
                                         << "COMMENT =================================\n"
                                         << "COMMENT\n"
                                         << "COMMENT =================================\n"
                                         << "COMMENT           physics EE structure\n"
                                         << "COMMENT\n"
                                         << "COMMENT  EtSaturation (GeV), ttf_threshold_Low (GeV), "
                                            "ttf_threshold_High (GeV)\n"
                                         << "COMMENT  FG_Threshold (GeV), dummy, dummy, dummy\n"
                                         << "COMMENT =================================\n"
                                         << "COMMENT\n"
                                         << "COMMENT =================================\n"
                                         << "COMMENT           crystal structure (same for EB and EE)\n"
                                         << "COMMENT\n"
                                         << "COMMENT  ped, mult, shift [gain12]\n"
                                         << "COMMENT  ped, mult, shift [gain6]\n"
                                         << "COMMENT  ped, mult, shift [gain1]\n"
                                         << "COMMENT =================================\n"
                                         << "COMMENT\n"
                                         << "COMMENT =================================\n"
                                         << "COMMENT           strip EB structure\n"
                                         << "COMMENT\n"
                                         << "COMMENT  sliding_window\n"
                                         << "COMMENT  weightGroupId\n"
                                         << "COMMENT  threshold_sfg lut_sfg\n"
                                         << "COMMENT =================================\n"
                                         << "COMMENT\n"
                                         << "COMMENT =================================\n"
                                         << "COMMENT           strip EE structure\n"
                                         << "COMMENT\n"
                                         << "COMMENT  sliding_window\n"
                                         << "COMMENT  weightGroupId\n"
                                         << "COMMENT  threshold_fg lut_fg\n"
                                         << "COMMENT =================================\n"
                                         << "COMMENT\n"
                                         << "COMMENT =================================\n"
                                         << "COMMENT           tower EB structure\n"
                                         << "COMMENT\n"
                                         << "COMMENT  LUTGroupId\n"
                                         << "COMMENT  FgGroupId\n"
                                         << "COMMENT  spike_killing_threshold\n"
                                         << "COMMENT =================================\n"
                                         << "COMMENT\n"
                                         << "COMMENT =================================\n"
                                         << "COMMENT           tower EE structure\n"
                                         << "COMMENT\n"
                                         << "COMMENT  LUTGroupId\n"
                                         << "COMMENT  tower_lut_fg\n"
                                         << "COMMENT =================================\n"
                                         << "COMMENT\n"
                                         << "COMMENT =================================\n"
                                         << "COMMENT           Weight structure\n"
                                         << "COMMENT\n"
                                         << "COMMENT  weightGroupId\n"
                                         << "COMMENT  w0, w1, w2, w3, w4\n"
                                         << "COMMENT =================================\n"
                                         << "COMMENT\n"
                                         << "COMMENT =================================\n"
                                         << "COMMENT           lut structure\n"
                                         << "COMMENT\n"
                                         << "COMMENT  LUTGroupId\n"
                                         << "COMMENT  LUT[1-1024]\n"
                                         << "COMMENT =================================\n"
                                         << "COMMENT\n"
                                         << "COMMENT =================================\n"
                                         << "COMMENT           fg EB structure\n"
                                         << "COMMENT\n"
                                         << "COMMENT  FgGroupId\n"
                                         << "COMMENT  el, eh, tl, th, lut_fg\n"
                                         << "COMMENT =================================\n"
                                         << "COMMENT";
}

void EcalTPCondAnalyzer::printTOWEREB(const EcalTPGSpike *ecaltpgSpikeTh,
                                      const EcalTPGFineGrainEBGroup *ecaltpgFgEBGroup,
                                      const EcalTPGLutGroup *ecaltpgLutGroup) const {
  const EcalTPGGroups::EcalTPGGroupsMap &lutMap = ecaltpgLutGroup->getMap();
  EcalTPGGroups::EcalTPGGroupsMapItr lutGroupId;
  const EcalTPGGroups::EcalTPGGroupsMap &fgMap = ecaltpgFgEBGroup->getMap();
  EcalTPGGroups::EcalTPGGroupsMapItr it;

  const EcalTPGSpike::EcalTPGSpikeMap &spikeThMap = ecaltpgSpikeTh->getMap();
  EcalTPGSpike::EcalTPGSpikeMapIterator itSpikeTh;

  edm::LogVerbatim("EcalTPCondAnalyzer");
  for (it = fgMap.begin(); it != fgMap.end(); ++it) {
    edm::LogVerbatim("EcalTPCondAnalyzer") << "TOWER_EB " << std::dec << (*it).first;
    lutGroupId = lutMap.find((*it).first);
    itSpikeTh = spikeThMap.find((*it).first);
    edm::LogVerbatim("EcalTPCondAnalyzer") << " " << (*it).second << "\n"
                                           << " " << (*lutGroupId).second << "\n"
                                           << " " << (*itSpikeTh).second;
  }
}

void EcalTPCondAnalyzer::printTOWEREE(const EcalTPGFineGrainTowerEE *ecaltpgFineGrainTowerEE,
                                      const EcalTPGLutGroup *ecaltpgLutGroup) const {
  EcalTPGFineGrainTowerEEMapIterator it;
  const EcalTPGFineGrainTowerEEMap &map = ecaltpgFineGrainTowerEE->getMap();
  const EcalTPGGroups::EcalTPGGroupsMap &lutMap = ecaltpgLutGroup->getMap();
  EcalTPGGroups::EcalTPGGroupsMapItr lutGroupId;

  edm::LogVerbatim("EcalTPCondAnalyzer");
  for (it = map.begin(); it != map.end(); ++it) {
    edm::LogVerbatim("EcalTPCondAnalyzer") << "TOWER_EE " << std::dec << (*it).first;
    lutGroupId = lutMap.find((*it).first);
    edm::LogVerbatim("EcalTPCondAnalyzer") << " " << (*lutGroupId).second << "\n" << std::hex << "0x" << (*it).second;
  }
}
