#ifndef SimCalorimetry_EcalEBTrigPrimProducers_EcalEBTrigPrimPhase2ESProducer_H
#define SimCalorimetry_EcalEBTrigPrimProducers_EcalEBTrigPrimPhase2ESProducer_H

// user include files
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/ModuleFactory.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"

#include "CondFormats/DataRecord/interface/EcalTPGCrystalStatusRcd.h"
#include "CondFormats/DataRecord/interface/EcalTPGPhysicsConstRcd.h"

//#include "CondFormats/DataRecord/interface/EcalTPGSpikeRcd.h"
#include "CondFormats/DataRecord/interface/EcalTPGStripStatusRcd.h"
//#include "CondFormats/DataRecord/interface/EcalTPGTowerStatusRcd.h"

#include "CondFormats/DataRecord/interface/EcalEBPhase2TPGLinearizationConstRcd.h"
#include "CondFormats/DataRecord/interface/EcalTPGWeightGroupRcd.h"
#include "CondFormats/DataRecord/interface/EcalEBPhase2TPGAmplWeightIdMapRcd.h"
#include "CondFormats/DataRecord/interface/EcalEBPhase2TPGTimeWeightIdMapRcd.h"
#include "CondFormats/DataRecord/interface/EcalEBPhase2TPGPedestalsRcd.h"
#include "CondFormats/EcalObjects/interface/EcalTPGCrystalStatus.h"
#include "CondFormats/EcalObjects/interface/EcalEBPhase2TPGLinearizationConst.h"
#include "CondFormats/EcalObjects/interface/EcalEBPhase2TPGPedestals.h"
#include "CondFormats/EcalObjects/interface/EcalTPGPhysicsConst.h"
//#include "CondFormats/EcalObjects/interface/EcalTPGSpike.h"
#include "CondFormats/EcalObjects/interface/EcalTPGStripStatus.h"
//#include "CondFormats/EcalObjects/interface/EcalTPGTowerStatus.h"
#include "CondFormats/EcalObjects/interface/EcalTPGWeightGroup.h"
#include "CondFormats/EcalObjects/interface/EcalEBPhase2TPGAmplWeightIdMap.h"
#include "CondFormats/EcalObjects/interface/EcalEBPhase2TPGTimeWeightIdMap.h"

#include "zlib.h"

/** \class EcalEBTrigPrimPhase2ESProducer
\author L. Lutton, N. Marinelli - Univ. of Notre Dame
 Description: forPhase II 
*/

class EcalEBTrigPrimPhase2ESProducer : public edm::ESProducer {
public:
  EcalEBTrigPrimPhase2ESProducer(const edm::ParameterSet &);
  ~EcalEBTrigPrimPhase2ESProducer() override;

  std::unique_ptr<EcalEBPhase2TPGLinearizationConst> produceLinearizationConst(
      const EcalEBPhase2TPGLinearizationConstRcd &);
  std::unique_ptr<EcalEBPhase2TPGPedestalsMap> producePedestals(const EcalEBPhase2TPGPedestalsRcd &);
  std::unique_ptr<EcalEBPhase2TPGAmplWeightIdMap> produceAmpWeight(const EcalEBPhase2TPGAmplWeightIdMapRcd &);
  std::unique_ptr<EcalEBPhase2TPGTimeWeightIdMap> produceTimeWeight(const EcalEBPhase2TPGTimeWeightIdMapRcd &);
  std::unique_ptr<EcalTPGWeightGroup> produceWeightGroup(const EcalTPGWeightGroupRcd &);
  //std::unique_ptr<EcalTPGLutGroup> produceLutGroup(const EcalTPGLutGroupRcd &);

  std::unique_ptr<EcalTPGPhysicsConst> producePhysicsConst(const EcalTPGPhysicsConstRcd &);
  std::unique_ptr<EcalTPGCrystalStatus> produceBadX(const EcalTPGCrystalStatusRcd &);
  //std::unique_ptr<EcalTPGStripStatus> produceBadStrip(const EcalTPGStripStatusRcd &);
  //std::unique_ptr<EcalTPGTowerStatus> produceBadTT(const EcalTPGTowerStatusRcd &);
  //std::unique_ptr<EcalTPGSpike> produceSpike(const EcalTPGSpikeRcd &);

private:
  void parseTextFile();
  std::vector<int> getRange(int subdet, int smNb, int towerNbInSm, int stripNbInTower = 0, int xtalNbInStrip = 0);
  void parseWeightsFile();

  // ----------member data ---------------------------
  std::string dbFilename_;
  //  std::string configFilename_;
  const edm::FileInPath configFilename_;
  bool flagPrint_;
  std::map<uint32_t, std::vector<uint32_t>> mapXtal_;
  std::map<uint32_t, std::vector<uint32_t>> mapStrip_[2];
  std::map<uint32_t, std::vector<uint32_t>> mapTower_[2];
  std::map<uint32_t, std::vector<uint32_t>> mapWeight_;
  std::map<uint32_t, std::vector<uint32_t>> mapTimeWeight_;
  std::map<int, std::vector<unsigned int>> mapXtalToGroup_;
  std::map<int, std::vector<unsigned int>> mapXtalToLin_;
  std::map<uint32_t, std::vector<float>> mapPhys_;
  static const int maxSamplesUsed_;
  static const int nLinConst_;


};

#endif
