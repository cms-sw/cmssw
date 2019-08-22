#ifndef SimCalorimetry_EcalTrigPrimProducers_EcalTrigPrimESProducer_H
#define SimCalorimetry_EcalTrigPrimProducers_EcalTrigPrimESProducer_H

// user include files
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/ModuleFactory.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "CondFormats/DataRecord/interface/EcalTPGCrystalStatusRcd.h"
#include "CondFormats/DataRecord/interface/EcalTPGFineGrainEBGroupRcd.h"
#include "CondFormats/DataRecord/interface/EcalTPGFineGrainEBIdMapRcd.h"
#include "CondFormats/DataRecord/interface/EcalTPGFineGrainStripEERcd.h"
#include "CondFormats/DataRecord/interface/EcalTPGFineGrainTowerEERcd.h"
#include "CondFormats/DataRecord/interface/EcalTPGLinearizationConstRcd.h"
#include "CondFormats/DataRecord/interface/EcalTPGLutGroupRcd.h"
#include "CondFormats/DataRecord/interface/EcalTPGLutIdMapRcd.h"
#include "CondFormats/DataRecord/interface/EcalTPGPedestalsRcd.h"
#include "CondFormats/DataRecord/interface/EcalTPGPhysicsConstRcd.h"
#include "CondFormats/DataRecord/interface/EcalTPGSlidingWindowRcd.h"
#include "CondFormats/DataRecord/interface/EcalTPGSpikeRcd.h"
#include "CondFormats/DataRecord/interface/EcalTPGStripStatusRcd.h"
#include "CondFormats/DataRecord/interface/EcalTPGTowerStatusRcd.h"
#include "CondFormats/DataRecord/interface/EcalTPGWeightGroupRcd.h"
#include "CondFormats/DataRecord/interface/EcalTPGWeightIdMapRcd.h"
#include "CondFormats/EcalObjects/interface/EcalTPGCrystalStatus.h"
#include "CondFormats/EcalObjects/interface/EcalTPGFineGrainEBGroup.h"
#include "CondFormats/EcalObjects/interface/EcalTPGFineGrainEBIdMap.h"
#include "CondFormats/EcalObjects/interface/EcalTPGFineGrainStripEE.h"
#include "CondFormats/EcalObjects/interface/EcalTPGFineGrainTowerEE.h"
#include "CondFormats/EcalObjects/interface/EcalTPGLinearizationConst.h"
#include "CondFormats/EcalObjects/interface/EcalTPGLutGroup.h"
#include "CondFormats/EcalObjects/interface/EcalTPGLutIdMap.h"
#include "CondFormats/EcalObjects/interface/EcalTPGPedestals.h"
#include "CondFormats/EcalObjects/interface/EcalTPGPhysicsConst.h"
#include "CondFormats/EcalObjects/interface/EcalTPGSlidingWindow.h"
#include "CondFormats/EcalObjects/interface/EcalTPGSpike.h"
#include "CondFormats/EcalObjects/interface/EcalTPGStripStatus.h"
#include "CondFormats/EcalObjects/interface/EcalTPGTowerStatus.h"
#include "CondFormats/EcalObjects/interface/EcalTPGWeightGroup.h"
#include "CondFormats/EcalObjects/interface/EcalTPGWeightIdMap.h"

#include "zlib.h"

//
// class declaration
//

class EcalTrigPrimESProducer : public edm::ESProducer {
public:
  EcalTrigPrimESProducer(const edm::ParameterSet &);
  ~EcalTrigPrimESProducer() override;

  std::unique_ptr<EcalTPGPedestals> producePedestals(const EcalTPGPedestalsRcd &);
  std::unique_ptr<EcalTPGLinearizationConst> produceLinearizationConst(const EcalTPGLinearizationConstRcd &);
  std::unique_ptr<EcalTPGSlidingWindow> produceSlidingWindow(const EcalTPGSlidingWindowRcd &);
  std::unique_ptr<EcalTPGFineGrainEBIdMap> produceFineGrainEB(const EcalTPGFineGrainEBIdMapRcd &);
  std::unique_ptr<EcalTPGFineGrainStripEE> produceFineGrainEEstrip(const EcalTPGFineGrainStripEERcd &);
  std::unique_ptr<EcalTPGFineGrainTowerEE> produceFineGrainEEtower(const EcalTPGFineGrainTowerEERcd &);
  std::unique_ptr<EcalTPGLutIdMap> produceLUT(const EcalTPGLutIdMapRcd &);
  std::unique_ptr<EcalTPGWeightIdMap> produceWeight(const EcalTPGWeightIdMapRcd &);
  std::unique_ptr<EcalTPGWeightGroup> produceWeightGroup(const EcalTPGWeightGroupRcd &);
  std::unique_ptr<EcalTPGLutGroup> produceLutGroup(const EcalTPGLutGroupRcd &);
  std::unique_ptr<EcalTPGFineGrainEBGroup> produceFineGrainEBGroup(const EcalTPGFineGrainEBGroupRcd &);
  std::unique_ptr<EcalTPGPhysicsConst> producePhysicsConst(const EcalTPGPhysicsConstRcd &);
  std::unique_ptr<EcalTPGCrystalStatus> produceBadX(const EcalTPGCrystalStatusRcd &);
  std::unique_ptr<EcalTPGStripStatus> produceBadStrip(const EcalTPGStripStatusRcd &);
  std::unique_ptr<EcalTPGTowerStatus> produceBadTT(const EcalTPGTowerStatusRcd &);
  std::unique_ptr<EcalTPGSpike> produceSpike(const EcalTPGSpikeRcd &);

private:
  void parseTextFile();
  std::vector<int> getRange(int subdet, int smNb, int towerNbInSm, int stripNbInTower = 0, int xtalNbInStrip = 0);

  // ----------member data ---------------------------
  std::string dbFilename_;
  bool flagPrint_;
  std::map<uint32_t, std::vector<uint32_t>> mapXtal_;
  std::map<uint32_t, std::vector<uint32_t>> mapStrip_[2];
  std::map<uint32_t, std::vector<uint32_t>> mapTower_[2];
  std::map<uint32_t, std::vector<uint32_t>> mapWeight_;
  std::map<uint32_t, std::vector<uint32_t>> mapFg_;
  std::map<uint32_t, std::vector<uint32_t>> mapLut_;
  std::map<uint32_t, std::vector<float>> mapPhys_;

  //   typedef voidp gzFile;
  //   bool getNextString(gzFile &gzf);
  //   int converthex();
  //   char buf_[100];
  //   std::string bufString_;
  //   std::string sub_;
  //   int bufpos_;
};

#endif
