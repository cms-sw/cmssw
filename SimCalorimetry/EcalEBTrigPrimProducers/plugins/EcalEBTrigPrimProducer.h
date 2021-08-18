#ifndef EcalEBTrigPrimProducer_h
#define EcalEBTrigPrimProducer_h

/** \class EcalEBTrigPrimProducer
 *  For Phase II 
 *
 ************************************************************/

#include <memory>

#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Utilities/interface/ESGetToken.h"

#include "DataFormats/Common/interface/Handle.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"
#include "CondFormats/DataRecord/interface/EcalTPGLinearizationConstRcd.h"
#include "CondFormats/DataRecord/interface/EcalTPGPedestalsRcd.h"
#include "CondFormats/DataRecord/interface/EcalTPGCrystalStatusRcd.h"

#include "CondFormats/EcalObjects/interface/EcalTPGLinearizationConst.h"
#include "CondFormats/EcalObjects/interface/EcalTPGPedestals.h"
#include "CondFormats/EcalObjects/interface/EcalTPGCrystalStatus.h"
#include "CondFormats/DataRecord/interface/EcalTPGWeightIdMapRcd.h"
#include "CondFormats/DataRecord/interface/EcalTPGWeightGroupRcd.h"
#include "CondFormats/DataRecord/interface/EcalTPGSlidingWindowRcd.h"
#include "CondFormats/EcalObjects/interface/EcalTPGSlidingWindow.h"
#include "CondFormats/DataRecord/interface/EcalTPGLutGroupRcd.h"
#include "CondFormats/DataRecord/interface/EcalTPGLutIdMapRcd.h"
#include "CondFormats/EcalObjects/interface/EcalTPGLutGroup.h"
#include "CondFormats/EcalObjects/interface/EcalTPGLutIdMap.h"
#include "CondFormats/DataRecord/interface/EcalTPGTowerStatusRcd.h"
#include "CondFormats/DataRecord/interface/EcalTPGSpikeRcd.h"
#include "CondFormats/EcalObjects/interface/EcalTPGSpike.h"
#include "CondFormats/EcalObjects/interface/EcalTPGWeightIdMap.h"
#include "CondFormats/EcalObjects/interface/EcalTPGWeightGroup.h"
#include "CondFormats/EcalObjects/interface/EcalTPGTowerStatus.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/CaloTopology/interface/EcalTrigTowerConstituentsMap.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"

class EcalEBTrigPrimTestAlgo;

class EcalEBTrigPrimProducer : public edm::stream::EDProducer<> {
public:
  explicit EcalEBTrigPrimProducer(const edm::ParameterSet& conf);

  ~EcalEBTrigPrimProducer() override;

  void beginRun(const edm::Run& run, const edm::EventSetup& es) override;
  void endRun(const edm::Run&, const edm::EventSetup&) override;
  void produce(edm::Event& e, const edm::EventSetup& c) override;

private:
  std::unique_ptr<EcalEBTrigPrimTestAlgo> algo_;
  bool barrelOnly_;
  bool tcpFormat_;
  bool debug_;
  bool famos_;
  int nSamples_;
  int nEvent_;

  edm::EDGetTokenT<EBDigiCollection> tokenEBdigi_;
  edm::ESGetToken<EcalTPGLinearizationConst, EcalTPGLinearizationConstRcd> theEcalTPGLinearization_Token_;
  edm::ESGetToken<EcalTPGPedestals, EcalTPGPedestalsRcd> theEcalTPGPedestals_Token_;
  edm::ESGetToken<EcalTPGCrystalStatus, EcalTPGCrystalStatusRcd> theEcalTPGCrystalStatus_Token_;
  edm::ESGetToken<EcalTPGWeightIdMap, EcalTPGWeightIdMapRcd> theEcalTPGWEightIdMap_Token_;
  edm::ESGetToken<EcalTPGWeightGroup, EcalTPGWeightGroupRcd> theEcalTPGWEightGroup_Token_;
  edm::ESGetToken<EcalTPGSlidingWindow, EcalTPGSlidingWindowRcd> theEcalTPGSlidingWindow_Token_;
  edm::ESGetToken<EcalTPGLutGroup, EcalTPGLutGroupRcd> theEcalTPGLutGroup_Token_;
  edm::ESGetToken<EcalTPGLutIdMap, EcalTPGLutIdMapRcd> theEcalTPGLutIdMap_Token_;
  edm::ESGetToken<EcalTPGTowerStatus, EcalTPGTowerStatusRcd> theEcalTPGTowerStatus_Token_;
  edm::ESGetToken<EcalTPGSpike, EcalTPGSpikeRcd> theEcalTPGSpike_Token_;
  //these are only used if we also handle the endcap
  edm::ESGetToken<EcalTrigTowerConstituentsMap, IdealGeometryRecord> eTTmapToken_;
  edm::ESGetToken<CaloGeometry, CaloGeometryRecord> theGeometryToken_;

  int binOfMaximum_;
  bool fillBinOfMaximumFromHistory_;

  unsigned long long getRecords(edm::EventSetup const& setup);
  unsigned long long cacheID_;
};

#endif
