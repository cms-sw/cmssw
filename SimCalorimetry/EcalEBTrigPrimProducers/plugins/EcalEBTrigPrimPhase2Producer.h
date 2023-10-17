#ifndef SimCalorimetry_EcalEBTrigPrimProducers_EcalEBTrigPrimPhase2Producer_h
#define SimCalorimetry_EcalEBTrigPrimProducers_EcalEBTrigPrimPhase2Producer_h

/** \class EcalEBTrigPrimPhase2Producer 
\author L. Lutton, N. Marinelli - Univ. of Notre Dame
 Description: forPhase II 
 It consumes the new Phase2 digis based on the new EB electronics
 and plugs in the main steering algo for TP emulation
 It produces the EcalEBPhase2TrigPrimDigiCollection
*/

#include <memory>

#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Utilities/interface/ESGetToken.h"

#include "DataFormats/Common/interface/Handle.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"

#include "CondFormats/DataRecord/interface/EcalTPGCrystalStatusRcd.h"
#include "CondFormats/EcalObjects/interface/EcalTPGCrystalStatus.h"

#include "CondFormats/EcalObjects/interface/EcalLiteDTUPedestals.h"
#include "CondFormats/DataRecord/interface/EcalLiteDTUPedestalsRcd.h"

#include "CondFormats/DataRecord/interface/EcalEBPhase2TPGPedestalsRcd.h"
#include "CondFormats/DataRecord/interface/EcalEBPhase2TPGLinearizationConstRcd.h"
#include "CondFormats/DataRecord/interface/EcalEBPhase2TPGAmplWeightIdMapRcd.h"
#include "CondFormats/DataRecord/interface/EcalEBPhase2TPGTimeWeightIdMapRcd.h"
#include "CondFormats/DataRecord/interface/EcalTPGWeightGroupRcd.h"
#include "CondFormats/EcalObjects/interface/EcalEBPhase2TPGAmplWeightIdMap.h"
#include "CondFormats/EcalObjects/interface/EcalEBPhase2TPGTimeWeightIdMap.h"
#include "CondFormats/EcalObjects/interface/EcalEBPhase2TPGLinearizationConst.h"
#include "CondFormats/EcalObjects/interface/EcalEBPhase2TPGPedestals.h"
#include "CondFormats/EcalObjects/interface/EcalTPGWeightGroup.h"

#include "CondFormats/DataRecord/interface/EcalTPGTowerStatusRcd.h"
#include "CondFormats/DataRecord/interface/EcalTPGSpikeRcd.h"
#include "CondFormats/EcalObjects/interface/EcalTPGSpike.h"
#include "CondFormats/EcalObjects/interface/EcalTPGTowerStatus.h"

#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/CaloTopology/interface/EcalTrigTowerConstituentsMap.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"

class EcalEBPhase2TrigPrimAlgo;

class EcalEBTrigPrimPhase2Producer : public edm::stream::EDProducer<> {
public:
  explicit EcalEBTrigPrimPhase2Producer(const edm::ParameterSet& conf);

  ~EcalEBTrigPrimPhase2Producer() override;

  void beginRun(const edm::Run& run, const edm::EventSetup& es) override;
  void endRun(const edm::Run&, const edm::EventSetup&) override;
  void produce(edm::Event& e, const edm::EventSetup& c) override;
  static void fillDescriptions(edm::ConfigurationDescriptions&);

private:
  std::unique_ptr<EcalEBPhase2TrigPrimAlgo> algo_;
  bool debug_;
  bool famos_;
  int nEvent_;

  edm::EDGetTokenT<EBDigiCollectionPh2> tokenEBdigi_;
  edm::ESGetToken<EcalEBPhase2TPGLinearizationConst, EcalEBPhase2TPGLinearizationConstRcd>
      theEcalEBPhase2TPGLinearization_Token_;
  edm::ESGetToken<EcalEBPhase2TPGPedestalsMap, EcalEBPhase2TPGPedestalsRcd> theEcalEBPhase2TPGPedestals_Token_;

  edm::ESGetToken<EcalLiteDTUPedestalsMap, EcalLiteDTUPedestalsRcd> theEcalTPGPedestals_Token_;

  edm::ESGetToken<EcalTPGCrystalStatus, EcalTPGCrystalStatusRcd> theEcalTPGCrystalStatus_Token_;
  edm::ESGetToken<EcalEBPhase2TPGAmplWeightIdMap, EcalEBPhase2TPGAmplWeightIdMapRcd> theEcalEBTPGAmplWeightIdMap_Token_;
  edm::ESGetToken<EcalEBPhase2TPGTimeWeightIdMap, EcalEBPhase2TPGTimeWeightIdMapRcd> theEcalEBTPGTimeWeightIdMap_Token_;

  edm::ESGetToken<EcalTPGWeightGroup, EcalTPGWeightGroupRcd> theEcalTPGWeightGroup_Token_;

  edm::ESGetToken<EcalTPGTowerStatus, EcalTPGTowerStatusRcd> theEcalTPGTowerStatus_Token_;
  edm::ESGetToken<EcalTPGSpike, EcalTPGSpikeRcd> theEcalTPGSpike_Token_;

  edm::ESGetToken<EcalTrigTowerConstituentsMap, IdealGeometryRecord> eTTmapToken_;
  edm::ESGetToken<CaloGeometry, CaloGeometryRecord> theGeometryToken_;

  int binOfMaximum_;
  bool fillBinOfMaximumFromHistory_;

  unsigned long long getRecords(edm::EventSetup const& setup);
  unsigned long long cacheID_;
};

#endif
