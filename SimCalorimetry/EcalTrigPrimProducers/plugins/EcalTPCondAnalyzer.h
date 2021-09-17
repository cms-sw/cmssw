// -*- C++ -*-
//
// Class:      EcalTPCondAnalyzer
//
/**\class EcalTPCondAnalyzer

 Description: prints the TPG conditions coming from the conditions DB
 Prints in exactly the same format as TPG.txt, such that a "diff" is possible.

*/
//
// Original Author:  Ursula Berthon
//         Created:  Wed Oct 15  11:38:38 CEST 2008
//
//

#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

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
#include "CondFormats/EcalObjects/interface/EcalTPGWeightGroup.h"
#include "CondFormats/EcalObjects/interface/EcalTPGWeightIdMap.h"

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
#include "CondFormats/DataRecord/interface/EcalTPGWeightGroupRcd.h"
#include "CondFormats/DataRecord/interface/EcalTPGWeightIdMapRcd.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"

//
// class declaration
//

class EcalTPCondAnalyzer : public edm::one::EDAnalyzer<edm::one::WatchRuns> {
public:
  explicit EcalTPCondAnalyzer(const edm::ParameterSet &);
  ~EcalTPCondAnalyzer() override;

  void analyze(const edm::Event &, const edm::EventSetup &) override;
  void beginJob() override;
  void beginRun(const edm::Run &run, const edm::EventSetup &evtSetup) override;
  void endJob() override;
  void endRun(edm::Run const &, edm::EventSetup const &) override;

private:
  unsigned long long getRecords(edm::EventSetup const &setup);
  unsigned long long cacheID_;

  edm::ESGetToken<CaloSubdetectorGeometry, EcalEndcapGeometryRecord> tokenEndcapGeom_;
  edm::ESGetToken<CaloSubdetectorGeometry, EcalBarrelGeometryRecord> tokenBarrelGeom_;
  edm::ESGetToken<EcalTPGPhysicsConst, EcalTPGPhysicsConstRcd> tokenEcalTPGPhysics_;
  edm::ESGetToken<EcalTPGLinearizationConst, EcalTPGLinearizationConstRcd> tokenEcalTPGLinearization_;
  edm::ESGetToken<EcalTPGPedestals, EcalTPGPedestalsRcd> tokenEcalTPGPedestals_;
  edm::ESGetToken<EcalTPGWeightIdMap, EcalTPGWeightIdMapRcd> tokenEcalTPGWeightIdMap_;
  edm::ESGetToken<EcalTPGFineGrainEBIdMap, EcalTPGFineGrainEBIdMapRcd> tokenEcalTPGFineGrainEBIdMap_;
  edm::ESGetToken<EcalTPGLutIdMap, EcalTPGLutIdMapRcd> tokenEcalTPGLutIdMap_;
  edm::ESGetToken<EcalTPGSlidingWindow, EcalTPGSlidingWindowRcd> tokenEcalTPGSlidingWindow_;
  edm::ESGetToken<EcalTPGFineGrainStripEE, EcalTPGFineGrainStripEERcd> tokenEcalTPGFineGrainStripEE_;
  edm::ESGetToken<EcalTPGWeightGroup, EcalTPGWeightGroupRcd> tokenEcalTPGWeightGroup_;
  edm::ESGetToken<EcalTPGLutGroup, EcalTPGLutGroupRcd> tokenEcalTPGLutGroup_;
  edm::ESGetToken<EcalTPGFineGrainEBGroup, EcalTPGFineGrainEBGroupRcd> tokenEcalTPGFineGrainEBGroup_;
  edm::ESGetToken<EcalTPGSpike, EcalTPGSpikeRcd> tokenEcalTPGSpike_;
  edm::ESGetToken<EcalTPGFineGrainTowerEE, EcalTPGFineGrainTowerEERcd> tokenEcalTPGFineGrainTowerEE_;

  const CaloSubdetectorGeometry *theEndcapGeometry_;
  const CaloSubdetectorGeometry *theBarrelGeometry_;

  void printComment() const;
  void printEcalTPGPhysicsConst(const EcalTPGPhysicsConst *) const;
  void printCRYSTAL(const EcalTPGPedestals *ecaltpPed, const EcalTPGLinearizationConst *ecaltpLin);
  void printSTRIP(const EcalTPGSlidingWindow *ecaltpgSlidW,
                  const EcalTPGWeightGroup *ecaltpgWeightGroup,
                  const EcalTPGFineGrainStripEE *ecaltpgFgStripEE) const;
  void printWEIGHT(const EcalTPGWeightIdMap *ecaltpgWeightMap) const;
  void printTOWEREB(const EcalTPGSpike *ecaltpgSpike,
                    const EcalTPGFineGrainEBGroup *ecaltpgFgEBGroup,
                    const EcalTPGLutGroup *ecaltpgLutGroup) const;
  void printEcalTPGLutIdMap(const EcalTPGLutIdMap *ecaltpgLut) const;
  void printEcalTPGFineGrainEBIdMap(const EcalTPGFineGrainEBIdMap *ecaltpgFineGrainEB) const;
  void printTOWEREE(const EcalTPGFineGrainTowerEE *ecaltpgFineGrainTowerEE,
                    const EcalTPGLutGroup *ecaltpgLutGroup) const;
};
