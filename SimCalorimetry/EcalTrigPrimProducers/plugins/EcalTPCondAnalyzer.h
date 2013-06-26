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
// $Id: EcalTPCondAnalyzer.h,v 1.6 2013/02/28 20:06:33 wmtan Exp $
//
//


// system include files
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "CondFormats/EcalObjects/interface/EcalTPGLinearizationConst.h"
#include "CondFormats/EcalObjects/interface/EcalTPGPedestals.h"
#include "CondFormats/EcalObjects/interface/EcalTPGSlidingWindow.h"
#include "CondFormats/EcalObjects/interface/EcalTPGWeightIdMap.h"
#include "CondFormats/EcalObjects/interface/EcalTPGWeightGroup.h"
#include "CondFormats/EcalObjects/interface/EcalTPGFineGrainStripEE.h"
#include "CondFormats/EcalObjects/interface/EcalTPGFineGrainEBGroup.h"
#include "CondFormats/EcalObjects/interface/EcalTPGLutGroup.h"
#include "CondFormats/EcalObjects/interface/EcalTPGLutIdMap.h"
#include "CondFormats/EcalObjects/interface/EcalTPGFineGrainEBIdMap.h"
#include "CondFormats/EcalObjects/interface/EcalTPGFineGrainTowerEE.h"
#include "CondFormats/EcalObjects/interface/EcalTPGPhysicsConst.h"
#include "CondFormats/EcalObjects/interface/EcalTPGCrystalStatus.h"
#include "CondFormats/EcalObjects/interface/EcalTPGTowerStatus.h"
#include "CondFormats/EcalObjects/interface/EcalTPGStripStatus.h"
#include "CondFormats/EcalObjects/interface/EcalTPGSpike.h"

class CaloSubdetectorGeometry ;

#include <vector>
#include <string>

//
// class declaration
//

class EcalTPCondAnalyzer : public edm::EDAnalyzer {
 public:
  explicit EcalTPCondAnalyzer(const edm::ParameterSet&);
  ~EcalTPCondAnalyzer();


  virtual void analyze(const edm::Event&, const edm::EventSetup&) override;
  virtual void beginJob();
  void beginRun(const edm::Run & run, const edm::EventSetup& es) override;
  virtual void endJob();

 private:

  unsigned long long getRecords(edm::EventSetup const& setup);
  unsigned long long cacheID_;

  const CaloSubdetectorGeometry * theEndcapGeometry_ ;
  const CaloSubdetectorGeometry * theBarrelGeometry_ ;

  void printComment() const;
  void printEcalTPGPhysicsConst(const EcalTPGPhysicsConst *) const;
  void printCRYSTAL(const EcalTPGPedestals * ecaltpPed, const EcalTPGLinearizationConst * ecaltpLin ) ;
  void printSTRIP(const EcalTPGSlidingWindow *ecaltpgSlidW,const EcalTPGWeightGroup *ecaltpgWeightGroup,const EcalTPGFineGrainStripEE * ecaltpgFgStripEE) const;
  void printWEIGHT(const EcalTPGWeightIdMap * ecaltpgWeightMap) const ;
  void printTOWEREB(const EcalTPGSpike *ecaltpgSpike, const EcalTPGFineGrainEBGroup *ecaltpgFgEBGroup,const EcalTPGLutGroup *ecaltpgLutGroup) const;
  void printEcalTPGLutIdMap(const EcalTPGLutIdMap *ecaltpgLut) const;
  void printEcalTPGFineGrainEBIdMap(const EcalTPGFineGrainEBIdMap *ecaltpgFineGrainEB) const ;
  void printTOWEREE(const EcalTPGFineGrainTowerEE *ecaltpgFineGrainTowerEE,const EcalTPGLutGroup *ecaltpgLutGroup) const ;
  void printBadX(const EcalTPGCrystalStatus *ecaltpgBadX) const; 
  void printBadTT(const EcalTPGTowerStatus *ecaltpgBadTT) const;
  void printBadStrip(const EcalTPGStripStatus *ecaltpgBadStrip) const;
  void printSpikeTh(const EcalTPGSpike *ecaltpgSpike) const;

};

