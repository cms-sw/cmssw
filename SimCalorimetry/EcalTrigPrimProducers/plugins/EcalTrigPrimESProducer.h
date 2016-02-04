#ifndef SimCalorimetry_EcalTrigPrimProducers_EcalTrigPrimESProducer_H
#define SimCalorimetry_EcalTrigPrimProducers_EcalTrigPrimESProducer_H

// user include files
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "CondFormats/EcalObjects/interface/EcalTPGPedestals.h"
#include "CondFormats/EcalObjects/interface/EcalTPGLinearizationConst.h"
#include "CondFormats/EcalObjects/interface/EcalTPGSlidingWindow.h"
#include "CondFormats/EcalObjects/interface/EcalTPGFineGrainEBIdMap.h"
#include "CondFormats/EcalObjects/interface/EcalTPGFineGrainStripEE.h"
#include "CondFormats/EcalObjects/interface/EcalTPGFineGrainTowerEE.h"
#include "CondFormats/EcalObjects/interface/EcalTPGLutIdMap.h"
#include "CondFormats/EcalObjects/interface/EcalTPGWeightIdMap.h"
#include "CondFormats/EcalObjects/interface/EcalTPGWeightGroup.h"
#include "CondFormats/EcalObjects/interface/EcalTPGLutGroup.h"
#include "CondFormats/EcalObjects/interface/EcalTPGFineGrainEBGroup.h"
#include "CondFormats/EcalObjects/interface/EcalTPGPhysicsConst.h"
#include "CondFormats/EcalObjects/interface/EcalTPGCrystalStatus.h"
#include "CondFormats/EcalObjects/interface/EcalTPGTowerStatus.h"
#include "CondFormats/DataRecord/interface/EcalTPGPedestalsRcd.h"
#include "CondFormats/DataRecord/interface/EcalTPGLinearizationConstRcd.h"
#include "CondFormats/DataRecord/interface/EcalTPGSlidingWindowRcd.h"
#include "CondFormats/DataRecord/interface/EcalTPGFineGrainEBIdMapRcd.h"
#include "CondFormats/DataRecord/interface/EcalTPGFineGrainStripEERcd.h"
#include "CondFormats/DataRecord/interface/EcalTPGFineGrainTowerEERcd.h"
#include "CondFormats/DataRecord/interface/EcalTPGLutIdMapRcd.h"
#include "CondFormats/DataRecord/interface/EcalTPGWeightIdMapRcd.h"
#include "CondFormats/DataRecord/interface/EcalTPGWeightGroupRcd.h"
#include "CondFormats/DataRecord/interface/EcalTPGLutGroupRcd.h"
#include "CondFormats/DataRecord/interface/EcalTPGFineGrainEBGroupRcd.h"
#include "CondFormats/DataRecord/interface/EcalTPGPhysicsConstRcd.h"
#include "CondFormats/DataRecord/interface/EcalTPGCrystalStatusRcd.h"
#include "CondFormats/DataRecord/interface/EcalTPGTowerStatusRcd.h"

#include "zlib.h"

//
// class declaration
//

class EcalTrigPrimESProducer : public edm::ESProducer {
 public:
  EcalTrigPrimESProducer(const edm::ParameterSet&);
  ~EcalTrigPrimESProducer();

  std::auto_ptr<EcalTPGPedestals> producePedestals(const EcalTPGPedestalsRcd &) ;
  std::auto_ptr<EcalTPGLinearizationConst> produceLinearizationConst(const EcalTPGLinearizationConstRcd &) ;
  std::auto_ptr<EcalTPGSlidingWindow> produceSlidingWindow(const EcalTPGSlidingWindowRcd &) ;
  std::auto_ptr<EcalTPGFineGrainEBIdMap> produceFineGrainEB(const EcalTPGFineGrainEBIdMapRcd &) ;
  std::auto_ptr<EcalTPGFineGrainStripEE> produceFineGrainEEstrip(const EcalTPGFineGrainStripEERcd &) ;
  std::auto_ptr<EcalTPGFineGrainTowerEE> produceFineGrainEEtower(const EcalTPGFineGrainTowerEERcd &) ;
  std::auto_ptr<EcalTPGLutIdMap> produceLUT(const EcalTPGLutIdMapRcd &) ;
  std::auto_ptr<EcalTPGWeightIdMap> produceWeight(const EcalTPGWeightIdMapRcd &) ;
  std::auto_ptr<EcalTPGWeightGroup> produceWeightGroup(const EcalTPGWeightGroupRcd &) ;
  std::auto_ptr<EcalTPGLutGroup> produceLutGroup(const EcalTPGLutGroupRcd &) ;
  std::auto_ptr<EcalTPGFineGrainEBGroup> produceFineGrainEBGroup(const EcalTPGFineGrainEBGroupRcd &) ;
  std::auto_ptr<EcalTPGPhysicsConst> producePhysicsConst(const EcalTPGPhysicsConstRcd &) ;
  std::auto_ptr<EcalTPGCrystalStatus> produceBadX(const EcalTPGCrystalStatusRcd &) ;
  std::auto_ptr<EcalTPGTowerStatus> produceBadTT(const EcalTPGTowerStatusRcd &) ;
  
 private:

  void parseTextFile() ;
  std::vector<int> getRange(int subdet, int smNb, int towerNbInSm, int stripNbInTower=0, int xtalNbInStrip=0) ;

  // ----------member data ---------------------------
  std::string dbFilename_;
  std::map<uint32_t, std::vector<uint32_t> > mapXtal_ ;
  std::map<uint32_t, std::vector<uint32_t> > mapStrip_[2] ;
  std::map<uint32_t, std::vector<uint32_t> > mapTower_[2] ;
  std::map<uint32_t, std::vector<uint32_t> > mapWeight_ ;
  std::map<uint32_t, std::vector<uint32_t> > mapFg_ ;
  std::map<uint32_t, std::vector<uint32_t> > mapLut_ ;
  std::map<uint32_t, std::vector<float> > mapPhys_ ;

//   typedef voidp gzFile;
//   bool getNextString(gzFile &gzf);
//   int converthex();
//   char buf_[100];
//   std::string bufString_;
//   std::string sub_;
//   int bufpos_;

};


#endif
