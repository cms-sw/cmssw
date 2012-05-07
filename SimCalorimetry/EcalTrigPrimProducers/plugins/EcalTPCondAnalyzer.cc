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
// $Id: EcalTPCondAnalyzer.cc,v 1.1 2008/10/15 15:51:15 uberthon Exp $
//
//
//


// system include files
#include <memory>
#include <utility>
#include <iostream>

// user include files
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "CondFormats/DataRecord/interface/EcalTPGPhysicsConstRcd.h"
#include "CondFormats/DataRecord/interface/EcalTPGFineGrainEBGroupRcd.h"
#include "CondFormats/DataRecord/interface/EcalTPGLutGroupRcd.h"
#include "CondFormats/DataRecord/interface/EcalTPGLutIdMapRcd.h"
#include "CondFormats/DataRecord/interface/EcalTPGFineGrainEBIdMapRcd.h"
#include "CondFormats/DataRecord/interface/EcalTPGFineGrainTowerEERcd.h"
#include "CondFormats/DataRecord/interface/EcalTPGLinearizationConstRcd.h"
#include "CondFormats/DataRecord/interface/EcalTPGPedestalsRcd.h"
#include "CondFormats/DataRecord/interface/EcalTPGSlidingWindowRcd.h"
#include "CondFormats/DataRecord/interface/EcalTPGWeightIdMapRcd.h"
#include "CondFormats/DataRecord/interface/EcalTPGWeightGroupRcd.h"
#include "CondFormats/DataRecord/interface/EcalTPGFineGrainStripEERcd.h"

#include "DataFormats/EcalDetId/interface/EcalTrigTowerDetId.h"
#include "DataFormats/EcalDetId/interface/EcalTriggerElectronicsId.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"

#include "SimCalorimetry/EcalTrigPrimProducers/plugins/EcalTPCondAnalyzer.h"

EcalTPCondAnalyzer::EcalTPCondAnalyzer(const edm::ParameterSet&  iConfig)

{}

void EcalTPCondAnalyzer::beginRun(const edm::Run & run, edm::EventSetup const& evtSetup){
 // get geometry
  
  edm::ESHandle<CaloSubdetectorGeometry> theEndcapGeometry_handle, theBarrelGeometry_handle;
  evtSetup.get<EcalEndcapGeometryRecord>().get("EcalEndcap",theEndcapGeometry_handle);
  evtSetup.get<EcalBarrelGeometryRecord>().get("EcalBarrel",theBarrelGeometry_handle);
  theEndcapGeometry_ = &(*theEndcapGeometry_handle);
  theBarrelGeometry_ = &(*theBarrelGeometry_handle);

  cacheID_=this->getRecords(evtSetup);
}

void EcalTPCondAnalyzer::beginJob() 
{}


EcalTPCondAnalyzer::~EcalTPCondAnalyzer() {
}

unsigned long long  EcalTPCondAnalyzer::getRecords(edm::EventSetup const& setup) {
  //
  // get Eventsetup records and print them
  //
   printComment();

   edm::ESHandle<EcalTPGPhysicsConst> theEcalTPGPhysConst_handle;
  setup.get<EcalTPGPhysicsConstRcd>().get(theEcalTPGPhysConst_handle);
  const EcalTPGPhysicsConst * ecaltpPhysConst = theEcalTPGPhysConst_handle.product();
  printEcalTPGPhysicsConst(ecaltpPhysConst);
  // for EcalFenixStrip...

  // get parameter records for xtals
  edm::ESHandle<EcalTPGLinearizationConst> theEcalTPGLinearization_handle;
  setup.get<EcalTPGLinearizationConstRcd>().get(theEcalTPGLinearization_handle);
  const EcalTPGLinearizationConst * ecaltpLin = theEcalTPGLinearization_handle.product();

  edm::ESHandle<EcalTPGPedestals> theEcalTPGPedestals_handle;
  setup.get<EcalTPGPedestalsRcd>().get(theEcalTPGPedestals_handle);
  const EcalTPGPedestals * ecaltpPed = theEcalTPGPedestals_handle.product();
  printCRYSTAL(ecaltpPed,ecaltpLin );


  //weight
  edm::ESHandle<EcalTPGWeightIdMap> theEcalTPGWEightIdMap_handle;
  setup.get<EcalTPGWeightIdMapRcd>().get(theEcalTPGWEightIdMap_handle);
  const EcalTPGWeightIdMap * ecaltpgWeightMap = theEcalTPGWEightIdMap_handle.product();
  printWEIGHT(ecaltpgWeightMap);

   // .. and for EcalFenixTcp

  edm::ESHandle<EcalTPGFineGrainEBIdMap> theEcalTPGFineGrainEBIdMap_handle;
  setup.get<EcalTPGFineGrainEBIdMapRcd>().get(theEcalTPGFineGrainEBIdMap_handle);
  const EcalTPGFineGrainEBIdMap * ecaltpgFineGrainEB = theEcalTPGFineGrainEBIdMap_handle.product();
  printEcalTPGFineGrainEBIdMap(ecaltpgFineGrainEB);


  edm::ESHandle<EcalTPGLutIdMap> theEcalTPGLutIdMap_handle;
  setup.get<EcalTPGLutIdMapRcd>().get(theEcalTPGLutIdMap_handle);
  const EcalTPGLutIdMap * ecaltpgLut = theEcalTPGLutIdMap_handle.product();
  printEcalTPGLutIdMap(ecaltpgLut);

  //for strips
  edm::ESHandle<EcalTPGSlidingWindow> theEcalTPGSlidingWindow_handle;
  setup.get<EcalTPGSlidingWindowRcd>().get(theEcalTPGSlidingWindow_handle);
  const EcalTPGSlidingWindow * ecaltpgSlidW = theEcalTPGSlidingWindow_handle.product();
  edm::ESHandle<EcalTPGFineGrainStripEE> theEcalTPGFineGrainStripEE_handle;
  setup.get<EcalTPGFineGrainStripEERcd>().get(theEcalTPGFineGrainStripEE_handle);
  const EcalTPGFineGrainStripEE * ecaltpgFgStripEE = theEcalTPGFineGrainStripEE_handle.product();     
  edm::ESHandle<EcalTPGWeightGroup> theEcalTPGWEightGroup_handle;
  setup.get<EcalTPGWeightGroupRcd>().get(theEcalTPGWEightGroup_handle);
  const EcalTPGWeightGroup * ecaltpgWeightGroup = theEcalTPGWEightGroup_handle.product();
  printSTRIP(ecaltpgSlidW,ecaltpgWeightGroup,ecaltpgFgStripEE);

  // get parameter records for towers
  edm::ESHandle<EcalTPGLutGroup> theEcalTPGLutGroup_handle;
  setup.get<EcalTPGLutGroupRcd>().get(theEcalTPGLutGroup_handle);
  const EcalTPGLutGroup * ecaltpgLutGroup = theEcalTPGLutGroup_handle.product();

  edm::ESHandle<EcalTPGFineGrainEBGroup> theEcalTPGFineGrainEBGroup_handle;
  setup.get<EcalTPGFineGrainEBGroupRcd>().get(theEcalTPGFineGrainEBGroup_handle);
  const EcalTPGFineGrainEBGroup * ecaltpgFgEBGroup = theEcalTPGFineGrainEBGroup_handle.product();
  printTOWEREB(ecaltpgFgEBGroup,ecaltpgLutGroup);
  edm::ESHandle<EcalTPGFineGrainTowerEE> theEcalTPGFineGrainTowerEE_handle;
  setup.get<EcalTPGFineGrainTowerEERcd>().get(theEcalTPGFineGrainTowerEE_handle);
  const EcalTPGFineGrainTowerEE * ecaltpgFineGrainTowerEE = theEcalTPGFineGrainTowerEE_handle.product();
  printTOWEREE(ecaltpgFineGrainTowerEE,ecaltpgLutGroup);

  std::cout<<"EOF"<<std::endl;

  return setup.get<EcalTPGFineGrainTowerEERcd>().cacheIdentifier();
}

// ------------ method called to analyze the data  ------------
void
EcalTPCondAnalyzer::analyze(const edm::Event& iEvent, const  edm::EventSetup & iSetup){
}

void
EcalTPCondAnalyzer::endJob(){
}

void EcalTPCondAnalyzer::printEcalTPGPhysicsConst(const EcalTPGPhysicsConst *ecaltpgPhysConst) const {
  EcalTPGPhysicsConstMapIterator it;
  const EcalTPGPhysicsConstMap mymap=ecaltpgPhysConst->getMap();
  for (it=mymap.begin();it!=mymap.end();++it) {
    if (it==mymap.begin()) {
      std::cout<<"\nPHYSICS_EB "<<(*it).first<<std::endl;
    } else {
      std::cout<<"\nPHYSICS_EE "<<(*it).first<<std::endl;
    }
    std::cout<<(*it).second.EtSat<<" "<<(*it).second.ttf_threshold_Low<<" "<<(*it).second.ttf_threshold_High<<std::endl;
    std::cout<<(*it).second.FG_lowThreshold<<" "<<(*it).second.FG_highThreshold<<" "<<(*it).second.FG_lowRatio<<" "<<(*it).second.FG_highRatio<<std::endl;
  }
}

void  EcalTPCondAnalyzer::printSTRIP(const EcalTPGSlidingWindow *slWin,const EcalTPGWeightGroup *ecaltpgWeightGroup,const EcalTPGFineGrainStripEE * ecaltpgFgStripEE) const {
  // print STRIP information
  const EcalTPGSlidingWindowMap &slwinmap = slWin -> getMap();
  const EcalTPGFineGrainStripEEMap &fgstripEEmap= ecaltpgFgStripEE->getMap();
  EcalTPGSlidingWindowMapIterator it;
  const EcalTPGGroups::EcalTPGGroupsMap &gMap=ecaltpgWeightGroup->getMap();
  EcalTPGGroups::EcalTPGGroupsMapItr groupId;

  std::cout<<std::endl;
  for (int mysub=1;mysub<=2;++mysub) {
    std::cout<<std::endl;
    for (it=slwinmap.begin();it!=slwinmap.end();++it) {
      EcalTriggerElectronicsId elid((*it).first);
      groupId=gMap.find((*it).first);
      int subdet =elid.subdet();
      if (subdet==mysub) {
	if (subdet==1) {
	  std::cout<<"STRIP_EB "<<std::dec<<(*it).first<<std::endl;
	  std::cout << std::hex << "0x" <<(*it).second<<std::endl ;
	  std::cout  <<" " <<(*groupId).second<< std::endl ; //weightgroupid
	}else if (subdet==2) {
	  std::cout<<"STRIP_EE "<<std::dec<<(*it).first<<std::endl;
	  std::cout << std::hex << "0x" <<(*it).second<<std::endl ;
	  std::cout <<" " <<(*groupId).second<<std::endl ;//weightgroupid
	  EcalTPGFineGrainStripEEMapIterator it2=fgstripEEmap.find((*it).first);
	  if (it2==fgstripEEmap.end()) {
	    edm::LogWarning("EcalTPGCondAnalyzer") <<" could not find strip Id "<<(*it).first<<", given in sliding window, inside the EcalTPGFineGranStripEEMap!!!";
	  } else {
	    EcalTPGFineGrainStripEE::Item item=(*it2).second;
	    std::cout<<std::hex<<"0x"<<item.threshold<<" 0x"<<item.lut<<std::endl ;  
	  }
	}
      }
    }
  }
}

void EcalTPCondAnalyzer::printWEIGHT(const EcalTPGWeightIdMap * ecaltpgWeightIdMap) const {
    
  std::cout<<std::endl;
    EcalTPGWeightIdMap::EcalTPGWeightMapItr it;
    uint32_t w0,w1,w2,w3,w4;
    const EcalTPGWeightIdMap::EcalTPGWeightMap map=ecaltpgWeightIdMap->getMap();
    for (it=map.begin();it!=map.end();++it) {
      std::cout <<"WEIGHT "<<(*it).first<<std::endl;
      (*it).second.getValues(w0,w1,w2,w3,w4);
      std::cout <<std::hex<<"0x"<<w0<<" 0x"<<w1<<" 0x"<<w2<<" 0x"<<w3<<" 0x"<<w4<<" "<<std::endl;
    }
}

void  EcalTPCondAnalyzer::printEcalTPGFineGrainEBIdMap(const EcalTPGFineGrainEBIdMap *ecaltpgFineGrainEB) const {
    EcalTPGFineGrainEBIdMap::EcalTPGFineGrainEBMapItr it;
    const EcalTPGFineGrainEBIdMap::EcalTPGFineGrainEBMap map=ecaltpgFineGrainEB->getMap();
    uint32_t ThresholdETLow, ThresholdETHigh, RatioLow, RatioHigh, LUT;

    std::cout<<std::endl;
    for (it=map.begin();it!=map.end();++it) {
      std::cout <<"FG "<<(*it).first<<std::endl;
      (*it).second.getValues(ThresholdETLow, ThresholdETHigh, RatioLow, RatioHigh, LUT);
      std::cout <<std::hex<<"0x"<<ThresholdETLow<<" 0x"<<ThresholdETHigh<<" 0x"<<RatioLow<<" 0x"<<RatioHigh<<" 0x"<<LUT<<std::endl;
    }
}


void EcalTPCondAnalyzer::printEcalTPGLutIdMap(const EcalTPGLutIdMap *ecaltpgLut) const {
    EcalTPGLutIdMap::EcalTPGLutMapItr it;
    const EcalTPGLutIdMap::EcalTPGLutMap map=ecaltpgLut->getMap();

    std::cout<<std::endl;
    for (it=map.begin();it!=map.end();++it) {
      std::cout <<"LUT "<<(*it).first<<std::endl;
      const unsigned int * lut=(*it).second.getLut();
      for (unsigned int i=0;i<1024;++i)  std::cout <<std::hex<<"0x"<<*lut++<<std::endl;
    }
}

void EcalTPCondAnalyzer::printCRYSTAL(const EcalTPGPedestals * ecaltpPed, const EcalTPGLinearizationConst * ecaltpLin ) {

  std::cout<<std::endl;
  const EcalTPGPedestalsMap pedMap=ecaltpPed->getMap();
  const EcalTPGLinearizationConstMap linMap=ecaltpLin->getMap();

  const std::vector<DetId> & ebCells = theBarrelGeometry_->getValidDetIds(DetId::Ecal, EcalBarrel);

  std::cout<<"COMMENT ====== barrel crystals ====== "<<std::endl; 
  for (std::vector<DetId>::const_iterator it = ebCells.begin(); it != ebCells.end(); ++it) {
    EBDetId id(*it) ;
    std::cout <<"CRYSTAL "<<std::dec<<id.rawId()<<std::endl;
    const EcalTPGPedestal &ped=pedMap[id.rawId()];
    const EcalTPGLinearizationConstant &lin=linMap[id.rawId()];
    std::cout<<std::hex<<" 0x"<<ped.mean_x12<<" 0x"<<lin.mult_x12<<" 0x"<<lin.shift_x12<<std::endl;
    std::cout<<std::hex<<" 0x"<<ped.mean_x6 <<" 0x"<<lin.mult_x6 <<" 0x"<<lin.shift_x6<<std::endl;
    std::cout<<std::hex<<" 0x"<<ped.mean_x1 <<" 0x"<<lin.mult_x1 <<" 0x"<<lin.shift_x1<<std::endl;
  }

  const std::vector<DetId> & eeCells = theEndcapGeometry_->getValidDetIds(DetId::Ecal, EcalEndcap);
  std::cout<<"COMMENT ====== endcap crystals ====== "<<std::endl; 
  for (std::vector<DetId>::const_iterator it = eeCells.begin(); it != eeCells.end(); ++it) {
    EEDetId id(*it) ;
    std::cout <<"CRYSTAL "<<std::dec<<id.rawId()<<std::endl;
    const EcalTPGPedestal &ped=pedMap[id.rawId()];
    const EcalTPGLinearizationConstant &lin=linMap[id.rawId()];
    std::cout<<std::hex<<" 0x"<<ped.mean_x12<<" 0x"<<lin.mult_x12<<" 0x"<<lin.shift_x12<<std::endl;
    std::cout<<std::hex<<" 0x"<<ped.mean_x6 <<" 0x"<<lin.mult_x6 <<" 0x"<<lin.shift_x6<<std::endl;
    std::cout<<std::hex<<" 0x"<<ped.mean_x1 <<" 0x"<<lin.mult_x1 <<" 0x"<<lin.shift_x1<<std::endl;
  }
}
void EcalTPCondAnalyzer::printComment() const {
  std::cout<<"COMMENT put your comments here\n"<<
  "COMMENT =================================\n"<<
  "COMMENT           physics EB structure\n"<<
  "COMMENT\n"<<
  "COMMENT  EtSaturation (GeV), ttf_threshold_Low (GeV), ttf_threshold_High (GeV)\n"<<
  "COMMENT  FG_lowThreshold (GeV), FG_highThreshold (GeV), FG_lowRatio, FG_highRatio\n"<<
  "COMMENT =================================\n"<<
  "COMMENT\n"<<
  "COMMENT =================================\n"<<
  "COMMENT           physics EE structure\n"<<
  "COMMENT\n"<<
  "COMMENT  EtSaturation (GeV), ttf_threshold_Low (GeV), ttf_threshold_High (GeV)\n"<<
  "COMMENT  FG_Threshold (GeV), dummy, dummy, dummy\n"<<
  "COMMENT =================================\n"<<
  "COMMENT\n"<<
  "COMMENT =================================\n"<<
  "COMMENT           crystal structure (same for EB and EE)\n"<<
  "COMMENT\n"<<
  "COMMENT  ped, mult, shift [gain12]\n"<<
  "COMMENT  ped, mult, shift [gain6]\n"<<
  "COMMENT  ped, mult, shift [gain1]\n"<<
  "COMMENT =================================\n"<<
  "COMMENT\n"<<
  "COMMENT =================================\n"<<
  "COMMENT           strip EB structure\n"<<
  "COMMENT\n"<<
  "COMMENT  sliding_window\n"<<
  "COMMENT  weightGroupId\n"<<
  "COMMENT =================================\n"<<
  "COMMENT\n"<<
  "COMMENT =================================\n"<<
  "COMMENT           strip EE structure\n"<<
  "COMMENT\n"<<
  "COMMENT  sliding_window\n"<<
  "COMMENT  weightGroupId\n"<<
  "COMMENT  threshold_fg strip_lut_fg\n"<<
  "COMMENT =================================\n"<<
  "COMMENT\n"<<
  "COMMENT =================================\n"<<
  "COMMENT           tower EB structure\n"<<
  "COMMENT\n"<<
  "COMMENT  LUTGroupId\n"<<
  "COMMENT  FgGroupId\n"<<
  "COMMENT =================================\n"<<
  "COMMENT\n"<<
  "COMMENT =================================\n"<<
  "COMMENT           tower EE structure\n"<<
  "COMMENT\n"<<
  "COMMENT  LUTGroupId\n"<<
  "COMMENT  tower_lut_fg\n"<<
  "COMMENT =================================\n"<<
  "COMMENT\n"<<
  "COMMENT =================================\n"<<
  "COMMENT           Weight structure\n"<<
  "COMMENT\n"<<
  "COMMENT  weightGroupId\n"<<
  "COMMENT  w0, w1, w2, w3, w4\n"<<
  "COMMENT =================================\n"<<
  "COMMENT\n"<<
  "COMMENT =================================\n"<<
  "COMMENT           lut structure\n"<<
  "COMMENT\n"<<
  "COMMENT  LUTGroupId\n"<<
  "COMMENT  LUT[1-1024]\n"<<
  "COMMENT =================================\n"<<
  "COMMENT\n"<<
  "COMMENT =================================\n"<<
  "COMMENT           fg EB structure\n"<<
  "COMMENT\n"<<
  "COMMENT  FgGroupId\n"<<
  "COMMENT  el, eh, tl, th, lut_fg\n"<<
  "COMMENT =================================\n"<<
  "COMMENT"<<std::endl;
}

void EcalTPCondAnalyzer::printTOWEREB(const EcalTPGFineGrainEBGroup *ecaltpgFgEBGroup,const EcalTPGLutGroup *ecaltpgLutGroup) const {

    const EcalTPGGroups::EcalTPGGroupsMap &lutMap=ecaltpgLutGroup->getMap();
    EcalTPGGroups::EcalTPGGroupsMapItr lutGroupId;
    const EcalTPGGroups::EcalTPGGroupsMap &fgMap=ecaltpgFgEBGroup->getMap();
    EcalTPGGroups::EcalTPGGroupsMapItr it;

    std::cout<<std::endl;
    for (it=fgMap.begin();it!=fgMap.end();++it) {
      std::cout <<"TOWER_EB "<<std::dec<<(*it).first<<std::endl;
      lutGroupId=lutMap.find((*it).first);
      std::cout <<" "<<(*it).second<<std::endl;
      std::cout <<" "<<(*lutGroupId).second<<std::endl;
    }
}

void EcalTPCondAnalyzer::printTOWEREE(const EcalTPGFineGrainTowerEE *ecaltpgFineGrainTowerEE,const EcalTPGLutGroup *ecaltpgLutGroup) const {

    EcalTPGFineGrainTowerEEMapIterator it;
    const EcalTPGFineGrainTowerEEMap map=ecaltpgFineGrainTowerEE->getMap();
    const EcalTPGGroups::EcalTPGGroupsMap &lutMap=ecaltpgLutGroup->getMap();
    EcalTPGGroups::EcalTPGGroupsMapItr lutGroupId;

    std::cout<<std::endl;
    for (it=map.begin();it!=map.end();++it) {
      std::cout <<"TOWER_EE "<<std::dec<<(*it).first<<std::endl;
      lutGroupId=lutMap.find((*it).first);
      std::cout <<" "<<(*lutGroupId).second<<std::endl;
      std::cout <<std::hex<<"0x"<<(*it).second<<std::endl;
    }
}
