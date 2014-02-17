// -*- C++ -*-
//
// Package:    NavigationSchoolAnalyzer
// Class:      NavigationSchoolAnalyzer
// 
/**\class NavigationSchoolAnalyzer NavigationSchoolAnalyzer.cc RecoTracker/NavigationSchoolAnalyzer/src/NavigationSchoolAnalyzer.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Jean-Roch Vlimant
//         Created:  Fri Mar 16 13:19:20 CDT 2007
// $Id: NavigationSchoolAnalyzer.cc,v 1.6 2012/11/27 09:52:21 slava77 Exp $
//
//


// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/ESHandle.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include <FWCore/MessageLogger/interface/MessageLogger.h>

#include "DataFormats/SiStripDetId/interface/TECDetId.h"
#include "DataFormats/SiStripDetId/interface/TIBDetId.h"
#include "DataFormats/SiStripDetId/interface/TIDDetId.h"
#include "DataFormats/SiStripDetId/interface/TOBDetId.h"
#include "DataFormats/SiPixelDetId/interface/PXBDetId.h"
#include "DataFormats/SiPixelDetId/interface/PXFDetId.h"
#include "DataFormats/MuonDetId/interface/RPCDetId.h"
#include "DataFormats/MuonDetId/interface/CSCDetId.h"
#include "DataFormats/MuonDetId/interface/DTChamberId.h"

#include "TrackingTools/DetLayers/interface/NavigationSchool.h"
#include "RecoTracker/Record/interface/NavigationSchoolRecord.h"

#include "TrackingTools/DetLayers/interface/DetLayer.h"

// class definition
class NavigationSchoolAnalyzer : public edm::EDAnalyzer {
public:
  explicit NavigationSchoolAnalyzer(const edm::ParameterSet&);
  ~NavigationSchoolAnalyzer();
  
  
private:
  virtual void beginRun(edm::Run & run, const edm::EventSetup&) ;
  virtual void analyze(const edm::Event&, const edm::EventSetup&);
  virtual void endJob() ;

  std::string theNavigationSchoolName;
};

//navigation printouts
std::ostream& operator<<(std::ostream& os,const DetLayer* dl){
  const std::vector<const GeomDet*>& bComponents = dl->basicComponents();

  if (bComponents.empty()){/* t'es pas dans la merde */;return os;}

  const GeomDet* tag = bComponents.front();
  unsigned int LorW=0;
  unsigned int side=0;

  switch (dl->subDetector()){
  case GeomDetEnumerators::PixelBarrel :
    LorW = PXBDetId(tag->geographicalId()).layer(); break;
  case GeomDetEnumerators::TIB :
    LorW = TIBDetId(tag->geographicalId()).layer(); break;
  case GeomDetEnumerators::TOB :
    LorW = TOBDetId(tag->geographicalId()).layer(); break;
  case GeomDetEnumerators::DT :
    LorW = DTChamberId(tag->geographicalId().rawId()).station(); break;
  case GeomDetEnumerators::RPCEndcap :
    side = (unsigned int)((RPCDetId(tag->geographicalId().rawId()).region()/2.+1)*2.);
  case GeomDetEnumerators::RPCBarrel :
    LorW = RPCDetId(tag->geographicalId().rawId()).station(); break;

  case GeomDetEnumerators::PixelEndcap :    
    LorW = PXFDetId(tag->geographicalId()).disk(); 
    side = PXFDetId(tag->geographicalId()).side();break;
  case GeomDetEnumerators::TID :
    LorW = TIDDetId(tag->geographicalId()).wheel(); 
    side = TIDDetId(tag->geographicalId()).side();break;
  case GeomDetEnumerators::TEC :
    LorW = TECDetId(tag->geographicalId()).wheel();
    side = TECDetId(tag->geographicalId()).side(); break;
  case GeomDetEnumerators::CSC :
    LorW = CSCDetId(tag->geographicalId().rawId()).layer();
    side = CSCDetId(tag->geographicalId().rawId()).endcap(); break;
  case GeomDetEnumerators::invalidDet: // make gcc happy
  default:
    // edm::LogError("InvalidDet") << "At " << __FILE__ << ", line " << __LINE__ << "\n";
    break;
  }
  
  switch (dl->location()){
  case GeomDetEnumerators::barrel:
    os<<"barrel subDetector: "<<dl->subDetector()<<"\n"
      <<"layer: "<<LorW<<"\n";
    break;
  case GeomDetEnumerators::endcap:
    os<<"endcap subDetector: "<<dl->subDetector()<<"\n"
      <<"wheel: "<<LorW<<"\n"
      <<"side: "<<((side==1)?"Minus":"Plus")<<"\n";
    break;
  case GeomDetEnumerators::invalidLoc: // make gcc happy
    // edm::LogError("InvalidLoc") << "At " << __FILE__ << ", line " << __LINE__ << "\n";
    break;
  }
  os<< (void*)dl <<"\n";
  return os;
}

std::ostream& operator<<(std::ostream&os, const NavigationSchool::StateType & layers){
  for (NavigationSchool::StateType::const_iterator l = layers.begin(); l!=layers.end();++l)
    {
      std::vector<const DetLayer*> displayThose;

      os<<"####################\n"	 
	<<"Layer: \n"
	<<(*l)->detLayer();
      
      displayThose= (*l)->nextLayers(insideOut);
      if (displayThose.empty())
        {os<<"*** no INsideOUT connection ***\n";}
      else{
	os<<"*** INsideOUT CONNECTED TO ***\n";
	for(std::vector<const DetLayer*>::iterator nl =displayThose.begin();nl!=displayThose.end();++nl)
	  {os<<(*nl)<<"-----------------\n";}}

      displayThose = (*l)->nextLayers(outsideIn);
      if (displayThose.empty())
	{os<<"*** no OUTsideIN connection ***\n";}
      else{
	os<<"*** OUTsideIN CONNECTED TO ***\n";
	for(std::vector<const DetLayer*>::iterator nl =displayThose.begin();nl!=displayThose.end();++nl)
	  {os<<(*nl)<<"-----------------\n";}}
    }
  return os<<"\n";
}


std::ostream& operator<<(std::ostream&os, const NavigationSchool *nav){
  NavigationSchool::StateType layer=nav->navigableLayers();
  os<<layer;
  return os;}


// the analyzer itself
NavigationSchoolAnalyzer::NavigationSchoolAnalyzer(const edm::ParameterSet& iConfig) :theNavigationSchoolName(iConfig.getParameter<std::string>("navigationSchoolName")) {}

NavigationSchoolAnalyzer::~NavigationSchoolAnalyzer() {}

void NavigationSchoolAnalyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {}

void NavigationSchoolAnalyzer::beginRun(edm::Run & run, const edm::EventSetup& iSetup) {
  //get the navigation school
  edm::ESHandle<NavigationSchool> nav;
  iSetup.get<NavigationSchoolRecord>().get(theNavigationSchoolName, nav);
  edm::LogInfo("NavigationSchoolAnalyzer")<<"hello";
  edm::LogInfo("NavigationSchoolAnalyzer")<<"NavigationSchool display of: "<<theNavigationSchoolName<<"\n"<<nav.product();
}

void NavigationSchoolAnalyzer::endJob() {}

//define this as a plug-in
DEFINE_FWK_MODULE(NavigationSchoolAnalyzer);
