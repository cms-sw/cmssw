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

#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
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
  virtual void beginRun(edm::Run const& run, edm::EventSetup const&) override;
  virtual void analyze(const edm::Event&, const edm::EventSetup&);
  virtual void endJob() ;

  std::string theNavigationSchoolName;
  const TrackerTopology *tTopo;
  void print(std::ostream& os,const DetLayer* dl);
  void print(std::ostream&os, const NavigationSchool::StateType & layers);
  void print(std::ostream&os, const NavigationSchool *nav);

};

//navigation printouts
void NavigationSchoolAnalyzer::print(std::ostream& os,const DetLayer* dl){
  const std::vector<const GeomDet*>& bComponents = dl->basicComponents();

  if (bComponents.empty()){/* t'es pas dans la merde */;return;}

  const GeomDet* tag = bComponents.front();
  unsigned int LorW=0;
  unsigned int side=0;

  switch (dl->subDetector()){
  case GeomDetEnumerators::PixelBarrel :
    LorW = tTopo->pxbLayer(tag->geographicalId()); break;
  case GeomDetEnumerators::TIB :
    LorW = tTopo->tibLayer(tag->geographicalId()); break;
  case GeomDetEnumerators::TOB :
    LorW = tTopo->tobLayer(tag->geographicalId()); break;
  case GeomDetEnumerators::DT :
    LorW = DTChamberId(tag->geographicalId().rawId()).station(); break;
  case GeomDetEnumerators::RPCEndcap :
    side = (unsigned int)((RPCDetId(tag->geographicalId().rawId()).region()/2.+1)*2.);
  case GeomDetEnumerators::RPCBarrel :
    LorW = RPCDetId(tag->geographicalId().rawId()).station(); break;

  case GeomDetEnumerators::PixelEndcap :    
    LorW = tTopo->pxfDisk(tag->geographicalId());
    side = tTopo->pxfSide(tag->geographicalId());break;
  case GeomDetEnumerators::TID :
    LorW = tTopo->tidWheel(tag->geographicalId()); 
    side = tTopo->tidSide(tag->geographicalId());break;
  case GeomDetEnumerators::TEC :
    LorW = tTopo->tecWheel(tag->geographicalId());
    side = tTopo->tecSide(tag->geographicalId()); break;
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
  return;
}

void NavigationSchoolAnalyzer::print(std::ostream&os, const NavigationSchool::StateType & layers){
  for (NavigationSchool::StateType::const_iterator l = layers.begin(); l!=layers.end();++l)
    {
      std::vector<const DetLayer*> displayThose;

      os<<"####################\n"	 
	<<"Layer: \n";
      print(os,(*l)->detLayer());
      
      displayThose= (*l)->nextLayers(insideOut);
      if (displayThose.empty())
        {os<<"*** no INsideOUT connection ***\n";}
      else{
	os<<"*** INsideOUT CONNECTED TO ***\n";
	for(std::vector<const DetLayer*>::iterator nl =displayThose.begin();nl!=displayThose.end();++nl)
	  {print(os,*nl); os<<"-----------------\n";}}

      displayThose = (*l)->nextLayers(outsideIn);
      if (displayThose.empty())
	{os<<"*** no OUTsideIN connection ***\n";}
      else{
	os<<"*** OUTsideIN CONNECTED TO ***\n";
	for(std::vector<const DetLayer*>::iterator nl =displayThose.begin();nl!=displayThose.end();++nl)
	  {print(os,*nl); os<<"-----------------\n";}}
    }
  os<<"\n";
  return;
}


void NavigationSchoolAnalyzer::print(std::ostream&os, const NavigationSchool *nav){
  NavigationSchool::StateType layer=nav->navigableLayers();
  print(os,layer);
  return;
}


// the analyzer itself
NavigationSchoolAnalyzer::NavigationSchoolAnalyzer(const edm::ParameterSet& iConfig) :theNavigationSchoolName(iConfig.getParameter<std::string>("navigationSchoolName")) {}

NavigationSchoolAnalyzer::~NavigationSchoolAnalyzer() {}

void NavigationSchoolAnalyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {}

void NavigationSchoolAnalyzer::beginRun(edm::Run const& run, edm::EventSetup const& iSetup) {
  edm::ESHandle<TrackerTopology> tTopoHandle;
  iSetup.get<IdealGeometryRecord>().get(tTopoHandle);
  tTopo = tTopoHandle.product(); 

  //get the navigation school
  edm::ESHandle<NavigationSchool> nav;
  iSetup.get<NavigationSchoolRecord>().get(theNavigationSchoolName, nav);
  edm::LogInfo("NavigationSchoolAnalyzer")<<"hello";
  edm::LogInfo("NavigationSchoolAnalyzer")<<"NavigationSchool display of: " <<theNavigationSchoolName<<"\n";
  print (std::cout,nav.product());
}

void NavigationSchoolAnalyzer::endJob() {}

//define this as a plug-in
DEFINE_FWK_MODULE(NavigationSchoolAnalyzer);
