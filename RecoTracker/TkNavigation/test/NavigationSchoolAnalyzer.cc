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
#include "Geometry/Records/interface/TrackerTopologyRcd.h"
#include "DataFormats/MuonDetId/interface/RPCDetId.h"
#include "DataFormats/MuonDetId/interface/CSCDetId.h"
#include "DataFormats/MuonDetId/interface/DTChamberId.h"

#include "TrackingTools/DetLayers/interface/NavigationSchool.h"
#include "RecoTracker/Record/interface/NavigationSchoolRecord.h"

#include "TrackingTools/DetLayers/interface/DetLayer.h"
// #include "TrackingTools/DetLayers/interface/NavigationSetter.h"

// class definition
class NavigationSchoolAnalyzer : public edm::EDAnalyzer {
public:
  explicit NavigationSchoolAnalyzer(const edm::ParameterSet&);
  ~NavigationSchoolAnalyzer();

private:
  virtual void beginRun(edm::Run const& run, const edm::EventSetup&) override;
  virtual void analyze(const edm::Event&, const edm::EventSetup&) override;

  std::string theNavigationSchoolName;
  const TrackerTopology* tTopo;
  void print(std::ostream& os, const DetLayer* dl);
  void print(std::ostream& os, const NavigationSchool::StateType& layers);
  void print(std::ostream& os, const NavigationSchool* nav);
  void printUsingGeom(std::ostream& os, const NavigationSchool& nav);
};

//navigation printouts
void NavigationSchoolAnalyzer::print(std::ostream& os, const DetLayer* dl) {
  const std::vector<const GeomDet*>& bComponents = dl->basicComponents();

  if (bComponents.empty()) { /* t'es pas dans la merde */
    return;
  }

  const GeomDet* tag = bComponents.front();
  unsigned int LorW = 0;
  unsigned int side = 0;

  if (GeomDetEnumerators::isTracker(dl->subDetector())) {
    LorW = tTopo->layer(tag->geographicalId());
    side = tTopo->side(tag->geographicalId());
  } else {
    switch (dl->subDetector()) {
      case GeomDetEnumerators::DT:
        LorW = DTChamberId(tag->geographicalId().rawId()).station();
        break;
      case GeomDetEnumerators::RPCEndcap:
        side = (unsigned int)((RPCDetId(tag->geographicalId().rawId()).region() / 2. + 1) * 2.);
        [[fallthrough]];
      case GeomDetEnumerators::RPCBarrel:
        LorW = RPCDetId(tag->geographicalId().rawId()).station();
        break;

      case GeomDetEnumerators::CSC:
        LorW = CSCDetId(tag->geographicalId().rawId()).layer();
        side = CSCDetId(tag->geographicalId().rawId()).endcap();
        break;
      case GeomDetEnumerators::invalidDet:  // make gcc happy
        // edm::LogError("InvalidDet") << "At " << __FILE__ << ", line " << __LINE__ << "\n";
        break;
      default:
        break;
    }
  }

  switch (dl->location()) {
    case GeomDetEnumerators::barrel:
      os << "barrel subDetector: " << dl->subDetector() << "\n"
         << "layer: " << LorW << "\n";
      break;
    case GeomDetEnumerators::endcap:
      os << "endcap subDetector: " << dl->subDetector() << "\n"
         << "wheel: " << LorW << "\n"
         << "side: " << ((side == 1) ? "Minus" : "Plus") << "\n";
      break;
    case GeomDetEnumerators::invalidLoc:  // make gcc happy
      // edm::LogError("InvalidLoc") << "At " << __FILE__ << ", line " << __LINE__ << "\n";
      break;
  }
  os << (void*)dl << "\n";
  return;
}

void NavigationSchoolAnalyzer::print(std::ostream& os, const NavigationSchool::StateType& layers) {
  for (NavigationSchool::StateType::const_iterator l = layers.begin(); l != layers.end(); ++l) {
    std::vector<const DetLayer*> displayThose;

    os << "####################\n"
       << "Layer: \n";
    print(os, (*l)->detLayer());

    displayThose = (*l)->nextLayers(insideOut);
    if (displayThose.empty()) {
      os << "*** no INsideOUT connection ***\n";
    } else {
      os << "*** INsideOUT CONNECTED TO ***\n";
      for (std::vector<const DetLayer*>::iterator nl = displayThose.begin(); nl != displayThose.end(); ++nl) {
        print(os, *nl);
        os << "-----------------\n";
      }
    }

    displayThose = (*l)->nextLayers(outsideIn);
    if (displayThose.empty()) {
      os << "*** no OUTsideIN connection ***\n";
    } else {
      os << "*** OUTsideIN CONNECTED TO ***\n";
      for (std::vector<const DetLayer*>::iterator nl = displayThose.begin(); nl != displayThose.end(); ++nl) {
        print(os, *nl);
        os << "-----------------\n";
      }
    }
  }
  os << "\n";
  return;
}

void NavigationSchoolAnalyzer::print(std::ostream& os, const NavigationSchool* nav) {
  NavigationSchool::StateType layer = const_cast<NavigationSchool*>(nav)->navigableLayers();
  print(os, layer);
  return;
}

void NavigationSchoolAnalyzer::printUsingGeom(std::ostream& os, const NavigationSchool& nav) {
  auto dls = nav.allLayersInSystem();  // ok let's' keep it for debug
  for (auto dl : dls) {
    os << "####################\n"
       << "Layer: \n";
    print(os, dl);

    auto displayThose = nav.nextLayers(*dl, insideOut);
    if (displayThose.empty()) {
      os << "*** no INsideOUT connection ***\n";
    } else {
      os << "*** INsideOUT CONNECTED TO ***\n";
      for (std::vector<const DetLayer*>::iterator nl = displayThose.begin(); nl != displayThose.end(); ++nl) {
        print(os, *nl);
        os << "-----------------\n";
      }
    }

    displayThose = nav.nextLayers(*dl, outsideIn);
    if (displayThose.empty()) {
      os << "*** no OUTsideIN connection ***\n";
    } else {
      os << "*** OUTsideIN CONNECTED TO ***\n";
      for (std::vector<const DetLayer*>::iterator nl = displayThose.begin(); nl != displayThose.end(); ++nl) {
        print(os, *nl);
        os << "-----------------\n";
      }
    }
  }
  os << "\n";
}

void printOldStyle(std::ostream& os, const NavigationSchool& nav) {
  /*
  NavigationSetter setter(nav);

  auto dls = nav.allLayersInSystem(); // ok let's' keep it for debug
  for ( auto dl : dls) {
     os<<"####################\n"	 
	<< "Layer: \n"
       << (dl);
      
     auto displayThose=  dl->nextLayers(insideOut);
      if (displayThose.empty())
        {os<<"*** no INsideOUT connection ***\n";}
      else{
	os<<"*** INsideOUT CONNECTED TO ***\n";
	for(std::vector<const DetLayer*>::iterator nl =displayThose.begin();nl!=displayThose.end();++nl)
	  {os<<(*nl)<<"-----------------\n";}}

      displayThose = dl->nextLayers(outsideIn);
      if (displayThose.empty())
	{os<<"*** no OUTsideIN connection ***\n";}
      else{
	os<<"*** OUTsideIN CONNECTED TO ***\n";
	for(std::vector<const DetLayer*>::iterator nl =displayThose.begin();nl!=displayThose.end();++nl)
	  {os<<(*nl)<<"-----------------\n";}}
  }
  os<<"\n";
*/
}

// the analyzer itself
NavigationSchoolAnalyzer::NavigationSchoolAnalyzer(const edm::ParameterSet& iConfig)
    : theNavigationSchoolName(iConfig.getParameter<std::string>("navigationSchoolName")) {}

NavigationSchoolAnalyzer::~NavigationSchoolAnalyzer() {}

#include <sstream>
#include <fstream>

void NavigationSchoolAnalyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  edm::ESHandle<TrackerTopology> tTopoHandle;
  iSetup.get<TrackerTopologyRcd>().get(tTopoHandle);
  tTopo = tTopoHandle.product();

  std::ostringstream byNav;
  std::ostringstream byGeom;
  std::ostringstream oldStyle;

  std::ofstream ByNavFile(theNavigationSchoolName + "_ByNav.log");
  std::ofstream ByGeomFile(theNavigationSchoolName + "_ByGeom.log");
  std::ofstream oldStyleFile(theNavigationSchoolName + "_oldStyle.log");

  //get the navigation school
  edm::ESHandle<NavigationSchool> nav;
  iSetup.get<NavigationSchoolRecord>().get(theNavigationSchoolName, nav);
  print(byNav, nav.product());
  printUsingGeom(byGeom, *nav.product());
  printOldStyle(oldStyle, *nav.product());

  ByNavFile << byNav.str() << std::endl;
  ByGeomFile << byGeom.str() << std::endl;
  oldStyleFile << oldStyle.str() << std::endl;

  if (oldStyle.str() != byGeom.str())
    std::cout << "Error: Navigation by Geom is not consistent with old Style Navigation in " << theNavigationSchoolName
              << "\n"
              << std::endl;

  // NavigationSetter setter(*nav.product());
  std::cout << "NavigationSchoolAnalyzer "
            << "hello at event" << std::endl;
  std::cout << "NavigationSchoolAnalyzer "
            << "NavigationSchool display of: " << theNavigationSchoolName << "\n"
            << byNav.str() << std::endl;

  std::cout << "\n\nNavigationSchoolAnalyzer "
            << "NavigationSchool display using Geometry" << std::endl;
  std::cout << byGeom.str() << std::endl;
}

void NavigationSchoolAnalyzer::beginRun(edm::Run const& run, const edm::EventSetup& iSetup) {
  edm::ESHandle<TrackerTopology> tTopoHandle;
  iSetup.get<TrackerTopologyRcd>().get(tTopoHandle);
  tTopo = tTopoHandle.product();

  //get the navigation school
  edm::ESHandle<NavigationSchool> nav;
  iSetup.get<NavigationSchoolRecord>().get(theNavigationSchoolName, nav);
  // NavigationSetter setter(*nav.product());
  edm::LogInfo("NavigationSchoolAnalyzer") << "NavigationSchool display of: " << theNavigationSchoolName << "\n";
  print(std::cout, nav.product());
}

//define this as a plug-in
DEFINE_FWK_MODULE(NavigationSchoolAnalyzer);
