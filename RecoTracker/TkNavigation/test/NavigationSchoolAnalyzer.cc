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
#include "DataFormats/ForwardDetId/interface/BTLDetId.h"
#include "DataFormats/ForwardDetId/interface/ETLDetId.h"
#include "DataFormats/MuonDetId/interface/CSCDetId.h"
#include "DataFormats/MuonDetId/interface/DTChamberId.h"
#include "DataFormats/MuonDetId/interface/RPCDetId.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "Geometry/Records/interface/TrackerTopologyRcd.h"
#include "RecoTracker/Record/interface/NavigationSchoolRecord.h"
#include "TrackingTools/DetLayers/interface/DetLayer.h"
#include "TrackingTools/DetLayers/interface/NavigationSchool.h"

// class definition
class NavigationSchoolAnalyzer : public edm::one::EDAnalyzer<edm::one::WatchRuns> {
public:
  explicit NavigationSchoolAnalyzer(const edm::ParameterSet&);
  ~NavigationSchoolAnalyzer() override = default;

private:
  virtual void beginRun(edm::Run const& run, const edm::EventSetup&) override;
  virtual void analyze(const edm::Event&, const edm::EventSetup&) override;
  virtual void endRun(edm::Run const& run, const edm::EventSetup&) override{};

  const std::string theNavigationSchoolName_;
  const edm::ESGetToken<NavigationSchool, NavigationSchoolRecord> navSchoolToken_;
  const edm::ESGetToken<TrackerTopology, TrackerTopologyRcd> tTopoTokenBR_;
  const edm::ESGetToken<NavigationSchool, NavigationSchoolRecord> navSchoolTokenBR_;

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
  } else if (dl->subDetector() == GeomDetEnumerators::TimingEndcap) {
    ETLDetId id(dl->basicComponents().front()->geographicalId().rawId());
    LorW = id.nDisc();
    //The MTD side returns 0 for the negative side and 1 for the positive side
    //In order to be comp
    side = id.mtdSide() + 1;
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
    : theNavigationSchoolName_(iConfig.getParameter<std::string>("navigationSchoolName")),
      navSchoolToken_(esConsumes(edm::ESInputTag("", theNavigationSchoolName_))),
      tTopoTokenBR_(esConsumes<edm::Transition::BeginRun>()),
      navSchoolTokenBR_(esConsumes<edm::Transition::BeginRun>(edm::ESInputTag("", theNavigationSchoolName_))) {}

#include <sstream>
#include <fstream>

void NavigationSchoolAnalyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  std::ostringstream byNav;
  std::ostringstream byGeom;
  std::ostringstream oldStyle;

  std::ofstream ByNavFile(theNavigationSchoolName_ + "_ByNav.log");
  std::ofstream ByGeomFile(theNavigationSchoolName_ + "_ByGeom.log");
  std::ofstream oldStyleFile(theNavigationSchoolName_ + "_oldStyle.log");

  //get the navigation school
  const NavigationSchool* nav = &iSetup.getData(navSchoolToken_);
  print(byNav, nav);
  printUsingGeom(byGeom, *nav);
  printOldStyle(oldStyle, *nav);

  ByNavFile << byNav.str() << std::endl;
  ByGeomFile << byGeom.str() << std::endl;
  oldStyleFile << oldStyle.str() << std::endl;

  if (oldStyle.str() != byGeom.str())
    edm::LogPrint("NavigationSchoolAnalyzer")
        << "Error: Navigation by Geom is not consistent with old Style Navigation in " << theNavigationSchoolName_
        << "\n";

  // NavigationSetter setter(*nav);
  edm::LogPrint("NavigationSchoolAnalyzer") << "NavigationSchoolAnalyzer "
                                            << "hello at event";
  edm::LogPrint("NavigationSchoolAnalyzer") << "NavigationSchoolAnalyzer "
                                            << "NavigationSchool display of: " << theNavigationSchoolName_ << "\n"
                                            << byNav.str();

  edm::LogPrint("NavigationSchoolAnalyzer") << "\n\nNavigationSchoolAnalyzer "
                                            << "NavigationSchool display using Geometry";
  edm::LogPrint("NavigationSchoolAnalyzer") << byGeom.str();
}

void NavigationSchoolAnalyzer::beginRun(edm::Run const& run, const edm::EventSetup& iSetup) {
  tTopo = &iSetup.getData(tTopoTokenBR_);

  //get the navigation school
  const NavigationSchool* nav = &iSetup.getData(navSchoolTokenBR_);
  // NavigationSetter setter(*nav.product());
  edm::LogInfo("NavigationSchoolAnalyzer") << "NavigationSchool display of: " << theNavigationSchoolName_ << "\n";
  print(std::cout, nav);
}

//define this as a plug-in
DEFINE_FWK_MODULE(NavigationSchoolAnalyzer);
