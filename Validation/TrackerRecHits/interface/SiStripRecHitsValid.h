#ifndef SiStripRecHitsValid_h
#define SiStripRecHitsValid_h

/* \class SiStripRecHitsValid
 *
 * Analyzer to validate RecHits in the Strip tracker
 *
 * \author Patrizia Azzi, INFN PD
 *
 * \version   1st version May 2006
 *
 ************************************************************/

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
//only mine
#include "FWCore/Framework/interface/Handle.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/Common/interface/EDProduct.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

//DQM services for histogram
#include "DQMServices/Core/interface/DaqMonitorBEInterface.h"
#include "DQMServices/Daemon/interface/MonitorDaemon.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

//--- for SimHit 
#include "SimDataFormats/TrackingHit/interface/PSimHit.h" 
#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h" 
#include "SimTracker/TrackerHitAssociation/interface/TrackerHitAssociator.h" 

#include "Geometry/CommonTopologies/interface/PixelTopology.h"
#include "Geometry/CommonTopologies/interface/StripTopology.h"
#include "Geometry/TrackerGeometryBuilder/interface/StripGeomDetUnit.h"
#include "Geometry/CommonDetUnit/interface/GeomDetType.h" 
#include "Geometry/CommonDetUnit/interface/GeomDetUnit.h" 
#include "Geometry/TrackerGeometryBuilder/interface/GluedGeomDet.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/TrackerNumberingBuilder/interface/GeometricDet.h"
#include "Geometry/TrackerGeometryBuilder/interface/PixelGeomDetType.h"
#include "Geometry/TrackerGeometryBuilder/interface/StripGeomDetType.h"

#include <string>


using namespace std;
using namespace edm;


class SiStripRecHitsValid : public edm::EDAnalyzer {

 public:
  
  SiStripRecHitsValid(const edm::ParameterSet& conf);
  
  ~SiStripRecHitsValid();

 protected:

  virtual void analyze(const edm::Event& e, const edm::EventSetup& c);
  void beginJob(const EventSetup& c);
  void endJob();
  
 private: 
  //Back-End Interface
  DaqMonitorBEInterface* dbe_;
  string outputFile_;
  //SiStripRecHit2DLocalPos
  MonitorElement* meNstpRphiTIB[4];
  MonitorElement* meAdcRphiTIB[4];
  MonitorElement* mePosxRphiTIB[4];
  MonitorElement* meErrxRphiTIB[4];
  MonitorElement* meResRphiTIB[4];
  MonitorElement* meNstpSasTIB[4];
  MonitorElement* meAdcSasTIB[4];
  MonitorElement* mePosxSasTIB[4];
  MonitorElement* meErrxSasTIB[4];
  MonitorElement* meResSasTIB[4];

  //SiStripRecHit2DMatchedLocalPos
  MonitorElement* mePosxMatchedTIB[2];
  MonitorElement* mePosyMatchedTIB[2];
  MonitorElement* meErrxMatchedTIB[2];
  MonitorElement* meErryMatchedTIB[2];
  MonitorElement* meResxMatchedTIB[2];
  MonitorElement* meResyMatchedTIB[2];

  std::vector<PSimHit> matched;
  std::pair<LocalPoint,LocalVector> projectHit( const PSimHit& hit, const StripGeomDetUnit* stripDet,
							const BoundPlane& plane);
  edm::ParameterSet conf_;
  const StripTopology* topol;
  std::vector<PSimHit> theStripHits;
  typedef std::map<unsigned int, std::vector<PSimHit> > simhit_map;
  typedef simhit_map::iterator simhit_map_iterator;
  simhit_map SimHitMap;


  static const int MAXHIT = 100;
  float simhitx[MAXHIT];
  float simhity[MAXHIT];
  float simhitz[MAXHIT];
  float simhitphi[MAXHIT];
  float simhiteta[MAXHIT];
  float rechitrphix[MAXHIT];
  float rechitrphierrx[MAXHIT];
  float rechitrphiy[MAXHIT];
  float rechitrphiz[MAXHIT];
  float rechitrphiphi[MAXHIT];
  float rechitrphires[MAXHIT];
  int   clusizrphi[MAXHIT];
  float cluchgrphi[MAXHIT];
  float rechitsasx[MAXHIT];
  float rechitsaserrx[MAXHIT];
  float rechitsasy[MAXHIT];
  float rechitsasz[MAXHIT];
  float rechitsasphi[MAXHIT];
  float rechitsasres[MAXHIT];
  int   clusizsas[MAXHIT];
  float cluchgsas[MAXHIT];
  float rechitmatchedx[MAXHIT];
  float rechitmatchedy[MAXHIT];
  float rechitmatchedz[MAXHIT];
  float rechitmatchederrxx[MAXHIT];
  float rechitmatchederrxy[MAXHIT];
  float rechitmatchederryy[MAXHIT];
  float rechitmatchedphi[MAXHIT];
  float rechitmatchedresx[MAXHIT];
  float rechitmatchedresy[MAXHIT];

};

#endif
