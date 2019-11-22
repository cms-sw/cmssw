// -*- C++ -*-
//
// Package:    PixelDigisTest
// Class:      PixelDigisTest
//
/**\class PixelDigisTest PixelDigisTest.cc 

 Description: Test pixel digis. 
 Barrel & Forward digis. Uses root histos.
 Adopted for the new simLinks. 
 Added the online detector index. d.k. 11/09
 Works with CMSSW_7
 New detector ID.
 Modified to use "byToken"

*/
//
// Original Author:  d.k.
//         Created:  Jan CET 2006
//
//
// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "DataFormats/Common/interface/Handle.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Utilities/interface/InputTag.h"

// my includes
//#include "Geometry/CommonDetUnit/interface/GeomDet.h"

#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"

#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/CommonDetUnit/interface/PixelGeomDetUnit.h"
#include "Geometry/CommonDetUnit/interface/PixelGeomDetType.h"

#include "DataFormats/SiPixelDetId/interface/PixelSubdetector.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "Geometry/Records/interface/TrackerTopologyRcd.h"
#include "DataFormats/SiPixelDetId/interface/PixelBarrelName.h"

// data formats
#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/SiPixelDigi/interface/PixelDigi.h"
//#include "DataFormats/SiPixelDigi/interface/PixelDigiCollection.h"
#include "DataFormats/DetId/interface/DetId.h"

#include "SimDataFormats/TrackerDigiSimLink/interface/PixelDigiSimLink.h"

// For the big pixel recongnition
#include "Geometry/TrackerGeometryBuilder/interface/RectangularPixelTopology.h"

// for simulated Tracker hits
//#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"

// For L1
#include "L1Trigger/GlobalTriggerAnalyzer/interface/L1GtUtils.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutSetupFwd.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutSetup.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutRecord.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerObjectMapRecord.h"

// For HLT
#include "DataFormats/HLTReco/interface/TriggerEvent.h"
#include "DataFormats/HLTReco/interface/TriggerTypeDefs.h"
#include "DataFormats/Common/interface/TriggerResults.h"
#include "HLTrigger/HLTcore/interface/HLTConfigProvider.h"
#include "FWCore/Common/interface/TriggerNames.h"

// To use root histos
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"

// For ROOT
#include <TROOT.h>
#include <TChain.h>
#include <TFile.h>
#include <TF1.h>
#include <TH2F.h>
#include <TH1F.h>

#define HISTOS
#define L1
#define HLT

using namespace std;

// Enable this to look at simlinks (link simhit->digis)
// Can be used only with simulated data.
//#define USE_SIM_LINKS

//
// class decleration
//

class PixelDigisTest : public edm::EDAnalyzer {
public:
  explicit PixelDigisTest(const edm::ParameterSet &);
  ~PixelDigisTest() override;
  void beginJob() override;
  void analyze(const edm::Event &, const edm::EventSetup &) override;
  void endJob() override;

private:
  // ----------member data ---------------------------
  bool PRINT;
  edm::EDGetTokenT<edm::DetSetVector<PixelDigi>> tPixelDigi;
#ifdef USE_SIM_LINKS
  edm::EDGetTokenT<edm::DetSetVector<PixelDigiSimLink>> tPixelDigiSimLink;
#endif

#ifdef HISTOS

  //TFile* hFile;
  TH1F *hdetunit;
  TH1F *heloss1, *heloss2, *heloss3;
  TH1F *hneloss1, *hneloss2, *hneloss3;
  TH1F *helossF1, *helossF2;
  TH1F *hpixid, *hpixsubid, *hlayerid, *hshellid, *hsectorid, *hladder1id, *hladder2id, *hladder3id, *hz1id, *hz2id,
      *hz3id;
  TH1F *hcols1, *hcols2, *hcols3, *hrows1, *hrows2, *hrows3;
  TH1F *hcolsF1, *hcolsF2, *hcolsF3, *hrowsF1, *hrowsF2, *hrowsF3;
  TH1F *hdigisPerDet1, *hdigisPerDet2, *hdigisPerDet3;
  TH1F *hdigisPerLay1, *hdigisPerLay2, *hdigisPerLay3;
  TH1F *hdetsPerLay1, *hdetsPerLay2, *hdetsPerLay3;
  TH1F *hdigisPerDetF1, *hdigisPerDetF2, *hdigisPerDetF3;
  TH1F *hdigisPerLayF1, *hdigisPerLayF2, *hdigisPerLayF3;
  TH1F *hdetsPerLayF1, *hdetsPerLayF2, *hdetsPerLayF3;
  TH1F *hdetr, *hdetz, *hdetrF, *hdetzF;
  TH1F *hcolsB, *hrowsB, *hcolsF, *hrowsF;
  TH1F *hcols1big, *hrows1big, *heloss1bigx, *heloss1bigy;
  TH1F *hsimlinks, *hfract;
  TH1F *hblade1, *hblade2;

  //TH2F *htest, *htest2;
  TH2F *hdetMap3, *hdetMap2, *hdetMap1, *hpixMap1, *hpixMap2, *hpixMap3, *hpixMapNoise;

  TH1F *hevent, *hlumi, *horbit, *hbx0, *hlumi0, *hlumi1, *hbx1, *hbx2, *hbx3, *hbx4, *hbx5, *hbx6;
  TH1F *hdets, *hdigis, *hdigis1, *hdigis2, *hdigis3, *hdigis4, *hdigis5;

#endif

  edm::InputTag src_;
};

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
PixelDigisTest::PixelDigisTest(const edm::ParameterSet &iConfig) {
  //We put this here for the moment since there is no better place
  //edm::Service<MonitorDaemon> daemon;
  //daemon.operator->();

  PRINT = iConfig.getUntrackedParameter<bool>("Verbosity", false);
  src_ = iConfig.getParameter<edm::InputTag>("src");
  tPixelDigi = consumes<edm::DetSetVector<PixelDigi>>(src_);
#ifdef USE_SIM_LINKS
  tPixelDigiSimLink = consumes<edm::DetSetVector<PixelDigiSimLink>>(src_);
#endif

  cout << " Construct PixelDigisTest " << endl;
}

PixelDigisTest::~PixelDigisTest() {
  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)
  cout << " Destroy PixelDigisTest " << endl;
}

//
// member functions
//
// ------------ method called at the begining   ------------
void PixelDigisTest::beginJob() {
  using namespace edm;
  cout << "Initialize PixelDigisTest " << endl;

#ifdef HISTOS
  edm::Service<TFileService> fs;

  // Histos go to a subdirectory "PixRecHits")
  //TFileDirectory subDir = fs->mkdir( "mySubDirectory" );
  //TFileDirectory subSubDir = subDir.mkdir( "mySubSubDirectory" );

  // put here whatever you want to do at the beginning of the job
  //hFile = new TFile ( "histo.root", "RECREATE" );

  hdetunit = fs->make<TH1F>("hdetunit", "Det unit", 1000, 302000000., 302300000.);
  hpixid = fs->make<TH1F>("hpixid", "Pix det id", 10, 0., 10.);
  hpixsubid = fs->make<TH1F>("hpixsubid", "Pix Barrel id", 10, 0., 10.);
  hlayerid = fs->make<TH1F>("hlayerid", "Pix layer id", 10, 0., 10.);
  hsectorid = fs->make<TH1F>("hsectorid", "Pix sector ", 10, 0., 10.);
  hshellid = fs->make<TH1F>("hshellid", "Shell", 5, 0., 5.);
  hladder1id = fs->make<TH1F>("hladder1id", "Ladder L1 id", 23, -11.5, 11.5);
  hladder2id = fs->make<TH1F>("hladder2id", "Ladder L2 id", 35, -17.5, 17.5);
  hladder3id = fs->make<TH1F>("hladder3id", "Ladder L3 id", 47, -23.5, 23.5);
  hz1id = fs->make<TH1F>("hz1id", "Z-index id L1", 11, -5.5, 5.5);
  hz2id = fs->make<TH1F>("hz2id", "Z-index id L2", 11, -5.5, 5.5);
  hz3id = fs->make<TH1F>("hz3id", "Z-index id L3", 11, -5.5, 5.5);

  hdigisPerDet1 = fs->make<TH1F>("hdigisPerDet1", "Digis per det l1", 200, -0.5, 199.5);
  hdigisPerDet2 = fs->make<TH1F>("hdigisPerDet2", "Digis per det l2", 200, -0.5, 199.5);
  hdigisPerDet3 = fs->make<TH1F>("hdigisPerDet3", "Digis per det l3", 200, -0.5, 199.5);
  hdigisPerLay1 = fs->make<TH1F>("hdigisPerLay1", "Digis per layer l1", 200, -0.5, 199.5);
  hdigisPerLay2 = fs->make<TH1F>("hdigisPerLay2", "Digis per layer l2", 200, -0.5, 199.5);
  hdigisPerLay3 = fs->make<TH1F>("hdigisPerLay3", "Digis per layer l3", 200, -0.5, 199.5);
  hdetsPerLay1 = fs->make<TH1F>("hdetsPerLay1", "Full dets per layer l1", 161, -0.5, 160.5);
  hdetsPerLay3 = fs->make<TH1F>("hdetsPerLay3", "Full dets per layer l3", 353, -0.5, 352.5);
  hdetsPerLay2 = fs->make<TH1F>("hdetsPerLay2", "Full dets per layer l2", 257, -0.5, 256.5);

  hdigisPerDetF1 = fs->make<TH1F>("hdigisPerDetF1", "Digis per det d1", 200, -0.5, 199.5);
  hdigisPerDetF2 = fs->make<TH1F>("hdigisPerDetF2", "Digis per det d2", 200, -0.5, 199.5);
  hdigisPerLayF1 = fs->make<TH1F>("hdigisPerLayF1", "Digis per layer d1", 2000, -0.5, 1999.5);
  hdigisPerLayF2 = fs->make<TH1F>("hdigisPerLayF2", "Digis per layer d2", 2000, -0.5, 1999.5);
  hdetsPerLayF1 = fs->make<TH1F>("hdetsPerLayF1", "Full dets per layer d1", 161, -0.5, 160.5);
  hdetsPerLayF2 = fs->make<TH1F>("hdetsPerLayF2", "Full dets per layer d2", 257, -0.5, 256.5);

  heloss1 = fs->make<TH1F>("heloss1", "Pix charge l1", 256, 0., 256.);
  heloss2 = fs->make<TH1F>("heloss2", "Pix charge l2", 256, 0., 256.);
  heloss3 = fs->make<TH1F>("heloss3", "Pix charge l3", 256, 0., 256.);
  hneloss1 = fs->make<TH1F>("hneloss1", "Pix noise charge l1", 256, 0., 256.);
  hneloss2 = fs->make<TH1F>("hneloss2", "Pix noise charge l2", 256, 0., 256.);
  hneloss3 = fs->make<TH1F>("hneloss3", "Pix noise charge l3", 256, 0., 256.);
  heloss1bigx = fs->make<TH1F>("heloss1bigx", "L1 big-x pix", 256, 0., 256.);
  heloss1bigy = fs->make<TH1F>("heloss1bigy", "L1 big-y pix", 256, 0., 256.);

  hcols1 = fs->make<TH1F>("hcols1", "Layer 1 cols", 500, -1.5, 498.5);
  hcols2 = fs->make<TH1F>("hcols2", "Layer 2 cols", 500, -1.5, 498.5);
  hcols3 = fs->make<TH1F>("hcols3", "Layer 3 cols", 500, -1.5, 498.5);
  hcols1big = fs->make<TH1F>("hcols1big", "Layer 1 big cols", 500, -1.5, 498.5);

  hrows1 = fs->make<TH1F>("hrows1", "Layer 1 rows", 200, -1.5, 198.5);
  hrows2 = fs->make<TH1F>("hrows2", "Layer 2 rows", 200, -1.5, 198.5);
  hrows3 = fs->make<TH1F>("hrows3", "layer 3 rows", 200, -1.5, 198.5);
  hrows1big = fs->make<TH1F>("hrows1big", "Layer 1 big rows", 200, -1.5, 198.5);

  hblade1 = fs->make<TH1F>("hblade1", "blade num, disk1", 24, 0., 24.);
  hblade2 = fs->make<TH1F>("hblade2", "blade num, disk2", 24, 0., 24.);

  helossF1 = fs->make<TH1F>("helossF1", "Pix charge d1", 100, 0., 300.);
  helossF2 = fs->make<TH1F>("helossF2", "Pix charge d2", 100, 0., 300.);
  hcolsF1 = fs->make<TH1F>("hcolsF1", "Disk 1 cols", 500, -1.5, 498.5);
  hcolsF2 = fs->make<TH1F>("hcolsF2", "Disk 2 cols", 500, -1.5, 498.5);
  hrowsF1 = fs->make<TH1F>("hrowsF1", "Disk 1 rows", 200, -1.5, 198.5);
  hrowsF2 = fs->make<TH1F>("hrowsF2", "Disk 2 rows", 200, -1.5, 198.5);

  hdetr = fs->make<TH1F>("hdetr", "det r", 150, 0., 15.);
  hdetz = fs->make<TH1F>("hdetz", "det z", 520, -26., 26.);
  hdetrF = fs->make<TH1F>("hdetrF", "det r", 150, 0., 15.);
  hdetzF = fs->make<TH1F>("hdetzF", "det z", 700, -70., 70.);

  hcolsB = fs->make<TH1F>("hcolsB", "cols per bar det", 450, 0., 450.);
  hrowsB = fs->make<TH1F>("hrowsB", "rows per bar det", 200, 0., 200.);
  hcolsF = fs->make<TH1F>("hcolsF", "cols per for det", 300, 0., 300.);
  hrowsF = fs->make<TH1F>("hrowsF", "rows per for det", 200, 0., 200.);

  hsimlinks = fs->make<TH1F>("hsimlinks", " track ids", 200, 0., 200.);
  hfract = fs->make<TH1F>("hfract", " track rractions", 100, 0., 1.);
  //                                             mod      ladder
  hdetMap1 = fs->make<TH2F>("hdetMap1", " ", 9, -4.5, 4.5, 21, -10.5, 10.5);
  hdetMap1->SetOption("colz");
  hdetMap2 = fs->make<TH2F>("hdetMap2", " ", 9, -4.5, 4.5, 33, -16.5, 16.5);
  hdetMap2->SetOption("colz");
  hdetMap3 = fs->make<TH2F>("hdetMap3", " ", 9, -4.5, 4.5, 45, -22.5, 22.5);
  hdetMap3->SetOption("colz");
  hpixMap1 = fs->make<TH2F>("hpixMap1", " ", 416, 0., 416., 160, 0., 160.);
  hpixMap1->SetOption("colz");
  hpixMap2 = fs->make<TH2F>("hpixMap2", " ", 416, 0., 416., 160, 0., 160.);
  hpixMap2->SetOption("colz");
  hpixMap3 = fs->make<TH2F>("hpixMap3", " ", 416, 0., 416., 160, 0., 160.);
  hpixMap3->SetOption("colz");
  hpixMapNoise = fs->make<TH2F>("hpixMapNoise", " ", 416, 0., 416., 160, 0., 160.);
  hpixMapNoise->SetOption("colz");

  //htest = fs->make<TH2F>("htest"," ",10,0.,10.,20,0.,20.);
  //htest2 = fs->make<TH2F>("htest2"," ",10,0.,10.,300,0.,300.);
  //htest->SetOption("colz");
  //htest2->SetOption("colz");

  hevent = fs->make<TH1F>("hevent", "event", 1000, 0, 10000000.);
  horbit = fs->make<TH1F>("horbit", "orbit", 100, 0, 100000000.);

  hlumi1 = fs->make<TH1F>("hlumi1", "lumi", 2000, 0, 2000.);
  hlumi0 = fs->make<TH1F>("hlumi0", "lumi", 2000, 0, 2000.);

  hbx6 = fs->make<TH1F>("hbx6", "bx", 4000, 0, 4000.);
  hbx5 = fs->make<TH1F>("hbx5", "bx", 4000, 0, 4000.);
  hbx4 = fs->make<TH1F>("hbx4", "bx", 4000, 0, 4000.);
  hbx3 = fs->make<TH1F>("hbx3", "bx", 4000, 0, 4000.);
  hbx2 = fs->make<TH1F>("hbx2", "bx", 4000, 0, 4000.);
  hbx1 = fs->make<TH1F>("hbx1", "bx", 4000, 0, 4000.);
  hbx0 = fs->make<TH1F>("hbx0", "bx", 4000, 0, 4000.);

  hdets = fs->make<TH1F>("hdets", "Dets with hits", 2000, -0.5, 1999.5);
  hdigis = fs->make<TH1F>("hdigis", "All Digis", 2000, -0.5, 1999.5);
  hdigis1 = fs->make<TH1F>("hdigis1", "All Digis for full events", 2000, -0.5, 1999.5);
  hdigis2 = fs->make<TH1F>("hdigis2", "BPix Digis", 2000, -0.5, 1999.5);
  hdigis3 = fs->make<TH1F>("hdigis3", "Fpix Digis", 2000, -0.5, 1999.5);
  hdigis4 = fs->make<TH1F>("hdigis4", "All Digis - on bunch", 2000, -0.5, 1999.5);
  hdigis5 = fs->make<TH1F>("hdigis5", "All Digis - off bunch ", 2000, -0.5, 1999.5);
#endif
}

// ------------ method called to produce the data  ------------
void PixelDigisTest::analyze(const edm::Event &iEvent, const edm::EventSetup &iSetup) {
  //Retrieve tracker topology from geometry
  edm::ESHandle<TrackerTopology> tTopo;
  iSetup.get<TrackerTopologyRcd>().get(tTopo);

  using namespace edm;
  if (PRINT)
    cout << " Analyze PixelDigisTest " << endl;

  //  int run       = iEvent.id().run();
  int event = iEvent.id().event();
  int lumiBlock = iEvent.luminosityBlock();
  int bx = iEvent.bunchCrossing();
  int orbit = iEvent.orbitNumber();

  hbx0->Fill(float(bx));
  hlumi0->Fill(float(lumiBlock));

  // eliminate bunches with beam
  bool bunch = false;
  if (bx == 410 || bx == 460 || bx == 510)
    bunch = true;
  else if (bx == 560 || bx == 610 || bx == 660)
    bunch = true;
  else if (bx == 1292 || bx == 1342 || bx == 1392)
    bunch = true;
  else if (bx == 1454 || bx == 1504 || bx == 1554)
    bunch = true;
  else if (bx == 2501 || bx == 2601)
    bunch = true;
  else if (bx == 3080 || bx == 3030 || bx == 3180)
    bunch = true;

  if (bx >= 1 && bx <= 351) {
    if ((bx % 50) == 1)
      bunch = true;
  } else if (bx >= 892 && bx <= 1245) {
    if (((bx - 892) % 50) == 0)
      bunch = true;
    else if (((bx - 895) % 50) == 0)
      bunch = true;
  } else if (bx >= 1786 && bx <= 2286) {
    if (((bx - 1786) % 50) == 0)
      bunch = true;
  } else if (bx >= 2671 && bx <= 3021) {
    if (((bx - 2671) % 50) == 0)
      bunch = true;
  }

  if (bunch) {
    //cout<<" reject "<<bx<<endl;
    hbx2->Fill(float(bx));
  } else {
    if (bx == 892)
      cout << " something wrong" << endl;
    if (bx == 1245)
      cout << " something wrong" << endl;
    if (bx == 3021)
      cout << " something wrong" << endl;
    if (bx == 2286)
      cout << " something wrong" << endl;
  }

  // Get digis
  edm::Handle<edm::DetSetVector<PixelDigi>> pixelDigis;
  iEvent.getByToken(tPixelDigi, pixelDigis);

#ifdef USE_SIM_LINKS
  // Get simlink data
  edm::Handle<edm::DetSetVector<PixelDigiSimLink>> pixelSimLinks;
  iEvent.getByToken(tPixelDigiSimLink, pixelSimLinks);
#endif

  // Get event setup (to get global transformation)
  edm::ESHandle<TrackerGeometry> geom;
  iSetup.get<TrackerDigiGeometryRecord>().get(geom);
  const TrackerGeometry &theTracker(*geom);

  int numberOfDetUnits = 0;
  int totalNumOfDigis = 0;

  int numberOfDetUnits1 = 0;
  int totalNumOfDigis1 = 0;
  int numberOfDetUnits2 = 0;
  int totalNumOfDigis2 = 0;
  int numberOfDetUnits3 = 0;
  int totalNumOfDigis3 = 0;
  int numOfDigisPerDet1 = 0;
  int numOfDigisPerDet2 = 0;
  int numOfDigisPerDet3 = 0;

  int numberOfDetUnitsF1 = 0;
  int totalNumOfDigisF1 = 0;
  int numberOfDetUnitsF2 = 0;
  int totalNumOfDigisF2 = 0;
  int numOfDigisPerDetF1 = 0;
  int numOfDigisPerDetF2 = 0;

  // Iterate on detector units
  edm::DetSetVector<PixelDigi>::const_iterator DSViter;
  for (DSViter = pixelDigis->begin(); DSViter != pixelDigis->end(); DSViter++) {
    bool valid = false;
    unsigned int detid = DSViter->id;  // = rawid
    DetId detId(detid);
    //const GeomDetUnit      * geoUnit = geom->idToDetUnit( detId );
    //const PixelGeomDetUnit * pixDet  = dynamic_cast<const PixelGeomDetUnit*>(geoUnit);
    unsigned int detType = detId.det();     // det type, tracker=1
    unsigned int subid = detId.subdetId();  //subdetector type, barrel=1

    if (PRINT)
      cout << "Det: " << detId.rawId() << " " << detId.null() << " " << detType << " " << subid << endl;

#ifdef HISTOS
    hdetunit->Fill(float(detid));
    hpixid->Fill(float(detType));
    hpixsubid->Fill(float(subid));
#endif

    if (detType != 1)
      continue;  // look only at tracker
    ++numberOfDetUnits;

    // Get the geom-detector
    const PixelGeomDetUnit *theGeomDet = dynamic_cast<const PixelGeomDetUnit *>(theTracker.idToDet(detId));
    double detZ = theGeomDet->surface().position().z();
    double detR = theGeomDet->surface().position().perp();
    //const BoundPlane plane = theGeomDet->surface(); // does not work

    //     int cols = theGeomDet->specificTopology().ncolumns();
    //     int rows = theGeomDet->specificTopology().nrows();
    //     float pitchX = theGeomDet->specificTopology().pitch().first;
    //     float pitchY = theGeomDet->specificTopology().pitch().second;

    const PixelTopology &topology = theGeomDet->specificTopology();
    int cols = topology.ncolumns();
    int rows = topology.nrows();
    float pitchX = topology.pitch().first;
    float pitchY = topology.pitch().second;

    unsigned int layerC = 0;
    unsigned int ladderC = 0;
    unsigned int zindex = 0;

    // Shell { mO = 1, mI = 2 , pO =3 , pI =4 };
    int shell = 0;      // shell id
    int sector = 0;     // 1-8
    int ladder = 0;     // 1-22
    int layer = 0;      // 1-3
    int module = 0;     // 1-4
    bool half = false;  //

    unsigned int disk = 0;   //1,2,3
    unsigned int blade = 0;  //1-24
    unsigned int side = 0;   //size=1 for -z, 2 for +z
    unsigned int panel = 0;  //panel=1

    // Subdet it, pix barrel=1, forward=2
    if (subid == 2) {  // forward

#ifdef HISTOS
      hdetrF->Fill(detR);
      hdetzF->Fill(detZ);
      hcolsF->Fill(float(cols));
      hrowsF->Fill(float(rows));
#endif

      disk = tTopo->pxfDisk(detid);      //1,2,3
      blade = tTopo->pxfBlade(detid);    //1-24
      zindex = tTopo->pxfModule(detid);  //
      side = tTopo->pxfSide(detid);      //size=1 for -z, 2 for +z
      panel = tTopo->pxfPanel(detid);    //panel=1

      if (PRINT) {
        cout << "Forward det " << subid << ", disk " << disk << ", blade " << blade << ", module " << zindex
             << ", side " << side << ", panel " << panel << endl;
        cout << " col/row, pitch " << cols << " " << rows << " " << pitchX << " " << pitchY << endl;
      }

    } else if (subid == 1) {  // Barrel

      // Barell layer = 1,2,3
      layerC = tTopo->pxbLayer(detid);
      // Barrel ladder id 1-20,32,44.
      ladderC = tTopo->pxbLadder(detid);
      // Barrel Z-index=1,8
      zindex = tTopo->pxbModule(detid);
      // Convert to online
      PixelBarrelName pbn(detid);
      // Shell { mO = 1, mI = 2 , pO =3 , pI =4 };
      PixelBarrelName::Shell sh = pbn.shell();  //enum
      sector = pbn.sectorName();
      ladder = pbn.ladderName();
      layer = pbn.layerName();
      module = pbn.moduleName();
      half = pbn.isHalfModule();
      shell = int(sh);
      // change the module sign for z<0
      if (shell == 1 || shell == 2)
        module = -module;
      // change ladeer sign for Outer )x<0)
      if (shell == 1 || shell == 3)
        ladder = -ladder;

      if (PRINT) {
        cout << " Barrel layer, ladder, module " << layerC << " " << ladderC << " " << zindex << " " << sh << "("
             << shell << ") " << sector << " " << layer << " " << ladder << " " << module << " " << half << endl;
        //cout<<" Barrel det, thick "<<detThick<<" "
        //  <<" layer, ladder, module "
        //  <<layer<<" "<<ladder<<" "<<zindex<<endl;
        //cout<<" col/row, pitch "<<cols<<" "<<rows<<" "
        //  <<pitchX<<" "<<pitchY<<endl;
      }

    }  // end fb-bar

#ifdef USE_SIM_LINKS
    // Look at simlink information (simulated data only)

    int numberOfSimLinks = 0;
    edm::DetSetVector<PixelDigiSimLink>::const_iterator isearch = pixelSimLinks->find(detid);

    if (isearch != pixelSimLinks->end()) {  //if it is not empty
      edm::DetSet<PixelDigiSimLink> link_detset = (*pixelSimLinks)[detid];
      edm::DetSet<PixelDigiSimLink>::const_iterator it;
      // Loop over DigisSimLink in this det unit
      for (it = link_detset.data.begin(); it != link_detset.data.end(); it++) {
        numberOfSimLinks++;
        // these methods should be declared const, fixed by M.P.
        // wait for next releasse and then uncomment
        unsigned int chan = it->channel();
        unsigned int simTrack = it->SimTrackId();
        float frac = it->fraction();
#ifdef HISTOS
        hsimlinks->Fill(float(simTrack));
        hfract->Fill(frac);
#endif

        if (PRINT)
          cout << " Sim link " << numberOfSimLinks << " " << chan << " " << simTrack << " " << frac << endl;
      }  // end simlink det loop

    }  // end simlink if

#endif  // USE_SIM_LINKS

    unsigned int numberOfDigis = 0;

    // Look at digis now
    edm::DetSet<PixelDigi>::const_iterator di;
    for (di = DSViter->data.begin(); di != DSViter->data.end(); di++) {
      //for(di = begin; di != end; di++) {

      numberOfDigis++;
      totalNumOfDigis++;
      int adc = di->adc();     // charge, modifued to unsiged short
      int col = di->column();  // column
      int row = di->row();     // row
      //int tof = di->time();    // tof always 0, method deleted

      // channel index needed to look for the simlink to simtracks
      int channel = PixelChannelIdentifier::pixelToChannel(row, col);
      if (PRINT)
        cout << numberOfDigis << " Col: " << col << " Row: " << row << " ADC: " << adc << " channel = " << channel
             << endl;

      if (col > 415)
        cout << " Error: column index too large " << col << " Barrel layer, ladder, module " << layer << " " << ladder
             << " " << zindex << endl;
      if (row > 159)
        cout << " Error: row index too large " << row << endl;

#ifdef HISTOS
      if (layer == 1) {
        bool noise = false;  //(ladder==6) || (module==-2) || (col==364) || (row==1);
        if (!noise) {
          valid = valid || true;
          heloss1->Fill(float(adc));
          hcols1->Fill(float(col));
          hrows1->Fill(float(row));
          hpixMap1->Fill(float(col), float(row));
          totalNumOfDigis1++;
          //htest2->Fill(float(module),float(adc));
          numOfDigisPerDet1++;

          //old 	   if(RectangularPixelTopology::isItBigPixelInX(row)) {
          //new	   if(topology.isItBigPixelInX(row)) {
          // 	     //cout<<" big in X "<<row<<endl;
          // 	     heloss1bigx->Fill(float(adc));
          // 	     hrows1big->Fill(float(row));
          // 	   }
          //old	   if(RectangularPixelTopology::isItBigPixelInY(col)) {
          //new	   if(topology.isItBigPixelInY(col)) {
          // 	     //cout<<" big in Y "<<col<<endl;
          // 	     heloss1bigy->Fill(float(adc));
          // 	     hcols1big->Fill(float(col));
          // 	   }

        }  // noise
      } else if (layer == 2) {
        // look for the noisy pixel
        bool noise = false;  // (ladder==6) && (module==-2) && (col==364) && (row==1);
        if (noise) {
          //cout<<" noise pixel "<<layer<<" "<<sector<<" "<<shell<<endl;
          hpixMapNoise->Fill(float(col), float(row));
          hneloss2->Fill(float(adc));
        } else {
          valid = valid || true;
          heloss2->Fill(float(adc));
          hcols2->Fill(float(col));
          hrows2->Fill(float(row));
          hpixMap2->Fill(float(col), float(row));
          totalNumOfDigis2++;
          numOfDigisPerDet2++;
        }  // noise
      } else if (layer == 3) {
        bool noise = false;  //(ladder==6) || (module==-2) || (col==364) || (row==1);
        if (!noise) {
          valid = valid || true;
          heloss3->Fill(float(adc));
          hcols3->Fill(float(col));
          hrows3->Fill(float(row));
          hpixMap3->Fill(float(col), float(row));
          totalNumOfDigis3++;
          numOfDigisPerDet3++;
        }  // noise
      } else if (disk == 1) {
        bool noise = false;  //(ladder==6) || (module==-2) || (col==364) || (row==1);
        if (!noise) {
          valid = valid || true;
          helossF1->Fill(float(adc));
          hcolsF1->Fill(float(col));
          hrowsF1->Fill(float(row));
          totalNumOfDigisF1++;
          numOfDigisPerDetF1++;
        }  // noise

      } else if (disk == 2) {
        bool noise = false;  //(ladder==6) || (module==-2) || (col==364) || (row==1);
        if (!noise) {
          valid = valid || true;
          helossF2->Fill(float(adc));
          hcolsF2->Fill(float(col));
          hrowsF2->Fill(float(row));
          totalNumOfDigisF2++;
          numOfDigisPerDetF2++;
        }  // noise
      }    // end if layer
#endif

    }  // end for digis in detunit
       //if(PRINT)
       //cout<<" for det "<<detid<<" digis = "<<numberOfDigis<<endl;

#ifdef HISTOS
    // Some histos
    if (valid) {         // to igore noise pixels
      if (subid == 2) {  // forward

        hdetrF->Fill(detR);
        hdetzF->Fill(detZ);
        hcolsF->Fill(float(cols));
        hrowsF->Fill(float(rows));

        if (disk == 1) {
          hblade1->Fill(float(blade));
          ++numberOfDetUnitsF1;
          hdigisPerDetF1->Fill(float(numOfDigisPerDetF1));
          numOfDigisPerDetF1 = 0;

        } else if (disk == 2) {
          hblade2->Fill(float(blade));
          ++numberOfDetUnitsF2;
          hdigisPerDetF2->Fill(float(numOfDigisPerDetF2));
          numOfDigisPerDetF2 = 0;
        }  // if disk

      } else if (subid == 1) {  // barrel

        hdetr->Fill(detR);
        hdetz->Fill(detZ);
        hcolsB->Fill(float(cols));
        hrowsB->Fill(float(rows));

        hlayerid->Fill(float(layer));
        hshellid->Fill(float(shell));
        hsectorid->Fill(float(sector));

        if (layer == 1) {
          hladder1id->Fill(float(ladder));
          hz1id->Fill(float(module));
          hdetMap1->Fill(float(module), float(ladder));
          ++numberOfDetUnits1;
          hdigisPerDet1->Fill(float(numOfDigisPerDet1));
          numOfDigisPerDet1 = 0;

        } else if (layer == 2) {
          hladder2id->Fill(float(ladder));
          hz2id->Fill(float(module));
          hdetMap2->Fill(float(module), float(ladder));
          ++numberOfDetUnits2;
          hdigisPerDet2->Fill(float(numOfDigisPerDet2));
          numOfDigisPerDet2 = 0;

        } else if (layer == 3) {
          hladder3id->Fill(float(ladder));
          hz3id->Fill(float(module));
          hdetMap3->Fill(float(module), float(ladder));
          ++numberOfDetUnits3;
          hdigisPerDet3->Fill(float(numOfDigisPerDet3));
          numOfDigisPerDet3 = 0;

        }  // layer
      }    // if bpix
    }      // if valid
#endif

  }  // end for det-units

  if (PRINT)
    cout << " Number of full det-units = " << numberOfDetUnits << " total digis = " << totalNumOfDigis << endl;
  hdets->Fill(float(numberOfDetUnits));
  hdigis->Fill(float(totalNumOfDigis));

  if (numberOfDetUnits > 0) {
    hevent->Fill(float(event));
    hlumi1->Fill(float(lumiBlock));
    hbx1->Fill(float(bx));
    horbit->Fill(float(orbit));

    hdigis1->Fill(float(totalNumOfDigis));
    float tmp = float(totalNumOfDigis1) + float(totalNumOfDigis2) + float(totalNumOfDigis3);
    hdigis2->Fill(tmp);
    tmp = float(totalNumOfDigisF1) + float(totalNumOfDigisF2);
    hdigis3->Fill(tmp);

    if (bunch)
      hdigis4->Fill(float(totalNumOfDigis));
    else
      hdigis5->Fill(float(totalNumOfDigis));

    if (numberOfDetUnits > 20)
      hbx3->Fill(float(bx));

    if (totalNumOfDigis > 100)
      hbx4->Fill(float(bx));
    else if (totalNumOfDigis > 4)
      hbx5->Fill(float(bx));
    else
      hbx6->Fill(float(bx));
  }

#ifdef HISTOS
  hdigisPerLay1->Fill(float(totalNumOfDigis1));
  hdigisPerLay2->Fill(float(totalNumOfDigis2));
  hdigisPerLay3->Fill(float(totalNumOfDigis3));
  if (totalNumOfDigis1 > 0)
    hdetsPerLay1->Fill(float(numberOfDetUnits1));
  if (totalNumOfDigis2 > 0)
    hdetsPerLay2->Fill(float(numberOfDetUnits2));
  if (totalNumOfDigis3 > 0)
    hdetsPerLay3->Fill(float(numberOfDetUnits3));

  hdigisPerLayF1->Fill(float(totalNumOfDigisF1));
  hdigisPerLayF2->Fill(float(totalNumOfDigisF2));
  hdetsPerLayF1->Fill(float(numberOfDetUnitsF1));
  hdetsPerLayF2->Fill(float(numberOfDetUnitsF2));
#endif
}
// ------------ method called to at the end of the job  ------------
void PixelDigisTest::endJob() {
  cout << " End PixelDigisTest " << endl;
  //hFile->Write();
  //hFile->Close();
}

//define this as a plug-in
DEFINE_FWK_MODULE(PixelDigisTest);
