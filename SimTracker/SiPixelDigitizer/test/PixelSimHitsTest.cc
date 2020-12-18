// -*- C++ -*-
//
// Package:    PixelSimHitsTest
// Class:      PixelSimHitsTest
//
/**\class PixelSimHitsTest PixelSimHitsTest.cc 

 Description: Test pixel simhits. Barrel only. Uses root histos.
 Modifed for module() method in PXBDetId. 2/06
 Add global coordiantes. 21/2/06
 Update 28/2/12. Works with CMSSW_5_3_8, not in cvs, d.k.
 Needs updating for 6.2 (in cvs) 
New det-id. 
 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  d.k.
//         Created:  Jan CET 2006
// $Id: PixelSimHitsTest.cc,v 1.8 2009/11/13 14:14:23 fambrogl Exp $
//
//
// system include files
#include <iostream>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "DataFormats/Common/interface/Handle.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Framework/interface/ESHandle.h"

// my includes
//#include "Geometry/CommonDetUnit/interface/GeomDet.h"

#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/CommonDetUnit/interface/PixelGeomDetUnit.h"
#include "Geometry/CommonDetUnit/interface/PixelGeomDetType.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"  //

// for det id
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "DataFormats/TrackerCommon/interface/PixelBarrelName.h"

#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"

#include "DataFormats/SiPixelDetId/interface/PixelSubdetector.h"
//#include "Geometry/Surface/interface/Surface.h"

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
#include <TProfile.h>

using namespace std;
using namespace edm;

//#define CHECK_GEOM

class PixelSimHitsTest : public edm::EDAnalyzer {
public:
  explicit PixelSimHitsTest(const edm::ParameterSet &);
  ~PixelSimHitsTest() override;
  void beginJob() override;
  void analyze(const edm::Event &, const edm::EventSetup &) override;
  void endJob() override;

private:
  // ----------member data ---------------------------
  edm::ParameterSet conf_;
  bool PRINT;
  string mode_;  // select bpix/fpix
  edm::EDGetTokenT<PSimHitContainer> tPixelSimHits;
  edm::ESGetToken<TrackerTopology, TrackerTopologyRcd> trackerTopoToken;
  edm::ESGetToken<TrackerGeometry, TrackerDigiGeometryRecord> trackerGeomToken;
  int numEvents;
  double numSimHits, numSimHitsGood;

  TH1F *heloss1, *heloss2, *heloss3, *hdetunit, *hpabs, *hpid, *htof, *htid;
  TH1F *hpixid, *hpixsubid, *hlayerid, *hladder1id, *hladder2id, *hladder3id, *hz1id, *hz2id, *hz3id;
  //TH1F *hladder1idUp, *hladder2idUp, *hladder3idUp;
  TH1F *hthick1, *hthick2, *hthick3, *hlength1, *hlength2, *hlength3;
  TH1F *hwidth1, *hwidth2, *hwidth3;
  TH1F *hsimHitsPerDet1, *hsimHitsPerDet2, *hsimHitsPerDet3;
  TH1F *hsimHitsPerLay1, *hsimHitsPerLay2, *hsimHitsPerLay3;
  TH1F *hsimHits, *hsimHitsGood;
  TH1F *hdetsPerLay1, *hdetsPerLay2, *hdetsPerLay3;
  TH1F *heloss1e, *heloss2e, *heloss3e;
  TH1F *heloss1mu, *heloss2mu, *heloss3mu;
  TH1F *htheta1, *htheta2, *htheta3;
  TH1F *hphi1, *hphi2, *hphi3;
  TH1F *hdetr, *hdetz, *hdetphi1, *hdetphi2, *hdetphi3;
  TH1F *hglobr1, *hglobr2, *hglobr3, *hglobz1, *hglobz2, *hglobz3;
  TH1F *hcolsB, *hrowsB, *hcolsF, *hrowsF;

  TH2F *htest, *htest2, *htest3, *htest4, *htest5;
  //TProfile *hp1, *hp2, *hp3, *hp4, *hp5;

#ifdef CHECK_GEOM
  float modulePositionZ[3][44][8];
  float modulePositionR[3][44][8];
  float modulePositionPhi[3][44][8];
#endif
};
//
PixelSimHitsTest::PixelSimHitsTest(const edm::ParameterSet &iConfig) : conf_(iConfig) {
  std::string src_, list_;
  src_ = iConfig.getParameter<std::string>("src");
  list_ = iConfig.getParameter<std::string>("list");

  edm::InputTag tag(src_, list_);  // for the ByToken
  tPixelSimHits = consumes<PSimHitContainer>(tag);

  trackerTopoToken = esConsumes<TrackerTopology, TrackerTopologyRcd>();
  trackerGeomToken = esConsumes<TrackerGeometry, TrackerDigiGeometryRecord>();

  mode_ = iConfig.getUntrackedParameter<std::string>("mode", "bpix");  // select bpix or fpix
  PRINT = iConfig.getUntrackedParameter<bool>("Verbosity", false);     // printout
  cout << " Construct PixelSimHitsTest " << endl;
}

PixelSimHitsTest::~PixelSimHitsTest() {
  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)
  cout << " Destroy PixelSimHitsTest " << endl;
}

// ------------ method called at the begining   ------------
void PixelSimHitsTest::beginJob() {
  using namespace edm;
  cout << "Initialize PixelSimHitsTest " << endl;

  // put here whatever you want to do at the beginning of the job
  numEvents = 0;
  numSimHits = 0.0;
  numSimHitsGood = 0.0;

  edm::Service<TFileService> fs;

  const float max_charge = 200.;  // in ke
  heloss1 = fs->make<TH1F>("heloss1", "Eloss l1", 100, 0., max_charge);
  heloss2 = fs->make<TH1F>("heloss2", "Eloss l2", 100, 0., max_charge);
  heloss3 = fs->make<TH1F>("heloss3", "Eloss l3", 100, 0., max_charge);

  hdetunit = fs->make<TH1F>("hdetunit", "Det unit", 1000, 302000000., 302300000.);
  hpabs = fs->make<TH1F>("hpabs", "Pabs", 100, 0., 110.);
  htof = fs->make<TH1F>("htof", "TOF", 50, -25., 25.);
  hpid = fs->make<TH1F>("hpid", "PID", 1000, 0., 1000.);
  htid = fs->make<TH1F>("htid", "Track id", 100, 0., 100.);

  hpixid = fs->make<TH1F>("hpixid", "Pix det id", 10, 0., 10.);
  hpixsubid = fs->make<TH1F>("hpixsubid", "Pix Barrel id", 10, 0., 10.);
  hlayerid = fs->make<TH1F>("hlayerid", "Pix layer id", 10, 0., 10.);
  hladder1id = fs->make<TH1F>("hladder1id", "Ladder L1 id", 102, -25.5, 25.5);
  hladder2id = fs->make<TH1F>("hladder2id", "Ladder L2 id", 102, -25.5, 25.5);
  hladder3id = fs->make<TH1F>("hladder3id", "Ladder L3 id", 102, -25.5, 25.5);
  hz1id = fs->make<TH1F>("hz1id", "Z-index id L1", 10, -5., 5.);
  hz2id = fs->make<TH1F>("hz2id", "Z-index id L2", 10, -5., 5.);
  hz3id = fs->make<TH1F>("hz3id", "Z-index id L3", 10, -5., 5.);

  hthick1 = fs->make<TH1F>("hthick1", "Det 1 Thinckess", 400, 0., 0.04);
  hthick2 = fs->make<TH1F>("hthick2", "Det 2 Thinckess", 400, 0., 0.04);
  hthick3 = fs->make<TH1F>("hthick3", "Det 3 Thinckess", 400, 0., 0.04);

  hlength1 = fs->make<TH1F>("hlength1", "Det 1 Length", 700, -3.5, 3.5);
  hlength2 = fs->make<TH1F>("hlength2", "Det 2 Length", 700, -3.5, 3.5);
  hlength3 = fs->make<TH1F>("hlength3", "Det 3 Length", 700, -3.5, 3.5);

  hwidth1 = fs->make<TH1F>("hwidth1", "Det 1 Width", 200, -1., 1.);
  hwidth2 = fs->make<TH1F>("hwidth2", "Det 2 Width", 200, -1., 1.);
  hwidth3 = fs->make<TH1F>("hwidth3", "Det 3 Width", 200, -1., 1.);

  hsimHitsPerDet1 = fs->make<TH1F>("hsimHitsPerDet1", "SimHits per det l1", 200, -0.5, 199.5);
  hsimHitsPerDet2 = fs->make<TH1F>("hsimHitsPerDet2", "SimHits per det l2", 200, -0.5, 199.5);
  hsimHitsPerDet3 = fs->make<TH1F>("hsimHitsPerDet3", "SimHits per det l3", 200, -0.5, 199.5);
  hsimHitsPerLay1 = fs->make<TH1F>("hsimHitsPerLay1", "SimHits per layer l1", 2000, -0.5, 1999.5);
  hsimHitsPerLay2 = fs->make<TH1F>("hsimHitsPerLay2", "SimHits per layer l2", 2000, -0.5, 1999.5);
  hsimHitsPerLay3 = fs->make<TH1F>("hsimHitsPerLay3", "SimHits per layer l3", 2000, -0.5, 1999.5);
  hdetsPerLay1 = fs->make<TH1F>("hdetsPerLay1", "Full dets per layer l1", 161, -0.5, 160.5);
  hdetsPerLay3 = fs->make<TH1F>("hdetsPerLay3", "Full dets per layer l3", 353, -0.5, 352.5);
  hdetsPerLay2 = fs->make<TH1F>("hdetsPerLay2", "Full dets per layer l2", 257, -0.5, 256.5);
  hsimHits = fs->make<TH1F>("hsimHits", "SimHits for bpix", 2000, -0.5, 1999.5);
  hsimHitsGood = fs->make<TH1F>("hsimHitsGood", "SimHits for bpix", 2000, -0.5, 1999.5);  // no deltas

  heloss1e = fs->make<TH1F>("heloss1e", "Eloss e l1", 100, 0., max_charge);
  heloss2e = fs->make<TH1F>("heloss2e", "Eloss e l2", 100, 0., max_charge);
  heloss3e = fs->make<TH1F>("heloss3e", "Eloss e l3", 100, 0., max_charge);

  heloss1mu = fs->make<TH1F>("heloss1mu", "Eloss mu l1", 100, 0., max_charge);
  heloss2mu = fs->make<TH1F>("heloss2mu", "Eloss mu l2", 100, 0., max_charge);
  heloss3mu = fs->make<TH1F>("heloss3mu", "Eloss mu l3", 100, 0., max_charge);

  htheta1 = fs->make<TH1F>("htheta1", "Theta l1", 350, 0.0, 3.5);
  htheta2 = fs->make<TH1F>("htheta2", "Theta l2", 350, 0.0, 3.5);
  htheta3 = fs->make<TH1F>("htheta3", "Theta l3", 350, 0.0, 3.5);
  hphi1 = fs->make<TH1F>("hphi1", "phi l1", 1400, -3.5, 3.5);
  hphi2 = fs->make<TH1F>("hphi2", "phi l2", 1400, -3.5, 3.5);
  hphi3 = fs->make<TH1F>("hphi3", "phi l3", 1400, -3.5, 3.5);

  hdetr = fs->make<TH1F>("hdetr", "det r", 1500, 0., 15.);
  hdetz = fs->make<TH1F>("hdetz", "det z", 5200, -26., 26.);

  hdetphi1 = fs->make<TH1F>("hdetphi1", "det phi l1", 700, -3.5, 3.5);
  hdetphi2 = fs->make<TH1F>("hdetphi2", "det phi l2", 700, -3.5, 3.5);
  hdetphi3 = fs->make<TH1F>("hdetphi3", "det phi l3", 700, -3.5, 3.5);

  hcolsB = fs->make<TH1F>("hcolsB", "cols per bar det", 450, 0., 450.);
  hrowsB = fs->make<TH1F>("hrowsB", "rows per bar det", 200, 0., 200.);
  hcolsF = fs->make<TH1F>("hcolsF", "cols per for det", 300, 0., 300.);
  hrowsF = fs->make<TH1F>("hrowsF", "rows per for det", 200, 0., 200.);

  //hladder1idUp = fs->make<TH1F>( "hladder1idUp", "Ladder L1 id", 100, -0.5, 49.5);
  //hladder2idUp = fs->make<TH1F>( "hladder2idUp", "Ladder L2 id", 100, -0.5, 49.5);
  //hladder3idUp = fs->make<TH1F>( "hladder3idUp", "Ladder L3 id", 100, -0.5, 49.5);

  hglobr1 = fs->make<TH1F>("hglobr1", "global r1", 150, 0., 15.);
  hglobz1 = fs->make<TH1F>("hglobz1", "global z1", 540, -27., 27.);
  hglobr2 = fs->make<TH1F>("hglobr2", "global r2", 150, 0., 15.);
  hglobz2 = fs->make<TH1F>("hglobz2", "global z2", 540, -27., 27.);
  hglobr3 = fs->make<TH1F>("hglobr3", "global r3", 150, 0., 15.);
  hglobz3 = fs->make<TH1F>("hglobz3", "global z3", 540, -27., 27.);

  // layer 1 only
  htest = fs->make<TH2F>("htest", " ", 108, -27., 27., 35, -3.5, 3.5);     // global z versus local y
  htest2 = fs->make<TH2F>("htest2", " ", 108, -27., 27., 60, 0., 600.);    // global z versus eloss
  htest3 = fs->make<TH2F>("htest3", " ", 240, -12., 12., 240, -12., 12.);  // x-y plane
  //htest4 = fs->make<TH2F>("htest4"," ",80,-4.,4.,100,-5.,5.);

  //hp1 = fs->make<TProfile>("hp1"," ",50,0.,50.);    // default option
  //hp2 = fs->make<TProfile>("hp2"," ",50,0.,50.," "); // option set to " "
  //hp3 = fs->make<TProfile>("hp3"," ",50,0.,50.,-100.,100.); //

#ifdef CHECK_GEOM
  // To get the module position
  for (int i = 0; i < 3; i++) {
    for (int n = 0; n < 44; n++) {
      for (int m = 0; m < 8; m++) {
        modulePositionR[i][n][m] = -1;
        modulePositionZ[i][n][m] = -1;
        modulePositionPhi[i][n][m] = -1;
      }
    }
  }
#endif
}

// ------------ method called to produce the data  ------------
void PixelSimHitsTest::analyze(const edm::Event &iEvent, const edm::EventSetup &iSetup) {
  const bool DEBUG = false;
  //string mode = "bpix";

  using namespace edm;
  if (PRINT)
    cout << " Analyze PixelSimHitsTest " << endl;

  // Get event setup (to get global transformation)
  edm::ESHandle<TrackerGeometry> geom = iSetup.getHandle(trackerGeomToken);
  const TrackerGeometry &theTracker(*geom);

  //Retrieve tracker topology from geometry (for det id)
  edm::ESHandle<TrackerTopology> tTopo = iSetup.getHandle(trackerTopoToken);

  // Get input data
  int totalNumOfSimHits = 0;
  int totalNumOfSimHits1 = 0;
  int totalNumOfSimHits2 = 0;
  int totalNumOfSimHits3 = 0;
  int goodHits = 0;  // above pt=0.1GeV

  // To count simhits per det module
  //typedef std::map<unsigned int, std::vector<PSimHit>,
  //std::less<unsigned int>> simhit_map;
  //typedef simhit_map::iterator simhit_map_iterator;
  //simhit_map SimHitMap;
  map<unsigned int, vector<PSimHit>, less<unsigned int> > SimHitMap1;
  map<unsigned int, vector<PSimHit>, less<unsigned int> > SimHitMap2;
  map<unsigned int, vector<PSimHit>, less<unsigned int> > SimHitMap3;

  // Access hits containers
  if (DEBUG)
    cout << "Define simhit container" << endl;
  Handle<PSimHitContainer> PixelHits;
  //Handle<PSimHitContainer> PixelHitsLowTof;
  //Handle<PSimHitContainer> PixelHitsHighTof;
  iEvent.getByToken(tPixelSimHits, PixelHits);

  //if(mode=="bpix") {
  //iEvent.getByLabel( src_ ,"TrackerHitsPixelBarrelLowTof" ,PixelHitsLowTof);
  //iEvent.getByLabel( src_ ,"TrackerHitsPixelBarrelHighTof",PixelHitsHighTof);
  //} else if(mode=="fpix") {
  //iEvent.getByLabel( src_ ,"TrackerHitsPixelEndcapLowTof",PixelHitsLowTof);
  //iEvent.getByLabel( src_ ,"TrackerHitsPixelEndcapHighTof",PixelHitsHighTof);
  //}

  if (DEBUG)
    cout << "Loop over SimHits LowTof" << endl;
  //for(vector<PSimHit>::const_iterator isim = PixelHitsLowTof->begin();
  // isim != PixelHitsLowTof->end(); ++isim) {
  for (vector<PSimHit>::const_iterator isim = PixelHits->begin(); isim != PixelHits->end(); ++isim) {
    totalNumOfSimHits++;
    // Det id
    DetId detId = DetId((*isim).detUnitId());
    unsigned int dettype = detId.det();     // for pixel=1
    unsigned int subid = detId.subdetId();  // barrel=1
    unsigned int detid = detId.rawId();     // raw det id

    if (dettype != 1 && subid != 1)
      cout << " error in det id " << dettype << " " << subid << endl;
    if (PRINT)
      cout << totalNumOfSimHits << " det id " << detid << " " << dettype << " " << subid << endl;
    if (DEBUG)
      cout << " det unit " << (*isim).detUnitId() << detId.null() << endl;

    // Global variables
    const PixelGeomDetUnit *theGeomDet = dynamic_cast<const PixelGeomDetUnit *>(theTracker.idToDet(detId));
    double detZ = theGeomDet->surface().position().z();      // module z position
    double detR = theGeomDet->surface().position().perp();   //        r
    double detPhi = theGeomDet->surface().position().phi();  //        phi
    hdetr->Fill(detR);
    hdetz->Fill(detZ);

    double detThick = theGeomDet->specificSurface().bounds().thickness();
    double detLength = theGeomDet->specificSurface().bounds().length();
    double detWidth = theGeomDet->specificSurface().bounds().width();

    int cols = theGeomDet->specificTopology().ncolumns();
    int rows = theGeomDet->specificTopology().nrows();

    if (DEBUG)
      cout << "det z/r " << detZ << "/" << detR << " thick/len/wid " << detThick << " " << detLength << " " << detWidth
           << " cols/rows " << cols << " " << rows << endl;

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

    unsigned int disk = 0;   //1,2
    unsigned int blade = 0;  //1-24
    unsigned int side = 0;   //size=1 for -z, 2 for +z
    unsigned int panel = 0;  //panel=1

    if (mode_ == "fpix") {
      disk = tTopo->pxfDisk(detid);      //1,2,3
      blade = tTopo->pxfBlade(detid);    //1-24
      zindex = tTopo->pxfModule(detid);  //
      side = tTopo->pxfSide(detid);      //size=1 for -z, 2 for +z
      panel = tTopo->pxfPanel(detid);    //panel=1

      if (PRINT) {
        cout << "Forward det " << subid << ", disk " << disk << ", blade " << blade << ", module " << zindex
             << ", side " << side << ", panel " << panel << endl;
        //cout<<" col/row, pitch "<<cols<<" "<<rows<<" "
        //<<pitchX<<" "<<pitchY<<endl;
      }

      hcolsF->Fill(float(cols));
      hrowsF->Fill(float(rows));

    } else if (mode_ == "bpix") {  // Barrel

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

      hcolsB->Fill(float(cols));
      hrowsB->Fill(float(rows));

    }  // end fb-bar

#ifdef CHECK_GEOM
    // To get the module position
    modulePositionR[layer - 1][ladderC - 1][zindex - 1] = detR;
    modulePositionZ[layer - 1][ladderC - 1][zindex - 1] = detZ;
    modulePositionPhi[layer - 1][ladderC - 1][zindex - 1] = detPhi;
#endif

    // SimHit information
    float eloss = (*isim).energyLoss() * 1000000 / 3.7;  //convert GeV to ke
    float tof = (*isim).timeOfFlight();
    float p = (*isim).pabs();
    float pt = (*isim).momentumAtEntry().perp();
    float theta = (*isim).thetaAtEntry();
    float phi = (*isim).phiAtEntry();
    int pid = ((*isim).particleType());
    int tid = (*isim).trackId();
    int procType = (*isim).processType();

    float x = (*isim).entryPoint().x();  // width (row index, in col direction)
    float y = (*isim).entryPoint().y();  // length (col index, in row direction)
    float z = (*isim).entryPoint().z();  // thick

    float x2 = (*isim).exitPoint().x();
    float y2 = (*isim).exitPoint().y();
    float z2 = (*isim).exitPoint().z();

    float dz = abs(z - z2);
    bool moduleDirectionUp = (z < z2);  // for positive direction z2>z

    float xpos = (x + x2) / 2.;
    float ypos = (y + y2) / 2.;
    float zpos = (z + z2) / 2.;

    if (pt > 0.1) {
      goodHits++;
    }
    if (PRINT) {
      if (pt > 0.1)
        cout << " simhit: ";
      else if (pid == 11)
        cout << " delta: ";
      else
        cout << " low pt (sec?): ";
      cout << " id " << pid << " " << tid << " proc " << procType << " tof " << tof << " de " << eloss << " pt " << pt
           << " entry " << x << "/" << y << "/" << z << " lenz " << dz << " " << moduleDirectionUp << endl;
    }
    if (DEBUG)
      cout << "  center pos " << xpos << " " << ypos << " " << zpos;

    LocalPoint loc(xpos, ypos, zpos);
    double gloX = theGeomDet->surface().toGlobal(loc).x();     //
    double gloY = theGeomDet->surface().toGlobal(loc).y();     //
    double gloR = theGeomDet->surface().toGlobal(loc).perp();  //
    double gloZ = theGeomDet->surface().toGlobal(loc).z();     //
    if (DEBUG)
      cout << ", global pos " << gloX << " " << gloY << " " << gloR << " " << gloZ << endl;

    htest3->Fill(gloX, gloY);
    hdetunit->Fill(float(detId.rawId()));
    hpabs->Fill(p);
    htof->Fill(tof);
    hpid->Fill(float(abs(pid)));
    htid->Fill(float(tid));
    hpixid->Fill(float(detid));
    hpixsubid->Fill(float(subid));
    hlayerid->Fill(float(layer));

    // Transform the theta from local module coordinates to global
    //if(theta<= PI/2.) theta = PI/2. - theta; // For +z global
    //else theta = (PI/2. + PI) - theta;

    if (mode_ == "fpix") {
      if (disk == 1) {
        //cout<<" disk "<<disk<<endl;
        totalNumOfSimHits1++;
        heloss1->Fill(eloss);
        if (pid == 11)
          heloss1e->Fill(eloss);
        else
          heloss1mu->Fill(eloss);
        hladder1id->Fill(float(blade));
        hz1id->Fill(float(zindex));
        hthick1->Fill(dz);
        hlength1->Fill(y);
        hwidth1->Fill(x);

        //SimHitMap1[detId.rawId()].push_back((*isim));
        htheta1->Fill(theta);
        hglobr1->Fill(gloR);
        hglobz1->Fill(gloZ);
        hdetphi1->Fill(detPhi);

      } else if (disk == 2) {
        //cout<<" disk "<<disk<<endl;
        totalNumOfSimHits2++;
        heloss2->Fill(eloss);
        if (pid == 11)
          heloss2e->Fill(eloss);
        else
          heloss2mu->Fill(eloss);
        hladder2id->Fill(float(blade));
        hz2id->Fill(float(zindex));
        hthick2->Fill(dz);
        hlength2->Fill(y);
        hwidth2->Fill(x);

        //SimHitMap2[detId.rawId()].push_back((*isim));
        hglobr2->Fill(gloR);
        hglobz2->Fill(gloZ);
        hdetphi2->Fill(detPhi);

      }  // end disks

    } else if (mode_ == "bpix") {
      if (layer == 1) {
        //cout<<" layer "<<layer<<endl;
        totalNumOfSimHits1++;
        heloss1->Fill(eloss);
        if (abs(pid) == 11)
          heloss1e->Fill(eloss);
        else
          heloss1mu->Fill(eloss);
        hladder1id->Fill(float(ladder));
        hz1id->Fill(float(module));
        hthick1->Fill(dz);
        hlength1->Fill(y);
        hwidth1->Fill(x);
        if (abs(pid) == 13 && p > 1.)
          hphi1->Fill(phi);

        // Test half modules
        //        if(ladder==5 || ladder==6 || ladder==15 || ladder==16 ) { // half-modules
        // 	 hwidth1h->Fill(x);
        // 	 if(pid==13 && p>1.) {  // select primary muons with mom above 1.
        // 	   hphi1h->Fill(phi);
        // 	   hglobr1h->Fill(gloR);
        // 	 }
        //        } else {
        //        }

        SimHitMap1[detId.rawId()].push_back((*isim));
        htheta1->Fill(theta);
        hglobr1->Fill(gloR);
        hglobz1->Fill(gloZ);

        // Check the coordinate system and counting
        htest->Fill(gloZ, ypos);
        if (abs(pid) != 11)
          htest2->Fill(gloZ, eloss);

        //if(pid!=11 && moduleDirectionUp)  hladder1idUp->Fill(float(ladder));

        //hp1->Fill(float(ladder),detR,1);
        //hp2->Fill(float(ladder),detPhi);
        hdetphi1->Fill(detPhi);

      } else if (layer == 2) {
        //cout<<" layer "<<layer<<endl;
        totalNumOfSimHits2++;
        heloss2->Fill(eloss);
        if (abs(pid) == 11)
          heloss2e->Fill(eloss);
        else
          heloss2mu->Fill(eloss);
        hladder2id->Fill(float(ladder));
        hz2id->Fill(float(module));
        hthick2->Fill(dz);
        hlength2->Fill(y);
        hwidth2->Fill(x);
        if (abs(pid) == 13 && p > 1.)
          hphi2->Fill(phi);

        // check half modules
        //        if(ladder==8 || ladder==9 || ladder==24 || ladder==25 ) {
        // 	 hwidth2h->Fill(x);
        //        } else {
        //        }

        SimHitMap2[detId.rawId()].push_back((*isim));
        hglobr2->Fill(gloR);
        hglobz2->Fill(gloZ);
        hdetphi2->Fill(detPhi);

        // check up/down modules
        //if(pid!=11 && moduleDirectionUp) hladder2idUp->Fill(float(ladder));

      } else if (layer == 3) {
        //cout<<" layer "<<layer<<endl;
        totalNumOfSimHits3++;
        heloss3->Fill(eloss);
        if (abs(pid) == 11)
          heloss3e->Fill(eloss);
        else
          heloss3mu->Fill(eloss);

        hladder3id->Fill(float(ladder));
        hz3id->Fill(float(module));
        hthick3->Fill(dz);
        hlength3->Fill(y);
        hwidth3->Fill(x);
        if (abs(pid) == 13 && p > 1.)
          hphi3->Fill(phi);

        // check half modules
        //       if(ladder==11 || ladder==12 || ladder==33 || ladder==34 ) {
        // 	 hwidth3h->Fill(x);
        //        } else {
        //        }

        SimHitMap3[detId.rawId()].push_back((*isim));
        hglobr3->Fill(gloR);
        hglobz3->Fill(gloZ);
        hdetphi3->Fill(detPhi);
        // check up/down modules
        //if(pid!=11 && moduleDirectionUp) hladder3idUp->Fill(float(ladder));

      }  // layers
    }    // end fpix/bpix
  }

  hsimHitsPerLay1->Fill(float(totalNumOfSimHits1));
  hsimHitsPerLay2->Fill(float(totalNumOfSimHits2));
  hsimHitsPerLay3->Fill(float(totalNumOfSimHits3));
  hsimHits->Fill(float(totalNumOfSimHits));
  hsimHitsGood->Fill(float(goodHits));

  int numberOfDetUnits1 = SimHitMap1.size();
  int numberOfDetUnits2 = SimHitMap2.size();
  int numberOfDetUnits3 = SimHitMap3.size();
  int numberOfDetUnits = numberOfDetUnits1 + numberOfDetUnits2 + numberOfDetUnits3;

  if (PRINT)
    cout << " Number of full det-units = " << numberOfDetUnits << " total simhits = " << totalNumOfSimHits
         << " good simhits (pt>0.1) " << goodHits << endl;

  hdetsPerLay1->Fill(float(numberOfDetUnits1));
  hdetsPerLay2->Fill(float(numberOfDetUnits2));
  hdetsPerLay3->Fill(float(numberOfDetUnits3));

  numEvents++;
  numSimHits += totalNumOfSimHits;
  numSimHitsGood += goodHits;

  if (mode_ == "bpix") {
    map<unsigned int, vector<PSimHit>, less<unsigned int> >::iterator simhit_map_iterator;
    for (simhit_map_iterator = SimHitMap1.begin(); simhit_map_iterator != SimHitMap1.end(); simhit_map_iterator++) {
      //if(PRINT) cout << " Lay1 det = "<<simhit_map_iterator->first <<" simHits = "
      //	      << (simhit_map_iterator->second).size()<< endl;
      hsimHitsPerDet1->Fill(float((simhit_map_iterator->second).size()));
    }
    for (simhit_map_iterator = SimHitMap2.begin(); simhit_map_iterator != SimHitMap2.end(); simhit_map_iterator++) {
      //if(PRINT) cout << " Lay2 det = "<<simhit_map_iterator->first <<" simHits = "
      //	      << (simhit_map_iterator->second).size()<< endl;
      hsimHitsPerDet2->Fill(float((simhit_map_iterator->second).size()));
    }
    for (simhit_map_iterator = SimHitMap3.begin(); simhit_map_iterator != SimHitMap3.end(); simhit_map_iterator++) {
      //if(PRINT) cout << " Lay3 det = "<<simhit_map_iterator->first <<" simHits = "
      //	      << (simhit_map_iterator->second).size() << endl;
      hsimHitsPerDet3->Fill(float((simhit_map_iterator->second).size()));
    }
  }  // of bpix
}
// ------------ method called to at the end of the job  ------------
void PixelSimHitsTest::endJob() {
  cout << " End PixelSimHitsTest " << endl;

  numEvents++;
  numSimHits = numSimHits / float(numEvents);
  numSimHitsGood = numSimHitsGood / float(numEvents);

  cout << " Events " << numEvents << " simhits " << numSimHits << " simhits > 0.1gev " << numSimHitsGood << endl;

#ifdef CHECK_GEOM
  // To get module positions
  cout << " Module position" << endl;
  cout << " Layer Ladder Zindex    R      Z      Phi " << endl;
  for (int i = 0; i < 3; i++) {
    int max_lad = 0;
    if (i == 0)
      max_lad = 20;
    else if (i == 1)
      max_lad = 32;
    else if (i == 2)
      max_lad = 44;
    for (int n = 0; n < max_lad; n++) {
      for (int m = 0; m < 8; m++) {
        cout << "   " << i + 1 << "      " << n + 1 << "      " << m + 1 << "    " << modulePositionR[i][n][m] << " "
             << modulePositionZ[i][n][m] << " " << modulePositionPhi[i][n][m] << endl;
      }
    }
  }
#endif
}

//define this as a plug-in
DEFINE_FWK_MODULE(PixelSimHitsTest);
