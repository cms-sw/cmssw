// -*- C++ -*-
//
// Package:    PixelDigisTest
// Class:      PixelDigisTest
// 
/**\class PixelDigisTest PixelDigisTest.cc 

 Description: Test pixel digis. 
 Barrel & Forward digis. Uses root histos.
 Works with CMSSW_0_9_0_pre3 
 Adopted for the new simLinks. 

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  d.k.
//         Created:  Jan CET 2006
// $Id: PixelDigisTest.cc,v 1.11 2006/08/07 13:09:42 dkotlins Exp $
//
//
// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/Handle.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/InputTag.h"

// my includes
//#include "Geometry/CommonDetUnit/interface/GeomDetUnit.h"

#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h" 

#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/TrackerGeometryBuilder/interface/PixelGeomDetUnit.h"
#include "Geometry/TrackerGeometryBuilder/interface/PixelGeomDetType.h"

#include "DataFormats/SiPixelDetId/interface/PixelSubdetector.h"
#include "DataFormats/SiPixelDetId/interface/PXBDetId.h"
#include "DataFormats/SiPixelDetId/interface/PXFDetId.h"

// data formats
#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/SiPixelDigi/interface/PixelDigi.h"
//#include "DataFormats/SiPixelDigi/interface/PixelDigiCollection.h"
#include "DataFormats/DetId/interface/DetId.h"

#include "SimDataFormats/TrackerDigiSimLink/interface/PixelDigiSimLink.h"

// For the big pixel recongnition
#include "Geometry/TrackerTopology/interface/RectangularPixelTopology.h"

// for simulated Tracker hits
//#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"

// For ROOT
#include <TROOT.h>
#include <TChain.h>
#include <TFile.h>
#include <TF1.h>
#include <TH2F.h>
#include <TH1F.h>

using namespace std;

//
// class decleration
//

class PixelDigisTest : public edm::EDAnalyzer {

public:

  explicit PixelDigisTest(const edm::ParameterSet&);
  ~PixelDigisTest();
  virtual void beginJob(const edm::EventSetup& iSetup);
  virtual void analyze(const edm::Event&, const edm::EventSetup&);
  virtual void endJob(); 

private:
  // ----------member data ---------------------------
  bool PRINT;

  TFile* hFile;
  TH1F *hdetunit;
  TH1F *heloss1,*heloss2, *heloss3;
  TH1F *helossF1,*helossF2;
  TH1F *hpixid,*hpixsubid,*hlayerid,*hladder1id,*hladder2id,*hladder3id,
    *hz1id,*hz2id,*hz3id;
  TH1F *hcols1,*hcols2,*hcols3,*hrows1,*hrows2,*hrows3;
  TH1F *hcolsF1,*hcolsF2,*hcolsF3,*hrowsF1,*hrowsF2,*hrowsF3;
  TH1F *hdigisPerDet1,*hdigisPerDet2,*hdigisPerDet3;
  TH1F *hdigisPerLay1,*hdigisPerLay2,*hdigisPerLay3;
  TH1F *hdetsPerLay1,*hdetsPerLay2,*hdetsPerLay3;
  TH1F *hdigisPerDetF1,*hdigisPerDetF2,*hdigisPerDetF3;
  TH1F *hdigisPerLayF1,*hdigisPerLayF2,*hdigisPerLayF3;
  TH1F *hdetsPerLayF1,*hdetsPerLayF2,*hdetsPerLayF3;
  TH1F *hdetr, *hdetz, *hdetrF, *hdetzF;
  TH1F *hcolsB,  *hrowsB,  *hcolsF,  *hrowsF;
  TH1F *hcols1big, *hrows1big, *heloss1bigx, *heloss1bigy;
  TH1F *hsimlinks, *hfract;

  TH2F *htest, *htest2;

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
PixelDigisTest::PixelDigisTest(const edm::ParameterSet& iConfig) {
  //We put this here for the moment since there is no better place 
  //edm::Service<MonitorDaemon> daemon;
  //daemon.operator->();

  PRINT = iConfig.getUntrackedParameter<bool>("Verbosity",false);
  src_ =  iConfig.getParameter<edm::InputTag>( "src" );
  cout<<" Construct PixelDigisTest "<<endl;
}


PixelDigisTest::~PixelDigisTest() {
 
  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)
  cout<<" Destroy PixelDigisTest "<<endl;

}

//
// member functions
//
// ------------ method called at the begining   ------------
void PixelDigisTest::beginJob(const edm::EventSetup& iSetup) {

   using namespace edm;
   cout << "Initialize PixelDigisTest " <<endl;


   // put here whatever you want to do at the beginning of the job
    hFile = new TFile ( "histo.root", "RECREATE" );
     
    hdetunit = new TH1F( "hdetunit", "Det unit", 1000,
                              302000000.,302300000.);
    hpixid = new TH1F( "hpixid", "Pix det id", 10, 0., 10.);
    hpixsubid = new TH1F( "hpixsubid", "Pix Barrel id", 10, 0., 10.);
    hlayerid = new TH1F( "hlayerid", "Pix layer id", 10, 0., 10.);
    hladder1id = new TH1F( "hladder1id", "Ladder L1 id", 50, 0., 50.);
    hladder2id = new TH1F( "hladder2id", "Ladder L2 id", 50, 0., 50.);
    hladder3id = new TH1F( "hladder3id", "Ladder L3 id", 50, 0., 50.);
    hz1id = new TH1F( "hz1id", "Z-index id L1", 10, 0., 10.);
    hz2id = new TH1F( "hz2id", "Z-index id L2", 10, 0., 10.);
    hz3id = new TH1F( "hz3id", "Z-index id L3", 10, 0., 10.);
 
    hdigisPerDet1 = new TH1F( "hdigisPerDet1", "Digis per det l1", 
			      200, -0.5, 199.5);
    hdigisPerDet2 = new TH1F( "hdigisPerDet2", "Digis per det l2", 
			      200, -0.5, 199.5);
    hdigisPerDet3 = new TH1F( "hdigisPerDet3", "Digis per det l3", 
			      200, -0.5, 199.5);
    hdigisPerLay1 = new TH1F( "hdigisPerLay1", "Digis per layer l1", 
			      2000, -0.5, 3999.5);
    hdigisPerLay2 = new TH1F( "hdigisPerLay2", "Digis per layer l2", 
			      2000, -0.5, 3999.5);
    hdigisPerLay3 = new TH1F( "hdigisPerLay3", "Digis per layer l3", 
			      2000, -0.5, 3999.5);
    hdetsPerLay1 = new TH1F( "hdetsPerLay1", "Full dets per layer l1", 
			      161, -0.5, 160.5);
    hdetsPerLay3 = new TH1F( "hdetsPerLay3", "Full dets per layer l3", 
			      353, -0.5, 352.5);
    hdetsPerLay2 = new TH1F( "hdetsPerLay2", "Full dets per layer l2", 
			      257, -0.5, 256.5);

    hdigisPerDetF1 = new TH1F( "hdigisPerDetF1", "Digis per det d1", 
			      200, -0.5, 199.5);
    hdigisPerDetF2 = new TH1F( "hdigisPerDetF2", "Digis per det d2", 
			      200, -0.5, 199.5);
    hdigisPerLayF1 = new TH1F( "hdigisPerLayF1", "Digis per layer d1", 
			      2000, -0.5, 1999.5);
    hdigisPerLayF2 = new TH1F( "hdigisPerLayF2", "Digis per layer d2", 
			      2000, -0.5, 1999.5);
    hdetsPerLayF1 = new TH1F( "hdetsPerLayF1", "Full dets per layer d1", 
			      161, -0.5, 160.5);
    hdetsPerLayF2 = new TH1F( "hdetsPerLayF2", "Full dets per layer d2", 
			      257, -0.5, 256.5);

    heloss1 = new TH1F( "heloss1", "Pix charge l1", 100, 0., 300.);
    heloss2 = new TH1F( "heloss2", "Pix charge l2", 100, 0., 300.);
    heloss3 = new TH1F( "heloss3", "Pix charge l3", 100, 0., 300.);
    heloss1bigx = new TH1F( "heloss1bigx", "L1 big-x pix", 100, 0., 300.);
    heloss1bigy = new TH1F( "heloss1bigy", "L1 big-y pix", 100, 0., 300.);

    hcols1 = new TH1F( "hcols1", "Layer 1 cols", 500,-1.5,498.5);
    hcols2 = new TH1F( "hcols2", "Layer 2 cols", 500,-1.5,498.5);
    hcols3 = new TH1F( "hcols3", "Layer 3 cols", 500,-1.5,498.5);
    hcols1big = new TH1F( "hcols1big", "Layer 1 big cols", 500,-1.5,498.5);
 
    hrows1 = new TH1F( "hrows1", "Layer 1 rows", 200,-1.5,198.5);
    hrows2 = new TH1F( "hrows2", "Layer 2 rows", 200,-1.5,198.5);
    hrows3 = new TH1F( "hrows3", "layer 3 rows", 200,-1.5,198.5);
    hrows1big = new TH1F( "hrows1big", "Layer 1 big rows", 200,-1.5,198.5);
 
    helossF1 = new TH1F( "helossF1", "Pix charge d1", 100, 0., 300.);
    helossF2 = new TH1F( "helossF2", "Pix charge d2", 100, 0., 300.);
    hcolsF1 = new TH1F( "hcolsF1", "Disk 1 cols", 500,-1.5,498.5);
    hcolsF2 = new TH1F( "hcolsF2", "Disk 2 cols", 500,-1.5,498.5);
    hrowsF1 = new TH1F( "hrowsF1", "Disk 1 rows", 200,-1.5,198.5);
    hrowsF2 = new TH1F( "hrowsF2", "Disk 2 rows", 200,-1.5,198.5);



    hdetr = new TH1F("hdetr","det r",150,0.,15.);
    hdetz = new TH1F("hdetz","det z",520,-26.,26.);
    hdetrF = new TH1F("hdetrF","det r",150,0.,15.);
    hdetzF = new TH1F("hdetzF","det z",700,-70.,70.);

    hcolsB = new TH1F("hcolsB","cols per bar det",450,0.,450.);
    hrowsB = new TH1F("hrowsB","rows per bar det",200,0.,200.);
    hcolsF = new TH1F("hcolsF","cols per for det",300,0.,300.);
    hrowsF = new TH1F("hrowsF","rows per for det",200,0.,200.);

    hsimlinks = new TH1F("hsimlinks"," track ids",200,0.,200.);
    hfract = new TH1F("hfract"," track rractions",100,0.,1.);

    htest = new TH2F("htest"," ",10,0.,10.,20,0.,20.);
    htest2 = new TH2F("htest2"," ",10,0.,10.,300,0.,300.);

}

// ------------ method called to produce the data  ------------
void PixelDigisTest::analyze(const edm::Event& iEvent, 
			   const edm::EventSetup& iSetup) {
  using namespace edm;
  if(PRINT) cout<<" Analyze PixelDigisTest "<<endl;

    // Get digis
  edm::Handle< edm::DetSetVector<PixelDigi> > pixelDigis;
  iEvent.getByLabel( src_ , pixelDigis);

  // Get simlink data
  //edm::Handle<PixelDigiSimLinkCollection> pixelSimLinks;
  //iEvent.getByLabel("siPixelDigis", pixelSimLinks);
  edm::Handle< edm::DetSetVector<PixelDigiSimLink> > pixelSimLinks;
  iEvent.getByLabel( src_ ,   pixelSimLinks);

  // Get event setup (to get global transformation)
  edm::ESHandle<TrackerGeometry> geom;
  iSetup.get<TrackerDigiGeometryRecord>().get( geom );
  const TrackerGeometry& theTracker(*geom);

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
  for(DSViter = pixelDigis->begin(); DSViter != pixelDigis->end(); DSViter++) {

    unsigned int detid = DSViter->id; // = rawid
    DetId detId(detid);
    //const GeomDetUnit      * geoUnit = geom->idToDetUnit( detId );
    //const PixelGeomDetUnit * pixDet  = dynamic_cast<const PixelGeomDetUnit*>(geoUnit);
    unsigned int detType=detId.det(); // det type, tracker=1
    unsigned int subid=detId.subdetId(); //subdetector type, barrel=1
    
    if(PRINT) 
      cout<<"Det: "<<detId.rawId()<<" "<<detId.null()<<" "<<detType<<" "<<subid<<endl;
    
    hdetunit->Fill(float(detid));
    hpixid->Fill(float(detType));
    hpixsubid->Fill(float(subid));
    
    if(detType!=1) continue; // look only at tracker
    ++numberOfDetUnits;
    
    // Get the geom-detector 
    const PixelGeomDetUnit * theGeomDet = 
      dynamic_cast<const PixelGeomDetUnit*> ( theTracker.idToDet(detId) );
    double detZ = theGeomDet->surface().position().z();
    double detR = theGeomDet->surface().position().perp();
    //const BoundPlane plane = theGeomDet->surface(); // does not work
    
    double detThick = theGeomDet->specificSurface().bounds().thickness();
    int cols = theGeomDet->specificTopology().ncolumns();
    int rows = theGeomDet->specificTopology().nrows();
    float pitchX = theGeomDet->specificTopology().pitch().first;
    float pitchY = theGeomDet->specificTopology().pitch().second;
    
    unsigned int layer=0;
    unsigned int ladder=0;
    unsigned int zindex=0;
    unsigned int disk=0; //1,2,3

    // Subdet it, pix barrel=1, forward=2
    if(subid==2) {   // forward

      hdetrF->Fill(detR);
      hdetzF->Fill(detZ);
      hcolsF->Fill(float(cols));
      hrowsF->Fill(float(rows));

      PXFDetId pdetId = PXFDetId(detid);
      disk=pdetId.disk(); //1,2,3
      unsigned int blade=pdetId.blade(); //1-24
      unsigned int zindex=pdetId.module(); //
      unsigned int side=pdetId.side(); //size=1 for -z, 2 for +z
      unsigned int panel=pdetId.panel(); //panel=1
      
      if(PRINT) {
	cout<<"Forward det "<<subid<<", disk "<<disk<<", blade "
		    <<blade<<", module "<<zindex<<", side "<<side<<", panel "
		    <<panel<<" pos = "<<detZ<<" "<<detR<<endl;
	cout<<" col/row, pitch "<<cols<<" "<<rows<<" "
		    <<pitchX<<" "<<pitchY<<endl;
      }

    } else if(subid == 1) { // Barrel 
      
      hdetr->Fill(detR);
      hdetz->Fill(detZ);
      hcolsB->Fill(float(cols));
      hrowsB->Fill(float(rows));
      
      PXBDetId pdetId = PXBDetId(detid);
      unsigned int detTypeP=pdetId.det();
      unsigned int subidP=pdetId.subdetId();
      // Barell layer = 1,2,3
      layer=pdetId.layer();
      // Barrel ladder id 1-20,32,44.
      ladder=pdetId.ladder();
      // Barrel Z-index=1,8
      zindex=pdetId.module();
      if(PRINT) { 
	cout<<" Barrel det, z/r "<<detZ<<" "<<detR<<" thick "<<detThick<<" "
	    <<" layer, ladder, module "
	    <<layer<<" "<<ladder<<" "<<zindex<<endl;
	cout<<" col/row, pitch "<<cols<<" "<<rows<<" "
	    <<pitchX<<" "<<pitchY<<endl;
      }      
      hlayerid->Fill(float(layer));

    } // end fb-bar

    // Some histos
    if(layer==1) {
      hladder1id->Fill(float(ladder));
      hz1id->Fill(float(zindex));
      ++numberOfDetUnits1;
      numOfDigisPerDet1=0;
      
    } else if(layer==2) {
      hladder2id->Fill(float(ladder));
      hz2id->Fill(float(zindex));
      ++numberOfDetUnits2;
      numOfDigisPerDet2=0;

    } else if(layer==3) {
      hladder3id->Fill(float(ladder));
      hz3id->Fill(float(zindex));
      ++numberOfDetUnits3;
      numOfDigisPerDet3=0;

    } else if(disk==1) {
      ++numberOfDetUnitsF1;
      numOfDigisPerDetF1=0;
 
   } else if(disk==2) {
      ++numberOfDetUnitsF2;
      numOfDigisPerDetF2=0;
   }
      

      // Has to be changed 
//      const PixelDigiSimLinkCollection::Range simLinkRange = 
//        pixelSimLinks->get(detid);
//      for(PixelDigiSimLinkCollection::ContainerIterator 
// 	   it = simLinkRange.first; it != simLinkRange.second; ++it) { 
//      }

    int numberOfSimLinks = 0;
    edm::DetSetVector<PixelDigiSimLink>::const_iterator 
      isearch = pixelSimLinks->find(detid);

    if(isearch != pixelSimLinks->end()) {      //if it is not empty
      edm::DetSet<PixelDigiSimLink> link_detset = (*pixelSimLinks)[detid];
      edm::DetSet<PixelDigiSimLink>::const_iterator it;
      // Loop over DigisSimLink in this det unit
      for(it = link_detset.data.begin(); 
	  it != link_detset.data.end(); it++) {
	
	numberOfSimLinks++;
	// these methods should be declared const, fixed by M.P.
	// wait for next releasse and then uncomment
	unsigned int chan = it->channel();
	unsigned int simTrack = it->SimTrackId();
	float frac = it->fraction();
	hsimlinks->Fill(float(simTrack));
	hfract->Fill(frac);
	if(PRINT) cout<<" Sim link "<<numberOfSimLinks<<" "<<chan<<" "
		      <<simTrack<<" "<<frac<<endl;
      } // end simlink det loop

    } // end simlink if

      unsigned int numberOfDigis = 0;

      // Look at digis now
      edm::DetSet<PixelDigi>::const_iterator  di;
      for(di = DSViter->data.begin(); di != DSViter->data.end(); di++) {
	//for(di = begin; di != end; di++) {
	
	numberOfDigis++;
	totalNumOfDigis++;
       int adc = di->adc();    // charge, modifued to unsiged short 
       int col = di->column(); // column 
       int row = di->row();    // row
       //int tof = di->time();    // tof always 0, method deleted

       // channel index needed to look for the simlink to simtracks
       int channel = PixelChannelIdentifier::pixelToChannel(row,col);
       if(PRINT) cout <<numberOfDigis<< " Col: " << col << " Row: " << row 
		      << " ADC: " << adc <<" channel = "<<channel<<endl;

       if(col>415) cout<<" Error: column index too large "<<col<<endl;
       if(row>159) cout<<" Error: row index too large "<<row<<endl;

       if(layer==1) {
	 heloss1->Fill(float(adc));
	 hcols1->Fill(float(col));
	 hrows1->Fill(float(row));
	 totalNumOfDigis1++;
	 htest2->Fill(float(zindex),float(adc));
	 numOfDigisPerDet1++;
	 if(RectangularPixelTopology::isItBigPixelInX(row)) {
	   //cout<<" big in X "<<row<<endl;
	   heloss1bigx->Fill(float(adc));
	   hrows1big->Fill(float(row));
	 }
	 if(RectangularPixelTopology::isItBigPixelInY(col)) {
	   //cout<<" big in Y "<<col<<endl;
	   heloss1bigy->Fill(float(adc));
	   hcols1big->Fill(float(col));
	 }

       } else if(layer==2) {
	 heloss2->Fill(float(adc));
	 hcols2->Fill(float(col));
	 hrows2->Fill(float(row));
	 totalNumOfDigis2++;
	 numOfDigisPerDet2++;

       } else if(layer==3) {
	 heloss3->Fill(float(adc));
	 hcols3->Fill(float(col));
	 hrows3->Fill(float(row));
	 totalNumOfDigis3++;
	 numOfDigisPerDet3++;

       } else if(disk==1) {
	 helossF1->Fill(float(adc));
	 hcolsF1->Fill(float(col));
	 hrowsF1->Fill(float(row));
	 totalNumOfDigisF1++;
	 numOfDigisPerDetF1++;

       } else if(disk==2) {
	 helossF2->Fill(float(adc));
	 hcolsF2->Fill(float(col));
	 hrowsF2->Fill(float(row));
	 totalNumOfDigisF2++;
	 numOfDigisPerDetF2++;
       } // end if layer
        
     } // end for digis
      //if(PRINT) 
      //cout<<" for det "<<detid<<" digis = "<<numberOfDigis<<endl;

     if(layer==1) {
       hdigisPerDet1->Fill(float(numOfDigisPerDet1));
       htest->Fill(float(zindex),float(numOfDigisPerDet1));
     } else if(layer==2) hdigisPerDet2->Fill(float(numOfDigisPerDet2));
     else if(layer==3) hdigisPerDet3->Fill(float(numOfDigisPerDet3));
     else if(disk==1) hdigisPerDetF1->Fill(float(numOfDigisPerDetF1));
     else if(disk==2) hdigisPerDetF2->Fill(float(numOfDigisPerDetF2));

   } // end for det-units

   if(PRINT) 
     cout << " Number of full det-units = " <<numberOfDetUnits
	  <<" total digis = "<<totalNumOfDigis<<endl;

   hdigisPerLay1 ->Fill(float(totalNumOfDigis1));
   hdigisPerLay2 ->Fill(float(totalNumOfDigis2));
   hdigisPerLay3 ->Fill(float(totalNumOfDigis3));
   hdetsPerLay1 ->Fill(float(numberOfDetUnits1));
   hdetsPerLay2 ->Fill(float(numberOfDetUnits2));
   hdetsPerLay3 ->Fill(float(numberOfDetUnits3));

   hdigisPerLayF1 ->Fill(float(totalNumOfDigisF1));
   hdigisPerLayF2 ->Fill(float(totalNumOfDigisF2));
   hdetsPerLayF1 ->Fill(float(numberOfDetUnitsF1));
   hdetsPerLayF2 ->Fill(float(numberOfDetUnitsF2));

}
// ------------ method called to at the end of the job  ------------
void PixelDigisTest::endJob(){
  cout << " End PixelDigisTest " << endl;
  hFile->Write();
  hFile->Close();
}

//define this as a plug-in
DEFINE_FWK_MODULE(PixelDigisTest)
