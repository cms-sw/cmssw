// -*- C++ -*-
//
// Class:      TestSuite
// 
/**\class TestSuite

 Description: test suite for Mixing Module

*/
//
// Original Author:  Ursula Berthon
//         Created:  Fri Sep 23 11:38:38 CEST 2005
// $Id: TestSuite.cc,v 1.7 2006/03/07 13:39:16 uberthon Exp $
//
//


// system include files
#include <memory>
#include <utility>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "SimDataFormats/CrossingFrame/interface/CrossingFrame.h"
#include "SimDataFormats/CrossingFrame/interface/MixCollection.h"
#include "SimGeneral/MixingModule/interface/TestSuite.h"
#include "SimDataFormats/TrackingHit/interface/PSimHit.h"
#include "TH1I.h"

using namespace edm;

TestSuite::TestSuite(const edm::ParameterSet& iConfig): filename_(iConfig.getParameter<std::string>("fileName")), bunchcr_(iConfig.getParameter<int>("BunchNr")), minbunch_(iConfig.getParameter<int>("minBunch")),maxbunch_(iConfig.getParameter<int>("maxBunch"))
{
  std::cout << "Constructed testSuite , bunchcr "<<bunchcr_<<" filename: "<<filename_<<std::endl;

  histfile_ = new TFile(filename_.c_str(),"UPDATE");
}


TestSuite::~TestSuite()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)
  histfile_->Write();
  histfile_->Close();

}


//
// member functions
//

// ------------ method called to analyze the data  ------------
void
TestSuite::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   using namespace edm;

// Get input
    edm::Handle<CrossingFrame> cf;
    iEvent.getByType(cf);

// use MixCollection and its iterator
// Please note that bunch() and getTrigger() are methods of the iterator itself
// while operator-> points to the templated objects!!!!

//track histo
    char histotracks[30],sighistotracks[30],histotracksindsig[30],histotracksind[30];
    sprintf(histotracks,"Tracks_bcr_%d",bunchcr_);
    sprintf(sighistotracks,"SignalTracks_bcr_%d",bunchcr_);
    sprintf(histotracksind,"VtxPointers_%d",bunchcr_);
    sprintf(histotracksindsig,"VtxPointers_signal_%d",bunchcr_);
    TH1I * trhist = new TH1I(histotracks,"Bunchcrossings",maxbunch_-minbunch_+1,minbunch_,maxbunch_+1);
    TH1I * trhistsig = new TH1I(sighistotracks,"Bunchcrossings",maxbunch_-minbunch_+1,minbunch_,maxbunch_+1);
    TH1I * trindhist = new TH1I(histotracksind,"Track to Vertex indices",100,0,500);
    TH1I * trindhistsig = new TH1I(histotracksindsig,"Signal Track to Vertex indices",100,0,500);
    std::auto_ptr<MixCollection<EmbdSimTrack> > col1(new MixCollection<EmbdSimTrack>(cf.product()));
    MixCollection<EmbdSimTrack>::iterator cfi1;
    for (cfi1=col1->begin(); cfi1!=col1->end();cfi1++) {
       if (cfi1.getTrigger()) {
	 trhist->Fill(cfi1.bunch());
	 trindhist->Fill(cfi1->vertIndex());
       } else {
	 trindhistsig->Fill(cfi1->vertIndex());
	 trhistsig->Fill(cfi1.bunch());
       }
     }
     std::cout <<"[OVAL] Mean of track pileup histo (bcrossings) "<<trhist->GetMean()<<std::endl;
     std::cout <<"[OVAL] Mean of track signal histo (bcrossings) "<<trhistsig->GetMean()<<std::endl;
     std::cout <<"[OVAL] Mean of track pileup histo (vtx indices) "<<trindhist->GetMean()<<std::endl;
     std::cout <<"[OVAL] Mean of track signal histo (vtx indices) "<<trindhistsig->GetMean()<<std::endl;

//vertex histo
    char histovertices[30], sighistovertices[30],histovertexindices[30],histovertexindicessig[30];
    sprintf(histovertices,"Vertices_bcr_%d",bunchcr_);
    sprintf(sighistovertices,"SignalVertices_bcr_%d",bunchcr_);
    sprintf(histovertexindices,"TrackPointers_%d",bunchcr_);
    sprintf(histovertexindicessig,"TrackPointers_signal_%d",bunchcr_);
    TH1I * vtxhist = new TH1I(histovertices,"Bunchcrossings",maxbunch_-minbunch_+1,minbunch_,maxbunch_+1);
    TH1I * vtxhistsig = new TH1I(sighistovertices,"Bunchcrossings",maxbunch_-minbunch_+1,minbunch_,maxbunch_+1);
    TH1I * vtxindhist = new TH1I(histovertexindices,"Vertex to Track Indices",100,0,300);
    TH1I * vtxindhistsig = new TH1I(histovertexindicessig,"Signal Vertex to Track Indices",100,0,300);
    std::auto_ptr<MixCollection<EmbdSimVertex> > col2(new MixCollection<EmbdSimVertex>(cf.product()));
    MixCollection<EmbdSimVertex>::iterator cfi2;
    for (cfi2=col2->begin(); cfi2!=col2->end();cfi2++) {
	if (cfi2.getTrigger()) {
	  vtxhist->Fill(cfi2.bunch());
	  if (!cfi2->noParent()) 	vtxindhist->Fill(cfi2->parentIndex());
	} else {
	  vtxhistsig->Fill(cfi2.bunch());
	  if (!cfi2->noParent()) 	vtxindhistsig->Fill(cfi2->parentIndex());
	}
    }
     std::cout <<"[OVAL] Mean of vertex pileup histo (bcrossings) "<<vtxhist->GetMean()<<std::endl;
     std::cout <<"[OVAL] Mean of vertex signal histo (bcrossings) "<<vtxhistsig->GetMean()<<std::endl;
     std::cout <<"[OVAL] Mean of vertex pileup histo (track indices) "<<vtxindhist->GetMean()<<std::endl;
     std::cout <<"[OVAL] Mean of vertex signal histo (track indices) "<<vtxindhistsig->GetMean()<<std::endl;

	
    int bsp=cf->getBunchSpace();
    char tof[30];

    //tracker
    sprintf(tof,"TrackerHit_Tof_bcr_%d",bunchcr_);
    TH1I * tofhist = new TH1I(tof,"TrackerHit_ToF",100,float(bsp*minbunch_),float(bsp*maxbunch_)+50.);
    sprintf(tof,"SignalTrackerHit_Tof_bcr_%d",bunchcr_);
    TH1I * tofhist_sig = new TH1I(tof,"TrackerHit_ToF",100,float(bsp*minbunch_),float(bsp*maxbunch_)+50.);
    std::string subdet("TrackerHitsTECLowTof");
    std::auto_ptr<MixCollection<PSimHit> > colsh(new MixCollection<PSimHit>(cf.product(),std::string(subdet)));
    MixCollection<PSimHit>::iterator cfish;
    for (cfish=colsh->begin(); cfish!=colsh->end();cfish++) {
      if (cfish.getTrigger())  tofhist->Fill(cfish->timeOfFlight());
      else  tofhist_sig->Fill(cfish->timeOfFlight());
    }
    std::cout <<"[OVAL] Mean of tracker pileup histo (ToF) "<<tofhist->GetMean()<<std::endl;
    std::cout <<"[OVAL] Mean of tracker signal histo (ToF) "<<tofhist_sig->GetMean()<<std::endl;

    //Ecal
    sprintf(tof,"EcalEBHit_Tof_bcr_%d",bunchcr_);
    TH1I * tofecalhist = new TH1I(tof,"EcalEBHit_ToF",100,float(bsp*minbunch_),float(bsp*maxbunch_)+50.);
    sprintf(tof,"SignalEcalEBHit_Tof_bcr_%d",bunchcr_);
    TH1I * tofecalhist_sig = new TH1I(tof,"EcalEBHit_ToF",100,float(bsp*minbunch_),float(bsp*maxbunch_)+50.);
    std::string ecalsubdet("EcalHitsEB");
    std::auto_ptr<MixCollection<PCaloHit> > colecal(new MixCollection<PCaloHit>(cf.product(),std::string(ecalsubdet)));
    MixCollection<PCaloHit>::iterator cfiecal;
    for (cfiecal=colecal->begin(); cfiecal!=colecal->end();cfiecal++) {
      if (cfiecal.getTrigger())    tofecalhist->Fill(cfiecal->time());
      else    tofecalhist_sig->Fill(cfiecal->time());
    }
    std::cout <<"[OVAL] Mean of Ecal pileup histo (ToF) "<<tofecalhist->GetMean()<<std::endl;
    std::cout <<"[OVAL] Mean of Ecal signal histo (ToF) "<<tofecalhist_sig->GetMean()<<std::endl;

    // Hcal
    sprintf(tof,"HcalHit_Tof_bcr_%d",bunchcr_);
    TH1I * tofhcalhist = new TH1I(tof,"HcalHit_ToF",100,float(bsp*minbunch_),float(bsp*maxbunch_)+50.);
    sprintf(tof,"SignalHcalHit_Tof_bcr_%d",bunchcr_);
    TH1I * tofhcalhist_sig = new TH1I(tof,"HcalHit_ToF",100,float(bsp*minbunch_),float(bsp*maxbunch_)+50.);
    std::string hcalsubdet("HcalHits");
    std::auto_ptr<MixCollection<PCaloHit> > colhcal(new MixCollection<PCaloHit>(cf.product(),std::string(hcalsubdet)));
    MixCollection<PCaloHit>::iterator cfihcal;
    for (cfihcal=colhcal->begin(); cfihcal!=colhcal->end();cfihcal++) {
      if (cfihcal.getTrigger())  tofhcalhist->Fill(cfihcal->time());
      else  tofhcalhist_sig->Fill(cfihcal->time());
    }
    std::cout <<"[OVAL] Mean of Hcal pileup histo (ToF) "<<tofhcalhist->GetMean()<<std::endl;
    std::cout <<"[OVAL] Mean of Hcal signal histo (ToF) "<<tofhcalhist_sig->GetMean()<<std::endl;
}

