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
//
//

#include "Validation/Mixing/interface/TestSuite.h"

// system include files
#include <memory>
#include <utility>

// user include files

#include "TFile.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"

using namespace edm;

TestSuite::TestSuite(const edm::ParameterSet& iConfig): filename_(iConfig.getParameter<std::string>("fileName")), bunchcr_(iConfig.getParameter<int>("BunchNr")), minbunch_(iConfig.getParameter<int>("minBunch")),maxbunch_(iConfig.getParameter<int>("maxBunch")),  dbe_(nullptr),
                                                        cfTrackToken_(consumes<CrossingFrame<SimTrack> > (
                                                            iConfig.getParameter<edm::InputTag>("cfTrackTag"))),
                                                        cfVertexToken_(consumes<CrossingFrame<SimTrack> > (
                                                            iConfig.getParameter<edm::InputTag>("cfVertexTag"))),
                                                        g4SimHits_Token_(consumes<CrossingFrame<PSimHit> > (
                                                            edm::InputTag("mix", "g4SimHitsTrackerHitsTECLowTof"))),
                                                        g4SimHits_Ecal_Token_(consumes<CrossingFrame<PCaloHit> > (
                                                            edm::InputTag("mix", "g4SimHitsEcalHitsEB"))),
                                                        g4SimHits_HCal_Token_(consumes<CrossingFrame<PCaloHit> > (
                                                            edm::InputTag("mix", "g4SimHitsHcalHits")))
{
  std::cout << "Constructed testSuite , bunchcr " << bunchcr_
            << " filename: " << filename_ << std::endl;
}


TestSuite::~TestSuite()
{

  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)
}

void TestSuite::beginJob() {

  // get hold of back-end interface
  dbe_ = Service<DQMStore>().operator->();
  dbe_->showDirStructure();
  dbe_->setCurrentFolder("MixingV/Mixing");
}

void TestSuite::endJob() {
  if (!filename_.empty() && dbe_ ) dbe_->save(filename_);
}



// ------------ method called to analyze the data  ------------
void
TestSuite::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  using namespace edm;

  // Get input
  edm::Handle<CrossingFrame<SimTrack> > cf_track;
  edm::Handle<CrossingFrame<SimVertex> > cf_vertex;
  edm::Handle<CrossingFrame<PSimHit> > cf_simhit;
  edm::Handle<CrossingFrame<PCaloHit> > cf_calohitEcal;
  edm::Handle<CrossingFrame<PCaloHit> > cf_calohitHcal;
  std::string subdetTracker("g4SimHitsTrackerHitsTECLowTof");
  std::string ecalsubdet("g4SimHitsEcalHitsEB");
  std::string hcalsubdet("g4SimHitsHcalHits");
  iEvent.getByToken(cfTrackToken_, cf_track);
  iEvent.getByToken(cfVertexToken_, cf_vertex);
  iEvent.getByToken(g4SimHits_Token_, cf_simhit);
  iEvent.getByToken(g4SimHits_Ecal_Token_, cf_calohitEcal);
  iEvent.getByToken(g4SimHits_HCal_Token_, cf_calohitHcal);

  // use MixCollection and its iterator
  // Please note that bunch() and getTrigger() are methods of the iterator itself
  // while operator-> points to the templated objects!!!!

  //track histo
  char histotracks[30],sighistotracks[30],histotracksindsig[30],histotracksind[30];
  sprintf(histotracks,"Tracks_bcr_%d",bunchcr_);
  sprintf(sighistotracks,"SignalTracks_bcr_%d",bunchcr_);
  sprintf(histotracksind,"VtxPointers_%d",bunchcr_);
  sprintf(histotracksindsig,"VtxPointers_signal_%d",bunchcr_);
  MonitorElement * trhist = dbe_->book1D(histotracks,"Bunchcrossings",maxbunch_-minbunch_+1,minbunch_,maxbunch_+1);
  MonitorElement * trhistsig = dbe_->book1D(sighistotracks,"Bunchcrossings",maxbunch_-minbunch_+1,minbunch_,maxbunch_+1);
  MonitorElement * trindhist = dbe_->book1D(histotracksind,"Track to Vertex indices",100,0,500);
  MonitorElement * trindhistsig = dbe_->book1D(histotracksindsig,"Signal Track to Vertex indices",100,0,500);
  std::unique_ptr<MixCollection<SimTrack> > col1(new MixCollection<SimTrack>(cf_track.product()));
  MixCollection<SimTrack>::iterator cfi1;
  for (cfi1=col1->begin(); cfi1!=col1->end();cfi1++) {
    if (cfi1.getTrigger()==0) {
      trhist->Fill(cfi1.bunch());
      trindhist->Fill(cfi1->vertIndex());
    } else {
      trindhistsig->Fill(cfi1->vertIndex());
      trhistsig->Fill(cfi1.bunch());
    }
  }


  //vertex histo
  char histovertices[30], sighistovertices[30],histovertexindices[30],histovertexindicessig[30];
  sprintf(histovertices,"Vertices_bcr_%d",bunchcr_);
  sprintf(sighistovertices,"SignalVertices_bcr_%d",bunchcr_);
  sprintf(histovertexindices,"TrackPointers_%d",bunchcr_);
  sprintf(histovertexindicessig,"TrackPointers_signal_%d",bunchcr_);
  MonitorElement * vtxhist = dbe_->book1D(histovertices,"Bunchcrossings",maxbunch_-minbunch_+1,minbunch_,maxbunch_+1);
  MonitorElement * vtxhistsig = dbe_->book1D(sighistovertices,"Bunchcrossings",maxbunch_-minbunch_+1,minbunch_,maxbunch_+1);
  MonitorElement * vtxindhist = dbe_->book1D(histovertexindices,"Vertex to Track Indices",100,0,300);
  MonitorElement * vtxindhistsig = dbe_->book1D(histovertexindicessig,"Signal Vertex to Track Indices",100,0,300);
  std::unique_ptr<MixCollection<SimVertex> > col2(new MixCollection<SimVertex>(cf_vertex.product()));
  MixCollection<SimVertex>::iterator cfi2;
  for (cfi2=col2->begin(); cfi2!=col2->end();cfi2++) {
    if (cfi2.getTrigger()==0) {
      vtxhist->Fill(cfi2.bunch());
      if (!cfi2->noParent()) 	vtxindhist->Fill(cfi2->parentIndex());
    } else {
      vtxhistsig->Fill(cfi2.bunch());
      if (!cfi2->noParent()) 	vtxindhistsig->Fill(cfi2->parentIndex());
    }
  }

  //tracker
  int bsp=cf_simhit->getBunchSpace();
  char tof[30];

  sprintf(tof,"TrackerHit_Tof_bcr_%d",bunchcr_);
  MonitorElement * tofhist = dbe_->book1D(tof,"TrackerHit_ToF",100,float(bsp*minbunch_),float(bsp*maxbunch_)+50.);
  sprintf(tof,"SignalTrackerHit_Tof_bcr_%d",bunchcr_);
  MonitorElement * tofhist_sig = dbe_->book1D(tof,"TrackerHit_ToF",100,float(bsp*minbunch_),float(bsp*maxbunch_)+50.);
  std::unique_ptr<MixCollection<PSimHit> > colsh(new MixCollection<PSimHit>(cf_simhit.product()));
  MixCollection<PSimHit>::iterator cfish;
  for (cfish=colsh->begin(); cfish!=colsh->end();cfish++) {
    if (cfish.getTrigger())  {
      tofhist_sig->Fill(cfish->timeOfFlight());
    }
    else  {
      tofhist->Fill(cfish->timeOfFlight());
    }
  }

  //Ecal
  sprintf(tof,"EcalEBHit_Tof_bcr_%d",bunchcr_);
  MonitorElement * tofecalhist = dbe_->book1D(tof,"EcalEBHit_ToF",100,float(bsp*minbunch_),float(bsp*maxbunch_)+50.);
  sprintf(tof,"SignalEcalEBHit_Tof_bcr_%d",bunchcr_);
  MonitorElement * tofecalhist_sig = dbe_->book1D(tof,"EcalEBHit_ToF",100,float(bsp*minbunch_),float(bsp*maxbunch_)+50.);
  //    std::string ecalsubdet("EcalHitsEB");
  std::unique_ptr<MixCollection<PCaloHit> > colecal(new MixCollection<PCaloHit>(cf_calohitEcal.product()));
  MixCollection<PCaloHit>::iterator cfiecal;
  for (cfiecal=colecal->begin(); cfiecal!=colecal->end();cfiecal++) {
    if (cfiecal.getTrigger())    tofecalhist_sig->Fill(cfiecal->time());
    else    tofecalhist->Fill(cfiecal->time());
  }

  // Hcal
  sprintf(tof,"HcalHit_Tof_bcr_%d",bunchcr_);
  MonitorElement * tofhcalhist = dbe_->book1D(tof,"HcalHit_ToF",100,float(bsp*minbunch_),float(bsp*maxbunch_)+50.);
  sprintf(tof,"SignalHcalHit_Tof_bcr_%d",bunchcr_);
  MonitorElement * tofhcalhist_sig = dbe_->book1D(tof,"HcalHit_ToF",100,float(bsp*minbunch_),float(bsp*maxbunch_)+50.);
  //    std::string hcalsubdet("HcalHits");
  std::unique_ptr<MixCollection<PCaloHit> > colhcal(new MixCollection<PCaloHit>(cf_calohitHcal.product()));
  MixCollection<PCaloHit>::iterator cfihcal;

  for (cfihcal=colhcal->begin(); cfihcal!=colhcal->end();cfihcal++) {
    if (cfihcal.getTrigger())  tofhcalhist_sig->Fill(cfihcal->time());
    else  tofhcalhist->Fill(cfihcal->time());
  }
}

