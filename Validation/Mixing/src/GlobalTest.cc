// -*- C++ -*-
//
// Class:      GlobalTest
// 
/**\class TestSuite

 Description: global physics test for Mixing Module

*/
//
// Original Author:  Ursula Berthon
//         Created:  Fri Sep 23 11:38:38 CEST 2005
// $Id: GlobalTest.cc,v 1.11 2012/10/10 14:39:02 wdd Exp $
//
//


// system include files
#include <memory>
#include <utility>

#include <string>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "SimDataFormats/CrossingFrame/interface/CrossingFrame.h"
#include "SimDataFormats/CrossingFrame/interface/MixCollection.h"
#include "SimDataFormats/TrackingHit/interface/PSimHit.h"

#include "SimDataFormats/CaloHit/interface/PCaloHitContainer.h"
#include "SimDataFormats/Track/interface/SimTrackContainer.h"

#include "Validation/Mixing/interface/GlobalTest.h"
#include "TFile.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"

using namespace edm;

GlobalTest::GlobalTest(const edm::ParameterSet& iConfig):
filename_(iConfig.getParameter<std::string>("fileName")),
minbunch_(iConfig.getParameter<int>("minBunch")),maxbunch_(iConfig.getParameter<int>("maxBunch")), dbe_(0),
cfTrackTag_(iConfig.getParameter<edm::InputTag>("cfTrackTag")),
cfVertexTag_(iConfig.getParameter<edm::InputTag>("cfVertexTag"))
{
  std::cout << "Constructed GlobalTest, filename: "<<filename_<<" minbunch: "<<minbunch_<<", maxbunch: "<<maxbunch_<<std::endl;

}

GlobalTest::~GlobalTest()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)
}

void GlobalTest::beginJob() {
  using namespace std;
  
  // get hold of back-end interface
  dbe_ = Service<DQMStore>().operator->(); 
  dbe_->showDirStructure();
  dbe_->setCurrentFolder("MixingV/Mixing");
  //book histos
  std::string NrPileupEvts = "NrPileupEvts";
  size_t NrPileupEvtsSize = NrPileupEvts.size() + 1;
  std::string NrVertices = "NrVertices";
  size_t NrVerticesSize = NrVertices.size() + 1;
  std::string NrTracks = "NrTracks";
  size_t NrTracksSize = NrTracks.size() + 1;
  std::string TrackPartId = "TrackPartId";
  size_t TrackPartIdSize = TrackPartId.size() + 1;
  std::string CaloEnergyEB = "CaloEnergyEB";
  size_t CaloEnergyEBSize = CaloEnergyEB.size() + 1;
  std::string CaloEnergyEE = "CaloEnergyEE";
  size_t CaloEnergyEESize = CaloEnergyEE.size() + 1;
  
  labels[0] = new char [NrPileupEvtsSize];
  strncpy(labels[0], NrPileupEvts.c_str(), NrPileupEvtsSize);
  labels[1] = new char [NrVerticesSize];
  strncpy(labels[1], NrVertices.c_str(), NrVerticesSize); 
  labels[2] = new char [NrTracksSize];
  strncpy(labels[2], NrTracks.c_str(), NrTracksSize);
  labels[3] = new char [TrackPartIdSize];
  strncpy(labels[3], TrackPartId.c_str(), TrackPartIdSize);
  labels[4] = new char [CaloEnergyEBSize];
  strncpy(labels[4], CaloEnergyEB.c_str(), CaloEnergyEBSize);
  labels[5] = new char [CaloEnergyEESize];
  strncpy(labels[5], CaloEnergyEE.c_str(), CaloEnergyEESize);
  
  //FIXME: test for max nr of histos
  for (int i=minbunch_;i<=maxbunch_;++i) {
    int ii=i-minbunch_;
    char label[50];
    sprintf(label,"%s_%d",labels[0],i);
    nrPileupsH_[ii]    = dbe_->book1D(label,label,100,0,100);
    sprintf(label,"%s_%d",labels[1],i);
    nrVerticesH_[ii]   = dbe_->book1D(label,label,100,0,5000);
    sprintf(label,"%s_%d",labels[2],i);
    nrTracksH_[ii]     = dbe_->book1D(label,label,100,0,10000);
    sprintf(label,"%s_%d",labels[3],i);
    trackPartIdH_[ii]  =  dbe_->book1D(label,label,100,0,100);
    sprintf(label,"%s_%d",labels[4],i);
    caloEnergyEBH_ [ii]  = dbe_->book1D(label,label,100,0.,1000.);
    sprintf(label,"%s_%d",labels[5],i);
    caloEnergyEEH_ [ii]  = dbe_->book1D(label,label,100,0.,1000.);
  }
} 


void GlobalTest::endJob() {
 if (filename_.size() != 0 && dbe_ ) dbe_->save(filename_);
 
 for (int i = 0; i < 6; i++) delete[] labels[i];
}

//
// member functions
//

// ------------ method called to analyze the data  ------------
void
GlobalTest::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   using namespace edm;
   using namespace std;
   
// Get input
    edm::Handle<CrossingFrame<SimTrack> > cf_track;
    edm::Handle<CrossingFrame<SimTrack> > cf_vertex;
    edm::Handle<CrossingFrame<PCaloHit> > cf_calohitE;
    edm::Handle<CrossingFrame<PCaloHit> > cf_calohitB;
    std::string ecalsubdetb("g4SimHitsEcalHitsEB");
    std::string ecalsubdete("g4SimHitsEcalHitsEE");
    iEvent.getByLabel(cfTrackTag_, cf_track);
    iEvent.getByLabel(cfVertexTag_, cf_vertex);
    iEvent.getByLabel("mix",ecalsubdetb,cf_calohitB);
    iEvent.getByLabel("mix",ecalsubdete,cf_calohitE);

    // number of events/bcr ??

    // number of tracks 
    for (int i=minbunch_;i<=maxbunch_;++i) {
       nrTracksH_[i-minbunch_]->Fill(cf_track->getNrPileups(i));
    }

    // number of vertices
    for (int i=minbunch_;i<=maxbunch_;++i) {
       nrVerticesH_[i-minbunch_]->Fill(cf_vertex->getNrPileups(i));
    }

   // part id for each track
    std::auto_ptr<MixCollection<SimTrack> > coltr(new MixCollection<SimTrack>(cf_track.product()));
    MixCollection<SimTrack>::iterator cfitr;
    for (cfitr=coltr->begin(); cfitr!=coltr->end();cfitr++) {
	 trackPartIdH_[cfitr.bunch()-minbunch_]->Fill(cfitr->type());
     }

    // energy sum
    double sumE[10]={0.,0.,0.,0.,0.,0.,0.,0.,0.,0.};
    std::auto_ptr<MixCollection<PCaloHit> > colecalb(new MixCollection<PCaloHit>(cf_calohitB.product()));
    MixCollection<PCaloHit>::iterator cfiecalb;
    for (cfiecalb=colecalb->begin(); cfiecalb!=colecalb->end();cfiecalb++) {
       sumE[cfiecalb.bunch()-minbunch_]+=cfiecalb->energy();
      //      if (cfiecal.getTrigger())    tofecalhist_sig->Fill(cfiecal->time());
      //      else    tofecalhist->Fill(cfiecal->time());
    }
    for (int i=minbunch_;i<=maxbunch_;++i) {
       caloEnergyEBH_[i-minbunch_]->Fill(sumE[i-minbunch_]);
    }
    double sumEE[10]={0.,0.,0.,0.,0.,0.,0.,0.,0.,0.};
    std::auto_ptr<MixCollection<PCaloHit> > colecale(new MixCollection<PCaloHit>(cf_calohitE.product()));
    MixCollection<PCaloHit>::iterator cfiecale;
    for (cfiecale=colecale->begin(); cfiecale!=colecale->end();cfiecale++) {
       sumEE[cfiecale.bunch()-minbunch_]+=cfiecale->energy();
      //      if (cfiecal.getTrigger())    tofecalhist_sig->Fill(cfiecal->time());
      //      else    tofecalhist->Fill(cfiecal->time());
    }
    for (int i=minbunch_;i<=maxbunch_;++i) {
       caloEnergyEEH_[i-minbunch_]->Fill(sumEE[i-minbunch_]);
    }
}

