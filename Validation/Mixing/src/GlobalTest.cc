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
// $Id: GlobalTest.cc,v 1.3 2007/03/09 10:23:35 uberthon Exp $
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
#include "SimDataFormats/TrackingHit/interface/PSimHit.h"

#include "Validation/Mixing/interface/GlobalTest.h"
#include "TFile.h"

using namespace edm;

GlobalTest::GlobalTest(const edm::ParameterSet& iConfig): filename_(iConfig.getParameter<std::string>("fileName")), minbunch_(iConfig.getParameter<int>("minBunch")),maxbunch_(iConfig.getParameter<int>("maxBunch")),  dbe_(0)
{
  std::cout << "Constructed GlobalTest, filename: "<<filename_<<" minbunch: "<<minbunch_<<", maxbunch: "<<maxbunch_<<std::endl;

}

GlobalTest::~GlobalTest()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)
}

void GlobalTest::beginJob(edm::EventSetup const&iSetup) {

  // get hold of back-end interface
  dbe_ = Service<DaqMonitorBEInterface>().operator->(); 
  dbe_->showDirStructure();
  dbe_->setCurrentFolder("Mixing");
  //book histos
  const int nrHistos=6;
  char * labels[nrHistos];
  labels[0]="NrPileupEvts";
  labels[1]="NrVertices";
  labels[2]="NrTracks";
  labels[3]="TrackPartId";
  labels[4]="CaloEnergyEB";
  labels[5]="CaloEnergyEE";

  /////    MonitorElement * vtxhistsig = dbe_->book1D(
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
}

//
// member functions
//

// ------------ method called to analyze the data  ------------
void
GlobalTest::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   using namespace edm;

// Get input
    edm::Handle<CrossingFrame<SimTrack> > cf_track;
    edm::Handle<CrossingFrame<SimTrack> > cf_vertex;
    edm::Handle<CrossingFrame<PCaloHit> > cf_calohitE;
    edm::Handle<CrossingFrame<PCaloHit> > cf_calohitB;
    std::string ecalsubdetb("EcalHitsEB");
    std::string ecalsubdete("EcalHitsEE");
    iEvent.getByType(cf_track);
    iEvent.getByType(cf_vertex);
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

