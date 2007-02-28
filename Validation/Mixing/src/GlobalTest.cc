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
// $Id: GlobalTest.cc,v 1.1 2007/02/27 17:05:09 uberthon Exp $
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
#include "TH1I.h"
#include "TFile.h"

using namespace edm;

GlobalTest::GlobalTest(const edm::ParameterSet& iConfig): filename_(iConfig.getParameter<std::string>("fileName")), minbunch_(iConfig.getParameter<int>("minBunch")),maxbunch_(iConfig.getParameter<int>("maxBunch")),  dbe_(0)
{
  std::cout << "Constructed GlobalTest, filename: "<<filename_<<" minbunch: "<<minbunch_<<", maxbunch: "<<maxbunch_<<std::endl;

  //  histfile_ = new TFile(filename_.c_str(),"RECREATE");

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
    edm::Handle<CrossingFrame> cf;
    iEvent.getByType(cf);

    // number of events/bcr ??

    // number of tracks 
    for (int i=minbunch_;i<=maxbunch_;++i) {
       nrTracksH_[i-minbunch_]->Fill(cf->getNrPileupTracks(i));
    }

    // number of vertices
    for (int i=minbunch_;i<=maxbunch_;++i) {
       nrVerticesH_[i-minbunch_]->Fill(cf->getNrPileupVertices(i));
    }

   // part id for each track
    std::auto_ptr<MixCollection<SimTrack> > coltr(new MixCollection<SimTrack>(cf.product()));
    MixCollection<SimTrack>::iterator cfitr;
    for (cfitr=coltr->begin(); cfitr!=coltr->end();cfitr++) {
      //       if (cfitr.getTrigger()==0) {
	 trackPartIdH_[cfitr.bunch()-minbunch_]->Fill(cfitr->type());
	 //       }
     }

    // energy sum
    double sumE[10]={0.,0.,0.,0.,0.,0.,0.,0.,0.,0.};
    std::string ecalsubdetb("EcalHitsEB");
    std::auto_ptr<MixCollection<PCaloHit> > colecalb(new MixCollection<PCaloHit>(cf.product(),std::string(ecalsubdetb)));
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
    std::string ecalsubdete("EcalHitsEE");
    std::auto_ptr<MixCollection<PCaloHit> > colecale(new MixCollection<PCaloHit>(cf.product(),std::string(ecalsubdete)));
    MixCollection<PCaloHit>::iterator cfiecale;
    for (cfiecale=colecale->begin(); cfiecale!=colecale->end();cfiecale++) {
       sumEE[cfiecale.bunch()-minbunch_]+=cfiecale->energy();
      //      if (cfiecal.getTrigger())    tofecalhist_sig->Fill(cfiecal->time());
      //      else    tofecalhist->Fill(cfiecal->time());
    }
    for (int i=minbunch_;i<=maxbunch_;++i) {
       caloEnergyEEH_[i-minbunch_]->Fill(sumEE[i-minbunch_]);
    }


//     // number of vertices
//     std::auto_ptr<MixCollection<SimVertex> > colvert(new MixCollection<SimVertex>(cf.product()));
//     MixCollection<SimVertex>::iterator cfivert;
//     for (cfivert=colvert->begin(); cfivert!=colvert->end();cfivert++) {
//       //       if (cfivert.getVertigger()==0) {
// 	 nrVerticesH_->Fill(cfivert.bunch());
// 	 //       }
//      }


// //vertex histo
//     char histovertices[30], sighistovertices[30],histovertexindices[30],histovertexindicessig[30];
//     sprintf(histovertices,"Vertices_bcr_%d",bunchcr_);
//     sprintf(sighistovertices,"SignalVertices_bcr_%d",bunchcr_);
//     sprintf(histovertexindices,"TrackPointers_%d",bunchcr_);
//     sprintf(histovertexindicessig,"TrackPointers_signal_%d",bunchcr_);
//     TH1I * vtxhist = new TH1I(histovertices,"Bunchcrossings",maxbunch_-minbunch_+1,minbunch_,maxbunch_+1);
//     TH1I * vtxhistsig = new TH1I(sighistovertices,"Bunchcrossings",maxbunch_-minbunch_+1,minbunch_,maxbunch_+1);
//     TH1I * vtxindhist = new TH1I(histovertexindices,"Vertex to Track Indices",100,0,300);
//     TH1I * vtxindhistsig = new TH1I(histovertexindicessig,"Signal Vertex to Track Indices",100,0,300);
//     std::auto_ptr<MixCollection<SimVertex> > col2(new MixCollection<SimVertex>(cf.product()));
//     MixCollection<SimVertex>::iterator cfi2;
//     for (cfi2=col2->begin(); cfi2!=col2->end();cfi2++) {
// 	if (cfi2.getTrigger()==0) {
// 	  vtxhist->Fill(cfi2.bunch());
// 	  if (!cfi2->noParent()) 	vtxindhist->Fill(cfi2->parentIndex());
// 	} else {
// 	  vtxhistsig->Fill(cfi2.bunch());
// 	  if (!cfi2->noParent()) 	vtxindhistsig->Fill(cfi2->parentIndex());
// 	}
//     }
	
//     int bsp=cf->getBunchSpace();
//     char tof[30];

//     //tracker
//     sprintf(tof,"TrackerHit_Tof_bcr_%d",bunchcr_);
//     TH1I * tofhist = new TH1I(tof,"TrackerHit_ToF",100,float(bsp*minbunch_),float(bsp*maxbunch_)+50.);
//     sprintf(tof,"SignalTrackerHit_Tof_bcr_%d",bunchcr_);
//     TH1I * tofhist_sig = new TH1I(tof,"TrackerHit_ToF",100,float(bsp*minbunch_),float(bsp*maxbunch_)+50.);
//     std::string subdet("TrackerHitsTECLowTof");
//     //    std::string subdet("TrackerHitsTIBLowTof");
//     std::auto_ptr<MixCollection<PSimHit> > colsh(new MixCollection<PSimHit>(cf.product(),std::string(subdet)));
//     MixCollection<PSimHit>::iterator cfish;
//     for (cfish=colsh->begin(); cfish!=colsh->end();cfish++) {
//       if (cfish.getTrigger())  {
// 	tofhist_sig->Fill(cfish->timeOfFlight());
//       }
//       else  {
// 	tofhist->Fill(cfish->timeOfFlight());
//       }
//     }

//     //Ecal
//     sprintf(tof,"EcalEBHit_Tof_bcr_%d",bunchcr_);
//     TH1I * tofecalhist = new TH1I(tof,"EcalEBHit_ToF",100,float(bsp*minbunch_),float(bsp*maxbunch_)+50.);
//     sprintf(tof,"SignalEcalEBHit_Tof_bcr_%d",bunchcr_);
//     TH1I * tofecalhist_sig = new TH1I(tof,"EcalEBHit_ToF",100,float(bsp*minbunch_),float(bsp*maxbunch_)+50.);
//     std::string ecalsubdet("EcalHitsEB");
//     std::auto_ptr<MixCollection<PCaloHit> > colecal(new MixCollection<PCaloHit>(cf.product(),std::string(ecalsubdet)));
//     MixCollection<PCaloHit>::iterator cfiecal;
//     for (cfiecal=colecal->begin(); cfiecal!=colecal->end();cfiecal++) {
//       if (cfiecal.getTrigger())    tofecalhist_sig->Fill(cfiecal->time());
//       else    tofecalhist->Fill(cfiecal->time());
//     }

//     // Hcal
//     sprintf(tof,"HcalHit_Tof_bcr_%d",bunchcr_);
//     TH1I * tofhcalhist = new TH1I(tof,"HcalHit_ToF",100,float(bsp*minbunch_),float(bsp*maxbunch_)+50.);
//     sprintf(tof,"SignalHcalHit_Tof_bcr_%d",bunchcr_);
//     TH1I * tofhcalhist_sig = new TH1I(tof,"HcalHit_ToF",100,float(bsp*minbunch_),float(bsp*maxbunch_)+50.);
//     std::string hcalsubdet("HcalHits");
//     std::auto_ptr<MixCollection<PCaloHit> > colhcal(new MixCollection<PCaloHit>(cf.product(),std::string(hcalsubdet)));
//     MixCollection<PCaloHit>::iterator cfihcal;
//     for (cfihcal=colhcal->begin(); cfihcal!=colhcal->end();cfihcal++) {
//       if (cfihcal.getTrigger())  tofhcalhist_sig->Fill(cfihcal->time());
//       else  tofhcalhist->Fill(cfihcal->time());
//     }
}

