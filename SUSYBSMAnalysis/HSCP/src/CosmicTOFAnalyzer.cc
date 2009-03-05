// -*- C++ -*-
//
// Package:    CosmicTOFAnalyzer
// Class:      CosmicTOFAnalyzer
// 
/**\class CosmicTOFAnalyzer CosmicTOFAnalyzer.cc SUSYBSMAnalysis/HSCP/src/CosmicTOFAnalyzer.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Andrea RIZZI
//         Created:  Sun Dec  7 12:41:44 CET 2008
// $Id$
//
//


// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "PhysicsTools/UtilAlgos/interface/TFileService.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

// user include files
// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ESHandle.h"


#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "TH2F.h"
#include "TProfile.h"
#include "TH1F.h"
#include <string>
#include <iostream>
#include <fstream>
#include "DataFormats/MuonDetId/interface/DTChamberId.h"
#include "DataFormats/MuonDetId/interface/MuonSubdetId.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHitFwd.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"
//
// class decleration
//

class CosmicTOFAnalyzer : public edm::EDAnalyzer {
   public:
      explicit CosmicTOFAnalyzer(const edm::ParameterSet&);
      ~CosmicTOFAnalyzer();


   private:
      virtual void beginJob(const edm::EventSetup&) ;
      virtual void analyze(const edm::Event&, const edm::EventSetup&);
      virtual void endJob() ;
      TFileDirectory * subDir;
      TH1F * h_diff; 
      TH1F * h_diffBiasCorrected; 
      TH1F * h_diffBiasCorrectedErr; 
      TH1F * h_diffBiasCorrectedErrPtCut; 
      TProfile * h_diffBiasCorrectedErrPt; 
      TH1F * h_pairs[5][15][5][15];
      float bias[5][15][5][15];
      float rms[5][15][5][15];
      float points[5][15][5][15];
      TFileDirectory * subDir1;
/*      TFileDirectory * subDir;
      TFileDirectory * subDir4;
      TFileDirectory * subDir5;
      TProfile * h_testc[8];
      TProfile * h_testd[8];
      TH1F * h_tof[8];*/

      // ----------member data ---------------------------
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
CosmicTOFAnalyzer::CosmicTOFAnalyzer(const edm::ParameterSet& iConfig)

{
   //now do what ever initialization is needed

}


CosmicTOFAnalyzer::~CosmicTOFAnalyzer()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called to for each event  ------------
void
CosmicTOFAnalyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   using namespace std;
   using namespace edm;
   using namespace reco;

Handle<MuonCollection> pIn;
iEvent.getByLabel("muons",pIn);
const MuonCollection & dtInfos  =  *pIn.product();

if(dtInfos.size()!=2) return;


if( dtInfos[0].bestTrack()->hitPattern().numberOfValidMuonDTHits() < 25 ) return;
if( dtInfos[1].bestTrack()->hitPattern().numberOfValidMuonDTHits() < 25 ) return;

MuonTime mt0 = dtInfos[0].time();
MuonTime mt1= dtInfos[1].time();

float t0 = mt0.timeAtIpInOut;
float t1 = mt1.timeAtIpInOut;

h_diff->Fill(t0-t1);
 int w0=0,s0=0,w1=0,s1=0;
 DTChamberId * id;
 cout << "matches0: " << dtInfos[0].matches().size() << endl;
 cout << "matches1: " << dtInfos[1].matches().size() << endl;
// id = dynamic_cast<const DTChamberId *> (& dtInfos[0].second.timeMeasurements[0].driftCell);
 for(trackingRecHit_iterator match = dtInfos[0].bestTrack()->recHitsBegin() ; match != dtInfos[0].bestTrack()->recHitsEnd() ; ++match)
 {
  DetId did=(*match)->geographicalId() ;
  if(did.det() == 2 && did.subdetId() == MuonSubdetId::DT)
  {
   id =  new DTChamberId(did);
   w0=id->wheel();
   s0=id->sector();
   delete id;
   break;
  }
 }

 for(trackingRecHit_iterator match = dtInfos[1].bestTrack()->recHitsBegin() ; match != dtInfos[1].bestTrack()->recHitsEnd() ; ++match)
 {
  DetId did=(*match)->geographicalId() ;
  if(did.det() == 2 && did.subdetId() == MuonSubdetId::DT)
  {
   id =  new DTChamberId(did);
   w1=id->wheel();
   s1=id->sector();
   delete id;
   break;
  }
 }


/* for(std::vector<MuonChamberMatch>::const_iterator match = dtInfos[1].matches().begin() ; match != dtInfos[1].matches().end() ; ++match)
 {
  cout << "det1: " << match->detector() << endl;
  if(match->detector() == MuonSubdetId::DT)
  {
   id =  new DTChamberId(match->id);
   w1=id->wheel();
   s1=id->sector();
   delete id;
  }
 }
*/
if(s0 ==0 || s1 ==0)
 {
    cout << "EEEEEEEEERRRRRRRRROOOOOORRRRRRRRRRRRR" << endl;
 return;
  }

//cout << "W" << w0 <<"S" << s0 << "_VS_" << "W" << w1 <<"S" << s1  << endl;

// cout << w0+2<< " " <<s0<< " " <<w1+2 << " " << s1 << endl
//if(dtInfos[0].pt() > 50) 
 h_pairs[w0+2][s0][w1+2][s1]->Fill(t0-t1);

 if(points[w0+2][s0][w1+2][s1]>=50)  h_diffBiasCorrected->Fill(t0-t1-bias[w0+2][s0][w1+2][s1]);

//TODO: try different Err cuts like: 0.8, 1, 1.2, 1.5, 2, 5
//TODO: Pull distribution (t-t0)/(sqrt(err0**2+err1**2))
if(points[w0+2][s0][w1+2][s1]>=50 &&  dtInfos[1].time().timeAtIpInOutErr < 10 && dtInfos[0].time().timeAtIpInOutErr < 10&&  (dtInfos[0].momentum() - dtInfos[1].momentum()).r() < 30   ) 
 {
  h_diffBiasCorrectedErr->Fill(t0-t1-bias[w0+2][s0][w1+2][s1]);
  h_diffBiasCorrectedErrPt->Fill(dtInfos[0].pt(),t0-t1-bias[w0+2][s0][w1+2][s1]);
  if(dtInfos[0].pt() > 50) h_diffBiasCorrectedErrPtCut->Fill(t0-t1-bias[w0+2][s0][w1+2][s1]);



if( fabs(t0-t1-bias[w0+2][s0][w1+2][s1]) > 15)
 {
  cout << "TAIL: e,r:" << iEvent.id().event() << " , "<< iEvent.id().run() << " Values: " << t0 << " " << t1 << " Err: " << dtInfos[0].time().timeAtIpInOutErr << " " << dtInfos[1].time().timeAtIpInOutErr << "  #hits " 
 << dtInfos[0].bestTrack()->hitPattern().numberOfValidMuonDTHits() << " " << dtInfos[1].bestTrack()->hitPattern().numberOfValidMuonDTHits() << " W/S " << w0 << "/" << s0 << " " << w1 << "/" << s1  <<
 " Momentum: " << dtInfos[0].momentum() << " " <<  dtInfos[1].momentum()  <<  " " << (dtInfos[0].momentum() - dtInfos[1].momentum()).r()/(dtInfos[0].momentum() + dtInfos[1].momentum()).r() <<   endl;
 }

 }
if( fabs(t0-t1) > 50)
 {
  cout << "OVERFLOW: Values: " << t0 << " " << t1 << " Err: " << dtInfos[0].time().timeAtIpInOutErr << " " << dtInfos[1].time().timeAtIpInOutErr << "  #hits " 
 << dtInfos[0].bestTrack()->hitPattern().numberOfValidMuonDTHits() << " " << dtInfos[1].bestTrack()->hitPattern().numberOfValidMuonDTHits() << " W/S " << w0 << "/" << s0 << " " << w1 << "/" << s1  << endl;
 }
}


// ------------ method called once each job just before starting event loop  ------------
void 
CosmicTOFAnalyzer::beginJob(const edm::EventSetup&)
{
   using namespace edm;
   using namespace std;

ifstream f("test.txt");

 while(!f.eof())
  {
   int w0,s0,w1,s1;
   float b,r,p;

   f >>  w0 >> s0 >> w1 >> s1 >> p >>  b >> r;
   if (!f.good()) break;

   points[w0][s0][w1][s1] = p;
   bias[w0][s0][w1][s1] = b;
   rms[w0][s0][w1][s1] = r;
   std::cout << w0 << " " << s0 << " " << w1 << " "<< s1 <<" " << b << " " << p << " " << r <<  std::endl;
  }


  edm::Service<TFileService> fs;
  subDir = new TFileDirectory(fs->mkdir( "Plots" ));
  h_diff = subDir->make<TH1F>("Diff","Diff", 100,-50,50);
  h_diffBiasCorrected = subDir->make<TH1F>("DiffBiasSub","DiffBiasSub", 100,-50,50);
  h_diffBiasCorrectedErr = subDir->make<TH1F>("DiffBiasSubErr","DiffBiasSub", 100,-50,50);
  h_diffBiasCorrectedErrPtCut = subDir->make<TH1F>("DiffBiasSubErrPtCut","DiffBiasSub (Pt > 50)", 100,-50,50);
  h_diffBiasCorrectedErrPt = subDir->make<TProfile>("DiffBiasSubErrPt","DiffBiasSub vs PT", 100,0,500,-50,50);
 

  subDir1 = new TFileDirectory(fs->mkdir( "Bias" ));
  for(int w1=0;w1<5;w1++)
   for(int w2=0;w2<5;w2++)
    for(int s1=1;s1<15;s1++)
     for(int s2=1;s2<15;s2++)
      {
          std::stringstream s;
          s<< "W" << w1-2 <<"S" << s1 << "_VS_" << "W" << w2-2 <<"S" << s2  ;
          h_pairs[w1][s1][w2][s2] = subDir1->make<TH1F>(s.str().c_str(),s.str().c_str(), 100,-50,50);
      } 


/*  subDir2 = new TFileDirectory(fs->mkdir( "Test2" ));
  subDir3 = new TFileDirectory(fs->mkdir( "FiberCorrected" ));
  subDir4 = new TFileDirectory(fs->mkdir( "FiberCorrectedNoMu67" ));
  h_testc[3] = subDir3->make<TProfile>("TIB","TIB", 60,-40,60,0,20);
  h_testc[4] = subDir3->make<TProfile>("TID","TID", 60,-40,60,0,20);
  h_testc[5] = subDir3->make<TProfile>("TOB","TOB", 60,-40,60,0,20);
  h_testc[6] = subDir3->make<TProfile>("TECm","TEC-", 60,-40,60,0,20);
  h_testc[7] = subDir3->make<TProfile>("TECp","TEC+", 60,-40,60,0,20);
  h_testd[1] = subDir4->make<TProfile>("PXB","PXB", 60,-40,60,0,20);
  h_testd[2] = subDir4->make<TProfile>("PXE","PXE", 60,-40,60,0,20);
  h_testd[3] = subDir4->make<TProfile>("TIB","TIB", 60,-40,60,0,20);
  h_testd[4] = subDir4->make<TProfile>("TID","TID", 60,-40,60,0,20);
  h_testd[5] = subDir4->make<TProfile>("TOB","TOB", 60,-40,60,0,20);
  h_testd[6] = subDir4->make<TProfile>("TECm","TEC-", 60,-40,60,0,20);
  h_testd[7] = subDir4->make<TProfile>("TECp","TEC+", 60,-40,60,0,20);*/

}

// ------------ method called once each job just after ending the event loop  ------------
void 
CosmicTOFAnalyzer::endJob() {

   using namespace edm;
   using namespace std;

   ofstream f("out.txt");



  for(int w1=0;w1<5;w1++)
   for(int w2=0;w2<5;w2++)
    for(int s1=1;s1<15;s1++)
     for(int s2=1;s2<15;s2++)
      {
          if(h_pairs[w1][s1][w2][s2]->GetEntries() > 0)
            {
             f << w1 << " " << s1 << " " << w2 << " " << s2 << " " << h_pairs[w1][s1][w2][s2]->GetEntries() << " " <<  h_pairs[w1][s1][w2][s2]->GetMean() << " " <<  h_pairs[w1][s1][w2][s2]->GetRMS() << endl;  
      /*       cout <<  "W" << w1-2 <<"S" << s1 << "_VS_" << "W" << w2-2 <<"S "  << s2 << " " <<  w1<< " " <<s1<< " " <<w2 << " " << s2 << " :"; 
             cout << " Entries : " << h_pairs[w1][s1][w2][s2]->GetEntries() ;
             cout << " Avg : " << h_pairs[w1][s1][w2][s2]->GetMean() ;
             cout << " RMS : " << h_pairs[w1][s1][w2][s2]->GetRMS() ;
             cout << endl;
        */    }
             else
            {
          //              cout <<  "W" << w1-2 <<"S" << s1 << "_VS_" << "W" << w2-2 <<"S : NOSTAT"  << endl ;
            }
      }


}

//define this as a plug-in
DEFINE_FWK_MODULE(CosmicTOFAnalyzer);
