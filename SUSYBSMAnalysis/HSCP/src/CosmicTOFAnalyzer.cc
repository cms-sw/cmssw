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
// $Id: CosmicTOFAnalyzer.cc,v 1.1 2009/03/05 09:42:09 arizzi Exp $
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
  using namespace edm;
  using namespace std;
  using namespace reco;

struct MuonCollectionDataAndHistograms
{
      TFileDirectory * subDir;
      TH1F * diff;
      TH1F * pull;
      TH1F * diffBiasCorrected;
      TH1F * diffBiasCorrectedErr;
      TH1F * diffBiasCorrectedErrPtCut;
      TProfile * diffBiasCorrectedErrPt;
      TProfile * diffBiasCorrectedVsErr;

      TH1F * pairs[5][15][5][15];
      float bias[5][15][5][15];
      float rms[5][15][5][15];
      float points[5][15][5][15];
      TFileDirectory * biasSubDir;

};

class CosmicTOFAnalyzer : public edm::EDAnalyzer {
   public:
      explicit CosmicTOFAnalyzer(const edm::ParameterSet&);
      ~CosmicTOFAnalyzer();


   private:
      virtual void beginJob(const edm::EventSetup&) ;
      virtual void analyze(const edm::Event&, const edm::EventSetup&);
      virtual void endJob() ;
      std::map<std::string,MuonCollectionDataAndHistograms> h_; 
      void readBias(std::string collName);
      void initHistos(std::string collName);
      void writeBias(std::string collName);
      void analyzeCollection(const MuonCollection & muons, std::string collName, const edm::Event& iEvent);

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

   Handle<MuonCollection> muH;
   iEvent.getByLabel("muons",muH);
   const MuonCollection & muons  =  *muH.product();
   analyzeCollection(muons,"muons",iEvent);

   Handle<MuonCollection> muH2;
   bool t0Coll = iEvent.getByLabel("muonsWitht0Correction",muH2);
  if(t0Coll)
  {  
     const MuonCollection & muonsT0  =  *muH2.product();
     analyzeCollection(muonsT0,"muonsWitht0Correction",iEvent);
  } 

}

void CosmicTOFAnalyzer::analyzeCollection(const MuonCollection & muons, std::string collName, const edm::Event& iEvent)
{ 

if(muons.size()!=2) return;
if( muons[0].bestTrack()->hitPattern().numberOfValidMuonDTHits() < 25 ) return;
if( muons[1].bestTrack()->hitPattern().numberOfValidMuonDTHits() < 25 ) return;


MuonTime mt0 = muons[0].time();
MuonTime mt1= muons[1].time();



float t0 = mt0.timeAtIpInOut;
float t1 = mt1.timeAtIpInOut;

h_[collName].diff->Fill(t0-t1);
 int w0=0,s0=0,w1=0,s1=0;
 DTChamberId * id;
 cout << "matches0: " << muons[0].matches().size() << endl;
 cout << "matches1: " << muons[1].matches().size() << endl;
// id = dynamic_cast<const DTChamberId *> (& muons[0].second.timeMeasurements[0].driftCell);
 for(trackingRecHit_iterator match = muons[0].bestTrack()->recHitsBegin() ; match != muons[0].bestTrack()->recHitsEnd() ; ++match)
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

 for(trackingRecHit_iterator match = muons[1].bestTrack()->recHitsBegin() ; match != muons[1].bestTrack()->recHitsEnd() ; ++match)
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

if(s0 ==0 || s1 ==0)
 {
    cout << "Error: cannot find the sector of this muon" << endl;
    return;
 }

 h_[collName].pairs[w0+2][s0][w1+2][s1]->Fill(t0-t1);

 // use only sectors for which we do know the bias quite well (at least 50 measurement)
 if(h_[collName].points[w0+2][s0][w1+2][s1]>=50) 
 { 
      h_[collName].diffBiasCorrected->Fill(t0-t1-h_[collName].bias[w0+2][s0][w1+2][s1]);
      float error = sqrt(muons[0].time().timeAtIpInOutErr*muons[0].time().timeAtIpInOutErr  + muons[1].time().timeAtIpInOutErr *muons[1].time().timeAtIpInOutErr ); 
      h_[collName].diffBiasCorrectedVsErr->Fill(error ,abs(t0-t1-h_[collName].bias[w0+2][s0][w1+2][s1])); 
      h_[collName].pull->Fill((t0-t1-h_[collName].bias[w0+2][s0][w1+2][s1])/error) ;

//TODO: try different Err cuts like: 0.8, 1, 1.2, 1.5, 2, 5
//TODO: Pull distribution (t-t0)/(sqrt(err0**2+err1**2))

       if( muons[1].time().timeAtIpInOutErr < 10 && muons[0].time().timeAtIpInOutErr < 10&&  (muons[0].momentum() - muons[1].momentum()).r() < 30   ) 
       {
          h_[collName].diffBiasCorrectedErr->Fill(t0-t1-h_[collName].bias[w0+2][s0][w1+2][s1]);
          h_[collName].diffBiasCorrectedErrPt->Fill(muons[0].pt(),t0-t1-h_[collName].bias[w0+2][s0][w1+2][s1]);

          if(muons[0].pt() > 50) h_[collName].diffBiasCorrectedErrPtCut->Fill(t0-t1-h_[collName].bias[w0+2][s0][w1+2][s1]);


          //Monitor details of events in the tails
          if( fabs(t0-t1-h_[collName].bias[w0+2][s0][w1+2][s1]) > 15)
          {
              cout << "TAIL ("  << collName << ") e,r:" << iEvent.id().event() << " , "<< iEvent.id().run() <<
                      " Values: " << t0 << " " << t1 << " Err: " << muons[0].time().timeAtIpInOutErr << " " << 
                      muons[1].time().timeAtIpInOutErr << "  #hits "  << muons[0].bestTrack()->hitPattern().numberOfValidMuonDTHits() << " " << 
                      muons[1].bestTrack()->hitPattern().numberOfValidMuonDTHits() << " W/S " << w0 << "/" << s0 << " " << w1 << "/" << s1  <<
                      " Momentum: " << muons[0].momentum() << " " <<  muons[1].momentum()  <<  " " << 
                      (muons[0].momentum() - muons[1].momentum()).r()/(muons[0].momentum() + muons[1].momentum()).r() 
                   <<   endl;
          }

 } // if error ok and muon pt matching 

} // if bias ok

if( fabs(t0-t1) > 50)
 {
  cout << "OVERFLOW ("  << collName << ") Values: " << t0 << " " << t1 << " Err: " << muons[0].time().timeAtIpInOutErr << " " << muons[1].time().timeAtIpInOutErr << "  #hits " 
 << muons[0].bestTrack()->hitPattern().numberOfValidMuonDTHits() << " " << muons[1].bestTrack()->hitPattern().numberOfValidMuonDTHits() << " W/S " << w0 << "/" << s0 << " " << w1 << "/" << s1  << endl;
 }
}


// ------------ method called once each job just before starting event loop  ------------
void 
CosmicTOFAnalyzer::beginJob(const edm::EventSetup&)
{
  readBias("muons");
  initHistos("muons");
  readBias("muonsWitht0Correction");
  initHistos("muonsWitht0Correction");
}

void CosmicTOFAnalyzer::readBias(std::string collName)
{
  ifstream f((collName+"_input-bias.txt").c_str());

  while(!f.eof())
  {
   int w0,s0,w1,s1;
   float b,r,p;

   f >>  w0 >> s0 >> w1 >> s1 >> p >>  b >> r;
   if (!f.good()) break;

   h_[collName].points[w0][s0][w1][s1] = p;
   h_[collName].bias[w0][s0][w1][s1] = b;
   h_[collName].rms[w0][s0][w1][s1] = r;
   std::cout << w0 << " " << s0 << " " << w1 << " "<< s1 <<" " << b << " " << p << " " << r <<  std::endl;
  }

}

void CosmicTOFAnalyzer::initHistos(std::string collName)
{
  edm::Service<TFileService> fs;
  h_[collName].subDir = new TFileDirectory(fs->mkdir( (collName+"Plots").c_str() ));
  h_[collName].diff = h_[collName].subDir->make<TH1F>("Diff","Diff", 100,-50,50);
  h_[collName].pull = h_[collName].subDir->make<TH1F>("Pulls","Pulls", 100,-5,5);
  h_[collName].diffBiasCorrected = h_[collName].subDir->make<TH1F>("DiffBiasSub","DiffBiasSub", 100,-50,50);
  h_[collName].diffBiasCorrectedErr = h_[collName].subDir->make<TH1F>("DiffBiasSubErr","DiffBiasSub (Err1 && Err2 < 10)", 100,-50,50);
  h_[collName].diffBiasCorrectedErrPtCut = h_[collName].subDir->make<TH1F>("DiffBiasSubErrPtCut","DiffBiasSub (Pt > 50)", 100,-50,50);
  h_[collName].diffBiasCorrectedErrPt = h_[collName].subDir->make<TProfile>("DiffBiasSubErrPt","DiffBiasSub vs PT (Err1 && Err2 <10)", 100,0,500,-50,50);
  h_[collName].diffBiasCorrectedVsErr = h_[collName].subDir->make<TProfile>("DiffBiasSubVsErr","DiffBiasSub vs Err", 100,0,50,-50,50);
 

  h_[collName].biasSubDir = new TFileDirectory(fs->mkdir( (collName+"Bias").c_str() ));
  for(int w1=0;w1<5;w1++)
   for(int w2=0;w2<5;w2++)
    for(int s1=1;s1<15;s1++)
     for(int s2=1;s2<15;s2++)
      {
          std::stringstream s;
          s<< "W" << w1-2 <<"S" << s1 << "_VS_" << "W" << w2-2 <<"S" << s2  ;
          h_[collName].pairs[w1][s1][w2][s2] = h_[collName].biasSubDir->make<TH1F>(s.str().c_str(),s.str().c_str(), 100,-50,50);
      } 


}

// ------------ method called once each job just after ending the event loop  ------------
void 
CosmicTOFAnalyzer::endJob() {
 writeBias("muons");
 writeBias("muonsWitht0Correction");
}
void 
CosmicTOFAnalyzer::writeBias(std::string collName) {

   using namespace edm;
   using namespace std;
   ofstream f((collName+"_output-bias.txt").c_str());



  for(int w1=0;w1<5;w1++)
   for(int w2=0;w2<5;w2++)
    for(int s1=1;s1<15;s1++)
     for(int s2=1;s2<15;s2++)
      {
          if(h_[collName].pairs[w1][s1][w2][s2]->GetEntries() > 0)
            {
             f << w1 << " " << s1 << " " << w2 << " " << s2 << " " << h_[collName].pairs[w1][s1][w2][s2]->GetEntries() << " " <<  h_[collName].pairs[w1][s1][w2][s2]->GetMean() << " " <<  h_[collName].pairs[w1][s1][w2][s2]->GetRMS() << endl;  
      /*       cout <<  "W" << w1-2 <<"S" << s1 << "_VS_" << "W" << w2-2 <<"S "  << s2 << " " <<  w1<< " " <<s1<< " " <<w2 << " " << s2 << " :"; 
             cout << " Entries : " << h_[collName].pairs[w1][s1][w2][s2]->GetEntries() ;
             cout << " Avg : " << h_[collName].pairs[w1][s1][w2][s2]->GetMean() ;
             cout << " RMS : " << h_[collName].pairs[w1][s1][w2][s2]->GetRMS() ;
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
