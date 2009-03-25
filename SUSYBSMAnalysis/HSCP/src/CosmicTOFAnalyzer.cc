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
// $Id: CosmicTOFAnalyzer.cc,v 1.4 2009/03/09 10:33:35 arizzi Exp $
//
//


// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDFilter.h"

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
#include "TTree.h"
#include "TBranch.h"
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


#include "AnalysisDataFormats/SUSYBSMObjects/interface/HSCParticle.h"

//
// class decleration
//
  using namespace susybsm;
  using namespace edm;
  using namespace std;
  using namespace reco;

struct DiMuonEvent
{
//timing
 float t0;
 float t1;
 float t0e;
 float t1e;
 float deltaT;
//momentum
 float pt0;
 float pt1;
 float eta0;
 float eta1;
 float phi0;
 float phi1;
//nhits
 float nh0;
 float nh1;
//sectr & wheel
 float s0; 
 float s1; 
 float w0; 
 float w1; 

//inner point
 float y0; 
 float y1; 
 float ev; 
 float run; 
};

struct MuonCollectionDataAndHistograms
{
      DiMuonEvent event;
      TBranch * branch;
      TFileDirectory * subDir;

      TH1F * nMuons;
      TH2F * hitsVsHits;
      TH1F * minHits;
      TH2F * minHitsVsPhi;
      TH2F * minHitsVsEta;
      TH2F * ptVsPt;
      TH1F * ptDiff;
      TH2F * posVsPos; 
      TH2F * ptVsPtSel; 
      TH1F * ptDiffSel;

      TH1F * diff;
      TH1F * pull;
      TH1F * diffBiasCorrected;
      TH1F * diffBiasCorrectedErr;
      TH1F * diffBiasCorrectedErrPtCut;
      TH1F * diffBiasCorrectedErrPtCutPhiCut;
      TProfile * diffBiasCorrectedErrPt;
      TProfile * diffBiasCorrectedVsErr;

      TH1F * pairs[5][15][5][15];
      float bias[5][15][5][15];
      float rms[5][15][5][15];
      float points[5][15][5][15];
      TFileDirectory * biasSubDir;
};

class CosmicTOFAnalyzer : public edm::EDFilter {
   public:
      explicit CosmicTOFAnalyzer(const edm::ParameterSet&);
      ~CosmicTOFAnalyzer();


   private:
      virtual void beginJob(const edm::EventSetup&) ;
      virtual bool filter(edm::Event&, const edm::EventSetup&);
      virtual void endJob() ;
      std::map<std::string,MuonCollectionDataAndHistograms> h_; 
      void readBias(std::string collName);
      void initHistos(std::string collName);
      void writeBias(std::string collName);
      bool analyzeCollection(const MuonCollection & muons, std::string collName, const edm::Event& iEvent);
      bool analyzeDiMuonEvent(const reco::Muon & muon0, const reco::Muon & muon1, std::string collName, const edm::Event& iEvent, MuonTime&  mt0, MuonTime & mt1);
      void initBranch(std::string collName, TTree * t);

      TTree * diMuEventTree;

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



bool
CosmicTOFAnalyzer::filter(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   using namespace std;
   using namespace edm;
   using namespace reco;
   bool result=false;
   Handle<MuonCollection> muH;
   iEvent.getByLabel("muons",muH);
   const MuonCollection & muons  =  *muH.product();
   if( analyzeCollection(muons,"muons",iEvent)) result=true;

   Handle<MuonCollection> muH2;
   bool t0Coll = iEvent.getByLabel("muonsWitht0Correction",muH2);
   if(t0Coll)
   {  
     const MuonCollection & muonsT0  =  *muH2.product();
     if( analyzeCollection(muonsT0,"muonsWitht0Correction",iEvent)) result=true;
   } 


   Handle<MuonTOFCollection> muTofOLD;
   bool oldColl =  iEvent.getByLabel("betaFromTOF",muTofOLD);
   if (oldColl)
    {
      const MuonTOFCollection & dtInfos  =  *muTofOLD.product();
      if(dtInfos.size()==2 && (dtInfos[0].second.nHits >= 25 ) && (dtInfos[1].second.nHits >= 25 ) )
      {
        MuonTime mt0;
        mt0.timeAtIpInOut=dtInfos[0].second.vertexTime;
        mt0.timeAtIpInOutErr=dtInfos[0].second.vertexTimeErr;
        MuonTime mt1;
        mt1.timeAtIpInOut=dtInfos[1].second.vertexTime;
        mt1.timeAtIpInOutErr=dtInfos[1].second.vertexTimeErr;
        
        if(dtInfos[0].first->outerTrack().isNull() ||  dtInfos[1].first->outerTrack().isNull())
        {
            cout << "Invalid Ref: " << dtInfos[0].second.nHits << " " << dtInfos[1].second.nHits << endl;
        } 
       else
         if(dtInfos[0].second.vertexTime!=0.000 && dtInfos[1].second.vertexTime!=0.000 )
          {
            if(analyzeDiMuonEvent(*dtInfos[0].first,*dtInfos[1].first,"oldBetaFromTOF",iEvent, mt0,mt1 )) result = true;
          }
     }  
   }



  if(result) diMuEventTree->Fill();
  return result;
}

bool CosmicTOFAnalyzer::analyzeCollection(const MuonCollection & muons, std::string collName, const edm::Event& iEvent)
{ 
   
h_[collName].nMuons->Fill(muons.size());

   if(muons.size()!=2) return false;
   
MuonTime mt0 = muons[0].time();
MuonTime mt1= muons[1].time();

 return analyzeDiMuonEvent(muons[0],muons[1],collName,iEvent, mt0,mt1 );
}

bool CosmicTOFAnalyzer::analyzeDiMuonEvent(const  reco::Muon & muon0, const  reco::Muon & muon1, std::string collName, const edm::Event& iEvent, MuonTime&  mt0, MuonTime & mt1)
{

   h_[collName].hitsVsHits->Fill(muon0.bestTrack()->hitPattern().numberOfValidMuonDTHits(), muon1.bestTrack()->hitPattern().numberOfValidMuonDTHits());
   if(muon0.outerTrack().isNonnull() && muon1.outerTrack().isNonnull())
          h_[collName].posVsPos->Fill(muon0.outerTrack()->innerPosition().y(),muon1.outerTrack()->innerPosition().y());
   h_[collName].ptVsPt->Fill(muon0.bestTrack()->pt(),muon1.bestTrack()->pt());
   h_[collName].ptDiff->Fill(muon0.bestTrack()->pt()-muon1.bestTrack()->pt());


   float etamin,phimin;
   int hitsmin;
   if(muon0.bestTrack()->hitPattern().numberOfValidMuonDTHits() < muon1.bestTrack()->hitPattern().numberOfValidMuonDTHits())
    {
       etamin=muon0.bestTrack()->eta();
       phimin=muon0.bestTrack()->phi();
       hitsmin=muon0.bestTrack()->hitPattern().numberOfValidMuonDTHits();
    } else {
       etamin=muon1.bestTrack()->eta();
       phimin=muon1.bestTrack()->phi();
       hitsmin=muon1.bestTrack()->hitPattern().numberOfValidMuonDTHits();
    }

  h_[collName].minHits->Fill(hitsmin);
  h_[collName].minHitsVsPhi->Fill(hitsmin,phimin);
  h_[collName].minHitsVsEta->Fill(hitsmin,etamin);

  if( muon0.bestTrack()->hitPattern().numberOfValidMuonDTHits() < 25 ) return false;
  if( muon1.bestTrack()->hitPattern().numberOfValidMuonDTHits() < 25 ) return false;


   h_[collName].ptVsPtSel->Fill(muon0.bestTrack()->pt(),muon1.bestTrack()->pt());
   h_[collName].ptDiffSel->Fill(muon0.bestTrack()->pt()-muon1.bestTrack()->pt());

h_[collName].event.t0 = 0;
h_[collName].event.t1 = 0;
h_[collName].event.t0e = 0;
h_[collName].event.t1e = 0;
h_[collName].event.deltaT = 0;
h_[collName].event.pt0=0;
h_[collName].event.pt1=0;
h_[collName].event.eta0=0;
h_[collName].event.eta1=0;
h_[collName].event.phi0=0;
h_[collName].event.phi1=0;
h_[collName].event.nh0=0;
h_[collName].event.nh1=0;
h_[collName].event.s0=0;
h_[collName].event.s1=0;
h_[collName].event.w0=0;
h_[collName].event.w1=0;



float t0 = mt0.timeAtIpInOut;
float t1 = mt1.timeAtIpInOut;

h_[collName].diff->Fill(t0-t1);
 int w0=0,s0=0,w1=0,s1=0;
 DTChamberId * id;
 cout << "matches0: " << muon0.matches().size() << endl;
 cout << "matches1: " << muon1.matches().size() << endl;
// id = dynamic_cast<const DTChamberId *> (& muon0.second.timeMeasurements[0].driftCell);
 std::set<unsigned int> dets1; 
//cout << "Mu1 " ; 
 for(trackingRecHit_iterator match = muon0.bestTrack()->recHitsBegin() ; match != muon0.bestTrack()->recHitsEnd() ; ++match)
 {
  DetId did=(*match)->geographicalId() ;
  if(did.det() == 2 && did.subdetId() == MuonSubdetId::DT)
  {
   if(s0==0)
   {
    id =  new DTChamberId(did);
    w0=id->wheel();
    s0=id->sector();
    delete id;
   // break;
   }
//   cout << did.rawId() << " ";
   dets1.insert(did.rawId());
  }
 }
// cout << endl;
//cout << "Mu2 " ; 
 for(trackingRecHit_iterator match = muon1.bestTrack()->recHitsBegin() ; match != muon1.bestTrack()->recHitsEnd() ; ++match)
 {
  DetId did=(*match)->geographicalId() ;
  if(did.det() == 2 && did.subdetId() == MuonSubdetId::DT)
  {
   if(s1==0)
   {
   id =  new DTChamberId(did);
   w1=id->wheel();
   s1=id->sector();
   delete id;
   // break;
   }
   if(dets1.find(did.rawId()) != dets1.end() ) 
    {
     cout << "Skipping event " << iEvent.id().event() << "same measurement used twice" << endl;
    }
//   cout << did.rawId() <<  " ";
  }
 }
// cout << endl;

if(s0 ==0 || s1 ==0)
 {
    cout << "Error: cannot find the sector of this muon" << endl;
    return true;
 }

if(muon0.pt() > 20 && muon1.pt()  > 20)  h_[collName].pairs[w0+2][s0][w1+2][s1]->Fill(t0-t1);

 // use only sectors for which we do know the bias quite well (at least 50 measurement)
 if(h_[collName].points[w0+2][s0][w1+2][s1]>=50) 
 { 
      h_[collName].diffBiasCorrected->Fill(t0-t1-h_[collName].bias[w0+2][s0][w1+2][s1]);
      float error = sqrt(mt0.timeAtIpInOutErr*mt0.timeAtIpInOutErr  + mt1.timeAtIpInOutErr *mt1.timeAtIpInOutErr ); 
      h_[collName].diffBiasCorrectedVsErr->Fill(error ,abs(t0-t1-h_[collName].bias[w0+2][s0][w1+2][s1])); 
      h_[collName].pull->Fill((t0-t1-h_[collName].bias[w0+2][s0][w1+2][s1])/error) ;

//TODO: try different Err cuts like: 0.8, 1, 1.2, 1.5, 2, 5
//TODO: Pull distribution (t-t0)/(sqrt(err0**2+err1**2))

       if( mt1.timeAtIpInOutErr < 10 && mt0.timeAtIpInOutErr < 10&&  (muon0.momentum() - muon1.momentum()).r() < 30   ) 
       {
          h_[collName].diffBiasCorrectedErr->Fill(t0-t1-h_[collName].bias[w0+2][s0][w1+2][s1]);
          h_[collName].diffBiasCorrectedErrPt->Fill(muon0.pt(),t0-t1-h_[collName].bias[w0+2][s0][w1+2][s1]);

          if(muon0.pt() > 50)
          {
             h_[collName].diffBiasCorrectedErrPtCut->Fill(t0-t1-h_[collName].bias[w0+2][s0][w1+2][s1]);
             if( fabs(muon0.phi()+1.5708 ) < 0.785  && fabs(muon1.phi()+1.5708 ) < 0.785  ) 
               {
                   h_[collName].diffBiasCorrectedErrPtCutPhiCut->Fill(t0-t1-h_[collName].bias[w0+2][s0][w1+2][s1]);
               }
          } 


          //Monitor details of events in the tails
          if( fabs(t0-t1-h_[collName].bias[w0+2][s0][w1+2][s1]) > 15)
          {
              cout << "TAIL ("  << collName << ") e,r:" << iEvent.id().event() << " , "<< iEvent.id().run() <<
                      " Values: " << t0 << " " << t1 << " Err: " << mt0.timeAtIpInOutErr << " " << 
                      mt1.timeAtIpInOutErr << "  #hits "  << muon0.bestTrack()->hitPattern().numberOfValidMuonDTHits() << " " << 
                      muon1.bestTrack()->hitPattern().numberOfValidMuonDTHits() << " W/S " << w0 << "/" << s0 << " " << w1 << "/" << s1  <<
                      " Momentum: " << muon0.momentum() << " " <<  muon1.momentum()  <<  " " << 
                      (muon0.momentum() - muon1.momentum()).r()/(muon0.momentum() + muon1.momentum()).r() 
                   <<   endl;
          }

 } // if error ok and muon pt matching 
 h_[collName].event.deltaT = t0-t1-h_[collName].bias[w0+2][s0][w1+2][s1]; 

} // if bias ok
 else
 { 
 h_[collName].event.deltaT = -1000; 
 }

 h_[collName].event.t0 = t0;
 h_[collName].event.t1 = t1;
 h_[collName].event.t0e = mt0.timeAtIpInOutErr;
 h_[collName].event.t1e = mt1.timeAtIpInOutErr;
 h_[collName].event.pt0=muon0.pt();
 h_[collName].event.pt1=muon1.pt();
 h_[collName].event.eta0=muon0.eta();
 h_[collName].event.eta1=muon1.eta();
 h_[collName].event.phi0=muon0.phi();
 h_[collName].event.phi1=muon1.phi();
 h_[collName].event.nh0= muon0.bestTrack()->hitPattern().numberOfValidMuonDTHits();
 h_[collName].event.nh1= muon1.bestTrack()->hitPattern().numberOfValidMuonDTHits();
 h_[collName].event.s0=s0;
 h_[collName].event.s1=s1;
 h_[collName].event.w0=w0;
 h_[collName].event.w1=w1;
 h_[collName].event.y0=muon0.outerTrack()->innerPosition().y();
 h_[collName].event.y1=muon1.outerTrack()->innerPosition().y();

 h_[collName].event.ev= iEvent.id().event();
 h_[collName].event.run = iEvent.id().run();
// h_[collName].branch->Fill();


if( fabs(t0-t1) > 50)
 {
  cout << "OVERFLOW ("  << collName << ") Values: " << t0 << " " << t1 << " Err: " << mt0.timeAtIpInOutErr << " " << mt1.timeAtIpInOutErr << "  #hits " 
 << muon0.bestTrack()->hitPattern().numberOfValidMuonDTHits() << " " << muon1.bestTrack()->hitPattern().numberOfValidMuonDTHits() << " W/S " << w0 << "/" << s0 << " " << w1 << "/" << s1  << endl;
 }
 return true;
}


// ------------ method called once each job just before starting event loop  ------------
void 
CosmicTOFAnalyzer::beginJob(const edm::EventSetup&)
{
  readBias("muons");
  initHistos("muons");
  readBias("muonsWitht0Correction");
  initHistos("muonsWitht0Correction");
  readBias("oldBetaFromTOF");
  initHistos("oldBetaFromTOF");
  edm::Service<TFileService> fs;
  TFileDirectory * ntupleDir = new TFileDirectory(fs->mkdir( "tree" ));
  diMuEventTree  = ntupleDir->make<TTree>("DiMuEventTree","Tree with di muon events");
  initBranch("muons",diMuEventTree);
  initBranch("muonsWitht0Correction",diMuEventTree);
  initBranch("oldBetaFromTOF",diMuEventTree);
 
}

void CosmicTOFAnalyzer::initBranch(std::string collName, TTree * t)
{
   h_[collName].branch = t->Branch(collName.c_str(),&(h_[collName].event),"t0:t1:t0e:t1e:deltaT:pt0:pt1:eta0:eta1:phi0:phi1:nh0:nh1:s0:s1:w0:w1:y0:y1:ev:run");

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
  h_[collName].diffBiasCorrectedErrPtCutPhiCut = h_[collName].subDir->make<TH1F>("DiffBiasSubErrPtCutPhiCut","DiffBiasSub (Pt > 50 && |phi+pi/2| < pi/4)", 100,-50,50);
  h_[collName].diffBiasCorrectedErrPt = h_[collName].subDir->make<TProfile>("DiffBiasSubErrPt","DiffBiasSub vs PT (Err1 && Err2 <10)", 100,0,500,-50,50);
  h_[collName].diffBiasCorrectedVsErr = h_[collName].subDir->make<TProfile>("DiffBiasSubVsErr","DiffBiasSub vs Err", 100,0,50,-50,50);


  h_[collName].nMuons = h_[collName].subDir->make<TH1F>("NMuons","Number of muons in the event", 25,-0.5,24.5);
  h_[collName].hitsVsHits = h_[collName].subDir->make<TH2F>("HitsVsHits","Number of Hits of muon 1 vs muon 2", 100,0,50,100,0,50);
  h_[collName].minHits = h_[collName].subDir->make<TH1F>("MinHits","Number of Hits of the muon with less hits", 100,0,50);
  h_[collName].minHitsVsPhi = h_[collName].subDir->make<TH2F>("MinHitsVsPhi","min Hits vs Phi", 100,0,50,100,-5,5);
  h_[collName].minHitsVsEta = h_[collName].subDir->make<TH2F>("MinHitsVsEta","min Hits vs Eta", 100,0,50,100,-5,5);
  h_[collName].posVsPos = h_[collName].subDir->make<TH2F>("PosVsPos","Y-pos of muon 1 vs muon 2", 250,-1500,1500,250,-1500,1500);
  h_[collName].ptVsPt = h_[collName].subDir->make<TH2F>("PtVsPt","Pt of muon 1 vs muon 2", 250,0,500,250,0,500);
  h_[collName].ptVsPtSel = h_[collName].subDir->make<TH2F>("PtVsPtSel","Pt of muon 1 vs muon 2 (selected  muons only)", 250,0,500,250,0,500);
  h_[collName].ptDiff = h_[collName].subDir->make<TH1F>("PtDiff","Pt of muon 1 - muon 2", 300,-50,50);
  h_[collName].ptDiffSel = h_[collName].subDir->make<TH1F>("PtDiffSel","Pt of muon 1 - muon 2 (selected  muons only)", 300,-50,50);


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
 writeBias("oldBetaFromTOF");
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
