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
// $Id: CosmicTOFAnalyzer.cc,v 1.6 2009/09/22 15:34:37 arizzi Exp $
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

#include "DataFormats/TrackReco/interface/DeDxData.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"

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
 float deltaTs;
 
 float t0ecal;
 float t1ecal;

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
 
 float tkdedx; 
 float tkdedxN; 

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
      TH1F * diffSingleOffsetCorrected;
      TH1F * diffBiasCorrected;
      TH1F * diffBiasCorrectedErr;
      TH1F * diffBiasCorrectedErrPtCut;
      TH1F * diffBiasCorrectedErrPtCutPhiCut;
      TProfile * diffBiasCorrectedErrPt;
      TProfile * diffBiasCorrectedVsErr;

      TH1F * single[5][15];
      TH1F * pairs[5][15][5][15];
      float sbias[5][15];
      float srms[5][15];
      float spoints[5][15];
      float bias[5][15][5][15];
      float rms[5][15][5][15];
      float points[5][15][5][15];
      TFileDirectory * biasSubDir;
      TFileDirectory * singleSubDir;
};

class CosmicTOFAnalyzer : public edm::EDFilter {
   public:
      explicit CosmicTOFAnalyzer(const edm::ParameterSet&);
      ~CosmicTOFAnalyzer();


   private:
      virtual void  beginJob ( const edm::EventSetup& ) ;
      virtual bool  beginRun( edm::Run&, const edm::EventSetup& ) ;
      virtual bool filter(edm::Event&, const edm::EventSetup&);
      virtual void endJob();
//      virtual void endRun( edm::Run&, const edm::EventSetup& ) ;
      std::map<int,std::map<std::string,MuonCollectionDataAndHistograms> > hr_; 
 //    std::map<std::string,MuonCollectionDataAndHistograms> h_; 
      void readBias(std::string collName, int runn);
      void initHistos(std::string collName, int runn);
      void writeBias(std::string collName, int runn );
      bool analyzeCollection(const MuonCollection & muons, std::string collName, const edm::Event& iEvent);
      bool analyzeDiMuonEvent(const reco::Muon & muon0, const reco::Muon & muon1, std::string collName, const edm::Event& iEvent, MuonTime&  mt0, MuonTime & mt1);
      void initBranch(std::string collName, TTree * t);
      bool byrun;
      TTree * diMuEventTree;
      MuonCollectionDataAndHistograms & h(std::string name,int run)
         {
           if(!byrun) run=0;
           return hr_[run][name];
         }
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
   byrun=iConfig.getParameter<bool>("ByRun");
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
int runn = iEvent.id().run() ;
h(collName,runn).nMuons->Fill(muons.size());

   if(muons.size()!=2) return false;
   
MuonTime mt0 = muons[0].time();
MuonTime mt1= muons[1].time();

 return analyzeDiMuonEvent(muons[0],muons[1],collName,iEvent, mt0,mt1 );
}

bool CosmicTOFAnalyzer::analyzeDiMuonEvent(const  reco::Muon & muon0, const  reco::Muon & muon1, std::string collName, const edm::Event& iEvent, MuonTime&  mt0, MuonTime & mt1)
{
   float dedxN=0, dedxV=0;
   Handle<DeDxDataValueMap> dedxH;
   iEvent.getByLabel("dedxTruncated40CTF",dedxH);
   const ValueMap<DeDxData> dEdxTrack = *dedxH.product();
   edm::Handle<reco::TrackCollection> trackCollectionHandle;
   iEvent.getByLabel("ctfWithMaterialTracksP5",trackCollectionHandle);
   for(unsigned int i=0; i<trackCollectionHandle->size(); i++) {
     reco::TrackRef track  = reco::TrackRef( trackCollectionHandle, i );
     const DeDxData& dedx = dEdxTrack[track];
     if( (track->normalizedChi2() < 5 && track->numberOfValidHits()>=8 ) && track->pt() > 5) //quality cuts
      {
         dedxN = dedx.dEdx();
         dedxV = dedx.numberOfMeasurements();
         break ;
       }
   }

int runn = iEvent.id().run() ;

   h(collName,runn).hitsVsHits->Fill(muon0.bestTrack()->hitPattern().numberOfValidMuonDTHits(), muon1.bestTrack()->hitPattern().numberOfValidMuonDTHits());
   if(muon0.outerTrack().isNonnull() && muon1.outerTrack().isNonnull())
          h(collName,runn).posVsPos->Fill(muon0.outerTrack()->innerPosition().y(),muon1.outerTrack()->innerPosition().y());
   h(collName,runn).ptVsPt->Fill(muon0.bestTrack()->pt(),muon1.bestTrack()->pt());
   h(collName,runn).ptDiff->Fill(muon0.bestTrack()->pt()-muon1.bestTrack()->pt());


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

  h(collName,runn).minHits->Fill(hitsmin);
  h(collName,runn).minHitsVsPhi->Fill(hitsmin,phimin);
  h(collName,runn).minHitsVsEta->Fill(hitsmin,etamin);

  if( muon0.bestTrack()->hitPattern().numberOfValidMuonDTHits() < 25 ) return false;
  if( muon1.bestTrack()->hitPattern().numberOfValidMuonDTHits() < 25 ) return false;


   h(collName,runn).ptVsPtSel->Fill(muon0.bestTrack()->pt(),muon1.bestTrack()->pt());
   h(collName,runn).ptDiffSel->Fill(muon0.bestTrack()->pt()-muon1.bestTrack()->pt());

h(collName,0).event.t0 = 0;
h(collName,0).event.t1 = 0;
h(collName,0).event.t0e = 0;
h(collName,0).event.t1e = 0;
h(collName,0).event.deltaT = 0;
h(collName,0).event.deltaTs = 0;
h(collName,0).event.pt0=0;
h(collName,0).event.pt1=0;
h(collName,0).event.eta0=0;
h(collName,0).event.eta1=0;
h(collName,0).event.phi0=0;
h(collName,0).event.phi1=0;
h(collName,0).event.nh0=0;
h(collName,0).event.nh1=0;
h(collName,0).event.s0=0;
h(collName,0).event.s1=0;
h(collName,0).event.w0=0;
h(collName,0).event.w1=0;

h(collName,0).event.tkdedx=0;
h(collName,0).event.tkdedxN=0;
h(collName,0).event.t0ecal=0;
h(collName,0).event.t1ecal=0;


float t0 = mt0.timeAtIpInOut;
float t1 = mt1.timeAtIpInOut;

h(collName,runn).diff->Fill(t0-t1);
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

if(muon0.pt() > 20 && muon1.pt()  > 20)  h(collName,runn).pairs[w0+2][s0][w1+2][s1]->Fill(t0-t1);

if(muon0.pt() > 20)  h(collName,runn).single[w0+2][s0]->Fill(t0);
if(muon1.pt() > 20)  h(collName,runn).single[w1+2][s1]->Fill(t1);

if(h(collName,runn).spoints[w0+2][s0]>=50 && h(collName,runn).spoints[w1+2][s1]>=50)
{
// correct with individual biases
      h(collName,runn).diffSingleOffsetCorrected->Fill( (t0-h(collName,runn).sbias[w0+2][s0])-(t1-h(collName,runn).sbias[w1+2][s1]));
      h(collName,0).event.deltaTs = (t0-h(collName,runn).sbias[w0+2][s0])-(t1-h(collName,runn).sbias[w1+2][s1]); 
}
else 
{
  h(collName,0).event.deltaTs = -1000;
} 

 // use only sectors for which we do know the bias quite well (at least 50 measurement)
// if(h(collName,runn).points[w0+2][s0][w1+2][s1]>=50) 
 if(h(collName,runn).points[w0+2][s0][w1+2][s1]>=10) 
 { 
// correct using pair bias
      h(collName,runn).diffBiasCorrected->Fill(t0-t1-h(collName,runn).bias[w0+2][s0][w1+2][s1]);
      float error = sqrt(mt0.timeAtIpInOutErr*mt0.timeAtIpInOutErr  + mt1.timeAtIpInOutErr *mt1.timeAtIpInOutErr ); 
      h(collName,runn).diffBiasCorrectedVsErr->Fill(error ,abs(t0-t1-h(collName,runn).bias[w0+2][s0][w1+2][s1])); 
      h(collName,runn).pull->Fill((t0-t1-h(collName,runn).bias[w0+2][s0][w1+2][s1])/error) ;

//TODO: try different Err cuts like: 0.8, 1, 1.2, 1.5, 2, 5
//TODO: Pull distribution (t-t0)/(sqrt(err0**2+err1**2))

       if( mt1.timeAtIpInOutErr < 10 && mt0.timeAtIpInOutErr < 10&&  (muon0.momentum() - muon1.momentum()).r() < 30   ) 
       {
          h(collName,runn).diffBiasCorrectedErr->Fill(t0-t1-h(collName,runn).bias[w0+2][s0][w1+2][s1]);
          h(collName,runn).diffBiasCorrectedErrPt->Fill(muon0.pt(),t0-t1-h(collName,runn).bias[w0+2][s0][w1+2][s1]);

          if(muon0.pt() > 50)
          {
             h(collName,runn).diffBiasCorrectedErrPtCut->Fill(t0-t1-h(collName,runn).bias[w0+2][s0][w1+2][s1]);
             if( fabs(muon0.phi()+1.5708 ) < 0.785  && fabs(muon1.phi()+1.5708 ) < 0.785  ) 
               {
                   h(collName,runn).diffBiasCorrectedErrPtCutPhiCut->Fill(t0-t1-h(collName,runn).bias[w0+2][s0][w1+2][s1]);
               }
          } 


          //Monitor details of events in the tails
          if( fabs(t0-t1-h(collName,runn).bias[w0+2][s0][w1+2][s1]) > 15)
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
 h(collName,0).event.deltaT = t0-t1-h(collName,runn).bias[w0+2][s0][w1+2][s1]; 

} // if bias ok
 else
 { 
 h(collName,0).event.deltaT = -1000; 
 }
 
 h(collName,0).event.t0 = t0;
 h(collName,0).event.t1 = t1;
 h(collName,0).event.t0e = mt0.timeAtIpInOutErr;
 h(collName,0).event.t1e = mt1.timeAtIpInOutErr;
 h(collName,0).event.pt0=muon0.pt();
 h(collName,0).event.pt1=muon1.pt();
 h(collName,0).event.eta0=muon0.eta();
 h(collName,0).event.eta1=muon1.eta();
 h(collName,0).event.phi0=muon0.phi();
 h(collName,0).event.phi1=muon1.phi();
 h(collName,0).event.nh0= muon0.bestTrack()->hitPattern().numberOfValidMuonDTHits();
 h(collName,0).event.nh1= muon1.bestTrack()->hitPattern().numberOfValidMuonDTHits();
 h(collName,0).event.s0=s0;
 h(collName,0).event.s1=s1;
 h(collName,0).event.w0=w0;
 h(collName,0).event.w1=w1;
 h(collName,0).event.y0=muon0.outerTrack()->innerPosition().y();
 h(collName,0).event.y1=muon1.outerTrack()->innerPosition().y();
 
    h(collName,0).event.tkdedxN = dedxN;
    h(collName,0).event.tkdedx = dedxV;

 h(collName,0).event.t0ecal=muon0.calEnergy().ecal_time;
 h(collName,0).event.t1ecal=muon1.calEnergy().ecal_time;

 h(collName,0).event.ev= iEvent.id().event();
 h(collName,0).event.run = iEvent.id().run();
// h(collName,0).branch->Fill();


if( fabs(t0-t1) > 50)
 {
  cout << "OVERFLOW ("  << collName << ") Values: " << t0 << " " << t1 << " Err: " << mt0.timeAtIpInOutErr << " " << mt1.timeAtIpInOutErr << "  #hits " 
 << muon0.bestTrack()->hitPattern().numberOfValidMuonDTHits() << " " << muon1.bestTrack()->hitPattern().numberOfValidMuonDTHits() << " W/S " << w0 << "/" << s0 << " " << w1 << "/" << s1  << endl;
 }
 return true;
}


// ------------ method called once each job just before starting event loop  ------------
bool 
CosmicTOFAnalyzer::beginRun(edm::Run& r, const edm::EventSetup&)
{
int runn = r.run();
if(!byrun) runn=0;

/*if(hr_.find(0) == hr_.end())//check if 0 is there
 {
  readBias("muons",0);
  initHistos("muons",0);
  readBias("muonsWitht0Correction",0);
  initHistos("muonsWitht0Correction",0);
  readBias("oldBetaFromTOF",0);
  initHistos("oldBetaFromTOF",0);

 }
*/
std::cout << "hereeeeeeeeeeeeee" << std::endl;
if(hr_.find(runn) != hr_.end())
  {
   return true;
  }
   cout << "Booking RUN " << runn << endl;
  readBias("muons",runn);
  initHistos("muons",runn);
  readBias("muonsWitht0Correction",runn);
  initHistos("muonsWitht0Correction",runn);
  readBias("oldBetaFromTOF",runn);
  initHistos("oldBetaFromTOF",runn);

 return true;
}

void CosmicTOFAnalyzer::beginJob(const edm::EventSetup&)
{
if(!byrun)
 {
  readBias("muons",0);
  initHistos("muons",0);
  readBias("muonsWitht0Correction",0);
  initHistos("muonsWitht0Correction",0);
  readBias("oldBetaFromTOF",0);
  initHistos("oldBetaFromTOF",0);
}


  edm::Service<TFileService> fs;
  TFileDirectory * ntupleDir = new TFileDirectory(fs->mkdir( "tree" ));
  diMuEventTree  = ntupleDir->make<TTree>("DiMuEventTree","Tree with di muon events");
  initBranch("muons",diMuEventTree);
  initBranch("muonsWitht0Correction",diMuEventTree);
  initBranch("oldBetaFromTOF",diMuEventTree);
}

void CosmicTOFAnalyzer::initBranch(std::string collName, TTree * t)
{
   h(collName,0).branch = t->Branch(collName.c_str(),&(h(collName,0).event),"t0:t1:t0e:t1e:deltaT:deltaTs:t0ecal:t1ecal:pt0:pt1:eta0:eta1:phi0:phi1:nh0:nh1:s0:s1:w0:w1:y0:y1:ev:run:tkdedx:tkdedxN");

}

void CosmicTOFAnalyzer::readBias(std::string collName,int runn)
{
if(!byrun) runn=0; 
  std::ostringstream stm;
  stm << collName << "_" << runn << "_input-bias.txt"  ;
  ifstream f(stm.str().c_str());
  std::cout << stm <<std::endl;

  while(!f.eof())
  {
   int w0,s0,w1,s1;
   float b,r,p;

   f >>  w0 >> s0 >> w1 >> s1 >> p >>  b >> r;
   if (!f.good()) break;

   h(collName,runn).points[w0][s0][w1][s1] = p;
   h(collName,runn).bias[w0][s0][w1][s1] = b;
   h(collName,runn).rms[w0][s0][w1][s1] = r;
   std::cout << w0 << " " << s0 << " " << w1 << " "<< s1 <<" " << b << " " << p << " " << r <<  std::endl;
  }

  std::ostringstream stm2;
  stm2<< collName << "_" << runn << "_input-singleOffset.txt"  ;
  ifstream f1(stm2.str().c_str());

  while(!f1.eof())
  {
   int w0,s0;
   float b,r,p;

   f1 >>  w0 >> s0 >> p >>  b >> r;
   if (!f1.good()) break;

   h(collName,runn).spoints[w0][s0] = p;
   h(collName,runn).sbias[w0][s0] = b;
   h(collName,runn).srms[w0][s0] = r;
   std::cout << w0 << " " << s0 << " " << b << " " << p << " " << r <<  std::endl;
  }


}

void CosmicTOFAnalyzer::initHistos(std::string collName,int runn)
{
if(!byrun) runn=0; 
  edm::Service<TFileService> fs;
  std::ostringstream stm;
  stm << collName << "Plots"  << runn;
  h(collName,runn).subDir = new TFileDirectory(fs->mkdir( stm.str().c_str() ));
  h(collName,runn).diff = h(collName,runn).subDir->make<TH1F>("Diff","Diff", 100,-50,50);
  h(collName,runn).pull = h(collName,runn).subDir->make<TH1F>("Pulls","Pulls", 100,-5,5);
  h(collName,runn).diffBiasCorrected = h(collName,runn).subDir->make<TH1F>("DiffBiasSub","DiffBiasSub", 100,-50,50);
  h(collName,runn).diffSingleOffsetCorrected = h(collName,runn).subDir->make<TH1F>("DiffSingleOffsetSub","DiffSingleOffsetSub", 100,-50,50);
  h(collName,runn).diffBiasCorrectedErr = h(collName,runn).subDir->make<TH1F>("DiffBiasSubErr","DiffBiasSub (Err1 && Err2 < 10)", 100,-50,50);
  h(collName,runn).diffBiasCorrectedErrPtCut = h(collName,runn).subDir->make<TH1F>("DiffBiasSubErrPtCut","DiffBiasSub (Pt > 50)", 100,-50,50);
  h(collName,runn).diffBiasCorrectedErrPtCutPhiCut = h(collName,runn).subDir->make<TH1F>("DiffBiasSubErrPtCutPhiCut","DiffBiasSub (Pt > 50 && |phi+pi/2| < pi/4)", 100,-50,50);
  h(collName,runn).diffBiasCorrectedErrPt = h(collName,runn).subDir->make<TProfile>("DiffBiasSubErrPt","DiffBiasSub vs PT (Err1 && Err2 <10)", 100,0,500,-50,50);
  h(collName,runn).diffBiasCorrectedVsErr = h(collName,runn).subDir->make<TProfile>("DiffBiasSubVsErr","DiffBiasSub vs Err", 100,0,50,-50,50);


  h(collName,runn).nMuons = h(collName,runn).subDir->make<TH1F>("NMuons","Number of muons in the event", 25,-0.5,24.5);
  h(collName,runn).hitsVsHits = h(collName,runn).subDir->make<TH2F>("HitsVsHits","Number of Hits of muon 1 vs muon 2", 100,0,50,100,0,50);
  h(collName,runn).minHits = h(collName,runn).subDir->make<TH1F>("MinHits","Number of Hits of the muon with less hits", 100,0,50);
  h(collName,runn).minHitsVsPhi = h(collName,runn).subDir->make<TH2F>("MinHitsVsPhi","min Hits vs Phi", 100,0,50,100,-5,5);
  h(collName,runn).minHitsVsEta = h(collName,runn).subDir->make<TH2F>("MinHitsVsEta","min Hits vs Eta", 100,0,50,100,-5,5);
  h(collName,runn).posVsPos = h(collName,runn).subDir->make<TH2F>("PosVsPos","Y-pos of muon 1 vs muon 2", 250,-1500,1500,250,-1500,1500);
  h(collName,runn).ptVsPt = h(collName,runn).subDir->make<TH2F>("PtVsPt","Pt of muon 1 vs muon 2", 250,0,500,250,0,500);
  h(collName,runn).ptVsPtSel = h(collName,runn).subDir->make<TH2F>("PtVsPtSel","Pt of muon 1 vs muon 2 (selected  muons only)", 250,0,500,250,0,500);
  h(collName,runn).ptDiff = h(collName,runn).subDir->make<TH1F>("PtDiff","Pt of muon 1 - muon 2", 300,-50,50);
  h(collName,runn).ptDiffSel = h(collName,runn).subDir->make<TH1F>("PtDiffSel","Pt of muon 1 - muon 2 (selected  muons only)", 300,-50,50);


  std::ostringstream stm2;
  stm2 << collName << "Bias"  << runn;
  h(collName,runn).biasSubDir = new TFileDirectory(fs->mkdir( stm2.str().c_str() ));
  for(int w1=0;w1<5;w1++)
   for(int w2=0;w2<5;w2++)
    for(int s1=1;s1<15;s1++)
     for(int s2=1;s2<15;s2++)
      {
          std::stringstream s;
          s<< "W" << w1-2 <<"S" << s1 << "_VS_" << "W" << w2-2 <<"S" << s2  ;
          h(collName,runn).pairs[w1][s1][w2][s2] = h(collName,runn).biasSubDir->make<TH1F>(s.str().c_str(),s.str().c_str(), 100,-50,50);
      } 

  std::ostringstream stm3;
  stm3 << collName << "SingleOffset"  << runn;
  h(collName,runn).singleSubDir = new TFileDirectory(fs->mkdir( stm3.str().c_str() ));
  for(int w1=0;w1<5;w1++)
   for(int s1=1;s1<15;s1++)
      {
          std::stringstream s;
          s<< "W" << w1-2 <<"S" << s1  ;
          h(collName,runn).single[w1][s1] = h(collName,runn).singleSubDir->make<TH1F>(s.str().c_str(),s.str().c_str(), 100,-50,50);
      }


}

// ------------ method called once each job just after ending the event loop  ------------
void 
CosmicTOFAnalyzer::endJob() {
std::cout << "here" << std::endl;
for(std::map<int,std::map<std::string,MuonCollectionDataAndHistograms> >::iterator it = hr_.begin(); it!=hr_.end(); ++it)
 {
 if(byrun && it->first == 0) continue; 
 writeBias("muons",it->first);
 writeBias("muonsWitht0Correction",it->first);
 writeBias("oldBetaFromTOF",it->first);
  }
}
void 
CosmicTOFAnalyzer::writeBias(std::string collName,int runn) {
if(!byrun) runn=0; 

   using namespace edm;
   using namespace std;
  std::ostringstream stm;
  stm << collName << "_" << runn << "_output-bias.txt"  ;
   ofstream f(stm.str().c_str());



  for(int w1=0;w1<5;w1++)
   for(int w2=0;w2<5;w2++)
    for(int s1=1;s1<15;s1++)
     for(int s2=1;s2<15;s2++)
      {
          if(h(collName,runn).pairs[w1][s1][w2][s2]->GetEntries() > 0)
            {
             f << w1 << " " << s1 << " " << w2 << " " << s2 << " " << h(collName,runn).pairs[w1][s1][w2][s2]->GetEntries() << " " <<  h(collName,runn).pairs[w1][s1][w2][s2]->GetMean() << " " <<  h(collName,runn).pairs[w1][s1][w2][s2]->GetRMS() << endl;  
      /*       cout <<  "W" << w1-2 <<"S" << s1 << "_VS_" << "W" << w2-2 <<"S "  << s2 << " " <<  w1<< " " <<s1<< " " <<w2 << " " << s2 << " :"; 
             cout << " Entries : " << h(collName,runn).pairs[w1][s1][w2][s2]->GetEntries() ;
             cout << " Avg : " << h(collName,runn).pairs[w1][s1][w2][s2]->GetMean() ;
             cout << " RMS : " << h(collName,runn).pairs[w1][s1][w2][s2]->GetRMS() ;
             cout << endl;
        */    }
             else
            {
          //              cout <<  "W" << w1-2 <<"S" << s1 << "_VS_" << "W" << w2-2 <<"S : NOSTAT"  << endl ;
            }
      }
  std::ostringstream stm2;
  stm2<< collName << "_" << runn << "_output-singleOffset.txt"  ;
  ofstream f1(stm2.str().c_str());
  for(int w1=0;w1<5;w1++)
    for(int s1=1;s1<15;s1++)
      {
          if(h(collName,runn).single[w1][s1]->GetEntries() > 0)
            {
             f1 << w1 << " " << s1 << " " << h(collName,runn).single[w1][s1]->GetEntries() << " " <<  h(collName,runn).single[w1][s1]->GetMean() << " " <<  h(collName,runn).single[w1][s1]->GetRMS() << endl;
      /*       cout <<  "W" << w1-2 <<"S" << s1 << "_VS_" << "W" << w2-2 <<"S "  << s2 << " " <<  w1<< " " <<s1<< " " <<w2 << " " << s2 << " :";
             cout << " Entries : " << h(collName,runn).pairs[w1][s1][w2][s2]->GetEntries() ;
             cout << " Avg : " << h(collName,runn).pairs[w1][s1][w2][s2]->GetMean() ;
             cout << " RMS : " << h(collName,runn).pairs[w1][s1][w2][s2]->GetRMS() ;
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
