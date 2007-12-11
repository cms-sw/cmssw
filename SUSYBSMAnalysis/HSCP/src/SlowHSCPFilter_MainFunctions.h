// -*- C++ -*-
//
// Original Author:  Loic QUERTENMONT
//         Created:  Fri Dec  7 10:40:51 CET 2007
// $Id: SlowHSCPFilter.cc,v 1.1 2007/12/10 18:57:12 querten Exp $
//
//


// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDFilter.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"



#include "FWCore/Framework/interface/TriggerNames.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutRecord.h"

#include "DataFormats/HLTReco/interface/HLTFilterObject.h"
#include "DataFormats/Common/interface/TriggerResults.h"
#include "DataFormats/L1Trigger/interface/L1ParticleMap.h"
#include "DataFormats/L1Trigger/interface/L1MuonParticle.h"
#include "DataFormats/L1Trigger/interface/L1MuonParticleFwd.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/Candidate/interface/CandidateFwd.h"
#include "DataFormats/Candidate/interface/LeafCandidate.h"
#include "DataFormats/RecoCandidate/interface/RecoChargedCandidate.h"
#include "DataFormats/RecoCandidate/interface/RecoChargedCandidateFwd.h"

#include "SimDataFormats/Track/interface/SimTrack.h"
#include "SimDataFormats/Vertex/interface/SimVertex.h"
#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"
#include "SimDataFormats/TrackingHit/interface/PSimHit.h"

#include "Geometry/CSCGeometry/interface/CSCGeometry.h"
#include <Geometry/CSCGeometry/interface/CSCLayer.h>
#include "Geometry/DTGeometry/interface/DTLayer.h"
#include "Geometry/DTGeometry/interface/DTGeometry.h"
#include "Geometry/RPCGeometry/interface/RPCGeometry.h"
#include "Geometry/RPCGeometry/interface/RPCRoll.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"

using namespace edm;


double DeltaR               (double phi1, double eta1, double phi2, double eta2);
int    ClosestL1Muon        (double phi, double eta, double dRMax, const l1extra::L1MuonParticleCollection L1_Muons);
int    ComesFromWhichHSCP   (unsigned int TrackId, std::vector<SimTrack*> HSCPs, std::vector<SimTrack>  TrackColl, std::vector<SimVertex> VertexColl);
void   GetTrueL1MuonsAndTime(edm::Event&, const edm::EventSetup&,  int* recoL1Muon, double* MinDt);  


void
GetTrueL1MuonsAndTime(edm::Event& iEvent, const edm::EventSetup& iSetup, int* recoL1Muon, double* MinDt)
{
   Handle<reco::CandidateCollection>  MC_Cand_h ;
   iEvent.getByLabel("genParticleCandidates",MC_Cand_h);
   const reco::CandidateCollection MC_Cand = *MC_Cand_h.product();

   for(unsigned int i=0;i<MC_Cand.size();i++){
//      if( abs(MC_Cand[i].pdgId())>10000 && MC_Cand[i].status()==1 )
//         printf("MC  Cand %2i : phi = %6.2f eta = %6.2f  Pt = %8.2f Beta = %6.2f\n",i,MC_Cand[i].phi(), MC_Cand[i].eta(), MC_Cand[i].pt() ,MC_Cand[i].p()/MC_Cand[i].energy());
   }

   Handle<l1extra::L1MuonParticleCollection> L1_Muons_h;
   iEvent.getByLabel("l1extraParticles", L1_Muons_h);
   const l1extra::L1MuonParticleCollection L1_Muons = *L1_Muons_h.product();

//   for(unsigned int i=0;i<L1_Muons.size();i++){
//      printf("L1  Muons %i : phi = %6.2f eta = %6.2f  Pt = %8.2f Beta = %6.2f\n",i,L1_Muons[i].phi(), L1_Muons[i].eta(), L1_Muons[i].pt(), L1_Muons[i].p()/L1_Muons[i].energy() );
//   }

//  Handle<reco::RecoChargedCandidateCollection>  HLT_Muons_h ;
//  try{iEvent.getByLabel("hltL3MuonCandidates",HLT_Muons_h);}catch(...){printf("No hltL3MuonCandidates__HLT\n");}
//  reco::RecoChargedCandidateCollection HLT_Muons;
//  if(HLT_Muons_h.isValid())HLT_Muons = *HLT_Muons_h.product();

   Handle<std::vector< SimTrack > > h_Tracks;
   iEvent.getByLabel("g4SimHits", h_Tracks);
   std::vector< SimTrack > TrackColl = *h_Tracks.product();

   Handle<std::vector< SimVertex > > h_Vertex;
   iEvent.getByLabel("g4SimHits", h_Vertex);
   std::vector< SimVertex > VertexColl = *h_Vertex.product();

   std::vector<SimTrack*> PrimaryHSCPTracks;
   for(unsigned int i=0;i<TrackColl.size();i++){
      if(abs(TrackColl[i].type()) > 10000 && TrackColl[i].vertIndex()==0){
         PrimaryHSCPTracks.push_back( &(TrackColl[i]) );
      }
   }

   for(unsigned int j=0;j<PrimaryHSCPTracks.size();j++){
//      printf("HSCP PRIMARY TRACK : phi = %6.2f  eta = %6.2f\n",PrimaryHSCPTracks[j]->momentum().phi(), PrimaryHSCPTracks[j]->momentum().eta());
      MinDt[j] = 9999;
   }

   edm::Handle<std::vector< PSimHit > > h_CSC_Hits;
   iEvent.getByLabel("g4SimHits","MuonCSCHits", h_CSC_Hits);
   std::vector< PSimHit > CSC_Hits = *h_CSC_Hits.product();

   edm::Handle<std::vector< PSimHit > > h_DT_Hits;
   iEvent.getByLabel("g4SimHits","MuonDTHits", h_DT_Hits);
   std::vector< PSimHit > DT_Hits = *h_DT_Hits.product();

//  edm::Handle<std::vector< PSimHit > > h_RPC_Hits;
//  iEvent.getByLabel("g4SimHits","MuonRPCHits", h_RPC_Hits);
//  std::vector< PSimHit > RPC_Hits = *h_RPC_Hits.product();

   ESHandle<CSCGeometry> CSC_Geom;
   iSetup.get<MuonGeometryRecord>().get(CSC_Geom);

   ESHandle<DTGeometry> DT_Geom;
   iSetup.get<MuonGeometryRecord>().get(DT_Geom);

//  ESHandle<RPCGeometry> RPC_Geom;
//  iSetup.get<MuonGeometryRecord>().get(RPC_Geom);

   unsigned int k;
   for(k=0 ; k<CSC_Hits.size();k++)
   {
      if(abs(CSC_Hits[k].particleType()) >10000){

         unsigned int HSCP_Id      = ComesFromWhichHSCP(CSC_Hits[k].trackId(), PrimaryHSCPTracks, TrackColl, VertexColl);
         const CSCLayer* CSC_Layer = CSC_Geom->layer( (CSCDetId) CSC_Hits[k].detUnitId() );
         GlobalPoint GP            = CSC_Layer->toGlobal(CSC_Hits[k].entryPoint());
         double DistFromIP         = sqrt( GP.x()*GP.x() + GP.y()*GP.y() + GP.z()*GP.z() ) ;
         double T0                 = DistFromIP / 29.9;  //(10/c)
//        printf("CSC_Hits %3i : TrackID = %6i   :  DistFromIP = %8.2fcm : tof = %6.2fns",k,HSCP_Id, DistFromIP, CSC_Hits[k].timeOfFlight());
//        printf(" : tof (if beta=1) =  %6.2fns --> DeltaT = %6.2f\n", T0, CSC_Hits[k].timeOfFlight() - T0);

         if(MinDt[HSCP_Id] > CSC_Hits[k].timeOfFlight() - T0 &&  CSC_Hits[k].timeOfFlight() - T0 > 0) MinDt[HSCP_Id] = CSC_Hits[k].timeOfFlight() - T0;
      }
   }

   for(k=0 ; k<DT_Hits.size();k++)
   {
      if(abs(DT_Hits[k].particleType()) >10000){
         unsigned int HSCP_Id      = ComesFromWhichHSCP(DT_Hits[k].trackId(), PrimaryHSCPTracks, TrackColl, VertexColl);
         const DTLayer* DT_Layer   = DT_Geom->layer( (DTLayerId) DT_Hits[k].detUnitId() );
         GlobalPoint GP            = DT_Layer->toGlobal(DT_Hits[k].entryPoint());
         double DistFromIP         = sqrt( GP.x()*GP.x() + GP.y()*GP.y() + GP.z()*GP.z() ) ;
         double T0                 = DistFromIP / 29.9;  //(10/c)
//        printf(" DT_Hits %3i : TrackID = %6i   :  DistFromIP = %8.2fcm : tof = %6.2fns",k,HSCP_Id, DistFromIP, DT_Hits[k].timeOfFlight());
//        printf(" : tof (if beta=1) =  %6.2fns --> DeltaT = %6.2f\n", T0, DT_Hits[k].timeOfFlight() - T0);

        if(MinDt[HSCP_Id] > DT_Hits[k].timeOfFlight() - T0 &&  DT_Hits[k].timeOfFlight() - T0 > 0) MinDt[HSCP_Id] = DT_Hits[k].timeOfFlight() - T0;
      }
   }



//  for(k=0 ; k<RPC_Hits.size();k++)
//  {
//     if(abs(RPC_Hits[k].particleType()) >10000){
//        unsigned int HSCP_Id      = ComesFromWhichHSCP(RPC_Hits[k].trackId(), PrimaryHSCPTracks);
//        const RPCRoll* RPC_Layer  = RPC_Geom->roll( (RPCDetId) RPC_Hits[k].detUnitId() );
//        GlobalPoint GP            = RPC_Layer->toGlobal(DT_Hits[k].entryPoint());
//        double DistFromIP         = sqrt( GP.x()*GP.x() + GP.y()*GP.y() + GP.z()*GP.z() ) ;
//        double T0                 = DistFromIP / 29.9;  //(10/c)
////        printf("RPC_Hits %3i : TrackID = %6i   :  DistFromIP = %8.2fcm : tof = %6.2fns",k,HSCP_Id, DistFromIP, RPC_Hits[k].timeOfFlight());
////        printf(" : tof (if beta=1) =  %6.2fns --> DelatT = %6.2f\n", T0, RPC_Hits[k].timeOfFlight() - T0);
//
//        if(MinDt[HSCP_Id] > RPC_Hits[k].timeOfFlight() - T0 &&  RPC_Hits[k].timeOfFlight() - T0 > 0) MinDt[HSCP_Id] = RPC_Hits[k].timeOfFlight() - T0;
//     }
//  } 


  for(unsigned int i=0;i<PrimaryHSCPTracks.size();i++){

     recoL1Muon[i] = ClosestL1Muon(PrimaryHSCPTracks[i]->momentum().phi(), PrimaryHSCPTracks[i]->momentum().eta(), 0.3, L1_Muons);

     if(MinDt[i]!=9999.0){
//        printf("HSCP %i : Min Delta T = %6.2f | HSCP Beta = %6.2f",i, MinDt[i], PrimaryHSCPTracks[i]->momentum().beta());
//        printf(" | Reco L1 muon = %i\n",recoL1Muon[i]);
     }
  }

  return;
}



int
ClosestL1Muon(double phi, double eta, double dRMax, const l1extra::L1MuonParticleCollection L1_Muons)
{
       double dR = 99999; int J=-1;

       for(unsigned int j=0;j<L1_Muons.size();j++){
          if(dR > DeltaR(phi,eta, L1_Muons[j].phi(), L1_Muons[j].eta()) ){
             dR = DeltaR(phi,eta, L1_Muons[j].phi(), L1_Muons[j].eta());
             J  = j;
          }
       }

       if(J>=0 && dR<=dRMax)  return J;
       return -1;
}

int
ComesFromWhichHSCP(unsigned int TrackId, std::vector<SimTrack*> HSCPs, std::vector<SimTrack>  TrackColl, std::vector<SimVertex> VertexColl)
{
   SimTrack* Track = NULL;
   for(unsigned int j=0;(j<TrackColl.size() && Track==NULL);j++){
      if( TrackId == TrackColl[j].trackId() ) Track = &TrackColl[j];
   }
   if(Track==NULL)return -5;
  
   int ID = -3;
   double DRMin = 9999;
   for(unsigned int i=0;i<HSCPs.size();i++){
      double DR = DeltaR(Track->momentum().phi(), Track->momentum().eta(), HSCPs[i]->momentum().phi(), HSCPs[i]->momentum().eta() );
      if(DR<DRMin){DRMin = DR; ID = i;}
   }

  if(DRMin<0.3)return ID;
  
  return -1;
}



double
DeltaR(double phi1, double eta1, double phi2, double eta2)
{
        double deltaphi=phi1-phi2;

        if(fabs(deltaphi)>3.14)deltaphi=2*3.14-fabs(deltaphi);
        else if(fabs(deltaphi)<3.14)deltaphi=fabs(deltaphi);
        return (sqrt(pow(deltaphi,2)+pow(eta1 - eta2,2)));
}


