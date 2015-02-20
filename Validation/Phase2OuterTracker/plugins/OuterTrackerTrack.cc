// -*- C++ -*-
//
// Package:    Phase2OuterTracker
// Class:      OuterTrackerTrack
// 
/**\class Phase2OuterTracker OuterTrackerTrack.cc Validation/Phase2OuterTracker/plugins/OuterTrackerTrack.cc

 Description: [one line class summary]

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  Lieselotte Moreels
//         Created:  Tue, 17 Feb 2015 13:46:36 GMT
// $Id$
//
//


// system include files
#include <memory>
#include <vector>
#include <numeric>
#include <fstream>
#include <math.h>
#include "TNamed.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/Common/interface/DetSetVectorNew.h"
#include "DQM/SiStripCommon/interface/SiStripFolderOrganizer.h"
#include "Validation/Phase2OuterTracker/interface/OuterTrackerTrack.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"

// For TrackingParticles
#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticle.h"
#include "SimTracker/TrackTriggerAssociation/interface/TTClusterAssociationMap.h"
#include "SimTracker/TrackTriggerAssociation/interface/TTStubAssociationMap.h"
#include "SimTracker/TrackTriggerAssociation/interface/TTTrackAssociationMap.h"
#include "DataFormats/L1TrackTrigger/interface/TTTypes.h"
#include "DataFormats/L1TrackTrigger/interface/TTTrack.h"

#include "Geometry/TrackerGeometryBuilder/interface/StackedTrackerGeometry.h"
#include "Geometry/Records/interface/StackedTrackerGeometryRecord.h"

#include "TMath.h"
#include <iostream>

//
// constructors and destructor
//
OuterTrackerTrack::OuterTrackerTrack(const edm::ParameterSet& iConfig)
: dqmStore_(edm::Service<DQMStore>().operator->()), conf_(iConfig)
{
  //now do what ever initialization is needed
  topFolderName_ = conf_.getParameter<std::string>("TopFolderName");
}


OuterTrackerTrack::~OuterTrackerTrack()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called for each event  ------------
void
OuterTrackerTrack::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  /// Track Trigger
  edm::Handle< std::vector< TTTrack< Ref_PixelDigi_ > > >            PixelDigiTTTrackHandle;
  //iEvent.getByLabel( tagTTTracks, PixelDigiTTTrackHandle );		//initialise tags from cfg file in constructor
  iEvent.getByLabel( "TTTracksFromPixelDigis", "Level1TTTracks",PixelDigiTTTrackHandle );
  
  /// Track Trigger MC Truth
  edm::Handle< TTClusterAssociationMap< Ref_PixelDigi_ > > MCTruthTTClusterHandle;
  edm::Handle< TTStubAssociationMap< Ref_PixelDigi_ > >    MCTruthTTStubHandle;
  edm::Handle< TTTrackAssociationMap< Ref_PixelDigi_ > >   MCTruthTTTrackHandle;
  //iEvent.getByLabel( tagTTClusterMCTruth, MCTruthTTClusterHandle );
  iEvent.getByLabel( "TTClusterAssociatorFromPixelDigis", "ClusterInclusive", MCTruthTTClusterHandle );
  //iEvent.getByLabel( tagTTStubMCTruth, MCTruthTTStubHandle );
  iEvent.getByLabel( "TTStubAssociatorFromPixelDigis", "StubAccepted",        MCTruthTTStubHandle );
  //iEvent.getByLabel( tagTTTrackMCTruth, MCTruthTTTrackHandle );
  iEvent.getByLabel( "TTTrackAssociatorFromPixelDigis", "Level1TTTracks", MCTruthTTTrackHandle );
	
	
  /// TrackingParticles
  edm::Handle< std::vector< TrackingParticle > > TrackingParticleHandle;
  iEvent.getByLabel( "mix", "MergedTrackTruth", TrackingParticleHandle );
  edm::Handle< std::vector< TrackingVertex > > TrackingVertexHandle;
  iEvent.getByLabel( "mix", "MergedTrackTruth", TrackingVertexHandle );
  
  
  unsigned int num3Stubs = 0;
  unsigned int num2Stubs = 0;
  
  /// Go on only if there are TTTracks from PixelDigis
  if ( PixelDigiTTTrackHandle->size() > 0 )
  {
    /// Loop over TTTracks
    unsigned int tkCnt = 0;
    std::vector< TTTrack< Ref_PixelDigi_ > >::const_iterator iterTTTrack;
    for ( iterTTTrack = PixelDigiTTTrackHandle->begin();
         iterTTTrack != PixelDigiTTTrackHandle->end();
         ++iterTTTrack )
    {
      /// Make the pointer
      edm::Ptr< TTTrack< Ref_PixelDigi_ > > tempTrackPtr( PixelDigiTTTrackHandle, tkCnt++ );
      
      
      unsigned int nStubs     = tempTrackPtr->getStubRefs().size();
      
      //double trackRInv  = tempTrackPtr->getRInv();
      double trackPt    = tempTrackPtr->getMomentum().perp();
      double trackPhi   = tempTrackPtr->getMomentum().phi();
      double trackEta   = tempTrackPtr->getMomentum().eta();
//      double trackTheta = tempTrackPtr->getMomentum().theta();
      double trackVtxZ0 = tempTrackPtr->getPOCA().z();
      double trackChi2  = tempTrackPtr->getChi2();
      double trackChi2R = tempTrackPtr->getChi2Red();
      
      
      /// Check if TTTrack is genuine
      bool genuineTrack = MCTruthTTTrackHandle->isGenuine( tempTrackPtr );
      
      if ( !genuineTrack ) continue;
      
      edm::Ptr< TrackingParticle > tpPtr = MCTruthTTTrackHandle->findTrackingParticlePtr( tempTrackPtr );
      
      /// Get the corresponding vertex and reject the track
      /// if its vertex is outside the beampipe
      if ( tpPtr->vertex().rho() >= 2 )
        continue;
      
      Track_NStubs->Fill(nStubs);
      
      double tpPt = tpPtr->p4().pt();
      double tpEta = tpPtr->momentum().eta();
//      double tpTheta = tpPtr->momentum().theta();
      double tpPhi = tpPtr->momentum().phi();
      double tpVtxZ0 = tpPtr->vertex().z();
      
      if ( nStubs > 2 )
      {
        
        //TPart_Eta_Normalization->Fill( tpEta );
        //TPart_Eta_NStubs->Fill( tpEta, nStubs );
        
        Track_3Stubs_Pt->Fill( trackPt );
        Track_3Stubs_Eta->Fill( trackEta );
        Track_3Stubs_Phi->Fill( trackPhi );
        
        num3Stubs++;
        Track_3Stubs_Pt_TPart_Pt->Fill( tpPt, trackPt );
        Track_3Stubs_PtRes_TPart_Eta->Fill( tpEta, trackPt - tpPt );
        Track_3Stubs_InvPt_TPart_InvPt->Fill( 1./tpPt, 1./trackPt );
        Track_3Stubs_InvPtRes_TPart_Eta->Fill( tpEta, 1./trackPt - 1./tpPt );
        Track_3Stubs_Phi_TPart_Phi->Fill( tpPhi, trackPhi );
        Track_3Stubs_PhiRes_TPart_Eta->Fill( tpEta, trackPhi - tpPhi );
        Track_3Stubs_Eta_TPart_Eta->Fill( tpEta, trackEta );
        Track_3Stubs_EtaRes_TPart_Eta->Fill( tpEta, trackEta - tpEta );
        Track_3Stubs_VtxZ0_TPart_VtxZ0->Fill( tpVtxZ0, trackVtxZ0 );
        Track_3Stubs_VtxZ0Res_TPart_Eta->Fill( tpEta, trackVtxZ0 - tpVtxZ0 );
        Track_3Stubs_Chi2_NStubs->Fill( nStubs, trackChi2 );
        //Track_3Stubs_Chi2_TPart_Eta->Fill( tpEta, trackChi2 );
        Track_3Stubs_Chi2Red_NStubs->Fill( nStubs, trackChi2R );
        //Track_3Stubs_Chi2Red_TPart_Eta->Fill( tpEta, trackChi2R );
      }
      else
      {
        Track_2Stubs_Pt->Fill( trackPt );
        Track_2Stubs_Eta->Fill( trackEta );
        Track_2Stubs_Phi->Fill( trackPhi );
        
        num2Stubs++;
        Track_2Stubs_Pt_TPart_Pt->Fill( tpPt, trackPt );
        Track_2Stubs_PtRes_TPart_Eta->Fill( tpEta, trackPt - tpPt );
        Track_2Stubs_InvPt_TPart_InvPt->Fill( 1./tpPt, 1./trackPt );
        Track_2Stubs_InvPtRes_TPart_Eta->Fill( tpEta, 1./trackPt - 1./tpPt );
        Track_2Stubs_Phi_TPart_Phi->Fill( tpPhi, trackPhi );
        Track_2Stubs_PhiRes_TPart_Eta->Fill( tpEta, trackPhi - tpPhi );
        Track_2Stubs_Eta_TPart_Eta->Fill( tpEta, trackEta );
        Track_2Stubs_EtaRes_TPart_Eta->Fill( tpEta, trackEta - tpEta );
        Track_2Stubs_VtxZ0_TPart_VtxZ0->Fill( tpVtxZ0, trackVtxZ0 );
        Track_2Stubs_VtxZ0Res_TPart_Eta->Fill( tpEta, trackVtxZ0 - tpVtxZ0 );
        Track_2Stubs_Chi2_NStubs->Fill( nStubs, trackChi2 );
        //Track_2Stubs_Chi2_TPart_Eta->Fill( tpEta, trackChi2 );
        Track_2Stubs_Chi2Red_NStubs->Fill( nStubs, trackChi2R );
        //Track_2Stubs_Chi2Red_TPart_Eta->Fill( tpEta, trackChi2R );      
      }
    } /// End of loop over TTTracks
  }
  
  Track_2Stubs_N->Fill( num2Stubs );
  Track_3Stubs_N->Fill( num3Stubs );
	
}


// ------------ method called when starting to process a run  ------------

void 
OuterTrackerTrack::beginRun(edm::Run const&, edm::EventSetup const&)
{
  SiStripFolderOrganizer folder_organizer;
  folder_organizer.setSiStripFolderName(topFolderName_);
  folder_organizer.setSiStripFolder();
  std::string HistoName;
  
  dqmStore_->setCurrentFolder(topFolderName_+"/Tracks/");
  
  // Number of TTStubs per TTTrack
  edm::ParameterSet psTrack_NStubs =  conf_.getParameter<edm::ParameterSet>("TH1TTTrack_NStubs");
  HistoName = "Track_NStubs";
  Track_NStubs = dqmStore_->book1D(HistoName, HistoName,
      psTrack_NStubs.getParameter<int32_t>("Nbinsx"),
      psTrack_NStubs.getParameter<double>("xmin"),
      psTrack_NStubs.getParameter<double>("xmax"));
  Track_NStubs->setAxisTitle("# TTStubs", 1);
  Track_NStubs->setAxisTitle("# Genuine TTTracks", 2);
  
  
  
  
  /// Plots where all TTTracks are made from up to 2 TTStubs
  dqmStore_->setCurrentFolder(topFolderName_+"/Tracks/2Stubs");
  
  // Number of TTTracks
  edm::ParameterSet psTrack_N =  conf_.getParameter<edm::ParameterSet>("TH1TTTrack_N");
  HistoName = "Track_2Stubs_N";
  Track_2Stubs_N = dqmStore_->book1D(HistoName, HistoName,
      psTrack_N.getParameter<int32_t>("Nbinsx"),
      psTrack_N.getParameter<double>("xmin"),
      psTrack_N.getParameter<double>("xmax"));
  Track_2Stubs_N->setAxisTitle("# Genuine TTTracks from up to 2 TTStubs", 1);
  Track_2Stubs_N->setAxisTitle("# Events", 2);
  
  // Pt
  edm::ParameterSet psTrack_Pt =  conf_.getParameter<edm::ParameterSet>("TH1TTTrack_Pt");
  HistoName = "Track_2Stubs_Pt";
  Track_2Stubs_Pt = dqmStore_->book1D(HistoName, HistoName,
      psTrack_Pt.getParameter<int32_t>("Nbinsx"),
      psTrack_Pt.getParameter<double>("xmin"),
      psTrack_Pt.getParameter<double>("xmax"));
  Track_2Stubs_Pt->setAxisTitle("TTTrack Pt", 1);
  Track_2Stubs_Pt->setAxisTitle("# TTTracks", 2);
  
  // Eta
  edm::ParameterSet psTrack_Eta =  conf_.getParameter<edm::ParameterSet>("TH1TTTrack_Eta");
  HistoName = "Track_2Stubs_Eta";
  Track_2Stubs_Eta = dqmStore_->book1D(HistoName, HistoName,
      psTrack_Eta.getParameter<int32_t>("Nbinsx"),
      psTrack_Eta.getParameter<double>("xmin"),
      psTrack_Eta.getParameter<double>("xmax"));
  Track_2Stubs_Eta->setAxisTitle("TTTrack Eta", 1);
  Track_2Stubs_Eta->setAxisTitle("# TTTracks", 2);
  
  // Phi
  edm::ParameterSet psTrack_Phi =  conf_.getParameter<edm::ParameterSet>("TH1TTTrack_Phi");
  HistoName = "Track_2Stubs_Phi";
  Track_2Stubs_Phi = dqmStore_->book1D(HistoName, HistoName,
      psTrack_Phi.getParameter<int32_t>("Nbinsx"),
      psTrack_Phi.getParameter<double>("xmin"),
      psTrack_Phi.getParameter<double>("xmax"));
  Track_2Stubs_Phi->setAxisTitle("TTTrack Phi", 1);
  Track_2Stubs_Phi->setAxisTitle("# TTTracks", 2);
  
  // TTTrack Chi2 vs Nstubs
  edm::ParameterSet psTrack_Chi2 =  conf_.getParameter<edm::ParameterSet>("TH2TTTrack_Chi2");
  HistoName = "Track_2Stubs_Chi2_NStubs";
  Track_2Stubs_Chi2_NStubs = dqmStore_->book2D(HistoName, HistoName,
      psTrack_Chi2.getParameter<int32_t>("Nbinsx"),
      psTrack_Chi2.getParameter<double>("xmin"),
      psTrack_Chi2.getParameter<double>("xmax"),
      psTrack_Chi2.getParameter<int32_t>("Nbinsy"),
      psTrack_Chi2.getParameter<double>("ymin"),
      psTrack_Chi2.getParameter<double>("ymax"));
  Track_2Stubs_Chi2_NStubs->setAxisTitle("# TTStubs", 1);
  Track_2Stubs_Chi2_NStubs->setAxisTitle("TTTrack #chi^{2}", 2);
  
  // TTTrack Chi2/ndf vs Nstubs
  edm::ParameterSet psTrack_Chi2Red =  conf_.getParameter<edm::ParameterSet>("TH2TTTrack_Chi2Red");
  HistoName = "Track_2Stubs_Chi2_NStubs";
  Track_2Stubs_Chi2Red_NStubs = dqmStore_->book2D(HistoName, HistoName,
      psTrack_Chi2Red.getParameter<int32_t>("Nbinsx"),
      psTrack_Chi2Red.getParameter<double>("xmin"),
      psTrack_Chi2Red.getParameter<double>("xmax"),
      psTrack_Chi2Red.getParameter<int32_t>("Nbinsy"),
      psTrack_Chi2Red.getParameter<double>("ymin"),
      psTrack_Chi2Red.getParameter<double>("ymax"));
  Track_2Stubs_Chi2Red_NStubs->setAxisTitle("# TTStubs", 1);
  Track_2Stubs_Chi2Red_NStubs->setAxisTitle("TTTrack #chi^{2}/ndf", 2);
  
  
  
  
  /// Plots where all TTTracks are made from at least 3 TTStubs
  dqmStore_->setCurrentFolder(topFolderName_+"/Tracks/3Stubs/");
  
  // Number of TTTracks
  HistoName = "Track_3Stubs_N";
  Track_3Stubs_N = dqmStore_->book1D(HistoName, HistoName,
      psTrack_N.getParameter<int32_t>("Nbinsx"),
      psTrack_N.getParameter<double>("xmin"),
      psTrack_N.getParameter<double>("xmax"));
  Track_3Stubs_N->setAxisTitle("# Genuine TTTracks from at least 3 TTStubs", 1);
  Track_3Stubs_N->setAxisTitle("# Events", 2);
  
  // Pt
  HistoName = "Track_3Stubs_Pt";
  Track_3Stubs_Pt = dqmStore_->book1D(HistoName, HistoName,
      psTrack_Pt.getParameter<int32_t>("Nbinsx"),
      psTrack_Pt.getParameter<double>("xmin"),
      psTrack_Pt.getParameter<double>("xmax"));
  Track_3Stubs_Pt->setAxisTitle("TTTrack Pt", 1);
  Track_3Stubs_Pt->setAxisTitle("# TTTracks", 2);
  
  // Eta
  HistoName = "Track_3Stubs_Eta";
  Track_3Stubs_Eta = dqmStore_->book1D(HistoName, HistoName,
      psTrack_Eta.getParameter<int32_t>("Nbinsx"),
      psTrack_Eta.getParameter<double>("xmin"),
      psTrack_Eta.getParameter<double>("xmax"));
  Track_3Stubs_Eta->setAxisTitle("TTTrack Eta", 1);
  Track_3Stubs_Eta->setAxisTitle("# TTTracks", 2);
  
  // Phi
  HistoName = "Track_3Stubs_Phi";
  Track_3Stubs_Phi = dqmStore_->book1D(HistoName, HistoName,
      psTrack_Phi.getParameter<int32_t>("Nbinsx"),
      psTrack_Phi.getParameter<double>("xmin"),
      psTrack_Phi.getParameter<double>("xmax"));
  Track_3Stubs_Phi->setAxisTitle("TTTrack Phi", 1);
  Track_3Stubs_Phi->setAxisTitle("# TTTracks", 2);
  
  // TTTrack Chi2 vs Nstubs
  HistoName = "Track_3Stubs_Chi2_NStubs";
  Track_3Stubs_Chi2_NStubs = dqmStore_->book2D(HistoName, HistoName,
      psTrack_Chi2.getParameter<int32_t>("Nbinsx"),
      psTrack_Chi2.getParameter<double>("xmin"),
      psTrack_Chi2.getParameter<double>("xmax"),
      psTrack_Chi2.getParameter<int32_t>("Nbinsy"),
      psTrack_Chi2.getParameter<double>("ymin"),
      psTrack_Chi2.getParameter<double>("ymax"));
  Track_3Stubs_Chi2_NStubs->setAxisTitle("# TTStubs", 1);
  Track_3Stubs_Chi2_NStubs->setAxisTitle("TTTrack #chi^{2}", 2);
  
  // TTTrack Chi2/ndf vs Nstubs
  HistoName = "Track_3Stubs_Chi2_NStubs";
  Track_3Stubs_Chi2Red_NStubs = dqmStore_->book2D(HistoName, HistoName,
      psTrack_Chi2Red.getParameter<int32_t>("Nbinsx"),
      psTrack_Chi2Red.getParameter<double>("xmin"),
      psTrack_Chi2Red.getParameter<double>("xmax"),
      psTrack_Chi2Red.getParameter<int32_t>("Nbinsy"),
     	psTrack_Chi2Red.getParameter<double>("ymin"),
      psTrack_Chi2Red.getParameter<double>("ymax"));
  Track_3Stubs_Chi2Red_NStubs->setAxisTitle("# TTStubs", 1);
  Track_3Stubs_Chi2Red_NStubs->setAxisTitle("TTTrack #chi^{2}/ndf", 2);
  
  
  
  
  /// Track properties compared to TParticles
  dqmStore_->setCurrentFolder(topFolderName_+"/TTTrackVSTPart/2Stubs/");
  
  // Pt
  edm::ParameterSet psTrack_Sim_Pt =  conf_.getParameter<edm::ParameterSet>("TH2TTTrack_Sim_Pt");
  HistoName = "Track_2Stubs_Pt_TPart_Pt";
  Track_2Stubs_Pt_TPart_Pt = dqmStore_->book2D(HistoName, HistoName,
      psTrack_Sim_Pt.getParameter<int32_t>("Nbinsx"),
      psTrack_Sim_Pt.getParameter<double>("xmin"),
      psTrack_Sim_Pt.getParameter<double>("xmax"),
      psTrack_Sim_Pt.getParameter<int32_t>("Nbinsy"),
      psTrack_Sim_Pt.getParameter<double>("ymin"),
      psTrack_Sim_Pt.getParameter<double>("ymax"));
  Track_2Stubs_Pt_TPart_Pt->setAxisTitle("TPart Pt", 1);
  Track_2Stubs_Pt_TPart_Pt->setAxisTitle("TTTrack Pt", 2);
  
  // PtRes
  edm::ParameterSet psTrack_Sim_PtRes =  conf_.getParameter<edm::ParameterSet>("TH2TTTrack_Sim_PtRes");
  HistoName = "Track_2Stubs_PtRes_TPart_Eta";
  Track_2Stubs_PtRes_TPart_Eta = dqmStore_->book2D(HistoName, HistoName,
      psTrack_Sim_PtRes.getParameter<int32_t>("Nbinsx"),
      psTrack_Sim_PtRes.getParameter<double>("xmin"),
      psTrack_Sim_PtRes.getParameter<double>("xmax"),
      psTrack_Sim_PtRes.getParameter<int32_t>("Nbinsy"),
      psTrack_Sim_PtRes.getParameter<double>("ymin"),
      psTrack_Sim_PtRes.getParameter<double>("ymax"));
  Track_2Stubs_PtRes_TPart_Eta->setAxisTitle("TPart Eta", 1);
  Track_2Stubs_PtRes_TPart_Eta->setAxisTitle("TTTrack Pt - TPart Pt", 2);
  
  // InvPt
  edm::ParameterSet psTrack_Sim_InvPt =  conf_.getParameter<edm::ParameterSet>("TH2TTTrack_Sim_InvPt");
  HistoName = "Track_2Stubs_InvPt_TPart_InvPt";
  Track_2Stubs_InvPt_TPart_InvPt = dqmStore_->book2D(HistoName, HistoName,
      psTrack_Sim_InvPt.getParameter<int32_t>("Nbinsx"),
      psTrack_Sim_InvPt.getParameter<double>("xmin"),
      psTrack_Sim_InvPt.getParameter<double>("xmax"),
      psTrack_Sim_InvPt.getParameter<int32_t>("Nbinsy"),
      psTrack_Sim_InvPt.getParameter<double>("ymin"),
      psTrack_Sim_InvPt.getParameter<double>("ymax"));
  Track_2Stubs_InvPt_TPart_InvPt->setAxisTitle("TPart 1/Pt", 1);
  Track_2Stubs_InvPt_TPart_InvPt->setAxisTitle("TTTrack 1/Pt", 2);
  
  // InvPtRes
  edm::ParameterSet psTrack_Sim_InvPtRes =  conf_.getParameter<edm::ParameterSet>("TH2TTTrack_Sim_InvPtRes");
  HistoName = "Track_2Stubs_InvPtRes_TPart_Eta";
  Track_2Stubs_InvPtRes_TPart_Eta = dqmStore_->book2D(HistoName, HistoName,
      psTrack_Sim_InvPtRes.getParameter<int32_t>("Nbinsx"),
      psTrack_Sim_InvPtRes.getParameter<double>("xmin"),
      psTrack_Sim_InvPtRes.getParameter<double>("xmax"),
      psTrack_Sim_InvPtRes.getParameter<int32_t>("Nbinsy"),
      psTrack_Sim_InvPtRes.getParameter<double>("ymin"),
      psTrack_Sim_InvPtRes.getParameter<double>("ymax"));
  Track_2Stubs_InvPtRes_TPart_Eta->setAxisTitle("TPart Eta", 1);
  Track_2Stubs_InvPtRes_TPart_Eta->setAxisTitle("TTTrack 1/Pt - TPart 1/Pt", 2);
  
  // Phi
  edm::ParameterSet psTrack_Sim_Phi =  conf_.getParameter<edm::ParameterSet>("TH2TTTrack_Sim_Phi");
  HistoName = "Track_2Stubs_Phi_TPart_Phi";
  Track_2Stubs_Phi_TPart_Phi = dqmStore_->book2D(HistoName, HistoName,
      psTrack_Sim_Phi.getParameter<int32_t>("Nbinsx"),
      psTrack_Sim_Phi.getParameter<double>("xmin"),
      psTrack_Sim_Phi.getParameter<double>("xmax"),
      psTrack_Sim_Phi.getParameter<int32_t>("Nbinsy"),
      psTrack_Sim_Phi.getParameter<double>("ymin"),
      psTrack_Sim_Phi.getParameter<double>("ymax"));
  Track_2Stubs_Phi_TPart_Phi->setAxisTitle("TPart Phi", 1);
  Track_2Stubs_Phi_TPart_Phi->setAxisTitle("TTTrack Phi", 2);
  
  // PhiRes
  edm::ParameterSet psTrack_Sim_PhiRes =  conf_.getParameter<edm::ParameterSet>("TH2TTTrack_Sim_PhiRes");
  HistoName = "Track_2Stubs_PhiRes_TPart_Eta";
  Track_2Stubs_PhiRes_TPart_Eta = dqmStore_->book2D(HistoName, HistoName,
      psTrack_Sim_PhiRes.getParameter<int32_t>("Nbinsx"),
      psTrack_Sim_PhiRes.getParameter<double>("xmin"),
      psTrack_Sim_PhiRes.getParameter<double>("xmax"),
      psTrack_Sim_PhiRes.getParameter<int32_t>("Nbinsy"),
      psTrack_Sim_PhiRes.getParameter<double>("ymin"),
      psTrack_Sim_PhiRes.getParameter<double>("ymax"));
  Track_2Stubs_PhiRes_TPart_Eta->setAxisTitle("TPart Eta", 1);
  Track_2Stubs_PhiRes_TPart_Eta->setAxisTitle("TTTrack Phi - TPart Phi", 2);
  
  // Eta
  edm::ParameterSet psTrack_Sim_Eta =  conf_.getParameter<edm::ParameterSet>("TH2TTTrack_Sim_Eta");
  HistoName = "Track_2Stubs_Eta_TPart_Eta";
  Track_2Stubs_Eta_TPart_Eta = dqmStore_->book2D(HistoName, HistoName,
      psTrack_Sim_Eta.getParameter<int32_t>("Nbinsx"),
      psTrack_Sim_Eta.getParameter<double>("xmin"),
      psTrack_Sim_Eta.getParameter<double>("xmax"),
      psTrack_Sim_Eta.getParameter<int32_t>("Nbinsy"),
      psTrack_Sim_Eta.getParameter<double>("ymin"),
      psTrack_Sim_Eta.getParameter<double>("ymax"));
  Track_2Stubs_Eta_TPart_Eta->setAxisTitle("TPart Eta", 1);
  Track_2Stubs_Eta_TPart_Eta->setAxisTitle("TTTrack Eta", 2);
  
  // EtaRes
  edm::ParameterSet psTrack_Sim_EtaRes =  conf_.getParameter<edm::ParameterSet>("TH2TTTrack_Sim_EtaRes");
  HistoName = "Track_2Stubs_EtaRes_TPart_Eta";
  Track_2Stubs_EtaRes_TPart_Eta = dqmStore_->book2D(HistoName, HistoName,
      psTrack_Sim_EtaRes.getParameter<int32_t>("Nbinsx"),
      psTrack_Sim_EtaRes.getParameter<double>("xmin"),
      psTrack_Sim_EtaRes.getParameter<double>("xmax"),
      psTrack_Sim_EtaRes.getParameter<int32_t>("Nbinsy"),
      psTrack_Sim_EtaRes.getParameter<double>("ymin"),
      psTrack_Sim_EtaRes.getParameter<double>("ymax"));
  Track_2Stubs_EtaRes_TPart_Eta->setAxisTitle("TPart Eta", 1);
  Track_2Stubs_EtaRes_TPart_Eta->setAxisTitle("TTTrack Eta - TPart Eta", 2);
  
  // Vertex position in z
  edm::ParameterSet psTrack_Sim_Vtx =  conf_.getParameter<edm::ParameterSet>("TH2TTTrack_Sim_Vtx");
  HistoName = "Track_2Stubs_VtxZ0_TPart_VtxZ0";
  Track_2Stubs_VtxZ0_TPart_VtxZ0 = dqmStore_->book2D(HistoName, HistoName,
      psTrack_Sim_Vtx.getParameter<int32_t>("Nbinsx"),
      psTrack_Sim_Vtx.getParameter<double>("xmin"),
      psTrack_Sim_Vtx.getParameter<double>("xmax"),
      psTrack_Sim_Vtx.getParameter<int32_t>("Nbinsy"),
      psTrack_Sim_Vtx.getParameter<double>("ymin"),
      psTrack_Sim_Vtx.getParameter<double>("ymax"));
  Track_2Stubs_VtxZ0_TPart_VtxZ0->setAxisTitle("TPart Vertex Position in z", 1);
  Track_2Stubs_VtxZ0_TPart_VtxZ0->setAxisTitle("TTTrack Eta Vertex Position in z", 2);
  
  // Vertex position in z
  edm::ParameterSet psTrack_Sim_VtxRes =  conf_.getParameter<edm::ParameterSet>("TH2TTTrack_Sim_VtxRes");
  HistoName = "Track_2Stubs_VtxZ0Res_TPart_Eta";
  Track_2Stubs_VtxZ0Res_TPart_Eta = dqmStore_->book2D(HistoName, HistoName,
      psTrack_Sim_VtxRes.getParameter<int32_t>("Nbinsx"),
      psTrack_Sim_VtxRes.getParameter<double>("xmin"),
      psTrack_Sim_VtxRes.getParameter<double>("xmax"),
      psTrack_Sim_VtxRes.getParameter<int32_t>("Nbinsy"),
      psTrack_Sim_VtxRes.getParameter<double>("ymin"),
      psTrack_Sim_VtxRes.getParameter<double>("ymax"));
  Track_2Stubs_VtxZ0Res_TPart_Eta->setAxisTitle("TPart Eta", 1);
  Track_2Stubs_VtxZ0Res_TPart_Eta->setAxisTitle("TTTrack - TPart Vertex Position in z", 2);
  
  
  
  dqmStore_->setCurrentFolder(topFolderName_+"/TTTrackVSTPart/3Stubs");
	
	// Pt
  HistoName = "Track_3Stubs_Pt_TPart_Pt";
  Track_3Stubs_Pt_TPart_Pt = dqmStore_->book2D(HistoName, HistoName,
      psTrack_Sim_Pt.getParameter<int32_t>("Nbinsx"),
      psTrack_Sim_Pt.getParameter<double>("xmin"),
      psTrack_Sim_Pt.getParameter<double>("xmax"),
      psTrack_Sim_Pt.getParameter<int32_t>("Nbinsy"),
      psTrack_Sim_Pt.getParameter<double>("ymin"),
      psTrack_Sim_Pt.getParameter<double>("ymax"));
  Track_3Stubs_Pt_TPart_Pt->setAxisTitle("TPart Pt", 1);
  Track_3Stubs_Pt_TPart_Pt->setAxisTitle("TTTrack Pt", 2);
  
  // PtRes
  HistoName = "Track_3Stubs_PtRes_TPart_Eta";
  Track_3Stubs_PtRes_TPart_Eta = dqmStore_->book2D(HistoName, HistoName,
      psTrack_Sim_PtRes.getParameter<int32_t>("Nbinsx"),
      psTrack_Sim_PtRes.getParameter<double>("xmin"),
      psTrack_Sim_PtRes.getParameter<double>("xmax"),
      psTrack_Sim_PtRes.getParameter<int32_t>("Nbinsy"),
      psTrack_Sim_PtRes.getParameter<double>("ymin"),
      psTrack_Sim_PtRes.getParameter<double>("ymax"));
  Track_3Stubs_PtRes_TPart_Eta->setAxisTitle("TPart Eta", 1);
  Track_3Stubs_PtRes_TPart_Eta->setAxisTitle("TTTrack Pt - TPart Pt", 2);
  
  // InvPt
  HistoName = "Track_3Stubs_InvPt_TPart_InvPt";
  Track_3Stubs_InvPt_TPart_InvPt = dqmStore_->book2D(HistoName, HistoName,
      psTrack_Sim_InvPt.getParameter<int32_t>("Nbinsx"),
      psTrack_Sim_InvPt.getParameter<double>("xmin"),
      psTrack_Sim_InvPt.getParameter<double>("xmax"),
      psTrack_Sim_InvPt.getParameter<int32_t>("Nbinsy"),
      psTrack_Sim_InvPt.getParameter<double>("ymin"),
      psTrack_Sim_InvPt.getParameter<double>("ymax"));
  Track_3Stubs_InvPt_TPart_InvPt->setAxisTitle("TPart 1/Pt", 1);
  Track_3Stubs_InvPt_TPart_InvPt->setAxisTitle("TTTrack 1/Pt", 2);
  
  // InvPtRes
  HistoName = "Track_3Stubs_InvPtRes_TPart_Eta";
  Track_3Stubs_InvPtRes_TPart_Eta = dqmStore_->book2D(HistoName, HistoName,
      psTrack_Sim_InvPtRes.getParameter<int32_t>("Nbinsx"),
      psTrack_Sim_InvPtRes.getParameter<double>("xmin"),
      psTrack_Sim_InvPtRes.getParameter<double>("xmax"),
      psTrack_Sim_InvPtRes.getParameter<int32_t>("Nbinsy"),
      psTrack_Sim_InvPtRes.getParameter<double>("ymin"),
      psTrack_Sim_InvPtRes.getParameter<double>("ymax"));
  Track_3Stubs_InvPtRes_TPart_Eta->setAxisTitle("TPart Eta", 1);
  Track_3Stubs_InvPtRes_TPart_Eta->setAxisTitle("TTTrack 1/Pt - TPart 1/Pt", 2);
  
  // Phi
  HistoName = "Track_3Stubs_Phi_TPart_Phi";
  Track_3Stubs_Phi_TPart_Phi = dqmStore_->book2D(HistoName, HistoName,
      psTrack_Sim_Phi.getParameter<int32_t>("Nbinsx"),
      psTrack_Sim_Phi.getParameter<double>("xmin"),
      psTrack_Sim_Phi.getParameter<double>("xmax"),
      psTrack_Sim_Phi.getParameter<int32_t>("Nbinsy"),
      psTrack_Sim_Phi.getParameter<double>("ymin"),
      psTrack_Sim_Phi.getParameter<double>("ymax"));
  Track_3Stubs_Phi_TPart_Phi->setAxisTitle("TPart Phi", 1);
  Track_3Stubs_Phi_TPart_Phi->setAxisTitle("TTTrack Phi", 2);
  
  // PhiRes
  HistoName = "Track_3Stubs_PhiRes_TPart_Eta";
  Track_3Stubs_PhiRes_TPart_Eta = dqmStore_->book2D(HistoName, HistoName,
      psTrack_Sim_PhiRes.getParameter<int32_t>("Nbinsx"),
      psTrack_Sim_PhiRes.getParameter<double>("xmin"),
      psTrack_Sim_PhiRes.getParameter<double>("xmax"),
      psTrack_Sim_PhiRes.getParameter<int32_t>("Nbinsy"),
      psTrack_Sim_PhiRes.getParameter<double>("ymin"),
      psTrack_Sim_PhiRes.getParameter<double>("ymax"));
  Track_3Stubs_PhiRes_TPart_Eta->setAxisTitle("TPart Eta", 1);
  Track_3Stubs_PhiRes_TPart_Eta->setAxisTitle("TTTrack Phi - TPart Phi", 2);
  
  // Eta
  HistoName = "Track_3Stubs_Eta_TPart_Eta";
  Track_3Stubs_Eta_TPart_Eta = dqmStore_->book2D(HistoName, HistoName,
      psTrack_Sim_Eta.getParameter<int32_t>("Nbinsx"),
      psTrack_Sim_Eta.getParameter<double>("xmin"),
      psTrack_Sim_Eta.getParameter<double>("xmax"),
      psTrack_Sim_Eta.getParameter<int32_t>("Nbinsy"),
      psTrack_Sim_Eta.getParameter<double>("ymin"),
      psTrack_Sim_Eta.getParameter<double>("ymax"));
  Track_3Stubs_Eta_TPart_Eta->setAxisTitle("TPart Eta", 1);
  Track_3Stubs_Eta_TPart_Eta->setAxisTitle("TTTrack Eta", 2);
  
  // EtaRes
  HistoName = "Track_3Stubs_EtaRes_TPart_Eta";
  Track_3Stubs_EtaRes_TPart_Eta = dqmStore_->book2D(HistoName, HistoName,
      psTrack_Sim_EtaRes.getParameter<int32_t>("Nbinsx"),
      psTrack_Sim_EtaRes.getParameter<double>("xmin"),
      psTrack_Sim_EtaRes.getParameter<double>("xmax"),
      psTrack_Sim_EtaRes.getParameter<int32_t>("Nbinsy"),
      psTrack_Sim_EtaRes.getParameter<double>("ymin"),
      psTrack_Sim_EtaRes.getParameter<double>("ymax"));
  Track_3Stubs_EtaRes_TPart_Eta->setAxisTitle("TPart Eta", 1);
  Track_3Stubs_EtaRes_TPart_Eta->setAxisTitle("TTTrack Eta - TPart Eta", 2);
  
  // Vertex position in z
  HistoName = "Track_3Stubs_VtxZ0_TPart_VtxZ0";
  Track_3Stubs_VtxZ0_TPart_VtxZ0 = dqmStore_->book2D(HistoName, HistoName,
      psTrack_Sim_Vtx.getParameter<int32_t>("Nbinsx"),
      psTrack_Sim_Vtx.getParameter<double>("xmin"),
      psTrack_Sim_Vtx.getParameter<double>("xmax"),
      psTrack_Sim_Vtx.getParameter<int32_t>("Nbinsy"),
      psTrack_Sim_Vtx.getParameter<double>("ymin"),
      psTrack_Sim_Vtx.getParameter<double>("ymax"));
  Track_3Stubs_VtxZ0_TPart_VtxZ0->setAxisTitle("TPart Vertex Position in z", 1);
  Track_3Stubs_VtxZ0_TPart_VtxZ0->setAxisTitle("TTTrack Vertex Position in z", 2);
  
  // Vertex position in z
  HistoName = "Track_3Stubs_VtxZ0Res_TPart_Eta";
  Track_3Stubs_VtxZ0Res_TPart_Eta = dqmStore_->book2D(HistoName, HistoName,
      psTrack_Sim_VtxRes.getParameter<int32_t>("Nbinsx"),
      psTrack_Sim_VtxRes.getParameter<double>("xmin"),
      psTrack_Sim_VtxRes.getParameter<double>("xmax"),
      psTrack_Sim_VtxRes.getParameter<int32_t>("Nbinsy"),
      psTrack_Sim_VtxRes.getParameter<double>("ymin"),
      psTrack_Sim_VtxRes.getParameter<double>("ymax"));
  Track_3Stubs_VtxZ0Res_TPart_Eta->setAxisTitle("TPart Eta", 1);
  Track_3Stubs_VtxZ0Res_TPart_Eta->setAxisTitle("TTTrack - TPart Vertex Position in z", 2);
}


// ------------ method called when ending the processing of a run  ------------
/*
void 
OuterTrackerTrack::endRun(edm::Run const&, edm::EventSetup const&)
{
}
*/
// ------------ method called once each job just before starting event loop  ------------
/*
void 
OuterTrackerTrack::beginJob()
{
}
*/

// ------------ method called once each job just after ending the event loop  ------------
/*
void 
OuterTrackerTrack::endJob() 
{
}
*/


//define this as a plug-in
DEFINE_FWK_MODULE(OuterTrackerTrack);
