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
  tagTTTracks_ = conf_.getParameter< edm::InputTag >("TTTracks");
  tagTTTrackMCTruth_ = conf_.getParameter< edm::InputTag >("TTTrackMCTruth");
  HQDelim_ = conf_.getParameter<int>("HQDelim");
  verbosePlots_ = conf_.getUntrackedParameter<bool>("verbosePlots",true);
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
  iEvent.getByLabel( tagTTTracks_, PixelDigiTTTrackHandle );
  //iEvent.getByLabel( "TTTracksFromPixelDigis", "Level1TTTracks",PixelDigiTTTrackHandle );
  
  /// Track Trigger MC Truth
  edm::Handle< TTClusterAssociationMap< Ref_PixelDigi_ > > MCTruthTTClusterHandle;
  edm::Handle< TTStubAssociationMap< Ref_PixelDigi_ > >    MCTruthTTStubHandle;
  edm::Handle< TTTrackAssociationMap< Ref_PixelDigi_ > >   MCTruthTTTrackHandle;
  iEvent.getByLabel( tagTTTrackMCTruth_, MCTruthTTTrackHandle );
  //iEvent.getByLabel( "TTTrackAssociatorFromPixelDigis", "Level1TTTracks", MCTruthTTTrackHandle );
	
	
  /// TrackingParticles
  edm::Handle< std::vector< TrackingParticle > > TrackingParticleHandle;
  iEvent.getByLabel( "mix", "MergedTrackTruth", TrackingParticleHandle );
  edm::Handle< std::vector< TrackingVertex > > TrackingVertexHandle;
  iEvent.getByLabel( "mix", "MergedTrackTruth", TrackingVertexHandle );
  
  
  unsigned int numHQTracks = 0;
  unsigned int numLQTracks = 0;
  
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
      
      if ( nStubs >= HQDelim_ )
      {
        Track_HQ_Pt->Fill( trackPt );
        Track_HQ_Eta->Fill( trackEta );
        Track_HQ_Phi->Fill( trackPhi );
        Track_HQ_VtxZ0->Fill( trackVtxZ0 );
        Track_HQ_Chi2->Fill( trackChi2 );
        Track_HQ_Chi2Red->Fill( trackChi2R );
        
        numHQTracks++;
        Track_HQ_Pt_TPart_Pt->Fill( tpPt, trackPt );
        Track_HQ_PtRes_TPart_Eta->Fill( tpEta, trackPt - tpPt );
        Track_HQ_InvPt_TPart_InvPt->Fill( 1./tpPt, 1./trackPt );
        Track_HQ_InvPtRes_TPart_Eta->Fill( tpEta, 1./trackPt - 1./tpPt );
        Track_HQ_Phi_TPart_Phi->Fill( tpPhi, trackPhi );
        Track_HQ_PhiRes_TPart_Eta->Fill( tpEta, trackPhi - tpPhi );
        Track_HQ_Eta_TPart_Eta->Fill( tpEta, trackEta );
        Track_HQ_EtaRes_TPart_Eta->Fill( tpEta, trackEta - tpEta );
        Track_HQ_VtxZ0_TPart_VtxZ0->Fill( tpVtxZ0, trackVtxZ0 );
        Track_HQ_VtxZ0Res_TPart_Eta->Fill( tpEta, trackVtxZ0 - tpVtxZ0 );
        Track_HQ_Chi2_NStubs->Fill( nStubs, trackChi2 );
        Track_HQ_Chi2Red_NStubs->Fill( nStubs, trackChi2R );
      }
      else
      {
        Track_LQ_Pt->Fill( trackPt );
        Track_LQ_Eta->Fill( trackEta );
        Track_LQ_Phi->Fill( trackPhi );
        Track_LQ_VtxZ0->Fill( trackVtxZ0 );
        Track_LQ_Chi2->Fill( trackChi2 );
        Track_LQ_Chi2Red->Fill( trackChi2R );
        
        numLQTracks++;
        Track_LQ_Pt_TPart_Pt->Fill( tpPt, trackPt );
        Track_LQ_PtRes_TPart_Eta->Fill( tpEta, trackPt - tpPt );
        Track_LQ_InvPt_TPart_InvPt->Fill( 1./tpPt, 1./trackPt );
        Track_LQ_InvPtRes_TPart_Eta->Fill( tpEta, 1./trackPt - 1./tpPt );
        Track_LQ_Phi_TPart_Phi->Fill( tpPhi, trackPhi );
        Track_LQ_PhiRes_TPart_Eta->Fill( tpEta, trackPhi - tpPhi );
        Track_LQ_Eta_TPart_Eta->Fill( tpEta, trackEta );
        Track_LQ_EtaRes_TPart_Eta->Fill( tpEta, trackEta - tpEta );
        Track_LQ_VtxZ0_TPart_VtxZ0->Fill( tpVtxZ0, trackVtxZ0 );
        Track_LQ_VtxZ0Res_TPart_Eta->Fill( tpEta, trackVtxZ0 - tpVtxZ0 );
        Track_LQ_Chi2_NStubs->Fill( nStubs, trackChi2 );
        Track_LQ_Chi2Red_NStubs->Fill( nStubs, trackChi2R );     
      }
    } /// End of loop over TTTracks
  }
  
  Track_LQ_N->Fill( numLQTracks );
  Track_HQ_N->Fill( numHQTracks );
	
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
  Track_NStubs->setAxisTitle("# Level-1 Stubs", 1);
  Track_NStubs->setAxisTitle("# Genuine Level-1 Tracks", 2);
  
  
  
  
  /// Plots where all TTTracks are made from up to X TTStubs
  dqmStore_->setCurrentFolder(topFolderName_+"/Tracks/LQ");
  
  // Number of TTTracks
  edm::ParameterSet psTrack_N =  conf_.getParameter<edm::ParameterSet>("TH1TTTrack_N");
  HistoName = "Track_LQ_N";
  Track_LQ_N = dqmStore_->book1D(HistoName, HistoName,
      psTrack_N.getParameter<int32_t>("Nbinsx"),
      psTrack_N.getParameter<double>("xmin"),
      psTrack_N.getParameter<double>("xmax"));
  Track_LQ_N->setAxisTitle("# Genuine low-quality Level-1 Tracks", 1);
  Track_LQ_N->setAxisTitle("# Events", 2);
  
  // Pt
  edm::ParameterSet psTrack_Pt =  conf_.getParameter<edm::ParameterSet>("TH1TTTrack_Pt");
  HistoName = "Track_LQ_Pt";
  Track_LQ_Pt = dqmStore_->book1D(HistoName, HistoName,
      psTrack_Pt.getParameter<int32_t>("Nbinsx"),
      psTrack_Pt.getParameter<double>("xmin"),
      psTrack_Pt.getParameter<double>("xmax"));
  Track_LQ_Pt->setAxisTitle("Level-1 Track Pt", 1);
  Track_LQ_Pt->setAxisTitle("# Level-1 Tracks", 2);
  
  // Eta
  edm::ParameterSet psTrack_Eta =  conf_.getParameter<edm::ParameterSet>("TH1TTTrack_Eta");
  HistoName = "Track_LQ_Eta";
  Track_LQ_Eta = dqmStore_->book1D(HistoName, HistoName,
      psTrack_Eta.getParameter<int32_t>("Nbinsx"),
      psTrack_Eta.getParameter<double>("xmin"),
      psTrack_Eta.getParameter<double>("xmax"));
  Track_LQ_Eta->setAxisTitle("Level-1 Track Eta", 1);
  Track_LQ_Eta->setAxisTitle("# Level-1 Tracks", 2);
  
  // Phi
  edm::ParameterSet psTrack_Phi =  conf_.getParameter<edm::ParameterSet>("TH1TTTrack_Phi");
  HistoName = "Track_LQ_Phi";
  Track_LQ_Phi = dqmStore_->book1D(HistoName, HistoName,
      psTrack_Phi.getParameter<int32_t>("Nbinsx"),
      psTrack_Phi.getParameter<double>("xmin"),
      psTrack_Phi.getParameter<double>("xmax"));
  Track_LQ_Phi->setAxisTitle("Level-1 Track Phi", 1);
  Track_LQ_Phi->setAxisTitle("# Level-1 Tracks", 2);
  
  // VtxZ0
  edm::ParameterSet psTrack_VtxZ0 =  conf_.getParameter<edm::ParameterSet>("TH1TTTrack_VtxZ0");
  HistoName = "Track_LQ_VtxZ0";
  Track_LQ_VtxZ0 = dqmStore_->book1D(HistoName, HistoName,
      psTrack_VtxZ0.getParameter<int32_t>("Nbinsx"),
      psTrack_VtxZ0.getParameter<double>("xmin"),
      psTrack_VtxZ0.getParameter<double>("xmax"));
  Track_LQ_VtxZ0->setAxisTitle("Level-1 Track Vertex Position in z", 1);
  Track_LQ_VtxZ0->setAxisTitle("# Level-1 Tracks", 2);
  
  // TTTrack Chi2
  edm::ParameterSet psTrack_Chi2 =  conf_.getParameter<edm::ParameterSet>("TH1TTTrack_Chi2");
  HistoName = "Track_LQ_Chi2";
  Track_LQ_Chi2 = dqmStore_->book1D(HistoName, HistoName,
      psTrack_Chi2.getParameter<int32_t>("Nbinsx"),
      psTrack_Chi2.getParameter<double>("xmin"),
      psTrack_Chi2.getParameter<double>("xmax"));
  Track_LQ_Chi2->setAxisTitle("Level-1 Track #chi^{2}", 1);
  Track_LQ_Chi2->setAxisTitle("# Level-1 Tracks", 2);
  
  // TTTrack Chi2/ndf
  edm::ParameterSet psTrack_Chi2Red =  conf_.getParameter<edm::ParameterSet>("TH1TTTrack_Chi2Red");
  HistoName = "Track_LQ_Chi2Red";
  Track_LQ_Chi2Red = dqmStore_->book1D(HistoName, HistoName,
      psTrack_Chi2Red.getParameter<int32_t>("Nbinsx"),
      psTrack_Chi2Red.getParameter<double>("xmin"),
      psTrack_Chi2Red.getParameter<double>("xmax"));
  Track_LQ_Chi2Red->setAxisTitle("Level-1 Track #chi^{2}/ndf", 1);
  Track_LQ_Chi2Red->setAxisTitle("# Level-1 Tracks", 2);
  
  // TTTrack Chi2 vs Nstubs
  edm::ParameterSet psTrack_Chi2_NStubs =  conf_.getParameter<edm::ParameterSet>("TH2TTTrack_Chi2");
  HistoName = "Track_LQ_Chi2_NStubs";
  Track_LQ_Chi2_NStubs = dqmStore_->book2D(HistoName, HistoName,
      psTrack_Chi2_NStubs.getParameter<int32_t>("Nbinsx"),
      psTrack_Chi2_NStubs.getParameter<double>("xmin"),
      psTrack_Chi2_NStubs.getParameter<double>("xmax"),
      psTrack_Chi2_NStubs.getParameter<int32_t>("Nbinsy"),
      psTrack_Chi2_NStubs.getParameter<double>("ymin"),
      psTrack_Chi2_NStubs.getParameter<double>("ymax"));
  Track_LQ_Chi2_NStubs->setAxisTitle("# Level-1 Stubs", 1);
  Track_LQ_Chi2_NStubs->setAxisTitle("Level-1 Track #chi^{2}", 2);
  
  // TTTrack Chi2/ndf vs Nstubs
  edm::ParameterSet psTrack_Chi2Red_NStubs =  conf_.getParameter<edm::ParameterSet>("TH2TTTrack_Chi2Red");
  HistoName = "Track_LQ_Chi2Red_NStubs";
  Track_LQ_Chi2Red_NStubs = dqmStore_->book2D(HistoName, HistoName,
      psTrack_Chi2Red_NStubs.getParameter<int32_t>("Nbinsx"),
      psTrack_Chi2Red_NStubs.getParameter<double>("xmin"),
      psTrack_Chi2Red_NStubs.getParameter<double>("xmax"),
      psTrack_Chi2Red_NStubs.getParameter<int32_t>("Nbinsy"),
      psTrack_Chi2Red_NStubs.getParameter<double>("ymin"),
      psTrack_Chi2Red_NStubs.getParameter<double>("ymax"));
  Track_LQ_Chi2Red_NStubs->setAxisTitle("# Level-1 Stubs", 1);
  Track_LQ_Chi2Red_NStubs->setAxisTitle("Level-1 Track #chi^{2}/ndf", 2);
  
  
  
  
  /// Plots where all TTTracks are made from at least 3 TTStubs
  dqmStore_->setCurrentFolder(topFolderName_+"/Tracks/HQ/");
  
  // Number of TTTracks
  HistoName = "Track_HQ_N";
  Track_HQ_N = dqmStore_->book1D(HistoName, HistoName,
      psTrack_N.getParameter<int32_t>("Nbinsx"),
      psTrack_N.getParameter<double>("xmin"),
      psTrack_N.getParameter<double>("xmax"));
  Track_HQ_N->setAxisTitle("# Genuine high-quality Level-1 Tracks", 1);
  Track_HQ_N->setAxisTitle("# Events", 2);
  
  // Pt
  HistoName = "Track_HQ_Pt";
  Track_HQ_Pt = dqmStore_->book1D(HistoName, HistoName,
      psTrack_Pt.getParameter<int32_t>("Nbinsx"),
      psTrack_Pt.getParameter<double>("xmin"),
      psTrack_Pt.getParameter<double>("xmax"));
  Track_HQ_Pt->setAxisTitle("Level-1 Track Pt", 1);
  Track_HQ_Pt->setAxisTitle("# Level-1 Tracks", 2);
  
  // Eta
  HistoName = "Track_HQ_Eta";
  Track_HQ_Eta = dqmStore_->book1D(HistoName, HistoName,
      psTrack_Eta.getParameter<int32_t>("Nbinsx"),
      psTrack_Eta.getParameter<double>("xmin"),
      psTrack_Eta.getParameter<double>("xmax"));
  Track_HQ_Eta->setAxisTitle("Level-1 Track Eta", 1);
  Track_HQ_Eta->setAxisTitle("# Level-1 Tracks", 2);
  
  // Phi
  HistoName = "Track_HQ_Phi";
  Track_HQ_Phi = dqmStore_->book1D(HistoName, HistoName,
      psTrack_Phi.getParameter<int32_t>("Nbinsx"),
      psTrack_Phi.getParameter<double>("xmin"),
      psTrack_Phi.getParameter<double>("xmax"));
  Track_HQ_Phi->setAxisTitle("Level-1 Track Phi", 1);
  Track_HQ_Phi->setAxisTitle("# Level-1 Tracks", 2);
  
  // VtxZ0
  HistoName = "Track_HQ_VtxZ0";
  Track_HQ_VtxZ0 = dqmStore_->book1D(HistoName, HistoName,
      psTrack_VtxZ0.getParameter<int32_t>("Nbinsx"),
      psTrack_VtxZ0.getParameter<double>("xmin"),
      psTrack_VtxZ0.getParameter<double>("xmax"));
  Track_HQ_VtxZ0->setAxisTitle("Level-1 Track Vertex Position in z", 1);
  Track_HQ_VtxZ0->setAxisTitle("# Level-1 Tracks", 2);
  
  // TTTrack Chi2
  HistoName = "Track_HQ_Chi2";
  Track_HQ_Chi2 = dqmStore_->book1D(HistoName, HistoName,
      psTrack_Chi2.getParameter<int32_t>("Nbinsx"),
      psTrack_Chi2.getParameter<double>("xmin"),
      psTrack_Chi2.getParameter<double>("xmax"));
  Track_HQ_Chi2->setAxisTitle("Level-1 Track #chi^{2}", 1);
  Track_HQ_Chi2->setAxisTitle("# Level-1 Tracks", 2);
  
  // TTTrack Chi2/ndf
  HistoName = "Track_HQ_Chi2Red";
  Track_HQ_Chi2Red = dqmStore_->book1D(HistoName, HistoName,
      psTrack_Chi2Red.getParameter<int32_t>("Nbinsx"),
      psTrack_Chi2Red.getParameter<double>("xmin"),
      psTrack_Chi2Red.getParameter<double>("xmax"));
  Track_HQ_Chi2Red->setAxisTitle("Level-1 Track #chi^{2}/ndf", 1);
  Track_HQ_Chi2Red->setAxisTitle("# Level-1 Tracks", 2);
  
  // TTTrack Chi2 vs Nstubs
  HistoName = "Track_HQ_Chi2_NStubs";
  Track_HQ_Chi2_NStubs = dqmStore_->book2D(HistoName, HistoName,
      psTrack_Chi2_NStubs.getParameter<int32_t>("Nbinsx"),
      psTrack_Chi2_NStubs.getParameter<double>("xmin"),
      psTrack_Chi2_NStubs.getParameter<double>("xmax"),
      psTrack_Chi2_NStubs.getParameter<int32_t>("Nbinsy"),
      psTrack_Chi2_NStubs.getParameter<double>("ymin"),
      psTrack_Chi2_NStubs.getParameter<double>("ymax"));
  Track_HQ_Chi2_NStubs->setAxisTitle("# Level-1 Stubs", 1);
  Track_HQ_Chi2_NStubs->setAxisTitle("Level-1 Track #chi^{2}", 2);
  
  // TTTrack Chi2/ndf vs Nstubs
  HistoName = "Track_HQ_Chi2Red_NStubs";
  Track_HQ_Chi2Red_NStubs = dqmStore_->book2D(HistoName, HistoName,
      psTrack_Chi2Red_NStubs.getParameter<int32_t>("Nbinsx"),
      psTrack_Chi2Red_NStubs.getParameter<double>("xmin"),
      psTrack_Chi2Red_NStubs.getParameter<double>("xmax"),
      psTrack_Chi2Red_NStubs.getParameter<int32_t>("Nbinsy"),
     	psTrack_Chi2Red_NStubs.getParameter<double>("ymin"),
      psTrack_Chi2Red_NStubs.getParameter<double>("ymax"));
  Track_HQ_Chi2Red_NStubs->setAxisTitle("# Level-1 Stubs", 1);
  Track_HQ_Chi2Red_NStubs->setAxisTitle("Level-1 Track #chi^{2}/ndf", 2);
  
  
  
  
  /// Track properties compared to TParticles
  dqmStore_->setCurrentFolder(topFolderName_+"/TTTrackVSTPart/LQ/");
  
  // Pt
  edm::ParameterSet psTrack_Sim_Pt =  conf_.getParameter<edm::ParameterSet>("TH2TTTrack_Sim_Pt");
  HistoName = "Track_LQ_Pt_TPart_Pt";
  Track_LQ_Pt_TPart_Pt = dqmStore_->book2D(HistoName, HistoName,
      psTrack_Sim_Pt.getParameter<int32_t>("Nbinsx"),
      psTrack_Sim_Pt.getParameter<double>("xmin"),
      psTrack_Sim_Pt.getParameter<double>("xmax"),
      psTrack_Sim_Pt.getParameter<int32_t>("Nbinsy"),
      psTrack_Sim_Pt.getParameter<double>("ymin"),
      psTrack_Sim_Pt.getParameter<double>("ymax"));
  Track_LQ_Pt_TPart_Pt->setAxisTitle("TPart Pt", 1);
  Track_LQ_Pt_TPart_Pt->setAxisTitle("TTTrack Pt", 2);
  
  // PtRes
  edm::ParameterSet psTrack_Sim_PtRes =  conf_.getParameter<edm::ParameterSet>("TH2TTTrack_Sim_PtRes");
  HistoName = "Track_LQ_PtRes_TPart_Eta";
  Track_LQ_PtRes_TPart_Eta = dqmStore_->book2D(HistoName, HistoName,
      psTrack_Sim_PtRes.getParameter<int32_t>("Nbinsx"),
      psTrack_Sim_PtRes.getParameter<double>("xmin"),
      psTrack_Sim_PtRes.getParameter<double>("xmax"),
      psTrack_Sim_PtRes.getParameter<int32_t>("Nbinsy"),
      psTrack_Sim_PtRes.getParameter<double>("ymin"),
      psTrack_Sim_PtRes.getParameter<double>("ymax"));
  Track_LQ_PtRes_TPart_Eta->setAxisTitle("TPart Eta", 1);
  Track_LQ_PtRes_TPart_Eta->setAxisTitle("TTTrack Pt - TPart Pt", 2);
  
  // InvPt
  edm::ParameterSet psTrack_Sim_InvPt =  conf_.getParameter<edm::ParameterSet>("TH2TTTrack_Sim_InvPt");
  HistoName = "Track_LQ_InvPt_TPart_InvPt";
  Track_LQ_InvPt_TPart_InvPt = dqmStore_->book2D(HistoName, HistoName,
      psTrack_Sim_InvPt.getParameter<int32_t>("Nbinsx"),
      psTrack_Sim_InvPt.getParameter<double>("xmin"),
      psTrack_Sim_InvPt.getParameter<double>("xmax"),
      psTrack_Sim_InvPt.getParameter<int32_t>("Nbinsy"),
      psTrack_Sim_InvPt.getParameter<double>("ymin"),
      psTrack_Sim_InvPt.getParameter<double>("ymax"));
  Track_LQ_InvPt_TPart_InvPt->setAxisTitle("TPart 1/Pt", 1);
  Track_LQ_InvPt_TPart_InvPt->setAxisTitle("TTTrack 1/Pt", 2);
  
  // InvPtRes
  edm::ParameterSet psTrack_Sim_InvPtRes =  conf_.getParameter<edm::ParameterSet>("TH2TTTrack_Sim_InvPtRes");
  HistoName = "Track_LQ_InvPtRes_TPart_Eta";
  Track_LQ_InvPtRes_TPart_Eta = dqmStore_->book2D(HistoName, HistoName,
      psTrack_Sim_InvPtRes.getParameter<int32_t>("Nbinsx"),
      psTrack_Sim_InvPtRes.getParameter<double>("xmin"),
      psTrack_Sim_InvPtRes.getParameter<double>("xmax"),
      psTrack_Sim_InvPtRes.getParameter<int32_t>("Nbinsy"),
      psTrack_Sim_InvPtRes.getParameter<double>("ymin"),
      psTrack_Sim_InvPtRes.getParameter<double>("ymax"));
  Track_LQ_InvPtRes_TPart_Eta->setAxisTitle("TPart Eta", 1);
  Track_LQ_InvPtRes_TPart_Eta->setAxisTitle("TTTrack 1/Pt - TPart 1/Pt", 2);
  
  // Phi
  edm::ParameterSet psTrack_Sim_Phi =  conf_.getParameter<edm::ParameterSet>("TH2TTTrack_Sim_Phi");
  HistoName = "Track_LQ_Phi_TPart_Phi";
  Track_LQ_Phi_TPart_Phi = dqmStore_->book2D(HistoName, HistoName,
      psTrack_Sim_Phi.getParameter<int32_t>("Nbinsx"),
      psTrack_Sim_Phi.getParameter<double>("xmin"),
      psTrack_Sim_Phi.getParameter<double>("xmax"),
      psTrack_Sim_Phi.getParameter<int32_t>("Nbinsy"),
      psTrack_Sim_Phi.getParameter<double>("ymin"),
      psTrack_Sim_Phi.getParameter<double>("ymax"));
  Track_LQ_Phi_TPart_Phi->setAxisTitle("TPart Phi", 1);
  Track_LQ_Phi_TPart_Phi->setAxisTitle("TTTrack Phi", 2);
  
  // PhiRes
  edm::ParameterSet psTrack_Sim_PhiRes =  conf_.getParameter<edm::ParameterSet>("TH2TTTrack_Sim_PhiRes");
  HistoName = "Track_LQ_PhiRes_TPart_Eta";
  Track_LQ_PhiRes_TPart_Eta = dqmStore_->book2D(HistoName, HistoName,
      psTrack_Sim_PhiRes.getParameter<int32_t>("Nbinsx"),
      psTrack_Sim_PhiRes.getParameter<double>("xmin"),
      psTrack_Sim_PhiRes.getParameter<double>("xmax"),
      psTrack_Sim_PhiRes.getParameter<int32_t>("Nbinsy"),
      psTrack_Sim_PhiRes.getParameter<double>("ymin"),
      psTrack_Sim_PhiRes.getParameter<double>("ymax"));
  Track_LQ_PhiRes_TPart_Eta->setAxisTitle("TPart Eta", 1);
  Track_LQ_PhiRes_TPart_Eta->setAxisTitle("TTTrack Phi - TPart Phi", 2);
  
  // Eta
  edm::ParameterSet psTrack_Sim_Eta =  conf_.getParameter<edm::ParameterSet>("TH2TTTrack_Sim_Eta");
  HistoName = "Track_LQ_Eta_TPart_Eta";
  Track_LQ_Eta_TPart_Eta = dqmStore_->book2D(HistoName, HistoName,
      psTrack_Sim_Eta.getParameter<int32_t>("Nbinsx"),
      psTrack_Sim_Eta.getParameter<double>("xmin"),
      psTrack_Sim_Eta.getParameter<double>("xmax"),
      psTrack_Sim_Eta.getParameter<int32_t>("Nbinsy"),
      psTrack_Sim_Eta.getParameter<double>("ymin"),
      psTrack_Sim_Eta.getParameter<double>("ymax"));
  Track_LQ_Eta_TPart_Eta->setAxisTitle("TPart Eta", 1);
  Track_LQ_Eta_TPart_Eta->setAxisTitle("TTTrack Eta", 2);
  
  // EtaRes
  edm::ParameterSet psTrack_Sim_EtaRes =  conf_.getParameter<edm::ParameterSet>("TH2TTTrack_Sim_EtaRes");
  HistoName = "Track_LQ_EtaRes_TPart_Eta";
  Track_LQ_EtaRes_TPart_Eta = dqmStore_->book2D(HistoName, HistoName,
      psTrack_Sim_EtaRes.getParameter<int32_t>("Nbinsx"),
      psTrack_Sim_EtaRes.getParameter<double>("xmin"),
      psTrack_Sim_EtaRes.getParameter<double>("xmax"),
      psTrack_Sim_EtaRes.getParameter<int32_t>("Nbinsy"),
      psTrack_Sim_EtaRes.getParameter<double>("ymin"),
      psTrack_Sim_EtaRes.getParameter<double>("ymax"));
  Track_LQ_EtaRes_TPart_Eta->setAxisTitle("TPart Eta", 1);
  Track_LQ_EtaRes_TPart_Eta->setAxisTitle("TTTrack Eta - TPart Eta", 2);
  
  // Vertex position in z
  edm::ParameterSet psTrack_Sim_Vtx =  conf_.getParameter<edm::ParameterSet>("TH2TTTrack_Sim_Vtx");
  HistoName = "Track_LQ_VtxZ0_TPart_VtxZ0";
  Track_LQ_VtxZ0_TPart_VtxZ0 = dqmStore_->book2D(HistoName, HistoName,
      psTrack_Sim_Vtx.getParameter<int32_t>("Nbinsx"),
      psTrack_Sim_Vtx.getParameter<double>("xmin"),
      psTrack_Sim_Vtx.getParameter<double>("xmax"),
      psTrack_Sim_Vtx.getParameter<int32_t>("Nbinsy"),
      psTrack_Sim_Vtx.getParameter<double>("ymin"),
      psTrack_Sim_Vtx.getParameter<double>("ymax"));
  Track_LQ_VtxZ0_TPart_VtxZ0->setAxisTitle("TPart Vertex Position in z", 1);
  Track_LQ_VtxZ0_TPart_VtxZ0->setAxisTitle("TTTrack Vertex Position in z", 2);
  
  // Vertex position in z
  edm::ParameterSet psTrack_Sim_VtxRes =  conf_.getParameter<edm::ParameterSet>("TH2TTTrack_Sim_VtxRes");
  HistoName = "Track_LQ_VtxZ0Res_TPart_Eta";
  Track_LQ_VtxZ0Res_TPart_Eta = dqmStore_->book2D(HistoName, HistoName,
      psTrack_Sim_VtxRes.getParameter<int32_t>("Nbinsx"),
      psTrack_Sim_VtxRes.getParameter<double>("xmin"),
      psTrack_Sim_VtxRes.getParameter<double>("xmax"),
      psTrack_Sim_VtxRes.getParameter<int32_t>("Nbinsy"),
      psTrack_Sim_VtxRes.getParameter<double>("ymin"),
      psTrack_Sim_VtxRes.getParameter<double>("ymax"));
  Track_LQ_VtxZ0Res_TPart_Eta->setAxisTitle("TPart Eta", 1);
  Track_LQ_VtxZ0Res_TPart_Eta->setAxisTitle("TTTrack - TPart Vertex Position in z", 2);
  
  
  
  dqmStore_->setCurrentFolder(topFolderName_+"/TTTrackVSTPart/HQ");
	
	// Pt
  HistoName = "Track_HQ_Pt_TPart_Pt";
  Track_HQ_Pt_TPart_Pt = dqmStore_->book2D(HistoName, HistoName,
      psTrack_Sim_Pt.getParameter<int32_t>("Nbinsx"),
      psTrack_Sim_Pt.getParameter<double>("xmin"),
      psTrack_Sim_Pt.getParameter<double>("xmax"),
      psTrack_Sim_Pt.getParameter<int32_t>("Nbinsy"),
      psTrack_Sim_Pt.getParameter<double>("ymin"),
      psTrack_Sim_Pt.getParameter<double>("ymax"));
  Track_HQ_Pt_TPart_Pt->setAxisTitle("TPart Pt", 1);
  Track_HQ_Pt_TPart_Pt->setAxisTitle("TTTrack Pt", 2);
  
  // PtRes
  HistoName = "Track_HQ_PtRes_TPart_Eta";
  Track_HQ_PtRes_TPart_Eta = dqmStore_->book2D(HistoName, HistoName,
      psTrack_Sim_PtRes.getParameter<int32_t>("Nbinsx"),
      psTrack_Sim_PtRes.getParameter<double>("xmin"),
      psTrack_Sim_PtRes.getParameter<double>("xmax"),
      psTrack_Sim_PtRes.getParameter<int32_t>("Nbinsy"),
      psTrack_Sim_PtRes.getParameter<double>("ymin"),
      psTrack_Sim_PtRes.getParameter<double>("ymax"));
  Track_HQ_PtRes_TPart_Eta->setAxisTitle("TPart Eta", 1);
  Track_HQ_PtRes_TPart_Eta->setAxisTitle("TTTrack Pt - TPart Pt", 2);
  
  // InvPt
  HistoName = "Track_HQ_InvPt_TPart_InvPt";
  Track_HQ_InvPt_TPart_InvPt = dqmStore_->book2D(HistoName, HistoName,
      psTrack_Sim_InvPt.getParameter<int32_t>("Nbinsx"),
      psTrack_Sim_InvPt.getParameter<double>("xmin"),
      psTrack_Sim_InvPt.getParameter<double>("xmax"),
      psTrack_Sim_InvPt.getParameter<int32_t>("Nbinsy"),
      psTrack_Sim_InvPt.getParameter<double>("ymin"),
      psTrack_Sim_InvPt.getParameter<double>("ymax"));
  Track_HQ_InvPt_TPart_InvPt->setAxisTitle("TPart 1/Pt", 1);
  Track_HQ_InvPt_TPart_InvPt->setAxisTitle("TTTrack 1/Pt", 2);
  
  // InvPtRes
  HistoName = "Track_HQ_InvPtRes_TPart_Eta";
  Track_HQ_InvPtRes_TPart_Eta = dqmStore_->book2D(HistoName, HistoName,
      psTrack_Sim_InvPtRes.getParameter<int32_t>("Nbinsx"),
      psTrack_Sim_InvPtRes.getParameter<double>("xmin"),
      psTrack_Sim_InvPtRes.getParameter<double>("xmax"),
      psTrack_Sim_InvPtRes.getParameter<int32_t>("Nbinsy"),
      psTrack_Sim_InvPtRes.getParameter<double>("ymin"),
      psTrack_Sim_InvPtRes.getParameter<double>("ymax"));
  Track_HQ_InvPtRes_TPart_Eta->setAxisTitle("TPart Eta", 1);
  Track_HQ_InvPtRes_TPart_Eta->setAxisTitle("TTTrack 1/Pt - TPart 1/Pt", 2);
  
  // Phi
  HistoName = "Track_HQ_Phi_TPart_Phi";
  Track_HQ_Phi_TPart_Phi = dqmStore_->book2D(HistoName, HistoName,
      psTrack_Sim_Phi.getParameter<int32_t>("Nbinsx"),
      psTrack_Sim_Phi.getParameter<double>("xmin"),
      psTrack_Sim_Phi.getParameter<double>("xmax"),
      psTrack_Sim_Phi.getParameter<int32_t>("Nbinsy"),
      psTrack_Sim_Phi.getParameter<double>("ymin"),
      psTrack_Sim_Phi.getParameter<double>("ymax"));
  Track_HQ_Phi_TPart_Phi->setAxisTitle("TPart Phi", 1);
  Track_HQ_Phi_TPart_Phi->setAxisTitle("TTTrack Phi", 2);
  
  // PhiRes
  HistoName = "Track_HQ_PhiRes_TPart_Eta";
  Track_HQ_PhiRes_TPart_Eta = dqmStore_->book2D(HistoName, HistoName,
      psTrack_Sim_PhiRes.getParameter<int32_t>("Nbinsx"),
      psTrack_Sim_PhiRes.getParameter<double>("xmin"),
      psTrack_Sim_PhiRes.getParameter<double>("xmax"),
      psTrack_Sim_PhiRes.getParameter<int32_t>("Nbinsy"),
      psTrack_Sim_PhiRes.getParameter<double>("ymin"),
      psTrack_Sim_PhiRes.getParameter<double>("ymax"));
  Track_HQ_PhiRes_TPart_Eta->setAxisTitle("TPart Eta", 1);
  Track_HQ_PhiRes_TPart_Eta->setAxisTitle("TTTrack Phi - TPart Phi", 2);
  
  // Eta
  HistoName = "Track_HQ_Eta_TPart_Eta";
  Track_HQ_Eta_TPart_Eta = dqmStore_->book2D(HistoName, HistoName,
      psTrack_Sim_Eta.getParameter<int32_t>("Nbinsx"),
      psTrack_Sim_Eta.getParameter<double>("xmin"),
      psTrack_Sim_Eta.getParameter<double>("xmax"),
      psTrack_Sim_Eta.getParameter<int32_t>("Nbinsy"),
      psTrack_Sim_Eta.getParameter<double>("ymin"),
      psTrack_Sim_Eta.getParameter<double>("ymax"));
  Track_HQ_Eta_TPart_Eta->setAxisTitle("TPart Eta", 1);
  Track_HQ_Eta_TPart_Eta->setAxisTitle("TTTrack Eta", 2);
  
  // EtaRes
  HistoName = "Track_HQ_EtaRes_TPart_Eta";
  Track_HQ_EtaRes_TPart_Eta = dqmStore_->book2D(HistoName, HistoName,
      psTrack_Sim_EtaRes.getParameter<int32_t>("Nbinsx"),
      psTrack_Sim_EtaRes.getParameter<double>("xmin"),
      psTrack_Sim_EtaRes.getParameter<double>("xmax"),
      psTrack_Sim_EtaRes.getParameter<int32_t>("Nbinsy"),
      psTrack_Sim_EtaRes.getParameter<double>("ymin"),
      psTrack_Sim_EtaRes.getParameter<double>("ymax"));
  Track_HQ_EtaRes_TPart_Eta->setAxisTitle("TPart Eta", 1);
  Track_HQ_EtaRes_TPart_Eta->setAxisTitle("TTTrack Eta - TPart Eta", 2);
  
  // Vertex position in z
  HistoName = "Track_HQ_VtxZ0_TPart_VtxZ0";
  Track_HQ_VtxZ0_TPart_VtxZ0 = dqmStore_->book2D(HistoName, HistoName,
      psTrack_Sim_Vtx.getParameter<int32_t>("Nbinsx"),
      psTrack_Sim_Vtx.getParameter<double>("xmin"),
      psTrack_Sim_Vtx.getParameter<double>("xmax"),
      psTrack_Sim_Vtx.getParameter<int32_t>("Nbinsy"),
      psTrack_Sim_Vtx.getParameter<double>("ymin"),
      psTrack_Sim_Vtx.getParameter<double>("ymax"));
  Track_HQ_VtxZ0_TPart_VtxZ0->setAxisTitle("TPart Vertex Position in z", 1);
  Track_HQ_VtxZ0_TPart_VtxZ0->setAxisTitle("TTTrack Vertex Position in z", 2);
  
  // Vertex position in z
  HistoName = "Track_HQ_VtxZ0Res_TPart_Eta";
  Track_HQ_VtxZ0Res_TPart_Eta = dqmStore_->book2D(HistoName, HistoName,
      psTrack_Sim_VtxRes.getParameter<int32_t>("Nbinsx"),
      psTrack_Sim_VtxRes.getParameter<double>("xmin"),
      psTrack_Sim_VtxRes.getParameter<double>("xmax"),
      psTrack_Sim_VtxRes.getParameter<int32_t>("Nbinsy"),
      psTrack_Sim_VtxRes.getParameter<double>("ymin"),
      psTrack_Sim_VtxRes.getParameter<double>("ymax"));
  Track_HQ_VtxZ0Res_TPart_Eta->setAxisTitle("TPart Eta", 1);
  Track_HQ_VtxZ0Res_TPart_Eta->setAxisTitle("TTTrack - TPart Vertex Position in z", 2);
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
