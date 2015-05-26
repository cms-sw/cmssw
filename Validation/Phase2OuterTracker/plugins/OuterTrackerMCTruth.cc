// -*- C++ -*-
//
// Package:    Phase2OuterTracker
// Class:      Phase2OuterTracker
// 
/**\class Phase2OuterTracker OuterTrackerMCTruth.cc Validation/Phase2OuterTracker/plugins/OuterTrackerMCTruth.cc

 Description: [one line class summary]

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  Lieselotte Moreels
//         Created:  Fri, 13 Jun 2014 09:57:34 GMT
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
#include "Validation/Phase2OuterTracker/interface/OuterTrackerMCTruth.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "DataFormats/SiPixelCluster/interface/SiPixelCluster.h"

// For TrackingParticles
#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticle.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingVertex.h"
#include "SimTracker/TrackTriggerAssociation/interface/TTClusterAssociationMap.h"
#include "SimTracker/TrackTriggerAssociation/interface/TTStubAssociationMap.h"
#include "SimTracker/TrackTriggerAssociation/interface/TTTrackAssociationMap.h"
#include "DataFormats/L1TrackTrigger/interface/TTTypes.h"
#include "DataFormats/L1TrackTrigger/interface/TTCluster.h"

#include "Geometry/TrackerGeometryBuilder/interface/StackedTrackerGeometry.h"
#include "Geometry/Records/interface/StackedTrackerGeometryRecord.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"



#include "TMath.h"
#include <iostream>

//
// constructors and destructor
//
OuterTrackerMCTruth::OuterTrackerMCTruth(const edm::ParameterSet& iConfig)
: dqmStore_(edm::Service<DQMStore>().operator->()), conf_(iConfig)

{
  topFolderName_ = conf_.getParameter<std::string>("TopFolderName");
  tagTTClusters_ = conf_.getParameter< edm::InputTag >("TTClusters");
  tagTTClusterMCTruth_ = conf_.getParameter< edm::InputTag >("TTClusterMCTruth");
  tagTTStubs_ = conf_.getParameter< edm::InputTag >("TTStubs");
  tagTTStubMCTruth_ = conf_.getParameter< edm::InputTag >("TTStubMCTruth");
  tagTTTracks_ = conf_.getParameter< edm::InputTag >("TTTracks");
  tagTTTrackMCTruth_ = conf_.getParameter< edm::InputTag >("TTTrackMCTruth");
  HQDelim_ = conf_.getParameter<int>("HQDelim");
  verbosePlots_ = conf_.getUntrackedParameter<bool>("verbosePlots",false);
}


OuterTrackerMCTruth::~OuterTrackerMCTruth()
{
	
  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)
	
}


//
// member functions
//

// ------------ method called for each event  ------------
void
OuterTrackerMCTruth::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  
  /// Geometry handles etc
  edm::ESHandle< TrackerGeometry >                GeometryHandle;
  edm::ESHandle< StackedTrackerGeometry >         StackedGeometryHandle;
  const StackedTrackerGeometry*                   theStackedGeometry;
  StackedTrackerGeometry::StackContainerIterator  StackedTrackerIterator;
	
  /// Geometry setup
  /// Set pointers to Geometry
  iSetup.get< TrackerDigiGeometryRecord >().get(GeometryHandle);
  /// Set pointers to Stacked Modules
  iSetup.get< StackedTrackerGeometryRecord >().get(StackedGeometryHandle);
  theStackedGeometry = StackedGeometryHandle.product(); /// Note this is different from the "global" geometry
  
  /// Magnetic Field
  edm::ESHandle< MagneticField > magneticFieldHandle;
  iSetup.get< IdealMagneticFieldRecord >().get(magneticFieldHandle);
  const MagneticField* theMagneticField = magneticFieldHandle.product();
  double mMagneticFieldStrength = theMagneticField->inTesla(GlobalPoint(0,0,0)).z();
  
  
  /// TrackingParticles
  edm::Handle< std::vector< TrackingParticle > > TrackingParticleHandle;
  iEvent.getByLabel( "mix", "MergedTrackTruth", TrackingParticleHandle );
  
  /// Track Trigger
  edm::Handle< edmNew::DetSetVector< TTCluster< Ref_PixelDigi_ > > > PixelDigiTTClusterHandle;
  edm::Handle< edmNew::DetSetVector< TTStub< Ref_PixelDigi_ > > >    PixelDigiTTStubHandle;
  edm::Handle< std::vector< TTTrack< Ref_PixelDigi_ > > >            PixelDigiTTTrackHandle;
  iEvent.getByLabel( tagTTClusters_, PixelDigiTTClusterHandle );
  iEvent.getByLabel( tagTTStubs_, PixelDigiTTStubHandle );
  iEvent.getByLabel( tagTTTracks_, PixelDigiTTTrackHandle );
  
  /// Track Trigger MC Truth
  edm::Handle< TTClusterAssociationMap< Ref_PixelDigi_ > > MCTruthTTClusterHandle;
  edm::Handle< TTStubAssociationMap< Ref_PixelDigi_ > >    MCTruthTTStubHandle;
  edm::Handle< TTTrackAssociationMap< Ref_PixelDigi_ > >   MCTruthTTTrackHandle;
  iEvent.getByLabel( tagTTClusterMCTruth_, MCTruthTTClusterHandle );
  iEvent.getByLabel( tagTTStubMCTruth_, MCTruthTTStubHandle );
  iEvent.getByLabel( tagTTTrackMCTruth_, MCTruthTTTrackHandle );
  
  
  /// Go on only if there are TrackingParticles
  if( TrackingParticleHandle->size() > 0)
  {
    /// Loop over the TrackingParticles
    unsigned int tpCnt = 0;
    std::vector< TrackingParticle >::const_iterator iterTP;
    for(iterTP = TrackingParticleHandle->begin(); iterTP !=	TrackingParticleHandle->end(); ++iterTP)
    {
      /// Make the pointer
      edm::Ptr<TrackingParticle> tempTPPtr( TrackingParticleHandle, tpCnt++ );
      
      /// Get the corresponding vertex
      /// Assume perfectly round beamspot
      /// Correct and get the correct TrackingParticle Vertex position wrt beam centre
      if ( tempTPPtr->vertex().rho() >= 2.0 )
        continue;
      
      /// Check beamspot and correction
      SimVtx_XY->Fill( tempTPPtr->vertex().x(), tempTPPtr->vertex().y() );
      SimVtx_RZ->Fill( tempTPPtr->vertex().z(), tempTPPtr->vertex().rho() );
      
      /// Here we have only tracks form primary vertices
      /// Check Pt spectrum and pseudorapidity for over-threshold tracks
      TPart_Pt->Fill( tempTPPtr->p4().pt() );
      if ( tempTPPtr->p4().pt() > 10.0 )
      {
        TPart_Eta_Pt10->Fill( tempTPPtr->momentum().eta() );
        TPart_Phi_Pt10->Fill( tempTPPtr->momentum().phi() > M_PI ?
                              tempTPPtr->momentum().phi() - 2*M_PI :
                              tempTPPtr->momentum().phi() );
      }
      
      
      /// Check if this TP produced any clusters
      std::vector< edm::Ref< edmNew::DetSetVector< TTCluster< Ref_PixelDigi_ > >, TTCluster< Ref_PixelDigi_ > > > theseClusters = MCTruthTTClusterHandle->findTTClusterRefs( tempTPPtr );
      
      if ( theseClusters.size() > 0 )
      {
        if ( verbosePlots_ )
        {
          TPart_Cluster_Pt->Fill (tempTPPtr->p4().pt() );
          if ( tempTPPtr->p4().pt() > 10.0 )
          {
            TPart_Cluster_Phi_Pt10->Fill( tempTPPtr->momentum().phi() );
            TPart_Cluster_Eta_Pt10->Fill( tempTPPtr->momentum().eta() );
          }
        } /// End verbosePlots
        
        bool normIClu = false;
        bool normOClu = false;
        
        /// Loop over the Clusters
        for ( unsigned int jc = 0; jc < theseClusters.size(); jc++ )
        {
          /// Check if it is good
          bool genuineClu = MCTruthTTClusterHandle->isGenuine( theseClusters.at(jc) );
          if ( !genuineClu )
            continue;
          
          unsigned int stackMember = theseClusters.at(jc)->getStackMember();
          unsigned int clusterWidth = theseClusters.at(jc)->findWidth();
          
          if ( stackMember == 0 )
          {
            if ( verbosePlots_ && normIClu == false )
            {
              TPart_Eta_INormalization->Fill( tempTPPtr->momentum().eta() );
              normIClu = true;
            }
            
            if ( clusterWidth == 1 )
            {
              TPart_Eta_ICW_1->Fill( tempTPPtr->momentum().eta() );
            }
            else if ( clusterWidth == 2 )
            {
              TPart_Eta_ICW_2->Fill( tempTPPtr->momentum().eta() );
            }
            else
            {
              TPart_Eta_ICW_3->Fill( tempTPPtr->momentum().eta() );
            }
          }
          if ( stackMember == 1 )
          {
            if ( verbosePlots_ && normOClu == false )
            {
              TPart_Eta_ONormalization->Fill( tempTPPtr->momentum().eta() );
              normOClu = true;
            }
            
            if ( clusterWidth == 1 )
            {
              TPart_Eta_OCW_1->Fill( tempTPPtr->momentum().eta() );
            }
            else if ( clusterWidth == 2 )
            {
              TPart_Eta_OCW_2->Fill( tempTPPtr->momentum().eta() );
            }
            else
            {
              TPart_Eta_OCW_3->Fill( tempTPPtr->momentum().eta() );
            }
          }
        }
      }
      
      
      /// Check if the TP produced any stubs
      std::vector< edm::Ref< edmNew::DetSetVector< TTStub< Ref_PixelDigi_ > >, TTStub< Ref_PixelDigi_ > > > theseStubs = MCTruthTTStubHandle->findTTStubRefs( tempTPPtr );
      
      if ( theseStubs.size() > 0 )
      {
        if ( verbosePlots_ )
        {
          TPart_Stub_Pt->Fill( tempTPPtr->p4().pt() );
          if ( tempTPPtr->p4().pt() > 10.0 )
          {
            TPart_Stub_Phi_Pt10->Fill( tempTPPtr->momentum().phi() );
            TPart_Stub_Eta_Pt10->Fill( tempTPPtr->momentum().eta() );
          }
        } /// End verbosePlots
        
        if ( tempTPPtr->p4().pt() <= 10 )
        continue;
        
        bool normStub = false;
        
        /// Loop over the Stubs
        for ( unsigned int js = 0; js < theseStubs.size(); js++ )
        {
          /// Check if it is good
          bool genuineStub = MCTruthTTStubHandle->isGenuine( theseStubs.at(js) );
          if ( !genuineStub )
            continue;
          
          if ( verbosePlots_ && normStub == false )
          {
            TPart_Stub_Eta_Pt10_Normalization->Fill( tempTPPtr->momentum().eta() );
            normStub = true;
          }
          /// Classify the stub
          StackedTrackerDetId stDetId( theseStubs.at(js)->getDetId() );
          /// Check if there are PS modules in seed or candidate
          const GeomDetUnit* det0 = theStackedGeometry->idToDetUnit( stDetId, 0 );
          const GeomDetUnit* det1 = theStackedGeometry->idToDetUnit( stDetId, 1 );
          /// Find pixel pitch and topology related information
          const PixelGeomDetUnit* pix0 = dynamic_cast< const PixelGeomDetUnit* >( det0 );
          const PixelGeomDetUnit* pix1 = dynamic_cast< const PixelGeomDetUnit* >( det1 );
          const PixelTopology* top0 = dynamic_cast< const PixelTopology* >( &(pix0->specificTopology()) );
          const PixelTopology* top1 = dynamic_cast< const PixelTopology* >( &(pix1->specificTopology()) );
          int cols0 = top0->ncolumns();
          int cols1 = top1->ncolumns();
          int ratio = cols0/cols1; /// This assumes the ratio is integer!
          
          if ( ratio == 1 ) /// 2S Modules
          {
            TPart_Stub_Eta_Pt10_Num2S->Fill( tempTPPtr->momentum().eta() );
          }
          else /// PS
          {
            TPart_Stub_Eta_Pt10_NumPS->Fill( tempTPPtr->momentum().eta() );
          }
        } /// End of loop over the Stubs generated by this TrackingParticle
      }
      
      
      if ( verbosePlots_ )
      {
        /// Check if the TP produced any tracks
        std::vector< edm::Ptr< TTTrack< Ref_PixelDigi_ > > > theseTracks =  MCTruthTTTrackHandle->findTTTrackPtrs( tempTPPtr );

        if ( theseTracks.size() > 0 )
        {
          /// Distinguish between low-quality (< HQDelim_ stubs) and high-quality tracks (at least HQDelim_ stubs)
          bool foundLQ_ = false;
          bool foundHQ_ = false;

          for ( unsigned int jt = 0; jt < theseTracks.size(); jt++ )
          {
            if ( foundLQ_ && foundHQ_ )
            {
              jt = theseTracks.size();
              continue;
            }

            if ( theseTracks.at(jt)->getStubRefs().size() == (HQDelim_-1) )
            {
              foundLQ_ = true;
            }
            else if ( theseTracks.at(jt)->getStubRefs().size() >= HQDelim_ )
            {
              foundHQ_ = true;
            }
          } /// End theseTracks

          if ( foundLQ_ )
          {
            TPart_Track_LQ_Pt->Fill( tempTPPtr->p4().pt() );
            if ( tempTPPtr->p4().pt() > 10.0 )
            {
              TPart_Track_LQ_Phi_Pt10->Fill( tempTPPtr->momentum().phi() );
              TPart_Track_LQ_Eta_Pt10->Fill( tempTPPtr->momentum().eta() );
            }
          }

          if ( foundHQ_ )
          {
            TPart_Track_HQ_Pt->Fill( tempTPPtr->p4().pt() );
            if ( tempTPPtr->p4().pt() > 10.0 )
            {
              TPart_Track_HQ_Phi_Pt10->Fill( tempTPPtr->momentum().phi() );
              TPart_Track_HQ_Eta_Pt10->Fill( tempTPPtr->momentum().eta() );
            }
          }
        }
      } /// End verbosePlots
      
    } /// End TrackingParticles
  } /// End if there are TrackingParticles
  
  
  /// Loop over the input Clusters
  typename edmNew::DetSetVector< TTCluster< Ref_PixelDigi_ > >::const_iterator inputIter;
  typename edmNew::DetSet< TTCluster< Ref_PixelDigi_ > >::const_iterator contentIter;
  for ( inputIter = PixelDigiTTClusterHandle->begin();
       inputIter != PixelDigiTTClusterHandle->end();
       ++inputIter )
  {
    for ( contentIter = inputIter->begin();
         contentIter != inputIter->end();
         ++contentIter )
    {
      /// Make the reference to be put in the map
      edm::Ref< edmNew::DetSetVector< TTCluster< Ref_PixelDigi_ > >, TTCluster< Ref_PixelDigi_ > > tempCluRef = edmNew::makeRefTo( PixelDigiTTClusterHandle, contentIter );
      
      //StackedTrackerDetId detIdClu( tempCluRef->getDetId() );
      unsigned int memberClu = tempCluRef->getStackMember();
      bool genuineClu     = MCTruthTTClusterHandle->isGenuine( tempCluRef );
      int partClu         = 999999999;
      if ( genuineClu )
      {
        edm::Ptr< TrackingParticle > thisTP = MCTruthTTClusterHandle->findTrackingParticlePtr( tempCluRef );
        partClu = thisTP->pdgId();
      }
      
      Cluster_PID->Fill( partClu, memberClu );
      
    }
  } /// End of Loop over TTClusters
  
  
  /// Loop over the input Stubs
  typename edmNew::DetSetVector< TTStub< Ref_PixelDigi_ > >::const_iterator otherInputIter;
  typename edmNew::DetSet< TTStub< Ref_PixelDigi_ > >::const_iterator otherContentIter;
  for ( otherInputIter = PixelDigiTTStubHandle->begin();
       otherInputIter != PixelDigiTTStubHandle->end();
       ++otherInputIter )
  {
    for ( otherContentIter = otherInputIter->begin();
         otherContentIter != otherInputIter->end();
         ++otherContentIter )
    {
      /// Make the reference to be put in the map
      edm::Ref< edmNew::DetSetVector< TTStub< Ref_PixelDigi_ > >, TTStub< Ref_PixelDigi_ > > tempStubRef = edmNew::makeRefTo( PixelDigiTTStubHandle, otherContentIter );
      
      StackedTrackerDetId detIdStub( tempStubRef->getDetId() );
      
      bool genuineStub    = MCTruthTTStubHandle->isGenuine( tempStubRef );
      int partStub        = 999999999;
      
      if ( !genuineStub ) continue;
      
      edm::Ptr< TrackingParticle > tpPtr = MCTruthTTStubHandle->findTrackingParticlePtr( tempStubRef );
      partStub = tpPtr->pdgId();
      
      Stub_PID->Fill( partStub );
      
      
      if ( verbosePlots_ )
      {
        /// Get the corresponding vertex and reject the track
        /// if its vertex is outside the beampipe
        if ( tpPtr->vertex().rho() >= 2.0 )
          continue;

        /// Compare to TrackingParticle

        if ( tpPtr.isNull() ) continue; /// This prevents to fill the vector if the TrackingParticle is not found
        TrackingParticle thisTP = *tpPtr;

        double simPt  = thisTP.p4().pt();
        double simEta = thisTP.momentum().eta();
        double simPhi = thisTP.momentum().phi();
        double recPt  = theStackedGeometry->findRoughPt( mMagneticFieldStrength, &(*tempStubRef) );
        double recEta = theStackedGeometry->findGlobalDirection( &(*tempStubRef) ).eta();
        double recPhi = theStackedGeometry->findGlobalDirection( &(*tempStubRef) ).phi();

        if ( simPhi > M_PI )
        {
          simPhi -= 2*M_PI;
        }
        if ( recPhi > M_PI )
        {
          recPhi -= 2*M_PI;
        }

        double displStub    = tempStubRef->getTriggerDisplacement();
        double offsetStub   = tempStubRef->getTriggerOffset();

        if ( detIdStub.isBarrel() )
        {
          Stub_InvPt_TPart_InvPt_AllLayers->Fill( 1./simPt, 1./recPt ); 
          Stub_Pt_TPart_Pt_AllLayers->Fill( simPt, recPt );        
          Stub_Eta_TPart_Eta_AllLayers->Fill( simEta, recEta );
          Stub_Phi_TPart_Phi_AllLayers->Fill( simPhi, recPhi );

          Stub_InvPtRes_TPart_Eta_AllLayers->Fill(simEta, 1./recPt - 1./simPt);
          Stub_PtRes_TPart_Eta_AllLayers->Fill(simEta, recPt - simPt);
          Stub_EtaRes_TPart_Eta_AllLayers->Fill( simEta, recEta - simEta );
          Stub_PhiRes_TPart_Eta_AllLayers->Fill( simEta, recPhi - simPhi );

          Stub_W_TPart_Pt_AllLayers->Fill( simPt, displStub - offsetStub );
          Stub_W_TPart_InvPt_AllLayers->Fill( 1./simPt, displStub - offsetStub );
        }
        else if ( detIdStub.isEndcap() )
        {
          Stub_InvPt_TPart_InvPt_AllDisks->Fill( 1./simPt, 1./recPt );
          Stub_Pt_TPart_Pt_AllDisks->Fill( simPt, recPt ); 
          Stub_Eta_TPart_Eta_AllDisks->Fill( simEta, recEta );
          Stub_Phi_TPart_Phi_AllDisks->Fill( simPhi, recPhi );

          Stub_InvPtRes_TPart_Eta_AllDisks->Fill(simEta, 1./recPt - 1./simPt);
          Stub_PtRes_TPart_Eta_AllDisks->Fill(simEta, recPt - simPt);
          Stub_EtaRes_TPart_Eta_AllDisks->Fill( simEta, recEta - simEta );
          Stub_PhiRes_TPart_Eta_AllDisks->Fill( simEta, recPhi - simPhi );

          Stub_W_TPart_Pt_AllDisks->Fill( simPt, displStub - offsetStub );
          Stub_W_TPart_InvPt_AllDisks->Fill( 1./simPt, displStub - offsetStub );
        }
      } /// End verbosePlots
    }
  } /// End of loop over TTStubs
  
  
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
      //double trackTheta = tempTrackPtr->getMomentum().theta();
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
      
      double tpPt = tpPtr->p4().pt();
      double tpEta = tpPtr->momentum().eta();
//      double tpTheta = tpPtr->momentum().theta();
      double tpPhi = tpPtr->momentum().phi();
      double tpVtxZ0 = tpPtr->vertex().z();
      
      if ( nStubs >= HQDelim_ )
      {
        Track_HQ_Chi2_TPart_Eta->Fill( tpEta, trackChi2 );
        Track_HQ_Chi2Red_TPart_Eta->Fill( tpEta, trackChi2R );
        
        if ( verbosePlots_ )
        {
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
        } /// End verbosePlots
      }
      else
      {
        Track_LQ_Chi2_TPart_Eta->Fill( tpEta, trackChi2 );
        Track_LQ_Chi2Red_TPart_Eta->Fill( tpEta, trackChi2R ); 
        
        if ( verbosePlots_ )
        {
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
        } /// End verbosePlots
      }
    } /// End of loop over TTTracks
  }
  
}


// ------------ method called once each job just before starting event loop  ------------
void
OuterTrackerMCTruth::beginRun(const edm::Run& run, const edm::EventSetup& es)
{
  
  SiStripFolderOrganizer folder_organizer;
  folder_organizer.setSiStripFolderName(topFolderName_);
  folder_organizer.setSiStripFolder();
  std::string HistoName;
  
  dqmStore_->setCurrentFolder(topFolderName_+"/MCTruth/");
  
  /// TrackingParticle and TrackingVertex
  edm::ParameterSet psSimVtx_XY =  conf_.getParameter<edm::ParameterSet>("TH2SimVtx_XY");
  HistoName = "SimVtx_XY";
  SimVtx_XY = dqmStore_->book2D(HistoName, HistoName,
      psSimVtx_XY.getParameter<int32_t>("Nbinsx"),
      psSimVtx_XY.getParameter<double>("xmin"),
      psSimVtx_XY.getParameter<double>("xmax"),
      psSimVtx_XY.getParameter<int32_t>("Nbinsy"),
      psSimVtx_XY.getParameter<double>("ymin"),
      psSimVtx_XY.getParameter<double>("ymax"));
  SimVtx_XY->setAxisTitle("SimVtx position x [cm]", 1);
  SimVtx_XY->setAxisTitle("SimVtx position y [cm]", 2);
  
  edm::ParameterSet psSimVtx_RZ =  conf_.getParameter<edm::ParameterSet>("TH2SimVtx_RZ");
  HistoName = "SimVtx_RZ";
  SimVtx_RZ = dqmStore_->book2D(HistoName, HistoName,
      psSimVtx_RZ.getParameter<int32_t>("Nbinsx"),
      psSimVtx_RZ.getParameter<double>("xmin"),
      psSimVtx_RZ.getParameter<double>("xmax"),
      psSimVtx_RZ.getParameter<int32_t>("Nbinsy"),
      psSimVtx_RZ.getParameter<double>("ymin"),
      psSimVtx_RZ.getParameter<double>("ymax"));
  SimVtx_RZ->setAxisTitle("SimVtx position z [cm]", 1);
  SimVtx_RZ->setAxisTitle("SimVtx position #rho [cm]", 2);
  
  
  dqmStore_->setCurrentFolder(topFolderName_+"/MCTruth/Pt10");
  
  edm::ParameterSet psTPart_Pt =  conf_.getParameter<edm::ParameterSet>("TH1TPart_Pt");
  HistoName = "TPart_Pt";
  TPart_Pt = dqmStore_->book1D(HistoName, HistoName,
      psTPart_Pt.getParameter<int32_t>("Nbinsx"),
      psTPart_Pt.getParameter<double>("xmin"),
      psTPart_Pt.getParameter<double>("xmax"));
  TPart_Pt->setAxisTitle("TPart p_{T} [GeV]", 1);
  TPart_Pt->setAxisTitle("# TParticles", 2);
  
  edm::ParameterSet psTPart_Angle_Pt10 =  conf_.getParameter<edm::ParameterSet>("TH1TPart_Angle_Pt10");
  HistoName = "TPart_Eta_Pt10";
  TPart_Eta_Pt10 = dqmStore_->book1D(HistoName, HistoName,
      psTPart_Angle_Pt10.getParameter<int32_t>("Nbinsx"),
      psTPart_Angle_Pt10.getParameter<double>("xmin"),
      psTPart_Angle_Pt10.getParameter<double>("xmax"));
  TPart_Eta_Pt10->setAxisTitle("TPart #eta", 1);
  TPart_Eta_Pt10->setAxisTitle("# TParticles (with p_{T}>10 GeV)", 2);
  
  HistoName = "TPart_Phi_Pt10";
  TPart_Phi_Pt10 = dqmStore_->book1D(HistoName, HistoName,
      psTPart_Angle_Pt10.getParameter<int32_t>("Nbinsx"),
      psTPart_Angle_Pt10.getParameter<double>("xmin"),
      psTPart_Angle_Pt10.getParameter<double>("xmax"));
  TPart_Phi_Pt10->setAxisTitle("TPart #phi", 1);
  TPart_Phi_Pt10->setAxisTitle("# TParticles (with p_{T}>10 GeV)", 2);
  
  
  if ( verbosePlots_ )
  {
    HistoName = "TPart_Cluster_Pt";
    TPart_Cluster_Pt = dqmStore_->book1D(HistoName, HistoName,
        psTPart_Pt.getParameter<int32_t>("Nbinsx"),
        psTPart_Pt.getParameter<double>("xmin"),
        psTPart_Pt.getParameter<double>("xmax"));
    TPart_Cluster_Pt->setAxisTitle("TPart p_{T} [GeV]", 1);
    TPart_Cluster_Pt->setAxisTitle("# TParticles", 2);

    HistoName = "TPart_Cluster_Eta_Pt10";
    TPart_Cluster_Eta_Pt10 = dqmStore_->book1D(HistoName, HistoName,
        psTPart_Angle_Pt10.getParameter<int32_t>("Nbinsx"),
        psTPart_Angle_Pt10.getParameter<double>("xmin"),
        psTPart_Angle_Pt10.getParameter<double>("xmax"));
    TPart_Cluster_Eta_Pt10->setAxisTitle("TPart #eta", 1);
    TPart_Cluster_Eta_Pt10->setAxisTitle("# TParticles (with p_{T}>10 GeV)", 2);

    HistoName = "TPart_Cluster_Phi_Pt10";
    TPart_Cluster_Phi_Pt10 = dqmStore_->book1D(HistoName, HistoName,
        psTPart_Angle_Pt10.getParameter<int32_t>("Nbinsx"),
        psTPart_Angle_Pt10.getParameter<double>("xmin"),
        psTPart_Angle_Pt10.getParameter<double>("xmax"));
    TPart_Cluster_Phi_Pt10->setAxisTitle("TPart #phi", 1);
    TPart_Cluster_Phi_Pt10->setAxisTitle("# TParticles (with p_{T}>10 GeV)", 2);
    
    HistoName = "TPart_Stub_Pt";
    TPart_Stub_Pt = dqmStore_->book1D(HistoName, HistoName,
        psTPart_Pt.getParameter<int32_t>("Nbinsx"),
        psTPart_Pt.getParameter<double>("xmin"),
        psTPart_Pt.getParameter<double>("xmax"));
    TPart_Stub_Pt->setAxisTitle("TPart p_{T} [GeV]", 1);
    TPart_Stub_Pt->setAxisTitle("# TParticles", 2);

    HistoName = "TPart_Stub_Eta_Pt10";
    TPart_Stub_Eta_Pt10 = dqmStore_->book1D(HistoName, HistoName,
        psTPart_Angle_Pt10.getParameter<int32_t>("Nbinsx"),
        psTPart_Angle_Pt10.getParameter<double>("xmin"),
        psTPart_Angle_Pt10.getParameter<double>("xmax"));
    TPart_Stub_Eta_Pt10->setAxisTitle("TPart #eta", 1);
    TPart_Stub_Eta_Pt10->setAxisTitle("# TParticles (with p_{T}>10 GeV)", 2);

    HistoName = "TPart_Stub_Phi_Pt10";
    TPart_Stub_Phi_Pt10 = dqmStore_->book1D(HistoName, HistoName,
        psTPart_Angle_Pt10.getParameter<int32_t>("Nbinsx"),
        psTPart_Angle_Pt10.getParameter<double>("xmin"),
        psTPart_Angle_Pt10.getParameter<double>("xmax"));
    TPart_Stub_Phi_Pt10->setAxisTitle("TPart #phi", 1);
    TPart_Stub_Phi_Pt10->setAxisTitle("# TParticles (with p_{T}>10 GeV)", 2);
    
    HistoName = "TPart_Track_LQ_Pt";
    TPart_Track_LQ_Pt = dqmStore_->book1D(HistoName, HistoName,
        psTPart_Pt.getParameter<int32_t>("Nbinsx"),
        psTPart_Pt.getParameter<double>("xmin"),
        psTPart_Pt.getParameter<double>("xmax"));
    TPart_Track_LQ_Pt->setAxisTitle("TPart p_{T} [GeV]", 1);
    TPart_Track_LQ_Pt->setAxisTitle("# TParticles", 2);

    HistoName = "TPart_Track_LQ_Eta_Pt10";
    TPart_Track_LQ_Eta_Pt10 = dqmStore_->book1D(HistoName, HistoName,
        psTPart_Angle_Pt10.getParameter<int32_t>("Nbinsx"),
        psTPart_Angle_Pt10.getParameter<double>("xmin"),
        psTPart_Angle_Pt10.getParameter<double>("xmax"));
    TPart_Track_LQ_Eta_Pt10->setAxisTitle("TPart #eta", 1);
    TPart_Track_LQ_Eta_Pt10->setAxisTitle("# TParticles (with p_{T}>10 GeV)", 2);

    HistoName = "TPart_Track_LQ_Phi_Pt10";
    TPart_Track_LQ_Phi_Pt10 = dqmStore_->book1D(HistoName, HistoName,
        psTPart_Angle_Pt10.getParameter<int32_t>("Nbinsx"),
        psTPart_Angle_Pt10.getParameter<double>("xmin"),
        psTPart_Angle_Pt10.getParameter<double>("xmax"));
    TPart_Track_LQ_Phi_Pt10->setAxisTitle("TPart #phi", 1);
    TPart_Track_LQ_Phi_Pt10->setAxisTitle("# TParticles (with p_{T}>10 GeV)", 2);
    
    HistoName = "TPart_Track_HQ_Pt";
    TPart_Track_HQ_Pt = dqmStore_->book1D(HistoName, HistoName,
        psTPart_Pt.getParameter<int32_t>("Nbinsx"),
        psTPart_Pt.getParameter<double>("xmin"),
        psTPart_Pt.getParameter<double>("xmax"));
    TPart_Track_HQ_Pt->setAxisTitle("TPart p_{T} [GeV]", 1);
    TPart_Track_HQ_Pt->setAxisTitle("# TParticles", 2);

    HistoName = "TPart_Track_HQ_Eta_Pt10";
    TPart_Track_HQ_Eta_Pt10 = dqmStore_->book1D(HistoName, HistoName,
        psTPart_Angle_Pt10.getParameter<int32_t>("Nbinsx"),
        psTPart_Angle_Pt10.getParameter<double>("xmin"),
        psTPart_Angle_Pt10.getParameter<double>("xmax"));
    TPart_Track_HQ_Eta_Pt10->setAxisTitle("TPart #eta", 1);
    TPart_Track_HQ_Eta_Pt10->setAxisTitle("# TParticles (with p_{T}>10 GeV)", 2);

    HistoName = "TPart_Track_HQ_Phi_Pt10";
    TPart_Track_HQ_Phi_Pt10 = dqmStore_->book1D(HistoName, HistoName,
        psTPart_Angle_Pt10.getParameter<int32_t>("Nbinsx"),
        psTPart_Angle_Pt10.getParameter<double>("xmin"),
        psTPart_Angle_Pt10.getParameter<double>("xmax"));
    TPart_Track_HQ_Phi_Pt10->setAxisTitle("TPart #phi", 1);
    TPart_Track_HQ_Phi_Pt10->setAxisTitle("# TParticles (with p_{T}>10 GeV)", 2);
  } /// End verbosePlots
  
  
  /// Eta distribution of Tracking Particles (Stubs in PS/2S modules)
  edm::ParameterSet psTPart_Eta_PS2S =  conf_.getParameter<edm::ParameterSet>("TH1TPart_Eta_PS2S");
  if ( verbosePlots_ )
  {
    HistoName = "TPart_Stub_Eta_Pt10_Normalization";
    TPart_Stub_Eta_Pt10_Normalization = dqmStore_->book1D(HistoName, HistoName,
        psTPart_Eta_PS2S.getParameter<int32_t>("Nbinsx"),
        psTPart_Eta_PS2S.getParameter<double>("xmin"),
        psTPart_Eta_PS2S.getParameter<double>("xmax"));
    TPart_Stub_Eta_Pt10_Normalization->setAxisTitle("TPart #eta (when p_{T}>10 GeV)", 1);
    TPart_Stub_Eta_Pt10_Normalization->setAxisTitle("# Genuine L1 Stubs", 2);
  }
  
  HistoName = "TPart_Stub_Eta_Pt10_NumPS";
  TPart_Stub_Eta_Pt10_NumPS = dqmStore_->book1D(HistoName, HistoName,
      psTPart_Eta_PS2S.getParameter<int32_t>("Nbinsx"),
      psTPart_Eta_PS2S.getParameter<double>("xmin"),
      psTPart_Eta_PS2S.getParameter<double>("xmax"));
  TPart_Stub_Eta_Pt10_NumPS->setAxisTitle("TPart #eta (when p_{T}>10 GeV)", 1);
  TPart_Stub_Eta_Pt10_NumPS->setAxisTitle("# Genuine L1 Stubs (PS modules)", 2);
  
  HistoName = "TPart_Stub_Eta_Pt10_Num2S";
  TPart_Stub_Eta_Pt10_Num2S = dqmStore_->book1D(HistoName, HistoName,
      psTPart_Eta_PS2S.getParameter<int32_t>("Nbinsx"),
      psTPart_Eta_PS2S.getParameter<double>("xmin"),
      psTPart_Eta_PS2S.getParameter<double>("xmax"));
  TPart_Stub_Eta_Pt10_Num2S->setAxisTitle("TPart #eta (when p_{T}>10 GeV)", 1);
  TPart_Stub_Eta_Pt10_Num2S->setAxisTitle("# Genuine L1 Stubs (2S modules)", 2);
  
  
  
  dqmStore_->setCurrentFolder(topFolderName_+"/MCTruth/");
  
  /// Eta distribution of Tracking Particles (Cluster width)
  // Inner
  edm::ParameterSet psTPart_Eta_CW =  conf_.getParameter<edm::ParameterSet>("TH1TPart_Eta_CW");
  HistoName = "TPart_Eta_ICW_1";
  TPart_Eta_ICW_1 = dqmStore_->book1D(HistoName, HistoName,
      psTPart_Eta_CW.getParameter<int32_t>("Nbinsx"),
      psTPart_Eta_CW.getParameter<double>("xmin"),
      psTPart_Eta_CW.getParameter<double>("xmax"));
  TPart_Eta_ICW_1->setAxisTitle("#eta", 1);
  TPart_Eta_ICW_1->setAxisTitle("# TParticles", 2);
  
  HistoName = "TPart_Eta_ICW_2";
  TPart_Eta_ICW_2 = dqmStore_->book1D(HistoName, HistoName,
      psTPart_Eta_CW.getParameter<int32_t>("Nbinsx"),
      psTPart_Eta_CW.getParameter<double>("xmin"),
      psTPart_Eta_CW.getParameter<double>("xmax"));
  TPart_Eta_ICW_2->setAxisTitle("#eta", 1);
  TPart_Eta_ICW_2->setAxisTitle("# TParticles", 2);
  
  HistoName = "TPart_Eta_ICW_3";
  TPart_Eta_ICW_3 = dqmStore_->book1D(HistoName, HistoName,
      psTPart_Eta_CW.getParameter<int32_t>("Nbinsx"),
      psTPart_Eta_CW.getParameter<double>("xmin"),
      psTPart_Eta_CW.getParameter<double>("xmax"));
  TPart_Eta_ICW_3->setAxisTitle("#eta", 1);
  TPart_Eta_ICW_3->setAxisTitle("# TParticles", 2);
  
  
  // Outer
  HistoName = "TPart_Eta_OCW_1";
  TPart_Eta_OCW_1 = dqmStore_->book1D(HistoName, HistoName,
      psTPart_Eta_CW.getParameter<int32_t>("Nbinsx"),
      psTPart_Eta_CW.getParameter<double>("xmin"),
      psTPart_Eta_CW.getParameter<double>("xmax"));
  TPart_Eta_OCW_1->setAxisTitle("#eta", 1);
  TPart_Eta_OCW_1->setAxisTitle("# TParticles", 2);
  
  HistoName = "TPart_Eta_OCW_2";
  TPart_Eta_OCW_2 = dqmStore_->book1D(HistoName, HistoName,
      psTPart_Eta_CW.getParameter<int32_t>("Nbinsx"),
      psTPart_Eta_CW.getParameter<double>("xmin"),
      psTPart_Eta_CW.getParameter<double>("xmax"));
  TPart_Eta_OCW_2->setAxisTitle("#eta", 1);
  TPart_Eta_OCW_2->setAxisTitle("# TParticles", 2);
  
  HistoName = "TPart_Eta_OCW_3";
  TPart_Eta_OCW_3 = dqmStore_->book1D(HistoName, HistoName,
      psTPart_Eta_CW.getParameter<int32_t>("Nbinsx"),
      psTPart_Eta_CW.getParameter<double>("xmin"),
      psTPart_Eta_CW.getParameter<double>("xmax"));
  TPart_Eta_OCW_3->setAxisTitle("#eta", 1);
  TPart_Eta_OCW_3->setAxisTitle("# TParticles", 2);
  
  // Normalization
  if ( verbosePlots_ )
  {
    HistoName = "TPart_Eta_INormalization";
    TPart_Eta_INormalization = dqmStore_->book1D(HistoName, HistoName,
        psTPart_Eta_CW.getParameter<int32_t>("Nbinsx"),
        psTPart_Eta_CW.getParameter<double>("xmin"),
        psTPart_Eta_CW.getParameter<double>("xmax"));
    TPart_Eta_INormalization->setAxisTitle("TPart_Eta_INormalization", 1);
    TPart_Eta_INormalization->setAxisTitle("# TParticles", 2);

    HistoName = "TPart_Eta_ONormalization";
    TPart_Eta_ONormalization = dqmStore_->book1D(HistoName, HistoName,
        psTPart_Eta_CW.getParameter<int32_t>("Nbinsx"),
        psTPart_Eta_CW.getParameter<double>("xmin"),
        psTPart_Eta_CW.getParameter<double>("xmax"));
    TPart_Eta_ONormalization->setAxisTitle("TPart_Eta_ONormalization", 1);
    TPart_Eta_ONormalization->setAxisTitle("# TParticles", 2);
  }
  
  
  /// PID
  edm::ParameterSet psCluster_PID =  conf_.getParameter<edm::ParameterSet>("TH2Cluster_PID");
  HistoName = "Cluster_PID";
  Cluster_PID = dqmStore_->book2D(HistoName, HistoName,
      psCluster_PID.getParameter<int32_t>("Nbinsx"),
      psCluster_PID.getParameter<double>("xmin"),
      psCluster_PID.getParameter<double>("xmax"),
      psCluster_PID.getParameter<int32_t>("Nbinsy"),
      psCluster_PID.getParameter<double>("ymin"),
      psCluster_PID.getParameter<double>("ymax"));
  Cluster_PID->setAxisTitle("L1 Cluster pdgID", 1);
  Cluster_PID->setAxisTitle("Stack Member", 2);
  
  edm::ParameterSet psStub_PID =  conf_.getParameter<edm::ParameterSet>("TH1Stub_PID");
  HistoName = "Stub_PID";
  Stub_PID = dqmStore_->book1D(HistoName, HistoName,
      psStub_PID.getParameter<int32_t>("Nbinsx"),
      psStub_PID.getParameter<double>("xmin"),
      psStub_PID.getParameter<double>("xmax"));
  Stub_PID->setAxisTitle("L1 Stub pdgID", 1);
  Stub_PID->setAxisTitle("# L1 Stubs", 2);
  
  
  // TTTrack Chi2 vs TPart Eta
  edm::ParameterSet psTrack_Chi2_TPart_Eta =  conf_.getParameter<edm::ParameterSet>("TH2Track_Chi2");
  HistoName = "Track_LQ_Chi2_TPart_Eta";
  Track_LQ_Chi2_TPart_Eta = dqmStore_->book2D(HistoName, HistoName,
      psTrack_Chi2_TPart_Eta.getParameter<int32_t>("Nbinsx"),
      psTrack_Chi2_TPart_Eta.getParameter<double>("xmin"),
      psTrack_Chi2_TPart_Eta.getParameter<double>("xmax"),
      psTrack_Chi2_TPart_Eta.getParameter<int32_t>("Nbinsy"),
      psTrack_Chi2_TPart_Eta.getParameter<double>("ymin"),
      psTrack_Chi2_TPart_Eta.getParameter<double>("ymax"));
  Track_LQ_Chi2_TPart_Eta->setAxisTitle("TPart #eta", 1);
  Track_LQ_Chi2_TPart_Eta->setAxisTitle("L1 Track #chi^{2}", 2);
  
  HistoName = "Track_HQ_Chi2_TPart_Eta";
  Track_HQ_Chi2_TPart_Eta = dqmStore_->book2D(HistoName, HistoName,
      psTrack_Chi2_TPart_Eta.getParameter<int32_t>("Nbinsx"),
      psTrack_Chi2_TPart_Eta.getParameter<double>("xmin"),
      psTrack_Chi2_TPart_Eta.getParameter<double>("xmax"),
      psTrack_Chi2_TPart_Eta.getParameter<int32_t>("Nbinsy"),
      psTrack_Chi2_TPart_Eta.getParameter<double>("ymin"),
      psTrack_Chi2_TPart_Eta.getParameter<double>("ymax"));
  Track_HQ_Chi2_TPart_Eta->setAxisTitle("TPart #eta", 1);
  Track_HQ_Chi2_TPart_Eta->setAxisTitle("L1 Track #chi^{2}", 2);
  
  // TTTrack Chi2/ndf vs Eta
  edm::ParameterSet psTrack_Chi2Red_TPart_Eta =  conf_.getParameter<edm::ParameterSet>("TH2Track_Chi2Red");
  HistoName = "Track_LQ_Chi2Red_TPart_Eta";
  Track_LQ_Chi2Red_TPart_Eta = dqmStore_->book2D(HistoName, HistoName,
      psTrack_Chi2Red_TPart_Eta.getParameter<int32_t>("Nbinsx"),
      psTrack_Chi2Red_TPart_Eta.getParameter<double>("xmin"),
      psTrack_Chi2Red_TPart_Eta.getParameter<double>("xmax"),
      psTrack_Chi2Red_TPart_Eta.getParameter<int32_t>("Nbinsy"),
     	psTrack_Chi2Red_TPart_Eta.getParameter<double>("ymin"),
      psTrack_Chi2Red_TPart_Eta.getParameter<double>("ymax"));
  Track_LQ_Chi2Red_TPart_Eta->setAxisTitle("TPart #eta", 1);
  Track_LQ_Chi2Red_TPart_Eta->setAxisTitle("L1 Track #chi^{2}/ndf", 2);
  
  HistoName = "Track_HQ_Chi2Red_TPart_Eta";
  Track_HQ_Chi2Red_TPart_Eta = dqmStore_->book2D(HistoName, HistoName,
      psTrack_Chi2Red_TPart_Eta.getParameter<int32_t>("Nbinsx"),
      psTrack_Chi2Red_TPart_Eta.getParameter<double>("xmin"),
      psTrack_Chi2Red_TPart_Eta.getParameter<double>("xmax"),
      psTrack_Chi2Red_TPart_Eta.getParameter<int32_t>("Nbinsy"),
     	psTrack_Chi2Red_TPart_Eta.getParameter<double>("ymin"),
      psTrack_Chi2Red_TPart_Eta.getParameter<double>("ymax"));
  Track_HQ_Chi2Red_TPart_Eta->setAxisTitle("TPart #eta", 1);
  Track_HQ_Chi2Red_TPart_Eta->setAxisTitle("L1 Track #chi^{2}/ndf", 2);
  
  
  /// Plots for debugging
  if ( verbosePlots_ )
  {
    /// Stub properties compared to TParticles
    dqmStore_->setCurrentFolder(topFolderName_+"/TTStubVSTPart/");

    // InvpT
    edm::ParameterSet psStub_InvPt =  conf_.getParameter<edm::ParameterSet>("TH2Stub_InvPt");
    HistoName = "Stub_InvPt_TPart_InvPt_AllLayers";
    Stub_InvPt_TPart_InvPt_AllLayers = dqmStore_->book2D(HistoName, HistoName,
        psStub_InvPt.getParameter<int32_t>("Nbinsx"),
        psStub_InvPt.getParameter<double>("xmin"),
        psStub_InvPt.getParameter<double>("xmax"),
        psStub_InvPt.getParameter<int32_t>("Nbinsy"),
        psStub_InvPt.getParameter<double>("ymin"),
        psStub_InvPt.getParameter<double>("ymax"));
    Stub_InvPt_TPart_InvPt_AllLayers->setAxisTitle("TPart 1/p_{T} [1/GeV]", 1);
    Stub_InvPt_TPart_InvPt_AllLayers->setAxisTitle("L1 Stub 1/p_{T} [1/GeV]", 2);

    HistoName = "Stub_InvPt_TPart_InvPt_AllDisks";
    Stub_InvPt_TPart_InvPt_AllDisks = dqmStore_->book2D(HistoName, HistoName,
        psStub_InvPt.getParameter<int32_t>("Nbinsx"),
        psStub_InvPt.getParameter<double>("xmin"),
        psStub_InvPt.getParameter<double>("xmax"),
        psStub_InvPt.getParameter<int32_t>("Nbinsy"),
        psStub_InvPt.getParameter<double>("ymin"),
        psStub_InvPt.getParameter<double>("ymax"));
    Stub_InvPt_TPart_InvPt_AllDisks->setAxisTitle("TPart 1/p_{T} [1/GeV]", 1);
    Stub_InvPt_TPart_InvPt_AllDisks->setAxisTitle("L1 Stub 1/p_{T} [1/GeV]", 2);

    // pT
    edm::ParameterSet psStub_Pt =  conf_.getParameter<edm::ParameterSet>("TH2Stub_Pt");
    HistoName = "Stub_Pt_TPart_Pt_AllLayers";
    Stub_Pt_TPart_Pt_AllLayers = dqmStore_->book2D(HistoName, HistoName,
        psStub_Pt.getParameter<int32_t>("Nbinsx"),
        psStub_Pt.getParameter<double>("xmin"),
        psStub_Pt.getParameter<double>("xmax"),
        psStub_Pt.getParameter<int32_t>("Nbinsy"),
        psStub_Pt.getParameter<double>("ymin"),
        psStub_Pt.getParameter<double>("ymax"));
    Stub_Pt_TPart_Pt_AllLayers->setAxisTitle("TPart p_{T} [GeV]", 1);
    Stub_Pt_TPart_Pt_AllLayers->setAxisTitle("L1 Stub p_{T} [GeV]", 2);

    HistoName = "Stub_Pt_TPart_Pt_AllDisks";
    Stub_Pt_TPart_Pt_AllDisks = dqmStore_->book2D(HistoName, HistoName,
        psStub_Pt.getParameter<int32_t>("Nbinsx"),
        psStub_Pt.getParameter<double>("xmin"),
        psStub_Pt.getParameter<double>("xmax"),
        psStub_Pt.getParameter<int32_t>("Nbinsy"),
        psStub_Pt.getParameter<double>("ymin"),
        psStub_Pt.getParameter<double>("ymax"));
    Stub_Pt_TPart_Pt_AllDisks->setAxisTitle("TPart p_{T} [GeV]", 1);
    Stub_Pt_TPart_Pt_AllDisks->setAxisTitle("L1 Stub p_{T} [GeV]", 2);

    // Eta
    edm::ParameterSet psStub_Eta =  conf_.getParameter<edm::ParameterSet>("TH2Stub_Eta");
    HistoName = "Stub_Eta_TPart_Eta_AllLayers";
    Stub_Eta_TPart_Eta_AllLayers = dqmStore_->book2D(HistoName, HistoName,
        psStub_Eta.getParameter<int32_t>("Nbinsx"),
        psStub_Eta.getParameter<double>("xmin"),
        psStub_Eta.getParameter<double>("xmax"),
        psStub_Eta.getParameter<int32_t>("Nbinsy"),
        psStub_Eta.getParameter<double>("ymin"),
        psStub_Eta.getParameter<double>("ymax"));
    Stub_Eta_TPart_Eta_AllLayers->setAxisTitle("TPart #eta", 1);
    Stub_Eta_TPart_Eta_AllLayers->setAxisTitle("L1 Stub #eta", 2);

    HistoName = "Stub_Eta_TPart_Eta_AllDisks";
    Stub_Eta_TPart_Eta_AllDisks = dqmStore_->book2D(HistoName, HistoName,
        psStub_Eta.getParameter<int32_t>("Nbinsx"),
        psStub_Eta.getParameter<double>("xmin"),
        psStub_Eta.getParameter<double>("xmax"),
        psStub_Eta.getParameter<int32_t>("Nbinsy"),
        psStub_Eta.getParameter<double>("ymin"),
        psStub_Eta.getParameter<double>("ymax"));
    Stub_Eta_TPart_Eta_AllDisks->setAxisTitle("TPart #eta", 1);
    Stub_Eta_TPart_Eta_AllDisks->setAxisTitle("L1 Stub #eta", 2);

    // Phi
    edm::ParameterSet psStub_Phi =  conf_.getParameter<edm::ParameterSet>("TH2Stub_Phi");
    HistoName = "Stub_Phi_TPart_Phi_AllLayers";
    Stub_Phi_TPart_Phi_AllLayers = dqmStore_->book2D(HistoName, HistoName,
        psStub_Phi.getParameter<int32_t>("Nbinsx"),
        psStub_Phi.getParameter<double>("xmin"),
        psStub_Phi.getParameter<double>("xmax"),
        psStub_Phi.getParameter<int32_t>("Nbinsy"),
        psStub_Phi.getParameter<double>("ymin"),
        psStub_Phi.getParameter<double>("ymax"));
    Stub_Phi_TPart_Phi_AllLayers->setAxisTitle("TPart #phi", 1);
    Stub_Phi_TPart_Phi_AllLayers->setAxisTitle("L1 Stub #phi", 2);

    HistoName = "Stub_Phi_TPart_Phi_AllDisks";
    Stub_Phi_TPart_Phi_AllDisks = dqmStore_->book2D(HistoName, HistoName,
        psStub_Phi.getParameter<int32_t>("Nbinsx"),
        psStub_Phi.getParameter<double>("xmin"),
        psStub_Phi.getParameter<double>("xmax"),
        psStub_Phi.getParameter<int32_t>("Nbinsy"),
        psStub_Phi.getParameter<double>("ymin"),
        psStub_Phi.getParameter<double>("ymax"));
    Stub_Phi_TPart_Phi_AllDisks->setAxisTitle("TPart #phi", 1);
    Stub_Phi_TPart_Phi_AllDisks->setAxisTitle("L1 Stub #phi", 2);

    //InvPtRes
    edm::ParameterSet psStub_InvPtRes =  conf_.getParameter<edm::ParameterSet>("TH2Stub_InvPtRes");
    HistoName = "Stub_InvPtRes_TPart_Eta_AllLayers";
    Stub_InvPtRes_TPart_Eta_AllLayers = dqmStore_->book2D(HistoName, HistoName,
        psStub_InvPtRes.getParameter<int32_t>("Nbinsx"),
        psStub_InvPtRes.getParameter<double>("xmin"),
        psStub_InvPtRes.getParameter<double>("xmax"),
        psStub_InvPtRes.getParameter<int32_t>("Nbinsy"),
        psStub_InvPtRes.getParameter<double>("ymin"),
        psStub_InvPtRes.getParameter<double>("ymax"));
    Stub_InvPtRes_TPart_Eta_AllLayers->setAxisTitle("TPart #eta", 1);
    Stub_InvPtRes_TPart_Eta_AllLayers->setAxisTitle("L1 Stub 1/p_{T} - TPart 1/p_{T} [1/GeV]", 2);  

    HistoName = "Stub_InvPtRes_TPart_Eta_AllDisks";
    Stub_InvPtRes_TPart_Eta_AllDisks = dqmStore_->book2D(HistoName, HistoName,
        psStub_InvPtRes.getParameter<int32_t>("Nbinsx"),
        psStub_InvPtRes.getParameter<double>("xmin"),
        psStub_InvPtRes.getParameter<double>("xmax"),
        psStub_InvPtRes.getParameter<int32_t>("Nbinsy"),
        psStub_InvPtRes.getParameter<double>("ymin"),
        psStub_InvPtRes.getParameter<double>("ymax"));
    Stub_InvPtRes_TPart_Eta_AllDisks->setAxisTitle("TPart #eta", 1);
    Stub_InvPtRes_TPart_Eta_AllDisks->setAxisTitle("L1 Stub 1/p_{T} - TPart 1/p_{T} [1/GeV]", 2); 

    //PtRes
    edm::ParameterSet psStub_PtRes =  conf_.getParameter<edm::ParameterSet>("TH2Stub_PtRes");
    HistoName = "Stub_PtRes_TPart_Eta_AllLayers";
    Stub_PtRes_TPart_Eta_AllLayers = dqmStore_->book2D(HistoName, HistoName,
         psStub_PtRes.getParameter<int32_t>("Nbinsx"),
         psStub_PtRes.getParameter<double>("xmin"),
         psStub_PtRes.getParameter<double>("xmax"),
         psStub_PtRes.getParameter<int32_t>("Nbinsy"),
         psStub_PtRes.getParameter<double>("ymin"),
         psStub_PtRes.getParameter<double>("ymax"));
    Stub_PtRes_TPart_Eta_AllLayers->setAxisTitle("TPart #eta", 1);
    Stub_PtRes_TPart_Eta_AllLayers->setAxisTitle("L1 Stub p_{T} - TPart p_{T} [GeV]", 2);

    HistoName = "Stub_PtRes_TPart_Eta_AllDisks";
    Stub_PtRes_TPart_Eta_AllDisks = dqmStore_->book2D(HistoName, HistoName,
        psStub_PtRes.getParameter<int32_t>("Nbinsx"),
        psStub_PtRes.getParameter<double>("xmin"),
        psStub_PtRes.getParameter<double>("xmax"),
        psStub_PtRes.getParameter<int32_t>("Nbinsy"),
        psStub_PtRes.getParameter<double>("ymin"),
        psStub_PtRes.getParameter<double>("ymax"));
    Stub_PtRes_TPart_Eta_AllDisks->setAxisTitle("TPart #eta", 1);
    Stub_PtRes_TPart_Eta_AllDisks->setAxisTitle("L1 Stub p_{T} - TPart p_{T} [GeV]", 2); 	

    // EtaRes
    edm::ParameterSet psStub_EtaRes =  conf_.getParameter<edm::ParameterSet>("TH2Stub_EtaRes");
    HistoName = "Stub_EtaRes_TPart_Eta_AllLayers";
    Stub_EtaRes_TPart_Eta_AllLayers = dqmStore_->book2D(HistoName, HistoName,
        psStub_EtaRes.getParameter<int32_t>("Nbinsx"),
        psStub_EtaRes.getParameter<double>("xmin"),
        psStub_EtaRes.getParameter<double>("xmax"),
        psStub_EtaRes.getParameter<int32_t>("Nbinsy"),
        psStub_EtaRes.getParameter<double>("ymin"),
        psStub_EtaRes.getParameter<double>("ymax"));
    Stub_EtaRes_TPart_Eta_AllLayers->setAxisTitle("TPart #eta", 1);
    Stub_EtaRes_TPart_Eta_AllLayers->setAxisTitle("L1 Stub #eta - TPart #eta", 2);

    HistoName = "Stub_EtaRes_TPart_Eta_AllDisks";
    Stub_EtaRes_TPart_Eta_AllDisks = dqmStore_->book2D(HistoName, HistoName,
        psStub_EtaRes.getParameter<int32_t>("Nbinsx"),
        psStub_EtaRes.getParameter<double>("xmin"),
        psStub_EtaRes.getParameter<double>("xmax"),
        psStub_EtaRes.getParameter<int32_t>("Nbinsy"),
        psStub_EtaRes.getParameter<double>("ymin"),
        psStub_EtaRes.getParameter<double>("ymax"));
    Stub_EtaRes_TPart_Eta_AllDisks->setAxisTitle("TPart #eta", 1);
    Stub_EtaRes_TPart_Eta_AllDisks->setAxisTitle("L1 Stub #eta - TPart #eta", 2);

    // PhiRes
    edm::ParameterSet psStub_PhiRes =  conf_.getParameter<edm::ParameterSet>("TH2Stub_PhiRes");
    HistoName = "Stub_PhiRes_TPart_Eta_AllLayers";
    Stub_PhiRes_TPart_Eta_AllLayers = dqmStore_->book2D(HistoName, HistoName,
        psStub_PhiRes.getParameter<int32_t>("Nbinsx"),
        psStub_PhiRes.getParameter<double>("xmin"),
        psStub_PhiRes.getParameter<double>("xmax"),
        psStub_PhiRes.getParameter<int32_t>("Nbinsy"),
        psStub_PhiRes.getParameter<double>("ymin"),
        psStub_PhiRes.getParameter<double>("ymax"));
    Stub_PhiRes_TPart_Eta_AllLayers->setAxisTitle("TPart Eta", 1);
    Stub_PhiRes_TPart_Eta_AllLayers->setAxisTitle("L1 Stub #phi - TPart #phi", 2);

    HistoName = "Stub_PhiRes_TPart_Eta_AllDisks";
    Stub_PhiRes_TPart_Eta_AllDisks = dqmStore_->book2D(HistoName, HistoName,
        psStub_PhiRes.getParameter<int32_t>("Nbinsx"),
        psStub_PhiRes.getParameter<double>("xmin"),
        psStub_PhiRes.getParameter<double>("xmax"),
        psStub_PhiRes.getParameter<int32_t>("Nbinsy"),
        psStub_PhiRes.getParameter<double>("ymin"),
        psStub_PhiRes.getParameter<double>("ymax"));
    Stub_PhiRes_TPart_Eta_AllDisks->setAxisTitle("TPart #eta", 1);
    Stub_PhiRes_TPart_Eta_AllDisks->setAxisTitle("L1 Stub #phi - TPart #phi", 2);

    // Width vs. InvPt
    edm::ParameterSet psStub_W_InvPt =  conf_.getParameter<edm::ParameterSet>("TH2Stub_W_InvPt");
    HistoName = "Stub_W_TPart_InvPt_AllLayers";
    Stub_W_TPart_InvPt_AllLayers = dqmStore_->book2D(HistoName, HistoName,
        psStub_W_InvPt.getParameter<int32_t>("Nbinsx"),
        psStub_W_InvPt.getParameter<double>("xmin"),
        psStub_W_InvPt.getParameter<double>("xmax"),
        psStub_W_InvPt.getParameter<int32_t>("Nbinsy"),
        psStub_W_InvPt.getParameter<double>("ymin"),
        psStub_W_InvPt.getParameter<double>("ymax"));
    Stub_W_TPart_InvPt_AllLayers->setAxisTitle("TPart 1/p_{T} [1/GeV]", 1);
    Stub_W_TPart_InvPt_AllLayers->setAxisTitle("L1 Stub Width", 2);

    HistoName = "Stub_W_TPart_InvPt_AllDisks";
    Stub_W_TPart_InvPt_AllDisks = dqmStore_->book2D(HistoName, HistoName,
        psStub_W_InvPt.getParameter<int32_t>("Nbinsx"),
        psStub_W_InvPt.getParameter<double>("xmin"),
        psStub_W_InvPt.getParameter<double>("xmax"),
        psStub_W_InvPt.getParameter<int32_t>("Nbinsy"),
        psStub_W_InvPt.getParameter<double>("ymin"),
        psStub_W_InvPt.getParameter<double>("ymax"));
    Stub_W_TPart_InvPt_AllDisks->setAxisTitle("TPart 1/p_{T} [1/GeV]", 1);
    Stub_W_TPart_InvPt_AllDisks->setAxisTitle("L1 Stub Width", 2);

    // Width vs. Pt
    edm::ParameterSet psStub_W_Pt =  conf_.getParameter<edm::ParameterSet>("TH2Stub_W_Pt");
    HistoName = "Stub_W_TPart_Pt_AllLayers";
    Stub_W_TPart_Pt_AllLayers = dqmStore_->book2D(HistoName, HistoName,
        psStub_W_Pt.getParameter<int32_t>("Nbinsx"),
        psStub_W_Pt.getParameter<double>("xmin"),
        psStub_W_Pt.getParameter<double>("xmax"),
        psStub_W_Pt.getParameter<int32_t>("Nbinsy"),
        psStub_W_Pt.getParameter<double>("ymin"),
        psStub_W_Pt.getParameter<double>("ymax"));
    Stub_W_TPart_Pt_AllLayers->setAxisTitle("TPart p_{T} [GeV]", 1);
    Stub_W_TPart_Pt_AllLayers->setAxisTitle("L1 Stub Width", 2);

    HistoName = "Stub_W_TPart_Pt_AllDisks";
    Stub_W_TPart_Pt_AllDisks = dqmStore_->book2D(HistoName, HistoName,
        psStub_W_Pt.getParameter<int32_t>("Nbinsx"),
        psStub_W_Pt.getParameter<double>("xmin"),
        psStub_W_Pt.getParameter<double>("xmax"),
        psStub_W_Pt.getParameter<int32_t>("Nbinsy"),
        psStub_W_Pt.getParameter<double>("ymin"),
        psStub_W_Pt.getParameter<double>("ymax"));
    Stub_W_TPart_Pt_AllDisks->setAxisTitle("TPart p_{T} [GeV]", 1);
    Stub_W_TPart_Pt_AllDisks->setAxisTitle("L1 Stub Width", 2);
    
    
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
    Track_LQ_Pt_TPart_Pt->setAxisTitle("TPart p_{T} [GeV]", 1);
    Track_LQ_Pt_TPart_Pt->setAxisTitle("L1 Track p_{T} [GeV]", 2);

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
    Track_LQ_PtRes_TPart_Eta->setAxisTitle("TPart #eta", 1);
    Track_LQ_PtRes_TPart_Eta->setAxisTitle("L1 Track p_{T} - TPart p_{T} [GeV]", 2);

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
    Track_LQ_InvPt_TPart_InvPt->setAxisTitle("TPart 1/p_{T} [1/GeV]", 1);
    Track_LQ_InvPt_TPart_InvPt->setAxisTitle("L1 Track 1/p_{T} [1/GeV]", 2);

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
    Track_LQ_InvPtRes_TPart_Eta->setAxisTitle("TPart #eta", 1);
    Track_LQ_InvPtRes_TPart_Eta->setAxisTitle("L1 Track 1/p_{T} - TPart 1/p_{T} [1/GeV]", 2);

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
    Track_LQ_Phi_TPart_Phi->setAxisTitle("TPart #phi", 1);
    Track_LQ_Phi_TPart_Phi->setAxisTitle("L1 Track #phi", 2);

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
    Track_LQ_PhiRes_TPart_Eta->setAxisTitle("TPart #eta", 1);
    Track_LQ_PhiRes_TPart_Eta->setAxisTitle("L1 Track #phi - TPart #phi", 2);

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
    Track_LQ_Eta_TPart_Eta->setAxisTitle("TPart #eta", 1);
    Track_LQ_Eta_TPart_Eta->setAxisTitle("L1 Track #eta", 2);

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
    Track_LQ_EtaRes_TPart_Eta->setAxisTitle("TPart #eta", 1);
    Track_LQ_EtaRes_TPart_Eta->setAxisTitle("L1 Track #eta - TPart #eta", 2);

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
    Track_LQ_VtxZ0_TPart_VtxZ0->setAxisTitle("TPart Vertex z [cm]", 1);
    Track_LQ_VtxZ0_TPart_VtxZ0->setAxisTitle("L1 Track Vertex z [cm]", 2);

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
    Track_LQ_VtxZ0Res_TPart_Eta->setAxisTitle("TPart #eta", 1);
    Track_LQ_VtxZ0Res_TPart_Eta->setAxisTitle("L1 Track Vertex z - TPart Vertex z [cm]", 2);
    
    
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
    Track_HQ_Pt_TPart_Pt->setAxisTitle("TPart p_{T} [GeV]", 1);
    Track_HQ_Pt_TPart_Pt->setAxisTitle("L1 Track p_{T} [GeV]", 2);

    // PtRes
    HistoName = "Track_HQ_PtRes_TPart_Eta";
    Track_HQ_PtRes_TPart_Eta = dqmStore_->book2D(HistoName, HistoName,
        psTrack_Sim_PtRes.getParameter<int32_t>("Nbinsx"),
        psTrack_Sim_PtRes.getParameter<double>("xmin"),
        psTrack_Sim_PtRes.getParameter<double>("xmax"),
        psTrack_Sim_PtRes.getParameter<int32_t>("Nbinsy"),
        psTrack_Sim_PtRes.getParameter<double>("ymin"),
        psTrack_Sim_PtRes.getParameter<double>("ymax"));
    Track_HQ_PtRes_TPart_Eta->setAxisTitle("TPart #eta", 1);
    Track_HQ_PtRes_TPart_Eta->setAxisTitle("L1 Track p_{T} - TPart p_{T} [GeV]", 2);

    // InvPt
    HistoName = "Track_HQ_InvPt_TPart_InvPt";
    Track_HQ_InvPt_TPart_InvPt = dqmStore_->book2D(HistoName, HistoName,
        psTrack_Sim_InvPt.getParameter<int32_t>("Nbinsx"),
        psTrack_Sim_InvPt.getParameter<double>("xmin"),
        psTrack_Sim_InvPt.getParameter<double>("xmax"),
        psTrack_Sim_InvPt.getParameter<int32_t>("Nbinsy"),
        psTrack_Sim_InvPt.getParameter<double>("ymin"),
        psTrack_Sim_InvPt.getParameter<double>("ymax"));
    Track_HQ_InvPt_TPart_InvPt->setAxisTitle("TPart 1/p_{T} [1/GeV]", 1);
    Track_HQ_InvPt_TPart_InvPt->setAxisTitle("L1 Track 1/p_{T} [1/GeV]", 2);

    // InvPtRes
    HistoName = "Track_HQ_InvPtRes_TPart_Eta";
    Track_HQ_InvPtRes_TPart_Eta = dqmStore_->book2D(HistoName, HistoName,
        psTrack_Sim_InvPtRes.getParameter<int32_t>("Nbinsx"),
        psTrack_Sim_InvPtRes.getParameter<double>("xmin"),
        psTrack_Sim_InvPtRes.getParameter<double>("xmax"),
        psTrack_Sim_InvPtRes.getParameter<int32_t>("Nbinsy"),
        psTrack_Sim_InvPtRes.getParameter<double>("ymin"),
        psTrack_Sim_InvPtRes.getParameter<double>("ymax"));
    Track_HQ_InvPtRes_TPart_Eta->setAxisTitle("TPart #eta", 1);
    Track_HQ_InvPtRes_TPart_Eta->setAxisTitle("L1 Track 1/p_{T} - TPart 1/p_{T} [1/GeV]", 2);

    // Phi
    HistoName = "Track_HQ_Phi_TPart_Phi";
    Track_HQ_Phi_TPart_Phi = dqmStore_->book2D(HistoName, HistoName,
        psTrack_Sim_Phi.getParameter<int32_t>("Nbinsx"),
        psTrack_Sim_Phi.getParameter<double>("xmin"),
        psTrack_Sim_Phi.getParameter<double>("xmax"),
        psTrack_Sim_Phi.getParameter<int32_t>("Nbinsy"),
        psTrack_Sim_Phi.getParameter<double>("ymin"),
        psTrack_Sim_Phi.getParameter<double>("ymax"));
    Track_HQ_Phi_TPart_Phi->setAxisTitle("TPart #phi", 1);
    Track_HQ_Phi_TPart_Phi->setAxisTitle("L1 Track #phi", 2);

    // PhiRes
    HistoName = "Track_HQ_PhiRes_TPart_Eta";
    Track_HQ_PhiRes_TPart_Eta = dqmStore_->book2D(HistoName, HistoName,
        psTrack_Sim_PhiRes.getParameter<int32_t>("Nbinsx"),
        psTrack_Sim_PhiRes.getParameter<double>("xmin"),
        psTrack_Sim_PhiRes.getParameter<double>("xmax"),
        psTrack_Sim_PhiRes.getParameter<int32_t>("Nbinsy"),
        psTrack_Sim_PhiRes.getParameter<double>("ymin"),
        psTrack_Sim_PhiRes.getParameter<double>("ymax"));
    Track_HQ_PhiRes_TPart_Eta->setAxisTitle("TPart #eta", 1);
    Track_HQ_PhiRes_TPart_Eta->setAxisTitle("L1 Track #phi - TPart #phi", 2);

    // Eta
    HistoName = "Track_HQ_Eta_TPart_Eta";
    Track_HQ_Eta_TPart_Eta = dqmStore_->book2D(HistoName, HistoName,
        psTrack_Sim_Eta.getParameter<int32_t>("Nbinsx"),
        psTrack_Sim_Eta.getParameter<double>("xmin"),
        psTrack_Sim_Eta.getParameter<double>("xmax"),
        psTrack_Sim_Eta.getParameter<int32_t>("Nbinsy"),
        psTrack_Sim_Eta.getParameter<double>("ymin"),
        psTrack_Sim_Eta.getParameter<double>("ymax"));
    Track_HQ_Eta_TPart_Eta->setAxisTitle("TPart #eta", 1);
    Track_HQ_Eta_TPart_Eta->setAxisTitle("L1 Track #eta", 2);

    // EtaRes
    HistoName = "Track_HQ_EtaRes_TPart_Eta";
    Track_HQ_EtaRes_TPart_Eta = dqmStore_->book2D(HistoName, HistoName,
        psTrack_Sim_EtaRes.getParameter<int32_t>("Nbinsx"),
        psTrack_Sim_EtaRes.getParameter<double>("xmin"),
        psTrack_Sim_EtaRes.getParameter<double>("xmax"),
        psTrack_Sim_EtaRes.getParameter<int32_t>("Nbinsy"),
        psTrack_Sim_EtaRes.getParameter<double>("ymin"),
        psTrack_Sim_EtaRes.getParameter<double>("ymax"));
    Track_HQ_EtaRes_TPart_Eta->setAxisTitle("TPart #eta", 1);
    Track_HQ_EtaRes_TPart_Eta->setAxisTitle("L1 Track #eta - TPart #eta", 2);

    // Vertex position in z
    HistoName = "Track_HQ_VtxZ0_TPart_VtxZ0";
    Track_HQ_VtxZ0_TPart_VtxZ0 = dqmStore_->book2D(HistoName, HistoName,
        psTrack_Sim_Vtx.getParameter<int32_t>("Nbinsx"),
        psTrack_Sim_Vtx.getParameter<double>("xmin"),
        psTrack_Sim_Vtx.getParameter<double>("xmax"),
        psTrack_Sim_Vtx.getParameter<int32_t>("Nbinsy"),
        psTrack_Sim_Vtx.getParameter<double>("ymin"),
        psTrack_Sim_Vtx.getParameter<double>("ymax"));
    Track_HQ_VtxZ0_TPart_VtxZ0->setAxisTitle("TPart Vertex z [cm]", 1);
    Track_HQ_VtxZ0_TPart_VtxZ0->setAxisTitle("L1 Track Vertex z [cm]", 2);

    // Vertex position in z
    HistoName = "Track_HQ_VtxZ0Res_TPart_Eta";
    Track_HQ_VtxZ0Res_TPart_Eta = dqmStore_->book2D(HistoName, HistoName,
        psTrack_Sim_VtxRes.getParameter<int32_t>("Nbinsx"),
        psTrack_Sim_VtxRes.getParameter<double>("xmin"),
        psTrack_Sim_VtxRes.getParameter<double>("xmax"),
        psTrack_Sim_VtxRes.getParameter<int32_t>("Nbinsy"),
        psTrack_Sim_VtxRes.getParameter<double>("ymin"),
        psTrack_Sim_VtxRes.getParameter<double>("ymax"));
    Track_HQ_VtxZ0Res_TPart_Eta->setAxisTitle("TPart #eta", 1);
    Track_HQ_VtxZ0Res_TPart_Eta->setAxisTitle("L1 Track Vertex z - TPart Vertex z [cm]", 2);
    
  } /// End verbosePlots
  	
}//end of method


// ------------ method called once each job just after ending the event loop  ------------
void 
OuterTrackerMCTruth::endJob(void) 
{
	
}

//define this as a plug-in
DEFINE_FWK_MODULE(OuterTrackerMCTruth);
