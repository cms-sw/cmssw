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
#include "DataFormats/L1TrackTrigger/interface/TTTypes.h"
#include "DataFormats/L1TrackTrigger/interface/TTCluster.h"

#include "SimTracker/TrackTriggerAssociation/interface/TTStubAssociationMap.h"
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
	
	/// TrackingParticles
	edm::Handle< std::vector< TrackingParticle > > TrackingParticleHandle;
	iEvent.getByLabel( "mix", "MergedTrackTruth", TrackingParticleHandle );
	/// Track Trigger
	edm::Handle< edmNew::DetSetVector< TTCluster< Ref_PixelDigi_ > > > PixelDigiTTClusterHandle;
	iEvent.getByLabel( "TTClustersFromPixelDigis", "ClusterInclusive", PixelDigiTTClusterHandle );
	edm::Handle< edmNew::DetSetVector< TTStub< Ref_PixelDigi_ > > >    PixelDigiTTStubHandle;
	iEvent.getByLabel( "TTStubsFromPixelDigis", "StubAccepted",        PixelDigiTTStubHandle );
	/// Track Trigger MC Truth
	edm::Handle< TTClusterAssociationMap< Ref_PixelDigi_ > > MCTruthTTClusterHandle;
	iEvent.getByLabel( "TTClusterAssociatorFromPixelDigis", "ClusterInclusive", MCTruthTTClusterHandle );
	edm::Handle< TTStubAssociationMap< Ref_PixelDigi_ > >    MCTruthTTStubHandle;
	iEvent.getByLabel( "TTStubAssociatorFromPixelDigis", "StubAccepted",        MCTruthTTStubHandle );
	
	
	/// Go on only if there are TrackingParticles
  if( TrackingParticleHandle->size() > 0)
  {
  	/// Loop over the TrackingParticles
		unsigned int tpCnt = 0;
		std::vector< TrackingParticle >::const_iterator iterTP;
		for(iterTP = TrackingParticleHandle->begin(); iterTP !=	TrackingParticleHandle->end(); ++iterTP)
		{
			/// Get the corresponding vertex
			/// Assume perfectly round beamspot
			/// Correct and get the correct TrackingParticle Vertex position wrt beam center
			if ( iterTP->vertex().rho() >= 2 )
				continue;
			
			/// Check beamspot and correction
			SimVtx_XY->Fill( iterTP->vertex().x(), iterTP->vertex().y() );
			SimVtx_RZ->Fill( iterTP->vertex().z(), iterTP->vertex().rho() );
			
			/// Here we have only tracks form primary vertices
			/// Check Pt spectrum and pseudorapidity for over-threshold tracks
			TPart_Pt->Fill( iterTP->p4().pt() );
			if ( iterTP->p4().pt() > 10.0 )
			{
				TPart_Eta_Pt10->Fill( iterTP->momentum().eta() );
				TPart_Phi_Pt10->Fill( iterTP->momentum().phi() > M_PI ?
															iterTP->momentum().phi() - 2*M_PI :
															iterTP->momentum().phi() );
			}

			
			/// Eta coverage
			/// Make the pointer
  		edm::Ptr<TrackingParticle> tempTPPtr( TrackingParticleHandle, tpCnt++ );
			
			/// Search the cluster MC map
			std::vector< edm::Ref< edmNew::DetSetVector< TTCluster< Ref_PixelDigi_ > >, TTCluster< Ref_PixelDigi_ > > > theseClusters = MCTruthTTClusterHandle->findTTClusterRefs( tempTPPtr );
			
			if ( theseClusters.size() > 0 )
			{
				
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
						if ( normIClu == false )
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
						if ( normOClu == false )
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
			
			/// Search the stub MC truth map
			std::vector< edm::Ref< edmNew::DetSetVector< TTStub< Ref_PixelDigi_ > >, TTStub< Ref_PixelDigi_ > > > theseStubs = MCTruthTTStubHandle->findTTStubRefs( tempTPPtr );

			if ( tempTPPtr->p4().pt() <= 10 )
			continue;

			if ( theseStubs.size() > 0 )
			{
				bool normStub = false;

				/// Loop over the Stubs
				for ( unsigned int js = 0; js < theseStubs.size(); js++ )
				{
					/// Check if it is good
					bool genuineStub = MCTruthTTStubHandle->isGenuine( theseStubs.at(js) );
					if ( !genuineStub )
						continue;

					if ( normStub == false )
					{
						TPart_Eta_Pt10_Normalization->Fill( tempTPPtr->momentum().eta() );
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
						TPart_Eta_Pt10_Num2S->Fill( tempTPPtr->momentum().eta() );
					}
					else /// PS
					{
						TPart_Eta_Pt10_NumPS->Fill( tempTPPtr->momentum().eta() );
					}
				} /// End of loop over the Stubs generated by this TrackingParticle
			}
		}	// end loop TrackingParticles
  } // end if there are TrackingParticles
	
	
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
			//bool combinClu      = MCTruthTTClusterHandle->isCombinatoric( tempCluRef );
			//bool unknownClu     = MCTruthTTClusterHandle->isUnknown( tempCluRef );
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
			//bool combinStub     = MCTruthTTStubHandle->isCombinatoric( tempStubRef );
			//bool unknownStub    = MCTruthTTStubHandle->isUnknown( tempStubRef );
			int partStub         = 999999999;
			if ( genuineStub )
			{
				edm::Ptr< TrackingParticle > thisTP = MCTruthTTStubHandle->findTrackingParticlePtr( tempStubRef );
				partStub = thisTP->pdgId();
			}
			
			Stub_PID->Fill( partStub );
			
			/// Store Track information in maps, skip if the Cluster is not good
			if ( !genuineStub ) continue;
			
			edm::Ptr< TrackingParticle > tpPtr = MCTruthTTStubHandle->findTrackingParticlePtr( tempStubRef );
			
			/// Get the corresponding vertex and reject the track
			/// if its vertex is outside the beampipe
			if ( tpPtr->vertex().rho() >= 2.0 )
				continue;
			
			/// Compare to TrackingParticle
			
			if ( tpPtr.isNull() ) continue; /// This prevents to fill the vector if the TrackingParticle is not found
			TrackingParticle thisTP = *tpPtr;
			
			double simPt = thisTP.p4().pt();
			double simEta = thisTP.momentum().eta();
			double simPhi = thisTP.momentum().phi();
// 			double recPt = theStackedGeometry->findRoughPt( mMagneticFieldStrength, &(*tempStubRef) );
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
				Stub_Eta_TPart_Eta_AllLayers->Fill( simEta, recEta );
				Stub_Phi_TPart_Phi_AllLayers->Fill( simPhi, recPhi );
				
				Stub_EtaRes_TPart_Eta_AllLayers->Fill( simEta, recEta - simEta );
				Stub_PhiRes_TPart_Eta_AllLayers->Fill( simEta, recPhi - simPhi );
				
				Stub_W_TPart_Pt_AllLayers->Fill( simPt, displStub - offsetStub );
				Stub_W_TPart_InvPt_AllLayers->Fill( 1./simPt, displStub - offsetStub );
			}
			else if ( detIdStub.isEndcap() )
			{
				Stub_Eta_TPart_Eta_AllDisks->Fill( simEta, recEta );
				Stub_Phi_TPart_Phi_AllDisks->Fill( simPhi, recPhi );
				
				Stub_EtaRes_TPart_Eta_AllDisks->Fill( simEta, recEta - simEta );
				Stub_PhiRes_TPart_Eta_AllDisks->Fill( simEta, recPhi - simPhi );
				
				Stub_W_TPart_Pt_AllDisks->Fill( simPt, displStub - offsetStub );
				Stub_W_TPart_InvPt_AllDisks->Fill( 1./simPt, displStub - offsetStub );
			}
			
		}
	} /// End of loop over TTStubs
	
}


// ------------ method called once each job just before starting event loop  ------------
void
OuterTrackerMCTruth::beginRun(const edm::Run& run, const edm::EventSetup& es)
{
	
	SiStripFolderOrganizer folder_organizer;
	folder_organizer.setSiStripFolderName(topFolderName_);
	folder_organizer.setSiStripFolder();
	
	dqmStore_->setCurrentFolder(topFolderName_+"/MCTruth/");
	
	/// TrackingParticle and TrackingVertex
	edm::ParameterSet psTPart_Pt =  conf_.getParameter<edm::ParameterSet>("TH1TPart_Pt");
	std::string HistoName = "TPart_Pt";
	TPart_Pt = dqmStore_->book1D(HistoName, HistoName,
																			psTPart_Pt.getParameter<int32_t>("Nbinsx"),
																			psTPart_Pt.getParameter<double>("xmin"),
																			psTPart_Pt.getParameter<double>("xmax"));
	TPart_Pt->setAxisTitle("TPart_Pt", 1);
	TPart_Pt->setAxisTitle("# TParticles", 2);
	
	edm::ParameterSet psTPart_Angle_Pt10 =  conf_.getParameter<edm::ParameterSet>("TH1TPart_Angle_Pt10");
	HistoName = "TPart_Eta_Pt10";
	TPart_Eta_Pt10 = dqmStore_->book1D(HistoName, HistoName,
																			psTPart_Angle_Pt10.getParameter<int32_t>("Nbinsx"),
																			psTPart_Angle_Pt10.getParameter<double>("xmin"),
																			psTPart_Angle_Pt10.getParameter<double>("xmax"));
	TPart_Eta_Pt10->setAxisTitle("TPart_Eta_Pt10", 1);
	TPart_Eta_Pt10->setAxisTitle("# TParticles", 2);
	
	HistoName = "TPart_Phi_Pt10";
	TPart_Phi_Pt10 = dqmStore_->book1D(HistoName, HistoName,
																			psTPart_Angle_Pt10.getParameter<int32_t>("Nbinsx"),
																			psTPart_Angle_Pt10.getParameter<double>("xmin"),
																			psTPart_Angle_Pt10.getParameter<double>("xmax"));
	TPart_Phi_Pt10->setAxisTitle("TPart_Phi_Pt10", 1);
	TPart_Phi_Pt10->setAxisTitle("# TParticles", 2);
	
	edm::ParameterSet psSimVtx_XY =  conf_.getParameter<edm::ParameterSet>("TH2SimVtx_XY");
	HistoName = "SimVtx_XY";
	SimVtx_XY = dqmStore_->book2D(HistoName, HistoName,
																			psSimVtx_XY.getParameter<int32_t>("Nbinsx"),
																			psSimVtx_XY.getParameter<double>("xmin"),
																			psSimVtx_XY.getParameter<double>("xmax"),
																			psSimVtx_XY.getParameter<int32_t>("Nbinsy"),
																			psSimVtx_XY.getParameter<double>("ymin"),
																			psSimVtx_XY.getParameter<double>("ymax"));
	SimVtx_XY->setAxisTitle("SimVtx x", 1);
	SimVtx_XY->setAxisTitle("SimVtx y", 2);
	
	edm::ParameterSet psSimVtx_RZ =  conf_.getParameter<edm::ParameterSet>("TH2SimVtx_RZ");
	HistoName = "SimVtx_RZ";
	SimVtx_RZ = dqmStore_->book2D(HistoName, HistoName,
																			psSimVtx_RZ.getParameter<int32_t>("Nbinsx"),
																			psSimVtx_RZ.getParameter<double>("xmin"),
																			psSimVtx_RZ.getParameter<double>("xmax"),
																			psSimVtx_RZ.getParameter<int32_t>("Nbinsy"),
																			psSimVtx_RZ.getParameter<double>("ymin"),
																			psSimVtx_RZ.getParameter<double>("ymax"));
	SimVtx_RZ->setAxisTitle("SimVtx z", 1);
	SimVtx_RZ->setAxisTitle("SimVtx #rho", 2);
	
	
	/// Eta distribution of Tracking Particles (Cluster width)
	// Inner
	edm::ParameterSet psTPart_Eta_CW =  conf_.getParameter<edm::ParameterSet>("TH1TPart_Eta_CW");
	HistoName = "TPart_Eta_ICW_1";
	TPart_Eta_ICW_1 = dqmStore_->book1D(HistoName, HistoName,
																			psTPart_Eta_CW.getParameter<int32_t>("Nbinsx"),
																			psTPart_Eta_CW.getParameter<double>("xmin"),
																			psTPart_Eta_CW.getParameter<double>("xmax"));
	TPart_Eta_ICW_1->setAxisTitle("TPart_Eta_ICW_1", 1);
	TPart_Eta_ICW_1->setAxisTitle("# TParticles", 2);
	
	HistoName = "TPart_Eta_ICW_2";
	TPart_Eta_ICW_2 = dqmStore_->book1D(HistoName, HistoName,
																			psTPart_Eta_CW.getParameter<int32_t>("Nbinsx"),
																			psTPart_Eta_CW.getParameter<double>("xmin"),
																			psTPart_Eta_CW.getParameter<double>("xmax"));
	TPart_Eta_ICW_2->setAxisTitle("TPart_Eta_ICW_2", 1);
	TPart_Eta_ICW_2->setAxisTitle("# TParticles", 2);
	
	HistoName = "TPart_Eta_ICW_3";
	TPart_Eta_ICW_3 = dqmStore_->book1D(HistoName, HistoName,
																			psTPart_Eta_CW.getParameter<int32_t>("Nbinsx"),
																			psTPart_Eta_CW.getParameter<double>("xmin"),
																			psTPart_Eta_CW.getParameter<double>("xmax"));
	TPart_Eta_ICW_3->setAxisTitle("TPart_Eta_ICW_3", 1);
	TPart_Eta_ICW_3->setAxisTitle("# TParticles", 2);
	
	HistoName = "TPart_Eta_INormalization";
	TPart_Eta_INormalization = dqmStore_->book1D(HistoName, HistoName,
																							 psTPart_Eta_CW.getParameter<int32_t>("Nbinsx"),
																							 psTPart_Eta_CW.getParameter<double>("xmin"),
																							 psTPart_Eta_CW.getParameter<double>("xmax"));
	TPart_Eta_INormalization->setAxisTitle("TPart_Eta_INormalization", 1);
	TPart_Eta_INormalization->setAxisTitle("# TParticles", 2);
	
	
	// Outer
	HistoName = "TPart_Eta_OCW_1";
	TPart_Eta_OCW_1 = dqmStore_->book1D(HistoName, HistoName,
																			psTPart_Eta_CW.getParameter<int32_t>("Nbinsx"),
																			psTPart_Eta_CW.getParameter<double>("xmin"),
																			psTPart_Eta_CW.getParameter<double>("xmax"));
	TPart_Eta_OCW_1->setAxisTitle("TPart_Eta_OCW_1", 1);
	TPart_Eta_OCW_1->setAxisTitle("# TParticles", 2);
	
	HistoName = "TPart_Eta_OCW_2";
	TPart_Eta_OCW_2 = dqmStore_->book1D(HistoName, HistoName,
																			psTPart_Eta_CW.getParameter<int32_t>("Nbinsx"),
																			psTPart_Eta_CW.getParameter<double>("xmin"),
																			psTPart_Eta_CW.getParameter<double>("xmax"));
	TPart_Eta_OCW_2->setAxisTitle("TPart_Eta_OCW_2", 1);
	TPart_Eta_OCW_2->setAxisTitle("# TParticles", 2);
	
	HistoName = "TPart_Eta_OCW_3";
	TPart_Eta_OCW_3 = dqmStore_->book1D(HistoName, HistoName,
																			psTPart_Eta_CW.getParameter<int32_t>("Nbinsx"),
																			psTPart_Eta_CW.getParameter<double>("xmin"),
																			psTPart_Eta_CW.getParameter<double>("xmax"));
	TPart_Eta_OCW_3->setAxisTitle("TPart_Eta_OCW_3", 1);
	TPart_Eta_OCW_3->setAxisTitle("# TParticles", 2);
	
	HistoName = "TPart_Eta_ONormalization";
	TPart_Eta_ONormalization = dqmStore_->book1D(HistoName, HistoName,
																							 psTPart_Eta_CW.getParameter<int32_t>("Nbinsx"),
																							 psTPart_Eta_CW.getParameter<double>("xmin"),
																							 psTPart_Eta_CW.getParameter<double>("xmax"));
	TPart_Eta_ONormalization->setAxisTitle("TPart_Eta_ONormalization", 1);
	TPart_Eta_ONormalization->setAxisTitle("# TParticles", 2);
	
	
	
	/// Eta distribution of Tracking Particles (Stubs in PS/2S modules)
	edm::ParameterSet psTPart_Eta_PS2S =  conf_.getParameter<edm::ParameterSet>("TH1TPart_Eta_PS2S");
	HistoName = "TPart_Eta_Pt10_Normalization";
	TPart_Eta_Pt10_Normalization = dqmStore_->book1D(HistoName, HistoName,
																			psTPart_Eta_PS2S.getParameter<int32_t>("Nbinsx"),
																			psTPart_Eta_PS2S.getParameter<double>("xmin"),
																			psTPart_Eta_PS2S.getParameter<double>("xmax"));
	TPart_Eta_Pt10_Normalization->setAxisTitle("TPart_Eta_Pt10_Normalization", 1);
	TPart_Eta_Pt10_Normalization->setAxisTitle("Average nb. of Stubs", 2);
	
	HistoName = "TPart_Eta_Pt10_NumPS";
	TPart_Eta_Pt10_NumPS = dqmStore_->book1D(HistoName, HistoName,
																			psTPart_Eta_PS2S.getParameter<int32_t>("Nbinsx"),
																			psTPart_Eta_PS2S.getParameter<double>("xmin"),
																			psTPart_Eta_PS2S.getParameter<double>("xmax"));
	TPart_Eta_Pt10_NumPS->setAxisTitle("TPart_Eta_Pt10_NumPS", 1);
	TPart_Eta_Pt10_NumPS->setAxisTitle("Average nb. of Stubs", 2);
	
	HistoName = "TPart_Eta_Pt10_Num2S";
	TPart_Eta_Pt10_Num2S = dqmStore_->book1D(HistoName, HistoName,
																			psTPart_Eta_PS2S.getParameter<int32_t>("Nbinsx"),
																			psTPart_Eta_PS2S.getParameter<double>("xmin"),
																			psTPart_Eta_PS2S.getParameter<double>("xmax"));
	TPart_Eta_Pt10_Num2S->setAxisTitle("TPart_Eta_Pt10_Num2S", 1);
	TPart_Eta_Pt10_Num2S->setAxisTitle("Average nb. of Stubs", 2);
	
	
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
	Cluster_PID->setAxisTitle("TTCluster pdgID", 1);
	Cluster_PID->setAxisTitle("Stack Member", 2);

	edm::ParameterSet psStub_PID =  conf_.getParameter<edm::ParameterSet>("TH1Stub_PID");
	HistoName = "Stub_PID";
	Stub_PID = dqmStore_->book1D(HistoName, HistoName,
																	psStub_PID.getParameter<int32_t>("Nbinsx"),
																	psStub_PID.getParameter<double>("xmin"),
																	psStub_PID.getParameter<double>("xmax"));
	Stub_PID->setAxisTitle("TTStub pdgID", 1);
	Stub_PID->setAxisTitle("# TTStubs", 2);
	
	
	/// Stub properties compared to TParticles
	dqmStore_->setCurrentFolder(topFolderName_+"/TStubVSTPart/");

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
	Stub_Eta_TPart_Eta_AllLayers->setAxisTitle("TPart Eta", 1);
	Stub_Eta_TPart_Eta_AllLayers->setAxisTitle("Stub Eta", 2);

	HistoName = "Stub_Eta_TPart_Eta_AllDisks";
	Stub_Eta_TPart_Eta_AllDisks = dqmStore_->book2D(HistoName, HistoName,
																									 psStub_Eta.getParameter<int32_t>("Nbinsx"),
																									 psStub_Eta.getParameter<double>("xmin"),
																									 psStub_Eta.getParameter<double>("xmax"),
																									 psStub_Eta.getParameter<int32_t>("Nbinsy"),
																									 psStub_Eta.getParameter<double>("ymin"),
																									 psStub_Eta.getParameter<double>("ymax"));
	Stub_Eta_TPart_Eta_AllDisks->setAxisTitle("TPart Eta", 1);
	Stub_Eta_TPart_Eta_AllDisks->setAxisTitle("Stub Eta", 2);

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
	Stub_Phi_TPart_Phi_AllLayers->setAxisTitle("TPart Phi", 1);
	Stub_Phi_TPart_Phi_AllLayers->setAxisTitle("Stub Phi", 2);

	HistoName = "Stub_Phi_TPart_Phi_AllDisks";
	Stub_Phi_TPart_Phi_AllDisks = dqmStore_->book2D(HistoName, HistoName,
																									 psStub_Phi.getParameter<int32_t>("Nbinsx"),
																									 psStub_Phi.getParameter<double>("xmin"),
																									 psStub_Phi.getParameter<double>("xmax"),
																									 psStub_Phi.getParameter<int32_t>("Nbinsy"),
																									 psStub_Phi.getParameter<double>("ymin"),
																									 psStub_Phi.getParameter<double>("ymax"));
	Stub_Phi_TPart_Phi_AllDisks->setAxisTitle("TPart Phi", 1);
	Stub_Phi_TPart_Phi_AllDisks->setAxisTitle("Stub Phi", 2);
	
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
	Stub_EtaRes_TPart_Eta_AllLayers->setAxisTitle("TPart Eta", 1);
	Stub_EtaRes_TPart_Eta_AllLayers->setAxisTitle("Stub Eta - TPart Eta", 2);

	HistoName = "Stub_EtaRes_TPart_Eta_AllDisks";
	Stub_EtaRes_TPart_Eta_AllDisks = dqmStore_->book2D(HistoName, HistoName,
																									psStub_EtaRes.getParameter<int32_t>("Nbinsx"),
																									psStub_EtaRes.getParameter<double>("xmin"),
																									psStub_EtaRes.getParameter<double>("xmax"),
																									psStub_EtaRes.getParameter<int32_t>("Nbinsy"),
																									psStub_EtaRes.getParameter<double>("ymin"),
																									psStub_EtaRes.getParameter<double>("ymax"));
	Stub_EtaRes_TPart_Eta_AllDisks->setAxisTitle("TPart Eta", 1);
	Stub_EtaRes_TPart_Eta_AllDisks->setAxisTitle("Stub Eta - TPart Eta", 2);

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
	Stub_PhiRes_TPart_Eta_AllLayers->setAxisTitle("Stub Phi - TPart Phi", 2);

	HistoName = "Stub_PhiRes_TPart_Eta_AllDisks";
	Stub_PhiRes_TPart_Eta_AllDisks = dqmStore_->book2D(HistoName, HistoName,
																									psStub_PhiRes.getParameter<int32_t>("Nbinsx"),
																									psStub_PhiRes.getParameter<double>("xmin"),
																									psStub_PhiRes.getParameter<double>("xmax"),
																									psStub_PhiRes.getParameter<int32_t>("Nbinsy"),
																									psStub_PhiRes.getParameter<double>("ymin"),
																									psStub_PhiRes.getParameter<double>("ymax"));
	Stub_PhiRes_TPart_Eta_AllDisks->setAxisTitle("TPart Eta", 1);
	Stub_PhiRes_TPart_Eta_AllDisks->setAxisTitle("Stub Phi - TPart Phi", 2);

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
	Stub_W_TPart_InvPt_AllLayers->setAxisTitle("TPart 1/Pt", 1);
	Stub_W_TPart_InvPt_AllLayers->setAxisTitle("Stub Width", 2);

	HistoName = "Stub_W_TPart_InvPt_AllDisks";
	Stub_W_TPart_InvPt_AllDisks = dqmStore_->book2D(HistoName, HistoName,
																											psStub_W_InvPt.getParameter<int32_t>("Nbinsx"),
																											psStub_W_InvPt.getParameter<double>("xmin"),
																											psStub_W_InvPt.getParameter<double>("xmax"),
																											psStub_W_InvPt.getParameter<int32_t>("Nbinsy"),
																											psStub_W_InvPt.getParameter<double>("ymin"),
																											psStub_W_InvPt.getParameter<double>("ymax"));
	Stub_W_TPart_InvPt_AllDisks->setAxisTitle("TPart 1/Pt", 1);
	Stub_W_TPart_InvPt_AllDisks->setAxisTitle("Stub Width", 2);

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
	Stub_W_TPart_Pt_AllLayers->setAxisTitle("TPart Pt", 1);
	Stub_W_TPart_Pt_AllLayers->setAxisTitle("Stub Width", 2);

	HistoName = "Stub_W_TPart_Pt_AllDisks";
	Stub_W_TPart_Pt_AllDisks = dqmStore_->book2D(HistoName, HistoName,
																								psStub_W_Pt.getParameter<int32_t>("Nbinsx"),
																								psStub_W_Pt.getParameter<double>("xmin"),
																								psStub_W_Pt.getParameter<double>("xmax"),
																								psStub_W_Pt.getParameter<int32_t>("Nbinsy"),
																								psStub_W_Pt.getParameter<double>("ymin"),
																								psStub_W_Pt.getParameter<double>("ymax"));
	Stub_W_TPart_Pt_AllDisks->setAxisTitle("TPart Pt", 1);
	Stub_W_TPart_Pt_AllDisks->setAxisTitle("Stub Width", 2);

	
}//end of method


// ------------ method called once each job just after ending the event loop  ------------
void 
OuterTrackerMCTruth::endJob(void) 
{
	
}

//define this as a plug-in
DEFINE_FWK_MODULE(OuterTrackerMCTruth);
