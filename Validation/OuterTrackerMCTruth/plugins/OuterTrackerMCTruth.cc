// -*- C++ -*-
//
// Package:    OuterTrackerMCTruth
// Class:      OuterTrackerMCTruth
// 
/**\class OuterTrackerMCTruth OuterTrackerMCTruth.cc Validation/OuterTrackerMCTruth/plugins/OuterTrackerMCTruth.cc

 Description: [one line class summary]

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  Isabelle Helena J De Bruyn
//         Created:  Wed, 17 Sep 2014 12:33:30 GMT
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
#include "Validation/OuterTrackerMCTruth/interface/OuterTrackerMCTruth.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "DataFormats/SiPixelCluster/interface/SiPixelCluster.h"

// For TPart_Eta_ICW_1 (TrackingParticles)
#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticle.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingVertex.h"
#include "SimTracker/TrackTriggerAssociation/interface/TTClusterAssociationMap.h"
#include "DataFormats/L1TrackTrigger/interface/TTTypes.h"
#include "DataFormats/L1TrackTrigger/interface/TTCluster.h"
#include "Geometry/CommonTopologies/interface/PixelTopology.h"
#include "Geometry/TrackerGeometryBuilder/interface/PixelGeomDetUnit.h"
// For TPart_Eta_Pt10_PS
#include "SimTracker/TrackTriggerAssociation/interface/TTStubAssociationMap.h"
#include "Geometry/TrackerGeometryBuilder/interface/StackedTrackerGeometry.h"
#include "Geometry/Records/interface/StackedTrackerGeometryRecord.h"



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
	edm::Handle< edmNew::DetSetVector< TTCluster< Ref_PixelDigi_ > > > PixelDigiTTClusterHandle;	// same for stubs
	iEvent.getByLabel( "TTClustersFromPixelDigis", "ClusterInclusive", PixelDigiTTClusterHandle );
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
							TPart_AbsEta_INormalization->Fill( fabs( tempTPPtr->momentum().eta() ) );
							TPart_Eta_INormalization->Fill( tempTPPtr->momentum().eta() );
							normIClu = true;
						}
						
						if ( clusterWidth == 1 )
						{
							TPart_AbsEta_ICW_1->Fill( fabs( tempTPPtr->momentum().eta() ) );
							TPart_Eta_ICW_1->Fill( tempTPPtr->momentum().eta() );
						}
						else if ( clusterWidth == 2 )
						{
							TPart_AbsEta_ICW_2->Fill( fabs( tempTPPtr->momentum().eta() ) );
							TPart_Eta_ICW_2->Fill( tempTPPtr->momentum().eta() );
						}
						else
						{
							TPart_AbsEta_ICW_3->Fill( fabs( tempTPPtr->momentum().eta() ) );
							TPart_Eta_ICW_3->Fill( tempTPPtr->momentum().eta() );
						}
					}
					if ( stackMember == 1 )
					{
						if ( normOClu == false )
            {
							TPart_AbsEta_ONormalization->Fill( fabs( tempTPPtr->momentum().eta() ) );
							TPart_Eta_ONormalization->Fill( tempTPPtr->momentum().eta() );
							normOClu = true;
						}
						
						if ( clusterWidth == 1 )
						{
							TPart_AbsEta_OCW_1->Fill( fabs( tempTPPtr->momentum().eta() ) );
							TPart_Eta_OCW_1->Fill( tempTPPtr->momentum().eta() );
						}
						else if ( clusterWidth == 2 )
						{
							TPart_AbsEta_OCW_2->Fill( fabs( tempTPPtr->momentum().eta() ) );
							TPart_Eta_OCW_2->Fill( tempTPPtr->momentum().eta() );
						}
						else
						{
							TPart_AbsEta_OCW_3->Fill( fabs( tempTPPtr->momentum().eta() ) );
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
						TPart_AbsEta_Pt10_Normalization->Fill( fabs( tempTPPtr->momentum().eta() ) );
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
						TPart_AbsEta_Pt10_Num2S->Fill( fabs( tempTPPtr->momentum().eta() ) );
						TPart_Eta_Pt10_Num2S->Fill( tempTPPtr->momentum().eta() );
					}
					else /// PS
					{
						TPart_AbsEta_Pt10_NumPS->Fill( fabs( tempTPPtr->momentum().eta() ) );
						TPart_Eta_Pt10_NumPS->Fill( tempTPPtr->momentum().eta() );
					}
				} /// End of loop over the Stubs generated by this TrackingParticle
			}
		}	// end loop TrackingParticles
  } // end if there are TrackingParticles
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
	TPart_Pt->setAxisTitle("Number of Events", 2);
	
	edm::ParameterSet psTPart_Angle_Pt10 =  conf_.getParameter<edm::ParameterSet>("TH1TPart_Angle_Pt10");
	HistoName = "TPart_Eta_Pt10";
	TPart_Eta_Pt10 = dqmStore_->book1D(HistoName, HistoName,
																			psTPart_Angle_Pt10.getParameter<int32_t>("Nbinsx"),
																			psTPart_Angle_Pt10.getParameter<double>("xmin"),
																			psTPart_Angle_Pt10.getParameter<double>("xmax"));
	TPart_Eta_Pt10->setAxisTitle("TPart_Eta_Pt10", 1);
	TPart_Eta_Pt10->setAxisTitle("Number of Events", 2);
	
	HistoName = "TPart_Phi_Pt10";
	TPart_Phi_Pt10 = dqmStore_->book1D(HistoName, HistoName,
																			psTPart_Angle_Pt10.getParameter<int32_t>("Nbinsx"),
																			psTPart_Angle_Pt10.getParameter<double>("xmin"),
																			psTPart_Angle_Pt10.getParameter<double>("xmax"));
	TPart_Phi_Pt10->setAxisTitle("TPart_Phi_Pt10", 1);
	TPart_Phi_Pt10->setAxisTitle("Number of Events", 2);
	
	edm::ParameterSet psSimVtx_XY =  conf_.getParameter<edm::ParameterSet>("TH1SimVtx_XY");
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
	
	edm::ParameterSet psSimVtx_RZ =  conf_.getParameter<edm::ParameterSet>("TH1SimVtx_RZ");
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
	TPart_Eta_ICW_1->setAxisTitle("Number of Events", 2);
	
	HistoName = "TPart_Eta_ICW_2";
	TPart_Eta_ICW_2 = dqmStore_->book1D(HistoName, HistoName,
																			psTPart_Eta_CW.getParameter<int32_t>("Nbinsx"),
																			psTPart_Eta_CW.getParameter<double>("xmin"),
																			psTPart_Eta_CW.getParameter<double>("xmax"));
	TPart_Eta_ICW_2->setAxisTitle("TPart_Eta_ICW_2", 1);
	TPart_Eta_ICW_2->setAxisTitle("Number of Events", 2);
	
	HistoName = "TPart_Eta_ICW_3";
	TPart_Eta_ICW_3 = dqmStore_->book1D(HistoName, HistoName,
																			psTPart_Eta_CW.getParameter<int32_t>("Nbinsx"),
																			psTPart_Eta_CW.getParameter<double>("xmin"),
																			psTPart_Eta_CW.getParameter<double>("xmax"));
	TPart_Eta_ICW_3->setAxisTitle("TPart_Eta_ICW_3", 1);
	TPart_Eta_ICW_3->setAxisTitle("Number of Events", 2);
	
	HistoName = "TPart_Eta_INormalization";
	TPart_Eta_INormalization = dqmStore_->book1D(HistoName, HistoName,
																							 psTPart_Eta_CW.getParameter<int32_t>("Nbinsx"),
																							 psTPart_Eta_CW.getParameter<double>("xmin"),
																							 psTPart_Eta_CW.getParameter<double>("xmax"));
	TPart_Eta_INormalization->setAxisTitle("TPart_Eta_INormalization", 1);
	TPart_Eta_INormalization->setAxisTitle("Number of Events", 2);
	
	
	edm::ParameterSet psTPart_AbsEta_CW =  conf_.getParameter<edm::ParameterSet>("TH1TPart_AbsEta_CW");
	HistoName = "TPart_AbsEta_ICW_1";
	TPart_AbsEta_ICW_1 = dqmStore_->book1D(HistoName, HistoName,
																			psTPart_AbsEta_CW.getParameter<int32_t>("Nbinsx"),
																			psTPart_AbsEta_CW.getParameter<double>("xmin"),
																			psTPart_AbsEta_CW.getParameter<double>("xmax"));
	TPart_AbsEta_ICW_1->setAxisTitle("TPart_AbsEta_ICW_1", 1);
	TPart_AbsEta_ICW_1->setAxisTitle("Number of Events", 2);
	
	HistoName = "TPart_AbsEta_ICW_2";
	TPart_AbsEta_ICW_2 = dqmStore_->book1D(HistoName, HistoName,
																			psTPart_AbsEta_CW.getParameter<int32_t>("Nbinsx"),
																			psTPart_AbsEta_CW.getParameter<double>("xmin"),
																			psTPart_AbsEta_CW.getParameter<double>("xmax"));
	TPart_AbsEta_ICW_2->setAxisTitle("TPart_AbsEta_ICW_2", 1);
	TPart_AbsEta_ICW_2->setAxisTitle("Number of Events", 2);
	
	HistoName = "TPart_AbsEta_ICW_3";
	TPart_AbsEta_ICW_3 = dqmStore_->book1D(HistoName, HistoName,
																			psTPart_AbsEta_CW.getParameter<int32_t>("Nbinsx"),
																			psTPart_AbsEta_CW.getParameter<double>("xmin"),
																			psTPart_AbsEta_CW.getParameter<double>("xmax"));
	TPart_AbsEta_ICW_3->setAxisTitle("TPart_AbsEta_ICW_3", 1);
	TPart_AbsEta_ICW_3->setAxisTitle("Number of Events", 2);
	
	HistoName = "TPart_AbsEta_INormalization";
	TPart_AbsEta_INormalization = dqmStore_->book1D(HistoName, HistoName,
																							 psTPart_AbsEta_CW.getParameter<int32_t>("Nbinsx"),
																							 psTPart_AbsEta_CW.getParameter<double>("xmin"),
																							 psTPart_AbsEta_CW.getParameter<double>("xmax"));
	TPart_AbsEta_INormalization->setAxisTitle("TPart_AbsEta_INormalization", 1);
	TPart_AbsEta_INormalization->setAxisTitle("Number of Events", 2);
	
	// Outer
	HistoName = "TPart_Eta_OCW_1";
	TPart_Eta_OCW_1 = dqmStore_->book1D(HistoName, HistoName,
																			psTPart_Eta_CW.getParameter<int32_t>("Nbinsx"),
																			psTPart_Eta_CW.getParameter<double>("xmin"),
																			psTPart_Eta_CW.getParameter<double>("xmax"));
	TPart_Eta_OCW_1->setAxisTitle("TPart_Eta_OCW_1", 1);
	TPart_Eta_OCW_1->setAxisTitle("Number of Events", 2);
	
	HistoName = "TPart_Eta_OCW_2";
	TPart_Eta_OCW_2 = dqmStore_->book1D(HistoName, HistoName,
																			psTPart_Eta_CW.getParameter<int32_t>("Nbinsx"),
																			psTPart_Eta_CW.getParameter<double>("xmin"),
																			psTPart_Eta_CW.getParameter<double>("xmax"));
	TPart_Eta_OCW_2->setAxisTitle("TPart_Eta_OCW_2", 1);
	TPart_Eta_OCW_2->setAxisTitle("Number of Events", 2);
	
	HistoName = "TPart_Eta_OCW_3";
	TPart_Eta_OCW_3 = dqmStore_->book1D(HistoName, HistoName,
																			psTPart_Eta_CW.getParameter<int32_t>("Nbinsx"),
																			psTPart_Eta_CW.getParameter<double>("xmin"),
																			psTPart_Eta_CW.getParameter<double>("xmax"));
	TPart_Eta_OCW_3->setAxisTitle("TPart_Eta_OCW_3", 1);
	TPart_Eta_OCW_3->setAxisTitle("Number of Events", 2);
	
	HistoName = "TPart_Eta_ONormalization";
	TPart_Eta_ONormalization = dqmStore_->book1D(HistoName, HistoName,
																							 psTPart_Eta_CW.getParameter<int32_t>("Nbinsx"),
																							 psTPart_Eta_CW.getParameter<double>("xmin"),
																							 psTPart_Eta_CW.getParameter<double>("xmax"));
	TPart_Eta_ONormalization->setAxisTitle("TPart_Eta_ONormalization", 1);
	TPart_Eta_ONormalization->setAxisTitle("Number of Events", 2);
	
	
	HistoName = "TPart_AbsEta_OCW_1";
	TPart_AbsEta_OCW_1 = dqmStore_->book1D(HistoName, HistoName,
																			psTPart_AbsEta_CW.getParameter<int32_t>("Nbinsx"),
																			psTPart_AbsEta_CW.getParameter<double>("xmin"),
																			psTPart_AbsEta_CW.getParameter<double>("xmax"));
	TPart_AbsEta_OCW_1->setAxisTitle("TPart_AbsEta_OCW_1", 1);
	TPart_AbsEta_OCW_1->setAxisTitle("Number of Events", 2);
	
	HistoName = "TPart_AbsEta_OCW_2";
	TPart_AbsEta_OCW_2 = dqmStore_->book1D(HistoName, HistoName,
																			psTPart_AbsEta_CW.getParameter<int32_t>("Nbinsx"),
																			psTPart_AbsEta_CW.getParameter<double>("xmin"),
																			psTPart_AbsEta_CW.getParameter<double>("xmax"));
	TPart_AbsEta_OCW_2->setAxisTitle("TPart_AbsEta_OCW_2", 1);
	TPart_AbsEta_OCW_2->setAxisTitle("Number of Events", 2);
	
	HistoName = "TPart_AbsEta_OCW_3";
	TPart_AbsEta_OCW_3 = dqmStore_->book1D(HistoName, HistoName,
																			psTPart_AbsEta_CW.getParameter<int32_t>("Nbinsx"),
																			psTPart_AbsEta_CW.getParameter<double>("xmin"),
																			psTPart_AbsEta_CW.getParameter<double>("xmax"));
	TPart_AbsEta_OCW_3->setAxisTitle("TPart_AbsEta_OCW_3", 1);
	TPart_AbsEta_OCW_3->setAxisTitle("Number of Events", 2);
	
	HistoName = "TPart_AbsEta_ONormalization";
	TPart_AbsEta_ONormalization = dqmStore_->book1D(HistoName, HistoName,
																							 psTPart_AbsEta_CW.getParameter<int32_t>("Nbinsx"),
																							 psTPart_AbsEta_CW.getParameter<double>("xmin"),
																							 psTPart_AbsEta_CW.getParameter<double>("xmax"));
	TPart_AbsEta_ONormalization->setAxisTitle("TPart_AbsEta_ONormalization", 1);
	TPart_AbsEta_ONormalization->setAxisTitle("Number of Events", 2);
	
	
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
	
	edm::ParameterSet psTPart_AbsEta_PS2S =  conf_.getParameter<edm::ParameterSet>("TH1TPart_AbsEta_PS2S");
	HistoName = "TPart_AbsEta_Pt10_Normalization";
	TPart_AbsEta_Pt10_Normalization = dqmStore_->book1D(HistoName, HistoName,
																			psTPart_AbsEta_PS2S.getParameter<int32_t>("Nbinsx"),
																			psTPart_AbsEta_PS2S.getParameter<double>("xmin"),
																			psTPart_AbsEta_PS2S.getParameter<double>("xmax"));
	TPart_AbsEta_Pt10_Normalization->setAxisTitle("TPart_AbsEta_Pt10_Normalization", 1);
	TPart_AbsEta_Pt10_Normalization->setAxisTitle("Average nb. of Stubs", 2);
	
	HistoName = "TPart_AbsEta_Pt10_NumPS";
	TPart_AbsEta_Pt10_NumPS = dqmStore_->book1D(HistoName, HistoName,
																			psTPart_AbsEta_PS2S.getParameter<int32_t>("Nbinsx"),
																			psTPart_AbsEta_PS2S.getParameter<double>("xmin"),
																			psTPart_AbsEta_PS2S.getParameter<double>("xmax"));
	TPart_AbsEta_Pt10_NumPS->setAxisTitle("TPart_AbsEta_Pt10_NumPS", 1);
	TPart_AbsEta_Pt10_NumPS->setAxisTitle("Average nb. of Stubs", 2);
	
	HistoName = "TPart_AbsEta_Pt10_Num2S";
	TPart_AbsEta_Pt10_Num2S = dqmStore_->book1D(HistoName, HistoName,
																			psTPart_AbsEta_PS2S.getParameter<int32_t>("Nbinsx"),
																			psTPart_AbsEta_PS2S.getParameter<double>("xmin"),
																			psTPart_AbsEta_PS2S.getParameter<double>("xmax"));
	TPart_AbsEta_Pt10_Num2S->setAxisTitle("TPart_AbsEta_Pt10_Num2S", 1);
	TPart_AbsEta_Pt10_Num2S->setAxisTitle("Average nb. of Stubs", 2);
	
}//end of method


// ------------ method called once each job just after ending the event loop  ------------
void 
OuterTrackerMCTruth::endJob(void) 
{
	
}

//define this as a plug-in
DEFINE_FWK_MODULE(OuterTrackerMCTruth);
