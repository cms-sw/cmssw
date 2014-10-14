// -*- C++ -*-
//
// Package:    OuterTrackerClusters
// Class:      OuterTrackerClusters
// 
/**\class OuterTrackerClusters OuterTrackerClusters.cc Validation/OuterTrackerClusters/plugins/OuterTrackerClusters.cc

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

#include "CalibTracker/Records/interface/SiStripDetCablingRcd.h"
#include "CalibFormats/SiStripObjects/interface/SiStripDetCabling.h"
#include "CondFormats/DataRecord/interface/SiStripCondDataRecords.h"
#include "CondFormats/SiStripObjects/interface/SiStripNoises.h"
#include "CalibFormats/SiStripObjects/interface/SiStripGain.h"
#include "CalibFormats/SiStripObjects/interface/SiStripQuality.h"
#include "DataFormats/SiStripCluster/interface/SiStripClusterCollection.h"
#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/Common/interface/DetSetVectorNew.h"
#include "DataFormats/SiStripCluster/interface/SiStripCluster.h"
#include "DQM/SiStripCommon/interface/SiStripFolderOrganizer.h"
#include "DQM/SiStripCommon/interface/SiStripHistoId.h"
#include "Validation/OuterTrackerClusters/interface/OuterTrackerClusters.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "DataFormats/SiStripDetId/interface/StripSubdetector.h"
#include "DataFormats/SiStripDetId/interface/SiStripSubStructure.h"
#include "CalibTracker/SiStripCommon/interface/SiStripDCSStatus.h"
#include "DataFormats/SiPixelCluster/interface/SiPixelCluster.h"

#include "DPGAnalysis/SiStripTools/interface/APVCyclePhaseCollection.h"
#include "DPGAnalysis/SiStripTools/interface/EventWithHistory.h"

#include "CommonTools/TriggerUtils/interface/GenericTriggerEventFlag.h"

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
OuterTrackerClusters::OuterTrackerClusters(const edm::ParameterSet& iConfig)
: dqmStore_(edm::Service<DQMStore>().operator->()), conf_(iConfig)

{
  clusterProducerStrip_ = conf_.getParameter<edm::InputTag>("ClusterProducerStrip");
  topFolderName_ = conf_.getParameter<std::string>("TopFolderName");
  
}


OuterTrackerClusters::~OuterTrackerClusters()
{
	
	// do anything here that needs to be done at desctruction time
	// (e.g. close files, deallocate resources etc.)
	
}


//
// member functions
//

// ------------ method called for each event  ------------
void
OuterTrackerClusters::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
	
	/// Geometry handles etc
  edm::ESHandle< TrackerGeometry >                GeometryHandle;
	
	/// Geometry setup
  /// Set pointers to Geometry
  iSetup.get< TrackerDigiGeometryRecord >().get(GeometryHandle);
	
	/// Track Trigger
	edm::Handle< edmNew::DetSetVector< TTCluster< Ref_PixelDigi_ > > > PixelDigiTTClusterHandle;	// same for stubs
	iEvent.getByLabel( "TTClustersFromPixelDigis", "ClusterInclusive", PixelDigiTTClusterHandle );
	/// Track Trigger MC Truth
	edm::Handle< TTClusterAssociationMap< Ref_PixelDigi_ > > MCTruthTTClusterHandle;
	iEvent.getByLabel( "TTClusterAssociatorFromPixelDigis", "ClusterInclusive", MCTruthTTClusterHandle );
		
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

			StackedTrackerDetId detIdClu( tempCluRef->getDetId() );		// find it!
			bool genuineClu     = MCTruthTTClusterHandle->isGenuine( tempCluRef );
			bool combinClu      = MCTruthTTClusterHandle->isCombinatoric( tempCluRef );
			//bool unknownClu     = MCTruthTTClusterHandle->isUnknown( tempCluRef );
			//int partClu         = 999999999;
			if ( genuineClu )
			{
				edm::Ptr< TrackingParticle > thisTP = MCTruthTTClusterHandle->findTrackingParticlePtr( tempCluRef );
				//partClu = thisTP->pdgId();
			}

			if ( detIdClu.isBarrel() )
			{
				if ( genuineClu )
				{
					Cluster_Gen_Barrel->Fill( detIdClu.iLayer() );
				}
				else if ( combinClu )
				{
					Cluster_Comb_Barrel->Fill( detIdClu.iLayer() );
				}
				else
				{
					Cluster_Unkn_Barrel->Fill( detIdClu.iLayer() );
				}

			}	// end if isBarrel()
			else if ( detIdClu.isEndcap() )
			{
				if ( genuineClu )
				{
					Cluster_Gen_Endcap->Fill( detIdClu.iDisk() );
				}
				else if ( combinClu )
				{
					Cluster_Comb_Endcap->Fill( detIdClu.iDisk() );
				}
				else
				{
					Cluster_Unkn_Endcap->Fill( detIdClu.iDisk() );
				}

			}	// end if isEndcap()
		}	// end loop contentIter
	}	// end loop inputIter
	
}


// ------------ method called once each job just before starting event loop  ------------
void
OuterTrackerClusters::beginRun(const edm::Run& run, const edm::EventSetup& es)
{
	
	SiStripFolderOrganizer folder_organizer;
	folder_organizer.setSiStripFolderName(topFolderName_);
	folder_organizer.setSiStripFolder();
	
	
	dqmStore_->setCurrentFolder(topFolderName_+"/Clusters/");
	
	/// TTCluster stacks
	edm::ParameterSet psTTClusterStacks =  conf_.getParameter<edm::ParameterSet>("TH1TTCluster_Stack");
	std::string HistoName = "Cluster_IMem_Barrel";
	Cluster_IMem_Barrel = dqmStore_->book1D(HistoName, HistoName,
																				psTTClusterStacks.getParameter<int32_t>("Nbinsx"),
																				psTTClusterStacks.getParameter<double>("xmin"),
																				psTTClusterStacks.getParameter<double>("xmax"));
	Cluster_IMem_Barrel->setAxisTitle("Inner TTCluster Stack", 1);
	Cluster_IMem_Barrel->setAxisTitle("Number of Events", 2);
	
	HistoName = "Cluster_IMem_Endcap";
	Cluster_IMem_Endcap = dqmStore_->book1D(HistoName, HistoName,
																				psTTClusterStacks.getParameter<int32_t>("Nbinsx"),
																				psTTClusterStacks.getParameter<double>("xmin"),
																				psTTClusterStacks.getParameter<double>("xmax"));
	Cluster_IMem_Endcap->setAxisTitle("Inner TTCluster Stack", 1);
	Cluster_IMem_Endcap->setAxisTitle("Number of Events", 2);
	
	HistoName = "Cluster_OMem_Barrel";
	Cluster_OMem_Barrel = dqmStore_->book1D(HistoName, HistoName,
																				psTTClusterStacks.getParameter<int32_t>("Nbinsx"),
																				psTTClusterStacks.getParameter<double>("xmin"),
																				psTTClusterStacks.getParameter<double>("xmax"));
	Cluster_OMem_Barrel->setAxisTitle("Outer TTCluster Stack", 1);
	Cluster_OMem_Barrel->setAxisTitle("Number of Events", 2);
	
	HistoName = "Cluster_OMem_Endcap";
	Cluster_OMem_Endcap = dqmStore_->book1D(HistoName, HistoName,
																				psTTClusterStacks.getParameter<int32_t>("Nbinsx"),
																				psTTClusterStacks.getParameter<double>("xmin"),
																				psTTClusterStacks.getParameter<double>("xmax"));
	Cluster_OMem_Endcap->setAxisTitle("Outer TTCluster Stack", 1);
	Cluster_OMem_Endcap->setAxisTitle("Number of Events", 2);
	
	HistoName = "Cluster_Gen_Barrel";
	Cluster_Gen_Barrel = dqmStore_->book1D(HistoName, HistoName,
																				psTTClusterStacks.getParameter<int32_t>("Nbinsx"),
																				psTTClusterStacks.getParameter<double>("xmin"),
																				psTTClusterStacks.getParameter<double>("xmax"));
	Cluster_Gen_Barrel->setAxisTitle("Genuine TTCluster Stack", 1);
	Cluster_Gen_Barrel->setAxisTitle("Number of Events", 2);
	
	HistoName = "Cluster_Unkn_Barrel";
	Cluster_Unkn_Barrel = dqmStore_->book1D(HistoName, HistoName,
																				psTTClusterStacks.getParameter<int32_t>("Nbinsx"),
																				psTTClusterStacks.getParameter<double>("xmin"),
																				psTTClusterStacks.getParameter<double>("xmax"));
	Cluster_Unkn_Barrel->setAxisTitle("Unknown TTCluster Stack", 1);
	Cluster_Unkn_Barrel->setAxisTitle("Number of Events", 2);
	
	HistoName = "Cluster_Comb_Barrel";
	Cluster_Comb_Barrel = dqmStore_->book1D(HistoName, HistoName,
																				psTTClusterStacks.getParameter<int32_t>("Nbinsx"),
																				psTTClusterStacks.getParameter<double>("xmin"),
																				psTTClusterStacks.getParameter<double>("xmax"));
	Cluster_Comb_Barrel->setAxisTitle("Combinatorial TTCluster Stack", 1);
	Cluster_Comb_Barrel->setAxisTitle("Number of Events", 2);
	
	HistoName = "Cluster_Gen_Endcap";
	Cluster_Gen_Endcap = dqmStore_->book1D(HistoName, HistoName,
																				psTTClusterStacks.getParameter<int32_t>("Nbinsx"),
																				psTTClusterStacks.getParameter<double>("xmin"),
																				psTTClusterStacks.getParameter<double>("xmax"));
	Cluster_Gen_Endcap->setAxisTitle("Genuine TTCluster Stack", 1);
	Cluster_Gen_Endcap->setAxisTitle("Number of Events", 2);
	
	HistoName = "Cluster_Unkn_Endcap";
	Cluster_Unkn_Endcap = dqmStore_->book1D(HistoName, HistoName,
																				psTTClusterStacks.getParameter<int32_t>("Nbinsx"),
																				psTTClusterStacks.getParameter<double>("xmin"),
																				psTTClusterStacks.getParameter<double>("xmax"));
	Cluster_Unkn_Endcap->setAxisTitle("Unknown TTCluster Stack", 1);
	Cluster_Unkn_Endcap->setAxisTitle("Number of Events", 2);
	
	HistoName = "Cluster_Comb_Endcap";
	Cluster_Comb_Endcap = dqmStore_->book1D(HistoName, HistoName,
																				psTTClusterStacks.getParameter<int32_t>("Nbinsx"),
																				psTTClusterStacks.getParameter<double>("xmin"),
																				psTTClusterStacks.getParameter<double>("xmax"));
	Cluster_Comb_Endcap->setAxisTitle("Combinatorial TTCluster Stack", 1);
	Cluster_Comb_Endcap->setAxisTitle("Number of Events", 2);
	
}//end of method


// ------------ method called once each job just after ending the event loop  ------------
void 
OuterTrackerClusters::endJob(void) 
{
	
}

//define this as a plug-in
DEFINE_FWK_MODULE(OuterTrackerClusters);
