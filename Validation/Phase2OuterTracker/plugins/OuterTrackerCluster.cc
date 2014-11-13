// -*- C++ -*-
//
// Package:    Phase2OuterTracker
// Class:      Phase2OuterTracker
// 
/**\class Phase2OuterTracker OuterTrackerCluster.cc Validation/Phase2OuterTracker/plugins/OuterTrackerCluster.cc

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
#include "Validation/Phase2OuterTracker/interface/OuterTrackerCluster.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "DataFormats/SiPixelCluster/interface/SiPixelCluster.h"

// For TrackingParticles
#include "SimTracker/TrackTriggerAssociation/interface/TTClusterAssociationMap.h"
#include "DataFormats/L1TrackTrigger/interface/TTTypes.h"
#include "DataFormats/L1TrackTrigger/interface/TTCluster.h"

#include "Geometry/TrackerGeometryBuilder/interface/StackedTrackerGeometry.h"
#include "Geometry/Records/interface/StackedTrackerGeometryRecord.h"

#include "TMath.h"
#include <iostream>

//
// constructors and destructor
//
OuterTrackerCluster::OuterTrackerCluster(const edm::ParameterSet& iConfig)
: dqmStore_(edm::Service<DQMStore>().operator->()), conf_(iConfig)

{
  topFolderName_ = conf_.getParameter<std::string>("TopFolderName");
  
}


OuterTrackerCluster::~OuterTrackerCluster()
{
	
	// do anything here that needs to be done at desctruction time
	// (e.g. close files, deallocate resources etc.)
	
}


//
// member functions
//

// ------------ method called for each event  ------------
void
OuterTrackerCluster::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
	
	/// Geometry handles etc
  edm::ESHandle< TrackerGeometry >                GeometryHandle;
	edm::ESHandle< StackedTrackerGeometry >         StackedGeometryHandle;
	const StackedTrackerGeometry*                   theStackedGeometry;
	
	/// Geometry setup
  /// Set pointers to Geometry
  iSetup.get< TrackerDigiGeometryRecord >().get(GeometryHandle);
	/// Set pointers to Stacked Modules
  iSetup.get< StackedTrackerGeometryRecord >().get(StackedGeometryHandle);
  theStackedGeometry = StackedGeometryHandle.product(); /// Note this is different from the "global" geometry
	
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

			StackedTrackerDetId detIdClu( tempCluRef->getDetId() );
			bool genuineClu     = MCTruthTTClusterHandle->isGenuine( tempCluRef );
			bool combinClu      = MCTruthTTClusterHandle->isCombinatoric( tempCluRef );
			//bool unknownClu     = MCTruthTTClusterHandle->isUnknown( tempCluRef );
			//int partClu         = 999999999;
			if ( genuineClu )
			{
				edm::Ptr< TrackingParticle > thisTP = MCTruthTTClusterHandle->findTrackingParticlePtr( tempCluRef );
				//partClu = thisTP->pdgId();
			}
			
			GlobalPoint posClu  = theStackedGeometry->findAverageGlobalPosition( &(*tempCluRef) );

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
			
			/// Eta distribution in function of genuine/combinatorial/unknown cluster
      if ( genuineClu )
      {
        Cluster_Gen_Eta->Fill( posClu.eta() );
      }
      else if ( combinClu )
      {
        Cluster_Comb_Eta->Fill( posClu.eta() );
      }
      else
      {
        Cluster_Unkn_Eta->Fill( posClu.eta() );
      }
			
		}	// end loop contentIter
	}	// end loop inputIter
	
}


// ------------ method called once each job just before starting event loop  ------------
void
OuterTrackerCluster::beginRun(const edm::Run& run, const edm::EventSetup& es)
{
	
	SiStripFolderOrganizer folder_organizer;
	folder_organizer.setSiStripFolderName(topFolderName_);
	folder_organizer.setSiStripFolder();
	
	
	dqmStore_->setCurrentFolder(topFolderName_+"/Clusters/");
	
	/// TTCluster stacks
	edm::ParameterSet psTTClusterStacks =  conf_.getParameter<edm::ParameterSet>("TH1TTCluster_Stack");
	std::string HistoName = "NClusters_Gen_Barrel";
	Cluster_Gen_Barrel = dqmStore_->book1D(HistoName, HistoName,
																				psTTClusterStacks.getParameter<int32_t>("Nbinsx"),
																				psTTClusterStacks.getParameter<double>("xmin"),
																				psTTClusterStacks.getParameter<double>("xmax"));
	Cluster_Gen_Barrel->setAxisTitle("Barrel layer", 1);
	Cluster_Gen_Barrel->setAxisTitle("# Clusters", 2);
	
	HistoName = "NClusters_Unkn_Barrel";
	Cluster_Unkn_Barrel = dqmStore_->book1D(HistoName, HistoName,
																				psTTClusterStacks.getParameter<int32_t>("Nbinsx"),
																				psTTClusterStacks.getParameter<double>("xmin"),
																				psTTClusterStacks.getParameter<double>("xmax"));
	Cluster_Unkn_Barrel->setAxisTitle("Barrel layer", 1);
	Cluster_Unkn_Barrel->setAxisTitle("# Clusters", 2);
	
	HistoName = "NClusters_Comb_Barrel";
	Cluster_Comb_Barrel = dqmStore_->book1D(HistoName, HistoName,
																				psTTClusterStacks.getParameter<int32_t>("Nbinsx"),
																				psTTClusterStacks.getParameter<double>("xmin"),
																				psTTClusterStacks.getParameter<double>("xmax"));
	Cluster_Comb_Barrel->setAxisTitle("Barrel layer", 1);
	Cluster_Comb_Barrel->setAxisTitle("# Clusters", 2);
	
	HistoName = "NClusters_Gen_Endcap";
	Cluster_Gen_Endcap = dqmStore_->book1D(HistoName, HistoName,
																				psTTClusterStacks.getParameter<int32_t>("Nbinsx"),
																				psTTClusterStacks.getParameter<double>("xmin"),
																				psTTClusterStacks.getParameter<double>("xmax"));
	Cluster_Gen_Endcap->setAxisTitle("Endcap disk", 1);
	Cluster_Gen_Endcap->setAxisTitle("# Clusters", 2);
	
	HistoName = "NClusters_Unkn_Endcap";
	Cluster_Unkn_Endcap = dqmStore_->book1D(HistoName, HistoName,
																				psTTClusterStacks.getParameter<int32_t>("Nbinsx"),
																				psTTClusterStacks.getParameter<double>("xmin"),
																				psTTClusterStacks.getParameter<double>("xmax"));
	Cluster_Unkn_Endcap->setAxisTitle("Endcap disk", 1);
	Cluster_Unkn_Endcap->setAxisTitle("# Clusters", 2);
	
	HistoName = "NClusters_Comb_Endcap";
	Cluster_Comb_Endcap = dqmStore_->book1D(HistoName, HistoName,
																				psTTClusterStacks.getParameter<int32_t>("Nbinsx"),
																				psTTClusterStacks.getParameter<double>("xmin"),
																				psTTClusterStacks.getParameter<double>("xmax"));
	Cluster_Comb_Endcap->setAxisTitle("Endcap disk", 1);
	Cluster_Comb_Endcap->setAxisTitle("# Clusters", 2);
	
	edm::ParameterSet psTTClusterEta =  conf_.getParameter<edm::ParameterSet>("TH1TTCluster_Eta");
	HistoName = "Cluster_Gen_Eta";
	Cluster_Gen_Eta = dqmStore_->book1D(HistoName, HistoName,
																				psTTClusterEta.getParameter<int32_t>("Nbinsx"),
																				psTTClusterEta.getParameter<double>("xmin"),
																				psTTClusterEta.getParameter<double>("xmax"));
	Cluster_Gen_Eta->setAxisTitle("Genuine TTCluster Eta", 1);
	Cluster_Gen_Eta->setAxisTitle("# Clusters", 2);
	
	HistoName = "Cluster_Unkn_Eta";
	Cluster_Unkn_Eta = dqmStore_->book1D(HistoName, HistoName,
																				psTTClusterEta.getParameter<int32_t>("Nbinsx"),
																				psTTClusterEta.getParameter<double>("xmin"),
																				psTTClusterEta.getParameter<double>("xmax"));
	Cluster_Unkn_Eta->setAxisTitle("Unknown TTCluster Eta", 1);
	Cluster_Unkn_Eta->setAxisTitle("# Clusters", 2);
	
	HistoName = "Cluster_Comb_Eta";
	Cluster_Comb_Eta = dqmStore_->book1D(HistoName, HistoName,
																				psTTClusterEta.getParameter<int32_t>("Nbinsx"),
																				psTTClusterEta.getParameter<double>("xmin"),
																				psTTClusterEta.getParameter<double>("xmax"));
	Cluster_Comb_Eta->setAxisTitle("Combinatorial TTCluster Eta", 1);
	Cluster_Comb_Eta->setAxisTitle("# Clusters", 2);
	
}//end of method


// ------------ method called once each job just after ending the event loop  ------------
void 
OuterTrackerCluster::endJob(void) 
{
	
}

//define this as a plug-in
DEFINE_FWK_MODULE(OuterTrackerCluster);
