// -*- C++ -*-
//
// Package:    Phase2OuterTracker
// Class:      Phase2OuterTracker
// 
/**\class Phase2OuterTracker OuterTrackerStub.cc Validation/Phase2OuterTracker/plugins/OuterTrackerStub.cc

 Description: [one line class summary]

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  Lieselotte Moreels
//         Created:  Mon, 27 Oct 2014 09:07:51 GMT
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
#include "Validation/Phase2OuterTracker/interface/OuterTrackerStub.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"


#include "SimTracker/TrackTriggerAssociation/interface/TTClusterAssociationMap.h"	//Needed for TTStubAssociationMap.h !
#include "SimTracker/TrackTriggerAssociation/interface/TTStubAssociationMap.h"
#include "DataFormats/L1TrackTrigger/interface/TTStub.h"
#include "DataFormats/L1TrackTrigger/interface/TTTypes.h"
#include "Geometry/TrackerGeometryBuilder/interface/StackedTrackerGeometry.h"
#include "Geometry/Records/interface/StackedTrackerGeometryRecord.h"

#include "TMath.h"
#include <iostream>

//
// constructors and destructor
//
OuterTrackerStub::OuterTrackerStub(const edm::ParameterSet& iConfig)
: dqmStore_(edm::Service<DQMStore>().operator->()), conf_(iConfig)

{
  topFolderName_ = conf_.getParameter<std::string>("TopFolderName");
  
}


OuterTrackerStub::~OuterTrackerStub()
{
	
	// do anything here that needs to be done at desctruction time
	// (e.g. close files, deallocate resources etc.)
	
}



//
// member functions
//

// ------------ method called for each event  ------------
void
OuterTrackerStub::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
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
	edm::Handle< edmNew::DetSetVector< TTStub< Ref_PixelDigi_ > > >    PixelDigiTTStubHandle;
	iEvent.getByLabel( "TTStubsFromPixelDigis", "StubAccepted",        PixelDigiTTStubHandle );
	/// Track Trigger MC Truth
	edm::Handle< TTStubAssociationMap< Ref_PixelDigi_ > >    MCTruthTTStubHandle;
	iEvent.getByLabel( "TTStubAssociatorFromPixelDigis", "StubAccepted",        MCTruthTTStubHandle );
	
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
			bool combinStub     = MCTruthTTStubHandle->isCombinatoric( tempStubRef );
			//bool unknownStub    = MCTruthTTStubHandle->isUnknown( tempStubRef );
			/* //Necessary for StubPID histo. Here or in MCTruth analyzer?
			int partStub         = 999999999;
			if ( genuineStub )
			{
				edm::Ptr< TrackingParticle > thisTP = MCTruthTTStubHandle->findTrackingParticlePtr( tempStubRef );
				partStub = thisTP->pdgId();
			}*/
			
			GlobalPoint posStub = theStackedGeometry->findGlobalPosition( &(*tempStubRef) );
			
			if ( detIdStub.isBarrel() )
			{
				
				if ( genuineStub )
				{
					Stub_Gen_Barrel->Fill( detIdStub.iLayer() );
				}
				else if ( combinStub )
				{
					Stub_Comb_Barrel->Fill( detIdStub.iLayer() );
				}
				else
				{
					Stub_Unkn_Barrel->Fill( detIdStub.iLayer() );
				}
				
			} // end if isBarrel()
			else if ( detIdStub.isEndcap() )
			{
				
				if ( genuineStub )
				{
					Stub_Gen_Endcap->Fill( detIdStub.iDisk() );
				}
				else if ( combinStub )
				{
					Stub_Comb_Endcap->Fill( detIdStub.iDisk() );
				}
				else
				{
					Stub_Unkn_Endcap->Fill( detIdStub.iDisk() );
				}
				
			}	// end if isEndcap()
			
			/// Eta distribution in function of genuine/combinatorial/unknown stub
			if ( genuineStub )
			{
				Stub_Gen_Eta->Fill( posStub.eta() );
			}
			else if ( combinStub )
			{
				Stub_Comb_Eta->Fill( posStub.eta() );
			}
			else
			{
				Stub_Unkn_Eta->Fill( posStub.eta() );
			}
			
			//hStub_PID->Fill( partStub );  // SHOULD BE HERE!
			
		}	// end loop contentIter
	}	// end loop inputIter
	
}


// ------------ method called when starting to processes a run  ------------

void 
OuterTrackerStub::beginRun(edm::Run const&, edm::EventSetup const&)
{
	SiStripFolderOrganizer folder_organizer;
	folder_organizer.setSiStripFolderName(topFolderName_);
	folder_organizer.setSiStripFolder();
	
	
	dqmStore_->setCurrentFolder(topFolderName_+"/Stubs/");
	
	/// TTStub stacks
	edm::ParameterSet psTTStubStacks =  conf_.getParameter<edm::ParameterSet>("TH1TTStub_Stack");
	std::string HistoName = "NStubs_Gen_Barrel";
	Stub_Gen_Barrel = dqmStore_->book1D(HistoName, HistoName,
																				psTTStubStacks.getParameter<int32_t>("Nbinsx"),
																				psTTStubStacks.getParameter<double>("xmin"),
																				psTTStubStacks.getParameter<double>("xmax"));
	Stub_Gen_Barrel->setAxisTitle("Barrel layer", 1);
	Stub_Gen_Barrel->setAxisTitle("# TTStubs", 2);
	
	HistoName = "NStubs_Unkn_Barrel";
	Stub_Unkn_Barrel = dqmStore_->book1D(HistoName, HistoName,
																				psTTStubStacks.getParameter<int32_t>("Nbinsx"),
																				psTTStubStacks.getParameter<double>("xmin"),
																				psTTStubStacks.getParameter<double>("xmax"));
	Stub_Unkn_Barrel->setAxisTitle("Barrel layer", 1);
	Stub_Unkn_Barrel->setAxisTitle("# TTStubs", 2);
	
	HistoName = "NStubs_Comb_Barrel";
	Stub_Comb_Barrel = dqmStore_->book1D(HistoName, HistoName,
																				psTTStubStacks.getParameter<int32_t>("Nbinsx"),
																				psTTStubStacks.getParameter<double>("xmin"),
																				psTTStubStacks.getParameter<double>("xmax"));
	Stub_Comb_Barrel->setAxisTitle("Barrel layer", 1);
	Stub_Comb_Barrel->setAxisTitle("# TTStubs", 2);
	
	HistoName = "NStubs_Gen_Endcap";
	Stub_Gen_Endcap = dqmStore_->book1D(HistoName, HistoName,
																				psTTStubStacks.getParameter<int32_t>("Nbinsx"),
																				psTTStubStacks.getParameter<double>("xmin"),
																				psTTStubStacks.getParameter<double>("xmax"));
	Stub_Gen_Endcap->setAxisTitle("Endcap disk", 1);
	Stub_Gen_Endcap->setAxisTitle("# TTStubs", 2);
	
	HistoName = "NStubs_Unkn_Endcap";
	Stub_Unkn_Endcap = dqmStore_->book1D(HistoName, HistoName,
																				psTTStubStacks.getParameter<int32_t>("Nbinsx"),
																				psTTStubStacks.getParameter<double>("xmin"),
																				psTTStubStacks.getParameter<double>("xmax"));
	Stub_Unkn_Endcap->setAxisTitle("Endcap disk", 1);
	Stub_Unkn_Endcap->setAxisTitle("# TTStubs", 2);
	
	HistoName = "NStubs_Comb_Endcap";
	Stub_Comb_Endcap = dqmStore_->book1D(HistoName, HistoName,
																				psTTStubStacks.getParameter<int32_t>("Nbinsx"),
																				psTTStubStacks.getParameter<double>("xmin"),
																				psTTStubStacks.getParameter<double>("xmax"));
	Stub_Comb_Endcap->setAxisTitle("Encap disk", 1);
	Stub_Comb_Endcap->setAxisTitle("# TTStubs", 2);
	
	edm::ParameterSet psTTStubEta =  conf_.getParameter<edm::ParameterSet>("TH1TTStub_Eta");
	HistoName = "Stub_Gen_Eta";
	Stub_Gen_Eta = dqmStore_->book1D(HistoName, HistoName,
																				psTTStubEta.getParameter<int32_t>("Nbinsx"),
																				psTTStubEta.getParameter<double>("xmin"),
																				psTTStubEta.getParameter<double>("xmax"));
	Stub_Gen_Eta->setAxisTitle("Genuine TTStub Eta", 1);
	Stub_Gen_Eta->setAxisTitle("# TTStubs", 2);
	
	HistoName = "Stub_Unkn_Eta";
	Stub_Unkn_Eta = dqmStore_->book1D(HistoName, HistoName,
																				psTTStubEta.getParameter<int32_t>("Nbinsx"),
																				psTTStubEta.getParameter<double>("xmin"),
																				psTTStubEta.getParameter<double>("xmax"));
	Stub_Unkn_Eta->setAxisTitle("Unknown TTStub Eta", 1);
	Stub_Unkn_Eta->setAxisTitle("# TTStubs", 2);
	
	HistoName = "Stub_Comb_Eta";
	Stub_Comb_Eta = dqmStore_->book1D(HistoName, HistoName,
																				psTTStubEta.getParameter<int32_t>("Nbinsx"),
																				psTTStubEta.getParameter<double>("xmin"),
																				psTTStubEta.getParameter<double>("xmax"));
	Stub_Comb_Eta->setAxisTitle("Combinatorial TTStub Eta", 1);
	Stub_Comb_Eta->setAxisTitle("# TTStubs", 2);
}


// ------------ method called once each job just after ending the event loop  ------------
void 
OuterTrackerStub::endJob() 
{
}

//define this as a plug-in
DEFINE_FWK_MODULE(OuterTrackerStub);
