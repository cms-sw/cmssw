#include <iostream>
#include <memory>
#include <string>


#include "TrackingTools/GeomPropagators/interface/Propagator.h"


#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/JetReco/interface/CaloJet.h"
#include "DataFormats/RecoCandidate/interface/RecoChargedCandidate.h"

#include "RecoTracker/SpecialSeedGenerators/interface/CosmicRegionalSeedGenerator.h"

#include <TMath.h>

using namespace std;
using namespace trigger;
using namespace reco;
using namespace edm;

CosmicRegionalSeedGenerator::CosmicRegionalSeedGenerator(edm::ParameterSet const& conf,
	   edm::ConsumesCollector && iC) : 
  conf_(conf)
{
  edm::LogInfo ("CosmicRegionalSeedGenerator") << "Begin Run:: Constructing  CosmicRegionalSeedGenerator";

  regionPSet = conf_.getParameter<edm::ParameterSet>("RegionPSet");
  ptMin_        = regionPSet.getParameter<double>("ptMin");
  rVertex_      = regionPSet.getParameter<double>("rVertex");
  zVertex_      = regionPSet.getParameter<double>("zVertex");
  deltaEta_     = regionPSet.getParameter<double>("deltaEtaRegion");
  deltaPhi_     = regionPSet.getParameter<double>("deltaPhiRegion");
  
  edm::ParameterSet toolsPSet  = conf_.getParameter<edm::ParameterSet>("ToolsPSet");
  thePropagatorName_           = toolsPSet.getParameter<std::string>("thePropagatorName");
  regionBase_                  = toolsPSet.getParameter<std::string>("regionBase");

  edm::ParameterSet collectionsPSet = conf_.getParameter<edm::ParameterSet>("CollectionsPSet");
  recoMuonsCollection_          = collectionsPSet.getParameter<edm::InputTag>("recoMuonsCollection");
  recoTrackMuonsCollection_     = collectionsPSet.getParameter<edm::InputTag>("recoTrackMuonsCollection");
  recoL2MuonsCollection_        = collectionsPSet.getParameter<edm::InputTag>("recoL2MuonsCollection");

  edm::ParameterSet regionInJetsCheckPSet = conf_.getParameter<edm::ParameterSet>("RegionInJetsCheckPSet");
  doJetsExclusionCheck_         = regionInJetsCheckPSet.getParameter<bool>("doJetsExclusionCheck");
  deltaRExclusionSize_          = regionInJetsCheckPSet.getParameter<double>("deltaRExclusionSize");
  jetsPtMin_                    = regionInJetsCheckPSet.getParameter<double>("jetsPtMin");
  recoCaloJetsCollection_       = regionInJetsCheckPSet.getParameter<edm::InputTag>("recoCaloJetsCollection");
  recoCaloJetsToken_            = iC.consumes<reco::CaloJetCollection>(recoCaloJetsCollection_);
  recoMuonsToken_     	        = iC.consumes<reco::MuonCollection>(recoMuonsCollection_);
  recoTrackMuonsToken_	        = iC.consumes<reco::TrackCollection>(recoTrackMuonsCollection_);
  recoL2MuonsToken_   	        = iC.consumes<reco::RecoChargedCandidateCollection>(recoL2MuonsCollection_);
  measurementTrackerEventToken_ = iC.consumes<MeasurementTrackerEvent>(edm::InputTag("MeasurementTrackerEvent"));

  edm::LogInfo ("CosmicRegionalSeedGenerator") << "Reco muons collection: "        << recoMuonsCollection_ << "\n"
					       << "Reco tracks muons collection: " << recoTrackMuonsCollection_<< "\n"
					       << "Reco L2 muons collection: "     << recoL2MuonsCollection_;
}
   
std::vector<std::unique_ptr<TrackingRegion>> CosmicRegionalSeedGenerator::regions(const edm::Event& event, const edm::EventSetup& es) const
{

  std::vector<std::unique_ptr<TrackingRegion> > result;


  const MeasurementTrackerEvent *measurementTracker = nullptr;
  if(!measurementTrackerEventToken_.isUninitialized()) {
    edm::Handle<MeasurementTrackerEvent> hmte;
    event.getByToken(measurementTrackerEventToken_, hmte);
    measurementTracker = hmte.product();
  }
  //________________________________________
  //
  //Seeding on Sta muon (MC && Datas)
  //________________________________________


  if(regionBase_=="seedOnStaMuon"||regionBase_=="") {

    LogDebug("CosmicRegionalSeedGenerator") << "Seeding on stand alone muons ";

    //get collections
    //+++++++++++++++

    //get the muon collection
    edm::Handle<reco::MuonCollection> muonsHandle;
    event.getByToken(recoMuonsToken_,muonsHandle);
    if (!muonsHandle.isValid())
      {
	edm::LogError("CollectionNotFound") << "Error::No reco muons collection (" << recoMuonsCollection_ << ") in the event - Please verify the name of the muon collection";
	return result;
      }

    LogDebug("CosmicRegionalSeedGenerator") << "Muons collection size = " << muonsHandle->size();

    //get the jet collection
    edm::Handle<CaloJetCollection> caloJetsHandle;
    event.getByToken(recoCaloJetsToken_,caloJetsHandle);

    //get the propagator 
    edm::ESHandle<Propagator> thePropagator;
    es.get<TrackingComponentsRecord>().get(thePropagatorName_, thePropagator); // thePropagatorName = "AnalyticalPropagator"

    //get tracker geometry
    edm::ESHandle<TrackerGeometry> theTrackerGeometry;
    es.get<TrackerDigiGeometryRecord>().get(theTrackerGeometry);
    //const TrackerGeometry& theTracker(*theTrackerGeometry);
    DetId outerid;
    

    //definition of the region
    //+++++++++++++++++++++++++

    int nmuons = 0;
    for (reco::MuonCollection::const_iterator staMuon = muonsHandle->begin();  staMuon != muonsHandle->end();  ++staMuon) {

      //select sta muons
      if (!staMuon->isStandAloneMuon()) {
	LogDebug("CosmicRegionalSeedGenerator") << "This muon is not a stand alone muon";
	continue;
      }
      
      //bit 25 as a coverage -1.4 < eta < 1.4
      if ( abs( staMuon->standAloneMuon()->eta() ) > 1.5 ) continue;

      //debug
      nmuons++;
      LogDebug("CosmicRegionalSeedGenerator") << "Muon stand alone found in the collection - in muons chambers: \n " 
					      << "Position = " << staMuon->standAloneMuon()->outerPosition() << "\n "
					      << "Momentum = " << staMuon->standAloneMuon()->outerMomentum() << "\n "
					      << "Eta = " << staMuon->standAloneMuon()->eta() << "\n "
					      << "Phi = " << staMuon->standAloneMuon()->phi();
      
      //initial position, momentum, charge
      
      GlobalPoint initialRegionPosition(staMuon->standAloneMuon()->referencePoint().x(), staMuon->standAloneMuon()->referencePoint().y(), staMuon->standAloneMuon()->referencePoint().z());
      GlobalVector initialRegionMomentum(staMuon->standAloneMuon()->momentum().x(), staMuon->standAloneMuon()->momentum().y(), staMuon->standAloneMuon()->momentum().z());
      int charge = (int) staMuon->standAloneMuon()->charge();
   
      LogDebug("CosmicRegionalSeedGenerator") << "Initial region - Reference point of the sta muon: \n " 
					      << "Position = " << initialRegionPosition << "\n "
					      << "Momentum = " << initialRegionMomentum << "\n "
					      << "Eta = " << initialRegionPosition.eta() << "\n "
					      << "Phi = " << initialRegionPosition.phi() << "\n "
					      << "Charge = " << charge;
   
      //propagation on the last layers of TOB
      if ( staMuon->standAloneMuon()->outerPosition().y()>0 ) initialRegionMomentum *=-1;
      GlobalTrajectoryParameters glb_parameters(initialRegionPosition,
						initialRegionMomentum,
						charge,
						thePropagator->magneticField());
      FreeTrajectoryState fts(glb_parameters);
      StateOnTrackerBound onBounds(thePropagator.product());
      TrajectoryStateOnSurface outer = onBounds(fts);
      
      if (!outer.isValid()) 
	{
	  //edm::LogError("FailedPropagation") << "Trajectory state on surface not valid" ;
	  LogDebug("CosmicRegionalSeedGenerator") << "Trajectory state on surface not valid" ;
	  continue;
	}


      //final position & momentum
      GlobalPoint  regionPosition = outer.globalPosition();
      GlobalVector regionMom      = outer.globalMomentum();
      
      LogDebug("CosmicRegionalSeedGenerator") << "Region after propagation: \n "
					      << "Position = " << outer.globalPosition() << "\n "
					      << "Momentum = " << outer.globalMomentum() << "\n "
					      << "R = " << regionPosition.perp() << " ---- z = " << regionPosition.z() << "\n "
					      << "Eta = " << outer.globalPosition().eta() << "\n "
					      << "Phi = " << outer.globalPosition().phi();
      

      //step back
      double stepBack = 1;
      GlobalPoint  center = regionPosition + stepBack * regionMom.unit();
      GlobalVector v = stepBack * regionMom.unit();
      LogDebug("CosmicRegionalSeedGenerator") << "Step back vector =  " << v << "\n";

      //exclude region built in jets
      if ( doJetsExclusionCheck_ ) {
	double delta_R_min = 1000.;
	for ( CaloJetCollection::const_iterator jet = caloJetsHandle->begin (); jet != caloJetsHandle->end(); jet++ ) {
	  if ( jet->pt() < jetsPtMin_ ) continue;
	  
	  double deta = center.eta() - jet->eta();
	  double dphi = fabs( center.phi() - jet->phi() );
	  if ( dphi > TMath::Pi() ) dphi = 2*TMath::Pi() - dphi;
	  
	  double delta_R = sqrt(deta*deta + dphi*dphi);
	  if ( delta_R < delta_R_min ) delta_R_min = delta_R;
	  
	}//end loop on jets
	
	if ( delta_R_min < deltaRExclusionSize_ ) {
	  LogDebug("CosmicRegionalSeedGenerator") << "Region built too close from a jet"; 
	  continue;
	}
      }//end if doJetsExclusionCheck
      
	
      //definition of the region

      result.push_back(std::make_unique<CosmicTrackingRegion>((-1)*regionMom,
                                                              center,
                                                              ptMin_,
                                                              rVertex_,
                                                              zVertex_,
                                                              deltaEta_,
                                                              deltaPhi_,
                                                              regionPSet,
                                                              measurementTracker));

      LogDebug("CosmicRegionalSeedGenerator")   << "Final CosmicTrackingRegion \n "
						<< "Position = "<< center << "\n "
						<< "Direction = "<< result.back()->direction() << "\n "
						<< "Distance from the region on the layer = " << (regionPosition -center).mag() << "\n "
						<< "Eta = " << center.eta() << "\n "
						<< "Phi = " << center.phi();
      

    }//end loop on muons

  }//end if SeedOnStaMuon





  //________________________________________
  //
  //Seeding on cosmic muons (MC && Datas)
  //________________________________________


  if(regionBase_=="seedOnCosmicMuon") {

    LogDebug("CosmicRegionalSeedGenerator") << "Seeding on cosmic muons tracks";

    //get collections
    //+++++++++++++++

    //get the muon collection
    edm::Handle<reco::TrackCollection> cosmicMuonsHandle;
    event.getByToken(recoTrackMuonsToken_,cosmicMuonsHandle);
    if (!cosmicMuonsHandle.isValid())
      {
	edm::LogError("CollectionNotFound") << "Error::No cosmic muons collection (" << recoTrackMuonsCollection_ << ") in the event - Please verify the name of the muon reco track collection";
	return result;
      }

    LogDebug("CosmicRegionalSeedGenerator") << "Cosmic muons tracks collection size = " << cosmicMuonsHandle->size();

    //get the jet collection
    edm::Handle<CaloJetCollection> caloJetsHandle;
    event.getByToken(recoCaloJetsToken_,caloJetsHandle);

    //get the propagator 
    edm::ESHandle<Propagator> thePropagator;
    es.get<TrackingComponentsRecord>().get(thePropagatorName_, thePropagator); // thePropagatorName = "AnalyticalPropagator"

    //get tracker geometry
    edm::ESHandle<TrackerGeometry> theTrackerGeometry;
    es.get<TrackerDigiGeometryRecord>().get(theTrackerGeometry);
    DetId outerid;
    

    //definition of the region
    //+++++++++++++++++++++++++

    int nmuons = 0;
    for (reco::TrackCollection::const_iterator cosmicMuon = cosmicMuonsHandle->begin();  cosmicMuon != cosmicMuonsHandle->end();  ++cosmicMuon) {
      
      //bit 25 as a coverage -1.4 < eta < 1.4
      if ( abs( cosmicMuon->eta() ) > 1.5 ) continue;

      nmuons++;
            
      //initial position, momentum, charge
      GlobalPoint initialRegionPosition(cosmicMuon->referencePoint().x(), cosmicMuon->referencePoint().y(), cosmicMuon->referencePoint().z());
      GlobalVector initialRegionMomentum(cosmicMuon->momentum().x(), cosmicMuon->momentum().y(), cosmicMuon->momentum().z());
      int charge = (int) cosmicMuon->charge();
   
      LogDebug("CosmicRegionalSeedGenerator") << "Position and momentum of the muon track in the muon chambers: \n "
					      << "x = " << cosmicMuon->outerPosition().x() << "\n "
					      << "y = " << cosmicMuon->outerPosition().y() << "\n "
					      << "y = " << cosmicMuon->pt() << "\n "
					      << "Initial region - Reference point of the cosmic muon track: \n " 
					      << "Position = " << initialRegionPosition << "\n "
					      << "Momentum = " << initialRegionMomentum << "\n "
					      << "Eta = " << initialRegionPosition.eta() << "\n "
					      << "Phi = " << initialRegionPosition.phi() << "\n "
					      << "Charge = " << charge;
   
      //propagation on the last layers of TOB
      if ( cosmicMuon->outerPosition().y()>0 && cosmicMuon->momentum().y()<0 ) initialRegionMomentum *=-1;
      GlobalTrajectoryParameters glb_parameters(initialRegionPosition,
						initialRegionMomentum,
						charge,
						thePropagator->magneticField());
      FreeTrajectoryState fts(glb_parameters);
      StateOnTrackerBound onBounds(thePropagator.product());
      TrajectoryStateOnSurface outer = onBounds(fts);
      
      if (!outer.isValid()) 
	{
	  //edm::LogError("FailedPropagation") << "Trajectory state on surface not valid" ;
	  LogDebug("CosmicRegionalSeedGenerator") << "Trajectory state on surface not valid" ;
	  continue;
	}


      //final position & momentum
      GlobalPoint  regionPosition = outer.globalPosition();
      GlobalVector regionMom      = outer.globalMomentum();
      
      LogDebug("CosmicRegionalSeedGenerator") << "Region after propagation: \n "
					      << "Position = " << outer.globalPosition() << "\n "
					      << "Momentum = " << outer.globalMomentum() << "\n "
					      << "R = " << regionPosition.perp() << " ---- z = " << regionPosition.z() << "\n "
					      << "Eta = " << outer.globalPosition().eta() << "\n "
					      << "Phi = " << outer.globalPosition().phi();
      

      //step back
      double stepBack = 1;
      GlobalPoint  center = regionPosition + stepBack * regionMom.unit();
      GlobalVector v = stepBack * regionMom.unit();
      LogDebug("CosmicRegionalSeedGenerator") << "Step back vector =  " << v << "\n";
      
      //exclude region built in jets
      if ( doJetsExclusionCheck_ ) {	
	double delta_R_min = 1000.;
	for ( CaloJetCollection::const_iterator jet = caloJetsHandle->begin (); jet != caloJetsHandle->end(); jet++ ) {
	  if ( jet->pt() < jetsPtMin_ ) continue;
	  
	  double deta = center.eta() - jet->eta();
	  double dphi = fabs( center.phi() - jet->phi() );
	  if ( dphi > TMath::Pi() ) dphi = 2*TMath::Pi() - dphi;
	  
	  double delta_R = sqrt(deta*deta + dphi*dphi);
	  if ( delta_R < delta_R_min ) delta_R_min = delta_R;
	  
	}//end loop on jets
	
	if ( delta_R_min < deltaRExclusionSize_ ) {
	  LogDebug("CosmicRegionalSeedGenerator") << "Region built too close from a jet"; 
	  continue;
	}
      }// end if doJetsExclusionCheck

      //definition of the region
      result.push_back(std::make_unique<CosmicTrackingRegion>((-1)*regionMom,
                                                              center,
                                                              ptMin_,
                                                              rVertex_,
                                                              zVertex_,
                                                              deltaEta_,
                                                              deltaPhi_,
                                                              regionPSet,
                                                              measurementTracker));

      LogDebug("CosmicRegionalSeedGenerator")   << "Final CosmicTrackingRegion \n "
						<< "Position = "<< center << "\n "
						<< "Direction = "<< result.back()->direction() << "\n "
						<< "Distance from the region on the layer = " << (regionPosition -center).mag() << "\n "
						<< "Eta = " << center.eta() << "\n "
						<< "Phi = " << center.phi();
      
    }//end loop on muons

  }//end if SeedOnCosmicMuon


  //________________________________________
  //
  //Seeding on L2 muons (MC && Datas)
  //________________________________________

  if(regionBase_=="seedOnL2Muon") {

    LogDebug("CosmicRegionalSeedGenerator") << "Seeding on L2 muons";

    //get collections
    //+++++++++++++++

    //get the muon collection
    edm::Handle<reco::RecoChargedCandidateCollection> L2MuonsHandle;
    event.getByToken(recoL2MuonsToken_,L2MuonsHandle);

    if (!L2MuonsHandle.isValid())
      {
	edm::LogError("CollectionNotFound") << "Error::No L2 muons collection (" << recoL2MuonsCollection_ <<") in the event - Please verify the name of the L2 muon collection";
	return result;
      }

    LogDebug("CosmicRegionalSeedGenerator") << "L2 muons collection size = " << L2MuonsHandle->size();

    //get the propagator 
    edm::ESHandle<Propagator> thePropagator;
    es.get<TrackingComponentsRecord>().get(thePropagatorName_, thePropagator); // thePropagatorName = "AnalyticalPropagator"

    //get tracker geometry
    edm::ESHandle<TrackerGeometry> theTrackerGeometry;
    es.get<TrackerDigiGeometryRecord>().get(theTrackerGeometry);
    DetId outerid;
    

    //definition of the region
    //+++++++++++++++++++++++++

    int nmuons = 0;
    for (reco::RecoChargedCandidateCollection::const_iterator L2Muon = L2MuonsHandle->begin();  L2Muon != L2MuonsHandle->end();  ++L2Muon) {
      reco::TrackRef tkL2Muon = L2Muon->get<reco::TrackRef>();

      //bit 25 as a coverage -1.4 < eta < 1.4
      if ( abs( tkL2Muon->eta() ) > 1.5 ) continue;

      nmuons++;
            
      //initial position, momentum, charge
      GlobalPoint initialRegionPosition(tkL2Muon->referencePoint().x(), tkL2Muon->referencePoint().y(), tkL2Muon->referencePoint().z());
      GlobalVector initialRegionMomentum(tkL2Muon->momentum().x(), tkL2Muon->momentum().y(), tkL2Muon->momentum().z());
      int charge = (int) tkL2Muon->charge();
   
      LogDebug("CosmicRegionalSeedGenerator") << "Position and momentum of the L2 muon track in the muon chambers: \n "
					      << "x = " << tkL2Muon->outerPosition().x() << "\n "
					      << "y = " << tkL2Muon->outerPosition().y() << "\n "
					      << "y = " << tkL2Muon->pt() << "\n "
					      << "Initial region - Reference point of the L2 muon track: \n " 
					      << "Position = " << initialRegionPosition << "\n "
					      << "Momentum = " << initialRegionMomentum << "\n "
					      << "Eta = " << initialRegionPosition.eta() << "\n "
					      << "Phi = " << initialRegionPosition.phi() << "\n "
					      << "Charge = " << charge;
   

      //seeding only in the bottom
      if ( tkL2Muon->outerPosition().y() > 0 )
	{
	  LogDebug("CosmicRegionalSeedGenerator") << "L2 muon in the TOP --- Region not created";
	  return result;
	}
      
      GlobalTrajectoryParameters glb_parameters(initialRegionPosition,
						initialRegionMomentum,
						charge,
						thePropagator->magneticField());
      FreeTrajectoryState fts(glb_parameters);
      StateOnTrackerBound onBounds(thePropagator.product());
      TrajectoryStateOnSurface outer = onBounds(fts);
      
      if (!outer.isValid()) 
	{
	  //edm::LogError("FailedPropagation") << "Trajectory state on surface not valid" ;
	  LogDebug("CosmicRegionalSeedGenerator") << "Trajectory state on surface not valid" ;
	  continue;
	}


      //final position & momentum
      GlobalPoint  regionPosition = outer.globalPosition();
      GlobalVector regionMom      = outer.globalMomentum();
      
      LogDebug("CosmicRegionalSeedGenerator")     << "Region after propagation: \n "
						  << "Position = " << outer.globalPosition() << "\n "
						  << "Momentum = " << outer.globalMomentum() << "\n "
						  << "R = " << regionPosition.perp() << " ---- z = " << regionPosition.z() << "\n "
						  << "Eta = " << outer.globalPosition().eta() << "\n "
						  << "Phi = " << outer.globalPosition().phi();
      

      //step back
      double stepBack = 1;
      GlobalPoint  center = regionPosition + stepBack * regionMom.unit();
      GlobalVector v = stepBack * regionMom.unit();
      LogDebug("CosmicRegionalSeedGenerator") << "Step back vector =  " << v << "\n";
      
	
      //definition of the region
      result.push_back(std::make_unique<CosmicTrackingRegion>((-1)*regionMom,
                                                              center,
                                                              ptMin_,
                                                              rVertex_,
                                                              zVertex_,
                                                              deltaEta_,
                                                              deltaPhi_,
                                                              regionPSet,
                                                              measurementTracker));

      LogDebug("CosmicRegionalSeedGenerator")       << "Final L2TrackingRegion \n "
						    << "Position = "<< center << "\n "
						    << "Direction = "<< result.back()->direction() << "\n "
						    << "Distance from the region on the layer = " << (regionPosition -center).mag() << "\n "
						    << "Eta = " << center.eta() << "\n "
						    << "Phi = " << center.phi();
      

    }//end loop on muons

  }//end if SeedOnL2Muon

  return result;

}

