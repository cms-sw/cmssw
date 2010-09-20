#include <iostream>
#include <memory>
#include <string>

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "TrackingTools/GeomPropagators/interface/Propagator.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "MagneticField/Engine/interface/MagneticField.h"

#include "DataFormats/TrackerRecHit2D/interface/SiStripMatchedRecHit2DCollection.h"
#include "DataFormats/SiStripDetId/interface/SiStripDetId.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"

#include "RecoTracker/SpecialSeedGenerators/interface/CosmicRegionalSeedGenerator.h"

using namespace std;

CosmicRegionalSeedGenerator::CosmicRegionalSeedGenerator(edm::ParameterSet const& conf) : 
  conf_(conf)
{
  edm::LogInfo ("CosmicRegionalSeedGenerator") << "Begin Run:: Constructing  CosmicRegionalSeedGenerator";

  regionPSet_ = conf_.getParameter<edm::ParameterSet>("RegionPSet");
  m_ptMin        = regionPSet_.getParameter<double>("ptMin");
  m_rVertex      = regionPSet_.getParameter<double>("rVertex");
  m_zVertex      = regionPSet_.getParameter<double>("zVertex");
  m_deltaEta     = regionPSet_.getParameter<double>("deltaEtaRegion");
  m_deltaPhi     = regionPSet_.getParameter<double>("deltaPhiRegion");

  edm::ParameterSet toolsPSet  = conf_.getParameter<edm::ParameterSet>("ToolsPSet");
  thePropagatorName_           = toolsPSet.getParameter<std::string>("thePropagatorName");
  regionBase_                  = toolsPSet.getParameter<std::string>("regionBase");

  edm::ParameterSet collectionsPSet = conf_.getParameter<edm::ParameterSet>("CollectionsPSet");
  recoMuonsCollection_          = collectionsPSet.getParameter<edm::InputTag>("recoMuonsCollection");
  recoTrackMuonsCollection_       = collectionsPSet.getParameter<edm::InputTag>("recoTrackMuonsCollection");

  edm::LogInfo ("CosmicRegionalSeedGenerator") << "Reco muons collection : "       << recoMuonsCollection_ << "\n"
					       << "Reco tracks muons collection: " << recoTrackMuonsCollection_;

}
   
std::vector<TrackingRegion*, std::allocator<TrackingRegion*> > CosmicRegionalSeedGenerator::regions(const edm::Event& event, const edm::EventSetup& es) const
{

  std::vector<TrackingRegion* > result;



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
    event.getByLabel(recoMuonsCollection_,muonsHandle);
    if (!muonsHandle.isValid())
      {
	edm::LogError("CosmicRegionalSeedGenerator") << "Error::No muons collection in the event - Please verify the name of the muon collection";
	return result;
      }

    LogDebug("CosmicRegionalSeedGenerator") << "Muons collection size = " << muonsHandle->size();

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
	  edm::LogError("CosmicRegionalSeedGenerator") << "Trajectory state on surface not valid" ;
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
      
	
      //definition of the region
      CosmicTrackingRegion *etaphiRegion = new CosmicTrackingRegion((-1)*regionMom,
								    center,
								    m_ptMin,
								    m_rVertex,
								    m_zVertex,
								    m_deltaEta,
								    m_deltaPhi,
								    regionPSet_
								    );
      
      result.push_back(etaphiRegion);      

      LogDebug("CosmicRegionalSeedGenerator")   << "Final CosmicTrackingRegion \n "
						<< "Position = "<< center << "\n "
						<< "Direction = "<< etaphiRegion->direction() << "\n "
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
    event.getByLabel(recoTrackMuonsCollection_,cosmicMuonsHandle);
    if (!cosmicMuonsHandle.isValid())
      {
	edm::LogError("CosmicRegionalSeedGenerator") << "Error::No muons collection in the event - Please verify the name of the muon reco track collection";
	return result;
      }

    LogDebug("CosmicRegionalSeedGenerator") << "Cosmic muons tracks collection size = " << cosmicMuonsHandle->size();

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
    for (reco::TrackCollection::const_iterator cosmicMuon = cosmicMuonsHandle->begin();  cosmicMuon != cosmicMuonsHandle->end();  ++cosmicMuon) {

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
	  edm::LogError("CosmicRegionalSeedGenerator") << "Trajectory state on surface not valid" ;
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
      
	
      //definition of the region
      CosmicTrackingRegion *etaphiRegion = new CosmicTrackingRegion((-1)*regionMom,
								    center,
								    m_ptMin,
								    m_rVertex,
								    m_zVertex,
								    m_deltaEta,
								    m_deltaPhi,
								    regionPSet_
								    );
      
      result.push_back(etaphiRegion);      

      LogDebug("CosmicRegionalSeedGenerator")   << "Final CosmicTrackingRegion \n "
						<< "Position = "<< center << "\n "
						<< "Direction = "<< etaphiRegion->direction() << "\n "
						<< "Distance from the region on the layer = " << (regionPosition -center).mag() << "\n "
						<< "Eta = " << center.eta() << "\n "
						<< "Phi = " << center.phi();
      

    }//end loop on muons

  }//end if SeedOnStaMuon





  return result;


}

