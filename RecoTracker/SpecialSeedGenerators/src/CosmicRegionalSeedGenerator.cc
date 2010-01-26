#include <iostream>
#include <memory>
#include <string>

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripMatchedRecHit2DCollection.h"
#include "RecoTracker/SpecialSeedGenerators/interface/CosmicRegionalSeedGenerator.h"

using namespace std;

CosmicRegionalSeedGenerator::CosmicRegionalSeedGenerator(edm::ParameterSet const& conf) : 
  conf_(conf)
{
  edm::LogInfo ("CosmicRegionalSeedGenerator") << "Begin Run:: Constructing  CosmicRegionalSeedGenerator";

  edm::ParameterSet regionPSet = conf_.getParameter<edm::ParameterSet>("RegionPSet");
  m_tp_label     = regionPSet.getParameter<edm::InputTag>("tp_label"); 
  m_ptMin        = regionPSet.getParameter<double>("ptMin");
  m_rVertex      = regionPSet.getParameter<double>("rVertex");
  m_zVertex      = regionPSet.getParameter<double>("zVertex");
  m_deltaEta     = regionPSet.getParameter<double>("deltaEtaRegion");
  m_deltaPhi     = regionPSet.getParameter<double>("deltaPhiRegion");

  edm::ParameterSet hltPSet = conf_.getParameter<edm::ParameterSet>("HLTPSet");
  hltTag_                   = hltPSet.getParameter<edm::InputTag>("hltTag");
  triggerSummaryLabel_      = hltPSet.getParameter<std::string>("triggerSummaryLabel");
  thePropagatorName_        = hltPSet.getParameter<std::string>("thePropagatorName");
  seeding_                  = hltPSet.getParameter<std::string>("seeding");

}
   
std::vector<TrackingRegion*, std::allocator<TrackingRegion*> > CosmicRegionalSeedGenerator::regions(const edm::Event& event, const edm::EventSetup& es) const
{

  std::vector<TrackingRegion* > result;


  //________________________________________
  //
  //Seeding on L2 muon (Datas)
  //________________________________________

      
  if(seeding_=="seedOnL2") {


    //get collections
    //+++++++++++++++


    //get the trigger result summary
    edm::Handle<trigger::TriggerEvent> triggerObj;
    event.getByLabel(triggerSummaryLabel_,triggerObj); // triggerSummaryLabel = "triggerSummaryAOD"

    const trigger::TriggerObjectCollection & toc(triggerObj->getObjects()); //get all the objects in the trigger


    //get the propagator 
    edm::ESHandle<Propagator> thePropagator;
    es.get<TrackingComponentsRecord>().get(thePropagatorName_, thePropagator); // thePropagatorName = AnalyticalPropagator


    //get the index of the HLT_TrackerCosmics
    const int index = triggerObj->filterIndex(hltTag_);
      
    //debug collections
    for (int i = 0; i< triggerObj->sizeFilters(); ++i ) 
      LogDebug("CosmicRegionalSeedGenerator") << "Triggers tag: " << "\n"
						   << triggerObj->filterTag(i);
      
    if ( index >= triggerObj->sizeFilters() ) {
      edm::LogError("CosmicRegionalSeedGenerator") << "Error::index size too high (" << triggerObj->sizeFilters() << " >= " << index << ")" <<"\n"
						   << "Possible bad typo for the HLT Tag";
      return result;
    }


    //get the indexes
    const trigger::Keys & k = triggerObj->filterKeys(index); //get the indexes of the objects that made the decision of the trigger above
      

    //definition of the region
    //+++++++++++++++++++++++++

      
    for (trigger::Keys::const_iterator ki = k.begin(); ki !=k.end(); ++ki ) {//loop on these indexes

      //initial global  position & momentum
      GlobalPoint position(0,0,0);//need major change of the trigger summary, the vertex position is not present!!!!!!!!!!!!!!!!!
      GlobalVector momentum(toc[*ki].px(),
			    toc[*ki].py(),
			    toc[*ki].pz());

      LogDebug("CosmicRegionalSeedGenerator") << "Momentum of the L2 4-vector = (" << toc[*ki].px() << ", " << toc[*ki].py() << ", " << toc[*ki].pz() << ")";


      //charge
      int charge = (int) (toc[*ki].id()>0) ? -1 : 1;


      //propagation on the last layers of TOB
      GlobalTrajectoryParameters glb_parameters(position,
						momentum,
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
      GlobalPoint regionPosition = outer.globalPosition();
      GlobalVector regionMom = outer.globalMomentum();
      LogDebug("CosmicRegionalSeedGenerator") << "Global position of the region = " << outer.globalPosition() << "\n"
						   << "Momentum = " << outer.globalMomentum();
	

      //step back
      double stepBack = 50;
      if ( regionPosition.basicVector().dot( regionMom.basicVector() ) > 0) stepBack *=-1;
      GlobalPoint  vertex = regionPosition + stepBack * regionMom.unit();


      //definition of the region
      CosmicTrackingRegion *etaphiRegion = new CosmicTrackingRegion( //GlobalVector(regionPosition.basicVector()),
								    (( stepBack > 0)?-1.:1.)*regionMom,
								    vertex,
								    m_ptMin,
								    m_rVertex,
								    m_zVertex,
								    m_deltaEta,
								    m_deltaPhi,
								    0
								    );
      
      result.push_back(etaphiRegion);      

      LogDebug("CosmicRegionalSeedGenerator")   << "CosmicTrackingRegion build with origin: "<< vertex
						     <<"\ndirection: "<<etaphiRegion->direction();
					      
    }//end loop on indexes
      
  }//end seedOnL2
      


  //________________________________________
  //
  //Seeding on reco L2 muon track (MC)
  //________________________________________


      
  if(seeding_=="seedOnL2MuonRecoTrack") {

      
    //get collections
    //+++++++++++++++


    //get the L2 track reco track collection
    edm::Handle<reco::TrackCollection> L2muonstrackCollectionHandle;
    event.getByLabel("hltL2Muons", L2muonstrackCollectionHandle);
    if (!L2muonstrackCollectionHandle.isValid())
      {
	edm::LogError("CosmicRegionalSeedGenerator") << "Error::No MuonsL2muonstrackCollection in the event";
	return result;
      }

    LogDebug("CosmicRegionalSeedGenerator") << "Track collection size = " << L2muonstrackCollectionHandle->size();

    if (L2muonstrackCollectionHandle->size() !=0 ) cout << " L2muonstrackCollectionHandle->size() = " << L2muonstrackCollectionHandle->size() << endl;


    //get the propagator 
    edm::ESHandle<Propagator> thePropagator;
    es.get<TrackingComponentsRecord>().get(thePropagatorName_, thePropagator); // thePropagatorName = "AnalyticalPropagator"

    //get tracker geometry
    edm::ESHandle<TrackerGeometry> theTrackerGeometry;
    es.get<TrackerDigiGeometryRecord>().get(theTrackerGeometry);
    const TrackerGeometry& theTracker(*theTrackerGeometry);

    //get the rechits
    edm::Handle<SiStripMatchedRecHit2DCollection> rechitsMatchedHandle;
    event.getByLabel("siStripMatchedRecHits","stereoRecHit",rechitsMatchedHandle);
    if (!rechitsMatchedHandle.isValid()) edm::LogError("CosmicRegionalSeedGenerator") << "Error::No SiStripMatchedRecHit2DCollection in the event";
    const SiStripMatchedRecHit2DCollection * theMatchedRecHitCollection = rechitsMatchedHandle.product();

    //definition of the region
    //+++++++++++++++++++++++++


    int nl2muonstrack = 0;
    for ( reco::TrackCollection::const_iterator l2muonstrack = L2muonstrackCollectionHandle->begin(); l2muonstrack != L2muonstrackCollectionHandle->end(); ++l2muonstrack ) {//loop on muons tracks

	
      //debug
      nl2muonstrack++;
      LogDebug("CosmicRegionalSeedGenerator") << "Muon track found in the collection (" << nl2muonstrack << ") with \n"
						   << "# reco hits = " << l2muonstrack->recHitsSize() << "\n"
						   << "eta = " << l2muonstrack->momentum().eta() << "\n"
						   << "phi = " << l2muonstrack->momentum().phi();
	
      for ( SiStripMatchedRecHit2DCollection::const_iterator det_iter = theMatchedRecHitCollection->begin(), det_end = theMatchedRecHitCollection->end(); det_iter != det_end; ++det_iter) {

	int nhits=0;
	
	SiStripMatchedRecHit2DCollection::DetSet rechitRange = *det_iter;
	DetId detid(rechitRange.detId());

	const StripGeomDetUnit* const theStripDet = dynamic_cast<const StripGeomDetUnit*>(theTracker.idToDet(detid));
 
	SiStripMatchedRecHit2DCollection::DetSet::const_iterator rechitRangeIteratorBegin = rechitRange.begin();
	SiStripMatchedRecHit2DCollection::DetSet::const_iterator rechitRangeIteratorEnd = rechitRange.end();
	SiStripMatchedRecHit2DCollection::DetSet::const_iterator iRecHit = rechitRangeIteratorBegin;

	for (; iRecHit != rechitRangeIteratorEnd; iRecHit++){ //loop on recHits
	  nhits++;
	  SiStripMatchedRecHit2D const rechit = *iRecHit;
	  GlobalPoint position;
	  position = theStripDet->surface().toGlobal(rechit.localPosition());

	  LogDebug("CosmicRegionalSeedGenerator") << "Hit matched to the track: \n" << "\n"
						       << "hit # " << nhits << "\n"
						       << "position = (" << position.x() << ", " << position.y() << ", " << position.z() << ")\n" 
						       << "eta = " << position.eta() << "\n"
						       << "phi = " << position.phi();
	  
	}
      }


      //initial position & momentum
      GlobalPoint initialRegionPosition(l2muonstrack->referencePoint().x(), l2muonstrack->referencePoint().y(), l2muonstrack->referencePoint().z());
      GlobalVector initialRegionMomentum(l2muonstrack->momentum().x(), l2muonstrack->momentum().y(), l2muonstrack->momentum().z());
      LogDebug("CosmicRegionalSeedGenerator") << "Initial global position of the region = " << initialRegionPosition << "\n"
						   << "Momentum = " << initialRegionMomentum;

	
      //charge
      int charge = (int) (l2muonstrack->charge()>0) ? -1 : 1;

	
      //propagation on the last layers of TOB
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
      GlobalPoint regionPosition = outer.globalPosition();
      GlobalVector regionMom = outer.globalMomentum();
      LogDebug("CosmicRegionalSeedGenerator") << "Global position of the region after propagation = " << outer.globalPosition() << "\n"
						   << "Momentum = " << outer.globalMomentum() << "\n"
						   << "eta = " << outer.globalPosition().eta() << "\n"
						   << "phi = " << outer.globalPosition().phi();


      //step back
      double stepBack = -50;
      if ( regionPosition.basicVector().dot( regionMom.basicVector() ) > 0) stepBack *=-1;
      GlobalPoint  center = regionPosition + stepBack * regionMom.unit();
      LogDebug("CosmicRegionalSeedGenerator") << "Step back =  " << stepBack << "\n";
						

	
      //definition of the region
      CosmicTrackingRegion *etaphiRegion = new CosmicTrackingRegion( //GlobalVector(regionPosition.basicVector()),
								    (( stepBack > 0)?-1.:1.)*regionMom,
								    center,
								    m_ptMin,
								    m_rVertex,
								    m_zVertex,
								    m_deltaEta,
								    m_deltaPhi,
								    0
								    );
      
      result.push_back(etaphiRegion);      

      LogDebug("CosmicRegionalSeedGenerator")   << "CosmicTrackingRegion build with ceter at "<< center
						     <<"\ndirection = "<< etaphiRegion->direction()
						     <<"\neta = " << center.eta()
						     <<"\nphi = " << center.phi();

    }//end loop on muons tracks

  }//end if seedOnRecoMuon


  return result;


}



