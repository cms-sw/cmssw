//////////////////////////
//  Producer by Anders  //
//     and Emmanuele    //
//    july 2012 @ CU    //
//////////////////////////


#ifndef L1TTRACK_PRDC_H
#define L1TTRACK_PRDC_H

////////////////////
// FRAMEWORK HEADERS
#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
//
// #include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
//
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

///////////////////////
// DATA FORMATS HEADERS
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/EDProduct.h"
#include "DataFormats/Common/interface/Ref.h"
//
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/SiPixelDetId/interface/PXBDetId.h" 
#include "DataFormats/Common/interface/DetSetVector.h"
//
#include "SimDataFormats/SLHC/interface/StackedTrackerTypes.h"
#include "SimDataFormats/SLHC/interface/slhcevent.hh"
#include "SimDataFormats/SLHC/interface/L1TRod.hh"
#include "SimDataFormats/SLHC/interface/L1TSector.hh"
#include "SimDataFormats/SLHC/interface/L1TStub.hh"
#include "SimDataFormats/SLHC/interface/L1TWord.hh"
#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"
#include "SimDataFormats/TrackingHit/interface/PSimHit.h"
#include "SimDataFormats/Track/interface/SimTrack.h"
#include "SimDataFormats/Track/interface/SimTrackContainer.h"
#include "SimDataFormats/Vertex/interface/SimVertex.h"
#include "SimDataFormats/Vertex/interface/SimVertexContainer.h"
//
#include "DataFormats/Math/interface/LorentzVector.h"
#include "DataFormats/Math/interface/Vector3D.h"
//
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "DataFormats/Candidate/interface/Candidate.h"
//
#include "DataFormats/GeometryCommonDetAlgo/interface/MeasurementPoint.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "TrackingTools/GeomPropagators/interface/HelixArbitraryPlaneCrossing.h"
////////////////////////
// FAST SIMULATION STUFF
#include "FastSimulation/Particle/interface/RawParticle.h"
#include "FastSimulation/BaseParticlePropagator/interface/BaseParticlePropagator.h"

////////////////////////////
// DETECTOR GEOMETRY HEADERS
#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/TrackerGeometryBuilder/interface/PixelGeomDetUnit.h"
#include "Geometry/TrackerGeometryBuilder/interface/PixelGeomDetType.h"
#include "Geometry/TrackerGeometryBuilder/interface/PixelTopologyBuilder.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/RectangularPixelTopology.h"
#include "Geometry/CommonDetUnit/interface/GeomDetType.h"
#include "Geometry/CommonDetUnit/interface/GeomDetUnit.h"
//
#include "Geometry/Records/interface/StackedTrackerGeometryRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/StackedTrackerGeometry.h"
#include "Geometry/TrackerGeometryBuilder/interface/StackedTrackerDetUnit.h"
#include "DataFormats/SiPixelDetId/interface/StackedTrackerDetId.h"
//#include "SLHCUpgradeSimulations/Utilities/interface/classInfo.h" REMOVE

////////////////
// PHYSICS TOOLS
#include "RecoTracker/TkSeedGenerator/interface/FastHelix.h"
#include "TrackingTools/TrajectoryState/interface/FreeTrajectoryState.h"
//
#include "DataFormats/GeometryCommonDetAlgo/interface/MeasurementVector.h"
#include "DataFormats/GeometrySurface/interface/BoundPlane.h"
//#include "SLHCUpgradeSimulations/Utilities/interface/constants.h" REMOVE

//////////////
// STD HEADERS
#include <memory>
#include <string>
#include <iostream>
#include <fstream>

//////////////
// NAMESPACES
// using namespace std;
// using namespace reco;
using namespace edm;
//using namespace cmsUpgrades;


//////////////////////////////
//                          //
//     CLASS DEFINITION     //
//                          //
//////////////////////////////

/////////////////////////////////////
// this class is needed to make a map
// between different types of stubs
class L1TStubCompare 
{
public:
  bool operator()(const L1TStub& x, const L1TStub& y) {
    if (x.layer() != y.layer()) return (y.layer()-x.layer())>0;
    else {
      if (x.ladder() != y.ladder()) return (y.ladder()-x.ladder())>0;
      else {
	if (x.module() != y.module()) return (y.module()-x.module())>0;
	else {
	  if (x.iz() != y.iz()) return (y.iz()-x.iz())>0;
	  else return (x.iphi()-y.iphi())>0;
	}
      }
    }
  }
};


class L1TrackProducer : public edm::EDProducer
{
public:

  typedef L1TkStub_PixelDigi_                           L1TkStubType;
  typedef std::vector< L1TkStubType >                                L1TkStubCollectionType;
  typedef edm::Ptr< L1TkStubType >                                   L1TkStubPtrType;
  typedef std::vector< L1TkStubPtrType >                             L1TkStubPtrCollection;
  typedef std::vector< L1TkStubPtrCollection >                       L1TkStubPtrCollVectorType;

  //typedef L1TkTracklet_PixelDigi_                       L1TkTrackletType;
  //typedef std::vector< L1TkTrackletType >                            L1TkTrackletCollectionType;
  //typedef edm::Ptr< L1TkTrackletType >                               L1TkTrackletPtrType;

  typedef L1TkTrack_PixelDigi_                          L1TkTrackType;
  typedef std::vector< L1TkTrackType >                               L1TkTrackCollectionType;
  typedef std::vector< L1TTrack >                                    L1TrackCollectionType;

  /// Constructor/destructor
  explicit L1TrackProducer(const edm::ParameterSet& iConfig);
  virtual ~L1TrackProducer();

protected:
                     
private:
  GeometryMap geom;
  int eventnum;

  /// Containers of parameters passed by python configuration file
  edm::ParameterSet config;

  /// ///////////////// ///
  /// MANDATORY METHODS ///
  virtual void beginRun( edm::Run& run, const edm::EventSetup& iSetup );
  virtual void endRun( edm::Run& run, const edm::EventSetup& iSetup );
  virtual void produce( edm::Event& iEvent, const edm::EventSetup& iSetup );
};


//////////////
// CONSTRUCTOR
L1TrackProducer::L1TrackProducer(edm::ParameterSet const& iConfig) // :   config(iConfig)
{
  //produces<L1TrackCollectionType>( "Level1Tracks" ).setBranchAlias("Level1Tracks");
  produces< L1TkStubPtrCollVectorType >( "L1TkStubs" ).setBranchAlias("L1TkStubs");
  produces< L1TkTrackCollectionType >( "Level1TkTracks" ).setBranchAlias("Level1TkTracks");
  // produces<L1TkStubMapType>( "L1TkStubMap" ).setBranchAlias("L1TkStubMap");
  // produces< L1TkTrackletCollectionType >( "L1TkTracklets" ).setBranchAlias("L1TkTracklets");
}

/////////////
// DESTRUCTOR
L1TrackProducer::~L1TrackProducer()
{
  /// Insert here what you need to delete
  /// when you close the class instance
}  

//////////
// END JOB
void L1TrackProducer::endRun(edm::Run& run, const edm::EventSetup& iSetup)
{
  /// Things to be done at the exit of the event Loop 

}

////////////
// BEGIN JOB
void L1TrackProducer::beginRun(edm::Run& run, const edm::EventSetup& iSetup )
{
  eventnum=0;
  std::cout << "L1TrackProducer" << std::endl;
}

//////////
// PRODUCE
void L1TrackProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{

  typedef std::map< L1TStub, L1TkStubPtrType, L1TStubCompare > stubMapType;

  /// Prepare output
  //std::auto_ptr< L1TrackCollectionType > L1TracksForOutput( new L1TrackCollectionType );
  std::auto_ptr< L1TkStubPtrCollVectorType > L1TkStubsForOutput( new L1TkStubPtrCollVectorType );
  //std::auto_ptr< L1TkTrackletCollectionType > L1TkTrackletsForOutput( new L1TkTrackletCollectionType );
  std::auto_ptr< L1TkTrackCollectionType > L1TkTracksForOutput( new L1TkTrackCollectionType );

  /// Geometry handles etc
  edm::ESHandle<TrackerGeometry>                               geometryHandle;
  const TrackerGeometry*                                       theGeometry;
  edm::ESHandle<StackedTrackerGeometry>           stackedGeometryHandle;
  const StackedTrackerGeometry*                   theStackedGeometry;
  StackedTrackerGeometry::StackContainerIterator  StackedTrackerIterator;

  /// Geometry setup
  /// Set pointers to Geometry
  iSetup.get<TrackerDigiGeometryRecord>().get(geometryHandle);
  theGeometry = &(*geometryHandle);
  /// Set pointers to Stacked Modules
  iSetup.get<StackedTrackerGeometryRecord>().get(stackedGeometryHandle);
  theStackedGeometry = stackedGeometryHandle.product(); /// Note this is different 
                                                        /// from the "global" geometry
  ////////////////////////
  // GET MAGNETIC FIELD //
  edm::ESHandle<MagneticField> magneticFieldHandle;
  iSetup.get<IdealMagneticFieldRecord>().get(magneticFieldHandle);
  const MagneticField* theMagneticField = magneticFieldHandle.product();
  double mMagneticFieldStrength = theMagneticField->inTesla(GlobalPoint(0,0,0)).z();

  ////////////
  // GET BS //
  ////////////
  //edm::Handle<reco::BeamSpot> recoBeamSpotHandle;
  //iEvent.getByLabel("dummy",recoBeamSpotHandle);
  //math::XYZPoint bsPosition=recoBeamSpotHandle->position();
  math::XYZPoint bsPosition(0.0,0.0,0.0);


  cout << "L1TrackProducer: B="<<mMagneticFieldStrength
       <<" vx reco="<<bsPosition.x()
       <<" vy reco="<<bsPosition.y()
       <<" vz reco="<<bsPosition.z()
       <<endl;

  SLHCEvent ev;
  ev.setIPx(bsPosition.x());
  ev.setIPy(bsPosition.y());
  eventnum++;


  ///////////////////
  // GET SIMTRACKS //
  edm::Handle<edm::SimTrackContainer>   simTrackHandle;
  edm::Handle<edm::SimVertexContainer>  simVtxHandle;
  //iEvent.getByLabel( "famosSimHits", simTrackHandle );
  //iEvent.getByLabel( "famosSimHits", simVtxHandle );
  iEvent.getByLabel( "g4SimHits", simTrackHandle );
  iEvent.getByLabel( "g4SimHits", simVtxHandle );

  //////////////////////
  // GET MC PARTICLES //
  edm::Handle<reco::GenParticleCollection> genpHandle;
  iEvent.getByLabel( "genParticles", genpHandle );

  /////////////////////
  // GET PIXEL DIGIS //
  edm::Handle<edm::DetSetVector<PixelDigi> >         pixelDigiHandle;
  edm::Handle<edm::DetSetVector<PixelDigiSimLink> >  pixelDigiSimLinkHandle;
  iEvent.getByLabel("simSiPixelDigis", pixelDigiHandle);
  iEvent.getByLabel("simSiPixelDigis", pixelDigiSimLinkHandle);

  ////////////////////////
  // GET THE PRIMITIVES //
  edm::Handle<L1TkCluster_PixelDigi_Collection>  pixelDigiL1TkClusterHandle;
  edm::Handle<L1TkStub_PixelDigi_Collection>     pixelDigiL1TkStubHandle;
  iEvent.getByLabel("L1TkClustersFromPixelDigis", pixelDigiL1TkClusterHandle);
  iEvent.getByLabel("L1TkStubsFromPixelDigis", "StubsPass", pixelDigiL1TkStubHandle);


  // dump map between inner and outer modules
  static bool first=true;

  if (first) {
    
    first=false;
    
    std::map< uint32_t, bool > detIdToInnerMap; /// stores TRUE for inner sensors and FALSE for outer ones
    /// Loop over the detector elements
    for ( StackedTrackerIterator = theStackedGeometry->stacks().begin();
	  StackedTrackerIterator != theStackedGeometry->stacks().end();
	  ++StackedTrackerIterator ) {
	
      StackedTrackerDetUnit* stackDetUnit = *StackedTrackerIterator;
      StackedTrackerDetId stackDetId = stackDetUnit->Id();
      assert(stackDetUnit == theStackedGeometry->idToStack(stackDetId));
	
      const GeomDet* det0 = theStackedGeometry->idToDet(stackDetId, 0);
      const GeomDet* det1 = theStackedGeometry->idToDet(stackDetId, 1);

      uint32_t detId0 = det0->geographicalId().rawId();
      uint32_t detId1 = det1->geographicalId().rawId();

      DetId tkId0(detId0);
      DetId tkId1(detId1);

      const GeomDetUnit* gDetUnit = theGeometry->idToDetUnit( tkId0 );
      MeasurementPoint mp0( 0.5, 0.5 );
      MeasurementPoint mp1( 1024 + 0.5, 0.5 );
      MeasurementPoint mp2( 0.5, 80 + 0.5 );
      GlobalPoint pdPos0 = gDetUnit->surface().toGlobal( gDetUnit->topology().localPosition( mp0 ) ) ;      
      GlobalPoint pdPos1 = gDetUnit->surface().toGlobal( gDetUnit->topology().localPosition( mp1 ) ) ;      
      GlobalPoint pdPos2 = gDetUnit->surface().toGlobal( gDetUnit->topology().localPosition( mp2 ) ) ;      

      PXBDetId pxbId0(tkId0);
      PXBDetId pxbId1(tkId1);


      geom.addModule(stackDetId.iLayer(),stackDetId.iPhi()+1,stackDetId.iZ(),
		     stackDetId.iLayer(),stackDetId.iPhi()+1,stackDetId.iZ(),
		     pdPos0.x(),pdPos0.y(),pdPos0.z(),
		     pdPos1.x(),pdPos1.y(),pdPos1.z(),
		     pdPos2.x(),pdPos2.y(),pdPos2.z());

      //std::cout << "pdPos0:"<<pdPos0<<endl;
      //std::cout << "pdPos1:"<<pdPos1<<endl;
      //std::cout << "pdPos2:"<<pdPos2<<endl;

      detIdToInnerMap.insert( make_pair(detId0, true) );
      detIdToInnerMap.insert( make_pair(detId1, false) );
    }
 
  }


  ////////////////////////
  /// LOOP OVER SimTracks
  SimTrackContainer::const_iterator iterSimTracks;
  for ( iterSimTracks = simTrackHandle->begin();
	iterSimTracks != simTrackHandle->end();
	++iterSimTracks ) {

    /// Get the corresponding vertex
    int vertexIndex = iterSimTracks->vertIndex();
    const SimVertex& theSimVertex = (*simVtxHandle)[vertexIndex];
    math::XYZTLorentzVectorD trkVtxPos = theSimVertex.position();
    GlobalPoint trkVtxCorr = GlobalPoint( trkVtxPos.x() - bsPosition.x(), 
					  trkVtxPos.y() - bsPosition.y(), 
					  trkVtxPos.z() - bsPosition.z() );
    
    double pt=iterSimTracks->momentum().pt();
    if (pt!=pt) pt=9999.999;
    ev.addL1SimTrack(iterSimTracks->trackId(),iterSimTracks->type(),pt,
		     iterSimTracks->momentum().eta(), 
		     iterSimTracks->momentum().phi(), 
		     trkVtxCorr.x(),
		     trkVtxCorr.y(),
		     trkVtxCorr.z());
   
    
  } /// End of Loop over SimTracks


  /*
  /// Loop over L1TkCluster hits
  for ( clusterHitsIter = clusterHits.begin();
	clusterHitsIter != clusterHits.end();
	clusterHitsIter++ ) {

  }
  */

  // End loop over L1TkClusters

  std::cout << "Will loop over digis:"<<std::endl;

  //////////////////////////////
  /// LOOP OVER Detector Modules
  DetSetVector<PixelDigi>::const_iterator iterDet;
  for ( iterDet = pixelDigiHandle->begin();
        iterDet != pixelDigiHandle->end();
        iterDet++ ) {

    /// Build Detector Id
    DetId tkId( iterDet->id );
    StackedTrackerDetId stdetid(tkId);
    /// Check if it is Pixel
    if ( tkId.subdetId() != 1 ) continue;

    /// Get the PixelDigiSimLink corresponding to this one
    PXBDetId pxbId(tkId);
    DetSetVector<PixelDigiSimLink>::const_iterator itDigiSimLink=pixelDigiSimLinkHandle->find(pxbId.rawId());
    if (itDigiSimLink==pixelDigiSimLinkHandle->end()){
      std::cout << "Here 012211 found no match"<<std::endl;
      continue;
    }
    DetSet<PixelDigiSimLink> digiSimLink = *itDigiSimLink;
    
    //DetSet<PixelDigiSimLink> digiSimLink = (*pixelDigiSimLinkHandle)[ pxbId.rawId() ]; REMOVE
    DetSet<PixelDigiSimLink>::const_iterator iterSimLink;
    /// Renormalize layer number from 5-14 to 0-9 and skip if inner pixels
    if ( pxbId.layer() < 5 ) continue;
    //int whichStack = pxbId.layer() - 5;  REMOVE next 4 lines
    //int whichDoubleStack;
    //if ( whichStack%2 == 0 ) whichDoubleStack = whichStack/2;
    //else whichDoubleStack = (whichStack - 1)/2;

    // Layer 0-20
    DetId digiDetId = iterDet->id;
    int sensorLayer = 0.5*(2*PXBDetId(digiDetId).layer() + (PXBDetId(digiDetId).ladder() + 1)%2 - 8);

    /// Loop over PixelDigis within Module and select those above threshold
    DetSet<PixelDigi>::const_iterator iterDigi;
    for ( iterDigi = iterDet->data.begin();
          iterDigi != iterDet->data.end();
          iterDigi++ ) {

      /// Threshold (here it is NOT redundant)
      if ( iterDigi->adc() <= 30 ) continue;

      /// Try to learn something from PixelDigi position
      const GeomDetUnit* gDetUnit = theGeometry->idToDetUnit( tkId );
      MeasurementPoint mp( iterDigi->row() + 0.5, iterDigi->column() + 0.5 );
      GlobalPoint pdPos = gDetUnit->surface().toGlobal( gDetUnit->topology().localPosition( mp ) ) ;

      std::vector<int> simtrackids;

      if (1) {
	/// Loop over PixelDigiSimLink to find the
	/// correct link to the SimTrack collection
	for ( iterSimLink = digiSimLink.data.begin();
	      iterSimLink != digiSimLink.data.end();
	      iterSimLink++) {
	  
	  /// When the channel is the same, the link is found
	  if ( (int)iterSimLink->channel() == iterDigi->channel() ) {
	    /// Map wrt SimTrack Id
	    unsigned int simTrackId = iterSimLink->SimTrackId();
	    simtrackids.push_back(simTrackId);	
	  }
	}
      }

      ev.addDigi(sensorLayer,iterDigi->row(),iterDigi->column(),
		 pxbId.layer(),pxbId.ladder(),pxbId.module(),
		 pdPos.x(),pdPos.y(),pdPos.z(),simtrackids);


    }
  }

  std::cout << "Will loop over stubs:"<<std::endl;


  ///////////////////////
  /// LOOP OVER L1TkStubs
  stubMapType stubMap;
  int iter=0;

  L1TkStub_PixelDigi_Collection::const_iterator iterL1TkStub;
  for ( iterL1TkStub = pixelDigiL1TkStubHandle->begin();
	iterL1TkStub != pixelDigiL1TkStubHandle->end();
	++iterL1TkStub ) {

    double stubPt = iterL1TkStub->findRoughPt(mMagneticFieldStrength,
					      theStackedGeometry);
    if (stubPt>10000.0) stubPt=9999.99;
    GlobalPoint  stubPosition = iterL1TkStub->findGlobalPosition(theStackedGeometry);
    //GlobalVector stubDirection = iterL1TkStub->findGlobalDirection(theStackedGeometry);

    StackedTrackerDetId stubDetId = iterL1TkStub->getDetId();
    unsigned int iStack = stubDetId.iLayer();
    unsigned int iPhi = stubDetId.iPhi();
    unsigned int iZ = stubDetId.iZ();

    //const StackedTrackerDetUnit* aDetUnit = theStackedGeometry->idToStack(stubDetId);
    //DetId id0 = aDetUnit->stackMember(0);
    //DetId id1 = aDetUnit->stackMember(1);

    //PXBDetId pxb0 = PXBDetId(id0);
    //PXBDetId pxb1 = PXBDetId(id1);


    std::vector<bool> innerStack;
    std::vector<int> irphi;
    std::vector<int> iz;
    std::vector<int> iladder;
    std::vector<int> imodule;


    /// Get the Inner and Outer L1TkCluster
    edm::Ptr<L1TkCluster_PixelDigi_> innerCluster = iterL1TkStub->getClusterPtr(0);
    edm::Ptr<L1TkCluster_PixelDigi_> outerCluster = iterL1TkStub->getClusterPtr(1);

    /// Get the Digis for each L1TkCluster
    //int innerChannel = innerCluster.getHits().at(0)->channel();
    //int outerChannel = outerCluster.getHits().at(0)->channel();
       
          
    /// Loop over Detector Modules
    DetSetVector<PixelDigi>::const_iterator iterDet;
    for ( iterDet = pixelDigiHandle->begin();
	  iterDet != pixelDigiHandle->end();
	  iterDet++ ) {
      
      /// Build Detector Id
      DetId tkId( iterDet->id );
      
      const DetId innerDetId = theStackedGeometry->idToDet( innerCluster->getDetId(), 0 )->geographicalId();
      const DetId outerDetId = theStackedGeometry->idToDet( outerCluster->getDetId(), 1 )->geographicalId();
      
      if (innerDetId.rawId()==tkId.rawId()) {
	// Layer 1-10.5
	//int sensorLayer = (2*PXBDetId(tkId).layer() + (PXBDetId(tkId).ladder() + 1)%2 - 8);

	/// Loop over PixelDigis within Module and select those above threshold
	DetSet<PixelDigi>::const_iterator iterDigi;
	for ( iterDigi = iterDet->data.begin();
	      iterDigi != iterDet->data.end();
	      iterDigi++ ) {

	  /// Threshold (here it is NOT redundant)
	  if ( iterDigi->adc() <= 30 ) continue;
          for (unsigned int ihit=0;ihit<innerCluster->getHits().size();ihit++){
	    if (iterDigi->channel() == innerCluster->getHits().at(ihit)->channel()) {

	      innerStack.push_back(true);
	      irphi.push_back(iterDigi->row());
	      iz.push_back(iterDigi->column());
	      iladder.push_back(PXBDetId(tkId).ladder());
	      imodule.push_back(PXBDetId(tkId).module());

	    }
	  }
	}
      }
    

      if (outerDetId.rawId()==tkId.rawId()) {
	// Layer 0-20
	//int sensorLayer = (2*PXBDetId(tkId).layer() + (PXBDetId(tkId).ladder() + 1)%2 - 8);

	/// Loop over PixelDigis within Module and select those above threshold
	DetSet<PixelDigi>::const_iterator iterDigi;
	for ( iterDigi = iterDet->data.begin();
	      iterDigi != iterDet->data.end();
	      iterDigi++ ) {

	  /// Threshold (here it is NOT redundant)
	  if ( iterDigi->adc() <= 30 ) continue;
          for (unsigned int ihit=0;ihit<outerCluster->getHits().size();ihit++){
	    if (iterDigi->channel() == outerCluster->getHits().at(ihit)->channel()) {

	      innerStack.push_back(false);
	      irphi.push_back(iterDigi->row());
	      iz.push_back(iterDigi->column());
	      iladder.push_back(PXBDetId(tkId).ladder());
	      imodule.push_back(PXBDetId(tkId).module());

	    }
	  }     
	}
      }

    }


    ev.addStub(iStack-1,iPhi,iZ,stubPt,
	       stubPosition.x(),stubPosition.y(),stubPosition.z(),
	       innerStack,irphi,iz,iladder,imodule);

    //std::cout << "Stub:"<<stubPosition.x()<<" "<<
    //  stubPosition.y()<<" "<<stubPosition.z()<<std::endl;

    Stub *aStub = new Stub;
    *aStub = ev.stub(iter);
    iter++;
    
    //int theSimtrackId=ev.simtrackid(*aStub);
    int theSimtrackId=-1;
    
    L1TStub L1Stub(theSimtrackId, aStub->iphi(), aStub->iz(),
    		   aStub->layer()+1, aStub->ladder(), aStub->module(),
    		   aStub->x(), aStub->y(), aStub->z());
    delete aStub;

    stubMap.insert( make_pair(L1Stub, L1TkStubPtrType(pixelDigiL1TkStubHandle, iterL1TkStub-pixelDigiL1TkStubHandle->begin()) ) );
  } // end loop over stubs

  std::cout << "Will actually do L1 tracking:"<<std::endl;


  //////////////////////////
  // NOW RUN THE L1 tracking

  Sector sectors[NSECTORS];

  vector<int> layer1=geom.ladders(1);
  vector<int> layer2=geom.ladders(2);
  vector<int> layer3=geom.ladders(3);
  vector<int> layer4=geom.ladders(4);
  vector<int> layer5=geom.ladders(5);
  vector<int> layer6=geom.ladders(6);
  vector<int> layer7=geom.ladders(7);
  vector<int> layer8=geom.ladders(8);
  //vector<int> layer9=geom.ladders(9);   HACK 6_1
  //vector<int> layer10=geom.ladders(10); HACK 6_1
  
  for(unsigned int isector=0;isector<NSECTORS;isector++){
    sectors[isector].setSector(isector);
    
    
    for(unsigned int i=0;i<layer1.size();i++){
      sectors[isector].addLadder(geom,1,layer1[i]);
    }
    for(unsigned int i=0;i<layer2.size();i++){
      sectors[isector].addLadder(geom,2,layer2[i]);
    }
    for(unsigned int i=0;i<layer3.size();i++){
      sectors[isector].addLadder(geom,3,layer3[i]);
    }
    for(unsigned int i=0;i<layer4.size();i++){
      sectors[isector].addLadder(geom,4,layer4[i]);
    }
    for(unsigned int i=0;i<layer5.size();i++){
      sectors[isector].addLadder(geom,5,layer5[i]);
    }
    for(unsigned int i=0;i<layer6.size();i++){
      sectors[isector].addLadder(geom,6,layer6[i]);
    }
    for(unsigned int i=0;i<layer7.size();i++){
      sectors[isector].addLadder(geom,7,layer7[i]);
    }
    for(unsigned int i=0;i<layer8.size();i++){
      sectors[isector].addLadder(geom,8,layer8[i]);
    }
    //for(unsigned int i=0;i<layer9.size();i++){     HACK 6_1
    //  sectors[isector].addLadder(geom,9,layer9[i]);
    //}
    //for(unsigned int i=0;i<layer10.size();i++){
    //  sectors[isector].addLadder(geom,10,layer10[i]);
    //}
    
  }
  
  L1TSector* Sectors[NSECTORS];

  for (unsigned int j=0;j<NSECTORS;j++){
    Sectors[j]=new L1TSector(j);
    for (unsigned int ladder=0;ladder<200;ladder++) { //hack
      for (unsigned int layer=1;layer<11;layer++) {
	if (sectors[j].contain(layer,ladder)>0) {
	  for (unsigned int module=0;module<200;module++){
	    if (!geom.moduleGeometryExists(layer,ladder,module)) continue;
	    const ModuleGeometry& mg=geom.moduleGeometry(layer,ladder,module);
	    
	    Sectors[j]->addGeom(layer,sectors[j].contain(layer,ladder),
				module,ladder,mg.r1(),mg.phi1(),mg.r2(),mg.phi2(),
				sectors[j].sectorCenter());
	  }
	}
      }
    }
  }


  for (int j=0;j<ev.nstubs();j++){

    Stub aStub=ev.stub(j);
    //int simtrackid=ev.simtrackid(aStub);
    int simtrackid=-1;
    int layer=aStub.layer()+1;
    int ladder=aStub.ladder()+1;
    int module=aStub.module();
    
    for (int k=0;k<NSECTORS;k++) {
      int contains=sectors[k].contain(layer,ladder);

      if (contains>0){
	L1TStub tmp(simtrackid,aStub.iphi(),aStub.iz(),
		    layer, ladder, module, aStub.x(), aStub.y(), aStub.z());

	if (layer==1) {
	  if (contains==1) Sectors[k]->addL11I(tmp);
	  if (contains==2) Sectors[k]->addL12I(tmp);
	  if (contains==3) Sectors[k]->addL13I(tmp);
	}
	  
	if (layer==2) {
	  if (contains==1) Sectors[k]->addL11O(tmp);
	  if (contains==2) Sectors[k]->addL12O(tmp);
	  if (contains==3) Sectors[k]->addL13O(tmp);
	}
	
	if (layer==3){ 
	  if (contains==1) Sectors[k]->addL31I(tmp);
	  if (contains==2) Sectors[k]->addL32I(tmp);
	  if (contains==3) Sectors[k]->addL33I(tmp);
	  if (contains==4) Sectors[k]->addL34I(tmp);
	  }
	if (layer==4) {
	  if (contains==1) Sectors[k]->addL31O(tmp);
	  if (contains==2) Sectors[k]->addL32O(tmp);
	  if (contains==3) Sectors[k]->addL33O(tmp);
	  if (contains==4) Sectors[k]->addL34O(tmp);
	}
	
	if (layer==5){ 
	  if (contains==1) Sectors[k]->addL5a1I(tmp);
	  if (contains==2) Sectors[k]->addL5a2I(tmp);
	  if (contains==3) Sectors[k]->addL5a3I(tmp);
	  if (contains==4) Sectors[k]->addL5a4I(tmp);
	}
	if (layer==6){ 
	  if (contains==1) Sectors[k]->addL5a1O(tmp);
	  if (contains==2) Sectors[k]->addL5a2O(tmp);
	  if (contains==3) Sectors[k]->addL5a3O(tmp);
	  if (contains==4) Sectors[k]->addL5a4O(tmp);
	}
	if (layer==7){ 
	  if (contains==1) Sectors[k]->addL5b1I(tmp);
	  if (contains==2) Sectors[k]->addL5b2I(tmp);
	  if (contains==3) Sectors[k]->addL5b3I(tmp);
	  if (contains==4) Sectors[k]->addL5b4I(tmp);
	}
	if (layer==8){ 
	  if (contains==1) Sectors[k]->addL5b1O(tmp);
	  if (contains==2) Sectors[k]->addL5b2O(tmp);
	  if (contains==3) Sectors[k]->addL5b3O(tmp);
	  if (contains==4) Sectors[k]->addL5b4O(tmp);
	}
	
	if (layer==9) {
	  if (contains==1) Sectors[k]->addL51I(tmp);
	  if (contains==2) Sectors[k]->addL52I(tmp);
	  if (contains==3) Sectors[k]->addL53I(tmp);
	  if (contains==4) Sectors[k]->addL54I(tmp);
	  if (contains==5) Sectors[k]->addL55I(tmp);
	  if (contains==6) Sectors[k]->addL56I(tmp);
	  if (contains==7) Sectors[k]->addL57I(tmp);
	}
	if (layer==10) {
	  if (contains==1) Sectors[k]->addL51O(tmp);
	  if (contains==2) Sectors[k]->addL52O(tmp);
	  if (contains==3) Sectors[k]->addL53O(tmp);
	  if (contains==4) Sectors[k]->addL54O(tmp);
	  if (contains==5) Sectors[k]->addL55O(tmp);
	  if (contains==6) Sectors[k]->addL56O(tmp);
	  if (contains==7) Sectors[k]->addL57O(tmp);
	}

      }
      
    }

  }

  for (int k=0;k<NSECTORS;k++) {
    int SL=0; 
    Sectors[k]->findTracklets(SL);      
    Sectors[k]->matchStubs(SL);
    
    // Look for tracks: we need to run matchStubs first,
    // otherwise we find zero tracks
    Sectors[k]->findTracks();
  }

  L1TTracks allTracks=Sectors[0]->getTracks();
  for(int isector=1;isector<NSECTORS;isector++) {
    allTracks.addTracks(Sectors[isector]->getTracks());
  }

  L1TTracks cleanedTracks=allTracks.purged();
  
  for (unsigned itrack=0; itrack<cleanedTracks.size(); itrack++) {
    L1TTrack track=cleanedTracks.get(itrack);

    //L1TkTrackType TkTrack(TkStubs, aSeedTracklet);
    L1TkTrackType TkTrack;
    //double frac;
    //TkTrack.setSimTrackId(track.simtrackid(frac));  FIXME
    //TkTrack.setRadius(1./track.rinv());  FIXME
    //GlobalPoint bsPosition(recoBeamSpotHandle->position().x(),
    //			   recoBeamSpotHandle->position().y(),
    //			   track.z0()
    //			   ); //store the L1 track vertex position 
    GlobalPoint bsPosition(0.0,
			   0.0,
			   track.z0()
			   ); //store the L1 track vertex position 
    //TkTrack.setVertex(bsPosition);  FIXME
    //TkTrack.setChi2RPhi(track.chisq1()); FIXME
    //TkTrack.setChi2ZPhi(track.chisq2()); FIXME
    cout << "L1TrackProducer::analyze Track with pt="<<track.pt(mMagneticFieldStrength)<<endl;
    TkTrack.setMomentum( GlobalVector ( GlobalVector::Cylindrical(fabs(track.pt(mMagneticFieldStrength)), 
								  track.phi0(), 
								  fabs(track.pt(mMagneticFieldStrength))*sinh(track.eta())) ) );

    L1TkTracksForOutput->push_back(TkTrack);

    vector<L1TkStubPtrType> TkStubs;
    L1TTracklet tracklet = track.getSeed();
    vector<L1TStub> stubComponents = tracklet.getStubComponents();
    vector<L1TStub> stubs = track.getStubs();
    //L1TkTrackletType TkTracklet;

    stubMapType::iterator it;
    for (it = stubMap.begin(); it != stubMap.end(); it++) {
      if (it->first == stubComponents[0] || it->first == stubComponents[1]) {
	L1TkStubPtrType TkStub = it->second;
	//if (TkStub->getStack()%2 == 0)
	//  TkTracklet.addStub(0, TkStub);
	//else
	//  TkTracklet.addStub(1, TkStub);
      }
      
      for (int j=0; j<(int)stubs.size(); j++) {
	if (it->first == stubs[j])
	  TkStubs.push_back(it->second);
      }
    }

    L1TkStubsForOutput->push_back( TkStubs );
    //TkTracklet.checkSimTrack();
    //TkTracklet.fitTracklet(mMagneticFieldStrength, GlobalPoint(bsPosition.x(), bsPosition.y(), 0.0), true);
    //L1TkTrackletsForOutput->push_back( TkTracklet );
  }



  // }

  iEvent.put( L1TkStubsForOutput, "L1TkStubs");
  //iEvent.put( L1TkTrackletsForOutput, "L1TkTracklets" );
  iEvent.put( L1TkTracksForOutput, "Level1TkTracks");

  for (unsigned int j=0;j<NSECTORS;j++){
    delete Sectors[j];
  }


} /// End of produce()


// ///////////////////////////
// // DEFINE THIS AS A PLUG-IN
DEFINE_FWK_MODULE(L1TrackProducer);

#endif
