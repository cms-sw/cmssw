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
#include "DataFormats/SiPixelDetId/interface/PXFDetId.h" 
#include "DataFormats/Common/interface/DetSetVector.h"
//
#include "SimDataFormats/SLHC/interface/StackedTrackerTypes.h"
#include "SimDataFormats/SLHC/interface/slhcevent.hh"
#include "SimDataFormats/SLHC/interface/L1TBarrel.hh"
#include "SimDataFormats/SLHC/interface/L1TDisk.hh"
#include "SimDataFormats/SLHC/interface/L1TStub.hh"
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
#include "DataFormats/L1TrackTrigger/interface/TTStub.h"
//
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "DataFormats/Candidate/interface/Candidate.h"
//
#include "DataFormats/GeometryCommonDetAlgo/interface/MeasurementPoint.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "DataFormats/SiPixelDetId/interface/PixelChannelIdentifier.h"
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

  int eventnum;

  /// Containers of parameters passed by python configuration file
  edm::ParameterSet config;

  string geometry_;

  /// ///////////////// ///
  /// MANDATORY METHODS ///
  virtual void beginRun( const edm::Run& run, const edm::EventSetup& iSetup );
  virtual void endRun( const edm::Run& run, const edm::EventSetup& iSetup );
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

  geometry_ = iConfig.getUntrackedParameter<string>("geometry","");
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
void L1TrackProducer::endRun(const edm::Run& run, const edm::EventSetup& iSetup)
{
  /// Things to be done at the exit of the event Loop 

}

////////////
// BEGIN JOB
void L1TrackProducer::beginRun(const edm::Run& run, const edm::EventSetup& iSetup )
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
  edm::Handle<reco::BeamSpot> recoBeamSpotHandle;
  iEvent.getByLabel("BeamSpotFromSim","BeamSpot",recoBeamSpotHandle);
  math::XYZPoint bsPosition=recoBeamSpotHandle->position();

  cout << "L1TrackProducer: B="<<mMagneticFieldStrength
       <<" vx reco="<<bsPosition.x()
       <<" vy reco="<<bsPosition.y()
       <<" vz reco="<<bsPosition.z()
       <<endl;

  SLHCEvent ev;
  ev.setIPx(bsPosition.x());
  ev.setIPy(bsPosition.y());
  eventnum++;

  cout << "Get simtracks"<<endl;

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

  cout << "Get pixel digis"<<endl;

  /////////////////////
  // GET PIXEL DIGIS //
  edm::Handle<edm::DetSetVector<PixelDigi> >         pixelDigiHandle;
  edm::Handle<edm::DetSetVector<PixelDigiSimLink> >  pixelDigiSimLinkHandle;
  iEvent.getByLabel("simSiPixelDigis", pixelDigiHandle);
  iEvent.getByLabel("simSiPixelDigis", pixelDigiSimLinkHandle);

  cout << "Get stubs and clusters"<<endl;

  ////////////////////////
  // GET THE PRIMITIVES //
  edm::Handle<L1TkCluster_PixelDigi_Collection>  pixelDigiL1TkClusterHandle;
  //edm::Handle<L1TkStub_PixelDigi_Collection>     pixelDigiL1TkStubHandle;
  iEvent.getByLabel("L1TkClustersFromPixelDigis", pixelDigiL1TkClusterHandle);
  //iEvent.getByLabel("L1TkStubsFromPixelDigis", "StubsPass", pixelDigiL1TkStubHandle);

  edm::Handle< edmNew::DetSetVector< TTStub< Ref_PixelDigi_ > > > TTStubHandle;
  iEvent.getByLabel( "TTStubsFromPixelDigis", "StubAccepted", TTStubHandle );


  cout << "Will loop over simtracks" <<endl;

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


  std::cout << "Will loop over digis:"<<std::endl;

  DetSetVector<PixelDigi>::const_iterator iterDet;
  for ( iterDet = pixelDigiHandle->begin();
        iterDet != pixelDigiHandle->end();
        iterDet++ ) {

    /// Build Detector Id
    DetId tkId( iterDet->id );
    StackedTrackerDetId stdetid(tkId);
    /// Check if it is Pixel
    if ( tkId.subdetId() == 2 ) {

      PXFDetId pxfId(tkId);
      DetSetVector<PixelDigiSimLink>::const_iterator itDigiSimLink1=pixelDigiSimLinkHandle->find(pxfId.rawId());
      if (itDigiSimLink1!=pixelDigiSimLinkHandle->end()){
	DetSet<PixelDigiSimLink> digiSimLink = *itDigiSimLink1;
	//DetSet<PixelDigiSimLink> digiSimLink = (*pixelDigiSimLinkHandle)[ pxfId.rawId() ];
	DetSet<PixelDigiSimLink>::const_iterator iterSimLink;
	/// Renormalize layer number from 5-14 to 0-9 and skip if inner pixels

	int disk = pxfId.disk();
	
	if (disk<4) {
	  continue;
	}

	disk-=3;
	
	// Layer 0-20
	//DetId digiDetId = iterDet->id;
	//int sensorLayer = 0.5*(2*PXFDetId(digiDetId).layer() + (PXFDetId(digiDetId).ladder() + 1)%2 - 8);
	
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
	    
	  int offset=1000;

	  if (pxfId.side()==1) {
	    offset=2000;
	  }

	  assert(pxfId.panel()==1);

	  vector<int> simtrackids;
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
	ev.addDigi(offset+disk,iterDigi->row(),iterDigi->column(),
		   pxfId.blade(),pxfId.panel(),pxfId.module(),
		   pdPos.x(),pdPos.y(),pdPos.z(),simtrackids);
	}
      }
    }

    if ( tkId.subdetId() == 1 ) {
      /// Get the PixelDigiSimLink corresponding to this one
      PXBDetId pxbId(tkId);
      DetSetVector<PixelDigiSimLink>::const_iterator itDigiSimLink=pixelDigiSimLinkHandle->find(pxbId.rawId());
      if (itDigiSimLink==pixelDigiSimLinkHandle->end()){
	continue;
      }
      DetSet<PixelDigiSimLink> digiSimLink = *itDigiSimLink;
      //DetSet<PixelDigiSimLink> digiSimLink = (*pixelDigiSimLinkHandle)[ pxbId.rawId() ];
      DetSet<PixelDigiSimLink>::const_iterator iterSimLink;
      /// Renormalize layer number from 5-14 to 0-9 and skip if inner pixels
      if ( pxbId.layer() < 5 ) {
	continue;
	
      }

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
	
	/// Loop over PixelDigiSimLink to find the
	/// correct link to the SimTrack collection
	vector<int > simtrackids;
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
	ev.addDigi(sensorLayer,iterDigi->row(),iterDigi->column(),
		   pxbId.layer(),pxbId.ladder(),pxbId.module(),
		   pdPos.x(),pdPos.y(),pdPos.z(),simtrackids);
      }
    }
  }    


  cout << "Will loop over stubs" << endl;

  /// Loop over L1TkStubs
  edmNew::DetSetVector<TTStub<Ref_PixelDigi_> >::const_iterator iterStubDet;
  for ( iterStubDet = TTStubHandle->begin();
	iterStubDet != TTStubHandle->end();
	++iterStubDet ) {

    edmNew::DetSet<TTStub<Ref_PixelDigi_> >::const_iterator iterTTStub;
    for ( iterTTStub = iterStubDet->begin();
	  iterTTStub != iterStubDet->end();
	  iterTTStub++ ) {

      const TTStub<Ref_PixelDigi_>* stub=iterTTStub;

      double stubPt = theStackedGeometry->findRoughPt(mMagneticFieldStrength,stub);
        
      if (stubPt>10000.0) stubPt=9999.99;
      GlobalPoint stubPosition = theStackedGeometry->findGlobalPosition(stub);

      StackedTrackerDetId stubDetId = stub->getDetId();
      unsigned int iStack = stubDetId.iLayer();
      unsigned int iRing = stubDetId.iRing();
      unsigned int iPhi = stubDetId.iPhi();
      unsigned int iZ = stubDetId.iZ();

      std::vector<bool> innerStack;
      std::vector<int> irphi;
      std::vector<int> iz;
      std::vector<int> iladder;
      std::vector<int> imodule;
      

      if (iStack==999999) {
	iStack=1000+iRing;
      }


      /// Get the Inner and Outer L1TkCluster
      std::vector< edm::Ref< edmNew::DetSetVector< TTCluster<Ref_PixelDigi_> >, TTCluster<Ref_PixelDigi_> > >  clusters = stub->getClusterRefs();

      assert(clusters.size()==2);

      edm::Ref< edmNew::DetSetVector< TTCluster<Ref_PixelDigi_> >, TTCluster<Ref_PixelDigi_> > innerClusters=clusters[0];

      const DetId innerDetId = innerClusters->getDetId();

      std::vector<Ref_PixelDigi_> hits=innerClusters->getHits();

      for (unsigned int ihit=0;ihit<hits.size();ihit++){

	std::pair<int,int> rowcol=PixelChannelIdentifier::channelToPixel(hits[ihit]->channel());
	    
	if (iStack<1000) {
	  innerStack.push_back(true);
	  irphi.push_back(rowcol.first);
	  iz.push_back(rowcol.second);
	  iladder.push_back(PXBDetId(innerDetId).ladder());
	  imodule.push_back(PXBDetId(innerDetId).module());
	}
	else {
	  innerStack.push_back(true);
	  irphi.push_back(rowcol.first);
	  iz.push_back(rowcol.second);
	  iladder.push_back(PXFDetId(innerDetId).disk());
	  imodule.push_back(PXFDetId(innerDetId).module());
	}    
      }

      edm::Ref< edmNew::DetSetVector< TTCluster<Ref_PixelDigi_> >, TTCluster<Ref_PixelDigi_> > outerClusters=clusters[1];

      const DetId outerDetId =outerClusters->getDetId();

      hits=outerClusters->getHits();

      for (unsigned int ihit=0;ihit<hits.size();ihit++){

	std::pair<int,int> rowcol=PixelChannelIdentifier::channelToPixel(hits[ihit]->channel());

    
	if (iStack<1000) {
	  innerStack.push_back(false);
	  irphi.push_back(rowcol.first);
	  iz.push_back(rowcol.second);
	  iladder.push_back(PXBDetId(outerDetId).ladder());
	  imodule.push_back(PXBDetId(outerDetId).module());
	}
	else {
	  innerStack.push_back(false);
	  irphi.push_back(rowcol.first);
	  iz.push_back(rowcol.second);
	  iladder.push_back(PXFDetId(outerDetId).disk());
	  imodule.push_back(PXFDetId(outerDetId).module());
	}    
      }    

      ev.addStub(iStack-1,iPhi,iZ,stubPt,
		 stubPosition.x(),stubPosition.y(),stubPosition.z(),
		 innerStack,irphi,iz,iladder,imodule);
        
    }
  }


  std::cout << "Will actually do L1 tracking:"<<std::endl;


  //////////////////////////
  // NOW RUN THE L1 tracking


  int mode = 0;

  // mode means:
  // 1 LB_6PS
  // 2 LB_4PS_2SS
  // 3 EB

  //cout << "geometry:"<<geometry_<<endl;

  if (geometry_=="LB_6PS") mode=1;
  if (geometry_=="LB_4PS_2SS") mode=2;
  if (geometry_=="BE") mode=3;
  if (geometry_=="BE5D") mode=4;



  assert(mode==1||mode==2||mode==3||mode==4);

#include "L1Tracking.icc"  


  
  for (unsigned itrack=0; itrack<purgedTracks.size(); itrack++) {
    L1TTrack track=purgedTracks.get(itrack);

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
    //cout << "L1TrackProducer Track with pt="<<track.pt(mMagneticFieldStrength)<<endl;
    TkTrack.setMomentum( GlobalVector ( GlobalVector::Cylindrical(fabs(track.pt(mMagneticFieldStrength)), 
								  track.phi0(), 
								  fabs(track.pt(mMagneticFieldStrength))*sinh(track.eta())) ) );

    L1TkTracksForOutput->push_back(TkTrack);

    vector<L1TkStubPtrType> TkStubs;
    L1TTracklet tracklet = track.getSeed();
    vector<L1TStub> stubComponents;// = tracklet.getStubComponents();
    vector<L1TStub> stubs = track.getStubs();
    //L1TkTrackletType TkTracklet;

    stubMapType::iterator it;
    //for (it = stubMap.begin(); it != stubMap.end(); it++) {
      //if (it->first == stubComponents[0] || it->first == stubComponents[1]) {
      //L1TkStubPtrType TkStub = it->second;
	//if (TkStub->getStack()%2 == 0)
	//  TkTracklet.addStub(0, TkStub);
	//else
	//  TkTracklet.addStub(1, TkStub);
      //}
      
      //for (int j=0; j<(int)stubs.size(); j++) {
    //	if (it->first == stubs[j])
    //  TkStubs.push_back(it->second);
    //}
    //}

    L1TkStubsForOutput->push_back( TkStubs );
    //TkTracklet.checkSimTrack();
    //TkTracklet.fitTracklet(mMagneticFieldStrength, GlobalPoint(bsPosition.x(), bsPosition.y(), 0.0), true);
    //L1TkTrackletsForOutput->push_back( TkTracklet );
  }



  // }

  iEvent.put( L1TkStubsForOutput, "L1TkStubs");
  //iEvent.put( L1TkTrackletsForOutput, "L1TkTracklets" );
  iEvent.put( L1TkTracksForOutput, "Level1TkTracks");

} /// End of produce()


// ///////////////////////////
// // DEFINE THIS AS A PLUG-IN
DEFINE_FWK_MODULE(L1TrackProducer);

#endif
