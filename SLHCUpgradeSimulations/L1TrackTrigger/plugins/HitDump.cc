//////////////////////////
//  Analyzer by Nicola  //
//    july 2010 @ PD    //
//////////////////////////

/////////////////////////
//       HEADERS       //
/////////////////////////

////////////////
// CLASS HEADER
// No more necessary in the current "no *.h file" implementation

////////////////////
// FRAMEWORK HEADERS
#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
//
#include "FWCore/Framework/interface/EDAnalyzer.h"
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
//#include "SimDataFormats/SLHC/interface/L1CaloCluster.h"
//
#include "DataFormats/L1Trigger/interface/L1EmParticle.h"
#include "DataFormats/L1Trigger/interface/L1EmParticleFwd.h"
#include "DataFormats/L1Trigger/interface/L1JetParticle.h"
#include "DataFormats/L1Trigger/interface/L1JetParticleFwd.h"
//
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "DataFormats/Candidate/interface/Candidate.h"
//
#include "DataFormats/GeometryCommonDetAlgo/interface/MeasurementPoint.h"
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


////////////////
// PHYSICS TOOLS
#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "RecoTracker/TkSeedGenerator/interface/FastHelix.h"
#include "TrackingTools/TrajectoryState/interface/FreeTrajectoryState.h"
#include "RecoTauTag/TauTagTools/interface/GeneratorTau.h"
//
#include "DataFormats/GeometryCommonDetAlgo/interface/MeasurementVector.h"
#include "DataFormats/GeometrySurface/interface/BoundPlane.h"
//#include "SLHCUpgradeSimulations/Utilities/interface/constants.h"

///////////////
// ROOT HEADERS
#include <TROOT.h>
#include <TTree.h>
#include <TFile.h>
#include <TF1.h>
#include <TH2F.h>
#include <TH1F.h>
#include <TH2D.h>
#include <TH1D.h>
#include <TH2.h>
#include <TH1.h>

//////////////
// STD HEADERS
#include <memory>
#include <string>
#include <iostream>
#include <fstream>

//////////////
// NAMESPACES
// I hate them to be used this way because
// I lose the feel of 'what is from where'
// but I need them, unfortunately...
using namespace std;
using namespace edm;
using namespace reco;
//using namespace cmsUpgrades;
//using namespace l1slhc;
using namespace l1extra;

//////////////////////////////
//                          //
//     CLASS DEFINITION     //
//                          //
//////////////////////////////

class TTree;
class TFile;
class TH1D;
class TH2D;
class TGraph;
class RectangularPixelTopology;
class TransientInitialStateEstimator;
class MagneticField;
class TrackerGeometry;
class TrajectoryStateOnSurface;
class PTrajectoryStateOnDet;
//
class HitDump : public edm::EDAnalyzer
{
  /// Public methods
  public:
    /// Constructor/destructor
    explicit HitDump(const edm::ParameterSet& iConfig);
    virtual ~HitDump();
    // Typical methods used on Loops over events
    virtual void beginJob();
    virtual void endJob();
    virtual void analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup);

    /// Some Type definitions

  /// Protected methods only internally used
  protected:
                     
  /// Private methods and variables
  private:

    edm::InputTag L1TkTrackletCollInputTag;
    
    bool testedGeometry;
    bool debugPrintouts;

    int eventnum;

    /// TO CHECK ALL EVENTS ARE PROCESSED
    TH1D* h_EvtCnt;

    /// Global Position of Digis
    TH2D* hDigi_XY;
    TH2D* hDigi_RZ;
    
    TH2D* hDet_LayMod;
    TH2D* hDet_LayLad;
    TH2D* hDet_LadMod;

    /// TO CHECK GEOMETRY
    TH2D* hGeom_Layer_R;
    TH2D* hGeom_iPhi_Phi;
    TH2D* hGeom_iZ_Z;


    /// Containers of parameters passed by python
    /// configuration file
    edm::ParameterSet config;

  string fileString_ ;
  std::ofstream myfile;
};


//////////////////////////////////
//                              //
//     CLASS IMPLEMENTATION     //
//                              //
//////////////////////////////////

//////////////
// CONSTRUCTOR
HitDump::HitDump(edm::ParameterSet const& iConfig) : 
  config(iConfig)
{
  /// Insert here what you need to initialize
  L1TkTrackletCollInputTag = iConfig.getParameter< edm::InputTag >("L1TkTrackletCollType");
  fileString_ = iConfig.getUntrackedParameter<string>("fileString","ForUlrich.txt");
}

/////////////
// DESTRUCTOR
HitDump::~HitDump()
{
  /// Insert here what you need to delete
  /// when you close the class instance
  myfile.close();
}  

//////////
// END JOB
void HitDump::endJob()//edm::Run& run, const edm::EventSetup& iSetup
{
  /// Things to be done at the exit of the event Loop
  std::cerr << " HitDump::endJob" << std::endl;
  /// End of things to be done at the exit from the event Loop
}

////////////
// BEGIN JOB
void HitDump::beginJob()
{

  /// Initialize all slave variables
  /// mainly histogram ranges and resolution
  //double trkMPt = 99999.9;

  eventnum=0;

  cout<<"HitDump::beginJob opening file:"<<fileString_<<endl;
  myfile.open(fileString_.c_str());


  testedGeometry = false;
  debugPrintouts = false;

  std::ostringstream histoName;
  std::ostringstream histoTitle;


  /// End of things to be done before entering the event Loop
}

//////////
// ANALYZE
void HitDump::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{


  eventnum++;

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
  ////////////////////////
  edm::ESHandle<MagneticField> magneticFieldHandle;
  iSetup.get<IdealMagneticFieldRecord>().get(magneticFieldHandle);
  const MagneticField* theMagneticField = magneticFieldHandle.product();
  double mMagneticFieldStrength = theMagneticField->inTesla(GlobalPoint(0,0,0)).z();

  ////////////
  // GET BS //
  ////////////
  //edm::Handle< std::vector< cmsUpgrades::L1TkBeam > > beamHandle;
  //iEvent.getByLabel( "L1TkBeams", beamHandle );
  GlobalPoint beamPos(0.0,0.0,0.0);

  ///////////////////
  // GET SIMTRACKS //
  ///////////////////
  edm::Handle<edm::SimTrackContainer>   simTrackHandle;
  edm::Handle<edm::SimVertexContainer>  simVtxHandle;
  //iEvent.getByLabel( "famosSimHits", simTrackHandle );
  //iEvent.getByLabel( "famosSimHits", simVtxHandle );
  iEvent.getByLabel( "g4SimHits", simTrackHandle );
  iEvent.getByLabel( "g4SimHits", simVtxHandle );

  //////////////////////
  // GET MC PARTICLES //
  //////////////////////
  edm::Handle<reco::GenParticleCollection> genpHandle;
  iEvent.getByLabel( "genParticles", genpHandle );


  /////////////////////
  // GET PIXEL DIGIS //
  /////////////////////
  edm::Handle<edm::DetSetVector<PixelDigi> >         pixelDigiHandle;
  edm::Handle<edm::DetSetVector<PixelDigiSimLink> >  pixelDigiSimLinkHandle;
  iEvent.getByLabel("simSiPixelDigis", pixelDigiHandle);
  iEvent.getByLabel("simSiPixelDigis", pixelDigiSimLinkHandle);


  ////////////////////////
  // GET THE PRIMITIVES //
  ////////////////////////
  edm::Handle<L1TkCluster_PixelDigi_Collection>  pixelDigiL1TkClusterHandle;
  edm::Handle<L1TkStub_PixelDigi_Collection>     pixelDigiL1TkStubHandle;
  //edm::Handle<L1TkTracklet_PixelDigi_Collection> pixelDigiL1TkTrackletHandle;
  edm::Handle<L1TkTrack_PixelDigi_Collection>    pixelDigiL1TkTrackHandle;
  iEvent.getByLabel("L1TkClustersFromPixelDigis", pixelDigiL1TkClusterHandle);
  iEvent.getByLabel("L1TkStubsFromPixelDigis","StubsPass",    pixelDigiL1TkStubHandle);
  //iEvent.getByLabel("L1TkTrackletsFromPixelDigis", "ShortTrackletsVtx00HelFit", pixelDigiL1TkTrackletHandle);
  //iEvent.getByLabel( L1TkTrackletCollInputTag, pixelDigiL1TkTrackletHandle);
  //iEvent.getByLabel("L1TkTracksFromPixelDigis",    "Level1TracksHelFitVtxYes", pixelDigiL1TkTrackHandle);

  // dump map between inner and outer modules

  static bool first=true;

  if (first) {
    
    first=false;
    
    //edm::ESHandle<cmsUpgrades::StackedTrackerGeometry> stackedGeometryHandle;
    //const cmsUpgrades::StackedTrackerGeometry* theStackedGeometry;
    //cmsUpgrades::StackedTrackerGeometry::StackContainerIterator
    //  StackedTrackerIterator;
    
    /// Geometry setup
    /// Set pointers to Geometry
    //iSetup.get<TrackerDigiGeometryRecord>().get(geometryHandle);
    //theGeometry = &(*geometryHandle);
    /// Set pointers to Stacked Modules
    // iSetup.get<StackedTrackerGeometryRecord>().get(stackedGeometryHandle);
    //theStackedGeometry = stackedGeometryHandle.product(); /// Note this
    //is different
      /// from the  "global" geometry
      

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

      int iStack=stackDetId.iLayer()+1;

      if (iStack==1000000){
	iStack=stackDetId.iRing()+1000;
      }

      if (myfile.is_open()) {      
	myfile << "Map: " 
	       << iStack << "\t" 
	       << stackDetId.iPhi()+1 << "\t" 
	       << stackDetId.iZ() << "\t" 
	       << iStack << "\t" 
	       << stackDetId.iPhi()+1 << "\t" 
	       << stackDetId.iZ() << "\t"
	       << pdPos0.x() << "\t"<< pdPos0.y() << "\t"<< pdPos0.z() << "\t"
	       << pdPos1.x() << "\t"<< pdPos1.y() << "\t"<< pdPos1.z() << "\t"
	       << pdPos2.x() << "\t"<< pdPos2.y() << "\t"<< pdPos2.z() << "\t"
	       << endl; 
      }	

      detIdToInnerMap.insert( make_pair(detId0, true) );
      detIdToInnerMap.insert( make_pair(detId1, false) );
    }
    myfile << "EndMap" <<endl;

  }


  if (myfile.is_open()) {
    myfile << "Event: "<<eventnum<<std::endl;
  }


  /// Loop over SimTracks
  SimTrackContainer::const_iterator iterSimTracks;
  for ( iterSimTracks = simTrackHandle->begin();
	iterSimTracks != simTrackHandle->end();
	++iterSimTracks ) {
    
    /// Get the corresponding vertex
    int vertexIndex = iterSimTracks->vertIndex();
    const SimVertex& theSimVertex = (*simVtxHandle)[vertexIndex];
    math::XYZTLorentzVectorD trkVtxPos = theSimVertex.position();
    GlobalPoint trkVtxCorr = GlobalPoint( trkVtxPos.x() - beamPos.x(), 
					  trkVtxPos.y() - beamPos.y(), 
					  trkVtxPos.z() - beamPos.z() );
    
    if (myfile.is_open()) {
      double pt=iterSimTracks->momentum().pt();
      if (pt!=pt) pt=9999.999;
      myfile << "SimTrack: " 
	     << iterSimTracks->trackId() << "\t" 
	     << iterSimTracks->type() << "\t" 
	     << pt << "\t" 
	     << iterSimTracks->momentum().eta() << "\t" 
	     << iterSimTracks->momentum().phi() << "\t" 
	     << trkVtxCorr.x() << "\t" 
	     << trkVtxCorr.y() << "\t" 
	     << trkVtxCorr.z() << "\t" 
	     << std::endl;
    }
    
  } /// End of Loop over SimTracks

  if (myfile.is_open()) {
    myfile << "SimTrackEnd"<<endl;
  }
 

  /*
  /// Loop over L1TkCluster hits
  for ( clusterHitsIter = clusterHits.begin();
	clusterHitsIter != clusterHits.end();
	clusterHitsIter++ ) {

  }
  */

  // End loop over L1TkClusters

  /// Loop over Detector Modules
  DetSetVector<PixelDigi>::const_iterator iterDet;
  for ( iterDet = pixelDigiHandle->begin();
        iterDet != pixelDigiHandle->end();
        iterDet++ ) {

    /// Build Detector Id
    DetId tkId( iterDet->id );
    StackedTrackerDetId stdetid(tkId);
    /// Check if it is Pixel
    if ( tkId.subdetId() == 2 ) {

      //cout << "Will create pxfId"<<endl;
      PXFDetId pxfId(tkId);
      DetSetVector<PixelDigiSimLink>::const_iterator itDigiSimLink1=pixelDigiSimLinkHandle->find(pxfId.rawId());
      if (itDigiSimLink1!=pixelDigiSimLinkHandle->end()){
	//cout << "Found forward digisim link"<<endl;
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

	  if (myfile.is_open()) {
	    myfile << "Digi: " 
		   << offset+disk << "\t" 
		   << iterDigi->row() << "\t" 
		   << iterDigi->column() << "\t" 
		   << pxfId.blade() << "\t" 
		   << pxfId.panel() << "\t" 
	      //     << stdetid.iPhi() << "\t" 
		   << pxfId.module() << "\t" 
		   << pdPos.x() << "\t"
		   << pdPos.y() << "\t"
		   << pdPos.z() << "\t"
		   << std::endl;
	  }
	  
	  /// Loop over PixelDigiSimLink to find the
	  /// correct link to the SimTrack collection
	  for ( iterSimLink = digiSimLink.data.begin();
		iterSimLink != digiSimLink.data.end();
		iterSimLink++) {
	    
	    /// When the channel is the same, the link is found
	    if ( (int)iterSimLink->channel() == iterDigi->channel() ) {
	      
	      /// Map wrt SimTrack Id
	      unsigned int simTrackId = iterSimLink->SimTrackId();
	      if (myfile.is_open()) {
		myfile << "SimTrackId: "<<simTrackId<<std::endl;
	      }
	    }
	  }
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
	
	
	if (myfile.is_open()) {
	  myfile << "Digi: " 
		 << sensorLayer << "\t" 
		 << iterDigi->row() << "\t" 
		 << iterDigi->column() << "\t" 
		 << pxbId.layer() << "\t" 
		 << pxbId.ladder() << "\t" 
	    //     << stdetid.iPhi() << "\t" 
		 << pxbId.module() << "\t" 
		 << pdPos.x() << "\t"
		 << pdPos.y() << "\t"
		 << pdPos.z() << "\t"
		 << std::endl;
	}
	
	/// Loop over PixelDigiSimLink to find the
	/// correct link to the SimTrack collection
	for ( iterSimLink = digiSimLink.data.begin();
	      iterSimLink != digiSimLink.data.end();
	      iterSimLink++) {
	  
	  /// When the channel is the same, the link is found
	  if ( (int)iterSimLink->channel() == iterDigi->channel() ) {
	    
	    /// Map wrt SimTrack Id
	    unsigned int simTrackId = iterSimLink->SimTrackId();
	    if (myfile.is_open()) {
	      myfile << "SimTrackId: "<<simTrackId<<std::endl;
	    }
	  }
	}
      }
    }
  }    

  if (myfile.is_open()) {
    myfile << "DigiEnd"<<endl;
  }

  


  /// Loop over L1TkStubs
  L1TkStub_PixelDigi_Collection::const_iterator iterL1TkStub;
  for ( iterL1TkStub = pixelDigiL1TkStubHandle->begin();
	iterL1TkStub != pixelDigiL1TkStubHandle->end();
	++iterL1TkStub ) {

    double stubPt = theStackedGeometry->findRoughPt(mMagneticFieldStrength,&(*iterL1TkStub));
						    
    if (stubPt>10000.0) stubPt=9999.99;
    GlobalPoint stubPosition = theStackedGeometry->findGlobalPosition(&(*iterL1TkStub));

    StackedTrackerDetId stubDetId = iterL1TkStub->getDetId();
    unsigned int iStack = stubDetId.iLayer();
    unsigned int iRing = stubDetId.iRing();
    unsigned int iPhi = stubDetId.iPhi();
    unsigned int iZ = stubDetId.iZ();

    //const StackedTrackerDetUnit* aDetUnit = theStackedGeometry->idToStack(stubDetId);
    //DetId id0 = aDetUnit->stackMember(0);
    //DetId id1 = aDetUnit->stackMember(1);

    //PXBDetId pxb0 = PXBDetId(id0);
    //PXBDetId pxb1 = PXBDetId(id1);

    if (iStack==999999) {
      iStack=1000+iRing;
    }

    if (myfile.is_open()) {
      myfile << "Stub: "<<
	iStack<<"\t"<<
	iPhi<<"\t"<<
	iZ<<"\t"<<
	stubPt<<"\t"<<
	stubPosition.x()<<"\t"<<
	stubPosition.y()<<"\t"<<
	stubPosition.z()<<"\t"<<
	std::endl;
    }

    /// Get the Inner and Outer L1TkCluster
    edm::Ptr<L1TkCluster_PixelDigi_> innerCluster = iterL1TkStub->getClusterPtr(0);

    const DetId innerDetId = theStackedGeometry->idToDet( innerCluster->getDetId(), 0 )->geographicalId();

    for (unsigned int ihit=0;ihit<innerCluster->getHits().size();ihit++){

      std::pair<int,int> rowcol=PixelChannelIdentifier::channelToPixel(innerCluster->getHits().at(ihit)->channel());
    
      if (myfile.is_open()) {
	if (iStack<1000) {
	  myfile << "InnerStackDigi: " 
		 << rowcol.first << "\t" 
		 << rowcol.second << "\t" 
		 << PXBDetId(innerDetId).ladder() << "\t" 
		 << PXBDetId(innerDetId).module() << "\t" 
		 << std::endl;
	}
	else {
	  myfile << "InnerStackDigi: " 
		 << rowcol.first << "\t" 
		 << rowcol.second << "\t" 
		 << PXFDetId(innerDetId).disk() << "\t" 
		 << PXFDetId(innerDetId).module() << "\t" 
		 << std::endl;
	}
      }    
    }


    edm::Ptr<L1TkCluster_PixelDigi_> outerCluster = iterL1TkStub->getClusterPtr(1);
      
    const DetId outerDetId = theStackedGeometry->idToDet( outerCluster->getDetId(), 1 )->geographicalId();

    for (unsigned int ihit=0;ihit<outerCluster->getHits().size();ihit++){

      std::pair<int,int> rowcol=PixelChannelIdentifier::channelToPixel(outerCluster->getHits().at(ihit)->channel());
    
      if (myfile.is_open()) {
	if (iStack<1000) {
	  myfile << "OuterStackDigi: " 
		 << rowcol.first << "\t" 
		 << rowcol.second << "\t" 
		 << PXBDetId(outerDetId).ladder() << "\t" 
		 << PXBDetId(outerDetId).module() << "\t" 
		 << std::endl;
	}
	else {
	  myfile << "OuterStackDigi: " 
		 << rowcol.first << "\t" 
		 << rowcol.second << "\t" 
		 << PXFDetId(outerDetId).disk() << "\t" 
		 << PXFDetId(outerDetId).module() << "\t" 
		 << std::endl;
	}
      }
    }    
    
    
    /*      
          
    /// Loop over Detector Modules
    DetSetVector<PixelDigi>::const_iterator iterDet;
    for ( iterDet = pixelDigiHandle->begin();
	  iterDet != pixelDigiHandle->end();
	  iterDet++ ) {
      
      /// Build Detector Id
      DetId tkId( iterDet->id );
      
      
      if (innerDetId.rawId()==outerDetId.rawId()) {
	std::cerr<<"STUB DEBUGGING INNER LAYER == OUTER LAYER RAW ID"<<std::endl;
      }
    
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
	      if (myfile.is_open()) {
		if (iStack<1000) {
		  myfile << "InnerStackDigi: " 
			 << iterDigi->row() << "\t" 
			 << iterDigi->column() << "\t" 
			 << PXBDetId(tkId).ladder() << "\t" 
			 << PXBDetId(tkId).module() << "\t" 
			 << std::endl;
		}
		else {
		  myfile << "InnerStackDigi: " 
			 << iterDigi->row() << "\t" 
			 << iterDigi->column() << "\t" 
			 << PXFDetId(tkId).disk() << "\t" 
			 << PXFDetId(tkId).module() << "\t" 
			 << std::endl;
		}
	      }
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
	      if (myfile.is_open()) {
		if (iStack<1000) {
		  myfile << "OuterStackDigi: " 
			 << iterDigi->row() << "\t" 
			 << iterDigi->column() << "\t" 
			 << PXBDetId(tkId).ladder() << "\t" 
			 << PXBDetId(tkId).module() << "\t" 
			 << std::endl;
		}
		else{
		  myfile << "OuterStackDigi: " 
			 << iterDigi->row() << "\t" 
			 << iterDigi->column() << "\t" 
			 << PXFDetId(tkId).disk() << "\t" 
			 << PXFDetId(tkId).module() << "\t" 
			 << std::endl;
		}
	      }
	    }
	  }     
	}
    */  
  }

      /*
     /// Get the PixelDigiSimLink corresponding to this one
     PXBDetId pxbId(tkId);
     /// Renormalize layer number from 5-14 to 0-9 and skip if inner pixels
     if ( pxbId.layer() < 5 ) continue;
     //std::cerr<<std::endl;
     
     unsigned int whichStack = pxbId.layer() - 5;
     //std::cerr<<iStack<<"\t"<<whichStack<<"\t"<<innerCluster.getStack()<<"\t"<<outerCluster.getStack()<<std::endl;
     if (whichStack != innerCluster.getStack()) continue;
     if (whichStack != outerCluster.getStack()) continue;

     unsigned int whichPhi = floor(pxbId.ladder() / 2);
     //std::cerr<<iPhi<<"\t"<<whichPhi<<"\t"<<innerCluster.getLadderPhi()<<"\t"<<outerCluster.getLadderPhi()<<std::endl;
     if (whichPhi != innerCluster.getLadderPhi()) continue;
     if (whichPhi != outerCluster.getLadderPhi()) continue;
     
     unsigned int whichZ = pxbId.module();
     //std::cerr<<iZ<<"\t"<<whichZ<<"\t"<<innerCluster.getLadderZ()<<"\t"<<outerCluster.getLadderZ()<<std::endl;
     if (whichZ != innerCluster.getLadderZ()) continue;
     if (whichZ != outerCluster.getLadderZ()) continue;
      */

  if (myfile.is_open()) {
    myfile << "StubEnd"<<endl;
  }


  /// Functions that gets called by framework every event
  //h_EvtCnt->Fill(iEvent.id().event()); /// The +0.2 is to be sure of being in the correct bin


  /// /////////////////////
  /// Test the Geometry ///
  /// of The Tracker    ///
  /// /////////////////////
  /// Do this only once, please
  std::vector<StackedTrackerDetUnit*> stackContainer = theStackedGeometry->stacks();
  if (testedGeometry==false) {
    /// Loop over Detector Pieces
    for ( unsigned int k=0; k<stackContainer.size(); k++ ) {
      //StackedTrackerDetUnit* detUnitIt = stackContainer.at(k);
      //StackedTrackerDetId stackDetId = detUnitIt->Id();
      //int layer = stackDetId.layer();
      //int iPhi  = stackDetId.iPhi();
      //int iZ    = stackDetId.iZ();
      //int doublestack;
      //if (layer%2==0) doublestack = layer/2;
      //else doublestack = (layer-1)/2;
      /*
	DetId testDet(stackDetId.rawId());
	PXBDetId testDetId(stackDetId.rawId());
	std::cerr<<"layer "<<layer<<" "<<testDetId.layer()<<std::endl;
	
	uint32_t uPhi = testDetId.ladder();
	if ( uPhi > 768 ) uPhi -= 768;
	else if ( uPhi > 512 ) uPhi -= 512;
	else if ( uPhi > 256 ) uPhi -= 256;

	uint32_t uZ = testDetId.module();
	if ( uZ%2 == 0 ) uZ = uZ/2;
	else uZ = (uZ-1)/2;

	std::cerr<<"ORIGINAL:"<<std::endl;
	std::cerr<<stackDetId<<std::endl;

	StackedTrackerDetId pippo( 1, testDetId.layer(), uPhi, uZ );
	std::cerr<<"TEST:"<<std::endl;
	std::cerr<<pippo<<std::endl;
      */
      //const GeomDetUnit* detUnit = theStackedGeometry->idToDetUnit( stackDetId, layer%2 );
      //GlobalPoint zeroZeroPoint = detUnit->toGlobal( detUnit->topology().localPosition( MeasurementPoint( 0, 0) ) );
      //hGeom_Layer_R->Fill( zeroZeroPoint.perp(), layer );
      //hGeom_iPhi_Phi->Fill( zeroZeroPoint.phi(), iPhi );
      //hGeom_iZ_Z->Fill( zeroZeroPoint.z(), iZ );

    } /// End of Loop over Detector Pieces
    /// End of Test the Geometry of The Tracker
    testedGeometry = true;
  }


} /// End of analyze()

/////////////////////////////////
//                             //
// SOME OTHER CUSTOM FUNCTIONS //
//                             //
/////////////////////////////////






///////////////////////////
// DEFINE THIS AS A PLUG-IN
DEFINE_FWK_MODULE(HitDump);




