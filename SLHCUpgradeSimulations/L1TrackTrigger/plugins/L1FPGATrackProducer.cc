//////////////////////////
//  Producer by Anders  //
//     and Emmanuele    //
//    july 2012 @ CU    //
//////////////////////////


#ifndef L1TFPGATRACK_PRDC_H
#define L1TFPGATRACK_PRDC_H

////////////////////
// FRAMEWORK HEADERS
#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
//
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/FileInPath.h" 
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
#include "SLHCUpgradeSimulations/L1TrackTrigger/interface/L1TStub.hh"
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
#include "DataFormats/L1TrackTrigger/interface/TTTrack.h"
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

///////////////
// FPGA emulation
#include "SLHCUpgradeSimulations/L1TrackTrigger/interface/FPGAConstants.hh"
#include "SLHCUpgradeSimulations/L1TrackTrigger/interface/FPGASector.hh"
#include "SLHCUpgradeSimulations/L1TrackTrigger/interface/FPGAWord.hh"
#include "SLHCUpgradeSimulations/L1TrackTrigger/interface/FPGATimer.hh"

////////////////
// PHYSICS TOOLS
#include "RecoTracker/TkSeedGenerator/interface/FastHelix.h"
#include "TrackingTools/TrajectoryState/interface/FreeTrajectoryState.h"
//
#include "DataFormats/GeometryCommonDetAlgo/interface/MeasurementVector.h"
#include "DataFormats/GeometrySurface/interface/BoundPlane.h"

#include "SLHCUpgradeSimulations/L1TrackTrigger/interface/StubPtConsistency.h"

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


//////////////////////////////
//                          //
//     CLASS DEFINITION     //
//                          //
//////////////////////////////

/////////////////////////////////////
// this class is needed to make a map
// between different types of stubs
struct L1TStubCompare 
{
public:
  bool operator()(const L1TStub& x, const L1TStub& y) const {
    if (x.layer() != y.layer()) return (y.layer()>x.layer());
    else {
      if (x.ladder() != y.ladder()) return (y.ladder()>x.ladder());
      else {
	if (x.module() != y.module()) return (y.module()>x.module());
	else {
	  if (x.iz() != y.iz()) return (y.iz()>x.iz());
	  else return (x.iphi()>y.iphi());
	}
      }
    }
  }
};


class L1FPGATrackProducer : public edm::EDProducer
{
public:

  typedef L1TkStub_PixelDigi_                           L1TkStubType;
  typedef std::vector< L1TkStubType >                                L1TkStubCollectionType;
  typedef edm::Ptr< L1TkStubType >                                   L1TkStubPtrType;
  typedef std::vector< L1TkStubPtrType >                             L1TkStubPtrCollection;
  typedef std::vector< L1TkStubPtrCollection >                       L1TkStubPtrCollVectorType;


  typedef L1TkTrack_PixelDigi_                          L1TkTrackType;
  typedef std::vector< L1TkTrackType >                               L1TkTrackCollectionType;
  typedef std::vector< L1TTrack >                                    L1TrackCollectionType;

  /// Constructor/destructor
  explicit L1FPGATrackProducer(const edm::ParameterSet& iConfig);
  virtual ~L1FPGATrackProducer();

protected:
                     
private:

  int eventnum;

  /// Containers of parameters passed by python configuration file
  edm::ParameterSet config;

  /// File path for configuration files
  edm::FileInPath fitPatternFile; 
  edm::FileInPath memoryModulesFile; 
  edm::FileInPath processingModulesFile; 
  edm::FileInPath wiresFile; 


  double phiWindowSF_;

  string asciiEventOutName_;
  std::ofstream asciiEventOut_;

  FPGASector** sectors;

  /// ///////////////// ///
  /// MANDATORY METHODS ///
  virtual void beginRun( const edm::Run& run, const edm::EventSetup& iSetup );
  virtual void endRun( const edm::Run& run, const edm::EventSetup& iSetup );
  virtual void produce( edm::Event& iEvent, const edm::EventSetup& iSetup );
};


//////////////
// CONSTRUCTOR
L1FPGATrackProducer::L1FPGATrackProducer(edm::ParameterSet const& iConfig) // :   config(iConfig)
{

  produces< std::vector< TTTrack< Ref_PixelDigi_ > > >( "Level1TTTracks" ).setBranchAlias("Level1TTTracks");

  phiWindowSF_ = iConfig.getUntrackedParameter<double>("phiWindowSF",1.0);

  asciiEventOutName_ = iConfig.getUntrackedParameter<string>("asciiFileName","");

  fitPatternFile = iConfig.getParameter<edm::FileInPath> ("fitPatternFile");
  processingModulesFile = iConfig.getParameter<edm::FileInPath> ("processingModulesFile");
  memoryModulesFile = iConfig.getParameter<edm::FileInPath> ("memoryModulesFile");
  wiresFile = iConfig.getParameter<edm::FileInPath> ("wiresFile");


  eventnum=0;
  if (asciiEventOutName_!="") {
    asciiEventOut_.open(asciiEventOutName_.c_str());
  }

  sectors=new FPGASector*[NSector];

  for (unsigned int i=0;i<NSector;i++) {
    sectors[i]=new FPGASector(i);
  }  

  cout << "fit pattern :     "<<fitPatternFile.fullPath()<<endl;
  cout << "process modules : "<<processingModulesFile.fullPath()<<endl;
  cout << "memory modules :  "<<memoryModulesFile.fullPath()<<endl;
  cout << "wires          :  "<<wiresFile.fullPath()<<endl;

  fitpatternfile=fitPatternFile.fullPath();

  cout << "Will read memory modules file"<<endl;

  ifstream inmem(memoryModulesFile.fullPath().c_str());
  assert(inmem.good());

  while (inmem.good()){
    string memType, memName, size;
    inmem >>memType>>memName>>size;
    if (!inmem.good()) continue;
    if (writetrace) {
      cout << "Read memory: "<<memType<<" "<<memName<<endl;
    }
    for (unsigned int i=0;i<NSector;i++) {
      sectors[i]->addMem(memType,memName);
    }
    
  }


  cout << "Will read processing modules file"<<endl;

  ifstream inproc(processingModulesFile.fullPath().c_str());
  assert(inproc.good());

  while (inproc.good()){
    string procType, procName;
    inproc >>procType>>procName;
    if (!inproc.good()) continue;
    if (writetrace) {
      cout << "Read process: "<<procType<<" "<<procName<<endl;
    }
    for (unsigned int i=0;i<NSector;i++) {
      sectors[i]->addProc(procType,procName);
    }
    
  }


  cout << "Will read wiring information"<<endl;


  ifstream inwire(wiresFile.fullPath().c_str());
  assert(inwire.good());


  while (inwire.good()){
    string line;
    getline(inwire,line);
    if (!inwire.good()) continue;
    if (writetrace) {
      cout << "Line : "<<line<<endl;
    }
    stringstream ss(line);
    string mem,tmp1,procin,tmp2,procout;
    ss>>mem>>tmp1>>procin;
    //cout <<"procin : "<<procin<<endl;
    if (procin=="output=>") {
      procin="";
      ss>>procout;
    }
    else{
      ss>>tmp2>>procout;
    }

    for (unsigned int i=0;i<NSector;i++) {
      sectors[i]->addWire(mem,procin,procout);
    }
  

  }



}

/////////////
// DESTRUCTOR
L1FPGATrackProducer::~L1FPGATrackProducer()
{
  /// Insert here what you need to delete
  /// when you close the class instance
  if (asciiEventOutName_!="") {
    asciiEventOut_.close();
  }

}  

//////////
// END JOB
void L1FPGATrackProducer::endRun(const edm::Run& run, const edm::EventSetup& iSetup)
{

  /// Things to be done at the exit of the event Loop 

}

////////////
// BEGIN JOB
void L1FPGATrackProducer::beginRun(const edm::Run& run, const edm::EventSetup& iSetup )
{
}

//////////
// PRODUCE
void L1FPGATrackProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{

  typedef std::map< L1TStub, edm::Ref< edmNew::DetSetVector< TTStub< Ref_PixelDigi_ > >, TTStub< Ref_PixelDigi_ >  >, L1TStubCompare > stubMapType;

  /// Prepare output
  //std::auto_ptr< L1TkStubPtrCollVectorType > L1TkStubsForOutput( new L1TkStubPtrCollVectorType );
  std::auto_ptr< std::vector< TTTrack< Ref_PixelDigi_ > > > L1TkTracksForOutput( new std::vector< TTTrack< Ref_PixelDigi_ > > );

  stubMapType stubMap;

  /// Geometry handles etc
  edm::ESHandle<StackedTrackerGeometry>           stackedGeometryHandle;
  const StackedTrackerGeometry*                   theStackedGeometry;
  StackedTrackerGeometry::StackContainerIterator  StackedTrackerIterator;

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

  //cout << "L1FPGATrackProducer: B="<<mMagneticFieldStrength
  //     <<" vx reco="<<bsPosition.x()
  //     <<" vy reco="<<bsPosition.y()
  //     <<" vz reco="<<bsPosition.z()
  //     <<endl;

  SLHCEvent ev;
  ev.setIPx(bsPosition.x());
  ev.setIPy(bsPosition.y());
  eventnum++;

  //cout << "Get simtracks"<<endl;

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

  //cout << "Get stubs and clusters"<<endl;

  ////////////////////////
  // GET THE PRIMITIVES //
  edm::Handle<L1TkCluster_PixelDigi_Collection>  pixelDigiL1TkClusterHandle;
  //edm::Handle<L1TkStub_PixelDigi_Collection>     pixelDigiL1TkStubHandle;
  iEvent.getByLabel("L1TkClustersFromPixelDigis", pixelDigiL1TkClusterHandle);
  //iEvent.getByLabel("L1TkStubsFromPixelDigis", "StubsPass", pixelDigiL1TkStubHandle);

  edm::Handle< edmNew::DetSetVector< TTStub< Ref_PixelDigi_ > > > TTStubHandle;
  iEvent.getByLabel( "TTStubsFromPixelDigis", "StubAccepted", TTStubHandle );


  //cout << "Will loop over simtracks" <<endl;

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
    //std::cout << "Simtrack pt eta phi : "<<pt<<" "
    //	      <<iterSimTracks->momentum().eta()<<" "
    //	      <<iterSimTracks->momentum().phi()<<std::endl;
      
    ev.addL1SimTrack(iterSimTracks->trackId(),iterSimTracks->type(),pt,
		     iterSimTracks->momentum().eta(), 
		     iterSimTracks->momentum().phi(), 
		     trkVtxCorr.x(),
		     trkVtxCorr.y(),
		     trkVtxCorr.z());
   
    
  } /// End of Loop over SimTracks


  //cout << "Will loop over stubs" << endl;

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

      StackedTrackerDetId stubDetId = stub->getDetId();
      unsigned int iStack = stubDetId.iLayer();
      unsigned int iRing = stubDetId.iRing();
      unsigned int iPhi = stubDetId.iPhi();
      unsigned int iZ = stubDetId.iZ();


      GlobalPoint stubPosition = theStackedGeometry->findGlobalPosition(stub);

      if (stub->getTriggerBend()<0.0) stubPt=-stubPt;
      if (iStack==999999 && stubPosition.z()>0) stubPt=-stubPt;
	
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

      std::vector< int > rows= innerClusters->getRows();
      std::vector< int > cols= innerClusters->getCols();

      for (unsigned int ihit=0;ihit<rows.size();ihit++){

	if (iStack<1000) {
	  innerStack.push_back(true);
	  irphi.push_back(rows[ihit]);
	  iz.push_back(cols[ihit]);
	  iladder.push_back(PXBDetId(innerDetId).ladder());
	  imodule.push_back(PXBDetId(innerDetId).module());
	}
	else {
	  innerStack.push_back(true);
	  irphi.push_back(rows[ihit]);
	  iz.push_back(cols[ihit]);
	  iladder.push_back(PXFDetId(innerDetId).disk());
	  imodule.push_back(PXFDetId(innerDetId).module());
	}    
      }

      edm::Ref< edmNew::DetSetVector< TTCluster<Ref_PixelDigi_> >, TTCluster<Ref_PixelDigi_> > outerClusters=clusters[1];

      const DetId outerDetId =outerClusters->getDetId();

      rows= outerClusters->getRows();
      cols= outerClusters->getCols();

      for (unsigned int ihit=0;ihit<rows.size();ihit++){

	if (iStack<1000) {
	  innerStack.push_back(false);
	  irphi.push_back(rows[ihit]);
	  iz.push_back(cols[ihit]);
	  iladder.push_back(PXBDetId(outerDetId).ladder());
	  imodule.push_back(PXBDetId(outerDetId).module());
	}
	else {
	  innerStack.push_back(false);
	  irphi.push_back(rows[ihit]);
	  iz.push_back(cols[ihit]);
	  iladder.push_back(PXFDetId(outerDetId).disk());
	  imodule.push_back(PXFDetId(outerDetId).module());
	}    
      }    


      int strip=-1;
      if (irphi.size()!=0) {
	strip=irphi[0];
      }
      //std::cout << "strip = "<<strip<<std::endl;

      if (ev.addStub(iStack-1,iPhi+1,iZ,strip,stubPt,stub->getTriggerBend(),
		 stubPosition.x(),stubPosition.y(),stubPosition.z(),
		     innerStack,irphi,iz,iladder,imodule)) {

	edm::Ref< edmNew::DetSetVector< TTStub< Ref_PixelDigi_ > >, TTStub< Ref_PixelDigi_ > > tempStubRef = edmNew::makeRefTo( TTStubHandle, iterTTStub);


	L1TStub lastStub=ev.lastStub();
	
	//cout << "Adding stub:"<<lastStub.layer()<<" "
	//     <<lastStub.ladder()<<" "
	//     <<lastStub.module()<<" "
	//     <<lastStub.iz()<<" "
	//     <<lastStub.iphi()<<endl;
	stubMap[lastStub]=tempStubRef;
      }
   
    }
  }


  //std::cout << "Will actually do L1 tracking:"<<std::endl;


  //////////////////////////
  // NOW RUN THE L1 tracking



  if (asciiEventOutName_!="") {
    ev.write(asciiEventOut_);
  }


  FPGATimer readTimer;
  FPGATimer cleanTimer;
  FPGATimer addStubTimer;
  FPGATimer layerdiskRouterTimer;
  FPGATimer VMRouterTimer;  
  FPGATimer TETimer;
  FPGATimer TCTimer;
  FPGATimer PTTimer;
  FPGATimer PRTimer;
  FPGATimer METimer;
  FPGATimer MCTimer;
  FPGATimer MTTimer;
  FPGATimer FTTimer;
  FPGATimer DuplicateTimer;

  bool first=true;

  std::vector<FPGATrack> tracks;

  int selectmu=0;
  L1SimTrack simtrk(0,0,0.0,0.0,0.0,0.0,0.0,0.0);

  ofstream outres;
  outres.open("trackres.txt");

  ofstream outeff;
  outeff.open("trackeff.txt");

#include "FPGA.icc"  


  int ntracks=0;

  
  for (unsigned itrack=0; itrack<tracks.size(); itrack++) {
    FPGATrack track=tracks[itrack];

    if (track.duplicate()) continue;

    ntracks++;

    TTTrack<Ref_PixelDigi_> aTrack;

    aTrack.setSector(999); //this is currently not retrained by the algorithm
    aTrack.setWedge(999); //not used by the tracklet implementations

    //First do the 4 parameter fit

    GlobalPoint bsPosition4par(0.0,0.0,track.z0());

    aTrack.setPOCA(bsPosition4par,4);
 
    double pt4par=fabs(track.pt(mMagneticFieldStrength));

    GlobalVector p34par(GlobalVector::Cylindrical(pt4par, 
						  track.phi0(), 
						  pt4par*sinh(track.eta())));

    aTrack.setMomentum(p34par,4);
    
    aTrack.setRInv(track.rinv(),4);

    aTrack.setChi2(track.chisq(),4);


    //std::cout<<"FPGA Track "<<track.duplicate()<<" "<<pt4par<<" "
    //	     <<track.eta()<<" "
    //	     <<track.phi0()<<std::endl;
    
    //Now do the 5 parameter fit

    GlobalPoint bsPosition5par(-track.d0()*sin(track.phi0()),track.d0()*cos(track.phi0()),track.z0());

    aTrack.setPOCA(bsPosition5par,5);
 
    double pt5par=fabs(track.pt(mMagneticFieldStrength));

    GlobalVector p35par(GlobalVector::Cylindrical(pt5par, 
						  track.phi0(), 
						  pt5par*sinh(track.eta())));

    aTrack.setMomentum(p35par,5);
    
    aTrack.setRInv(track.rinv(),5);

    aTrack.setChi2(track.chisq(),5);
    
    
    vector<L1TStub*> stubptrs = track.stubs();
    
    vector<L1TStub> stubs;

    //std::cout << "L1TStub size = " <<stubptrs.size()<<std::endl;

    for (unsigned int i=0;i<stubptrs.size();i++){ 
      stubs.push_back(*(stubptrs[i]));
    }

    
    stubMapType::const_iterator it;
    for (vector<L1TStub>::const_iterator itstubs = stubs.begin(); 
	 itstubs != stubs.end(); itstubs++) {
      it=stubMap.find(*itstubs);
      if (it!=stubMap.end()) {
	aTrack.addStubRef(it->second);
	//cout << "Found stub in stub map"<<endl;
	//cout << "stub:"<<itstubs->layer()<<" "
	//     <<itstubs->ladder()<<" "
	//     <<itstubs->module()<<" "
	//     <<itstubs->iz()<<" "
	//     <<itstubs->iphi()<<endl;
      }
      else{
	cout << "Could not find stub in stub map"<<endl;
	cout << "stub:"<<itstubs->layer()<<" "
	     <<itstubs->ladder()<<" "
	     <<itstubs->module()<<" "
	     <<itstubs->iz()<<" "
	     <<itstubs->iphi()<<endl;

      }
    }

    //std::cout << "tracks.size() = "<<tracks.size()<<" saved "<<ntracks<<std::endl;
    

    // pt consistency
    //float consistency4par = StubPtConsistency::getConsistency(aTrack, theStackedGeometry, mMagneticFieldStrength, 4); 
    //aTrack.setStubPtConsistency(consistency4par, 4);
    aTrack.setStubPtConsistency(-1.0, 4);
    //float consistency5par = StubPtConsistency::getConsistency(aTrack, theStackedGeometry, mMagneticFieldStrength, 5); 
    //aTrack.setStubPtConsistency(consistency5par,5);
    aTrack.setStubPtConsistency(-1.0,5);

    L1TkTracksForOutput->push_back(aTrack);

  }

  //cout << "size:"<<stubMap.size()<<endl;
  //for(stubMapType::const_iterator it=stubMap.begin();it!=stubMap.end();it++){
  //	cout << "iterating stub:"<<it->first.layer()<<" "
  //     <<it->first.ladder()<<" "
  //     <<it->first.module()<<" "
  //     <<it->first.iz()<<" "
  //	     <<it->first.iphi()<<endl;
  //}


  iEvent.put( L1TkTracksForOutput, "Level1TTTracks");

} /// End of produce()


// ///////////////////////////
// // DEFINE THIS AS A PLUG-IN
DEFINE_FWK_MODULE(L1FPGATrackProducer);

#endif
