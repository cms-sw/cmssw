#include "Validation/RecoVertex/interface/PrimaryVertexAnalyzer4PU.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Version/interface/GetReleaseVersion.h"
#include "MagneticField/Engine/interface/MagneticField.h"

// reco track and vertex 
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/Math/interface/Point3D.h"
#include "RecoVertex/VertexPrimitives/interface/TransientVertex.h"

// simulated vertices,..., add <use name=SimDataFormats/Vertex> and <../Track>
#include <SimDataFormats/Vertex/interface/SimVertex.h>
#include <SimDataFormats/Vertex/interface/SimVertexContainer.h>
#include <SimDataFormats/Track/interface/SimTrack.h>
#include <SimDataFormats/Track/interface/SimTrackContainer.h>

#include "SimDataFormats/CrossingFrame/interface/CrossingFrame.h"
#include "SimDataFormats/CrossingFrame/interface/CrossingFramePlaybackInfo.h"
#include "SimDataFormats/CrossingFrame/interface/MixCollection.h"
#include "SimDataFormats/TrackingHit/interface/PSimHit.h"

//generator level + CLHEP
#include "HepMC/GenEvent.h"
#include "HepMC/GenVertex.h"


// TrackingParticle
#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticleFwd.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingVertexContainer.h"
//associator
#include "SimTracker/Records/interface/TrackAssociatorRecord.h"


// fit
#include "RecoVertex/KalmanVertexFit/interface/KalmanVertexFitter.h"
#include "RecoVertex/AdaptiveVertexFit/interface/AdaptiveVertexFitter.h"

#include "CommonTools/Statistics/interface/ChiSquaredProbability.h"

// Root
#include <TH1.h>
#include <TH2.h>
#include <TFile.h>
#include <TProfile.h>
 
#include <cmath>
#include <gsl/gsl_math.h>
#include <gsl/gsl_eigen.h>


// cluster stufff
//#include "DataFormats/TrackRecoTrack.h"
#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHitCollection.h"
#include "Geometry/TrackerGeometryBuilder/interface/PixelGeomDetUnit.h"


using namespace edm;
using namespace reco;
using namespace std;
//
// constants, enums and typedefs
//
typedef reco::Vertex::trackRef_iterator trackit_t;
//
// static data member definitions
//

//
// constructors and destructor
//
PrimaryVertexAnalyzer4PU::PrimaryVertexAnalyzer4PU(const ParameterSet& iConfig) :
  theTrackFilter(iConfig.getParameter<edm::ParameterSet>("TkFilterParameters")),
  beamSpot_(iConfig.getParameter<edm::InputTag>("beamSpot"))
{
   //now do what ever initialization is needed
  simG4_=iConfig.getParameter<edm::InputTag>( "simG4" );
  recoTrackProducer_= iConfig.getUntrackedParameter<std::string>("recoTrackProducer");
  // open output file to store histograms}
  outputFile_  = iConfig.getUntrackedParameter<std::string>("outputFile");

  rootFile_ = TFile::Open(outputFile_.c_str(),"RECREATE");
  verbose_= iConfig.getUntrackedParameter<bool>("verbose", false);
  doMatching_= iConfig.getUntrackedParameter<bool>("matching", false);
  printXBS_= iConfig.getUntrackedParameter<bool>("XBS", false);
  simUnit_= 1.0;  // starting with CMSSW_1_2_x ??
  if ( (edm::getReleaseVersion()).find("CMSSW_1_1_",0)!=std::string::npos){
    simUnit_=0.1;  // for use in  CMSSW_1_1_1 tutorial
  }
  
  dumpPUcandidates_=iConfig.getUntrackedParameter<bool>("dumpPUcandidates", false);

  zmatch_=iConfig.getUntrackedParameter<double>("zmatch", 0.0500);
  cout << "PrimaryVertexAnalyzer4PU: zmatch=" << zmatch_ << endl;
  eventcounter_=0;
  dumpcounter_=0;
  ndump_=10;
  DEBUG_=false;
  //DEBUG_=true;
}




PrimaryVertexAnalyzer4PU::~PrimaryVertexAnalyzer4PU()
{
    // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)
  delete rootFile_;
}



//
// member functions
//


std::map<std::string, TH1*>  PrimaryVertexAnalyzer4PU::bookVertexHistograms(){
  std::map<std::string, TH1*> h;

  // release validation histograms used in DoCompare.C
  h["nbtksinvtx"]   = new TH1F("nbtksinvtx","reconstructed tracks in vertex",40,-0.5,39.5); 
  h["nbtksinvtxPU"] = new TH1F("nbtksinvtxPU","reconstructed tracks in vertex",40,-0.5,39.5); 
  h["nbtksinvtxTag"]= new TH1F("nbtksinvtxTag","reconstructed tracks in vertex",40,-0.5,39.5); 
  h["resx"]         = new TH1F("resx","residual x",100,-0.04,0.04);
  h["resy"]         = new TH1F("resy","residual y",100,-0.04,0.04);
  h["resz"]         = new TH1F("resz","residual z",100,-0.1,0.1);
  h["resz10"]       = new TH1F("resz10","residual z",100,-1.0,1.);
  h["pullx"]        = new TH1F("pullx","pull x",100,-25.,25.);
  h["pully"]        = new TH1F("pully","pull y",100,-25.,25.);
  h["pullz"]        = new TH1F("pullz","pull z",100,-25.,25.);
  h["vtxchi2"]      = new TH1F("vtxchi2","chi squared",100,0.,100.);
  h["vtxndf"]       = new TH1F("vtxndf","degrees of freedom",500,0.,100.);
  h["vtxndfc"]       = new TH1F("vtxndfc","expected 2nd highest ndof",500,0.,100.);

  h["vtxndfvsntk"]  = new TH2F("vtxndfvsntk","ndof vs #tracks",20,0.,100, 20, 0., 200.);
  h["vtxndfoverntk"]= new TH1F("vtxndfoverntk","ndof / #tracks",40,0.,2.);
  h["vtxndf2overntk"]= new TH1F("vtxndf2overntk","(ndof+2) / #tracks",40,0.,2.);
  h["tklinks"]      = new TH1F("tklinks","Usable track links",2,-0.5,1.5);
  h["nans"]         = new TH1F("nans","Illegal values for x,y,z,xx,xy,xz,yy,yz,zz",9,0.5,9.5);


  // raw
  add(h, new TH1F("szRecVtx","size of recvtx collection",20, -0.5, 19.5));
  add(h, new TH1F("isFake","fake vertex",2, -0.5, 1.5));
  add(h, new TH1F("isFake1","fake vertex or ndof<0",2, -0.5, 1.5));
  add(h, new TH1F("bunchCrossing","bunchCrossing",4000, 0., 4000.));
  add(h, new TH2F("bunchCrossingLogNtk","bunchCrossingLogNtk",4000, 0., 4000.,5,0., 5.));
  add(h, new TH1F("highpurityTrackFraction","fraction of high purity tracks",20, 0., 1.));
  add(h, new TH2F("trkchi2vsndof","vertices chi2 vs ndof",50, 0., 100., 50, 0., 200.));
  add(h, new TH1F("trkchi2overndof","vertices chi2 / ndof",50, 0., 5.));
  // two track vertices
  add(h,new TH2F("2trkchi2vsndof","two-track vertices chi2 vs ndof",40, 0., 10., 20, 0., 20.));
  add(h,new TH1F("2trkmassSS","two-track vertices mass (same sign)",100, 0., 2.));
  add(h,new TH1F("2trkmassOS","two-track vertices mass (opposite sign)",100, 0., 2.));
  add(h,new TH1F("2trkdphi","two-track vertices delta-phi",360, 0, 2*M_PI));
  add(h,new TH1F("2trkseta","two-track vertices sum-eta",50, -2., 2.));
  add(h,new TH1F("2trkdphicurl","two-track vertices delta-phi (sum eta<0.1)",360, 0, 2*M_PI));
  add(h,new TH1F("2trksetacurl","two-track vertices sum-eta (delta-phi<0.1)",50, -2., 2.));
  add(h,new TH1F("2trkdetaOS","two-track vertices delta-eta (same sign)",50, -0.5, 0.5));
  add(h,new TH1F("2trkdetaSS","two-track vertices delta-eta (opposite sign)",50, -0.5, 0.5));
  // two track PU vertices
  add(h,new TH1F("2trkmassSSPU","two-track vertices mass (same sign)",100, 0., 2.));
  add(h,new TH1F("2trkmassOSPU","two-track vertices mass (opposite sign)",100, 0., 2.));
  add(h,new TH1F("2trkdphiPU","two-track vertices delta-phi",360, 0, 2*M_PI));
  add(h,new TH1F("2trksetaPU","two-track vertices sum-eta",50, -2., 2.));
  add(h,new TH1F("2trkdphicurlPU","two-track vertices delta-phi (sum eta<0.1)",360, 0, 2*M_PI));
  add(h,new TH1F("2trksetacurlPU","two-track vertices sum-eta (delta-phi<0.1)",50, -2., 2.));
  add(h,new TH1F("2trkdetaOSPU","two-track vertices delta-eta (same sign)",50, -0.5, 0.5));
  add(h,new TH1F("2trkdetaSSPU","two-track vertices delta-eta (opposite sign)",50, -0.5, 0.5));
  // three track vertices
  add(h,new TH2F("2trkchi2vsndof","two-track vertices chi2 vs ndof",40, 0., 10., 20, 0., 20.));
  add(h,new TH2F("3trkchi2vsndof","three-track vertices chi2 vs ndof",40, 0., 10., 20, 0., 20.));
  add(h,new TH2F("4trkchi2vsndof","four-track vertices chi2 vs ndof",40, 0., 10., 20, 0., 20.));
  add(h,new TH2F("5trkchi2vsndof","five-track vertices chi2 vs ndof",40, 0., 10., 20, 0., 20.));
  // same for fakes
  add(h,new TH2F("fake2trkchi2vsndof","fake two-track vertices chi2 vs ndof",40, 0., 10., 20, 0., 20.));
  add(h,new TH2F("fake3trkchi2vsndof","fake three-track vertices chi2 vs ndof",40, 0., 10., 20, 0., 20.));
  add(h,new TH2F("fake4trkchi2vsndof","fake four-track vertices chi2 vs ndof",40, 0., 10., 20, 0., 20.));
  add(h,new TH2F("fake5trkchi2vsndof","fake five-track vertices chi2 vs ndof",40, 0., 10., 20, 0., 20.));
  h["resxr"]        = new TH1F("resxr","relative residual x",100,-0.04,0.04);
  h["resyr"]        = new TH1F("resyr","relative residual y",100,-0.04,0.04);
  h["reszr"]        = new TH1F("reszr","relative residual z",100,-0.1,0.1);
  h["pullxr"]       = new TH1F("pullxr","relative pull x",100,-25.,25.);
  h["pullyr"]       = new TH1F("pullyr","relative pull y",100,-25.,25.);
  h["pullzr"]       = new TH1F("pullzr","relative pull z",100,-25.,25.);
  h["vtxprob"]      = new TH1F("vtxprob","chisquared probability",100,0.,1.);
  h["eff"]          = new TH1F("eff","efficiency",2, -0.5, 1.5);
  h["efftag"]       = new TH1F("efftag","efficiency tagged vertex",2, -0.5, 1.5);
  h["zdistancetag"] = new TH1F("zdistancetag","z-distance between tagged and generated",100, -0.1, 0.1);
  h["abszdistancetag"] = new TH1F("abszdistancetag","z-distance between tagged and generated",1000, 0., 1.0);
  h["abszdistancetagcum"] = new TH1F("abszdistancetagcum","z-distance between tagged and generated",1000, 0., 1.0);
  h["puritytag"]    = new TH1F("puritytag","purity of primary vertex tags",2, -0.5, 1.5);
  h["effvsptsq"]    = new TProfile("effvsptsq","efficiency vs ptsq",20, 0., 10000., 0, 1.);
  h["effvsnsimtrk"] = new TProfile("effvsnsimtrk","efficiency vs # simtracks",50, 0., 50., 0, 1.);
  h["effvsnrectrk"] = new TProfile("effvsnrectrk","efficiency vs # rectracks",50, 0., 50., 0, 1.);
  h["effvsnseltrk"] = new TProfile("effvsnseltrk","efficiency vs # selected tracks",50, 0., 50., 0, 1.);
  h["effvsz"]       = new TProfile("effvsz","efficiency vs z",20, -20., 20., 0, 1.);
  h["effvsz2"]      = new TProfile("effvsz2","efficiency vs z (2mm)",20, -20., 20., 0, 1.);
  h["effvsr"]       = new TProfile("effvsr","efficiency vs r",20, 0., 1., 0, 1.);
  h["xresvsntrk"] = new TProfile("xresvsntrk","xresolution vs # vertex tracks",40, 0., 200., 0, 0.01);
  h["yresvsntrk"] = new TProfile("yresvsntrk","yresolution vs # vertex tracks",40, 0., 200., 0, 0.01);
  h["zresvsntrk"] = new TProfile("zresvsntrk","zresolution vs # vertex tracks",40, 0., 200., 0, 0.01);
  h["cpuvsntrk"] = new TProfile("cpuvsntrk","cpu time vs # of fitted tracks",40, 0., 200., 0, 200.);
  h["cpucluvsntrk"] = new TProfile("cpucluvsntrk","clustering cpu time # of tracks",40, 0., 200., 0, 10.);
  h["cpufit"]    = new TH1F("cpufit","cpu time for fitting",100, 0., 200.);
  h["cpuclu"]    = new TH1F("cpuclu","cpu time for clustering",100, 0., 200.);
  h["nbtksinvtx2"]   = new TH1F("nbtksinvtx2","reconstructed tracks in vertex",40,0.,200.); 
  h["nbtksinvtxPU2"]   = new TH1F("nbtksinvtxPU2","reconstructed tracks in vertex",40,0.,200.); 
  h["nbtksinvtxTag2"]   = new TH1F("nbtksinvtxTag2","reconstructed tracks in vertex",40,0.,200.); 

  h["xrec"]         = new TH1F("xrec","reconstructed x",100,-0.1,0.1);
  h["yrec"]         = new TH1F("yrec","reconstructed y",100,-0.1,0.1);
  h["zrec"]         = new TH1F("zrec","reconstructed z",100,-20.,20.);
  h["err1"]         = new TH1F("err1","error 1",100,0.,0.1);
  h["err2"]         = new TH1F("err2","error 2",100,0.,0.1);
  h["errx"]         = new TH1F("errx","error x",100,0.,0.1);
  h["erry"]         = new TH1F("erry","error y",100,0.,0.1);
  h["errz"]         = new TH1F("errz","error z",100,0.,2.0);
  h["errz1"]        = new TH1F("errz1","error z",100,0.,0.2);

  h["zrecNt100"]         = new TH1F("zrecNt100","reconstructed z for high multiplicity events",80,-40.,40.);
  add(h, new TH2F("zrecvsnt","reconstructed z vs number of tracks",100,-50., 50., 20, 0., 200.));
  add(h, new TH2F("xyrec","reconstructed xy",100, -4., 4., 100, -4., 4.));
  h["xrecBeam"]     = new TH1F("xrecBeam","reconstructed x - beam x",100,-0.1,0.1);
  h["yrecBeam"]     = new TH1F("yrecBeam","reconstructed y - beam y",100,-0.1,0.1);
  h["zrecBeam"]     = new TH1F("zrecBeam","reconstructed z - beam z",100,-20.,20.);
  h["xrecBeamvsz"] = new TH2F("xrecBeamvsz","reconstructed x - beam x vs z", 20, -20., 20.,100,-0.1,0.1);
  h["yrecBeamvsz"] = new TH2F("yrecBeamvsz","reconstructed y - beam y vs z", 20, -20., 20.,100,-0.1,0.1);
  h["xrecBeamvszprof"] = new TProfile("xrecBeamvszprof","reconstructed x - beam x vs z-z0", 20, -20., 20.,-0.1,0.1);
  h["yrecBeamvszprof"] = new TProfile("yrecBeamvszprof","reconstructed y - beam y vs z-z0", 20, -20., 20.,-0.1,0.1);
  add(h, new TH2F("xrecBeamvsdxXBS","reconstructed x - beam x vs resolution",10,0., 0.02, 100, -0.1,0.1)); // just a test
  add(h, new TH2F("yrecBeamvsdyXBS","reconstructed z - beam z vs resolution",10,0., 0.02, 100, -0.1,0.1));
  add(h, new TH2F("xrecBeamvsdx","reconstructed x - beam x vs resolution",10,0., 0.02, 100, -0.1,0.1));
  add(h, new TH2F("yrecBeamvsdy","reconstructed z - beam z vs resolution",10,0., 0.02, 100, -0.1,0.1));
  add(h, new TH2F("xrecBeamvsdxR2","reconstructed x - beam x vs resolution",20,0., 0.04, 100, -0.1,0.1));
  add(h, new TH2F("yrecBeamvsdyR2","reconstructed z - beam z vs resolution",20,0., 0.04, 100, -0.1,0.1));
  //  add(h, new TH2F("xrecBeamvsdx","reconstructed x - beam x vs resolution",100,-0.1,0.1, 10, 0., 0.04));
  //  add(h, new TH2F("yrecBeamvsdy","reconstructed y - beam y vs resolution",100,-0.1,0.1, 10, 0., 0.04));
  h["xrecBeamvsdxprof"] = new TProfile("xrecBeamvsdxprof","reconstructed x - beam x vs resolution",10, 0., 0.04,-0.1,0.1 );
  h["yrecBeamvsdyprof"] = new TProfile("yrecBeamvsdyprof","reconstructed y - beam y vs resolution",10, 0., 0.04,-0.1,0.1 );
  add(h, new TProfile("xrecBeam2vsdx2prof","reconstructed x - beam x vs resolution",10,0., 0.002, 0., 0.01));
  add(h, new TProfile("yrecBeam2vsdy2prof","reconstructed y - beam y vs resolution",10,0., 0.002, 0., 0.01));
  add(h,new TH2F("xrecBeamvsdx2","reconstructed x - beam x vs resolution",10,0., 0.002, 100, -0.01, 0.01));
  add(h,new TH2F("yrecBeamvsdy2","reconstructed y - beam y vs resolution",10,0., 0.002, 100, -0.01, 0.01));
  h["xrecb"]        = new TH1F("xrecb","reconstructed x - beam x",100,-0.01,0.01);
  h["yrecb"]        = new TH1F("yrecb","reconstructed y - beam y",100,-0.01,0.01);
  h["zrecb"]        = new TH1F("zrecb","reconstructed z - beam z",100,-20.,20.);
  h["xrec1"]        = new TH1F("xrec1","reconstructed x",100,-4,4);
  h["yrec1"]        = new TH1F("yrec1","reconstructed y",100,-4,4);  // should match the sim histos
  h["zrec1"]        = new TH1F("zrec1","reconstructed z",100,-80.,80.);
  h["xrec2"]        = new TH1F("xrec2","reconstructed x",100,-1,1);
  h["yrec2"]        = new TH1F("yrec2","reconstructed y",100,-1,1);
  h["zrec2"]        = new TH1F("zrec2","reconstructed z",100,-40.,40.);
  h["xrec3"]        = new TH1F("xrec3","reconstructed x",100,-0.1,0.1);
  h["yrec3"]        = new TH1F("yrec3","reconstructed y",100,-0.1,0.1);
  h["zrec3"]        = new TH1F("zrec3","reconstructed z",100,-20.,20.);
  add(h, new TH1F("xrecBeamPull","normalized residuals reconstructed x - beam x",100,-20,20));
  add(h, new TH1F("yrecBeamPull","normalized residuals reconstructed y - beam y",100,-20,20));
  add(h, new TH1F("zdiffrec","z-distance between vertices",200,-20., 20.));
  add(h, new TH1F("zdiffrec2","z-distance between ndof>2 vertices",200,-20., 20.));
  add(h, new TH1F("zdiffrec4","z-distance between ndof>4 vertices",200,-20., 20.));
  add(h, new TH1F("zdiffrec8","z-distance between ndof>8 vertices",200,-20., 20.));
  add(h, new TH2F("zvszrec2","z positions of multiple vertices",200,-20., 20., 200,-20., 20.));
  add(h, new TH2F("pzvspz2","prob(z) of multiple vertices",100, 0.,1.,100,0., 1.));
  add(h, new TH2F("zvszrec4","z positions of multiple vertices",100,-20., 20., 100,-20., 20.));
  add(h, new TH2F("pzvspz4","prob(z) of multiple vertices",100, 0.,1.,100,0., 1.));
  add(h, new TH2F("zdiffvsz","z-distance vs z",100,-10.,10.,30,-15.,15.));
  add(h, new TH2F("zdiffvsz4","z-distance vs z (ndof>4)",100,-10.,10.,30,-15.,15.));
  add(h, new TProfile("eff0vsntrec","efficiency vs # reconstructed tracks",50, 0., 50., 0, 1.));
  add(h, new TProfile("eff0vsntsel","efficiency vs # selected tracks",50, 0., 50., 0, 1.));
  add(h, new TProfile("eff0ndof0vsntsel","efficiency (ndof>0) vs # selected tracks",50, 0., 50., 0, 1.));
  add(h, new TProfile("eff0ndof8vsntsel","efficiency (ndof>8) vs # selected tracks",50, 0., 50., 0, 1.));
  add(h, new TProfile("eff0ndof2vsntsel","efficiency (ndof>2) vs # selected tracks",50, 0., 50., 0, 1.));
  add(h, new TProfile("eff0ndof4vsntsel","efficiency (ndof>4) vs # selected tracks",50, 0., 50., 0, 1.));
  add(h, new TH1F("xbeamPUcand","x - beam of PU candidates (ndof>4)",100,-1., 1.));
  add(h, new TH1F("ybeamPUcand","y - beam of PU candidates (ndof>4)",100,-1., 1.));
  add(h, new TH1F("xbeamPullPUcand","x - beam pull of PU candidates (ndof>4)",100,-20., 20.));
  add(h, new TH1F("ybeamPullPUcand","y - beam pull of PU candidates (ndof>4)",100,-20., 20.));
  add(h, new TH1F("ndofOverNtk","ndof / ntk of ndidates (ndof>4)",100,0., 2.));
  //add(h, new TH1F("sumwOverNtk","sumw / ntk of ndidates (ndof>4)",100,0., 2.));
  add(h, new TH1F("ndofOverNtkPUcand","ndof / ntk of ndidates (ndof>4)",100,0., 2.));
  //add(h, new TH1F("sumwOverNtkPUcand","sumw / ntk of ndidates (ndof>4)",100,0., 2.));
  add(h, new TH1F("zPUcand","z positions of PU candidates (all)",100,-20., 20.));
  add(h, new TH1F("zPUcand2","z positions of PU candidates (ndof>2)",100,-20., 20.));
  add(h, new TH1F("zPUcand4","z positions of PU candidates (ndof>4)",100,-20., 20.));
  add(h, new TH1F("ndofPUcand","ndof of PU candidates (all)",50,0., 100.));
  add(h, new TH1F("ndofPUcand2","ndof of PU candidates (ndof>2)",50,0., 100.));
  add(h, new TH1F("ndofPUcand4","ndof of PU candidates (ndof>4)",50,0., 100.));
  add(h, new TH1F("ndofnr2","second highest ndof",500,0., 100.));
  add(h, new TH1F("ndofnr2d1cm","second highest ndof (dz>1cm)",500,0., 100.));
  add(h, new TH1F("ndofnr2d2cm","second highest ndof (dz>2cm)",500,0., 100.));

  h["nrecvtx"]      = new TH1F("nrecvtx","# of reconstructed vertices", 50, -0.5, 49.5);
  h["nrecvtx8"]      = new TH1F("nrecvtx8","# of reconstructed vertices with ndof>8", 50, -0.5, 49.5);
  h["nrecvtx2"]      = new TH1F("nrecvtx2","# of reconstructed vertices with ndof>2", 50, -0.5, 49.5);
  h["nrecvtx4"]      = new TH1F("nrecvtx4","# of reconstructed vertices with ndof>4", 50, -0.5, 49.5);
  //  h["nsimvtx"]      = new TH1F("nsimvtx","# of simulated vertices", 50, -0.5, 49.5);
  h["nrectrk"]      = new TH1F("nrectrk","# of reconstructed tracks", 100, -0.5, 99.5);
  add(h, new TH1F("nsimtrk","# of simulated tracks", 100, -0.5, 99.5));
  add(h, new TH1F("nsimtrkSignal","# of simulated tracks (Signal)", 100, -0.5, 99.5));
  add(h, new TH1F("nsimtrkPU","# of simulated tracks (PU)", 100, -0.5, 99.5));
  h["nsimtrk"]->StatOverflows(kTRUE);
  h["nsimtrkPU"]->StatOverflows(kTRUE);
  h["nsimtrkSignal"]->StatOverflows(kTRUE);
  h["xrectag"]      = new TH1F("xrectag","reconstructed x, signal vtx",100,-0.05,0.05);
  h["yrectag"]      = new TH1F("yrectag","reconstructed y, signal vtx",100,-0.05,0.05);
  h["zrectag"]      = new TH1F("zrectag","reconstructed z, signal vtx",100,-20.,20.);
  h["nrectrk0vtx"] = new TH1F("nrectrk0vtx","# rec tracks no vertex ",100,-0.5, 99.5);
  h["nseltrk0vtx"] = new TH1F("nseltrk0vtx","# rec tracks no vertex ",100,-0.5, 99.5);
  h["nrecsimtrk"] = new TH1F("nrecsimtrk","# rec tracks matched to sim tracks in vertex",100,-0.5, 99.5);
  h["nrecnosimtrk"] = new TH1F("nrecsimtrk","# rec tracks not matched to sim tracks in vertex",100,-0.5, 99.5);
  h["trackAssEffvsPt"] =  new TProfile("trackAssEffvsPt","track association efficiency vs pt",20, 0., 100., 0, 1.);

  // cluster stuff
  h["nseltrk"]         = new TH1F("nseltrk","# of reconstructed tracks selected for PV", 100, -0.5, 99.5);
  h["nclutrkall"]      = new TH1F("nclutrkall","# of reconstructed tracks in clusters", 100, -0.5, 99.5);
  h["nclutrkvtx"]      = new TH1F("nclutrkvtx","# of reconstructed tracks in clusters of reconstructed vertices", 100, -0.5, 99.5);
  h["nclu"]            = new TH1F("nclu","# of clusters", 100, -0.5, 99.5);
  h["nclu0vtx"]        = new TH1F("nclu0vtx","# of clusters in events with no PV", 100, -0.5, 99.5);
  h["zlost1"]           = new TH1F("zlost1","z of lost vertices (bad z)", 100, -20., 20.);
  h["zlost2"]           = new TH1F("zlost2","z of lost vertices (no matching cluster)", 100, -20., 20.);
  h["zlost3"]           = new TH1F("zlost3","z of lost vertices (vertex too far from beam)", 100, -20., 20.);
  h["zlost4"]           = new TH1F("zlost4","z of lost vertices (invalid vertex)", 100, -20., 20.);
  h["selstat"]     = new TH1F("selstat","selstat", 5, -2.5, 2.5);
  

  // properties of fake vertices  (MC only)_
  add(h, new TH1F("fakeVtxZNdofgt05","z of fake vertices with ndof>0.5", 100, -20., 20.));
  add(h, new TH1F("fakeVtxZNdofgt2","z of fake vertices with ndof>2", 100, -20., 20.));
  add(h, new TH1F("fakeVtxZ","z of fake vertices", 100, -20., 20.));
  add(h, new TH1F("fakeVtxNdof","ndof of fake vertices", 500,0., 100.));
  add(h,new TH1F("fakeVtxNtrk","number of tracks in fake vertex",20,-0.5, 19.5));
  add(h,new TH1F("matchedVtxNdof","ndof of matched vertices", 500,0., 100.));


  //  histograms of track quality (Data and MC)
  string types[] = {"all","sel","bachelor","sellost","wgt05","wlt05","M","tagged","untagged","ndof4","PUcand","PUfake"};
  for(int t=0; t<12; t++){
    add(h, new TH1F(("rapidity_"+types[t]).c_str(),"rapidity ",100,-3., 3.));
    h["z0_"+types[t]] = new TH1F(("z0_"+types[t]).c_str(),"z0 ",80,-40., 40.);
    h["phi_"+types[t]] = new TH1F(("phi_"+types[t]).c_str(),"phi ",80,-3.14159, 3.14159);
    h["eta_"+types[t]] = new TH1F(("eta_"+types[t]).c_str(),"eta ",80,-4., 4.);
    h["pt_"+types[t]] = new TH1F(("pt_"+types[t]).c_str(),"pt ",100,0., 20.);
    h["found_"+types[t]]     = new TH1F(("found_"+types[t]).c_str(),"found hits",20, 0., 20.);
    h["lost_"+types[t]]      = new TH1F(("lost_"+types[t]).c_str(),"lost hits",20, 0., 20.);
    h["nchi2_"+types[t]]     = new TH1F(("nchi2_"+types[t]).c_str(),"normalized track chi2",100, 0., 20.);
    h["rstart_"+types[t]]    = new TH1F(("rstart_"+types[t]).c_str(),"start radius",100, 0., 20.);
    h["tfom_"+types[t]]      = new TH1F(("tfom_"+types[t]).c_str(),"track figure of merit",100, 0., 100.);
    h["expectedInner_"+types[t]]      = new TH1F(("expectedInner_"+types[t]).c_str(),"expected inner hits ",10, 0., 10.);
    h["expectedOuter_"+types[t]]      = new TH1F(("expectedOuter_"+types[t]).c_str(),"expected outer hits ",10, 0., 10.);
    h["logtresxy_"+types[t]] = new TH1F(("logtresxy_"+types[t]).c_str(),"log10(track r-phi resolution/um)",100, 0., 5.);
    h["logtresz_"+types[t]] = new TH1F(("logtresz_"+types[t]).c_str(),"log10(track z resolution/um)",100, 0., 5.);
    h["tpullxy_"+types[t]]   = new TH1F(("tpullxy_"+types[t]).c_str(),"track r-phi pull",100, -10., 10.);
    add(h, new TH2F( ("lvseta_"+types[t]).c_str(),"cluster length vs eta",60,-3., 3., 20, 0., 20));
    add(h, new TH2F( ("lvstanlambda_"+types[t]).c_str(),"cluster length vs tan lambda",60,-6., 6., 20, 0., 20));
    add(h, new TH1F( ("restrkz_"+types[t]).c_str(),"z-residuals (track vs vertex)", 200, -5., 5.));
    add(h, new TH2F( ("restrkzvsphi_"+types[t]).c_str(),"z-residuals (track - vertex)", 12,-3.14159,3.14159,100, -5., 5.));
    add(h, new TH2F( ("restrkzvseta_"+types[t]).c_str(),"z-residuals (track - vertex)", 12,-3.,3.,200, -5., 5.));
    add(h, new TH2F( ("pulltrkzvsphi_"+types[t]).c_str(),"normalized z-residuals (track - vertex)", 12,-3.14159,3.14159,100, -5., 5.));
    add(h, new TH2F( ("pulltrkzvseta_"+types[t]).c_str(),"normalized z-residuals (track - vertex)", 12,-3.,3.,100, -5., 5.));
    add(h, new TH1F( ("pulltrkz_"+types[t]).c_str(),"normalized z-residuals (track vs vertex)", 100, -5., 5.));
    add(h, new TH1F( ("sigmatrkz0_"+types[t]).c_str(),"z-resolution (excluding beam)", 100, 0., 5.));
    add(h, new TH1F( ("sigmatrkz_"+types[t]).c_str(),"z-resolution (including beam)", 100,0., 5.));
    add(h, new TH1F( ("nbarrelhits_"+types[t]).c_str(),"number of pixel barrel hits", 10, 0., 10.));
    add(h, new TH1F( ("nbarrelLayers_"+types[t]).c_str(),"number of pixel barrel layers", 10, 0., 10.));
    add(h, new TH1F( ("nPxLayers_"+types[t]).c_str(),"number of pixel layers (barrel+endcap)", 10, 0., 10.));
    add(h, new TH1F( ("nSiLayers_"+types[t]).c_str(),"number of Tracker layers ", 20, 0., 20.));
    add(h, new TH1F( ("trackAlgo_"+types[t]).c_str(),"track algorithm ", 30, 0., 30.));
    add(h, new TH1F( ("trackQuality_"+types[t]).c_str(),"track quality ", 7, -1., 6.));
  }
  add(h, new TH1F("trackWt","track weight in vertex",100,0., 1.));
   

  h["nrectrk"]->StatOverflows(kTRUE);
  h["nrectrk"]->StatOverflows(kTRUE);
  h["nrectrk0vtx"]->StatOverflows(kTRUE);
  h["nseltrk0vtx"]->StatOverflows(kTRUE);
  h["nrecsimtrk"]->StatOverflows(kTRUE);
  h["nrecnosimtrk"]->StatOverflows(kTRUE);
  h["nseltrk"]->StatOverflows(kTRUE);
  h["nbtksinvtx"]->StatOverflows(kTRUE);
  h["nbtksinvtxPU"]->StatOverflows(kTRUE);
  h["nbtksinvtxTag"]->StatOverflows(kTRUE);
  h["nbtksinvtx2"]->StatOverflows(kTRUE);
  h["nbtksinvtxPU2"]->StatOverflows(kTRUE);
  h["nbtksinvtxTag2"]->StatOverflows(kTRUE);

  // pile-up and track assignment related histograms (MC)
  h["npu0"]       =new TH1F("npu0","Number of simulated vertices",40,0.,40.);
  h["npu1"]       =new TH1F("npu1","Number of simulated vertices with >0 track",40,0.,40.);
  h["npu2"]       =new TH1F("npu2","Number of simulated vertices with >1 track",40,0.,40.);
  h["nrecv"]      =new TH1F("nrecv","# of reconstructed vertices", 40, 0, 40);
  add(h,new TH2F("nrecvsnpu","#rec vertices vs number of sim vertices with >0 tracks", 40,  0., 40, 40,  0, 40));
  add(h,new TH2F("nrecvsnpu2","#rec vertices vs number of sim vertices with >1 tracks", 40,  0., 40, 40,  0, 40));
  add(h,new TH1F("sumpt","sumpt of simulated tracks",100,0.,100.));
  add(h,new TH1F("sumptSignal","sumpt of simulated tracks in Signal events",100,0.,200.));
  add(h,new TH1F("sumptPU","sumpt of simulated tracks in PU events",100,0.,200.));
  add(h,new TH1F("sumpt2rec","sumpt2 of reconstructed and matched tracks",100,0.,100.));
  add(h,new TH1F("sumpt2","sumpt2 of simulated tracks",100,0.,100.));
  add(h,new TH1F("sumpt2Signal","sumpt2 of simulated tracks in Signal events",100,0.,200.));
  add(h,new TH1F("sumpt2PU","sumpt2 of simulated tracks in PU events",100,0.,200.));
  add(h,new TH1F("sumpt2rec","sumpt2 of reconstructed and matched tracks",100,0.,100.));
  add(h,new TH1F("sumpt2recSignal","sumpt2 of reconstructed and matched tracks in Signal events",100,0.,200.));
  add(h,new TH1F("sumpt2recPU","sumpt2 of reconstructed and matched tracks in PU events",100,0.,200.));
  add(h,new TH1F("nRecTrkInSimVtx","number of reco tracks matched to sim-vertex", 101, 0., 100));
  add(h,new TH1F("nRecTrkInSimVtxSignal","number of reco tracks matched to signal sim-vertex", 101, 0., 100));
  add(h,new TH1F("nRecTrkInSimVtxPU","number of reco tracks matched to PU-vertex", 101, 0., 100));
  add(h,new TH1F("nPrimRecTrkInSimVtx","number of reco primary tracks matched to sim-vertex", 101, 0., 100));
  add(h,new TH1F("nPrimRecTrkInSimVtxSignal","number of reco primary tracks matched to signal sim-vertex", 101, 0., 100));
  add(h,new TH1F("nPrimRecTrkInSimVtxPU","number of reco primary tracks matched to PU-vertex", 101, 0., 100));
  // obsolete, scheduled for removal
  add(h,new TH1F("recPurity","track purity of reconstructed vertices", 101, 0., 1.01));
  add(h,new TH1F("recPuritySignal","track purity of reconstructed Signal vertices", 101, 0., 1.01));
  add(h,new TH1F("recPurityPU","track purity of reconstructed PU vertices", 101, 0., 1.01));
  // end of obsolete
  add(h,new TH1F("recmatchPurity","track purity of all vertices", 101, 0., 1.01));
  add(h,new TH1F("recmatchPurityTag","track purity of tagged vertices", 101, 0., 1.01));
  add(h,new TH1F("recmatchPurityTagSignal","track purity of tagged Signal vertices", 101, 0., 1.01));
  add(h,new TH1F("recmatchPurityTagPU","track purity of tagged PU vertices", 101, 0., 1.01));
  add(h,new TH1F("recmatchPuritynoTag","track purity of untagged vertices", 101, 0., 1.01));
  add(h,new TH1F("recmatchPuritynoTagSignal","track purity of untagged Signal vertices", 101, 0., 1.01));
  add(h,new TH1F("recmatchPuritynoTagPU","track purity of untagged PU vertices", 101, 0., 1.01));
  add(h,new TH1F("recmatchvtxs","number of sim vertices contributing to recvtx", 10, 0., 10.));
  add(h,new TH1F("recmatchvtxsTag","number of sim vertices contributing to recvtx", 10, 0., 10.));
  add(h,new TH1F("recmatchvtxsnoTag","number of sim vertices contributing to recvtx", 10, 0., 10.));
  //
  add(h,new TH1F("trkAssignmentEfficiency", "track to vertex assignment efficiency", 101, 0., 1.01) );
  add(h,new TH1F("trkAssignmentEfficiencySignal", "track to signal vertex assignment efficiency", 101, 0., 1.01) );
  add(h,new TH1F("trkAssignmentEfficiencyPU", "track to PU vertex assignment efficiency", 101, 0., 1.01) );
  add(h,new TH1F("primtrkAssignmentEfficiency", "track to vertex assignment efficiency", 101, 0., 1.01) );
  add(h,new TH1F("primtrkAssignmentEfficiencySignal", "track to signal vertex assignment efficiency", 101, 0., 1.01) );
  add(h,new TH1F("primtrkAssignmentEfficiencyPU", "track to PU vertex assignment efficiency", 101, 0., 1.01) );
  add(h,new TH1F("vtxMultiplicity", "number of rec vertices containg tracks from one true vertex", 10, 0., 10.) );
  add(h,new TH1F("vtxMultiplicitySignal", "number of rec vertices containg tracks from the Signal Vertex", 10, 0., 10.) );
  add(h,new TH1F("vtxMultiplicityPU", "number of rec vertices containg tracks from a PU Vertex", 10, 0., 10.) );
  
  add(h,new TProfile("vtxFindingEfficiencyVsNtrk","finding efficiency vs number of associated rec tracks",100, 0., 100., 0., 1.) );
  add(h,new TProfile("vtxFindingEfficiencyVsNtrkSignal","Signal vertex finding efficiency vs number of associated rec tracks",100, 0., 100., 0., 1.) );
  add(h,new TProfile("vtxFindingEfficiencyVsNtrkPU","PU vertex finding efficiency vs number of associated rec tracks",100, 0., 100., 0., 1.) );

  add(h,new TH1F("TagVtxTrkPurity","TagVtxTrkPurity",100,0.,1.01));
  add(h,new TH1F("TagVtxTrkEfficiency","TagVtxTrkEfficiency",100,0.,1.01));
  
  add(h,new TH1F("matchVtxFraction","fraction of sim vertex tracks found in a recvertex",101,0,1.01));
  add(h,new TH1F("matchVtxFractionSignal","fraction of sim vertex tracks found in a recvertex",101,0,1.01));
  add(h,new TH1F("matchVtxFractionPU","fraction of sim vertex tracks found in a recvertex",101,0,1.01));
  add(h,new TH1F("matchVtxFractionCum","fraction of sim vertex tracks found in a recvertex",101,0,1.01));
  add(h,new TH1F("matchVtxFractionCumSignal","fraction of sim vertexs track found in a recvertex",101,0,1.01));
  add(h,new TH1F("matchVtxFractionCumPU","fraction of sim vertex tracks found in a recvertex",101,0,1.01));
  add(h,new TH1F("matchVtxEfficiency","efficiency for finding matching rec vertex",2,-0.5,1.5));
  add(h,new TH1F("matchVtxEfficiencySignal","efficiency for finding matching rec vertex",2,-0.5,1.5));
  add(h,new TH1F("matchVtxEfficiencyPU","efficiency for finding matching rec vertex",2,-0.5,1.5));
  add(h,new TH1F("matchVtxEfficiency2","efficiency for finding matching rec vertex (nt>1)",2,-0.5,1.5));
  add(h,new TH1F("matchVtxEfficiency2Signal","efficiency for finding matching rec vertex (nt>1)",2,-0.5,1.5));
  add(h,new TH1F("matchVtxEfficiency2PU","efficiency for finding matching rec vertex (nt>1)",2,-0.5,1.5));
  add(h,new TH1F("matchVtxEfficiency5","efficiency for finding matching rec vertex (purity>0.5)",2,-0.5,1.5));
  add(h,new TH1F("matchVtxEfficiency5Signal","efficiency for finding matching rec vertex (purity>0.5)",2,-0.5,1.5));
  add(h,new TH1F("matchVtxEfficiency5PU","efficiency for finding matching rec vertex (purity>0.5)",2,-0.5,1.5));


  add(h,new TH1F("matchVtxEfficiencyZ","efficiency for finding matching rec vertex within 1 mm",2,-0.5,1.5));
  add(h,new TH1F("matchVtxEfficiencyZSignal","efficiency for finding matching rec vertex within 1 mm",2,-0.5,1.5));
  add(h,new TH1F("matchVtxEfficiencyZPU","efficiency for finding matching rec vertex within 1 mm",2,-0.5,1.5));

  add(h,new TH1F("matchVtxEfficiencyZ1","efficiency for finding matching rec vertex within 1 mm (nt>0)",2,-0.5,1.5));
  add(h,new TH1F("matchVtxEfficiencyZ1Signal","efficiency for finding matching rec vertex within 1 mm (nt>0)",2,-0.5,1.5));
  add(h,new TH1F("matchVtxEfficiencyZ1PU","efficiency for finding matching rec vertex within 1 mm (nt>0)",2,-0.5,1.5));

  add(h,new TH1F("matchVtxEfficiencyZ2","efficiency for finding matching rec vertex within 1 mm (nt>1)",2,-0.5,1.5));
  add(h,new TH1F("matchVtxEfficiencyZ2Signal","efficiency for finding matching rec vertex within 1 mm (nt>1)",2,-0.5,1.5));
  add(h,new TH1F("matchVtxEfficiencyZ2PU","efficiency for finding matching rec vertex within 1 mm (nt>1)",2,-0.5,1.5));

  add(h,new TH1F("matchVtxZ","z distance to matched recvtx",100, -0.1, 0.1));
  add(h,new TH1F("matchVtxZPU","z distance to matched recvtx",100, -0.1, 0.1));
  add(h,new TH1F("matchVtxZSignal","z distance to matched recvtx",100, -0.1, 0.1));

  add(h,new TH1F("matchVtxZCum","z distance to matched recvtx",1001,0, 1.01));
  add(h,new TH1F("matchVtxZCumSignal","z distance to matched recvtx",1001,0, 1.01));
  add(h,new TH1F("matchVtxZCumPU","z distance to matched recvtx",1001,0, 1.01));

  add(h, new TH1F("unmatchedVtx","number of unmatched rec vertices (fakes)",10,0.,10.));
  add(h, new TH1F("unmatchedVtxFrac","fraction of unmatched rec vertices (fakes)",1000,0.,1.0));
  add(h, new TH1F("unmatchedVtxZ","z of unmached rec  vertices (fakes)",100,-20., 20.));
  add(h, new TH1F("unmatchedVtxNdof","ndof of unmatched rec vertices (fakes)", 500,0., 100.));
  add(h,new TH2F("correctlyassigned","pt and eta of correctly assigned tracks", 60,  -3., 3., 100, 0, 10.));
  add(h,new TH2F("misassigned","pt and eta of mis assigned tracks", 60,  -3., 3., 100, 0, 10.));

  add(h,new TH1F("ptcat","pt of correctly assigned tracks", 100, 0, 10.));
  add(h,new TH1F("etacat","eta of correctly assigned tracks", 60, -3., 3.));
  add(h,new TH1F("phicat","phi of correctly assigned tracks", 100, -3.14159, 3.14159));
  add(h,new TH1F("dzcat","dz of correctly assigned tracks", 100, 0., 1.));

  add(h,new TH1F("ptmis","pt of mis-assigned tracks", 100, 0, 10.));
  add(h,new TH1F("etamis","eta of mis-assigned tracks", 60, -3., 3.));
  add(h,new TH1F("phimis","phi of mis-assigned tracks",100, -3.14159, 3.14159));
  add(h,new TH1F("dzmis","dz of mis-assigned tracks", 100, 0., 1.));


  add(h,new TH1F("Tc","Tc computed with Truth matched Reco Tracks",100,0.,20.));
  add(h,new TH1F("TcSignal","Tc of signal vertices computed with Truth matched Reco Tracks",100,0.,20.));
  add(h,new TH1F("TcPU","Tc of PU vertices computed with Truth matched Reco Tracks",100,0.,20.));

  add(h,new TH1F("logTc","log Tc computed with Truth matched Reco Tracks",100,-2.,8.));
  add(h,new TH1F("logTcSignal","log Tc of signal vertices computed with Truth matched Reco Tracks",100,-2.,8.));
  add(h,new TH1F("logTcPU","log Tc of PU vertices computed with Truth matched Reco Tracks",100,-2.,8.));

  add(h,new TH1F("xTc","Tc of merged clusters",100,0.,20.));
  add(h,new TH1F("xTcSignal","Tc of signal vertices merged with PU",100,0.,20.));
  add(h,new TH1F("xTcPU","Tc of merged PU vertices",100,0.,20.));

  add(h,new TH1F("logxTc","log Tc merged vertices",100,-2.,8.));
  add(h,new TH1F("logxTcSignal","log Tc of signal vertices merged with PU",100,-2.,8.));
  add(h,new TH1F("logxTcPU","log Tc of merged PU vertices ",100,-2.,8.));

  add(h,new TH1F("logChisq","Chisq/ntrk computed with Truth matched Reco Tracks",100,-2.,8.));
  add(h,new TH1F("logChisqSignal","Chisq/ntrk of signal vertices computed with Truth matched Reco Tracks",100,-2.,8.));
  add(h,new TH1F("logChisqPU","Chisq/ntrk of PU vertices computed with Truth matched Reco Tracks",100,-2.,8.));

  add(h,new TH1F("logxChisq","Chisq/ntrk of merged clusters",100,-2.,8.));
  add(h,new TH1F("logxChisqSignal","Chisq/ntrk of signal vertices merged with PU",100,-2.,8.));
  add(h,new TH1F("logxChisqPU","Chisq/ntrk of merged PU vertices",100,-2.,8.));

  add(h,new TH1F("Chisq","Chisq/ntrk computed with Truth matched Reco Tracks",100,0.,20.));
  add(h,new TH1F("ChisqSignal","Chisq/ntrk of signal vertices computed with Truth matched Reco Tracks",100,0.,20.));
  add(h,new TH1F("ChisqPU","Chisq/ntrk of PU vertices computed with Truth matched Reco Tracks",100,0.,20.));

  add(h,new TH1F("xChisq","Chisq/ntrk of merged clusters",100,0.,20.));
  add(h,new TH1F("xChisqSignal","Chisq/ntrk of signal vertices merged with PU",100,0.,20.));
  add(h,new TH1F("xChisqPU","Chisq/ntrk of merged PU vertices",100,0.,20.));

  add(h,new TH1F("dzmax","dzmax computed with Truth matched Reco Tracks",100,0.,2.));
  add(h,new TH1F("dzmaxSignal","dzmax of signal vertices computed with Truth matched Reco Tracks",100,0.,2.));
  add(h,new TH1F("dzmaxPU","dzmax of PU vertices computed with Truth matched Reco Tracks",100,0.,2.));

  add(h,new TH1F("xdzmax","dzmax of merged clusters",100,0.,2.));
  add(h,new TH1F("xdzmaxSignal","dzmax of signal vertices merged with PU",100,0.,2.));
  add(h,new TH1F("xdzmaxPU","dzmax of merged PU vertices",100,0.,2.));

  add(h,new TH1F("dztrim","dzmax computed with Truth matched Reco Tracks",100,0.,2.));
  add(h,new TH1F("dztrimSignal","dzmax of signal vertices computed with Truth matched Reco Tracks",100,0.,2.));
  add(h,new TH1F("dztrimPU","dzmax of PU vertices computed with Truth matched Reco Tracks",100,0.,2.));

  add(h,new TH1F("xdztrim","dzmax of merged clusters",100,0.,2.));
  add(h,new TH1F("xdztrimSignal","dzmax of signal vertices merged with PU",100,0.,2.));
  add(h,new TH1F("xdztrimPU","dzmax of merged PU vertices",100,0.,2.));

  add(h,new TH1F("m4m2","m4m2 computed with Truth matched Reco Tracks",100,0.,100.));
  add(h,new TH1F("m4m2Signal","m4m2 of signal vertices computed with Truth matched Reco Tracks",100,0.,100.));
  add(h,new TH1F("m4m2PU","m4m2 of PU vertices computed with Truth matched Reco Tracks",100,0.,100.));

  add(h,new TH1F("xm4m2","m4m2 of merged clusters",100,0.,100.));
  add(h,new TH1F("xm4m2Signal","m4m2 of signal vertices merged with PU",100,0.,100.));
  add(h,new TH1F("xm4m2PU","m4m2 of merged PU vertices",100,0.,100.));

  return h;
}


//
// member functions
//
void PrimaryVertexAnalyzer4PU::beginJob(){
  std::cout << " PrimaryVertexAnalyzer4PU::beginJob  conversion from sim units to rec units is " << simUnit_ << std::endl;


  rootFile_->cd();

  TDirectory *noBS = rootFile_->mkdir("noBS");
  noBS->cd();
  hnoBS=bookVertexHistograms();
  for(std::map<std::string,TH1*>::const_iterator hist=hnoBS.begin(); hist!=hnoBS.end(); hist++){
    hist->second->SetDirectory(noBS);
  }

  TDirectory *withBS = rootFile_->mkdir("BS");
  withBS->cd();
  hBS=bookVertexHistograms();
  for(std::map<std::string,TH1*>::const_iterator hist=hBS.begin(); hist!=hBS.end(); hist++){
    hist->second->SetDirectory(withBS);
  }

  TDirectory *DA = rootFile_->mkdir("DA");
  DA->cd();
  hDA=bookVertexHistograms();
  for(std::map<std::string,TH1*>::const_iterator hist=hDA.begin(); hist!=hDA.end(); hist++){
    hist->second->SetDirectory(DA);
  }

//   TDirectory *PIX = rootFile_->mkdir("PIX");
//   PIX->cd();
//   hPIX=bookVertexHistograms();
//   for(std::map<std::string,TH1*>::const_iterator hist=hPIX.begin(); hist!=hPIX.end(); hist++){
//     hist->second->SetDirectory(PIX);
//   }

//   TDirectory *MVF = rootFile_->mkdir("MVF");
//   MVF->cd();
//   hMVF=bookVertexHistograms();
//   for(std::map<std::string,TH1*>::const_iterator hist=hMVF.begin(); hist!=hMVF.end(); hist++){
//     hist->second->SetDirectory(MVF);
//   }

  rootFile_->cd();
  hsimPV["rapidity"] = new TH1F("rapidity","rapidity ",100,-10., 10.);
  hsimPV["chRapidity"] = new TH1F("chRapidity","charged rapidity ",100,-10., 10.);
  hsimPV["recRapidity"] = new TH1F("recRapidity","reconstructed rapidity ",100,-10., 10.);
  hsimPV["pt"] = new TH1F("pt","pt ",100,0., 20.);

  hsimPV["xsim"]         = new TH1F("xsim","simulated x",100,-0.01,0.01); // 0.01cm = 100 um
  hsimPV["ysim"]         = new TH1F("ysim","simulated y",100,-0.01,0.01);
  hsimPV["zsim"]         = new TH1F("zsim","simulated z",100,-20.,20.);

  hsimPV["xsim1"]        = new TH1F("xsim1","simulated x",100,-4.,4.);
  hsimPV["ysim1"]        = new TH1F("ysim1","simulated y",100,-4.,4.);
  hsimPV["zsim1"]        = new TH1F("zsim1","simulated z",100,-40.,40.);

  add(hsimPV, new TH1F("xsim2PU","simulated x (Pile-up)",100,-1.,1.));
  add(hsimPV, new TH1F("ysim2PU","simulated y (Pile-up)",100,-1.,1.)); 
  add(hsimPV, new TH1F("zsim2PU","simulated z (Pile-up)",100,-20.,20.)); 
  add(hsimPV, new TH1F("xsim2Signal","simulated x (Signal)",100,-1.,1.));
  add(hsimPV, new TH1F("ysim2Signal","simulated y (Signal)",100,-1.,1.));
  add(hsimPV, new TH1F("zsim2Signal","simulated z (Signal)",100,-20.,20.));

  hsimPV["xsim2"]        = new TH1F("xsim2","simulated x",100,-1,1); // 0.01cm = 100 um
  hsimPV["ysim2"]        = new TH1F("ysim2","simulated y",100,-1,1);
  hsimPV["zsim2"]        = new TH1F("zsim2","simulated z",100,-20.,20.);
  hsimPV["xsim3"]        = new TH1F("xsim3","simulated x",100,-0.1,0.1); // 0.01cm = 100 um
  hsimPV["ysim3"]        = new TH1F("ysim3","simulated y",100,-0.1,0.1);
  hsimPV["zsim3"]        = new TH1F("zsim3","simulated z",100,-20.,20.);
  hsimPV["xsimb"]        = new TH1F("xsimb","simulated x",100,-0.01,0.01); // 0.01cm = 100 um
  hsimPV["ysimb"]        = new TH1F("ysimb","simulated y",100,-0.01,0.01);
  hsimPV["zsimb"]        = new TH1F("zsimb","simulated z",100,-20.,20.);
  hsimPV["xsimb1"]        = new TH1F("xsimb1","simulated x",100,-0.1,0.1); // 0.01cm = 100 um
  hsimPV["ysimb1"]        = new TH1F("ysimb1","simulated y",100,-0.1,0.1);
  hsimPV["zsimb1"]        = new TH1F("zsimb1","simulated z",100,-20.,20.);
  add(hsimPV, new TH1F("xbeam","beamspot x",100,-1.,1.));
  add(hsimPV, new TH1F("ybeam","beamspot y",100,-1.,1.));
  add(hsimPV, new TH1F("zbeam","beamspot z",100,-1.,1.));
  add(hsimPV, new TH1F("wxbeam","beamspot sigma x",100,-1.,1.));
  add(hsimPV, new TH1F("wybeam","beamspot sigma y",100,-1.,1.));
  add(hsimPV, new TH1F("wzbeam","beamspot sigma z",100,-1.,1.));
  hsimPV["xsim2"]->StatOverflows(kTRUE);
  hsimPV["ysim2"]->StatOverflows(kTRUE);
  hsimPV["zsim2"]->StatOverflows(kTRUE);
  hsimPV["xsimb"]->StatOverflows(kTRUE);
  hsimPV["ysimb"]->StatOverflows(kTRUE);
  hsimPV["zsimb"]->StatOverflows(kTRUE);
  hsimPV["nsimvtx"]      = new TH1F("nsimvtx","# of simulated vertices", 50, -0.5, 49.5);
  //  hsimPV["nsimtrk"]      = new TH1F("nsimtrk","# of simulated tracks", 100, -0.5, 99.5); //  not filled right now, also exists in hBS..
  //  hsimPV["nsimtrk"]->StatOverflows(kTRUE);
  hsimPV["nbsimtksinvtx"]= new TH1F("nbsimtksinvtx","simulated tracks in vertex",100,-0.5,99.5); 
  hsimPV["nbsimtksinvtx"]->StatOverflows(kTRUE);

}


void PrimaryVertexAnalyzer4PU::endJob() {
  std::cout << "this is void PrimaryVertexAnalyzer4PU::endJob() " << std::endl;
  //cumulate some histos
  double sumDA=0,sumBS=0,sumnoBS=0;
  // double sumPIX=0,sumMVF=0;
  for(int i=101; i>0; i--){
    sumDA+=hDA["matchVtxFractionSignal"]->GetBinContent(i)/hDA["matchVtxFractionSignal"]->Integral();
    hDA["matchVtxFractionCumSignal"]->SetBinContent(i,sumDA);
    sumBS+=hBS["matchVtxFractionSignal"]->GetBinContent(i)/hBS["matchVtxFractionSignal"]->Integral();
    hBS["matchVtxFractionCumSignal"]->SetBinContent(i,sumBS);
    sumnoBS+=hnoBS["matchVtxFractionSignal"]->GetBinContent(i)/hnoBS["matchVtxFractionSignal"]->Integral();
    hnoBS["matchVtxFractionCumSignal"]->SetBinContent(i,sumnoBS);
//     sumPIX+=hPIX["matchVtxFractionSignal"]->GetBinContent(i)/hPIX["matchVtxFractionSignal"]->Integral();
//     hPIX["matchVtxFractionCumSignal"]->SetBinContent(i,sumPIX);
//     sumMVF+=hMVF["matchVtxFractionSignal"]->GetBinContent(i)/hMVF["matchVtxFractionSignal"]->Integral();
//     hMVF["matchVtxFractionCumSignal"]->SetBinContent(i,sumMVF);
  }
  sumDA=0,sumBS=0,sumnoBS=0;
  //sumPIX=0,sumMVF=0;
  for(int i=1; i<1001; i++){
    sumDA+=hDA["abszdistancetag"]->GetBinContent(i);
    hDA["abszdistancetagcum"]->SetBinContent(i,sumDA/float(hDA["abszdistancetag"]->GetEntries()));
    sumBS+=hBS["abszdistancetag"]->GetBinContent(i);
    hBS["abszdistancetagcum"]->SetBinContent(i,sumBS/float(hBS["abszdistancetag"]->GetEntries()));
    sumnoBS+=hnoBS["abszdistancetag"]->GetBinContent(i);
    hnoBS["abszdistancetagcum"]->SetBinContent(i,sumnoBS/float(hnoBS["abszdistancetag"]->GetEntries()));
//     sumPIX+=hPIX["abszdistancetag"]->GetBinContent(i);
//     hPIX["abszdistancetagcum"]->SetBinContent(i,sumPIX/float(hPIX["abszdistancetag"]->GetEntries()));
//     sumMVF+=hMVF["abszdistancetag"]->GetBinContent(i);
//     hMVF["abszdistancetagcum"]->SetBinContent(i,sumMVF/float(hMVF["abszdistancetag"]->GetEntries()));
  }

  Cumulate(hBS["matchVtxZCum"]);   Cumulate(hBS["matchVtxZCumSignal"]);   Cumulate(hBS["matchVtxZCumPU"]); 
  Cumulate(hnoBS["matchVtxZCum"]);   Cumulate(hnoBS["matchVtxZCumSignal"]);   Cumulate(hnoBS["matchVtxZCumPU"]); 
  Cumulate(hDA["matchVtxZCum"]);   Cumulate(hDA["matchVtxZCumSignal"]);   Cumulate(hDA["matchVtxZCumPU"]); 
  /*
   h->ComputeIntegral();
   Double_t *integral = h->GetIntegral();
   h->SetContent(integral);
  */

  // make a reference for ndofnr2
  //hDA["vtxndof"]->ComputeIntegral();
  //Double_t *integral = hDA["vtxndf"]->GetIntegral();
  //   h->SetContent(integral);
  double p;
  for(int i=1; i<501; i++){
    if(hDA["vtxndf"]->GetEntries()>0){
      p=  hDA["vtxndf"]->Integral(i,501)/hDA["vtxndf"]->GetEntries();    hDA["vtxndfc"]->SetBinContent(i,p*hDA["vtxndf"]->GetBinContent(i));
    }
    if(hBS["vtxndf"]->GetEntries()>0){
      p=  hBS["vtxndf"]->Integral(i,501)/hBS["vtxndf"]->GetEntries();    hBS["vtxndfc"]->SetBinContent(i,p*hBS["vtxndf"]->GetBinContent(i));
    }
    if(hnoBS["vtxndf"]->GetEntries()>0){
      p=hnoBS["vtxndf"]->Integral(i,501)/hnoBS["vtxndf"]->GetEntries();  hnoBS["vtxndfc"]->SetBinContent(i,p*hnoBS["vtxndf"]->GetBinContent(i));
    }
//     if(hPIX["vtxndf"]->GetEntries()>0){
//       p=hPIX["vtxndf"]->Integral(i,501)/hPIX["vtxndf"]->GetEntries();  hPIX["vtxndfc"]->SetBinContent(i,p*hPIX["vtxndf"]->GetBinContent(i));
//     }
  }
  
  rootFile_->cd();
  for(std::map<std::string,TH1*>::const_iterator hist=hsimPV.begin(); hist!=hsimPV.end(); hist++){
    std::cout << "writing " << hist->first << std::endl;
    hist->second->Write();
  }
  rootFile_->Write();
  std::cout << "PrimaryVertexAnalyzer4PU::endJob: done" << std::endl;
}




// helper functions
std::vector<PrimaryVertexAnalyzer4PU::SimPart> PrimaryVertexAnalyzer4PU::getSimTrkParameters(
											     edm::Handle<edm::SimTrackContainer> & simTrks,
											     edm::Handle<edm::SimVertexContainer> & simVtcs,
											     double simUnit)
{
   std::vector<SimPart > tsim;
   if(simVtcs->begin()==simVtcs->end()){
     if(verbose_){
       cout << "  PrimaryVertexAnalyzer4PU::getSimTrkParameters  no simvtcs" << endl;
     }
     return tsim;
   }
   if(verbose_){
     cout << "  PrimaryVertexAnalyzer4PU::getSimTrkParameters simVtcs n=" << simVtcs->size() << endl;
     cout << "  PrimaryVertexAnalyzer4PU::getSimTrkParameters 1st position" << setw(8) << setprecision(4) << simVtcs->begin()->position() << endl;
   }
   double t0=simVtcs->begin()->position().e();

   for(edm::SimTrackContainer::const_iterator t=simTrks->begin();
       t!=simTrks->end(); ++t){
     if (t->noVertex()){
       std::cout << "simtrk  has no vertex" << std::endl;
     }else{
       // get the vertex position
       //HepLorentzVector v=(*simVtcs)[t->vertIndex()].position();
       math::XYZTLorentzVectorD v((*simVtcs)[t->vertIndex()].position().x(),
                          (*simVtcs)[t->vertIndex()].position().y(),
                          (*simVtcs)[t->vertIndex()].position().z(),
                          (*simVtcs)[t->vertIndex()].position().e());
       int pdgCode=t->type();

       if( pdgCode==-99 ){
         // such entries cause crashes, no idea what they are
         std::cout << "funny particle skipped  , code="  << pdgCode << std::endl;
       }else{
         double Q=0; //double Q=HepPDT::theTable().getParticleData(pdgCode)->charge();
         if ((pdgCode==11)||(pdgCode==13)||(pdgCode==15)||(pdgCode==-211)||(pdgCode==-2212)||(pdgCode==-321)||(pdgCode==-3222)){Q=-1;}
         else if((pdgCode==-11)||(pdgCode==-13)||(pdgCode==-15)||(pdgCode==211)||(pdgCode==2212)||(pdgCode==321)||(pdgCode==3222)){Q=1;}
         else {
           //std::cout << pdgCode << " " <<std::endl;
         }
         math::XYZTLorentzVectorD p(t->momentum().x(),t->momentum().y(),t->momentum().z(),t->momentum().e());
         if ( (Q != 0) && (p.pt()>0.1)  && (fabs(t->momentum().eta())<3.0)
              && fabs(v.z()*simUnit<20) && (sqrt(v.x()*v.x()+v.y()*v.y())<10.)){
           double x0=v.x()*simUnit;
           double y0=v.y()*simUnit;
           double z0=v.z()*simUnit;
           double kappa=-Q*0.002998*fBfield_/p.pt();
           double D0=x0*sin(p.phi())-y0*cos(p.phi())-0.5*kappa*(x0*x0+y0*y0);
           double q=sqrt(1.-2.*kappa*D0);
           double s0=(x0*cos(p.phi())+y0*sin(p.phi()))/q;
           double s1;
           if (fabs(kappa*s0)>0.001){
             s1=asin(kappa*s0)/kappa;
           }else{
             double ks02=(kappa*s0)*(kappa*s0);
             s1=s0*(1.+ks02/6.+3./40.*ks02*ks02+5./112.*pow(ks02,3));
           }
           SimPart sp;//ParameterVector par;
           sp.par[reco::TrackBase::i_qoverp] = Q/p.P();
           sp.par[reco::TrackBase::i_lambda] = M_PI/2.-p.theta();
           sp.par[reco::TrackBase::i_phi] = p.phi()-asin(kappa*s0);
           sp.par[reco::TrackBase::i_dxy] = -2.*D0/(1.+q);
           sp.par[reco::TrackBase::i_dsz] = z0*sin(p.theta())-s1*cos(p.theta());

	   sp.pdg=pdgCode;
           if (v.t()-t0<1e-15){
             sp.type=0;  // primary
           }else{
             sp.type=1;  //secondary
           }

           // now get zpca  (get perigee wrt beam)
           double x1=x0-0.033; double y1=y0-0.; // FIXME how do we get the simulated beam position?
           D0=x1*sin(p.phi())-y1*cos(p.phi())-0.5*kappa*(x1*x1+y1*y1);
           q=sqrt(1.-2.*kappa*D0);
           s0=(x1*cos(p.phi())+y1*sin(p.phi()))/q;
           if (fabs(kappa*s0)>0.001){
             s1=asin(kappa*s0)/kappa;
           }else{
             double ks02=(kappa*s0)*(kappa*s0);
             s1=s0*(1.+ks02/6.+3./40.*ks02*ks02+5./112.*pow(ks02,3));
           }
           sp.ddcap=-2.*D0/(1.+q);
           sp.zdcap=z0-s1/tan(p.theta());
           sp.zvtx=z0;
           sp.xvtx=x0;
           sp.yvtx=y0;

           tsim.push_back(sp);
         }
       }
     }// has vertex
   }//for loop
   return tsim;
}


int*  PrimaryVertexAnalyzer4PU::supf(std::vector<SimPart>& simtrks, const reco::TrackCollection & trks){
  int nsim=simtrks.size();
  int nrec=trks.size();
  int *rectosim=new int[nrec]; // pointer to associated simtrk
  double** pij=new double*[nrec];
  double mu=100.; // initial chi^2 cut-off  (5 dofs !)
  int nmatch=0;
  int i=0;
  for(reco::TrackCollection::const_iterator t=trks.begin(); t!=trks.end(); ++t){
    pij[i]=new double[nsim];
    rectosim[i]=-1;
    ParameterVector  par = t->parameters();
    //reco::TrackBase::CovarianceMatrix V = t->covariance();
    reco::TrackBase::CovarianceMatrix S = t->covariance();
    S.Invert();
    for(int j=0; j<nsim; j++){
      simtrks[j].rec=-1;
      SimPart s=simtrks[j];
      double c=0;
      for(int k=0; k<5; k++){
        for(int l=0; l<5; l++){
          c+=(par(k)-s.par[k])*(par(l)-s.par[l])*S(k,l);
        }
      }
      pij[i][j]=exp(-0.5*c);

//       double c0=pow((par[0]-s.par[0])/t->qoverpError(),2)*0.1
// 	+pow((par[1]-s.par[1])/t->lambdaError(),2)
// 	+pow((par[2]-s.par[2])/t->phiError(),2)
// 	+pow((par[3]-s.par[3])/t->dxyError(),2)*0.1;
//         +pow((par[4]-s.par[4])/t->dszError(),2)*0.1;
//       pij[i][j]=exp(-0.5*c0);

//       if( c0 <100 ){
//       cout << setw(3) << i << " rec " << setw(6) << par << endl;
//       cout << setw(3) << j << " sim " << setw(6) << s.par << " ---> C=" << c << endl;
//       cout <<  "       "  << setw(6)
// 	   << (par[0]-s.par[0])<< ","
// 	   << (par[1]-s.par[1])<< ","
// 	   << (par[2]-s.par[2])<< ","
// 	   << (par[3]-s.par[3])<< ","
// 	   << (par[4]-s.par[4])
// 	   << " match=" << match(simtrks[j].par, trks.at(i).parameters())
// 	   << endl;
//       cout <<  "       "  << setw(6)
// 	   << (par[0]-s.par[0])/t->qoverpError() << ","
// 	   << (par[1]-s.par[1])/t->lambdaError() << ","
// 	   << (par[2]-s.par[2])/t->phiError() << ","
// 	   << (par[3]-s.par[3])/t->dxyError() << ","
// 	   << (par[4]-s.par[4])/t->dszError() << " c0=" << c0
// 	   << endl <<endl;
//       }

    }
    i++;
  }

  for(int k=0; k<nrec; k++){
    int imatch=-1; int jmatch=-1;
    double pmatch=0;
    for(int j=0; j<nsim; j++){
      if ((simtrks[j].rec)<0){
        double psum=exp(-0.5*mu); //cutoff
        for(int i=0; i<nrec; i++){
          if (rectosim[i]<0){ psum+=pij[i][j];}
        }
        for(int i=0; i<nrec; i++){
          if ((rectosim[i]<0)&&(pij[i][j]/psum>pmatch)){
            pmatch=pij[i][j]/psum;
            imatch=i; jmatch=j;
          }
        }
      }
    }// sim loop
   if((jmatch>=0)||(imatch>=0)){
     //std::cout << pmatch << "    " << pij[imatch][jmatch] << "  match=" <<
     //	match(simtrks[jmatch].par, trks.at(imatch).parameters()) <<std::endl;
      if (pmatch>0.01){
        rectosim[imatch]=jmatch;
        simtrks[jmatch].rec=imatch;
        nmatch++;
      }else if (match(simtrks[jmatch].par, trks.at(imatch).parameters())){
        // accept it anyway if it matches crudely and relax the cut-off
        rectosim[imatch]=jmatch;
        simtrks[jmatch].rec=imatch;
        nmatch++;
        mu=mu*2;
      }
    }
  }

//   std::cout << ">>>>>>>>>>>>>>>--------------supf----------------------" << std::endl;
//   std::cout <<"nsim=" << nsim   << "   nrec=" << nrec << "    nmatch=" << nmatch << std::endl;
//   std::cout << "rec to sim " << std::endl;
//   for(int i=0; i<nrec; i++){
//     std::cout << i << " ---> " << rectosim[i] << std::endl;
//   }
//   std::cout << "sim to rec " << std::endl;
//   for(int j=0; j<nsim; j++){
//     std::cout << j << " ---> " << simtrks[j].rec << std::endl;
//   }

   std::cout << "unmatched sim " << std::endl;
   for(int j=0; j<nsim; j++){
     if(simtrks[j].rec<0){
       double pt= 1./simtrks[j].par[0]/tan(simtrks[j].par[1]);
       if((fabs(pt))>1.){
	 std::cout << setw(3) << j << setw(8) << simtrks[j].pdg 
		   << setw(8) << setprecision(4) << "  (" << simtrks[j].xvtx << "," << simtrks[j].yvtx <<  "," << simtrks[j].zvtx << ")" 
		   << " pt= " <<  pt
		   << " phi=" << simtrks[j].par[2] 
		   << " eta= " <<  -log(tan(0.5*(M_PI/2-simtrks[j].par[1]))) 
		   << std::endl; 
       }
     }
   }
//   std::cout << "<<<<<<<<<<<<<<<--------------supf----------------------" << std::endl;

  //delete rectosim; // or return it?
  for(int i=0; i<nrec; i++){delete pij[i];}
  delete pij;
  return rectosim;  // caller must delete it
}








bool PrimaryVertexAnalyzer4PU::match(const ParameterVector  &a,
				     const ParameterVector  &b){
  double dqoverp =a(0)-b(0);
  double dlambda =a(1)-b(1);
  double dphi    =a(2)-b(2);
  double dsz     =a(4)-b(4);
  if (dphi>M_PI){ dphi-=M_2_PI; }else if(dphi<-M_PI){dphi+=M_2_PI;}
  //  return ( (fabs(dqoverp)<0.2) && (fabs(dlambda)<0.02) && (fabs(dphi)<0.04) && (fabs(dsz)<0.1) );
  return ( (fabs(dqoverp)<0.2) && (fabs(dlambda)<0.02) && (fabs(dphi)<0.04) && (fabs(dsz)<1.0) );
}


bool PrimaryVertexAnalyzer4PU::matchVertex(const simPrimaryVertex  &vsim, 
				       const reco::Vertex       &vrec){
  return (fabs(vsim.z*simUnit_-vrec.z())<zmatch_);
}

bool PrimaryVertexAnalyzer4PU::isResonance(const HepMC::GenParticle * p){
  double ctau=(pdt_->particle( abs(p->pdg_id()) ))->lifetime();
  //std::cout << "isResonance   " << p->pdg_id() << " " << ctau << std::endl;
  return  ctau >0 && ctau <1e-6;
}

bool PrimaryVertexAnalyzer4PU::isFinalstateParticle(const HepMC::GenParticle * p){
  return ( !p->end_vertex() && p->status()==1 );
}


bool PrimaryVertexAnalyzer4PU::isCharged(const HepMC::GenParticle * p){
  const ParticleData * part = pdt_->particle( p->pdg_id() );
  if (part){
    return part->charge()!=0;
  }else{
    // the new/improved particle table doesn't know anti-particles
    return  pdt_->particle( -p->pdg_id() )!=0;
  }
}




void PrimaryVertexAnalyzer4PU::fillTrackHistos(std::map<std::string, TH1*> & h, const std::string & ttype, const reco::Track & t, const reco::Vertex * v){
    Fill(h,"rapidity_"+ttype,t.eta());
    Fill(h,"z0_"+ttype,t.vz());
    Fill(h,"phi_"+ttype,t.phi());
    Fill(h,"eta_"+ttype,t.eta());
    Fill(h,"pt_"+ttype,t.pt());
    Fill(h,"found_"+ttype,t.found());
    Fill(h,"lost_"+ttype,t.lost());
    Fill(h,"nchi2_"+ttype,t.normalizedChi2());
    Fill(h,"rstart_"+ttype,(t.innerPosition()).Rho());

    double d0Error=t.d0Error();
    double d0=t.dxy(vertexBeamSpot_.position());
    if (d0Error>0){ 
      Fill(h,"logtresxy_"+ttype,log(d0Error/0.0001)/log(10.));
      Fill(h,"tpullxy_"+ttype,d0/d0Error);
    }
    //double z0=t.vz();
    double dzError=t.dzError();
    if(dzError>0){
      Fill(h,"logtresz_"+ttype,log(dzError/0.0001)/log(10.));
    }

    //
    Fill(h,"sigmatrkz_"+ttype,sqrt(pow(t.dzError(),2)+wxy2_/pow(tan(t.theta()),2)));
    Fill(h,"sigmatrkz0_"+ttype,t.dzError());

    // track vs vertex 
    if((! (v==NULL)) && (v->ndof()>10.)) {
      // emulate clusterizer input
      //const TransientTrack & tt = theB_->build(&t);    wrong !!!!
      TransientTrack tt = theB_->build(&t);    tt.setBeamSpot(vertexBeamSpot_); // need the setBeamSpot !
      double z=(tt.stateAtBeamLine().trackStateAtPCA()).position().z();
      double tantheta=tan((tt.stateAtBeamLine().trackStateAtPCA()).momentum().theta());
      double dz2= pow(tt.track().dzError(),2)+wxy2_/pow(tantheta,2);
      
      Fill(h,"restrkz_"+ttype,z-v->position().z());
      Fill(h,"restrkzvsphi_"+ttype,t.phi(), z-v->position().z());
      Fill(h,"restrkzvseta_"+ttype,t.eta(), z-v->position().z());
      Fill(h,"pulltrkzvsphi_"+ttype,t.phi(), (z-v->position().z())/sqrt(dz2));
      Fill(h,"pulltrkzvseta_"+ttype,t.eta(), (z-v->position().z())/sqrt(dz2));

      Fill(h,"pulltrkz_"+ttype,(z-v->position().z())/sqrt(dz2));

      double x1=t.vx()-vertexBeamSpot_.x0(); double y1=t.vy()-vertexBeamSpot_.y0();
      double kappa=-0.002998*fBfield_*t.qoverp()/cos(t.theta());
      double D0=x1*sin(t.phi())-y1*cos(t.phi())-0.5*kappa*(x1*x1+y1*y1);
      double q=sqrt(1.-2.*kappa*D0);
      double s0=(x1*cos(t.phi())+y1*sin(t.phi()))/q; 
      // double s1;
      if (fabs(kappa*s0)>0.001){
	//s1=asin(kappa*s0)/kappa;
      }else{
	//double ks02=(kappa*s0)*(kappa*s0);
	//s1=s0*(1.+ks02/6.+3./40.*ks02*ks02+5./112.*pow(ks02,3));
      }
      //     sp.ddcap=-2.*D0/(1.+q);
      //double zdcap=t.vz()-s1/tan(t.theta());

    }
    //
    
    // collect some info on hits and clusters
    Fill(h,"nbarrelLayers_"+ttype,t.hitPattern().pixelBarrelLayersWithMeasurement());
    Fill(h,"nPxLayers_"+ttype,t.hitPattern().pixelLayersWithMeasurement());
    Fill(h,"nSiLayers_"+ttype,t.hitPattern().trackerLayersWithMeasurement());
    Fill(h,"expectedInner_"+ttype,t.trackerExpectedHitsInner().numberOfHits());
    Fill(h,"expectedOuter_"+ttype,t.trackerExpectedHitsOuter().numberOfHits());
    Fill(h,"trackAlgo_"+ttype,t.algo());
    Fill(h,"trackQuality_"+ttype,t.qualityMask());

    //
    int longesthit=0, nbarrel=0;
    for(trackingRecHit_iterator hit=t.recHitsBegin(); hit!=t.recHitsEnd(); hit++){
      if ((**hit).isValid()   && (**hit).geographicalId().det() == DetId::Tracker ){
       bool barrel = DetId((**hit).geographicalId()).subdetId() == static_cast<int>(PixelSubdetector::PixelBarrel);
       //bool endcap = DetId::DetId((**hit).geographicalId()).subdetId() == static_cast<int>(PixelSubdetector::PixelEndcap);
       if (barrel){
	 const SiPixelRecHit *pixhit = dynamic_cast<const SiPixelRecHit*>( &(**hit));
	 edm::Ref<edmNew::DetSetVector<SiPixelCluster>, SiPixelCluster> const& clust = (*pixhit).cluster();
	 if (clust.isNonnull()) {
	   nbarrel++;
	   if (clust->sizeY()-longesthit>0) longesthit=clust->sizeY();
	   if (clust->sizeY()>20.){
	     Fill(h,"lvseta_"+ttype,t.eta(), 19.9);
	     Fill(h,"lvstanlambda_"+ttype,tan(t.lambda()), 19.9);
	   }else{
	     Fill(h,"lvseta_"+ttype,t.eta(), float(clust->sizeY()));
	     Fill(h,"lvstanlambda_"+ttype,tan(t.lambda()), float(clust->sizeY()));
	   }
	 }
       }
      }
    }
    Fill(h,"nbarrelhits_"+ttype,float(nbarrel));
    //-------------------------------------------------------------------

}

void PrimaryVertexAnalyzer4PU::dumpHitInfo(const reco::Track & t){
    // collect some info on hits and clusters
  int longesthit=0, nbarrel=0;
  cout << Form("%5.2f  %5.2f  : ",t.pt(),t.eta());
  for(trackingRecHit_iterator hit=t.recHitsBegin(); hit!=t.recHitsEnd(); hit++){
    if ((**hit).isValid()   && (**hit).geographicalId().det() == DetId::Tracker ){
      bool barrel = DetId((**hit).geographicalId()).subdetId() == static_cast<int>(PixelSubdetector::PixelBarrel);
      //bool endcap = DetId::DetId((**hit).geographicalId()).subdetId() == static_cast<int>(PixelSubdetector::PixelEndcap);
      if (barrel){
	nbarrel++;
	const SiPixelRecHit *pixhit = dynamic_cast<const SiPixelRecHit*>( &(**hit));
	edm::Ref<edmNew::DetSetVector<SiPixelCluster>, SiPixelCluster> const& clust = (*pixhit).cluster();
	if (clust.isNonnull()) {
	  cout << Form("%4d",clust->sizeY());
	  if (clust->sizeY()-longesthit>0) longesthit=clust->sizeY();
	}
      }
    }
  }
  //cout << endl;
}

void PrimaryVertexAnalyzer4PU::printRecVtxs(const Handle<reco::VertexCollection> recVtxs, std::string title){
    int ivtx=0;
    std::cout << std::endl << title << std::endl;
    for(reco::VertexCollection::const_iterator v=recVtxs->begin(); 
	v!=recVtxs->end(); ++v){
      string vtype=" recvtx  ";
      if( v->isFake()){
	vtype=" fake   ";
      }else if (v->ndof()==-5){
	vtype=" cluster "; // pos=selector[iclu],cputime[iclu],clusterz[iclu]
      }else if(v->ndof()==-3){
	vtype=" event   ";
      }
      std::cout << "vtx "<< std::setw(3) << std::setfill(' ')<<ivtx++
	        << vtype
		<< " #trk " << std::fixed << std::setprecision(4) << std::setw(3) << v->tracksSize() 
		<< " chi2 " << std::fixed << std::setw(4) << std::setprecision(1) << v->chi2() 
		<< " ndof " << std::fixed << std::setw(5) << std::setprecision(2) << v->ndof()
		<< " x "  << std::setw(8) <<std::fixed << std::setprecision(4) << v->x() 
		<< " dx " << std::setw(8) << v->xError()// <<  std::endl 
		<< " y "  << std::setw(8) << v->y() 
		<< " dy " << std::setw(8) << v->yError()//<< std::endl
		<< " z "  << std::setw(8) << v->z() 
		<< " dz " << std::setw(8) << v->zError()
		<< std::endl;
    }
}


void PrimaryVertexAnalyzer4PU::printSimVtxs(const Handle<SimVertexContainer> simVtxs){
    int i=0;
    for(SimVertexContainer::const_iterator vsim=simVtxs->begin();
	vsim!=simVtxs->end(); ++vsim){
      if ( vsim->position().x()*vsim->position().x()+vsim->position().y()*vsim->position().y() < 1.){
	std::cout << i++ << ")" << std::scientific
		  << " evtid=" << vsim->eventId().event()  << ","  << vsim->eventId().bunchCrossing()
		  << " sim x=" << vsim->position().x()*simUnit_
		  << " sim y=" << vsim->position().y()*simUnit_
		  << " sim z=" << vsim->position().z()*simUnit_
		  << " sim t=" << vsim->position().t()
		  << " parent=" << vsim->parentIndex() 
		  << std::endl;
      }
    }
}







void PrimaryVertexAnalyzer4PU::printSimTrks(const Handle<SimTrackContainer> simTrks){
  std::cout <<  " simTrks   type, (momentum), vertIndex, genpartIndex"  << std::endl;
  int i=1;
  for(SimTrackContainer::const_iterator t=simTrks->begin();
      t!=simTrks->end(); ++t){
    //HepMC::GenParticle* gp=evtMC->GetEvent()->particle( (*t).genpartIndex() );
    std::cout << i++ << ")" 
	      << t->eventId().event()  << ","  << t->eventId().bunchCrossing()
	      << (*t)
	      << " index="
	      << (*t).genpartIndex();
    //if (gp) {
    //  HepMC::GenVertex *gv=gp->production_vertex();
    //  std::cout  <<  " genvertex =" << (*gv);
    //}
    std::cout << std::endl;
  }
}



void PrimaryVertexAnalyzer4PU::printRecTrks(const Handle<reco::TrackCollection> &recTrks  ){

  cout << "printRecTrks" << endl;
  int i =1;
  for(reco::TrackCollection::const_iterator t=recTrks->begin(); t!=recTrks->end(); ++t){
    //    reco::TrackBase::ParameterVector  par = t->parameters();
    
    cout << endl;
    cout << "Track "<<i << " " ; i++;
    //enum TrackQuality { undefQuality=-1, loose=0, tight=1, highPurity=2, confirmed=3, goodIterative=4, qualitySize=5};
    if( t->quality(reco::TrackBase::loose)){ cout << "loose ";};
    if( t->quality(reco::TrackBase::tight)){ cout << "tight ";};
    if( t->quality(reco::TrackBase::highPurity)){ cout << "highPurity ";};
    if( t->quality(reco::TrackBase::confirmed)){ cout << "confirmed  ";};
    if( t->quality(reco::TrackBase::goodIterative)){ cout << "goodIterative  ";};
    cout  << endl;

    TransientTrack  tk = theB_->build(&(*t)); tk.setBeamSpot(vertexBeamSpot_);   
    double ipsig=0;
    if (tk.stateAtBeamLine().isValid()){
      ipsig= tk.stateAtBeamLine().transverseImpactParameter().significance();
    }else{
      ipsig=-1;
    }

    cout << Form("pt=%8.3f phi=%6.3f eta=%6.3f z=%8.4f  dz=%6.4f, ipsig=%6.1f",t->pt(), t->phi(), t->eta(), t->vz(), t->dzError(),ipsig) << endl;


    cout << Form(" found=%6d  lost=%6d   chi2/ndof=%10.3f ",t->found(), t->lost(),t->normalizedChi2())<<endl;
    const reco::HitPattern & p= t->hitPattern();
    cout << "subdet layers valid lost" << endl;
    cout << Form(" barrel  %2d  %2d  %2d",p.pixelBarrelLayersWithMeasurement(),p.numberOfValidPixelBarrelHits(), p.numberOfLostPixelBarrelHits()) << endl;
    cout << Form(" fwd     %2d  %2d  %2d",p.pixelEndcapLayersWithMeasurement(),p.numberOfValidPixelEndcapHits(), p.numberOfLostPixelEndcapHits()) << endl;
    cout << Form(" pixel   %2d  %2d  %2d",p.pixelLayersWithMeasurement(), p.numberOfValidPixelHits(), p.numberOfLostPixelHits()) << endl;
    cout << Form(" tracker %2d  %2d  %2d",p.trackerLayersWithMeasurement(), p.numberOfValidTrackerHits(), p.numberOfLostTrackerHits()) << endl;
    cout << endl;
    const reco::HitPattern & pinner= t->trackerExpectedHitsInner();
    const reco::HitPattern & pouter= t->trackerExpectedHitsOuter();
    int ninner=pinner.numberOfHits();
    int nouter=pouter.numberOfHits();

    //    double d0Error=t.d0Error();
    //    double d0=t.dxy(myBeamSpot);
    
    //
    for(trackingRecHit_iterator hit=t->recHitsBegin(); hit!=t->recHitsEnd(); hit++){
      if ((**hit).isValid()   && (**hit).geographicalId().det() == DetId::Tracker ){
       bool barrel = DetId((**hit).geographicalId()).subdetId() == static_cast<int>(PixelSubdetector::PixelBarrel);
       bool endcap = DetId((**hit).geographicalId()).subdetId() == static_cast<int>(PixelSubdetector::PixelEndcap);
       if (barrel){
	 const SiPixelRecHit *pixhit = dynamic_cast<const SiPixelRecHit*>( &(**hit));
	 edm::Ref<edmNew::DetSetVector<SiPixelCluster>, SiPixelCluster> const& clust = (*pixhit).cluster();
	 if (clust.isNonnull()) {
	   cout << Form(" barrel cluster size=%2d   charge=%6.1f wx=%2d  wy=%2d, expected=%3.1f",clust->size(),clust->charge(),clust->sizeX(),clust->sizeY(),1.+2./fabs(tan(t->theta()))) << endl;
	 }
       }else if(endcap){
	 const SiPixelRecHit *pixhit = dynamic_cast<const SiPixelRecHit*>( &(**hit));
	 edm::Ref<edmNew::DetSetVector<SiPixelCluster>, SiPixelCluster> const& clust = (*pixhit).cluster();
	 if (clust.isNonnull()) {
	   cout << Form(" endcap cluster size=%2d   charge=%6.1f wx=%2d  wy=%2d",clust->size(),clust->charge(),clust->sizeX(),clust->sizeY()) << endl;
	 }
       }
      }
    }
    cout << "hitpattern" << endl;
    for(int i=0; i<p.numberOfHits(); i++){      p.printHitPattern(i,std::cout);    }
    cout << "expected inner " << ninner << endl;
    for(int i=0; i<pinner.numberOfHits(); i++){      pinner.printHitPattern(i,std::cout);    }
    cout << "expected outer " << nouter << endl;
    for(int i=0; i<pouter.numberOfHits(); i++){      pouter.printHitPattern(i,std::cout);    }
  }
}

namespace {

  bool recTrackLessZ(const reco::TransientTrack & tk1,
                     const reco::TransientTrack & tk2)
  {
    return tk1.stateAtBeamLine().trackStateAtPCA().position().z() < tk2.stateAtBeamLine().trackStateAtPCA().position().z();
  }

}




void PrimaryVertexAnalyzer4PU::printPVTrks(const Handle<reco::TrackCollection> &recTrks, 
					   const Handle<reco::VertexCollection> &recVtxs,  
					   std::vector<SimPart>& tsim,
					   std::vector<SimEvent>& simEvt,
					   bool selectedOnly){
  // make a printout of the tracks selected for PV reconstructions, show matching MC tracks, too

  vector<TransientTrack>  selTrks;
  for(reco::TrackCollection::const_iterator t=recTrks->begin();
      t!=recTrks->end(); ++t){
    TransientTrack  tt = theB_->build(&(*t));  tt.setBeamSpot(vertexBeamSpot_);
    if( (!selectedOnly) || (selectedOnly && theTrackFilter(tt))){     selTrks.push_back(tt);    }
  }
  if (selTrks.size()==0) return;
  stable_sort(selTrks.begin(), selTrks.end(), recTrackLessZ);

  // select tracks like for PV reconstruction and match them to sim tracks
  reco::TrackCollection selRecTrks;

  for(unsigned int i=0; i<selTrks.size(); i++){ selRecTrks.push_back(selTrks[i].track());} 
  int* rectosim=supf(tsim, selRecTrks);



  // now dump in a format similar to the clusterizer
  cout << "printPVTrks" << endl;
  cout << "----          z +/- dz     1bet-m     ip +/-dip         pt   phi   eta";
  if((tsim.size()>0)||(simEvt.size()>0)) {cout << " type     pdg    zvtx    zdca      dca    zvtx   zdca    dsz";}
  cout << endl;

  cout.precision(4);
  int isel=0;
  for(unsigned int i=0; i<selTrks.size(); i++){
    if  (selectedOnly || (theTrackFilter(selTrks[i]))) {
          cout <<  setw (3)<< isel;
	  isel++;
    }else{
      cout <<  "   ";
    }


    // is this track in the tracklist of a recvtx ?
    int vmatch=-1;
    int iv=0;
    for(reco::VertexCollection::const_iterator v=recVtxs->begin(); 
	v!=recVtxs->end(); ++v){
      if ( (v->ndof()<-1) || (v->chi2()<=0) ) continue;  // skip clusters 
      for(trackit_t tv=v->tracks_begin(); tv!=v->tracks_end(); tv++){
	const reco::Track & RTv=*(tv->get());  
	if(selTrks[i].track().vz()==RTv.vz()) {vmatch=iv;}
      }
      iv++;
    }

    double tz=(selTrks[i].stateAtBeamLine().trackStateAtPCA()).position().z();
    double tantheta=tan((selTrks[i].stateAtBeamLine().trackStateAtPCA()).momentum().theta());
    double tdz0= selTrks[i].track().dzError();
    double tdz2= pow(selTrks[i].track().dzError(),2)+ (pow(vertexBeamSpot_.BeamWidthX(),2)+pow(vertexBeamSpot_.BeamWidthY(),2))/pow(tantheta,2);
    
    if(vmatch>-1){
      cout << "["<<setw(2)<<vmatch<<"]";
    }else{
      //int status=theTrackFilter.status(selTrks[i]);
      int status=0;
      if(status==0){
	cout <<"    ";
      }else{
	if(status&0x1){cout << "i";}else{cout << ".";};
	if(status&0x2){cout << "c";}else{cout << ".";};
	if(status&0x4){cout << "h";}else{cout << ".";};
	if(status&0x8){cout << "q";}else{cout << ".";};
      }
    }
    cout  <<  setw (8) << fixed << setprecision(4)<<  tz << " +/-" <<  setw (6)<< sqrt(tdz2);
    

    // track quality and hit information, see DataFormats/TrackReco/interface/HitPattern.h
    if(selTrks[i].track().quality(reco::TrackBase::highPurity)){ cout << " *";}else{cout <<"  ";}
    if(selTrks[i].track().hitPattern().hasValidHitInFirstPixelBarrel()){cout <<"+";}else{cout << "-";}
    cout << setw(1) << selTrks[i].track().hitPattern().pixelBarrelLayersWithMeasurement();
    cout << setw(1) << selTrks[i].track().hitPattern().pixelEndcapLayersWithMeasurement(); 
    cout << setw(1) << hex << selTrks[i].track().hitPattern().trackerLayersWithMeasurement()-selTrks[i].track().hitPattern().pixelLayersWithMeasurement()<<dec;
    cout << "-" << setw(1)<<hex <<selTrks[i].track().trackerExpectedHitsOuter().numberOfHits() << dec;

    
    Measurement1D IP=selTrks[i].stateAtBeamLine().transverseImpactParameter();
    cout << setw (8) << IP.value() << "+/-" << setw (6) << IP.error();
    if(selTrks[i].track().ptError()<1){
      cout << " " << setw(8) << setprecision(2)  << selTrks[i].track().pt()*selTrks[i].track().charge();
    }else{
      cout << " " << setw(7) << setprecision(1)  << selTrks[i].track().pt()*selTrks[i].track().charge() << "*";
      //cout << "+/-" << setw(6)<< setprecision(2) << selTrks[i].track().ptError();
    }
    cout << " " << setw(5) << setprecision(2) << selTrks[i].track().phi()
	 << " " << setw(5) << setprecision(2) << selTrks[i].track().eta() ;



    // print MC info, if available
    if(simEvt.size()>0){
      reco::Track t=selTrks[i].track();
      try{
	TrackingParticleRef tpr = z2tp_[t.vz()];
	const TrackingVertex *parentVertex= tpr->parentVertex().get();
	//double vx=parentVertex->position().x(); // problems with tpr->vz()
	//double vy=parentVertex->position().y(); work in progress
	double vz=parentVertex->position().z();
	cout << " " << tpr->eventId().event();
	cout << " " << setw(5) << tpr->pdgId();
	cout << " " << setw(8) << setprecision(4) << vz;
      }catch (...){
	cout << " not matched";
      }
    }else{
      //    cout << "  " << rectosim[i];
      if(rectosim[i]>0){
	if(tsim[rectosim[i]].type==0){	cout << " prim " ;}else{cout << " sec  ";}
	cout << " " << setw(5) << tsim[rectosim[i]].pdg;
	cout << " " << setw(8) << setprecision(4) << tsim[rectosim[i]].zvtx;
	cout << " " << setw(8) << setprecision(4) << tsim[rectosim[i]].zdcap;
	cout << " " << setw(8) << setprecision(4) << tsim[rectosim[i]].ddcap;
	double zvtxpull=(tz-tsim[rectosim[i]].zvtx)/sqrt(tdz2);
	cout << setw(6) << setprecision(1) << zvtxpull;
	double zdcapull=(tz-tsim[rectosim[i]].zdcap)/tdz0;
	cout << setw(6) << setprecision(1) << zdcapull;
	double dszpull=(selTrks[i].track().dsz()-tsim[rectosim[i]].par[4])/selTrks[i].track().dszError();
	cout << setw(6) << setprecision(1) << dszpull;
      }
    }
    cout << endl;
  }
  delete rectosim;
}


void PrimaryVertexAnalyzer4PU::matchRecTracksToVertex(simPrimaryVertex & pv, 
						   const std::vector<SimPart > & tsim,
						   const edm::Handle<reco::TrackCollection> & recTrks)
{
  // find all recTracks that belong to this simulated vertex (not just the ones that are used in
  // matching recVertex)

  std::cout << "dump rec tracks: " << std::endl;
  int irec=0;
  for(reco::TrackCollection::const_iterator t=recTrks->begin();
      t!=recTrks->end(); ++t){
    reco::TrackBase::ParameterVector  p = t->parameters();
    std::cout  << irec++ << ") " << p <<  std::endl;
  }

  std::cout << "match sim tracks: " << std::endl;
  pv.matchedRecTrackIndex.clear();
  pv.nMatchedTracks=0;
  int isim=0;
  for(std::vector<SimPart>::const_iterator s=tsim.begin();
      s!=tsim.end(); ++s){
    std::cout  << isim++ << " " << s->par;// << std::endl;
    int imatch=-1;
    int irec=0;
    for(reco::TrackCollection::const_iterator t=recTrks->begin();
	t!=recTrks->end(); ++t){
      reco::TrackBase::ParameterVector  p = t->parameters();
      if (match(s->par,p)){	imatch=irec; }
      irec++;
    }
    pv.matchedRecTrackIndex.push_back(imatch);
    if(imatch>-1){ 
      pv.nMatchedTracks++; 
      std::cout << " matched to rec trk" << imatch << std::endl;
    }else{
      std::cout << " not matched" << std::endl;
    }
  }
}
/********************************************************************************************************/






/********************************************************************************************************/

void PrimaryVertexAnalyzer4PU::getTc(const std::vector<reco::TransientTrack>& tracks, 
			       double & Tc, double & chsq, double & dzmax, double & dztrim, double & m4m2){
  if (tracks.size()<2){ Tc=-1; chsq=-1; dzmax=-1; dztrim=-1; m4m2=-1; return;}

  double sumw=0, sumwz=0, sumww=0,sumwwz=0,sumwwzz=0;
  double zmin=1e10, zmin1=1e10, zmax1=-1e10, zmax=-1e10;
  double m4=0,m3=0,m2=0,m1=0,m0=0;
  for(vector<reco::TransientTrack>::const_iterator it=tracks.begin(); it!=tracks.end(); it++){
     double tantheta=tan(((*it).stateAtBeamLine().trackStateAtPCA()).momentum().theta());
     reco::BeamSpot beamspot=(it->stateAtBeamLine()).beamSpot();
     double z=((*it).stateAtBeamLine().trackStateAtPCA()).position().z();
     double dz2= pow((*it).track().dzError(),2)+pow(beamspot.BeamWidthX()/tantheta,2);
   //    t.dz2= pow((*it).track().dzError(),2) + pow(wxy0/tantheta,2) +  1./(1.+exp(pow(t.ip/t.dip,2)-pow(2.)))*pow(ip/tantheta,2);
     double w=1./dz2;  // take p=1
     sumw    += w;
     sumwz   += w*z;
     sumwwz  += w*w*z;;
     sumwwzz += w*w*z*z;
     sumww   += w*w;
     m0      += w;
     m1      += w*z;
     m2      += w*z*z;
     m3      += w*z*z*z;
     m4      += w*z*z*z*z;
     if(dz2<pow(0.1,2)){
       if(z<zmin1){zmin1=z;}    if(z<zmin){zmin1=zmin; zmin=z;}
       if(z>zmax1){zmax1=z;}    if(z>zmax){zmax1=zmax; zmax=z;}
     }
  }
  double z=sumwz/sumw;
  double a=sumwwzz-2*z*sumwwz+z*z*sumww;
  double b=sumw;
  if(tracks.size()>1){
    chsq=(m2-m0*z*z)/(tracks.size()-1);
    Tc=2.*a/b;
    m4m2=sqrt((m4-4*m3*z+6*m2*z*z-3*m1*z*z*z+m0*z*z*z*z)/(m2-2*m1*z+z*z*m0));
  }else{
    chsq=0;
    Tc=0;
    m4m2=0;
  }
  dzmax=zmax-zmin;
  dztrim=zmax1-zmin1;// truncated
}
/********************************************************************************************************/





/********************************************************************************************************/
bool PrimaryVertexAnalyzer4PU::truthMatchedTrack( edm::RefToBase<reco::Track> track, TrackingParticleRef & tpr)

/********************************************************************************************************/
// for a reco track select the matching tracking particle, always use this function to make sure we
// are consistent
// to get the TrackingParticle form the TrackingParticleRef, use ->get();
{
  double f=0;
  try{
    std::vector<std::pair<TrackingParticleRef, double> > tp = r2s_[track];
    for (std::vector<std::pair<TrackingParticleRef, double> >::const_iterator it = tp.begin(); 
	 it != tp.end(); ++it) {
      
      if (it->second>f){
	tpr=it->first;
	f=it->second;
      }
    }
  } catch (Exception event) {
    // silly way of testing whether track is in r2s_
  }
  
  // sanity check on track parameters?
  
  return (f>0.5);
}
/********************************************************************************************************/






/********************************************************************************************************/
std::vector< edm::RefToBase<reco::Track> >  PrimaryVertexAnalyzer4PU::getTruthMatchedVertexTracks(
				       const reco::Vertex& v
				       )
// for vertex v get a list of tracks for which truth matching is available 
/********************************************************************************************************/
{
  std::vector<  edm::RefToBase<reco::Track> > b;
  TrackingParticleRef tpr;

  for(trackit_t tv=v.tracks_begin(); tv!=v.tracks_end(); tv++){
    edm::RefToBase<reco::Track> track=*tv;
    if (truthMatchedTrack(track, tpr)){
      b.push_back(*tv);
    }
  }


  return b;
}
/********************************************************************************************************/





/********************************************************************************************************/
std::vector<PrimaryVertexAnalyzer4PU::SimEvent> PrimaryVertexAnalyzer4PU::getSimEvents
(
 // const Event& iEvent, const EventSetup& iSetup,
 edm::Handle<TrackingParticleCollection>  TPCollectionH,
 edm::Handle<TrackingVertexCollection>  TVCollectionH,
 edm::Handle<View<Track> > trackCollectionH
 ){

  const TrackingParticleCollection* simTracks = TPCollectionH.product();
  const View<Track>  tC = *(trackCollectionH.product());


  vector<SimEvent> simEvt;
  map<EncodedEventId, unsigned int> eventIdToEventMap;
  map<EncodedEventId, unsigned int>::iterator id;
  bool dumpTP=false;
  for(TrackingParticleCollection::const_iterator it=simTracks->begin(); it!=simTracks->end(); it++){
    
    if( fabs(it->parentVertex().get()->position().z())>100.) continue; // skip funny entries @ -13900

    unsigned int event=0;  //note, this is no longer the same as eventId().event()
    id=eventIdToEventMap.find(it->eventId());
    if (id==eventIdToEventMap.end()){

      // new event here
      SimEvent e;
      e.eventId=it->eventId();
      event=simEvt.size();
      const TrackingVertex *parentVertex= it->parentVertex().get();
      if(DEBUG_){
	cout << "getSimEvents: ";
	cout << it->eventId().bunchCrossing() << "," <<  it->eventId().event() 
	     << " z="<< it->vz() << " " 
	     << parentVertex->eventId().bunchCrossing() << ","  <<parentVertex->eventId().event() 
	     << " z=" << parentVertex->position().z() << endl;
      }
      if (it->eventId()==parentVertex->eventId()){
	//e.x=it->vx(); e.y=it->vy(); e.z=it->vz();// wrong ???
	e.x=parentVertex->position().x();
	e.y=parentVertex->position().y();
	e.z=parentVertex->position().z();
	if(e.z<-100){
	  dumpTP=true;
	}
      }else{
	e.x=0; e.y=0; e.z=-88.;
      }
      simEvt.push_back(e);
      eventIdToEventMap[e.eventId]=event;
    }else{
      event=id->second;
    }
      

    simEvt[event].tp.push_back(&(*it));
    if( (abs(it->eta())<2.5) && (it->charge()!=0) ){
      simEvt[event].sumpt2+=pow(it->pt(),2); // should keep track of decays ?
      simEvt[event].sumpt+=it->pt(); 
    }
  }

  if(dumpTP){
    for(TrackingParticleCollection::const_iterator it=simTracks->begin(); it!=simTracks->end(); it++){
      std::cout << *it << std::endl;
    } 
  }


  for(View<Track>::size_type i=0; i<tC.size(); ++i) {
    RefToBase<Track> track(trackCollectionH, i);
    TrackingParticleRef tpr;
    if( truthMatchedTrack(track,tpr)){

      if( eventIdToEventMap.find(tpr->eventId())==eventIdToEventMap.end() ){ cout << "Bug in getSimEvents" << endl; break; }

      z2tp_[track.get()->vz()]=tpr;

      unsigned int event=eventIdToEventMap[tpr->eventId()];
      double ipsig=0,ipdist=0;
      const TrackingVertex *parentVertex= tpr->parentVertex().get();
      double vx=parentVertex->position().x(); // problems with tpr->vz()
      double vy=parentVertex->position().y();
      double vz=parentVertex->position().z();
      double d=sqrt(pow(simEvt[event].x-vx,2)+pow(simEvt[event].y-vy,2)+pow(simEvt[event].z-vz,2))*1.e4;
      ipdist=d;
      double dxy=track->dxy(vertexBeamSpot_.position());
      ipsig=dxy/track->dxyError();


      TransientTrack  t = theB_->build(tC[i]); 
      t.setBeamSpot(vertexBeamSpot_);   
      if (theTrackFilter(t)){
 	simEvt[event].tk.push_back(t);
 	if(ipdist<5){simEvt[event].tkprim.push_back(t);}
 	if(ipsig<5){simEvt[event].tkprimsel.push_back(t);}
      }
      
    }
  }


  
  AdaptiveVertexFitter theFitter;
  cout << "SimEvents " << simEvt.size()  <<  endl;
  for(unsigned int i=0; i<simEvt.size(); i++){

    if(simEvt[i].tkprim.size()>0){

      getTc(simEvt[i].tkprimsel, simEvt[i].Tc, simEvt[i].chisq, simEvt[i].dzmax, simEvt[i].dztrim, simEvt[i].m4m2);
      TransientVertex v = theFitter.vertex(simEvt[i].tkprim, vertexBeamSpot_);
      if (v.isValid()){
	simEvt[i].xfit=v.position().x();
	simEvt[i].yfit=v.position().y();
	simEvt[i].zfit=v.position().z();
	//	if (simEvt[i].z<-80.){simEvt[i].z=v.position().z();} // for now
      }
    }


    if(DEBUG_){
      cout << i <<"  )   nTP="  << simEvt[i].tp.size()
	   << "   z=" <<  simEvt[i].z
	   << "    recTrks="  << simEvt[i].tk.size() 
	   << "    recTrksPrim="  << simEvt[i].tkprim.size() 
	   << "   zfit=" << simEvt[i].zfit
	   << endl;
    }
  }
 
  return simEvt;
}


std::vector<PrimaryVertexAnalyzer4PU::simPrimaryVertex> PrimaryVertexAnalyzer4PU::getSimPVs(
				      const Handle<HepMCProduct> evtMC)
{
  if(DEBUG_){std::cout << "getSimPVs HepMC " << std::endl;}

  std::vector<PrimaryVertexAnalyzer4PU::simPrimaryVertex> simpv;
  const HepMC::GenEvent* evt=evtMC->GetEvent();
  if (evt) {
//     std::cout << "process id " <<evt->signal_process_id()<<std::endl;
//     std::cout <<"signal process vertex "<< ( evt->signal_process_vertex() ?
// 					     evt->signal_process_vertex()->barcode() : 0 )   <<std::endl;
//     std::cout <<"number of vertices " << evt->vertices_size() << std::endl;


    //int idx=0;
    for(HepMC::GenEvent::vertex_const_iterator vitr= evt->vertices_begin();
	vitr != evt->vertices_end(); ++vitr ) 
      { // loop for vertex ...

	HepMC::FourVector pos = (*vitr)->position();
	//	if (pos.t()>0) { continue;} // skip secondary vertices, doesn't work for some samples

	if (fabs(pos.z())>1000) continue;  // skip funny junk vertices

	bool hasMotherVertex=false;
	//std::cout << "mothers" << std::endl;
	for ( HepMC::GenVertex::particle_iterator
	      mother  = (*vitr)->particles_begin(HepMC::parents);
	      mother != (*vitr)->particles_end(HepMC::parents);
              ++mother ) {
	  HepMC::GenVertex * mv=(*mother)->production_vertex();
	  if (mv) {hasMotherVertex=true;}
	  //std::cout << "\t"; (*mother)->print();
	}
	if(hasMotherVertex) {continue;}


	// could be a new vertex, check  all primaries found so far to avoid multiple entries
        const double mm=0.1;  
	simPrimaryVertex sv(pos.x()*mm,pos.y()*mm,pos.z()*mm);
	simPrimaryVertex *vp=NULL;  // will become non-NULL if a vertex is found and then point to it
	for(std::vector<simPrimaryVertex>::iterator v0=simpv.begin();
	    v0!=simpv.end(); v0++){
	  if( (fabs(sv.x-v0->x)<1e-5) && (fabs(sv.y-v0->y)<1e-5) && (fabs(sv.z-v0->z)<1e-5)){
	    vp=&(*v0);
	    break;
	  }
	}
	if(!vp){
	  // this is a new vertex
	  //std::cout << "this is a new vertex" << sv.x << " " << sv.y << " " << sv.z <<std::endl;
	  simpv.push_back(sv);
	  vp=&simpv.back();
// 	}else{
// 	  std::cout << "this is not a new vertex" << std::endl;
	}

	
	// store the gen vertex barcode with this simpv
	vp->genVertex.push_back((*vitr)->barcode());


	// collect final state descendants and sum up momenta etc
	for ( HepMC::GenVertex::particle_iterator
	      daughter  = (*vitr)->particles_begin(HepMC::descendants);
	      daughter != (*vitr)->particles_end(HepMC::descendants);
              ++daughter ) {
	  //std::cout << "checking daughter  type " << (*daughter)->pdg_id() << " final :" <<isFinalstateParticle(*daughter) << std::endl;
	  if (isFinalstateParticle(*daughter)){ 
	    if ( find(vp->finalstateParticles.begin(), vp->finalstateParticles.end(),(*daughter)->barcode())
		 == vp->finalstateParticles.end()){
	      vp->finalstateParticles.push_back((*daughter)->barcode());
	      HepMC::FourVector m=(*daughter)->momentum();
	      //std::cout << "adding particle to primary " << m.px()<< " "  << m.py() << " "  << m.pz() << std::endl; 
	      vp->ptot.setPx(vp->ptot.px()+m.px());
	      vp->ptot.setPy(vp->ptot.py()+m.py());
	      vp->ptot.setPz(vp->ptot.pz()+m.pz());
	      vp->ptot.setE(vp->ptot.e()+m.e());
	      vp->ptsq+=(m.perp())*(m.perp());
	      // count relevant particles
	      if ( (m.perp()>0.2) && (fabs(m.pseudoRapidity())<2.5) && isCharged( *daughter ) ){
		vp->nGenTrk++;
	      }
	      
	      hsimPV["rapidity"]->Fill(m.pseudoRapidity());
	      if( (m.perp()>0.8) &&  isCharged( *daughter ) ){
		hsimPV["chRapidity"]->Fill(m.pseudoRapidity());
	      }
	      hsimPV["pt"]->Fill(m.perp());
	    }//new final state particle for this vertex
	  }//daughter is a final state particle
	}


	//idx++;
      }
  }
  if(verbose_){
    cout << "------- PrimaryVertexAnalyzer4PU simPVs -------" <<  endl;
    for(std::vector<simPrimaryVertex>::iterator v0=simpv.begin();
	v0!=simpv.end(); v0++){
      cout << "z=" << v0->z 
	   << "  px=" << v0->ptot.px()
	   << "  py=" << v0->ptot.py()
	   << "  pz=" << v0->ptot.pz() 
	   << "  pt2="<< v0->ptsq 
	   << endl;
    }
    cout << "-----------------------------------------------" << endl;
  }
  return simpv;
}








/* get sim pv from TrackingParticles/TrackingVertex */
std::vector<PrimaryVertexAnalyzer4PU::simPrimaryVertex> PrimaryVertexAnalyzer4PU::getSimPVs(
											  const edm::Handle<TrackingVertexCollection> tVC
											  )
{
  std::vector<PrimaryVertexAnalyzer4PU::simPrimaryVertex> simpv;
  //std::cout <<"number of vertices " << tVC->size() << std::endl;

  if(DEBUG_){std::cout << "getSimPVs TrackingVertexCollection " << std::endl;}

  for (TrackingVertexCollection::const_iterator v = tVC -> begin(); v != tVC -> end(); ++v) {

    if(DEBUG_){
      std::cout << (v->eventId()).event() << v -> position() << v->g4Vertices().size() <<" "  <<v->genVertices().size() <<  "   t=" <<v->position().t()*1.e12 <<"    ==0:" <<(v->position().t()>0) << std::endl;
      for( TrackingVertex::g4v_iterator gv=v->g4Vertices_begin(); gv!=v->g4Vertices_end(); gv++){
	std::cout << *gv << std::endl;
      }
      std::cout << "----------" << std::endl;
    }
 
    //    bool hasMotherVertex=false;
    if ((unsigned int) v->eventId().event()<simpv.size()) continue;
	//if (v->position().t()>0) { continue;} // skip secondary vertices (obsolete + doesn't work)
    if (fabs(v->position().z())>1000) continue;  // skip funny junk vertices
    
    // could be a new vertex, check  all primaries found so far to avoid multiple entries
    const double mm=1.0; // for tracking vertices
    simPrimaryVertex sv(v->position().x()*mm,v->position().y()*mm,v->position().z()*mm);

    //cout << "sv: " << (v->eventId()).event() << endl;
    sv.eventId=v->eventId();

    for (TrackingParticleRefVector::iterator iTrack = v->daughterTracks_begin(); iTrack != v->daughterTracks_end(); ++iTrack){
      //cout <<((**iTrack).eventId()).event() << " ";  // an iterator of Refs, dereference twice. Cool eyh?
      sv.eventId=(**iTrack).eventId();
    }
    //cout <<endl;
    simPrimaryVertex *vp=NULL;  // will become non-NULL if a vertex is found and then point to it
    for(std::vector<simPrimaryVertex>::iterator v0=simpv.begin();
	v0!=simpv.end(); v0++){
      if( (sv.eventId==v0->eventId) && (fabs(sv.x-v0->x)<1e-5) && (fabs(sv.y-v0->y)<1e-5) && (fabs(sv.z-v0->z)<1e-5)){
	vp=&(*v0);
	break;
      }
    }
    if(!vp){
      // this is a new vertex
      if(DEBUG_){std::cout << "this is a new vertex " << sv.eventId.event() << "   "  << sv.x << " " << sv.y << " " << sv.z <<std::endl;}
      simpv.push_back(sv);
      vp=&simpv.back();
    }else{
      if(DEBUG_){std::cout << "this is not a new vertex"  << sv.x << " " << sv.y << " " << sv.z <<std::endl;}
    }


    // Loop over daughter track(s)
    if(DEBUG_){
      for (TrackingVertex::tp_iterator iTP = v -> daughterTracks_begin(); iTP != v -> daughterTracks_end(); ++iTP) {
	std::cout << "  Daughter momentum:      " << (*(*iTP)).momentum();
	std::cout << "  Daughter type     " << (*(*iTP)).pdgId();
	std::cout << std::endl;
      }
    }
  }
  if(DEBUG_){  
    cout << "------- PrimaryVertexAnalyzer4PU simPVs from TrackingVertices -------" <<  endl; 
    for(std::vector<simPrimaryVertex>::iterator v0=simpv.begin();
	v0!=simpv.end(); v0++){
      cout << "z=" << v0->z << "  event=" << v0->eventId.event() << endl;
    }
    cout << "-----------------------------------------------" << endl;
  }
  return simpv;
}






// ------------ method called to produce the data  ------------
void
PrimaryVertexAnalyzer4PU::analyze(const Event& iEvent, const EventSetup& iSetup)
{
  
  std::vector<simPrimaryVertex> simpv;  //  a list of primary MC vertices
  std::vector<SimPart> tsim;
  std::string mcproduct="generator";  // starting with 3_1_0 pre something

  eventcounter_++;
  run_             = iEvent.id().run();
  luminosityBlock_ = iEvent.luminosityBlock();
  event_           = iEvent.id().event();
  bunchCrossing_   = iEvent.bunchCrossing();
  orbitNumber_     = iEvent.orbitNumber();

  dumpThisEvent_ = false;



  if(verbose_){
    std::cout << endl 
	      << "PrimaryVertexAnalyzer4PU::analyze   event counter=" << eventcounter_
	      << " Run=" << run_ << "  LumiBlock " << luminosityBlock_ << "  event  " << event_
	      << " bx=" << bunchCrossing_ <<  " orbit=" << orbitNumber_ 
      //<< " selected = " << good
	      << std::endl;
  }


   try{
    iSetup.getData(pdt_);
  }catch(const Exception&){
    std::cout << "Some problem occurred with the particle data table. This may not work !" <<std::endl;
  }

  Handle<reco::VertexCollection> recVtxs;
  bool bnoBS=iEvent.getByLabel("offlinePrimaryVertices", recVtxs);
  
  Handle<reco::VertexCollection> recVtxsBS;
  bool bBS=iEvent.getByLabel("offlinePrimaryVerticesWithBS", recVtxsBS);
  
  Handle<reco::VertexCollection> recVtxsDA;
  bool bDA=iEvent.getByLabel("offlinePrimaryVerticesDA", recVtxsDA);

//   Handle<reco::VertexCollection> recVtxsPIX;
//   bool bPIX=iEvent.getByLabel("pixelVertices", recVtxsPIX);
//   bPIX=false;

//   Handle<reco::VertexCollection> recVtxsMVF;
//   bool bMVF=iEvent.getByLabel("offlinePrimaryVerticesMVF", recVtxsMVF);

  Handle<reco::TrackCollection> recTrks;
  iEvent.getByLabel(recoTrackProducer_, recTrks);


  int nhighpurity=0, ntot=0;
  for(reco::TrackCollection::const_iterator t=recTrks->begin(); t!=recTrks->end(); ++t){  
    ntot++;
    if(t->quality(reco::TrackBase::highPurity)) nhighpurity++;
  } 
  if(ntot>10)  hnoBS["highpurityTrackFraction"]->Fill(float(nhighpurity)/float(ntot));
  if((recoTrackProducer_=="generalTracks")&&(nhighpurity<0.25*ntot)){
    if(verbose_){ cout << "rejected, " << nhighpurity << " highpurity  out of  " << ntot << "  total tracks "<< endl<< endl;}
    return;
  }



  

  if(iEvent.getByLabel(beamSpot_, recoBeamSpotHandle_)){
    vertexBeamSpot_= *recoBeamSpotHandle_;
    wxy2_=pow(vertexBeamSpot_.BeamWidthX(),2)+pow(vertexBeamSpot_.BeamWidthY(),2);
    Fill(hsimPV, "xbeam",vertexBeamSpot_.x0()); Fill(hsimPV, "wxbeam",vertexBeamSpot_.BeamWidthX());
    Fill(hsimPV, "ybeam",vertexBeamSpot_.y0()); Fill(hsimPV, "wybeam",vertexBeamSpot_.BeamWidthY());
    Fill(hsimPV, "zbeam",vertexBeamSpot_.z0());// Fill("wzbeam",vertexBeamSpot_.BeamWidthZ());
  }else{
    cout << " beamspot not found, using dummy " << endl;
    vertexBeamSpot_=reco::BeamSpot();// dummy
  }


  if(bnoBS){
    if((recVtxs->begin()->isValid())&&(recVtxs->begin()->ndof()>1)&&(recVtxs->begin()->ndof()>(0.0*recVtxs->begin()->tracksSize()))){  // was 5 and 0.2
      Fill(hnoBS,"xrecBeamvsdxXBS",recVtxs->begin()->xError(),recVtxs->begin()->x()-vertexBeamSpot_.x0());
      Fill(hnoBS,"yrecBeamvsdyXBS",recVtxs->begin()->yError(),recVtxs->begin()->y()-vertexBeamSpot_.y0());

      if(printXBS_) {
	cout << Form("XBS %10d %5d %10d  %4d   %lu %5.2f    %8f %8f       %8f %8f      %8f %8f",
		      run_,luminosityBlock_,event_,bunchCrossing_,
		      (unsigned long)(recVtxs->begin()->tracksSize()), recVtxs->begin()->ndof(),
		      recVtxs->begin()->x(), 		   recVtxs->begin()->xError(), 
		      recVtxs->begin()->y(), 		   recVtxs->begin()->yError(), 
		      recVtxs->begin()->z(), 		   recVtxs->begin()->zError()
		     ) << endl; 
      }

    }
  }

 
  // for the associator
  Handle<View<Track> > trackCollectionH;
  iEvent.getByLabel(recoTrackProducer_,trackCollectionH);

  Handle<HepMCProduct> evtMC;

  Handle<SimVertexContainer> simVtxs;
  iEvent.getByLabel( simG4_, simVtxs);
  
  Handle<SimTrackContainer> simTrks;
  iEvent.getByLabel( simG4_, simTrks);





  edm::Handle<TrackingParticleCollection>  TPCollectionH ;
  edm::Handle<TrackingVertexCollection>    TVCollectionH ;
  bool gotTP=iEvent.getByLabel("mix","MergedTrackTruth",TPCollectionH);
  bool gotTV=iEvent.getByLabel("mix","MergedTrackTruth",TVCollectionH);


  iSetup.get<TransientTrackRecord>().get("TransientTrackBuilder",theB_);
  fBfield_=((*theB_).field()->inTesla(GlobalPoint(0.,0.,0.))).z();

 
  vector<SimEvent> simEvt;
  if (gotTP && gotTV ){

    edm::ESHandle<TrackAssociatorBase> theHitsAssociator;
    iSetup.get<TrackAssociatorRecord>().get("TrackAssociatorByHits",theHitsAssociator);
    associatorByHits_ = (TrackAssociatorBase *) theHitsAssociator.product();
    r2s_ =   associatorByHits_->associateRecoToSim (trackCollectionH,TPCollectionH, &iEvent, &iSetup ); 
    //simEvt=getSimEvents(iEvent, TPCollectionH, TVCollectionH, trackCollectionH);
    simEvt=getSimEvents(TPCollectionH, TVCollectionH, trackCollectionH);

    if (simEvt.size()==0){
      cout << " !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" << endl;
      cout << " !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" << endl;
      cout << " !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" << endl;
      cout << " !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" << endl;
      cout << " !!!!!!!!!!!!!!!!!!!!!!   got TrackingParticles but no simEvt   !!!!!!!!!!!!!!!!!" << endl;
      cout << " dumping Tracking particles " << endl;
      const TrackingParticleCollection* simTracks = TPCollectionH.product();
      for(TrackingParticleCollection::const_iterator it=simTracks->begin(); it!=simTracks->end(); it++){
	cout << *it << endl;
      }
      cout << " dumping Tracking Vertices " << endl;
      const TrackingVertexCollection* tpVtxs = TVCollectionH.product();
      for(TrackingVertexCollection::const_iterator iv=tpVtxs->begin(); iv!=tpVtxs->end(); iv++){
	cout << *iv << endl;
      }
      if(iEvent.getByLabel(mcproduct,evtMC)){
	cout << "dumping simTracks" << endl;
	for(SimTrackContainer::const_iterator t=simTrks->begin();  t!=simTrks->end(); ++t){
	  cout << *t << endl;
	}
	cout << "dumping simVertexes" << endl;
	for(SimVertexContainer::const_iterator vv=simVtxs->begin();  
	    vv!=simVtxs->end(); 
	    ++vv){
	  cout << *vv << endl;
	}
      }else{
	cout << "no hepMC" << endl;
      }
      cout << " !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" << endl;

      const HepMC::GenEvent* evt=evtMC->GetEvent();
      if(evt){
	std::cout << "process id " <<evt->signal_process_id()<<std::endl;
	std::cout <<"signal process vertex "<< ( evt->signal_process_vertex() ?
						 evt->signal_process_vertex()->barcode() : 0 )   <<std::endl;
	std::cout <<"number of vertices " << evt->vertices_size() << std::endl;
	evt->print();
      }else{
	cout << "no event in HepMC" << endl;
      }
      cout << " !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" << endl;

    }
  }


  

  if(gotTV){

    if(verbose_){
      cout << "Found Tracking Vertices " << endl;
    }
    simpv=getSimPVs(TVCollectionH);
    

  }else if(iEvent.getByLabel(mcproduct,evtMC)){

    simpv=getSimPVs(evtMC);

    if(verbose_){
      cout << "Using HepMCProduct " << endl;
      std::cout << "simtrks " << simTrks->size() << std::endl;
    }
    tsim = PrimaryVertexAnalyzer4PU::getSimTrkParameters(simTrks, simVtxs, simUnit_);
    
  }else{
    // if(verbose_({cout << "No MC info at all" << endl;}
  }




  // get the beam spot from the appropriate dummy vertex, if available
  for(reco::VertexCollection::const_iterator v=recVtxs->begin(); 
      v!=recVtxs->end(); ++v){
    if ((v->ndof()==-3) && (v->chi2()==0) ){ 
      myBeamSpot=math::XYZPoint(v->x(), v->y(), v->z());
    }
  }

  


  hsimPV["nsimvtx"]->Fill(simpv.size());
  for(std::vector<simPrimaryVertex>::iterator vsim=simpv.begin();
       vsim!=simpv.end(); vsim++){
    if(doMatching_){  
      matchRecTracksToVertex(*vsim, tsim, recTrks); // hepmc, based on track parameters
     }

     hsimPV["nbsimtksinvtx"]->Fill(vsim->nGenTrk);
     hsimPV["xsim"]->Fill(vsim->x*simUnit_);
     hsimPV["ysim"]->Fill(vsim->y*simUnit_);
     hsimPV["zsim"]->Fill(vsim->z*simUnit_);
     hsimPV["xsim1"]->Fill(vsim->x*simUnit_);
     hsimPV["ysim1"]->Fill(vsim->y*simUnit_);
     hsimPV["zsim1"]->Fill(vsim->z*simUnit_);
     Fill(hsimPV,"xsim2",vsim->x*simUnit_,vsim==simpv.begin());
     Fill(hsimPV,"ysim2",vsim->y*simUnit_,vsim==simpv.begin());
     Fill(hsimPV,"zsim2",vsim->z*simUnit_,vsim==simpv.begin());
     hsimPV["xsim2"]->Fill(vsim->x*simUnit_);
     hsimPV["ysim2"]->Fill(vsim->y*simUnit_);
     hsimPV["zsim2"]->Fill(vsim->z*simUnit_);
     hsimPV["xsim3"]->Fill(vsim->x*simUnit_);
     hsimPV["ysim3"]->Fill(vsim->y*simUnit_);
     hsimPV["zsim3"]->Fill(vsim->z*simUnit_);
     hsimPV["xsimb"]->Fill(vsim->x*simUnit_-vertexBeamSpot_.x0());
     hsimPV["ysimb"]->Fill(vsim->y*simUnit_-vertexBeamSpot_.y0());
     hsimPV["zsimb"]->Fill(vsim->z*simUnit_-vertexBeamSpot_.z0());
     hsimPV["xsimb1"]->Fill(vsim->x*simUnit_-vertexBeamSpot_.x0());
     hsimPV["ysimb1"]->Fill(vsim->y*simUnit_-vertexBeamSpot_.y0());
     hsimPV["zsimb1"]->Fill(vsim->z*simUnit_-vertexBeamSpot_.z0());
  }




   
   if(bnoBS){
     analyzeVertexCollection(hnoBS, recVtxs, recTrks, simpv,"noBS");
     analyzeVertexCollectionTP(hnoBS, recVtxs, recTrks, simEvt,"noBS");
   }
   if(bBS){
     analyzeVertexCollection(hBS, recVtxsBS, recTrks, simpv,"BS");
     analyzeVertexCollectionTP(hBS, recVtxsBS, recTrks, simEvt,"BS");
   }
   if(bDA){
     analyzeVertexCollection(hDA, recVtxsDA, recTrks, simpv,"DA");
     analyzeVertexCollectionTP(hDA, recVtxsDA, recTrks, simEvt,"DA");
   }
//    if(bPIX){
//      analyzeVertexCollection(hPIX, recVtxsPIX, recTrks, simpv,"PIX");
//      analyzeVertexCollectionTP(hPIX, recVtxsPIX, recTrks, simEvt,"PIX");
//    }



   if((dumpThisEvent_&& (dumpcounter_<100)) ||(verbose_ && (eventcounter_<ndump_))){
     cout << endl << "Event dump" << endl
	  << "event counter=" << eventcounter_
	  << " Run=" << run_ << "  LumiBlock " << luminosityBlock_ << "  event  " << event_
	  << " bx=" << bunchCrossing_ <<  " orbit=" << orbitNumber_ 
	  << std::endl;
     dumpcounter_++;

     //evtMC->GetEvent()->print();
     //printRecTrks(recTrks);  // very verbose !!
     
//      if (bPIX) printRecVtxs(recVtxsPIX,"pixel vertices");
     if (bnoBS) {printRecVtxs(recVtxs,"Offline without Beamspot");}
     if (bnoBS && (!bDA)){ printPVTrks(recTrks, recVtxs, tsim, simEvt, false);}
     if (bBS) printRecVtxs(recVtxsBS,"Offline with Beamspot");
     if (bDA) {
       printRecVtxs(recVtxsDA,"Offline DA");
       printPVTrks(recTrks, recVtxsDA, tsim, simEvt, false);
     }
     if (dumpcounter_<2){cout << "beamspot " << vertexBeamSpot_ << endl;}
   }

  if(verbose_){
    std::cout << std::endl;
  }
}

namespace {
bool lt(const std::pair<double,unsigned int>& a,const std::pair<double,unsigned int>& b ){
  return a.first<b.first;
}
}

/***************************************************************************************/
void PrimaryVertexAnalyzer4PU::printEventSummary(std::map<std::string, TH1*> & h,
						 const edm::Handle<reco::VertexCollection> recVtxs,
						 const edm::Handle<reco::TrackCollection> recTrks, 
						 vector<SimEvent> & simEvt,
						 const string message){
  // make a readable summary of the vertex finding if the TrackingParticles are availabe
  if (simEvt.size()==0){return;}


  // sort vertices in z ... for nicer printout

  vector< pair<double,unsigned int> >  zrecv;
  for(unsigned int idx=0; idx<recVtxs->size(); idx++){
    if ( (recVtxs->at(idx).ndof()<0) || (recVtxs->at(idx).chi2()<=0) ) continue;  // skip clusters 
    zrecv.push_back( make_pair(recVtxs->at(idx).z(),idx) );
  }
  stable_sort(zrecv.begin(),zrecv.end(),lt);

  // same for simulated vertices
  vector< pair<double,unsigned int> >  zsimv;
  for(unsigned int idx=0; idx<simEvt.size(); idx++){
    zsimv.push_back(make_pair(simEvt[idx].z, idx));
  }
  stable_sort(zsimv.begin(), zsimv.end(),lt);




  cout << "---------------------------" << endl;
  cout << "event counter = " << eventcounter_ << "   " << message << endl;
  cout << "---------------------------" << endl;
  cout << " z[cm]       rec -->    ";
  cout.precision(4);
  for(vector< pair<double,unsigned int> >::iterator itrec=zrecv.begin(); itrec!=zrecv.end(); itrec++){
    cout << setw(7) << fixed << itrec->first;
    if (itrec->second==0){cout << "*" ;}else{cout << " " ;}
  }
  cout << endl;
  cout << "                        ";
  for(vector< pair<double,unsigned int> >::iterator itrec=zrecv.begin(); itrec!=zrecv.end(); itrec++){
    cout << setw(7) << fixed << recVtxs->at(itrec->second).tracksSize();
    if (itrec->second==0){cout << "*" ;}else{cout << " " ;}
  }
  cout << "   rec tracks" << endl;
  cout << "                        ";
  map<unsigned int, int> truthMatchedVertexTracks;
  for(vector< pair<double,unsigned int> >::iterator itrec=zrecv.begin(); itrec!=zrecv.end(); itrec++){
    truthMatchedVertexTracks[itrec->second]=getTruthMatchedVertexTracks(recVtxs->at(itrec->second)).size();
    cout << setw(7) << fixed << truthMatchedVertexTracks[itrec->second];
    if (itrec->second==0){cout << "*" ;}else{cout << " " ;}
  }
  cout << "   truth matched " << endl;

  cout << "sim ------- trk  prim ----" << endl;



  map<unsigned int, unsigned int> rvmatch; // reco vertex matched to sim vertex  (sim to rec)
  map<unsigned int, double > nmatch;  // highest number of truth-matched tracks of ev found in a recvtx
  map<unsigned int, double > purity;  // highest purity of a rec vtx (i.e. highest number of tracks from the same simvtx)
  map<unsigned int, double > wpurity;  // same for the sum of weights

  for(vector< pair<double,unsigned int> >::iterator itrec=zrecv.begin(); itrec!=zrecv.end(); itrec++){
    purity[itrec->second]=0.;
    wpurity[itrec->second]=0.;
  }

  for(vector< pair<double,unsigned int> >::iterator itsim=zsimv.begin(); itsim!=zsimv.end(); itsim++){
    SimEvent* ev =&(simEvt[itsim->second]);


    cout.precision(4);
    if (itsim->second==0){
      cout << setw(8) << fixed << ev->z << ")*" << setw(5) << ev->tk.size() << setw(5) << ev->tkprim.size() << "  | ";
    }else{
      cout << setw(8) << fixed << ev->z << ") " << setw(5) << ev->tk.size() << setw(5) << ev->tkprim.size() << "  | ";
    }

    nmatch[itsim->second]=0;  // highest number of truth-matched tracks of ev found in a recvtx
    double matchpurity=0,matchwpurity=0;

    for(vector< pair<double,unsigned int> >::iterator itrec=zrecv.begin(); itrec!=zrecv.end(); itrec++){
      const reco::Vertex *v = &(recVtxs->at(itrec->second));

      // count tracks found in both, sim and rec
      double n=0,wt=0;
      for(vector<TransientTrack>::iterator te=ev->tk.begin(); te!=ev->tk.end(); te++){
	const reco::Track&  RTe=te->track();
	 for(trackit_t tv=v->tracks_begin(); tv!=v->tracks_end(); tv++){
	   const reco::Track & RTv=*(tv->get());  
	   if(RTe.vz()==RTv.vz()) {n++; wt+=v->trackWeight(*tv);}
	}
      }
      cout << setw(7) << int(n)<< " ";

      if (n > nmatch[itsim->second]){
	nmatch[itsim->second]=n;
	rvmatch[itsim->second]=itrec->second;
	matchpurity=n/truthMatchedVertexTracks[itrec->second];
	matchwpurity=wt/truthMatchedVertexTracks[itrec->second];
      }

      if(n > purity[itrec->second]){
	purity[itrec->second]=n;
      }

      if(wt > wpurity[itrec->second]){
	wpurity[itrec->second]=wt;
      }

    }// end of reco vertex loop

    cout << "  | ";
    if  (nmatch[itsim->second]>0 ){
      if(matchpurity>0.5){
	cout << "found  ";
      }else{
	cout << "merged ";
      }	
      cout << "  max eff. = "  << setw(8) << nmatch[itsim->second]/ev->tk.size() << " p=" << matchpurity << " w=" << matchwpurity <<  endl;
    }else{
      if(ev->tk.size()==0){
	cout  << " invisible" << endl;
      }else if (ev->tk.size()==1){
	cout << "single track " << endl;
      }else{
	cout << "lost " << endl;
      }
    }
  }
  cout << "---------------------------" << endl;

  //  the purity of the reconstructed vertex
  cout << "               purity   ";
  for(vector< pair<double,unsigned int> >::iterator itrec=zrecv.begin(); itrec!=zrecv.end(); itrec++){
    cout << setw(7) << fixed << purity[itrec->second]/truthMatchedVertexTracks[itrec->second];
    if (itrec->second==0){cout << "*" ;}else{cout << " " ;}
  }
  cout << endl;

//   //  classification of reconstructed vertex fake/real
//   cout << "                        ";
//   for(vector< pair<double,unsigned int> >::iterator itrec=zrecv.begin(); itrec!=zrecv.end(); itrec++){
//     cout << setw(7) << fixed << purity[itrec->second]/truthMatchedVertexTracks[itrec->second];
//     if (itrec->second==0){cout << "*" ;}else{cout << " " ;}
//   }
//   cout << endl;
  cout << "---------------------------" << endl;




  // list problematic tracks
  for(vector< pair<double,unsigned int> >::iterator itsim=zsimv.begin(); itsim!=zsimv.end(); itsim++){
    SimEvent* ev =&(simEvt[itsim->second]);

    for(vector<TransientTrack>::iterator te=ev->tk.begin(); te!=ev->tk.end(); te++){
      const reco::Track&  RTe=te->track();
      
      int ivassign=-1;  // will become the index of the vertex to which a track was assigned
      
      for(vector< pair<double,unsigned int> >::iterator itrec=zrecv.begin(); itrec!=zrecv.end(); itrec++){
	const reco::Vertex *v = &(recVtxs->at(itrec->second));

	for(trackit_t tv=v->tracks_begin(); tv!=v->tracks_end(); tv++){
	  const reco::Track & RTv=*(tv->get());  
	  if(RTe.vz()==RTv.vz()) {ivassign=itrec->second;}
	}
      }
      double tantheta=tan((te->stateAtBeamLine().trackStateAtPCA()).momentum().theta());
      reco::BeamSpot beamspot=(te->stateAtBeamLine()).beamSpot();
      //double z=(te->stateAtBeamLine().trackStateAtPCA()).position().z();
      double dz2= pow(RTe.dzError(),2)+pow(beamspot.BeamWidthX()/tantheta,2);
      
      if(ivassign==(int)rvmatch[itsim->second]){
	Fill(h,"correctlyassigned",RTe.eta(),RTe.pt());
	Fill(h,"ptcat",RTe.pt());
	Fill(h,"etacat",RTe.eta());
	Fill(h,"phicat",RTe.phi());
	Fill(h,"dzcat",sqrt(dz2));
      }else{
	Fill(h,"misassigned",RTe.eta(),RTe.pt());
	Fill(h,"ptmis",RTe.pt());
	Fill(h,"etamis",RTe.eta());
	Fill(h,"phimis",RTe.phi());
	Fill(h,"dzmis",sqrt(dz2));
	cout << "vertex " << setw(8) << fixed << ev->z;

	if (ivassign<0){
	  cout << " track lost                ";
	  // for some clusterizers there shouldn't be any lost tracks,
	  // are there differences in the track selection?
	}else{
	  cout << " track misassigned " << setw(8) << fixed << recVtxs->at(ivassign).z();
	}

	cout << "  track z=" << setw(8) << fixed  << RTe.vz() << "+/-" << RTe.dzError() << "  pt=" <<  setw(8) << fixed<< RTe.pt() << "  eta=" << setw(8) << fixed << RTe.eta()<< " sel=" <<theTrackFilter(*te);

	//
	//cout << " ztrack=" << te->track().vz();
	TrackingParticleRef tpr = z2tp_[te->track().vz()];
	double zparent=tpr->parentVertex().get()->position().z();
	if(zparent==ev->z) {
	  cout << " prim";
	}else{
	  cout << " sec ";
	}
	cout << "  id=" << tpr->pdgId();
	cout << endl;

	//
      }
    }// next simvertex-track

  }//next simvertex

  cout << "---------------------------" << endl;

}
/***************************************************************************************/




/***************************************************************************************/
void PrimaryVertexAnalyzer4PU::analyzeVertexCollectionTP(std::map<std::string, TH1*> & h,
			       const edm::Handle<reco::VertexCollection> recVtxs,
			       const edm::Handle<reco::TrackCollection> recTrks, 
							vector<SimEvent> & simEvt,
							const string message){
  
  //  cout <<"PrimaryVertexAnalyzer4PU::analyzeVertexCollectionTP size=" << simEvt.size() << endl;
  if(simEvt.size()==0)return;

  printEventSummary(h, recVtxs,recTrks,simEvt,message);

  //const int iSignal=0;  
  EncodedEventId iSignal=simEvt[0].eventId;
  Fill(h,"npu0",simEvt.size());


  for(vector<SimEvent>::iterator ev=simEvt.begin(); ev!=simEvt.end(); ev++){
    Fill(h,"Tc",    ev->Tc,    ev==simEvt.begin());
    Fill(h,"Chisq", ev->chisq, ev==simEvt.begin());
    if(ev->chisq>0)Fill(h,"logChisq", log(ev->chisq), ev==simEvt.begin());
    Fill(h,"dzmax", ev->dzmax, ev==simEvt.begin());
    Fill(h,"dztrim",ev->dztrim,ev==simEvt.begin());
    Fill(h,"m4m2",  ev->m4m2,  ev==simEvt.begin());
    if(ev->Tc>0){  Fill(h,"logTc",log(ev->Tc)/log(10.),ev==simEvt.begin());}
    

    for(vector<SimEvent>::iterator ev2=ev+1; ev2!=simEvt.end(); ev2++){
      vector<TransientTrack> xt;
      if((ev->tkprimsel.size()>0)&&(ev2->tkprimsel.size()>0)&&(ev->tkprimsel.size()+ev2->tkprimsel.size())>1){
	xt.insert (xt.end() ,ev->tkprimsel.begin(),ev->tkprimsel.end());
	xt.insert (xt.end() ,ev2->tkprimsel.begin(),ev2->tkprimsel.end());
	double xTc,xChsq,xDzmax,xDztrim,xm4m2;
	getTc(xt, xTc, xChsq, xDzmax, xDztrim,xm4m2);
	if(xTc>0){
	  Fill(h,"xTc",xTc,ev==simEvt.begin());
	  Fill(h,"logxTc",   log(xTc)/log(10),ev==simEvt.begin());
	  Fill(h,"xChisq",   xChsq,ev==simEvt.begin());
	  if(xChsq>0){Fill(h,"logxChisq",   log(xChsq),ev==simEvt.begin());};
	  Fill(h,"xdzmax",   xDzmax,ev==simEvt.begin());
	  Fill(h,"xdztrim",  xDztrim,ev==simEvt.begin());
	  Fill(h,"xm4m2",    xm4m2,ev==simEvt.begin());
	  
	}
      }
    }
  }
  
  // --------------------------------------- count actual rec vtxs ----------------------
  int nrecvtxs=0;//, nrecvtxs1=0, nrecvtxs2=0;
  for(reco::VertexCollection::const_iterator v=recVtxs->begin(); 
      v!=recVtxs->end(); ++v){
    if ( (v->isFake()) || (v->ndof()<0) || (v->chi2()<=0) ) continue;  // skip clusters 
    nrecvtxs++;
  }

  // --------------------------------------- fill the track assignment matrix ----------------------
  for(vector<SimEvent>::iterator ev=simEvt.begin(); ev!=simEvt.end(); ev++){
    ev->ntInRecVz.clear();  // just in case
    ev->zmatch=-99.;
    ev->nmatch=0;
    for(reco::VertexCollection::const_iterator v=recVtxs->begin(); 
	v!=recVtxs->end(); ++v){
      double n=0, wt=0;
      for(vector<TransientTrack>::iterator te=ev->tk.begin(); te!=ev->tk.end(); te++){
	const reco::Track&  RTe=te->track();
	for(trackit_t tv=v->tracks_begin(); tv!=v->tracks_end(); tv++){
	  const reco::Track & RTv=*(tv->get());  
	  if(RTe.vz()==RTv.vz()) {n++; wt+=v->trackWeight(*tv);}
	}
      }
      ev->ntInRecVz[v->z()]=n;
      if (n > ev->nmatch){ ev->nmatch=n; ev->zmatch=v->z(); ev->zmatch=v->z(); }
    }
  }
  
    // call a vertex a fake if for every sim vertex there is another recvertex containing more tracks
    // from that sim vertex than the current recvertex
  double nfake=0;
  for(reco::VertexCollection::const_iterator v=recVtxs->begin(); 
      v!=recVtxs->end(); ++v){
    bool matched=false;
    for(vector<SimEvent>::iterator ev=simEvt.begin(); ev!=simEvt.end(); ev++){
      if ((ev->nmatch>0)&&(ev->zmatch==v->z())){
	matched=true;
      }
    }
    if(!matched && !v->isFake()) {
      nfake++;
      cout << " fake rec vertex at z=" << v->z() << endl;
      // some histograms of fake vertex properties here
      Fill(h,"unmatchedVtxZ",v->z());
      Fill(h,"unmatchedVtxNdof",v->ndof());
    }
  }
  if(nrecvtxs>0){
    Fill(h,"unmatchedVtx",nfake);
    Fill(h,"unmatchedVtxFrac",nfake/nrecvtxs);
  }

  // --------------------------------------- match rec to sim ---------------------------------------
  for(reco::VertexCollection::const_iterator v=recVtxs->begin(); 
      v!=recVtxs->end(); ++v){

    if ( (v->ndof()<0) || (v->chi2()<=0) ) continue;  // skip clusters 
    double  nmatch=-1;      // highest number of tracks in recvtx v found in any event
    EncodedEventId evmatch;
    bool matchFound=false;
    double nmatchvtx=0;     // number of simvtcs contributing to recvtx v

    for(vector<SimEvent>::iterator ev=simEvt.begin(); ev!=simEvt.end(); ev++){

      double n=0;  // number of tracks that are in both, the recvtx v and the event ev
      for(vector<TransientTrack>::iterator te=ev->tk.begin(); te!=ev->tk.end(); te++){

	const reco::Track&  RTe=te->track();
	 for(trackit_t tv=v->tracks_begin(); tv!=v->tracks_end(); tv++){
	   const reco::Track & RTv=*(tv->get());  
	   if(RTe.vz()==RTv.vz()){ n++;}
	}
      }     

      // find the best match in terms of the highest number of tracks 
      // from a simvertex in this rec vertex
      if (n > nmatch){
	nmatch=n;
	evmatch=ev->eventId;
	matchFound=true;
      }
      if(n>0){
	nmatchvtx++;
      }
    }

    double nmatchany=getTruthMatchedVertexTracks(*v).size();  
    if (matchFound && (nmatchany>0)){
      //           highest number of tracks in recvtx matched to (the same) sim vertex
      // purity := -----------------------------------------------------------------
      //                  number of truth matched tracks in this recvtx
      double purity =nmatch/nmatchany; 
      Fill(h,"recmatchPurity",purity);
      if(v==recVtxs->begin()){
	Fill(h,"recmatchPurityTag",purity, (bool)(evmatch==iSignal));
      }else{
	Fill(h,"recmatchPuritynoTag",purity,(bool)(evmatch==iSignal));
      }
    }
    Fill(h,"recmatchvtxs",nmatchvtx);
    if(v==recVtxs->begin()){
      Fill(h,"recmatchvtxsTag",nmatchvtx);
    }else{
      Fill(h,"recmatchvtxsnoTag",nmatchvtx);
    }


     
  } // recvtx loop
  Fill(h,"nrecv",nrecvtxs);


  // --------------------------------------- match sim to rec  ---------------------------------------

  int npu1=0, npu2=0;

  for(vector<SimEvent>::iterator ev=simEvt.begin(); ev!=simEvt.end(); ev++){

    if(ev->tk.size()>0) npu1++;
    if(ev->tk.size()>1) npu2++;

    bool isSignal= ev->eventId==iSignal;
    
    Fill(h,"nRecTrkInSimVtx",(double) ev->tk.size(),isSignal);
    Fill(h,"nPrimRecTrkInSimVtx",(double) ev->tkprim.size(),isSignal);
    Fill(h,"sumpt2rec",sqrt(ev->sumpt2rec),isSignal);
    Fill(h,"sumpt2",sqrt(ev->sumpt2),isSignal);
    Fill(h,"sumpt",sqrt(ev->sumpt),isSignal);

    double nRecVWithTrk=0;  // vertices with tracks from this simvertex
    double  nmatch=0, ntmatch=0, zmatch=-99;

    for(reco::VertexCollection::const_iterator v=recVtxs->begin(); 
	v!=recVtxs->end(); ++v){
      if ( (v->ndof()<-1) || (v->chi2()<=0) ) continue;  // skip clusters 
      // count tracks found in both, sim and rec
      double n=0, wt=0;
      for(vector<TransientTrack>::iterator te=ev->tk.begin(); te!=ev->tk.end(); te++){
	const reco::Track&  RTe=te->track();
	for(trackit_t tv=v->tracks_begin(); tv!=v->tracks_end(); tv++){
	   const reco::Track & RTv=*(tv->get());  
	   if(RTe.vz()==RTv.vz()) {n++; wt+=v->trackWeight(*tv);}
	}
      }

      if(n>0){	nRecVWithTrk++; }
      if (n > nmatch){
	nmatch=n; ntmatch=v->tracksSize(); zmatch=v->position().z();
      }
      
    }// end of reco vertex loop


    // nmatch is the highest number of tracks from this sim vertex found in a single reco-vertex
    if(ev->tk.size()>0){ Fill(h,"trkAssignmentEfficiency", nmatch/ev->tk.size(), isSignal); };
    if(ev->tkprim.size()>0){ Fill(h,"primtrkAssignmentEfficiency", nmatch/ev->tkprim.size(), isSignal); };

    // matched efficiency = efficiency for finding a reco vertex with > 50% of the simvertexs reconstructed tracks

    double ntsim=ev->tk.size(); // may be better to use the number of primary tracks here ?
    double matchpurity=nmatch/ntmatch;

    if(ntsim>0){

      Fill(h,"matchVtxFraction",nmatch/ntsim,isSignal);
      if(nmatch/ntsim>=0.5){
	Fill(h,"matchVtxEfficiency",1.,isSignal);
	if(ntsim>1){Fill(h,"matchVtxEfficiency2",1.,isSignal);}
	if(matchpurity>0.5){Fill(h,"matchVtxEfficiency5",1.,isSignal);}
      }else{
	Fill(h,"matchVtxEfficiency",0.,isSignal);
	if(ntsim>1){Fill(h,"matchVtxEfficiency2",0.,isSignal);}
	Fill(h,"matchVtxEfficiency5",0.,isSignal);  // no (matchpurity>5) here !!
	if(isSignal){
	cout << "Signal vertex not matched " <<  message << "  event=" << eventcounter_ << " nmatch=" << nmatch << "  ntsim=" << ntsim << endl;
	}
      }
    } // ntsim >0


    if(zmatch>-99){
      Fill(h,"matchVtxZ",zmatch-ev->z);
      Fill(h,"matchVtxZ",zmatch-ev->z,isSignal);
      Fill(h,"matchVtxZCum",fabs(zmatch-ev->z));
      Fill(h,"matchVtxZCum",fabs(zmatch-ev->z),isSignal);
    }else{
      Fill(h,"matchVtxZCum",1.0);
      Fill(h,"matchVtxZCum",1.0,isSignal);
    }
    if(fabs(zmatch-ev->z)<zmatch_){
      Fill(h,"matchVtxEfficiencyZ",1.,isSignal);
    }else{
      Fill(h,"matchVtxEfficiencyZ",0.,isSignal);
    }	
    
    if(ntsim>0) Fill(h, "matchVtxEfficiencyZ1", fabs(zmatch-ev->z)<zmatch_ , isSignal);
    if(ntsim>1) Fill(h, "matchVtxEfficiencyZ2", fabs(zmatch-ev->z)<zmatch_ , isSignal);


    Fill(h,"vtxMultiplicity",nRecVWithTrk,isSignal);

    // efficiency vs number of tracks, use your favorite definition of efficiency here
    //if(nmatch>=0.5*ntmatch){  // purity
    if(fabs(zmatch-ev->z)<zmatch_){  //  zmatch
      Fill(h,"vtxFindingEfficiencyVsNtrk",(double) ev->tk.size(),1.);
       if(isSignal){
	 Fill(h,"vtxFindingEfficiencyVsNtrkSignal",ev->tk.size(),1.);
      }else{
	Fill(h,"vtxFindingEfficiencyVsNtrkPU",(double) ev->tk.size(),1.);
      }
    }else{
      Fill(h,"vtxFindingEfficiencyVsNtrk",(double) ev->tk.size(),0.);
      if(isSignal){
	Fill(h,"vtxFindingEfficiencyVsNtrkSignal",(double) ev->tk.size(),1.);
      }else{
	Fill(h,"vtxFindingEfficiencyVsNtrkPU",(double) ev->tk.size(),1.);
      }
    }

    
  }
  
  Fill(h,"npu1",npu1);
  Fill(h,"npu2",npu2);

  Fill(h,"nrecvsnpu",npu1,float(nrecvtxs));
  Fill(h,"nrecvsnpu2",npu2,float(nrecvtxs));

  // ---------------------------------------  sim-signal vs rec-tag  ---------------------------------------
  SimEvent* ev=&(simEvt[0]);
  const reco::Vertex* v=&(*recVtxs->begin());

  double n=0;
  for(vector<TransientTrack>::iterator te=ev->tk.begin(); te!=ev->tk.end(); te++){
    const reco::Track&  RTe=te->track();
    for(trackit_t tv=v->tracks_begin(); tv!=v->tracks_end(); tv++){
      const reco::Track & RTv=*(tv->get());  
      if(RTe.vz()==RTv.vz()) {n++;}
    }
  }
  
  cout << "Number of tracks in reco tagvtx " << v->tracksSize() << endl;
  cout << "Number of selected tracks in sim event vtx " << ev->tk.size() << "    (prim=" << ev->tkprim.size() << ")"<<endl;
  cout << "Number of tracks in both         " << n << endl;
  double ntruthmatched=getTruthMatchedVertexTracks(*v).size();
  if (ntruthmatched>0){
    cout << "TrackPurity = "<< n/ntruthmatched <<endl;
    Fill(h,"TagVtxTrkPurity",n/ntruthmatched);
  }
  if (ev->tk.size()>0){
    cout << "TrackEfficiency = "<< n/ev->tk.size() <<endl;
    Fill(h,"TagVtxTrkEfficiency",n/ev->tk.size());
  }
}

/***************************************************************************************/






/***************************************************************************************/

void PrimaryVertexAnalyzer4PU::analyzeVertexCollection(std::map<std::string, TH1*> & h,
						 const Handle<reco::VertexCollection> recVtxs,
						 const Handle<reco::TrackCollection> recTrks, 
						 std::vector<simPrimaryVertex> & simpv,
						       const std::string message)
{
  //cout <<"PrimaryVertexAnalyzer4PU::analyzeVertexCollection (HepMC), simpvs=" << simpv.size() << endl;
  int nrectrks=recTrks->size();
  int nrecvtxs=recVtxs->size();
  int nseltrks=-1; 
  reco::TrackCollection selTrks;   // selected tracks
  reco::TrackCollection lostTrks;  // selected but lost tracks (part of dropped clusters)

  // extract dummy vertices representing clusters
  reco::VertexCollection clusters;
  reco::Vertex allSelected;
  double cpufit=0;
  double cpuclu=0;
  for(reco::VertexCollection::const_iterator v=recVtxs->begin(); 
      v!=recVtxs->end(); ++v){
    if ( (fabs(v->ndof()+3.)<0.0001) && (v->chi2()<=0) ){ 
      // this dummy vertex is for the full event
      allSelected=(*v);
      nseltrks=(allSelected.tracksSize());
      nrecvtxs--;
      cpuclu=-v->chi2();
      continue;
    }else if( (fabs(v->ndof()+2.)<0.0001) && (v->chi2()==0) ){
      // this is a cluster, not a vertex
      clusters.push_back(*v);
      Fill(h,"cpuvsntrk",(double) v->tracksSize(),fabs(v->y()));
      cpufit+=fabs(v->y());
      Fill(h,"nclutrkall",(double) v->tracksSize());
      Fill(h,"selstat",v->x());
      //Fill(h,"nclutrkvtx",);// see below
      nrecvtxs--;
    }
  }
  Fill(h,"cpuclu",cpuclu);
  Fill(h,"cpufit",cpufit);
  Fill(h,"cpucluvsntrk",nrectrks, cpuclu);
  

  
  if(simpv.size()>0){//this is mc
    double dsimrecx=0.;
    double dsimrecy=0.;//0.0011;
    double dsimrecz=0.;//0.0012;
    
    // vertex matching and efficiency bookkeeping
    int nsimtrk=0;
    for(std::vector<simPrimaryVertex>::iterator vsim=simpv.begin();
	vsim!=simpv.end(); vsim++){
      
      
      nsimtrk+=vsim->nGenTrk;
      // look for a matching reconstructed vertex
      vsim->recVtx=NULL;
      vsim->cluster=-1;
      
      for(reco::VertexCollection::const_iterator vrec=recVtxs->begin();  vrec!=recVtxs->end(); ++vrec){

	if( vrec->isFake() ) {
	  continue;  // skip fake vertices (=beamspot)
	  cout << "fake vertex" << endl;
	}

	if( vrec->ndof()<0. )continue;  // skip dummy clusters, if any
	//        if ( matchVertex(*vsim,*vrec) ){

	// if the matching critera are fulfilled, accept the rec-vertex that is closest in z
	if(    ((vsim->recVtx) && (fabs(vsim->recVtx->position().z()-vsim->z-dsimrecz)>fabs(vrec->z()-vsim->z-dsimrecz)))
	       || (!vsim->recVtx) )
	  {
	    vsim->recVtx=&(*vrec);

	    // find the corresponding cluster
	    for(unsigned int iclu=0; iclu<clusters.size(); iclu++){
	      if( fabs(clusters[iclu].position().z()-vrec->position().z()) < 0.001 ){
		vsim->cluster=iclu;
		vsim->nclutrk=clusters[iclu].position().y();
	      }
	    }
	  }

	// the following only works in MC samples without pile-up
	if ((simpv.size()==1) && ( fabs(vsim->recVtx->position().z()-vsim->z-dsimrecz)>zmatch_ )){
	    // now we have a recvertex without a matching simvertex, I would call this fake 
	    // however, the G4 info does not contain pile-up
	    Fill(h,"fakeVtxZ",vrec->z());
	    if (vrec->ndof()>=0.5) Fill(h,"fakeVtxZNdofgt05",vrec->z());
	    if (vrec->ndof()>=2.0) Fill(h,"fakeVtxZNdofgt2",vrec->z());
	    Fill(h,"fakeVtxNdof",vrec->ndof());
	    //Fill(h,"fakeVtxNdof1",vrec->ndof());
	    Fill(h,"fakeVtxNtrk",vrec->tracksSize());
	    if(vrec->tracksSize()==2){  Fill(h,"fake2trkchi2vsndof",vrec->ndof(),vrec->chi2());   }
	    if(vrec->tracksSize()==3){  Fill(h,"fake3trkchi2vsndof",vrec->ndof(),vrec->chi2());   }
	    if(vrec->tracksSize()==4){  Fill(h,"fake4trkchi2vsndof",vrec->ndof(),vrec->chi2());   }
	    if(vrec->tracksSize()==5){  Fill(h,"fake5trkchi2vsndof",vrec->ndof(),vrec->chi2());   }
	  }
      }
      

      Fill(h,"nsimtrk",float(nsimtrk));
      Fill(h,"nsimtrk",float(nsimtrk),vsim==simpv.begin());
      Fill(h,"nrecsimtrk",float(vsim->nMatchedTracks));
      Fill(h,"nrecnosimtrk",float(nsimtrk-vsim->nMatchedTracks));
      
      // histogram properties of matched vertices
      if (vsim->recVtx && ( fabs(vsim->recVtx->z()-vsim->z*simUnit_)<zmatch_ )){
	
	if(verbose_){std::cout <<"primary matched " << message << " " << setw(8) << setprecision(4) << vsim->x << " " << vsim->y << " " << vsim->z << std:: endl;}
	Fill(h,"matchedVtxNdof", vsim->recVtx->ndof());
	// residuals an pulls with respect to simulated vertex
	Fill(h,"resx", vsim->recVtx->x()-vsim->x*simUnit_ );
	Fill(h,"resy", vsim->recVtx->y()-vsim->y*simUnit_ );
	Fill(h,"resz", vsim->recVtx->z()-vsim->z*simUnit_ );
	Fill(h,"resz10", vsim->recVtx->z()-vsim->z*simUnit_ );
	Fill(h,"pullx", (vsim->recVtx->x()-vsim->x*simUnit_)/vsim->recVtx->xError() );
	Fill(h,"pully", (vsim->recVtx->y()-vsim->y*simUnit_)/vsim->recVtx->yError() );
	Fill(h,"pullz", (vsim->recVtx->z()-vsim->z*simUnit_)/vsim->recVtx->zError() );
	Fill(h,"resxr", vsim->recVtx->x()-vsim->x*simUnit_-dsimrecx);
	Fill(h,"resyr", vsim->recVtx->y()-vsim->y*simUnit_-dsimrecy );
	Fill(h,"reszr", vsim->recVtx->z()-vsim->z*simUnit_-dsimrecz);
	Fill(h,"pullxr", (vsim->recVtx->x()-vsim->x*simUnit_-dsimrecx)/vsim->recVtx->xError() );
	Fill(h,"pullyr", (vsim->recVtx->y()-vsim->y*simUnit_-dsimrecy)/vsim->recVtx->yError() );
	Fill(h,"pullzr", (vsim->recVtx->z()-vsim->z*simUnit_-dsimrecz)/vsim->recVtx->zError() );



	// efficiency with zmatch within 500 um (or whatever zmatch is)
	Fill(h,"eff", 1.);
	if(simpv.size()==1){
	  if (vsim->recVtx==&(*recVtxs->begin())){
	    Fill(h,"efftag", 1.); 
	  }else{
	    Fill(h,"efftag", 0.); 
	    cout << "signal vertex not tagged " << message << " " << eventcounter_ << endl;
	    // call XXClusterizerInZ.vertices(seltrks,3)
	    
	  }
	}
	
	Fill(h,"effvsptsq",vsim->ptsq,1.);
	Fill(h,"effvsnsimtrk",vsim->nGenTrk,1.);
	Fill(h,"effvsnrectrk",nrectrks,1.);
	Fill(h,"effvsnseltrk",nseltrks,1.);
	Fill(h,"effvsz",vsim->z*simUnit_,1.);
	Fill(h,"effvsz2",vsim->z*simUnit_,1.);
	Fill(h,"effvsr",sqrt(vsim->x*vsim->x+vsim->y*vsim->y)*simUnit_,1.);
	

      }else{  // no matching rec vertex found for this simvertex
	
	bool plapper=verbose_ && vsim->nGenTrk;
	if(plapper){
	  // be quiet about invisble vertices
	  std::cout << "primary not found "  << message << " " << eventcounter_ << "  x=" <<vsim->x*simUnit_ << "  y=" << vsim->y*simUnit_  << " z=" << vsim->z*simUnit_  << " nGenTrk=" << vsim->nGenTrk << std::endl;
	}
	int mistype=0;
	if (vsim->recVtx){
	  if(plapper){
	    std::cout << "nearest recvertex at " << vsim->recVtx->z() << "   dz=" << vsim->recVtx->z()-vsim->z*simUnit_ << std::endl;
	  }
	  
	  if (fabs(vsim->recVtx->z()-vsim->z*simUnit_)<0.2 ){
	    Fill(h,"effvsz2",vsim->z*simUnit_,1.);
	  }
	  
	  if (fabs(vsim->recVtx->z()-vsim->z*simUnit_)<0.5 ){
	    if(verbose_){std::cout << "type 1, lousy z vertex" << std::endl;}
	    Fill(h,"zlost1", vsim->z*simUnit_,1.);
	    mistype=1;
	  }else{
	    if(plapper){std::cout << "type 2a no vertex anywhere near" << std::endl;}
	    mistype=2;
	  }
	}else{// no recVtx at all
	  mistype=2;
	  if(plapper){std::cout << "type 2b, no vertex at all" << std::endl;}
	}
	
	if(mistype==2){
	  int selstat=-3;
	  // no matching vertex found, is there a cluster?
	  for(unsigned int iclu=0; iclu<clusters.size(); iclu++){
	    if( fabs(clusters[iclu].position().z()-vsim->z*simUnit_) < 0.1 ){
	      selstat=int(clusters[iclu].position().x()+0.1);
	      if(verbose_){std::cout << "matching cluster found with selstat=" << clusters[iclu].position().x() << std::endl;}
	    }
	  }
	  if (selstat==0){
	    if(plapper){std::cout << "vertex rejected (distance to beam)" << std::endl;}
	    Fill(h,"zlost3", vsim->z*simUnit_,1.);
	  }else if(selstat==-1){
	    if(plapper) {std::cout << "vertex invalid" << std::endl;}
	    Fill(h,"zlost4", vsim->z*simUnit_,1.);
	  }else if(selstat==1){
	    if(plapper){std::cout << "vertex accepted, this cannot be right!!!!!!!!!!" << std::endl;}
	  }else if(selstat==-2){
	    if(plapper){std::cout << "dont know what this means !!!!!!!!!!" << std::endl;}
	  }else if(selstat==-3){
	    if(plapper){std::cout << "no matching cluster found " << std::endl;}
	    Fill(h,"zlost2", vsim->z*simUnit_,1.);
	  }else{
	    if(plapper){std::cout << "dont know what this means either !!!!!!!!!!" << selstat << std::endl;}
	  }
	}//
	
	
	Fill(h,"eff", 0.);
	if(simpv.size()==1){ Fill(h,"efftag", 0.); }
	
	Fill(h,"effvsptsq",vsim->ptsq,0.);
	Fill(h,"effvsnsimtrk",float(vsim->nGenTrk),0.);
	Fill(h,"effvsnrectrk",nrectrks,0.);
	Fill(h,"effvsnseltrk",nseltrks,0.);
	Fill(h,"effvsz",vsim->z*simUnit_,0.);
	Fill(h,"effvsr",sqrt(vsim->x*vsim->x+vsim->y*vsim->y)*simUnit_,0.);
	
      } // no recvertex for this simvertex

    }

    // end of sim/rec matching
   
     
   // purity of event vertex tags
    if (recVtxs->size()>0){
      Double_t dz=(*recVtxs->begin()).z() - (*simpv.begin()).z*simUnit_;
      Fill(h,"zdistancetag",dz);
      Fill(h,"abszdistancetag",fabs(dz));
      if( fabs(dz)<zmatch_){
	Fill(h,"puritytag",1.);
      }else{
	// bad tag: the true primary was more than 500 um (or zmatch) away from the tagged primary
	Fill(h,"puritytag",0.);
      }
    }
    
  }else{
    //if(verbose_) cout << "PrimaryVertexAnalyzer4PU::analyzeVertexCollection:  simPV is empty!" << endl;
  }


  //******* the following code does not require MC and will/should work for data **********


  Fill(h,"bunchCrossing",bunchCrossing_);
  if(recTrks->size()>0)  Fill(h,"bunchCrossingLogNtk",bunchCrossing_,log(recTrks->size())/log(10.));
  
  // -----------------  reconstructed tracks  ------------------------
  // the list of selected tracks can only be correct if the selection parameterset  and track collection
  // is the same that was used for the reconstruction

  int nt=0;
  for(reco::TrackCollection::const_iterator t=recTrks->begin();
      t!=recTrks->end(); ++t){
    if((recVtxs->size()>0) && (recVtxs->begin()->isValid())){
      fillTrackHistos(h,"all",*t,&(*recVtxs->begin()));
    }else{
      fillTrackHistos(h,"all",*t);
    }
    if(recTrks->size()>100)    fillTrackHistos(h,"M",*t);



    TransientTrack  tt = theB_->build(&(*t));  tt.setBeamSpot(vertexBeamSpot_);
    if  (theTrackFilter(tt)){
      selTrks.push_back(*t);
      fillTrackHistos(h,"sel",*t);
      int foundinvtx=0;
      int nvtemp=-1;
      for(reco::VertexCollection::const_iterator v=recVtxs->begin(); 
	  v!=recVtxs->end(); ++v){
	nvtemp++;
	if(( v->isFake()) || (v->ndof()<-2) ) break;
	for(trackit_t tv=v->tracks_begin(); tv!=v->tracks_end(); tv++ ){
	  if( ((**tv).vz()==t->vz()&&((**tv).phi()==t->phi())) ) {
	    foundinvtx++;
	  }
	}
	
      }
      if(foundinvtx==0){
	fillTrackHistos(h,"sellost",*t);
      }else if(foundinvtx>1){
	cout << "hmmmm " << foundinvtx << endl;
      }
    }
    nt++;
  }


  if (nseltrks<0){
    nseltrks=selTrks.size();
  }else if( ! (nseltrks==(int)selTrks.size()) ){
    std::cout << "Warning: inconsistent track selection !" << std::endl;
  }



  // fill track histograms of vertex tracks
  int nrec=0,  nrec0=0, nrec8=0, nrec2=0, nrec4=0;
  for(reco::VertexCollection::const_iterator v=recVtxs->begin(); 
      v!=recVtxs->end(); ++v){
    
    if (! (v->isFake()) && v->ndof()>0 && v->chi2()>0 ){
      nrec++;
      if (v->ndof()>0) nrec0++;
      if (v->ndof()>8) nrec8++;
      if (v->ndof()>2) nrec2++;
      if (v->ndof()>4) nrec4++;
      for(trackit_t t=v->tracks_begin(); t!=v->tracks_end(); t++){
	if(v==recVtxs->begin()){
	  fillTrackHistos(h,"tagged",**t,  &(*v));
	}else{
	  fillTrackHistos(h,"untagged",**t,  &(*v));
	}

	Float_t wt=v->trackWeight(*t);
	//dumpHitInfo(**t); cout << "  w=" << wt << endl;
	Fill(h,"trackWt",wt);
	if(wt>0.5){
	  fillTrackHistos(h,"wgt05",**t, &(*v));
	  if(v->ndof()>4) fillTrackHistos(h,"ndof4",**t, &(*v));
	}else{
	  fillTrackHistos(h,"wlt05",**t, &(*v));
	}
      }
    }
  }


  // bachelor tracks (only available through clusters right now)
  for(unsigned int iclu=0; iclu<clusters.size(); iclu++){
    if (clusters[iclu].tracksSize()==1){
      for(trackit_t t = clusters[iclu].tracks_begin(); 
	  t!=clusters[iclu].tracks_end(); t++){
	fillTrackHistos(h,"bachelor",**t);
      }
    }
  }


  // -----------------  reconstructed vertices  ------------------------

  // event 
  Fill(h,"szRecVtx",recVtxs->size());
  Fill(h,"nclu",clusters.size());
  Fill(h,"nseltrk",nseltrks);
  Fill(h,"nrectrk",nrectrks);
  Fill(h,"nrecvtx",nrec);
  Fill(h,"nrecvtx2",nrec2);
  Fill(h,"nrecvtx4",nrec4);
  Fill(h,"nrecvtx8",nrec8);

  if(nrec>0){
    Fill(h,"eff0vsntrec",nrectrks,1.);
    Fill(h,"eff0vsntsel",nseltrks,1.);
  }else{
    Fill(h,"eff0vsntrec",nrectrks,0.);
    Fill(h,"eff0vsntsel",nseltrks,0.);
    if((nseltrks>1)&&(verbose_)){
      cout << Form("PrimaryVertexAnalyzer4PU: %s may have lost a vertex  %10d  %10d     %4d / %4d ",message.c_str(),run_, event_, nrectrks,nseltrks) << endl;
      dumpThisEvent_=true;
    }
  }
  if(nrec0>0) { Fill(h,"eff0ndof0vsntsel",nseltrks,1.);}else{ Fill(h,"eff0ndof0vsntsel",nseltrks,0.);}
  if(nrec2>0) { Fill(h,"eff0ndof2vsntsel",nseltrks,1.);}else{ Fill(h,"eff0ndof2vsntsel",nseltrks,0.);}
  if(nrec4>0) { Fill(h,"eff0ndof4vsntsel",nseltrks,1.);}else{ Fill(h,"eff0ndof4vsntsel",nseltrks,0.);}
  if(nrec8>0) { Fill(h,"eff0ndof8vsntsel",nseltrks,1.);}else{ Fill(h,"eff0ndof8vsntsel",nseltrks,0.);}

  if((nrec>1)&&(DEBUG_)) {
    cout << "multivertex event" << endl;
    dumpThisEvent_=true;
  }

  if((nrectrks>10)&&(nseltrks<3)){
    cout << "small fraction of selected tracks "  << endl;
    dumpThisEvent_=true;
  }

  // properties of events without a vertex
  if((nrec==0)||(recVtxs->begin()->isFake())){
    Fill(h,"nrectrk0vtx",nrectrks);
    Fill(h,"nseltrk0vtx",nseltrks);
    Fill(h,"nclu0vtx",clusters.size());
  }


  //  properties of (valid) vertices
  double ndof2=-10,ndof1=-10, zndof1=0, zndof2=0;
  for(reco::VertexCollection::const_iterator v=recVtxs->begin(); 
      v!=recVtxs->end(); ++v){
    if(v->isFake()){ Fill(h,"isFake",1.);}else{ Fill(h,"isFake",0.);}
    if(v->isFake()||((v->ndof()<0)&&(v->ndof()>-3))){ Fill(h,"isFake1",1.);}else{ Fill(h,"isFake1",0.);}

    if((v->isFake())||(v->ndof()<0)) continue;
    if(v->ndof()>ndof1){ ndof2=ndof1; zndof2=zndof1; ndof1=v->ndof(); zndof1=v->position().z();}
    else if(v->ndof()>ndof2){ ndof2=v->ndof(); zndof2=v->position().z();}


    // some special histogram for two track vertices
    if(v->tracksSize()==2){
      const TrackBaseRef& t1= *(v->tracks_begin());
      const TrackBaseRef& t2=*(v->tracks_begin()+1);
      bool os=(t1->charge()*t2->charge()<0);
      double dphi=t1->phi()-t2->phi(); if (dphi<0) dphi+=2*M_PI;
      double m12=sqrt(pow( sqrt(pow(0.139,2)+pow( t1->p(),2)) +sqrt(pow(0.139,2)+pow( t2->p(),2)) ,2)
			     -pow(t1->px()+t2->px(),2)
			     -pow(t1->py()+t2->py(),2)
			     -pow(t1->pz()+t2->pz(),2)
		    );
      if(os){
	Fill(h,"2trkdetaOS",t1->eta()-t2->eta());
	Fill(h,"2trkmassOS",m12);
      }else{
      	Fill(h,"2trkdetaSS",t1->eta()-t2->eta());
	Fill(h,"2trkmassSS",m12);
      }
      Fill(h,"2trkdphi",dphi);
      Fill(h,"2trkseta",t1->eta()+t2->eta());
      if(fabs(dphi-M_PI)<0.1)      Fill(h,"2trksetacurl",t1->eta()+t2->eta());
      if(fabs(t1->eta()+t2->eta())<0.1) Fill(h,"2trkdphicurl",dphi);
      // fill separately for extra vertices
      if(v!=recVtxs->begin()){
	if(os){
	  Fill(h,"2trkdetaOSPU",t1->eta()-t2->eta());
	  Fill(h,"2trkmassOSPU",m12);
	}else{
	  Fill(h,"2trkdetaSSPU",t1->eta()-t2->eta());
	  Fill(h,"2trkmassSSPU",m12);
	}
	Fill(h,"2trkdphiPU",dphi);
	Fill(h,"2trksetaPU",t1->eta()+t2->eta());
	if(fabs(dphi-M_PI)<0.1)      Fill(h,"2trksetacurlPU",t1->eta()+t2->eta());
	if(fabs(t1->eta()+t2->eta())<0.1) Fill(h,"2trkdphicurlPU",dphi);
      }
    }


    Fill(h,"trkchi2vsndof",v->ndof(),v->chi2());
    if(v->ndof()>0){    Fill(h,"trkchi2overndof",v->chi2()/v->ndof()); }
    if(v->tracksSize()==2){  Fill(h,"2trkchi2vsndof",v->ndof(),v->chi2());   }
    if(v->tracksSize()==3){  Fill(h,"3trkchi2vsndof",v->ndof(),v->chi2());   }
    if(v->tracksSize()==4){  Fill(h,"4trkchi2vsndof",v->ndof(),v->chi2());   }
    if(v->tracksSize()==5){  Fill(h,"5trkchi2vsndof",v->ndof(),v->chi2());   }

    Fill(h,"nbtksinvtx",v->tracksSize());
    Fill(h,"nbtksinvtx2",v->tracksSize());
    Fill(h,"vtxchi2",v->chi2());
    Fill(h,"vtxndf",v->ndof());
    Fill(h,"vtxprob",ChiSquaredProbability(v->chi2() ,v->ndof()));
    Fill(h,"vtxndfvsntk",v->tracksSize(), v->ndof());
    Fill(h,"vtxndfoverntk",v->ndof()/v->tracksSize());
    Fill(h,"vtxndf2overntk",(v->ndof()+2)/v->tracksSize());
    Fill(h,"zrecvsnt",v->position().z(),float(nrectrks));
    if(nrectrks>100){
      Fill(h,"zrecNt100",v->position().z());
    }

    if(v->ndof()>2.0){  // enter only vertices that really contain tracks
      Fill(h,"xrec",v->position().x());
      Fill(h,"yrec",v->position().y());
      Fill(h,"zrec",v->position().z());
      Fill(h,"xrec1",v->position().x());
      Fill(h,"yrec1",v->position().y());
      Fill(h,"zrec1",v->position().z());
      Fill(h,"xrec2",v->position().x());
      Fill(h,"yrec2",v->position().y());
      Fill(h,"zrec2",v->position().z());
      Fill(h,"xrec3",v->position().x());
      Fill(h,"yrec3",v->position().y());
      Fill(h,"zrec3",v->position().z());
      Fill(h,"xrecb",v->position().x()-vertexBeamSpot_.x0());
      Fill(h,"yrecb",v->position().y()-vertexBeamSpot_.y0());
      Fill(h,"zrecb",v->position().z()-vertexBeamSpot_.z0());
      Fill(h,"xrecBeam",v->position().x()-vertexBeamSpot_.x0());
      Fill(h,"yrecBeam",v->position().y()-vertexBeamSpot_.y0());
      Fill(h,"zrecBeam",v->position().z()-vertexBeamSpot_.z0());
      Fill(h,"xrecBeamPull",(v->position().x()-vertexBeamSpot_.x0())/sqrt(pow(v->xError(),2)+pow(vertexBeamSpot_.BeamWidthX(),2)));
      Fill(h,"yrecBeamPull",(v->position().y()-vertexBeamSpot_.y0())/sqrt(pow(v->yError(),2)+pow(vertexBeamSpot_.BeamWidthY(),2)));
      
      Fill(h,"xrecBeamvsdx",v->xError(),v->position().x()-vertexBeamSpot_.x0());
      Fill(h,"yrecBeamvsdy",v->yError(),v->position().y()-vertexBeamSpot_.y0());
      Fill(h,"xrecBeamvsdxR2",v->position().x()-vertexBeamSpot_.x0(),v->xError());
      Fill(h,"yrecBeamvsdyR2",v->position().y()-vertexBeamSpot_.y0(),v->yError());
      Fill(h,"xrecBeam2vsdx2prof",pow(v->xError(),2),pow(v->position().x()-vertexBeamSpot_.x0(),2));
      Fill(h,"yrecBeam2vsdy2prof",pow(v->yError(),2),pow(v->position().y()-vertexBeamSpot_.y0(),2));
      Fill(h,"xrecBeamvsdx2",pow(v->xError(),2),pow(v->position().x()-vertexBeamSpot_.x0(),2));
      Fill(h,"yrecBeamvsdy2",pow(v->yError(),2),pow(v->position().y()-vertexBeamSpot_.y0(),2));
      Fill(h,"xrecBeamvsz",v->position().z(),v->position().x()-vertexBeamSpot_.x0());
      Fill(h,"yrecBeamvsz",v->position().z(),v->position().y()-vertexBeamSpot_.y0());
      Fill(h,"xrecBeamvszprof",v->position().z(),v->position().x()-vertexBeamSpot_.x0());
      Fill(h,"yrecBeamvszprof",v->position().z(),v->position().y()-vertexBeamSpot_.y0());
      Fill(h,"xrecBeamvsdxprof",v->xError(),v->position().x()-vertexBeamSpot_.x0());
      Fill(h,"yrecBeamvsdyprof",v->yError(),v->position().y()-vertexBeamSpot_.y0());
      
      
      Fill(h,"errx",v->xError());
      Fill(h,"erry",v->yError());
      Fill(h,"errz",v->zError());
      double vxx=v->covariance(0,0);
      double vyy=v->covariance(1,1);
      double vxy=v->covariance(1,0);
      double dv=0.25*(vxx+vyy)*(vxx+vyy-(vxx*vyy-vxy*vxy));
      if(dv>0){
	double l1=0.5*(vxx+vyy)+sqrt(dv);
	Fill(h,"err1",sqrt(l1));
	double l2=sqrt(0.5*(vxx+vyy)-sqrt(dv));
	if(l2>0) Fill(h,"err2",sqrt(l2));
      }
      
      
      // look at the tagged vertex separately
      if (v==recVtxs->begin()){
	Fill(h,"nbtksinvtxTag",v->tracksSize());
	Fill(h,"nbtksinvtxTag2",v->tracksSize());
	Fill(h,"xrectag",v->position().x());
	Fill(h,"yrectag",v->position().y());
	Fill(h,"zrectag",v->position().z());
      }else{
	Fill(h,"nbtksinvtxPU",v->tracksSize());
	Fill(h,"nbtksinvtxPU2",v->tracksSize());
      }
	
      // vertex resolution vs number of tracks
      Fill(h,"xresvsntrk",v->tracksSize(),v->xError());
      Fill(h,"yresvsntrk",v->tracksSize(),v->yError());
      Fill(h,"zresvsntrk",v->tracksSize(),v->zError());
      
    }
    
    // cluster properties (if available)
    for(unsigned int iclu=0; iclu<clusters.size(); iclu++){
      if( fabs(clusters[iclu].position().z()-v->position().z()) < 0.0001 ){
	Fill(h,"nclutrkvtx",clusters[iclu].tracksSize());
      }
    }
    

        
    //  properties of (valid) neighbour vertices
    reco::VertexCollection::const_iterator v1=v;     v1++;
    for(; v1!=recVtxs->end(); ++v1){
      if((v1->isFake())||(v1->ndof()<0)) continue;
      Fill(h,"zdiffrec",v->position().z()-v1->position().z());
//       if(fabs(v->position().z()-v1->position().z())>1){
// 	cout << message << " zdiffrec=" << v->position().z()-v1->position().z() << " " << v->ndof() << " " << v1->ndof() << endl;
// 	dumpThisEvent_=true;
//       }

      double z0=v->position().z()-vertexBeamSpot_.z0();
      double z1=v1->position().z()-vertexBeamSpot_.z0();
      Fill(h,"zPUcand",z0);	Fill(h,"zPUcand",z1);
      Fill(h,"ndofPUcand",v->ndof()); Fill(h,"ndofPUcand",v1->ndof());
      
      Fill(h,"zdiffvsz",z1-z0,0.5*(z1+z0));

      if ((v->ndof()>2) && (v1->ndof()>2)){
	Fill(h,"zdiffrec2",v->position().z()-v1->position().z());
	Fill(h,"zPUcand2",z0);
	Fill(h,"zPUcand2",z1);
	Fill(h,"ndofPUcand2",v->ndof());
	Fill(h,"ndofPUcand2",v1->ndof());
	Fill(h,"zvszrec2",z0, z1);
	Fill(h,"pzvspz2",TMath::Freq(z0/2.16),TMath::Freq(z1/2.16) );
      }
      
      if ((v->ndof()>4) && (v1->ndof()>4)){
	Fill(h,"zdiffvsz4",z1-z0,0.5*(z1+z0));
	Fill(h,"zdiffrec4",v->position().z()-v1->position().z());
	Fill(h,"zvszrec4",z0, z1);
	Fill(h,"pzvspz4",TMath::Freq(z0/2.16),TMath::Freq(z1/2.16) );
	//cout << "ndof4 pu-candidate " << run_ << " " << event_ << endl ;
	if(fabs(z0-z1)>1.0){
	  Fill(h,"xbeamPUcand",v->position().x()-vertexBeamSpot_.x0());
	  Fill(h,"ybeamPUcand",v->position().y()-vertexBeamSpot_.y0());
	  Fill(h,"xbeamPullPUcand",(v->position().x()-vertexBeamSpot_.x0())/v->xError());
	  Fill(h,"ybeamPullPUcand",(v->position().y()-vertexBeamSpot_.y0())/v->yError());
	  //Fill(h,"sumwOverNtkPUcand",sumw/v->tracksSize());
	  //Fill(h,"sumwOverNtkPUcand",sumw/v1->tracksSize());
	  Fill(h,"ndofOverNtkPUcand",v->ndof()/v->tracksSize());
	  Fill(h,"ndofOverNtkPUcand",v1->ndof()/v1->tracksSize());
	  Fill(h,"xbeamPUcand",v1->position().x()-vertexBeamSpot_.x0());
	  Fill(h,"ybeamPUcand",v1->position().y()-vertexBeamSpot_.y0());
	  Fill(h,"xbeamPullPUcand",(v1->position().x()-vertexBeamSpot_.x0())/v1->xError());
	  Fill(h,"ybeamPullPUcand",(v1->position().y()-vertexBeamSpot_.y0())/v1->yError());
	  Fill(h,"zPUcand4",z0);
	  Fill(h,"zPUcand4",z1);
	  Fill(h,"ndofPUcand4",v->ndof());
	  Fill(h,"ndofPUcand4",v1->ndof());
	  for(trackit_t t=v->tracks_begin(); t!=v->tracks_end(); t++){ if(v->trackWeight(*t)>0.5) fillTrackHistos(h,"PUcand",**t, &(*v));  }
	  for(trackit_t t=v1->tracks_begin(); t!=v1->tracks_end(); t++){ if(v1->trackWeight(*t)>0.5) fillTrackHistos(h,"PUcand",**t, &(*v1));   }
	}
	}
      
      if ((v->ndof()>4) && (v1->ndof()>2) && (v1->ndof()<4)){
	for(trackit_t t=v1->tracks_begin(); t!=v1->tracks_end(); t++){ if(v1->trackWeight(*t)>0.5) fillTrackHistos(h,"PUfake",**t, &(*v1));   }
      }
      
      if ((v->ndof()>8) && (v1->ndof()>8)){
	Fill(h,"zdiffrec8",v->position().z()-v1->position().z());
	if(dumpPUcandidates_ && fabs(z0-z1)>1.0){
	  cout << "PU candidate " << run_ << " " << event_ << " " << message <<  " zdiff=" <<z0-z1 << endl;
// 	cerr << "PU candidate " << run_ << " "<<  event_ << " " << message <<  endl;
	  dumpThisEvent_=true;
	}
      }

    }
    
    // test track links, use reconstructed vertices
      bool problem = false;
      Fill(h,"nans",1.,std::isnan(v->position().x())*1.);
      Fill(h,"nans",2.,std::isnan(v->position().y())*1.);
      Fill(h,"nans",3.,std::isnan(v->position().z())*1.);
      
      int index = 3;
      for (int i = 0; i != 3; i++) {
	for (int j = i; j != 3; j++) {
	  index++;
	  Fill(h,"nans",index*1., std::isnan(v->covariance(i, j))*1.);
	  if (std::isnan(v->covariance(i, j))) problem = true;
	  // in addition, diagonal element must be positive
	  if (j == i && v->covariance(i, j) < 0) {
	    Fill(h,"nans",index*1., 1.);
	    problem = true;
	  }
	}
      }
      
      try {
	for(trackit_t t = v->tracks_begin(); 
	    t!=v->tracks_end(); t++) {
	  // illegal charge
	  if ( (**t).charge() < -1 || (**t).charge() > 1 ) {
	    Fill(h,"tklinks",0.);
	  }
	  else {
	    Fill(h,"tklinks",1.);
	  }
	}
      } catch (...) {
	// exception thrown when trying to use linked track
	Fill(h,"tklinks",0.);
      }

      if (problem) {
	// analyze track parameter covariance definiteness
	double data[25];
	try {
	  int itk = 0;
	  for(trackit_t t = v->tracks_begin(); 
	      t!=v->tracks_end(); t++) {
	    std::cout << "Track " << itk++ << std::endl;
	    int i2 = 0;
	    for (int i = 0; i != 5; i++) {
	      for (int j = 0; j != 5; j++) {
		data[i2] = (**t).covariance(i, j);
		std::cout << std:: scientific << data[i2] << " ";
		i2++;
	      }
	      std::cout << std::endl;
	    }
	    gsl_matrix_view m 
	      = gsl_matrix_view_array (data, 5, 5);
	    
	    gsl_vector *eval = gsl_vector_alloc (5);
	    gsl_matrix *evec = gsl_matrix_alloc (5, 5);
	    
	    gsl_eigen_symmv_workspace * w = 
	      gsl_eigen_symmv_alloc (5);
	    
	    gsl_eigen_symmv (&m.matrix, eval, evec, w);
	    
	    gsl_eigen_symmv_free (w);
	    
	    gsl_eigen_symmv_sort (eval, evec, 
				  GSL_EIGEN_SORT_ABS_ASC);
	    
	    // print sorted eigenvalues
	  {
	    int i;
	    for (i = 0; i < 5; i++) {
	      double eval_i 
		= gsl_vector_get (eval, i);
	      //gsl_vector_view evec_i 
	      //	= gsl_matrix_column (evec, i);
	      
	      printf ("eigenvalue = %g\n", eval_i);
	      //	      printf ("eigenvector = \n");
	      //	      gsl_vector_fprintf (stdout, 
	      //				  &evec_i.vector, "%g");
	    }
	  }
	  }
	}
      catch (...) {
	// exception thrown when trying to use linked track
	break;
      }// catch()
      }// if (problem)


    
  }  // vertex loop (v)


  // 2nd highest ndof
  if (ndof2>0){
      Fill(h,"ndofnr2",ndof2); 
      if(fabs(zndof1-zndof2)>1) Fill(h,"ndofnr2d1cm",ndof2); 
      if(fabs(zndof1-zndof2)>2) Fill(h,"ndofnr2d2cm",ndof2); 
  }


}

