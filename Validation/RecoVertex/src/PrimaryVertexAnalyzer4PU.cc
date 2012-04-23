#include "Validation/RecoVertex/interface/PrimaryVertexAnalyzer4PU.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "MagneticField/Engine/interface/MagneticField.h"

// reco track and vertex 
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/Math/interface/Point3D.h"
#include "RecoVertex/VertexPrimitives/interface/TransientVertex.h"
// associator stuff
#include "DataFormats/Math/interface/deltaPhi.h"

// simulated vertices,..., add <use name=SimDataFormats/Vertex> and <../Track>
#include <SimDataFormats/Vertex/interface/SimVertex.h>
#include <SimDataFormats/Vertex/interface/SimVertexContainer.h>
#include <SimDataFormats/Track/interface/SimTrack.h>
#include <SimDataFormats/Track/interface/SimTrackContainer.h>

#include "SimDataFormats/CrossingFrame/interface/CrossingFrame.h"
#include "SimDataFormats/CrossingFrame/interface/CrossingFramePlaybackInfo.h"
#include "SimDataFormats/CrossingFrame/interface/MixCollection.h"
#include "SimDataFormats/TrackingHit/interface/PSimHit.h"

#include "SimDataFormats/PileupSummaryInfo/interface/PileupSummaryInfo.h"
// see https://twiki.cern.ch/twiki/bin/view/CMS/PileupInformation

// Lumi
#include "DataFormats/Luminosity/interface/LumiDetails.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"


// AOD et al
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"

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
#include "TrackingTools/TrajectoryParametrization/interface/TrajectoryStateExceptions.h"

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
#include "DataFormats/SiPixelCluster/interface/SiPixelCluster.h"

#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/DetId/interface/DetId.h"

#include "DataFormats/SiPixelDetId/interface/PXBDetId.h"
#include "DataFormats/SiPixelDetId/interface/PXFDetId.h"
#include "DataFormats/SiPixelDetId/interface/PixelSubdetector.h"
#include "DataFormats/SiPixelDetId/interface/PixelBarrelName.h"


#include <assert.h>

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
PrimaryVertexAnalyzer4PU::PrimaryVertexAnalyzer4PU(const ParameterSet& iConfig):theTrackFilter(iConfig.getParameter<edm::ParameterSet>("TkFilterParameters"))
{
   //now do what ever initialization is needed
  simG4_=iConfig.getParameter<edm::InputTag>( "simG4" );
  recoTrackProducer_= iConfig.getUntrackedParameter<std::string>("recoTrackProducer");
  // open output file to store histograms}
  outputFile_  = iConfig.getUntrackedParameter<std::string>("outputFile");
  trackAssociatorLabel_ = iConfig.getUntrackedParameter<std::string>("trackAssociator","TrackAssociatorByChi2");// "TrackAssociatorByHits"
  if (trackAssociatorLabel_=="TrackAssociatorByChi2"){
    trackAssociatorMin_=-100.;
  }else if(trackAssociatorLabel_=="TrackAssociatorByHits"){
    trackAssociatorMin_=0.5;
  }else{
    trackAssociatorMin_=0.0;
  }

  cout << "trackAssociatorLabel=" << trackAssociatorLabel_ << "   trackAssociatorMin_=" << trackAssociatorMin_ << endl;

  rootFile_ = TFile::Open(outputFile_.c_str(),"RECREATE");
  info_=new TObjString("info");
  info_->SetString(iConfig.getUntrackedParameter<std::string>("info","").c_str());
  verbose_= iConfig.getUntrackedParameter<bool>("verbose", false);
  veryverbose_= iConfig.getUntrackedParameter<bool>("veryverbose", false);
  doMatching_= iConfig.getUntrackedParameter<bool>("matching", false);
  sigmaZoverride_= iConfig.getUntrackedParameter<double>("sigmaZ", 0.0); // 0 means use beamspot, >0 means use this value
  nPUmin_= abs(iConfig.getUntrackedParameter<int>("PUmin", 0));
  cout << "nPUMin="<< nPUmin_ << endl;
  nPUmax_= abs(iConfig.getUntrackedParameter<int>("PUmax", 1000000));
  useVertexFilter_ = iConfig.getUntrackedParameter<bool>("useVertexFilter", false);
  //filterBeamError_ = iConfig.getUntrackedParameter<bool>("filterBeamError", false);
  bxFilter_ = iConfig.getUntrackedParameter<int>("bxFilter", 0);  // <0 means all bx
  simUnit_= 1.0;  // starting with CMSSW_1_2_x ??

  dumpSignalVsTag_=iConfig.getUntrackedParameter<bool>("dumpSignalVsTag", false);
  nEventSummary_= iConfig.getUntrackedParameter<int>("eventSummaries", 1000);
  ndump_= iConfig.getUntrackedParameter<int>("nDump", 10);

  std::vector<std::string> defaultCollections;
  defaultCollections.push_back("offlinePrimaryVertices");
  defaultCollections.push_back("offlinePrimaryVerticesWithBS");
  vertexCollectionLabels_=iConfig.getUntrackedParameter< std::vector<std::string> >("vertexCollections", defaultCollections);
  nCompareCollections_=iConfig.getUntrackedParameter<int>("compareCollections", 0);  //-1= compare all, >0 dump n events
  
  currentLS_=-1;
  zmatch_=iConfig.getUntrackedParameter<double>("zmatch", 0.0500);
  cout << "PrimaryVertexAnalyzer4PU: zmatch=" << zmatch_ << endl;
  eventcounter_=0;
  emptyeventcounter_=0;
  dumpcounter_=0;
  eventSummaryCounter_=0;
  DEBUG_=false;
  //DEBUG_=true;
  RECO_=false;
  RECO_=iConfig.getUntrackedParameter< bool >("RECO",false);
  autoDumpCounter_=iConfig.getUntrackedParameter< int >("autodump",0);
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


  const int nbinzdiffrec=400.;


  // release validation histograms used in DoCompare.C
  add(h, new TH1F("nbtksinvtx","reconstructed tracks in vertex",40,-0.5,39.5)); 
  add(h, new TH1F("nbtksinvtxPU","reconstructed tracks in vertex",40,-0.5,39.5)); 
  add(h, new TH1F("nbtksinvtxTag","reconstructed tracks in vertex",40,-0.5,39.5)); 
  add(h, new TH1F("resx","residual x",100,-0.04,0.04));
  add(h, new TH1F("resy","residual y",100,-0.04,0.04));
  add(h, new TH1F("resz","residual z",100,-0.1,0.1));
  add(h, new TH1F("resz10","residual z",100,-1.0,1.));
  add(h, new TH1F("pullx","pull x",100,-25.,25.));
  add(h, new TH1F("pully","pull y",100,-25.,25.));
  add(h, new TH1F("pullz","pull z",100,-25.,25.));
  add(h, new TH1F("vtxchi2","chi squared",100,0.,100.));
  add(h, new TH1F("vtxcxy2","xy-beam compatibility",100,0.,50.));
  add(h, new TH1F("vtxcxy2Matched","xy-beam compatibility",100,0.,50.));
  add(h, new TH1F("vtxcxy2Fake","xy-beam compatibility",100,0.,50.));
  add(h, new TH1F("ntpfake", "fake vertices",10, 0., 10.));
  add(h, new TH1F("ntpfake4","fake vertices with ndof>4",  10, 0., 10.));
  add(h, new TH1F("ndofFake", "ndof of fake vertices", 50, 0., 100.));
  add(h, new TH1F("ntpfound", "found vertices",100, 0., 100.));
  add(h, new TH1F("ntpfound4","found vertices with ndof>4",  100, 0., 100.));
  add(h, new TH1F("zdiffrec4found","z-distance between truth-matched ndof>4 vertices",nbinzdiffrec,-2., 2.));
  add(h, new TH1F("zdiffrec4fake","z-distance between truth-matched and fake ndof>4 vertices",nbinzdiffrec,-2., 2.));
  add(h,  new TProfile("effvsnrectp","efficiency vs # truth matched rec tracks",50, 0., 50., 0, 1.));
  add(h,  new TProfile("effvsngentp","efficiency vs # tracking particles with tracker hits",50, 0., 50., 0, 1.));
  add(h,  new TProfile("effsigvsnrectp","signal efficiency vs # truth matched rec tracks",50, 0., 100., 0, 1.));
  add(h,  new TProfile("effsigvsngentp","signal efficiency vs # tracking particles with tracker hits",50, 0., 100., 0, 1.));

  add(h, new TH1F("vtxndf","degrees of freedom",               5000,0.,1000.));
  add(h, new TH1F("vtxndfc","expected lower ndof of two",      5000,0.,1000.));
  add(h, new TH1F("ndofnr2","expected lower ndof of two"      ,5000,0., 1000.));
  add(h, new TH1F("ndofnr2d1cm","lower ndof of pair (dz>1cm)",5000,0., 1000.));
  add(h, new TH1F("ndofnr2d2cm","lower ndof of pair (dz>2cm)",5000,0., 1000.));
  add(h, new TH1F("vtxndfIso","degrees of freedom (isolated vertex)",   5000,0.,1000.));


  add(h, new TH2F("vtxndfvsntk","ndof vs #tracks",40,0.,200, 40, 0., 400.));
  add(h, new TH1F("vtxndfoverntk","ndof / #tracks",40,0.,2.));
  add(h, new TH1F("vtxndf2overntk","(ndof+2) / #tracks",40,0.,2.));
  add(h, new TH1F("tklinks","Usable track links",2,-0.5,1.5));
  add(h, new TH1F("nans","Illegal values for x,y,z,xx,xy,xz,yy,yz,zz",9,0.5,9.5));


  // raw
  add(h, new TH1F("szRecVtx","size of recvtx collection",20, -0.5, 19.5));
  add(h, new TH1F("isFake","fake vertex",2, -0.5, 1.5));
  add(h, new TH1F("isFake1","fake vertex or ndof<0",2, -0.5, 1.5));
  add(h, new TH1F("bunchCrossing","bunchCrossing",4000, 0., 4000.));
  add(h, new TH2F("bunchCrossingLogNtk","bunchCrossingLogNtk",4000, 0., 4000.,5,0., 5.));
  add(h, new TH1F("highpurityTrackFraction","fraction of high purity tracks",20, 0., 1.));
  add(h, new TH2F("trkchi2vsndof","vertices chi2 vs ndof",50, 0., 100., 50, 0., 200.));
  add(h, new TH1F("trkchi2overndof","vertices chi2 / ndof",50, 0., 5.));
  add(h, new TH1F("z0trk","track z0 (tight, eta<1.5, pt>0.5))",100., -20., 20.));
  // two track vertices
  add(h, new TH2F("2trkchi2vsndof","two-track vertices chi2 vs ndof",40, 0., 10., 20, 0., 20.));
  add(h, new TH1F("2trkmassSS","two-track vertices mass (same sign)",100, 0., 2.));
  add(h, new TH1F("2trkmassOS","two-track vertices mass (opposite sign)",100, 0., 2.));
  add(h, new TH1F("2trkdphi","two-track vertices delta-phi",360, 0, 2*M_PI));
  add(h, new TH1F("2trkseta","two-track vertices sum-eta",50, -2., 2.));
  add(h, new TH1F("2trkdphicurl","two-track vertices delta-phi (sum eta<0.1)",360, 0, 2*M_PI));
  add(h, new TH1F("2trksetacurl","two-track vertices sum-eta (delta-phi<0.1)",50, -2., 2.));
  add(h, new TH1F("2trkdetaOS","two-track vertices delta-eta (same sign)",50, -0.5, 0.5));
  add(h, new TH1F("2trkdetaSS","two-track vertices delta-eta (opposite sign)",50, -0.5, 0.5));
  // two track PU vertices
  add(h, new TH1F("2trkmassSSPU","two-track vertices mass (same sign)",100, 0., 2.));
  add(h, new TH1F("2trkmassOSPU","two-track vertices mass (opposite sign)",100, 0., 2.));
  add(h, new TH1F("2trkdphiPU","two-track vertices delta-phi",360, 0, 2*M_PI));
  add(h, new TH1F("2trksetaPU","two-track vertices sum-eta",50, -2., 2.));
  add(h, new TH1F("2trkdphicurlPU","two-track vertices delta-phi (sum eta<0.1)",360, 0, 2*M_PI));
  add(h, new TH1F("2trksetacurlPU","two-track vertices sum-eta (delta-phi<0.1)",50, -2., 2.));
  add(h, new TH1F("2trkdetaOSPU","two-track vertices delta-eta (same sign)",50, -0.5, 0.5));
  add(h, new TH1F("2trkdetaSSPU","two-track vertices delta-eta (opposite sign)",50, -0.5, 0.5));
  // three track vertices
  add(h, new TH2F("2trkchi2vsndof","two-track vertices chi2 vs ndof",40, 0., 10., 20, 0., 20.));
  add(h, new TH2F("3trkchi2vsndof","three-track vertices chi2 vs ndof",40, 0., 10., 20, 0., 20.));
  add(h, new TH2F("4trkchi2vsndof","four-track vertices chi2 vs ndof",40, 0., 10., 20, 0., 20.));
  add(h, new TH2F("5trkchi2vsndof","five-track vertices chi2 vs ndof",40, 0., 10., 20, 0., 20.));
  // same for fakes
  add(h, new TH2F("fake2trkchi2vsndof","fake two-track vertices chi2 vs ndof",40, 0., 10., 20, 0., 20.));
  add(h, new TH2F("fake3trkchi2vsndof","fake three-track vertices chi2 vs ndof",40, 0., 10., 20, 0., 20.));
  add(h, new TH2F("fake4trkchi2vsndof","fake four-track vertices chi2 vs ndof",40, 0., 10., 20, 0., 20.));
  add(h, new TH2F("fake5trkchi2vsndof","fake five-track vertices chi2 vs ndof",40, 0., 10., 20, 0., 20.));
  // and unmatched (also fakes, but identified differently)
  add(h, new TH2F("unmatchedVtx2trkchi2vsndof","unmatched two-track vertices chi2 vs ndof",40, 0., 10., 20, 0., 20.));
  add(h, new TH2F("unmatchedVtx3trkchi2vsndof","unmatched three-track vertices chi2 vs ndof",40, 0., 10., 20, 0., 20.));
  add(h, new TH2F("unmatchedVtx4trkchi2vsndof","unmatched four-track vertices chi2 vs ndof",40, 0., 10., 20, 0., 20.));
  add(h, new TH2F("unmatchedVtx5trkchi2vsndof","unmatched five-track vertices chi2 vs ndof",40, 0., 10., 20, 0., 20.));
  add(h, new TH1F("resxr","relative residual x",100,-0.04,0.04));
  add(h, new TH1F("resyr","relative residual y",100,-0.04,0.04));
  add(h, new TH1F("reszr","relative residual z",100,-0.1,0.1));
  add(h, new TH1F("resx50","residual x (ndof>50)",100,-0.04,0.04));
  add(h, new TH1F("resy50","residual y (ndof>50)",100,-0.04,0.04));
  add(h, new TH1F("resz50","residual z (ndof>50)",100,-0.1,0.1));
  add(h, new TH1F("pullxr","relative pull x",100,-25.,25.));
  add(h, new TH1F("pullyr","relative pull y",100,-25.,25.));
  add(h, new TH1F("pullzr","relative pull z",100,-25.,25.));
  add(h, new TH1F("vtxprob","chisquared probability",100,0.,1.));
  h["eff"]          = new TH1F("eff","efficiency",2, -0.5, 1.5);
  h["efftag"]       = new TH1F("efftag","efficiency tagged vertex",2, -0.5, 1.5);
  h["zdistancetag"] = new TH1F("zdistancetag","z-distance between tagged and generated",100, -0.1, 0.1);
  h["abszdistancetag"] = new TH1F("abszdistancetag","z-distance between tagged and generated",1000, 0., 1.0);
  h["abszdistancetagcum"] = new TH1F("abszdistancetagcum","z-distance between tagged and generated",1000, 0., 1.0);

  add(h, new TH1F("zdistancenearest","z-distance between generated nearest rec",100, -0.1, 0.1));
  add(h, new TH1F("abszdistancenearest","z-distance between generated and nearest rec",1000, 0., 1.0));
  add(h, new TH1F("abszdistancenearestcum","z-distance between generated and nearest rec",1000, 0., 1.0));
  add(h, new TH1F("indexnearest","index of nearest rec vertex", 20, 0., 20.));

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

  add(h, new TH2F("xyrec","reconstructed xy",100, -4., 4., 100, -4., 4.));
  h["xrecBeam"]     = new TH1F("xrecBeam","reconstructed x - beam x",100,-0.1,0.1);
  h["yrecBeam"]     = new TH1F("yrecBeam","reconstructed y - beam y",100,-0.1,0.1);
  h["zrecBeam"]     = new TH1F("zrecBeam","reconstructed z - beam z",100,-20.,20.);
  h["xrecBeamvsz"] = new TH2F("xrecBeamvsz","reconstructed x - beam x vs z", 20, -20., 20.,100,-0.1,0.1);
  h["yrecBeamvsz"] = new TH2F("yrecBeamvsz","reconstructed y - beam y vs z", 20, -20., 20.,100,-0.1,0.1);
  h["xrecBeamvszprof"] = new TProfile("xrecBeamvszprof","reconstructed x - beam x vs z-z0", 20, -20., 20.,-0.1,0.1);
  h["yrecBeamvszprof"] = new TProfile("yrecBeamvszprof","reconstructed y - beam y vs z-z0", 20, -20., 20.,-0.1,0.1);
  h["xrecBeamvsNdofprof"] = new TProfile("xrecBeamvsNdofprof","reconstructed x - beam x vs ndof", 10, 0., 200.,-0.1,0.1);
  h["yrecBeamvsNdofprof"] = new TProfile("yrecBeamvsNdofprof","reconstructed y - beam y vs ndof", 10, 0., 200.,-0.1,0.1);

  h["resxvsNdofprof"] = new TProfile("resxvsNdofprof","reconstructed x - simulated x vs ndof", 10, 0., 200.,-0.1,0.1);
  h["resyvsNdofprof"] = new TProfile("resyvsNdofprof","reconstructed y - simulated y vs ndof", 10, 0., 200.,-0.1,0.1);

  h["resxvsNdoftest"] = new TProfile("resxvsNdoftest","reconstructed x - simulated x vs ndof", 10, 0., 200.,-0.1,0.1);
  h["resyvsNdoftest"] = new TProfile("resyvsNdoftest","reconstructed y - simulated y vs ndof", 10, 0., 200.,-0.1,0.1);
  h["resxvsNdofSpread"] = new TProfile("resxvsNdofSpread","reconstructed x - simulated x vs ndof", 10, 0., 200.,-0.1,0.1,"S");
  h["resyvsNdofSpread"] = new TProfile("resyvsNdofSpread","reconstructed y - simulated y vs ndof", 10, 0., 200.,-0.1,0.1,"S");

  add(h, new TH2F("zrecLS","reconstructed z vs LS",500, 0. , 500., 40., -20., 20.));


//   add(h, new TH2F("xrecBeamvsdxXBS","reconstructed x - beam x vs resolution",10,0., 0.02, 100, -0.1,0.1)); // just a test
//  add(h, new TH2F("yrecBeamvsdyXBS","reconstructed z - beam z vs resolution",10,0., 0.02, 100, -0.1,0.1));
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
  h["zrec2"]        = new TH1F("zrec2","reconstructed z",200,-40.,40.);

  h["xrec3"]        = new TH1F("xrec3","reconstructed x",100,-0.1,0.1);
  h["yrec3"]        = new TH1F("yrec3","reconstructed y",100,-0.1,0.1);
  h["zrec3"]        = new TH1F("zrec3","reconstructed z",100,-20.,20.);
  add(h, new TH1F("zrec3a","reconstructed z",100,-1.,1.));

  add(h, new TH1F("xrec8","reconstructed x (ndof>8)",100,-0.1,0.1));
  add(h, new TH1F("yrec8","reconstructed y (ndof>8)",100,-0.1,0.1));
  add(h, new TH1F("zrec8","reconstructed z (ndof>8)",100,-20.,20.));

  add(h, new TH1F("xrec12","reconstructed x (ndof>12)",100,-0.1,0.1));
  add(h, new TH1F("yrec12","reconstructed y (ndof>12)",100,-0.1,0.1));
  add(h, new TH1F("zrec12","reconstructed z (ndof>12)",100,-20.,20.));
  add(h, new TH1F("zrec12tag","reconstructed z (tagged,ndof>12)",100,-20.,20.));


  add(h, new TH1F("xrecBeamPull","normalized residuals reconstructed x - beam x",100,-10,10));
  add(h, new TH1F("yrecBeamPull","normalized residuals reconstructed y - beam y",100,-10,10));
  add(h, new TH1F("zrecBeamPull","normalized residuals reconstructed z - beam z",100,-10,10));
  add(h, new TH1F("zrecBeamPull0","normalized residuals reconstructed z - beam z",100,-10,10));
  add(h, new TH1F("zrecBeamPull12","normalized residuals reconstructed z - beam z (ndof>12)",100,-10,10));

  add(h, new TProfile("zvsls","z vs ls",200, 0., 2000., -20., 20.));
  add(h, new TProfile("zbeamvsls","zbeam vs ls",200, 0., 2000., -20., 20.));

  add(h, new TH1F("zdiffrec","z-distance between vertices",nbinzdiffrec,-20., 20.));
  add(h, new TH1F("zdiffrechr","z-distance between vertices",nbinzdiffrec,-2., 2.));
  add(h, new TH1F("zdiffrec2","z-distance between ndof>2 vertices",nbinzdiffrec,-20., 20.));
  add(h, new TH1F("zdiffrec3","z-distance between ndof>3 vertices",nbinzdiffrec,-20., 20.));
  add(h, new TH1F("zdiffrec4","z-distance between ndof>4 vertices",nbinzdiffrec,-20., 20.));
  add(h, new TH1F("zdiffrec4hr","z-distance between ndof>4 vertices",nbinzdiffrec,-2., 2.));
  add(h, new TH1F("zdiffrec4tag","z-distance tagged - other ndof>4 vertices",nbinzdiffrec,-20., 20.));
  add(h, new TH1F("zdiffrec4taghr","z-distance tagged - other ndof>4 vertices",nbinzdiffrec,-2., 2.));
  add(h, new TH1F("zdiffrec5","z-distance between ndof>5 vertices",nbinzdiffrec,-20., 20.));
  add(h, new TH1F("zdiffrec6","z-distance between ndof>6 vertices",nbinzdiffrec,-20., 20.));
  add(h, new TH1F("zdiffrec7","z-distance between ndof>7 vertices",nbinzdiffrec,-20., 20.));
  add(h, new TH1F("zdiffrec8","z-distance between ndof>8 vertices",nbinzdiffrec,-20., 20.));
  add(h, new TH1F("zdiffrec12","z-distance between ndof>12 vertices",2*nbinzdiffrec,-20., 20.));
  add(h, new TH1F("zdiffrec20","z-distance between ndof>20 vertices",2*nbinzdiffrec,-20., 20.));
  add(h, new TH1F("zdiffrecp","normalized z-distance between vertices",nbinzdiffrec,-20., 20.));
  add(h, new TH1F("zdiffrec2p","normalized z-distance between ndof>2 vertices",nbinzdiffrec,-5., 5.));
  add(h, new TH1F("zdiffrec3p","normalized z-distance between ndof>3 vertices",nbinzdiffrec,-5., 5.));
  add(h, new TH1F("zdiffrec4p","normalized z-distance between ndof>4 vertices",nbinzdiffrec,-5., 5.));
  add(h, new TH1F("zdiffrec5p","normalized z-distance between ndof>5 vertices",nbinzdiffrec,-5., 5.));
  add(h, new TH1F("zdiffrec6p","normalized z-distance between ndof>6 vertices",nbinzdiffrec,-5., 5.));
  add(h, new TH1F("zdiffrec7p","normalized z-distance between ndof>7 vertices",nbinzdiffrec,-5., 5.));
  add(h, new TH1F("zdiffrec8p","normalized z-distance between ndof>8 vertices",nbinzdiffrec,-5., 5.));
  add(h, new TH1F("zdiffrec12p","z-distance between ndof>12p vertices",nbinzdiffrec,-20., 20.));
  add(h, new TH2F("zvszrec2","z positions of multiple vertices",200,-20., 20., 200,-20., 20.));
  add(h, new TH2F("pzvspz2","prob(z) of multiple vertices",100, 0.,1.,100,0., 1.));
  add(h, new TH2F("zvszrec4","z positions of multiple vertices",100,-20., 20., 100,-20., 20.));
  add(h, new TH2F("pzvspz4","prob(z) of multiple vertices",100, 0.,1.,100,0., 1.));

  add(h, new TH1F("dzreccentral","z-distance between vertices",100,0., 2.));
  add(h, new TProfile("ndofcentral","<ndof> vs z-distance between vertices",100,0., 2., 0., 500.));
  add(h, new TProfile("ndoflocentral","lower ndof vs z-distance between vertices",100,0., 2., 0., 500.));
  add(h, new TProfile("ndofhicentral","higher ndof vs z-distance between vertices",100,0., 2., 0., 500.));
  add(h, new TProfile("ndofsumcentral","sum of ndof vs z-distance between vertices",100,0., 2., 0., 500.));

  add(h, new TProfile("n0dz","n vs dz ",200,0., 2., 0., 1.));
  add(h, new TProfile("n1dz","n vs dz ",200,0., 2., 0., 1.));
  add(h, new TProfile("n2dz","n vs dz ",200,0., 2., 0., 1.));
  add(h, new TProfile("n3dz","n vs dz ",200,0., 2., 0., 1.));
  add(h, new TProfile("n4dz","n vs dz ",200,0., 2., 0., 1.));
  add(h, new TProfile("n0dz0","n vs dz ",200,0., 2., 0., 1.));
  add(h, new TProfile("n1dz0","n vs dz ",200,0., 2., 0., 1.));
  add(h, new TProfile("n2dz0","n vs dz ",200,0., 2., 0., 1.));
  add(h, new TProfile("n3dz0","n vs dz ",200,0., 2., 0., 1.));
  add(h, new TProfile("n4dz0","n vs dz ",200,0., 2., 0., 1.));

  add(h, new TH2F("nc00","nc00",400, -2., 2., 400, -2., 2.));
  add(h, new TH2F("nc10","nc10",400, -2., 2., 400, -2., 2.));
  add(h, new TH2F("nc01","nc01",400, -2., 2., 400, -2., 2.));
  add(h, new TH2F("nc11","nc11",400, -2., 2., 400, -2., 2.));

  add(h, new TH1F("zdiffsimmerge","z distance of merged or lost simulated vertices",100, -5., 5.));
  add(h, new TH1F("zdiffsimfound","z distance of found simulated vertices",1000, -10., 10));
  add(h, new TH1F("zdiffsimall","z distance of simulated vertices",1000, -10., 10));
  add(h, new TH1F("zdiffsimfoundTP","z distance of found simulated distance",1000, -10., 10));
  add(h, new TH1F("zdiffsimfoundTP2","z distance of found simulated distance (2)",1000, -10., 10));
  add(h, new TH1F("zdiffsimallTP","z distance of simulated distance",1000, -10., 10));
  add(h, new TH2F("zdiffrecvssim","z distance rec vertices vs simulated distance",100, -1., 1.,100, -1., 1.));
  add(h, new TH2F("zdiffrecvssimTP","z distance rec vertices vs simulated distance",100, -1., 1.,100, -1., 1.));
  add(h, new TH2F("zdiffrec4vssim","z distance rec vertices (nd>4) vs simulated distance",100, -1., 1.,100, -1., 1.));
  add(h, new TH2F("zdiffrec4vssimTP","z distance rec vertices (nd>4) vs simulated distance",100, -1., 1.,100, -1., 1.));
  add(h, new TH2F("zdiffrec12vssim","z distance rec vertices (nd>12) vs simulated distance",100, -1., 1.,100, -1., 1.));
  add(h, new TH2F("zdiffrec12vssimTP","z distance rec vertices (nd>12) vs simulated distance",100, -1., 1.,100, -1., 1.));
  add(h, new TH2F("zdiffrecvsdsim","z distance rec vertices vs simulated distance",100, 0., 1.,100, -0.5, 0.5));
  add(h, new TH2F("zdiffrecvsdsimTP","z distance rec vertices vs simulated distance",100, -0.5, 0.5,100, -0.5, 0.5));
  add(h, new TProfile("zdiffrecvsdsimprof","z distance rec vertices vs simulated distance",200, -1.0, 1.0, -0.5, 0.5));
  add(h, new TProfile("zdiffrecvsdsimTPprof","z distance rec vertices vs simulated distance",200, -1.0, 1.0, -0.5, 0.5));
  add(h, new TH1F("zdiffrec4f","z-distance between ndof>4 vertices",1000, -10., 10.));

  add(h, new TH1F("zreciso","zrec-zsim of isolated sim vertices", 500, 0., 1.));
  add(h, new TH1F("nreciso","number of rec vertices near isolated sim vertices", 10, 0., 10.));
  add(h, new TH1F("zdiffsimisoall","z distance of simulated distance (isolated pairs)",500, 0., 10));
  add(h, new TH1F("zdiffsimiso0","simulated z distance (isolated pairs, 0 rec)",500, 0., 10));
  add(h, new TH1F("zdiffsimiso1","simulated z distance (isolated pairs, 1 rec)",500, 0., 10));
  add(h, new TH1F("zdiffsimiso2","simulated z distance (isolated pairs, 2 rec)",500, 0., 10));
  add(h, new TH1F("zdiffsimiso3","simulated z distance (isolated pairs, 3 rec)",500, 0., 10));
  add(h, new TH1F("zdiffreciso2","reconstructed z distance (isolated pairs, 2 rec)",500, 0., 10));
  add(h, new TH2F("dzrecvssimiso2","reconstructed vs simulated z distance (isolated pairs, 2 rec)",200, 0., 2, 200, 0., 2.));

  const int nbinzdiff=150;
  const float zdiffrange=20.;
  add(h, new TH1F("zrec8r","reconstructed (z-z0)*sqrt2 (ndof>8)",nbinzdiff,-zdiffrange,zdiffrange));
  add(h, new TH1F("zrec12r","reconstructed (z-z0)*sqrt2 (ndof>12)",nbinzdiff,-zdiffrange,zdiffrange));
  add(h, new TH1F("zrec12q","reconstructed (z-z0)/sqrt2 (ndof>12)",nbinzdiff,-zdiffrange,zdiffrange));

  add(h, new TH2F("zdiffvszNv2","z-distance vs z (nv=2)",nbinzdiff,-zdiffrange,zdiffrange,30,-15.,15.));
  add(h, new TH1F("zbarFakeEnriched","zbar fake enriched",100,-20.,20.));
  add(h, new TH1F("zbarFakeEnriched2","zbar fake enriched (ndof>2)",100,-20.,20.));
  add(h, new TH1F("zbarFakeEnriched5","zbar fake enriched (ndof>5)",100,-20.,20.));
  add(h, new TH1F("zbarFakeDepleted","zbar fake depleted",100,-20.,20.));
  add(h, new TH2F("zdiffvsz","z-distance vs z",nbinzdiff,-zdiffrange,zdiffrange,30,-15.,15.));
  add(h, new TH2F("zdiffvsz2","z-distance vs z (ndof>2)",nbinzdiff,-zdiffrange,zdiffrange,30,-15.,15.));
  add(h, new TH2F("zdiffvsz3","z-distance vs z (ndof>3)",nbinzdiff,-zdiffrange,zdiffrange,30,-15.,15.));
  add(h, new TH2F("zdiffvsz4","z-distance vs z (ndof>4)",nbinzdiff,-zdiffrange,zdiffrange,30,-15.,15.));
  add(h, new TH2F("zdiffvsz5","z-distance vs z (ndof>5)",nbinzdiff,-zdiffrange,zdiffrange,30,-15.,15.));
  add(h, new TH2F("zdiffvsz6","z-distance vs z (ndof>6)",nbinzdiff,-zdiffrange,zdiffrange,30,-15.,15.));
  add(h, new TH2F("zdiffvsz7","z-distance vs z (ndof>7)",nbinzdiff,-zdiffrange,zdiffrange,30,-15.,15.));
  add(h, new TH2F("zdiffvsz8","z-distance vs z (ndof>8)",nbinzdiff,-zdiffrange,zdiffrange,30,-15.,15.));
  add(h, new TH2F("zdiffvsz12","z-distance vs z (ndof>12)",nbinzdiff,-zdiffrange,zdiffrange,30,-15.,15.));

  add(h, new TH2F("zdiffvszp","z-distance vs z/sigmaZ",nbinzdiff,-zdiffrange,zdiffrange,30,-5.,5.));
  add(h, new TH2F("zdiffvszp2","z-distance vs z/sigmaZ (ndof>2)",nbinzdiff,-zdiffrange,zdiffrange,30,-5.,5.));
  add(h, new TH2F("zdiffvszp3","z-distance vs z/sigmaZ (ndof>3)",nbinzdiff,-zdiffrange,zdiffrange,30,-5.,5.));
  add(h, new TH2F("zdiffvszp4","z-distance vs z/sigmaZ (ndof>4)",nbinzdiff,-zdiffrange,zdiffrange,30,-5.,5.));
  add(h, new TH2F("zdiffvszp5","z-distance vs z/sigmaZ (ndof>5)",nbinzdiff,-zdiffrange,zdiffrange,30,-5.,5.));
  add(h, new TH2F("zdiffvszp6","z-distance vs z/sigmaZ (ndof>6)",nbinzdiff,-zdiffrange,zdiffrange,30,-5.,5.));
  add(h, new TH2F("zdiffvszp7","z-distance vs z/sigmaZ (ndof>7)",nbinzdiff,-zdiffrange,zdiffrange,30,-5.,5.));
  add(h, new TH2F("zdiffvszp8","z-distance vs z/sigmaZ (ndof>8)",nbinzdiff,-zdiffrange,zdiffrange,30,-5.,5.));
  add(h, new TH2F("zdiffvszp12","z-distance vs z/sigmaZ (ndof>12)",nbinzdiff,-zdiffrange,zdiffrange,30,-5.,5.));

  add(h, new TH2F("zdiffvsz2Nv2","z-distance vs z (ndof>2,Nv=2)",nbinzdiff,-zdiffrange,zdiffrange,30,-15.,15.));
  add(h, new TH2F("zdiffvsz3Nv2","z-distance vs z (ndof>3,Nv=2)",nbinzdiff,-zdiffrange,zdiffrange,30,-15.,15.));
  add(h, new TH2F("zdiffvsz4Nv2","z-distance vs z (ndof>4,Nv=2)",nbinzdiff,-zdiffrange,zdiffrange,30,-15.,15.));
  add(h, new TH2F("zdiffvsz5Nv2","z-distance vs z (ndof>5,Nv=2)",nbinzdiff,-zdiffrange,zdiffrange,30,-15.,15.));
  add(h, new TH2F("zdiffvsz6Nv2","z-distance vs z (ndof>6,Nv=2)",nbinzdiff,-zdiffrange,zdiffrange,30,-15.,15.));
  add(h, new TH2F("zdiffvsz7Nv2","z-distance vs z (ndof>7,Nv=2)",nbinzdiff,-zdiffrange,zdiffrange,30,-15.,15.));
  add(h, new TH2F("zdiffvsz8Nv2","z-distance vs z (ndof>8,Nv=2)",nbinzdiff,-zdiffrange,zdiffrange,30,-15.,15.));

  add(h, new TProfile("eff0vsntrec","efficiency vs # reconstructed tracks",50, 0., 50., 0, 1.));
  add(h, new TProfile("eff0vsntsel","efficiency vs # selected tracks",50, 0., 50., 0, 1.));
  add(h, new TProfile("eff0ndof0vsntsel","efficiency (ndof>0) vs # selected tracks",50, 0., 50., 0, 1.));
  add(h, new TProfile("eff0ndof8vsntsel","efficiency (ndof>8) vs # selected tracks",50, 0., 50., 0, 1.));
  add(h, new TProfile("eff0ndof2vsntsel","efficiency (ndof>2) vs # selected tracks",50, 0., 50., 0, 1.));
  add(h, new TProfile("eff0ndof4vsntsel","efficiency (ndof>4) vs # selected tracks",50, 0., 50., 0, 1.));
  add(h, new TH1F("ndofOverNtk","ndof / ntk of ndidates (ndof>4)",100,0., 2.));
  add(h, new TH1F("sumwoverntk","sumw / ntk of ndidates (ndof>4)",100,0., 1.));
  add(h, new TH1D("sumwvsz","sumw (ndof>4)",100, -20., 20.));
  add(h, new TH1D("sumntkvsz","sumntk (ndof>4)",100, -20., 20.));
  add(h, new TH1F("sumwoversumntkvsz","sumw over sumntk (ndof>4)",100, -20., 20.));
  add(h, new TProfile("sumwoverntkvsz","sumw / ntk of candidates (ndof>4)",100, -20., 20.,0., 1.));
  add(h, new TProfile("sumwoverntkvsz0","sumw / ntk of candidates (ndof>0)",100, -20., 20.,0., 1.));
  add(h, new TProfile("sumwoverntkvszlo","sumw / ntk of candidates (ndof<10)",100, -20., 20.,0., 1.));
  add(h, new TProfile("sumwoverntkvszhi","sumw / ntk of candidates (ndof>20)",100, -20., 20.,0., 1.));
  add(h, new TProfile("sumwoverntkvsztp","sumw / ntk of candidates (ndof>4)",100, -20., 20.,0., 1.));
  add(h, new TProfile("sumwoverntkwgt05vsz","sumw / ntk(w>0.5) of candidates (ndof>4)",100, -20., 20.,0., 1.));
  add(h, new TProfile("ntrkvsz","<ntrk> vs z (ndof>4)",100, -20., 20.,0., 1000.));
  add(h, new TProfile("ntrkpt1vsz","<ntrk pt>1.0> vs z (ndof>4)",100, -20., 20.,0., 1000.));
  add(h, new TProfile("ndofvsz","<ndof> vs z (ndof>4)",100, -20., 20.,0., 1000.));
  add(h, new TH2F("log10ndofvsz","log10(ndof) vs z",100, -20., 20.,20,-1, 3.));
  add(h, new TH1F("sumwoverntk0","sumw / ntk of ndidates (ndof>0)",100,0., 1.));
  add(h, new TH2F("sumwoverntkvsz4","sumw / ntk of candidates (ndof>4)",100, -20., 20.,20,0., 1.));

  add(h, new TH1F("nrecvtx","# of reconstructed vertices", 50, -0.5, 49.5));
  add(h, new TH1F("nrecvtx2","# of reconstructed vertices with ndof>2", 50, -0.5, 49.5));
  add(h, new TH1F("nrecvtx3","# of reconstructed vertices with ndof>3", 50, -0.5, 49.5));
  add(h, new TH1F("nrecvtx4","# of reconstructed vertices with ndof>4", 50, -0.5, 49.5));
  add(h, new TH1F("nrecvtx5","# of reconstructed vertices with ndof>5", 50, -0.5, 49.5));
  add(h, new TH1F("nrecvtx6","# of reconstructed vertices with ndof>6", 50, -0.5, 49.5));
  add(h, new TH1F("nrecvtx7","# of reconstructed vertices with ndof>7", 50, -0.5, 49.5));
  add(h, new TH1F("nrecvtx8","# of reconstructed vertices with ndof>8", 50, -0.5, 49.5));
  add(h, new TH1F("nrectrk","# of reconstructed tracks", 1000, -0.5, 999.5));
  add(h, new TH1F("nsimtrk","# of simulated tracks", 100, -0.5, 99.5));
  add(h, new TH1F("nsimtrkSignal","# of simulated tracks (Signal)", 100, -0.5, 99.5));
  add(h, new TH1F("nsimtrkPU","# of simulated tracks (PU)", 1-00, -0.5, 999.5));
  add(h, new TH2F("nrecvtxvsL","# of reconstructed vertices vs instbxlumi", 100, 0., 5., 30, -0.5, 29.5));
  add(h, new TH2F("nrecvtx4vsL","# of reconstructed vertices (ndof>4) vs instbxlumi", 100, 0., 5., 30, -0.5, 29.5));
  add(h, new TProfile("nrecvtxvsLprof","# of reconstructed vertices vs instbxlumi", 100, 0., 5.,  0., 100.));
  add(h, new TProfile("nrecvtx4vsLprof","# of reconstructed vertices (ndof>4) vs instbxlumi", 100, 0., 5.,0, 100.));
  add(h, new TH2F("nbtksinvtxvsL","reconstructed tracks in vertex",50, 0., 2.5, 80,-0.5,79.5)); 
  add(h, new TH2F("nbtksinvtx4vsL","reconstructed tracks in vertex(ndof>4)",50, 0., 2.5, 80,-0.5,79.5)); 
  add(h, new TH2F("Npixvsnrecvtx4","# of pixel clusters vs reconstructed vertices (ndof>4)",  30, -0.5, 29.5, 100, 0., 10000.));
  add(h, new TProfile("Npixvsnrecvtx4prof","# of pixel clusters vs reconstructed vertices(ndof>4)",  30, -0.5, 29.5, 0., 10000.));


  h["nsimtrk"]->StatOverflows(kTRUE);
  h["nsimtrkPU"]->StatOverflows(kTRUE);
  h["nsimtrkSignal"]->StatOverflows(kTRUE);
  h["xrectag"]      = new TH1F("xrectag","reconstructed x, signal vtx",100,-0.05,0.05);
  h["yrectag"]      = new TH1F("yrectag","reconstructed y, signal vtx",100,-0.05,0.05);
  h["zrectag"]      = new TH1F("zrectag","reconstructed z, signal vtx",100,-20.,20.);
  h["nrectrk0vtx"]  = new TH1F("nrectrk0vtx","# rec tracks no vertex ",100,-0.5, 99.5);
  h["nseltrk0vtx"]  = new TH1F("nseltrk0vtx","# rec tracks no vertex ",100,-0.5, 99.5);
  //h["nrecsimtrk"]   = new TH1F("nrecsimtrk","# rec tracks matched to sim tracks in vertex",100,-0.5, 99.5);
  //h["nrecnosimtrk"] = new TH1F("nrecsimtrk","# rec tracks not matched to sim tracks in vertex",100,-0.5, 99.5);
  h["trackAssEffvsPt"] =  new TProfile("trackAssEffvsPt","track association efficiency vs pt",20, 0., 100., 0, 1.);

  // cluster stuff
  h["nseltrk"]         = new TH1F("nseltrk","# of reconstructed tracks selected for PV", 1000, -0.5, 999.5);
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
  add(h, new TH1F("fakeVtxZNdofgt4","z of fake vertices with ndof>4", 100, -20., 20.));
  add(h, new TH1F("fakeVtxZNdofgt8","z of fake vertices with ndof>8", 100, -20., 20.));
  add(h, new TH1F("fakeVtxZ","z of fake vertices", 100, -20., 20.));
  add(h, new TH1F("fakeVtxNdof","ndof of fake vertices", 500,0., 100.));
  add(h,new TH1F("fakeVtxNtrk","number of tracks in fake vertex",20,-0.5, 19.5));
  add(h,new TH1F("matchedVtxNdof","ndof of matched vertices", 500,0., 100.));


  //  histograms of track quality (Data and MC)
  string types[] = {"all","sel","sellost","wgt05","wlt05","|z|<2","|z|>10",
		    "tagged","untagged","ndof4","unmatchedVtx",
		    "seltpmatched","seltpunmatched"};
  for(unsigned int t=0; t<13; t++){
    string st=types[t];
    const char* ct=types[t].c_str();
    string stp=" ("+types[t]+")";
    add(h, new TH1F(("rapidity_"+st).c_str(),"rapidity ",100,-3., 3.));
    add(h, new TH1F(("z0_"+st).c_str(),"z0 ",200,-40., 40.));
    add(h, new TH1F(("phi_"+st).c_str(),"phi ",80,-3.14159, 3.14159));
    add(h, new TH1F(("eta_"+st).c_str(),"eta ",80,-4., 4.));
    add(h, new TH1F(("pt_"+st).c_str(),"pt ",100,0., 5.));
    add(h, new TH1F(("pthi_"+st).c_str(),"pt ",100,0., 100.));
    add(h, new TH1F(("ptfwd_"+st).c_str(),"pt (forward)",100,0., 5.));
    add(h, new TH1F(("ptcentral_"+st).c_str(),"pt (central)",100,0., 5.));
    add(h, new TH1F(("found_"+st).c_str(),"found hits",20, 0., 20.));
    add(h, new TH1F(("lost_"+st).c_str(),"lost hits",20, 0., 20.));
    add(h, new TH1F(("nchi2_"+st).c_str(),"normalized track chi2",100, 0., 20.));
    add(h, new TH1F(("rstart_"+st).c_str(),"start radius",100, 0., 20.));
    add(h, new TH1F(("expectedInner_"+st).c_str(),"expected inner hits ",10, 0., 10.));
    add(h, new TH1F(("expectedOuter_"+st).c_str(),"expected outer hits ",10, 0., 10.));
    add(h, new TH1F(("logtresxy_"+st).c_str(),"log10(track r-phi resolution/um)",100, 0., 5.));
    add(h, new TH1F(("logtresz_"+st).c_str(),"log10(track z resolution/um)",100, 0., 5.));
    add(h, new TH1F(Form("tpullxy_%s",ct),Form("track r-phi pull (%s)",ct),100, -10., 10.));
    add(h, new TProfile(("tpullxyvsz_"+st).c_str(),("track r-phi pull"+stp).c_str(), 100, -20., 20., -10., 10.));
    add(h, new TProfile(("tpullzvsz_"+st).c_str(),"track z pull",100, -20., 20., -10., 10.));
    add(h, new TH2F( ("lvseta_"+st).c_str(),"cluster length vs eta",60,-3., 3., 20, 0., 20));
    add(h, new TH2F( ("lvstanlambda_"+st).c_str(),"cluster length vs tan lambda",60,-6., 6., 20, 0., 20));
    add(h, new TH1D( ("restrkz_"+st).c_str(),"z-residuals (track vs vertex)", 200, -2., 2.));
    add(h, new TH2F( ("restrkzvsphi_"+st).c_str(),"z-residuals (track - vertex)", 12,-3.14159,3.14159,100, -1., 1.));
    add(h, new TH2F( ("restrkzvseta_"+st).c_str(),"z-residuals (track - vertex)", 12,-3.,3.,100, -1., 1.));
    add(h, new TH2F( ("restrkzvsz_"+st).c_str(),"z-residuals (track - vertex) vs z", 100,-20.,20.,100, -1., 1.));
    add(h, new TH2F( ("pulltrkzvsphi_"+st).c_str(),"normalized z-residuals (track - vertex)", 12,-3.14159,3.14159,100, -5., 5.));
    add(h, new TH2F( ("pulltrkzvseta_"+st).c_str(),"normalized z-residuals (track - vertex)", 12,-3.,3.,100, -5., 5.));
    add(h, new TH2F( ("pulltrkzvsz_"+st).c_str(),"normalized z-residuals (track - vertex) vs z", 100,-20., 20., 100, -5., 5.));
    add(h, new TH1D( ("pulltrkz_"+st).c_str(),"normalized z-residuals (track vs vertex)", 100, -5., 5.));
    add(h, new TH1D( ("sigmatrkz0_"+st).c_str(),"z-resolution (excluding beam)", 100, 0., 5.));
    add(h, new TH1D( ("sigmatrkz_"+st).c_str(),"z-resolution (including beam)", 100,0., 5.));
    add(h, new TH1D( ("nbarrelhits_"+st).c_str(),"number of pixel barrel hits", 10, 0., 10.));
    add(h, new TH1D( ("nbarrelLayers_"+st).c_str(),"number of pixel barrel layers", 10, 0., 10.));
    add(h, new TH1D( ("nPxLayers_"+st).c_str(),"number of pixel layers (barrel+endcap)", 10, 0., 10.));
    add(h, new TH1D( ("nSiLayers_"+st).c_str(),"number of Tracker layers ", 20, 0., 20.));
    add(h, new TH1D( ("n3dLayers_"+st).c_str(),"number of 3d Tracker layers ", 20, 0., 20.));
    add(h, new TH1D( ("trackAlgo_"+st).c_str(),"track algorithm ", 30, 0., 30.));
    add(h, new TH2F( ("nPxLayersVsPt_"+st).c_str(),"number of pixel layers (barrel+endcap)", 8,0.,8.,10, 0., 10.));
    add(h, new TH1D( ("trackQuality_"+st).c_str(),"track quality ", 7, -1., 6.));
  }// track types
  add(h, new TH1F("wosfrac","wos fraction of matched vertex",256,0., 1.));
  add(h, new TH1F("nwosmatch","number of wos matche vertices (splitting)",5,0., 5.));
  //
  add(h, new TH1F("trackWt","track weight in vertex",256,0., 1.));
  add(h, new TH1F("allweight","track weight in vertex",256,0., 1.));
  add(h, new TH1F("minorityweight","minority track weight",256,0., 1.));
  add(h, new TH1F("minorityaweight","minority(a) track weight",256,0., 1.));
  add(h, new TH1F("minoritybweight","minority(b) track weight",256,0., 1.));
  add(h, new TH1F("majorityweight","majority track weight",256,0., 1.));
  add(h, new TH1F("unmatchedweight","unmatched track weight",256,0., 1.));
  add(h, new TH1F("unmatchedvtxtrkweight","unmatched vertex track weight",256,0., 1.));
  add(h, new TProfile("vtxpurityvsz","vertex purity",100, -20., 20., 0., 1.));
  add(h, new TProfile("ntrkwgt05vsz","number of w>0.5 tracks",100, -20., 20., 0., 1000.));
  add(h, new TProfile("ftrkwgt05vsz","fraction of w>0.5 tracks",100, -20., 20., 0., 1.));
  add(h, new TProfile("trackwtvsz","track weight vs z (all)",100, -20., 20., 0., 1.));
  add(h, new TProfile("trackwtgt05vsz","track weight vs z (w>0.5)",100, -20., 20., 0.5, 1.));
  add(h, new TProfile("allweightvsz","track weight vs z (all)",100, -20., 20., 0., 1.));
  add(h, new TProfile("minorityweightvsz","minority track weight",100, -20., 20., 0., 1.));
  add(h, new TProfile("minorityaweightvsz","minority(a) track weight",100, -20., 20., 0., 1.));
  add(h, new TProfile("minoritybweightvsz","minority(b) track weight",100, -20., 20., 0., 1.));
  add(h, new TProfile("majorityweightvsz","majority track weight",100, -20., 20., 0., 1.));
  add(h, new TProfile("unmatchedweightvsz","unmatched track weight",100, -20., 20., 0., 1.));
  add(h, new TProfile("minorityfractionvsz","minority track fraction",100, -20., 20., 0., 1.));
  add(h, new TProfile("minorityafractionvsz","minority(a) track fraction",100, -20., 20., 0., 1.));
  add(h, new TProfile("minoritybfractionvsz","minority(b) track fraction",100, -20., 20., 0., 1.));
  add(h, new TProfile("majorityfractionvsz","majority track fraction",100, -20., 20., 0., 1.));
  add(h, new TProfile("unmatchedfractionvsz","unmatched track fraction (in vtx)",100, -20., 20., 0., 1.));
  add(h, new TProfile("unmatchedvtxtrkfractionvsz","unmatched vertex track fraction",100, -20., 20., 0., 1.));
  add(h, new TProfile("unmatchedvtxtrkweightvsz","unmatched vertex track weight",100, -20., 20., 0., 1.));
   
  add(h, new TProfile("matchedselfractionvsz","matched fraction of selected tracks",100, -20., 20., 0., 1.));
  add(h, new TProfile("unmatchedselfractionvsz","unmatched fraction of selected tracks",100, -20., 20., 0., 1.));
  add(h, new TProfile("matchedallfractionvsz","matched fraction of reco tracks",100, -20., 20., 0., 1.));
  add(h, new TProfile("unmatchedallfractionvsz","unmatched fraction of reco tracks",100, -20., 20., 0., 1.));

      

  h["nrectrk"]->StatOverflows(kTRUE);
  h["nrectrk"]->StatOverflows(kTRUE);
  h["nrectrk0vtx"]->StatOverflows(kTRUE);
  h["nseltrk0vtx"]->StatOverflows(kTRUE);
  h["nseltrk"]->StatOverflows(kTRUE);
  h["nbtksinvtx"]->StatOverflows(kTRUE);
  h["nbtksinvtxPU"]->StatOverflows(kTRUE);
  h["nbtksinvtxTag"]->StatOverflows(kTRUE);
  h["nbtksinvtx2"]->StatOverflows(kTRUE);
  h["nbtksinvtxPU2"]->StatOverflows(kTRUE);
  h["nbtksinvtxTag2"]->StatOverflows(kTRUE);

  // pile-up and track assignment related histograms (MC with TP)
  add(h, new TH1F("npu0","Number of simulated vertices",40,0.,40.));
  add(h, new TH1F("npu1","Number of simulated vertices with >0 track",40,0.,40.));
  add(h, new TH1F("npu2","Number of simulated vertices with >1 track",40,0.,40.));
  add(h, new TH1F("npu3","Number of simulated vertices with >2 track",40,0.,40.));
  add(h, new TH1F("npu4","Number of simulated vertices with >3 track",40,0.,40.));
  add(h, new TH1F("npu5","Number of simulated vertices with >4 track",40,0.,40.));

  
  for(int nt=0; nt<5; nt++){
    add(h,new TH2F(Form("nrecvsnpus%d",nt), Form("# or reconstructed vertices vs number sim vertices with >=%d tracks",nt),40, 0., 40., 40, 0., 40.));
    add(h,new TH2F(Form("nrec4vsnpus%d",nt), Form("# or reconstructed vertices vs number sim vertices with >=%d tracks",nt),40, 0., 40., 40, 0., 40.));
    add(h,new TProfile(Form("nrec4vsnpus%dprof",nt), Form("# or reconstructed vertices vs number sim vertices with >=%d tracks",nt),40, 0., 40.,  0., 100.));
  }
  add(h,new TH1F("nrecv","# of reconstructed vertices", 40, 0, 40));
  add(h,new TH1F("nrecv4","# of reconstructed vertices (ndof>4)", 40, 0, 40));
  add(h,new TH2F("nrecvsnpu","#rec vertices vs number of sim vertices with >0 tracks", 40,  0., 40, 40,  0, 40));
  add(h,new TH2F("nrec2vsnpu","#rec vertices vs number of sim vertices with >0 tracks", 40,  0., 40, 40,  0, 40));
  add(h,new TH2F("nrec4vsnpu","#rec vertices vs number of sim vertices with >0 tracks", 40,  0., 40, 40,  0, 40));
  add(h,new TH2F("nrecvsnpu2","#rec vertices vs number of sim vertices with >1 tracks", 40,  0., 40, 40,  0, 40));
  add(h,new TH2F("nrec2vsnpu2","#rec vertices vs number of sim vertices with >1 tracks", 40,  0., 40, 40,  0, 40));
  add(h,new TH2F("nrec4vsnpu2","#rec vertices vs number of sim vertices with >1 tracks", 40,  0., 40, 40,  0, 40));
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

  add(h,new TH1F("recmatchPurity","track purity of all vertices", 101, 0., 1.01));
  add(h,new TH1F("recmatchPurityTag","track purity of tagged vertices", 101, 0., 1.01));
  add(h,new TH1F("recmatchPurityTagSignal","track purity of tagged Signal vertices", 101, 0., 1.01));
  add(h,new TH1F("recmatchPurityTagPU","track purity of tagged PU vertices", 101, 0., 1.01));
  add(h,new TH1F("recmatchPuritynoTag","track purity of untagged vertices", 101, 0., 1.01));
  add(h,new TH1F("recmatchPuritynoTagSignal","track purity of untagged Signal vertices", 101, 0., 1.01));
  add(h,new TH1F("recmatchPuritynoTagPU","track purity of untagged PU vertices", 101, 0., 1.01));
  add(h,new TH1F("recmatchvtxs","number of sim vertices contributing to a recvtx", 10, 0., 10.));
  add(h,new TH1F("recmatchvtxsTag","number of sim vertices contributing to a recvtx", 10, 0., 10.));
  add(h,new TH1F("recmatchvtxsnoTag","number of sim vertices contributing to a recvtx", 10, 0., 10.));
  add(h,new TH1F("recmatch30vtxs","number of sim vertices contributing >30% of their tracks to a recvtx", 10, 0., 10.));
  add(h,new TH1F("recmatch30vtxsTag","number of sim vertices contributing >30% of their tracks to a recvtx", 10, 0., 10.));
  add(h,new TH1F("recmatch30vtxsnoTag","number of sim vertices contributing >30% of their tracks to a recvtx", 10, 0., 10.));
  add(h,new TH1F("recmatch50vtxs","number of sim vertices contributing >50% of their tracks to a recvtx", 10, 0., 10.));
  add(h,new TH1F("recmatch50vtxsTag","number of sim vertices contributing >50% of their tracks to a recvtx", 10, 0., 10.));
  add(h,new TH1F("recmatch50vtxsnoTag","number of sim vertices contributing >50% of their tracks to a recvtx", 10, 0., 10.));
  //
  add(h,new TH1F("trkAssignmentEfficiency", "track to vertex assignment efficiency", 101, 0., 1.01) );
  add(h,new TH1F("trkAssignmentEfficiencySignal", "track to signal vertex assignment efficiency", 101, 0., 1.01) );
  add(h,new TH1F("trkAssignmentEfficiencyPU", "track to PU vertex assignment efficiency", 101, 0., 1.01) );
  add(h,new TH1F("primtrkAssignmentEfficiency", "track to vertex assignment efficiency", 101, 0., 1.01) );
  add(h,new TH1F("primtrkAssignmentEfficiencySignal", "track to signal vertex assignment efficiency", 101, 0., 1.01) );
  add(h,new TH1F("primtrkAssignmentEfficiencyPU", "track to PU vertex assignment efficiency", 101, 0., 1.01) );
  add(h,new TH1F("vtxMultiplicity", "number of rec vertices containing tracks from one true vertex", 10, 0., 10.) );
  add(h,new TH1F("vtxMultiplicitySignal", "number of rec vertices containing tracks from the Signal Vertex", 10, 0., 10.) );
  add(h,new TH1F("vtxMultiplicityPU", "number of rec vertices containing tracks from a PU Vertex", 10, 0., 10.) );
  add(h,new TH1F("vtxMultiplicity50", "number of rec vertices containing >=50% tracks from one true vertex", 10, 0., 10.) );
  add(h,new TH1F("vtxMultiplicity50Signal", "number of rec vertices containing tracks from the Signal Vertex", 10, 0., 10.) );
  add(h,new TH1F("vtxMultiplicity50PU", "number of rec vertices containing tracks from a PU Vertex", 10, 0., 10.) );
  add(h,new TH1F("vtxMultiplicity30", "number of rec vertices containing >=30% tracks from one true vertex", 10, 0., 10.) );
  add(h,new TH1F("vtxMultiplicity30Signal", "number of rec vertices containing tracks from the Signal Vertex", 10, 0., 10.) );
  add(h,new TH1F("vtxMultiplicity30PU", "number of rec vertices containing tracks from a PU Vertex", 10, 0., 10.) );
  
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
  add(h,new TH1F("matchVtxEfficiency","efficiency for finding matching rec vertex (ntsim>0)",2,-0.5,1.5));
  add(h,new TH1F("matchVtxEfficiencySignal","efficiency for finding matching rec vertex (ntsim>0)",2,-0.5,1.5));
  add(h,new TH1F("matchVtxEfficiencyPU","efficiency for finding matching rec vertex (ntsim>0)",2,-0.5,1.5));
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
  add(h, new TH1F("unmatchedVtx4","number of unmatched rec vertices ndof>4 (fakes)",10,0.,10.));
  add(h, new TH1F("unmatchedVtxW4","number of unmatched (by weight) rec vertices ndof>4 (fakes)",10,0.,10.));
  add(h, new TH1F("unmatchedVtxNtrk","number of tracks in unmatched vertex",20,-0.5, 19.5));
  add(h, new TH1F("unmatchedVtxFrac","fraction of unmatched rec vertices (fakes)",1000,0.,1.0));
  add(h, new TH1F("unmatchedVtxZ","z of unmached rec  vertices (fakes)",100,-20., 20.));
  add(h, new TH1F("unmatchedVtxDeltaZ","Delta z of unmached rec  vertices (fakes)",100,-20., 20.));
  add(h, new TH1F("unmatchedVtxNdof","ndof of unmatched rec vertices (fakes)", 500,0., 100.));
  add(h, new TH1F("unmatchedVtxNdof1","ndof of unmatched rec vertices (fakes, delta z>1cm)", 500,0., 100.));
  add(h, new TH1F("unmatchedVtxNdof2","ndof of unmatched rec vertices (fakes, delta z>2cm)", 500,0., 100.));
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

  MC_              = false;

  rootFile_->cd();

  for(std::vector<std::string>::const_iterator vCollection=vertexCollectionLabels_.begin(); vCollection!=vertexCollectionLabels_.end(); vCollection++){
    cout << "PrimaryVertexAnalyzer4PU: booking histograms for collection " << *vCollection <<endl;
    std::string s=*vCollection;
    TDirectory *dir = rootFile_->mkdir(s.c_str());
    dir->cd();
    histograms_[s]=bookVertexHistograms();
    for(std::map<std::string,TH1*>::const_iterator hist=histograms_[s].begin(); hist!=histograms_[s].end(); hist++){
      hist->second->SetDirectory(dir);
    }

  }


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
  add(hsimPV, new TH1F("zbeam","beamspot z",100,-1.,1));
  add(hsimPV, new TH1F("wxbeam","beamspot sigma x",100,0.,0.02));
  add(hsimPV, new TH1F("wybeam","beamspot sigma y",100,0.,0.02));
  add(hsimPV, new TH1F("sigmaZbeam","beamspot sigma z",100,0.,10.));
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



  add(hTrk, new TH1F("deltaphi","delta phi (sum eta<0.1)", 400, -M_PI, M_PI));
  add(hTrk, new TH1F("sumeta","sum eta (delta phi - pi <0.1)", 200, -1., 1.));
  add(hTrk, new TH1F("ptloop","pt of looper candidates", 100, 0., 2.));
  add(hTrk, new TH1F("dptloop","delta pt of looper candidates", 100, -1., 1.));
  add(hTrk, new TH1F("zloop","z of looper candidates", 100, -40., 40.));
  add(hTrk, new TH1F("dzloop","delta z of looper candidates", 100, -1., 1.));
  add(hTrk, new TH1F("sumdxyloop","sum dxy of looper candidates", 100, -1., 1.));
  add(hTrk, new TH1F("deltaphisel","delta phi (all cuts)", 400, -M_PI, M_PI));
  add(hTrk, new TH1F("deltaphi","delta phi (sum eta<0.1)", 400, -M_PI, M_PI));

  add(hTrk, new TH1F("deltaphi2","delta phi (sum eta<0.1)", 400, -M_PI, M_PI));
  add(hTrk, new TH1F("deta2","delta eta (delta phi <0.1)", 400, -M_PI, M_PI));
  add(hTrk, new TH1F("ptloop2","pt of same sign tracks", 100, 0., 2.));
  add(hTrk, new TH1F("dptloop2","delta pt of same sign tracks", 100, -1., 1.));
  add(hTrk, new TH1F("zloop2","z of same sign tracks", 100, -40., 40.));
  add(hTrk, new TH1F("dzloop2","delta z of same sign tracks", 100, -1., 1.));
  add(hTrk, new TH1D("dzall","delta z of all tracks", 200, -1., 1.)); // bin overflows!!

  add(hEvt, new TH1F("Lbx","instantaneous BX lumi",100, 0., 10.));
  add(hEvt, new TH2F("nDigiPixvsL","# of pixel hits vs instbxlumi", 50, 0., 2.5, 1000, 0, 10000.));
  add(hEvt, new TProfile("nDigiPixvsLprof","# of pixel hits vs instbxlumi", 50, 0., 2.5, 0, 10000.));
  add(hEvt, new TH1F("bunchCrossing","bunchCrossing",4000, 0., 4000.));
  add(hEvt, new TH2F("bunchCrossingLogNtk","bunchCrossingLogNtk",4000, 0., 4000.,5,0., 5.));
  //  add(hEvt, new TH1F("highpurityTrackFraction","fraction of high purity tracks",20, 0., 1.));
}


void PrimaryVertexAnalyzer4PU::endJob() {
  std::cout << "this is void PrimaryVertexAnalyzer4PU::endJob() " << std::endl;


  for(std::vector<std::string>::const_iterator vCollection=vertexCollectionLabels_.begin(); vCollection!=vertexCollectionLabels_.end(); vCollection++){
    std::map<std::string, TH1*>  h=histograms_[*vCollection];

    for(int i=1; i<101; i++){
      if (h["sumntkvsz"]->GetBinContent(i)>0){
	h["sumwoversumntkvsz"]->SetBinContent(i,h["sumwvsz"]->GetBinContent(i)/h["sumntkvsz"]->GetBinContent(i));
      }
    }

    double sum=0;
    for(int i=101; i>0; i--){
      sum+=h["matchVtxFractionSignal"]->GetBinContent(i)/h["matchVtxFractionSignal"]->Integral();
      h["matchVtxFractionCumSignal"]->SetBinContent(i,sum);
    }
    sum=0;
    for(int i=1; i<1001; i++){
      sum+=h["abszdistancetag"]->GetBinContent(i);
      h["abszdistancetagcum"]->SetBinContent(i,sum/float(h["abszdistancetag"]->GetEntries()));
    }

    sum=0;
    for(int i=1; i<1001; i++){
      sum+=h["abszdistancenearest"]->GetBinContent(i);
      h["abszdistancenearestcum"]->SetBinContent(i,sum/float(h["abszdistancenearest"]->GetEntries()));
    }

    Cumulate(h["matchVtxZCum"]);   Cumulate(h["matchVtxZCumSignal"]);   Cumulate(h["matchVtxZCumPU"]); 

 
    double p;
    unsigned int nbin=h["vtxndf"]->GetNbinsX();
    for(unsigned int i=1; i<=nbin; i++){
      if(h["vtxndf"]->GetEntries()>0){
	p=  h["vtxndf"]->Integral(i,nbin+1)/h["vtxndf"]->GetEntries();    h["vtxndfc"]->SetBinContent(i,p*h["vtxndf"]->GetBinContent(i));
      }
    }


   }
  
  rootFile_->cd();

  std::cout << "Info=" << info_->String() << std::endl;
  if(info_->GetString().Length()>0){
    info_->Write("Info");
  }

  for(std::map<std::string,TH1*>::const_iterator hist=hEvt.begin(); hist!=hEvt.end(); hist++){
    hist->second->Write();
  }

  for(std::map<std::string,TH1*>::const_iterator hist=hsimPV.begin(); hist!=hsimPV.end(); hist++){
    hist->second->Write();
  }
  for(std::map<std::string,TH1*>::const_iterator hist=hTrk.begin(); hist!=hTrk.end(); hist++){
    hist->second->Write();
    int nbin=hist->second->GetNbinsX();
    double I=hist->second->GetBinContent(0)+hist->second->GetBinContent(nbin+1)+hist->second->Integral();
    if (hist->second->GetEntries()>I){
      cout << "Warning !  possible bin content overflow in " << hist->first <<  "entries=" << hist->second->GetEntries() 
	   << "   integral=" << I << endl;
    }
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
           //double x1=x0-0.033; double y1=y0-0.; // FIXME how do we get the simulated beam position?
	   double x1=x0; double y1=y0;
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



std::vector<PrimaryVertexAnalyzer4PU::SimPart> PrimaryVertexAnalyzer4PU::getSimTrkParameters(
											     const Handle<reco::GenParticleCollection>  genParticles)
{
   std::vector<SimPart > tsim;
   double xp=0,yp=0,zp=-99;
  for(size_t i = 0; i < genParticles->size(); ++ i) {
     const GenParticle & gp = (*genParticles)[i];
     int pdgCode=gp.pdgId();
     int st=gp.status();

     if( (st==1)&&(xp==0)&&(yp==0)&&(zp==-99)) {
       xp=gp.vx(); yp=gp.vy(); zp=gp.vz();
     }
     if( pdgCode==-99 ){
       // such entries cause crashes, no idea what they are
       std::cout << "funny particle skipped  , code="  << pdgCode << std::endl;
     }else{
       double Q=gp.charge();
       if ( (st==1)&&(Q != 0) && (gp.pt()>0.1)  && (fabs(gp.eta())<3.0)
              && fabs(gp.vz()<20) && (sqrt(gp.vx()*gp.vx()+gp.vy()*gp.vy())<10.)){
           double x0=gp.vx();
           double y0=gp.vy();
           double z0=gp.vz();
           double kappa=-Q*0.002998*fBfield_/gp.pt();
           double D0=x0*sin(gp.phi())-y0*cos(gp.phi())-0.5*kappa*(x0*x0+y0*y0);
           double q=sqrt(1.-2.*kappa*D0);
           double s0=(x0*cos(gp.phi())+y0*sin(gp.phi()))/q;
           double s1;
           if (fabs(kappa*s0)>0.001){
             s1=asin(kappa*s0)/kappa;
           }else{
             double ks02=(kappa*s0)*(kappa*s0);
             s1=s0*(1.+ks02/6.+3./40.*ks02*ks02+5./112.*pow(ks02,3));
           }
           SimPart sp;//ParameterVector par;
           sp.par[reco::TrackBase::i_qoverp] = Q/gp.p();
           sp.par[reco::TrackBase::i_lambda] = M_PI/2.-gp.theta();
           sp.par[reco::TrackBase::i_phi] = gp.phi()-asin(kappa*s0);
           sp.par[reco::TrackBase::i_dxy] = -2.*D0/(1.+q);
           sp.par[reco::TrackBase::i_dsz] = z0*sin(gp.theta())-s1*cos(gp.theta());

	   sp.pdg=pdgCode;
	   double t=sqrt(pow(gp.vx()-xp,2)+pow(gp.vy()-yp,2)+pow(gp.vz()-zp,2));
           if (t<1e-6){
             sp.type=0;  // primary
           }else{
             sp.type=1;  //secondary
           }

           // now get zpca  (get perigee wrt beam)
           //double x1=x0-0.033; double y1=y0-0.; // FIXME how do we get the simulated beam position?
	   double x1=x0; double y1=y0;
           D0=x1*sin(gp.phi())-y1*cos(gp.phi())-0.5*kappa*(x1*x1+y1*y1);
           q=sqrt(1.-2.*kappa*D0);
           s0=(x1*cos(gp.phi())+y1*sin(gp.phi()))/q;
           if (fabs(kappa*s0)>0.001){
             s1=asin(kappa*s0)/kappa;
           }else{
             double ks02=(kappa*s0)*(kappa*s0);
             s1=s0*(1.+ks02/6.+3./40.*ks02*ks02+5./112.*pow(ks02,3));
           }
           sp.ddcap=-2.*D0/(1.+q);
           sp.zdcap=z0-s1/tan(gp.theta());
           sp.zvtx=z0;
           sp.xvtx=x0;
           sp.yvtx=y0;

           tsim.push_back(sp);
         }
     }
   }//for loop
   return tsim;
}




int*  PrimaryVertexAnalyzer4PU::supf(std::vector<SimPart>& simtrks, const reco::TrackCollection & trks){
  // track rec to sim matching for hepMC simtrks
  unsigned int nsim=simtrks.size();
  unsigned int nrec=trks.size();
  int *rectosim=new int[nrec]; // pointer to associated simtrk
  for(unsigned int i0=0; i0<nrec; i0++) rectosim[i0]=-1;
  if(nsim==0) return rectosim;

  double** pij=new double*[nrec];     //  pij[nrec][nsim]
  double mu=100.; // initial chi^2 cut-off  (5 dofs !)
  int nmatch=0;
  unsigned int i=0;
  for(reco::TrackCollection::const_iterator t=trks.begin(); t!=trks.end(); ++t){
    pij[i]=new double[nsim];
    ParameterVector  par = t->parameters();
    reco::TrackBase::CovarianceMatrix S = t->covariance();
    S.Invert();
    for(unsigned int j=0; j<nsim; j++){
      simtrks[j].rec=-1;
      SimPart s=simtrks[j];
      double c=0;
      for(int k=0; k<5; k++){
        for(int l=0; l<5; l++){
          c+=(par(k)-s.par[k])*(par(l)-s.par[l])*S(k,l);
        }
      }
      //assert((i<nrec)&&(j<nsim));
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

  for(unsigned int k=0; k<nrec; k++){
    int imatch=-1; int jmatch=-1;
    double pmatch=0;
    for(unsigned int j=0; j<nsim; j++){
      if ((simtrks[j].rec)<0){
        double psum=exp(-0.5*mu); //cutoff
        for(unsigned int i=0; i<nrec; i++){
          if (rectosim[i]<0){ psum+=pij[i][j];}
        }
        for(unsigned int i=0; i<nrec; i++){
          if ((rectosim[i]<0)&&(pij[i][j]/psum>pmatch)){
            pmatch=pij[i][j]/psum;
            imatch=i; jmatch=j;
          }
        }
      }
    }// sim loop
    if((jmatch>=0)||(imatch>=0)){
    //if((jmatch>=0)&&(imatch>=0)){
     //std::cout << pmatch << "    " << pij[imatch][jmatch] << "  match=" <<
     //	match(simtrks[jmatch].par, trks.at(imatch).parameters()) <<std::endl;
      //assert((jmatch>=0)&&(imatch>=0)&&(static_cast<unsigned int>(jmatch)<nsim)&&(static_cast<unsigned int>(imatch)<nrec));
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

   std::cout << "simtracks without a matching rec track: " << std::endl;
   for(unsigned int j=0; j<nsim; j++){
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

  for(unsigned int i=0; i<nrec; i++){delete [] pij[i];}
  delete [] pij;
  return rectosim;  // caller must delete it !!! delete [] rectosim
} //supf








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
    Fill(h,"pthi_"+ttype,t.pt());
    if(fabs(t.eta())>2.0)  Fill(h,"ptfwd_"+ttype,t.pt());
    if(fabs(t.eta())<1.0) Fill(h,"ptcentral_"+ttype,t.pt());
    Fill(h,"found_"+ttype,t.found());
    Fill(h,"lost_"+ttype,t.lost());
    Fill(h,"nchi2_"+ttype,t.normalizedChi2());
    if (RECO_) Fill(h,"rstart_"+ttype,(t.innerPosition()).Rho());  // innerPosition need TrackExtra

    double d0Error=t.d0Error();
    double d0=t.dxy(vertexBeamSpot_.position());
    if (d0Error>0){ 
      Fill(h,"logtresxy_"+ttype,log(d0Error/0.0001)/log(10.));
      Fill(h,"tpullxy_"+ttype,d0/d0Error);
      Fill(h,"tpullxyvsz_"+ttype,t.vz(), pow(d0/d0Error,2));
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
      double phi=(tt.stateAtBeamLine().trackStateAtPCA()).momentum().phi();
      double dz2= pow(tt.track().dzError(),2)+(pow(wx_*cos(phi),2)+pow(wy_*sin(phi),2))/pow(tantheta,2);
	  //      double dz2= pow(tt.track().dzError(),2)+wxy2_/pow(tantheta,2);
      
      Fill(h,"restrkz_"+ttype,z-v->position().z());
      Fill(h,"restrkzvsphi_"+ttype,t.phi(), z-v->position().z());
      Fill(h,"restrkzvseta_"+ttype,t.eta(), z-v->position().z());
      Fill(h,"pulltrkzvsphi_"+ttype,t.phi(), (z-v->position().z())/sqrt(dz2));
      Fill(h,"restrkzvsz_"+ttype, v->position().z(), z-v->position().z());
      Fill(h,"pulltrkzvsz_"+ttype, v->position().z(), (z-v->position().z())/sqrt(dz2));
      Fill(h,"pulltrkzvseta_"+ttype,t.eta(), (z-v->position().z())/sqrt(dz2));
      Fill(h,"tpullzvsz_"+ttype,t.vz(), pow(z-v->position().z(),2)/(pow(v->zError(),2)+dz2));

      Fill(h,"pulltrkz_"+ttype,(z-v->position().z())/sqrt(pow(v->zError(),2)+dz2));


//       double x1=t.vx()-vertexBeamSpot_.x0(); double y1=t.vy()-vertexBeamSpot_.y0();
//       double kappa=-0.002998*fBfield_*t.qoverp()/cos(t.theta());
//       double D0=x1*sin(t.phi())-y1*cos(t.phi())-0.5*kappa*(x1*x1+y1*y1);
//       double q=sqrt(1.-2.*kappa*D0);
//       double s0=(x1*cos(t.phi())+y1*sin(t.phi()))/q; 
//       double s1=0;
//       if (fabs(kappa*s0)>0.001){
// 	s1=asin(kappa*s0)/kappa;
//       }else{
// 	double ks02=(kappa*s0)*(kappa*s0);
// 	s1=s0*(1.+ks02/6.+3./40.*ks02*ks02+5./112.*pow(ks02,3));
//       }

    }

    // collect some info on hits and clusters
    Fill(h,"nbarrelLayers_"+ttype,static_cast<double>(t.hitPattern().pixelBarrelLayersWithMeasurement()));
    Fill(h,"nPxLayers_"+ttype,static_cast<double>(t.hitPattern().pixelLayersWithMeasurement()));
    if(fabs(t.eta()<2)) Fill(h,"nPxLayersVsPt_"+ttype, t.pt(), static_cast<double>(t.hitPattern().pixelLayersWithMeasurement()));
    Fill(h,"nSiLayers_"+ttype,static_cast<double>(t.hitPattern().trackerLayersWithMeasurement()));
    Fill(h,"n3dLayers_"+ttype,static_cast<double>(t.hitPattern().numberOfValidStripLayersWithMonoAndStereo()));
    Fill(h,"expectedInner_"+ttype,static_cast<double>(t.trackerExpectedHitsInner().numberOfHits()));
    Fill(h,"expectedOuter_"+ttype,static_cast<double>(t.trackerExpectedHitsOuter().numberOfHits()));
    Fill(h,"trackAlgo_"+ttype,static_cast<double>(t.algo()));
    Fill(h,"trackQuality_"+ttype,static_cast<double>(t.qualityMask()));


    if(RECO_){
      //
      int longesthit=0, nbarrel=0;
      for(trackingRecHit_iterator hit=t.recHitsBegin(); hit!=t.recHitsEnd(); hit++){
	//bool valid=((**hit).getType()&0xf) ==0; // kludgde needed for reading 52 data with pre1?
	//if (valid  && (**hit).geographicalId().det() == DetId::Tracker ){
	if ((**hit).isValid()   && (**hit).geographicalId().det() == DetId::Tracker ){
	  bool barrel = (**hit).geographicalId().subdetId() == static_cast<int>(PixelSubdetector::PixelBarrel);
	  //bool barrel = DetId::DetId((**hit).geographicalId()).subdetId() == static_cast<int>(PixelSubdetector::PixelBarrel);
	  if (barrel){
	    const SiPixelRecHit *pixhit = dynamic_cast<const SiPixelRecHit*>( &(**hit));
	    edm::Ref<edmNew::DetSetVector<SiPixelCluster>, SiPixelCluster> const& clust = (*pixhit).cluster();
	    if (clust.isNonnull()) {
	      nbarrel++;
	      if (clust->sizeY()>longesthit) longesthit=clust->sizeY();
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
    }
    //-------------------------------------------------------------------
    
}


//void PrimaryVertexAnalyzer4PU::printRecVtxs(const Handle<reco::VertexCollection> recVtxs, std::string title){
void PrimaryVertexAnalyzer4PU::printRecVtxs(const reco::VertexCollection * recVtxs, std::string title){
    int ivtx=0;
    std::cout << std::endl << title << "  "<<recVtxs->size()<<"" << std::endl;
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
		<< " chi2 " << std::fixed << std::setw(5) << std::setprecision(1) << v->chi2() 
		<< " ndof " << std::fixed << std::setw(6) << std::setprecision(2) << v->ndof()
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

    //
    for(trackingRecHit_iterator hit=t->recHitsBegin(); hit!=t->recHitsEnd(); hit++){
      if ((**hit).isValid()   && (**hit).geographicalId().det() == DetId::Tracker ){
	bool barrel = (**hit).geographicalId().subdetId() == static_cast<int>(PixelSubdetector::PixelBarrel);
	bool endcap = (**hit).geographicalId().subdetId() == static_cast<int>(PixelSubdetector::PixelEndcap);
	//       bool barrel = DetId::DetId((**hit).geographicalId()).subdetId() == static_cast<int>(PixelSubdetector::PixelBarrel);
	//       bool endcap = DetId::DetId((**hit).geographicalId()).subdetId() == static_cast<int>(PixelSubdetector::PixelEndcap);
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
    if(tk1.stateAtBeamLine().isValid() && tk2.stateAtBeamLine().isValid()){
      return tk1.stateAtBeamLine().trackStateAtPCA().position().z() < tk2.stateAtBeamLine().trackStateAtPCA().position().z();
    }else{
      return false;
    }
  }

}




void PrimaryVertexAnalyzer4PU::printPVTrks(const Handle<reco::TrackCollection> &recTrks, 
					   //const Handle<reco::VertexCollection> &recVtxs,  
					   const reco::VertexCollection  * recVtxs,  
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
  int* rectosim=NULL;
  if(MC_) rectosim=supf(tsim, selRecTrks);



  // now dump in a format similar to the clusterizer
  cout << "printPVTrks " << run_ << " :  " << event_ << endl;
  cout << "----          z +/- dz     1bet3-l      ip +/-dip        pt   phi   eta";
  if((tsim.size()>0)||(simEvt.size()>0)) {cout << " type     pdg    zvtx    zdca      dca    zvtx   zdca    dsz";}
  cout << endl;

  cout.precision(4);
  int isel=0;
  double tz0=-10000;

  for(unsigned int i=0; i<selTrks.size(); i++){

//     if  (selectedOnly || (theTrackFilter(selTrks[i]))) {
//           cout <<  setw (4)<< isel;
// 	  isel++;
//     }else if (!selTrks[i].stateAtBeamLine().isValid()){
//       cout << "XXXX";
//     }else{
//       cout <<  "    ";
//     }


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
    double phi=(selTrks[i].stateAtBeamLine().trackStateAtPCA()).momentum().phi();
    double tdz2= pow(selTrks[i].track().dzError(),2)+ (pow(wx_*cos(phi),2)+pow(wy_*sin(phi),2))/pow(tantheta,2);

    // print vertices in between tracks
    int iiv=0;
    for(reco::VertexCollection::const_iterator v=recVtxs->begin(); 
	v!=recVtxs->end(); ++v){
      if ( (v->ndof()>0) && (v->chi2()>0)  && (v->position().z()<tz) && (v->z()>tz0 )){
	cout << "****[" << setw(2) << iiv << "]" 
	     << setw (8) << fixed << setprecision(4)<<  v->z() << " +/-" <<  setw (6)<< v->zError() 
	     << "  ndof=" << v->ndof() 
	     <<" " << endl;
      }
      iiv++;
    }

    // print MC vertices in between tracks
    if(simEvt.size()>0){
      for(unsigned int event=0; event<simEvt.size(); event++){
	if ((simEvt[event].z<tz) && (simEvt[event].z>tz0 )){
	  cout << "----[" << setw(2) << event << "]" 
	       << setw (8) << fixed << setprecision(4)<<  simEvt[event].z << "          "
	       << "  ntrk=" << simEvt[event].tk.size()
	       <<" " << endl;
	}
      }
    }

    tz0=tz;

    if  (selectedOnly || (theTrackFilter(selTrks[i]))) {
          cout <<  setw (4)<< isel;
	  isel++;
    }else if (!selTrks[i].stateAtBeamLine().isValid()){
      cout << "XXXX";
    }else{
      cout <<  "    ";
    }


    
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
    //cout  <<  setw (8) << fixed << setprecision(4)<<  tz << " +/-" <<  setw (6)<< sqrt(tdz2);
    reco::Track t0=selTrks[i].track();
    if(fabs(t0.vz())<100){
      cout  <<  setw (8) << fixed << setprecision(4)<<  tz << " +/-" <<  setw (6)<< sqrt(tdz2);
    }else{
      cout  <<  setw (8) << fixed << setprecision(4)<<  99.99*tz/fabs(tz) << " +/-" <<  setw (6)<< sqrt(tdz2);
    }
    
    // track quality and hit information, see DataFormats/TrackReco/interface/HitPattern.h
    if(selTrks[i].track().quality(reco::TrackBase::highPurity)){ cout << " *";}else{cout <<"  ";}
    if(selTrks[i].track().hitPattern().hasValidHitInFirstPixelBarrel()){cout <<"+";}else{cout << "-";}
    cout << setw(1) << selTrks[i].track().hitPattern().pixelBarrelLayersWithMeasurement();
    cout << setw(1) << selTrks[i].track().hitPattern().pixelEndcapLayersWithMeasurement(); 
    int mm=selTrks[i].track().hitPattern().trackerLayersWithMeasurement()-selTrks[i].track().hitPattern().pixelLayersWithMeasurement();
    if(mm>=0){
      cout << setw(1) << hex << mm <<dec;
    }else{   cout << "X";  }
    cout << setw(1) << hex << selTrks[i].track().hitPattern().numberOfValidStripLayersWithMonoAndStereo() <<dec;
    //cout << "-" << setw(1)<<hex <<selTrks[i].track().trackerExpectedHitsOuter().numberOfHits() << dec;
    cout << "-" << setw(1)<<hex <<selTrks[i].track().hitPattern().numberOfLostTrackerHits() << dec;

    
    Measurement1D IP=selTrks[i].stateAtBeamLine().transverseImpactParameter();
    cout << setw (8) << IP.value() << "+/-" << setw (6) << IP.error();
    //if(selTrks[i].track().ptError()<1){
    double erel=selTrks[i].track().ptError()/selTrks[i].track().pt();
    if(erel<0.1){
      cout << " " << setw(7) << setprecision(2)  << selTrks[i].track().pt()*selTrks[i].track().charge();
    }else if (erel<0.5){
      cout << " " << setw(6) << setprecision(1)  << selTrks[i].track().pt()*selTrks[i].track().charge() << "-";
    }else{
      cout << " " << setw(6) << setprecision(1)  << selTrks[i].track().pt()*selTrks[i].track().charge() << "*";
      //cout << "+/-" << setw(6)<< setprecision(2) << selTrks[i].track().ptError();
    }
    cout << " " << setw(5) << setprecision(2) << selTrks[i].track().phi()
	 << " " << setw(5) << setprecision(2) << selTrks[i].track().eta() ;



    // print MC info, if available
    if(MC_){
      if(simEvt.size()>0){
	reco::Track t=selTrks[i].track();
	if (z2tp_.find(t.vz())==z2tp_.end()){
	  cout << " not matched";
	}else{
	  try{
	    TrackingParticleRef tpr = z2tp_[t.vz()];
	    const TrackingVertex *parentVertex= tpr->parentVertex().get();
	    if(parentVertex==0){
	      cout << " null parent vertex ";
	    }else{
	      if(parentVertex->sourceTracks().size()==0){ cout << " prim" ;}else{cout << " sec ";}
	      cout << " " << setw(3) << tpr->eventId().event();
	      cout << " " << setw(5) << tpr->pdgId();
	      double vz=parentVertex->position().z();
	      cout << " " << setw(8) << setprecision(4) << vz;
	    }
	    cout << " " << setw(8) << setprecision(4) << 0; //zdcap
	    cout << " " << setw(8) << setprecision(4) << 0; //ddcap

	  }catch (...){
	    cout << " not matched1";
	  }
	}//
      }else{
	// no tracking particles
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
    }
    cout << endl;
  }
  cout  << "----------------"<<endl;
  if(MC_) delete [] rectosim;
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



// /********************************************************************************************************/
// Int_t PrimaryVertexAnalyzer4PU::getAssociatedRecoTrackIndex(const edm::Handle<reco::TrackCollection> &recTrks, TrackingParticleRef tpr ) 
// // returns the index into the reco track collection, shamelessly stolen from an unknown author
// /********************************************************************************************************/
// {
//     reco::TrackRef associatedRecoTrack;
//     int matchedTrackIndex = -1;     // -1 unmatched        

//     if(r2s_.find(tpr) != s2r_.end())
//     {
//         // get the associated recoTrack
//         associatedRecoTrack = s2r_[tpr].begin()->first;
    
//         // get the index of the associated recoTrack
//         for(reco::TrackCollection::size_type i = 0; i < recTrks->size(); ++i)
//         {
//             reco::TrackRef recoTrack(recTrks, i);

//             if(associatedRecoTrack == recoTrack)
//             {
//                 matchedTrackIndex = i;
//                 break;
//             }
//         } 
//     }
//     return matchedTrackIndex;
// }
// /********************************************************************************************************/



/********************************************************************************************************/
bool PrimaryVertexAnalyzer4PU::truthMatchedTrack( edm::RefToBase<reco::Track> track, TrackingParticleRef & tpr)

/********************************************************************************************************/
// for a reco track select the matching tracking particle, always use this function to make sure we
// are consistent
// after calling truthMachtedTrack, tpr may have changed its value
// to get the TrackingParticle form the TrackingParticleRef, use ->get();
{
  double f=-1e10;
  if(r2s_.find(track)==r2s_.end()){
    return false;
  }else{
    std::vector<std::pair<TrackingParticleRef, double> > tp = r2s_[track];
    //cout << tp.begin()->first.get()->eventId().event() <<  "  " << tp.size() << " " << tp.begin()->second << endl;
  }

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
  return (f>trackAssociatorMin_);
}
/********************************************************************************************************/






/********************************************************************************************************/
std::vector< edm::RefToBase<reco::Track> >  PrimaryVertexAnalyzer4PU::getTruthMatchedVertexTracks(
				       const reco::Vertex& v
				       )
// for rec vertex v get a list of tracks for which truth matching is available 
/********************************************************************************************************/
{
  std::vector<  edm::RefToBase<reco::Track> > b;
  TrackingParticleRef tpr;

  for(trackit_t tv=v.tracks_begin(); tv!=v.tracks_end(); tv++){
    edm::RefToBase<reco::Track> track=*tv;
    if (truthMatchedTrack(track, tpr)){
      b.push_back(*tv);
//       cout << "  test  "
// 	   << "  tv.key=" << tv->key() 
// 	   << "  track.key=" << track.key() 
// 	   << "  tpr.key=" << tpr.key() 
// 	   << "  trkidx2tp_[track.key()].key=" << trkidx2tp_[ track.key()].key() 
// 	   << endl;
    }
  }

  return b;
}
/********************************************************************************************************/


/********************************************************************************************************/
/********************************************************************************************************/







/********************************************************************************************************/
std::vector<PrimaryVertexAnalyzer4PU::SimEvent> PrimaryVertexAnalyzer4PU::getSimEvents
(
 edm::Handle<TrackingParticleCollection>  TPCollectionH,
 edm::Handle<TrackingVertexCollection>  TVCollectionH,
 edm::Handle<View<Track> > trackCollectionH
 ){

  const TrackingParticleCollection* simTracks = TPCollectionH.product();
  const View<Track>  tC = *(trackCollectionH.product());


  vector<SimEvent> simEvt;
  map<EncodedEventId, unsigned int> eventIdToEventMap;
  map<EncodedEventId, unsigned int>::iterator id;
  z2tp_.clear();
  bool dumpTP=false;
  for(TrackingParticleCollection::const_iterator it=simTracks->begin(); it!=simTracks->end(); it++){
    
    if( fabs(it->parentVertex().get()->position().z())>100.) continue; // skip funny entries @ -13900

    unsigned int event=0;  //note, this is no longer the same as eventId().event()
    id=eventIdToEventMap.find(it->eventId());
    // skip out of time pile-up, irrelevant for tracking
    if (it->eventId().bunchCrossing()!=0) continue;
    //
    if (id==eventIdToEventMap.end()){

      // new event here
      SimEvent e;
      e.type=1;  //full
      e.eventId=it->eventId();
      event=simEvt.size();
      e.key=event;
      e.nChTP=0;
      const TrackingVertex *parentVertex= it->parentVertex().get();
      if(DEBUG_){
	cout << "getSimEvents: ";
	cout << it->eventId().bunchCrossing() << "," <<  it->eventId().event() 
	     << " z="<< it->vz() << " " 
	     << parentVertex->eventId().bunchCrossing() << ","  <<parentVertex->eventId().event() 
	     << " z=" << parentVertex->position().z() 
	     << endl;
      }
      if (it->eventId()==parentVertex->eventId()){
	e.x=parentVertex->position().x();
	e.y=parentVertex->position().y();
	e.z=parentVertex->position().z();
      }else{
	e.x=it->vx();e.y=it->vy();e.z=it->vz();
      }
      simEvt.push_back(e);
      eventIdToEventMap[e.eventId]=event;
    }else{
      event=id->second;
    }
      

    simEvt[event].tp.push_back(&(*it));
    if( (abs(it->eta())<2.4) && (it->charge()!=0)  && (it->trackPSimHit(DetId::Tracker).size()>0) ){
      if(it->pt()>0.1){  simEvt[event].nChTP++; }
      simEvt[event].sumpt2+=pow(it->pt(),2); // should keep track of decays ?
      simEvt[event].sumpt+=it->pt(); 
    }
  }

  if(dumpTP){
    for(TrackingParticleCollection::const_iterator it=simTracks->begin(); it!=simTracks->end(); it++){
      std::cout << *it << std::endl;
    } 
  }

  trkidx2tp_.clear();
  trkidx2simevt_.clear();
  for(View<Track>::size_type i=0; i<tC.size(); ++i) {

    RefToBase<Track> track(trackCollectionH, i);
    TrackingParticleRef tpr;
    if( truthMatchedTrack(track,tpr)){

      if( eventIdToEventMap.find(tpr->eventId())==eventIdToEventMap.end() ){ cout << "Bug in getSimEvents" << endl; break; }
      z2tp_[track.get()->vz()]=tpr;
      trkidx2tp_[i]=tpr;


      unsigned int event=eventIdToEventMap[tpr->eventId()];
      simEvt[event].trkidx.push_back(i);
      trkidx2simevt_[i]=event;

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
      
    }else{
      //cout << "unmatched track " << i << endl;
      trkidx2simevt_[i]=666;
    }
  }


  
  //AdaptiveVertexFitter theFitter;
  for(unsigned int i=0; i<simEvt.size(); i++){

    if(simEvt[i].tkprim.size()>0){

      getTc(simEvt[i].tkprimsel, simEvt[i].Tc, simEvt[i].chisq, simEvt[i].dzmax, simEvt[i].dztrim, simEvt[i].m4m2);
      simEvt[i].zfit=-99;
    }else{
      simEvt[i].Tc=0; simEvt[i].chisq=0; simEvt[i].dzmax=0; simEvt[i].dztrim=0; simEvt[i].m4m2=0; simEvt[i].zfit=-99;
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
									  const Handle<SimVertexContainer> simVtxs,
									  const Handle<SimTrackContainer> simTrks
)
{
  if(DEBUG_){std::cout << "getSimPVs from simVtxs/simTrks " << std::endl;}

  std::vector<PrimaryVertexAnalyzer4PU::simPrimaryVertex> simpv;
  SimVertexContainer::const_iterator vsim=simVtxs->begin();
  {
    simPrimaryVertex sv(vsim->position().x()*simUnit_,vsim->position().y()*simUnit_, vsim->position().z()*simUnit_);
    sv.type=1;
    for(edm::SimTrackContainer::const_iterator t=simTrks->begin();  t!=simTrks->end(); ++t){
      int pdgCode=abs(t->type());
      if ((pdgCode==11)||(pdgCode==13)||(pdgCode==15)||(pdgCode==-211)||(pdgCode==-2212)||(pdgCode==-321)||(pdgCode==-3222)){
	//math::XYZTLorentzVectorD p(t->momentum().x(),t->momentum().y(),t->momentum().z(),t->momentum().e());
	 // must be a simpler way
	 //if((Q>0)&&(p.pt()>0.1)){
	if((t->momentum().Pt()>0.1)&&(fabs(t->momentum().Eta())<2.5)){
	   sv.nGenTrk++;
	 }
      }
    }
    simpv.push_back(sv);
  }

  return simpv;
}




std::vector<PrimaryVertexAnalyzer4PU::simPrimaryVertex> PrimaryVertexAnalyzer4PU::getSimPVs(
											    const Handle<reco::GenParticleCollection> genParticles
)
{
  if(DEBUG_){std::cout << "getSimPVs from genParticles " << std::endl;}

  std::vector<PrimaryVertexAnalyzer4PU::simPrimaryVertex> simpv;
  //double x=0,y=0,z=0,t=-100;
  double x=0,y=0,z=0;
  for(size_t i = 0; i < genParticles->size(); ++ i) {
    const GenParticle & p = (*genParticles)[i];
    int st = p.status();  
    if(st==1){ x=p.vx(); y=p.vy(); z=p.vz(); break;}
  }
  simPrimaryVertex sv(x,y,z);
  sv.type=1;
  for(size_t i = 0; i < genParticles->size(); ++ i) {
     const GenParticle & p = (*genParticles)[i];
     int pdgCode = abs(p.pdgId());
     int st = p.status();  
     if ((st==1)&&((pdgCode==11)||(pdgCode==13)||(pdgCode==15)||(pdgCode==211)||(pdgCode==2212)||(pdgCode==321)||(pdgCode==3222))){
       if((p.pt()>0.1)&&(fabs(p.eta())<2.5)){
	 sv.nGenTrk++;
       }
     }
   }
   simpv.push_back(sv);
   return simpv;
}





std::vector<PrimaryVertexAnalyzer4PU::simPrimaryVertex> PrimaryVertexAnalyzer4PU::getSimPVs(
				      const Handle<HepMCProduct> evtMC)
{
  if(DEBUG_){std::cout << "getSimPVs HepMC " << std::endl;}

  std::vector<PrimaryVertexAnalyzer4PU::simPrimaryVertex> simpv;
  const HepMC::GenEvent* evt=evtMC->GetEvent();
  if (evt) {

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
	sv.type=1;
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
	  simpv.push_back(sv);
	  vp=&simpv.back();
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
	      if ( (m.perp()>0.1) && (fabs(m.pseudoRapidity())<2.5) && isCharged( *daughter ) ){
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

  if(DEBUG_){std::cout << "getSimPVs from TrackingVertexCollection " << std::endl;}

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
    if (fabs(v->position().z())>1000) continue;  // skip funny junk vertices
    if ((v->position().t()<-20)||(v->position().t()>20)) continue;    // what is this? out of time pu?
    
    // could be a new vertex, check  all primaries found so far to avoid multiple entries
    const double mm=1.0; // for tracking vertices
    simPrimaryVertex sv(v->position().x()*mm,v->position().y()*mm,v->position().z()*mm);

    sv.eventId=v->eventId();
    sv.type=1;

    for (TrackingParticleRefVector::iterator iTrack = v->daughterTracks_begin(); iTrack != v->daughterTracks_end(); ++iTrack){
      sv.eventId=(**iTrack).eventId();  // an iterator of Refs, dereference twice
    }

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
      // Loop over daughter tracks
      for (TrackingVertex::tp_iterator iTP = v -> daughterTracks_begin(); iTP != v -> daughterTracks_end(); ++iTP) {
	//double pt=(*(*iTP)).momentum().perp2();
	const TrackingParticle & tp=(*(*iTP));
	if ( !(tp.charge()==0) && (fabs(tp.momentum().eta())<2.5)){
	  if (tp.momentum().perp2()>0.1*0.1){
	    sv.nGenTrk++;
	    sv.sumpT+=sqrt(tp.momentum().perp2());
	  }
	}
      }

      simpv.push_back(sv);
      vp=&simpv.back();
    }else{
      if(DEBUG_){std::cout << "this is not a new vertex"  << sv.x << " " << sv.y << " " << sv.z <<std::endl;}
    }



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



reco::VertexCollection * PrimaryVertexAnalyzer4PU::vertexFilter(Handle<reco::VertexCollection> pvs, bool filter){
  reco::VertexCollection * pv = new reco::VertexCollection;
  if(filter){
    // ptmin filter
    for(reco::VertexCollection::const_iterator  ipv = pvs->begin(); ipv != pvs->end(); ipv++) {
      double ptmin=0;
      for(trackit_t tv=ipv->tracks_begin(); tv!=ipv->tracks_end(); tv++){
	double pt=tv->get()->pt();
	if (pt>ptmin) { ptmin=pt;}
      }
      if(ptmin>0.5){
	pv->push_back(*ipv);
      }
    }
  }else{
    for(reco::VertexCollection::const_iterator  ipv = pvs->begin(); ipv != pvs->end(); ipv++ ) { pv->push_back(*ipv);}
  }
  return pv;
}
// reco::VertexCollection * PrimaryVertexAnalyzer4PU::vertexFilter(Handle<reco::VertexCollection> pvs, bool filter){
//   reco::VertexCollection * pv = new reco::VertexCollection;
//   if(filter){
//     // dynamic ndof filter
//     for(reco::VertexCollection::const_iterator  ipv1 = pvs->begin(); ipv1 != pvs->end(); ipv1++) {
//       double deltaz=1e10;
//       for( reco::VertexCollection::const_iterator  ipv2 = pvs->begin(); ipv2 != pvs->end(); ) {
//       if ((ipv2->ndof() > ipv1->ndof()) && (fabs(ipv2->position().z()-ipv1->position().z())<fabs(deltaz))){
// 	deltaz=ipv2->position().z()-ipv1->position().z();
//       }
//       }
//       if(ipv1->ndof()>fabs(2.0/deltaz)) pv->push_back(*ipv1);
//     }
//   }else{
//     for(reco::VertexCollection::const_iterator  ipv1 = pvs->begin(); ipv1 != pvs->end(); ipv1++ ) { pv->push_back(*ipv1);}
//   }
//   return pv;
// }


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
  if(sigmaZoverride_>0) sigmaZ_=sigmaZoverride_;
  MC_              = false;
  dumpThisEvent_   = false;
  



  if(verbose_){
    std::cout << endl 
	      << "PrimaryVertexAnalyzer4PU::analyze   event counter=" << eventcounter_
	      << " Run=" << run_ << "  LumiBlock " << luminosityBlock_ << "  event  " << event_
	      << " bx=" << bunchCrossing_ <<  " orbit=" << orbitNumber_ 
      //<< " selected = " << good
	      << std::endl;
  }

  if((bxFilter_>0) && (bxFilter_!=bunchCrossing_)) return;

   try{
    iSetup.getData(pdt_);
  }catch(const Exception&){
    std::cout << "Some problem occurred with the particle data table. This may not work !" <<std::endl;
  }

   nDigiPix_=0;
   if(RECO_){
     // snippet from Danek: RecoLocalTracker/SiPixelClusterizer/test/ReadPixClusters.cc
     edm::Handle< edmNew::DetSetVector<SiPixelCluster> > clusters;
     iEvent.getByLabel( "siPixelClusters" , clusters);
     
     const edmNew::DetSetVector<SiPixelCluster>& input = *clusters;
     edmNew::DetSetVector<SiPixelCluster>::const_iterator DSViter=input.begin();
     for ( ; DSViter != input.end() ; DSViter++) {
       //for (clustIt = DSViter->begin(); clustIt != DSViter->end(); clustIt++) { sumClusters++; }
       nDigiPix_+=DSViter->size();
     }
     
     Fill(hEvt,"nDigiPixvsL",instBXLumi_, float(nDigiPix_));
     Fill(hEvt,"nDigiPixvsLprof",instBXLumi_, float(nDigiPix_));
   }


   PileupSummaryInfo  puInfo;
   Handle<PileupSummaryInfo> puInfoH;
   bool bPuInfo=iEvent.getByLabel("addPileupInfo", puInfoH);
   if ( bPuInfo ){
     // pre 3_11/4_1_3
     puInfo=*puInfoH;
   }else{
    // new style (> 3_11_3/4_1_3), a vector
    Handle<std::vector< PileupSummaryInfo > >  vpuInfoH;
    bPuInfo=iEvent.getByLabel("addPileupInfo", vpuInfoH);
    if (bPuInfo){
      std::vector<PileupSummaryInfo>::const_iterator PVI;
      for(PVI = vpuInfoH->begin(); PVI != vpuInfoH->end(); ++PVI) {
	if(verbose_){
	  std::cout << " Pileup Information: bunchXing, nvtx: " << PVI->getBunchCrossing() << " " << PVI->getPU_NumInteractions() << 
	    "  nz=" << PVI->getPU_zpositions().size() << std::endl;
	}
	if (PVI->getBunchCrossing()==0){
	  bPuInfo=true;
	  puInfo=*PVI;
	}
      }
    }
  }
   if (bPuInfo && ((puInfo.getPU_zpositions().size()<nPUmin_) || (puInfo.getPU_zpositions().size()>nPUmax_))){
     return;
   }

  // genParticles for AOD et al:  https://twiki.cern.ch/twiki/bin/view/CMSPublic/WorkBookGenParticleCandidate
   Handle<GenParticleCollection> genParticles;
   bool bgenParticles=iEvent.getByLabel("genParticles", genParticles);


   for(std::vector<std::string>::const_iterator vCollection=vertexCollectionLabels_.begin(); vCollection!=vertexCollectionLabels_.end(); vCollection++){
     Handle<reco::VertexCollection> recVtxsH;
     if(iEvent.getByLabel(*vCollection, recVtxsH)){
       recVtxs_[*vCollection] = vertexFilter(recVtxsH, useVertexFilter_);
     }else{
       recVtxs_[*vCollection] = NULL;
       cout <<"collection " << *vCollection << " not found " << endl;
     }
   }



  
  Handle<reco::TrackCollection> recTrks;
  iEvent.getByLabel(recoTrackProducer_, recTrks);


  if(recTrks->size()==0){
    emptyeventcounter_++;
    if (emptyeventcounter_<100){
      cout << "Event without tracks skipped " <<  eventcounter_ << " " <<  event_ << endl;
    }
    if(emptyeventcounter_==100){ cout << "This is the last message of this kind" << endl;}
    return;
  }

  int nhighpurity=0, ntot=0;
  for(reco::TrackCollection::const_iterator t=recTrks->begin(); t!=recTrks->end(); ++t){  
    ntot++;
    if(t->quality(reco::TrackBase::highPurity)) nhighpurity++;
  } 


  

  // a quick look at loopers
  for(reco::TrackCollection::const_iterator t1=recTrks->begin(); t1!=recTrks->end(); ++t1){  
    if(t1->charge()<0) continue;
    if(t1->hitPattern().pixelLayersWithMeasurement()<2) continue;
    if(t1->hitPattern().trackerLayersWithMeasurement()<4) continue;
    if(t1->normalizedChi2()>10) continue;
    for(reco::TrackCollection::const_iterator t2=recTrks->begin(); t2!=recTrks->end(); ++t2){  
      if(t2->charge()>0) continue;
      if(t2->hitPattern().pixelLayersWithMeasurement()<2) continue;
      if(t2->hitPattern().trackerLayersWithMeasurement()<4) continue;
      if(t2->normalizedChi2()>10) continue;
      double dphi=t1->phi()-t2->phi(); if (dphi<0) dphi+=2*M_PI;
      double seta=t1->eta()+t2->eta();
      double sumdxy=t1->dxy()+t2->dxy();
      if(fabs(dphi-M_PI)<0.1) Fill(hTrk,"sumeta",seta);
      if(fabs(seta)<0.1) Fill(hTrk,"deltaphi",dphi-M_PI);
      if( (fabs(dphi-M_PI)<0.05) && (fabs(seta)<0.05) ){
	if((fabs(t1->vz()-t2->vz())<0.05)&&(fabs(sumdxy)<0.2)) Fill(hTrk,"ptloop",0.5*(t1->pt()+t2->pt()));
	if((fabs(t1->vz()-t2->vz())<0.2)&&(fabs(sumdxy)<0.2)) Fill(hTrk,"dptloop",t1->pt()-t2->pt());
	if((fabs(t1->pt()-t2->pt())<0.05)&&(fabs(sumdxy)<0.5)) Fill(hTrk,"zloop",0.5*(t1->vz()+t2->vz()));
	if((fabs(t1->pt()-t2->pt())<0.05)&&(fabs(sumdxy)<0.5)) Fill(hTrk,"dzloop",t1->vz()-t2->vz());
	Fill(hTrk,"sumdxyloop",sumdxy);
      }
      if( (fabs(seta)<0.04)
	  &&(fabs(t1->vz()-t2->vz())<0.2)
	  &&(fabs(t1->pt()-t2->pt())<0.04)
	  &&(fabs(t1->dxy()+t2->dxy())<0.3)){
	Fill(hTrk,"deltaphisel",dphi-M_PI);
      }
    }
  } 

  // and now a look at double tracks
  int n1=0;
  for(reco::TrackCollection::const_iterator t1=recTrks->begin(); t1!=recTrks->end(); ++t1){  
    n1++;
    if(t1->hitPattern().pixelLayersWithMeasurement()<2) continue;
    if(t1->hitPattern().trackerLayersWithMeasurement()<5) continue;
    if(t1->normalizedChi2()>10) continue;
    int n2=0;
    for(reco::TrackCollection::const_iterator t2=recTrks->begin(); t2!=recTrks->end(); ++t2){  
      n2++;
      if(n1==n2) continue;
      
      if(t2->hitPattern().pixelLayersWithMeasurement()<2) continue;
      if(t2->hitPattern().trackerLayersWithMeasurement()<5) continue;
      if(t2->normalizedChi2()>10) continue;

      Fill(hTrk,"dzall",t1->vz()-t2->vz());
      if((t2->charge()*t1->charge())<0) continue;
      double dphi=t1->phi()-t2->phi(); 
      if (dphi<-M_PI){dphi+=2*M_PI;}else if(dphi>M_PI){dphi-=2*M_PI;}
      //      double seta=t1->eta()+t2->eta();
      double deta=t1->eta()-t2->eta();
      if(fabs(dphi)<0.1) Fill(hTrk,"deta2",deta);
      if(fabs(deta)<0.1) Fill(hTrk,"deltaphi2",dphi);
      if( (fabs(dphi)<0.05) && (fabs(deta)<0.05) ){
	Fill(hTrk,"ptloop2",0.5*(t1->pt()+t2->pt()));
	Fill(hTrk,"dptloop2",t1->pt()-t2->pt());
	Fill(hTrk,"zloop2",0.5*(t1->vz()+t2->vz()));
	Fill(hTrk,"dzloop2",t1->vz()-t2->vz());
      }
    }
  }

  

  if(iEvent.getByType(recoBeamSpotHandle_)){
    vertexBeamSpot_= *recoBeamSpotHandle_;
    wxy2_=pow(vertexBeamSpot_.BeamWidthX(),2)+pow(vertexBeamSpot_.BeamWidthY(),2);
    wx_=vertexBeamSpot_.BeamWidthX();
    wy_=vertexBeamSpot_.BeamWidthY();
    if ((vertexBeamSpot_.sigmaZ()<9)&&(sigmaZoverride_==0)){sigmaZ_=vertexBeamSpot_.sigmaZ();}
    Fill(hsimPV, "xbeam",vertexBeamSpot_.x0()); Fill(hsimPV, "wxbeam",vertexBeamSpot_.BeamWidthX());
    Fill(hsimPV, "ybeam",vertexBeamSpot_.y0()); Fill(hsimPV, "wybeam",vertexBeamSpot_.BeamWidthY());
    Fill(hsimPV, "zbeam",vertexBeamSpot_.z0()); Fill(hsimPV, "sigmaZbeam",vertexBeamSpot_.sigmaZ());
    if(luminosityBlock_!=currentLS_){
      cout << "BEAM " << run_ << " : " << std::setw(10) << luminosityBlock_ << " " <<  std::setw(8) <<std::fixed << std::setprecision(4) << vertexBeamSpot_.x0() << ", " <<  vertexBeamSpot_.y0() <<  ", " <<  vertexBeamSpot_.z0() << "+/-" << vertexBeamSpot_.z0Error()<<  ",  wx=" <<  vertexBeamSpot_.BeamWidthX() << ",  wy=" <<   vertexBeamSpot_.BeamWidthY() <<  " , sigmaZ=" << vertexBeamSpot_.sigmaZ() << "+/-" << vertexBeamSpot_.sigmaZ0Error() << endl;
      currentLS_=luminosityBlock_;
    }
//     // temp, reject badly known beamspots
//     if (filterBeamError_ && ((vertexBeamSpot_.sigmaZ0Error()> 0.1) || (vertexBeamSpot_.z0Error()>0.1))) return;
//     return;

  }else{
    cout << " beamspot not found, using dummy " << endl;
    vertexBeamSpot_=reco::BeamSpot();
  }



  instBXLumi_= -1.;
  edm::Handle<LumiDetails> lumiDetails;
  if (iEvent.getLuminosityBlock().getByLabel("lumiProducer",lumiDetails)){
    instBXLumi_ = lumiDetails->lumiValue(LumiDetails::kOCC1,iEvent.bunchCrossing());
  }else{
    // cout << "no lumi ?" << endl;
  }
  


 
  // for the associator
  Handle<View<Track> > trackCollectionH;
  iEvent.getByLabel(recoTrackProducer_,trackCollectionH);

  Handle<HepMCProduct> evtMC;

  Handle<SimVertexContainer> simVtxs;
  bool bSimVtxs = iEvent.getByLabel( simG4_, simVtxs);
  
  Handle<SimTrackContainer> simTrks;
  bool bSimTrks = iEvent.getByLabel( simG4_, simTrks);





  edm::Handle<TrackingParticleCollection>  TPCollectionH ;
  edm::Handle<TrackingVertexCollection>    TVCollectionH ;
  bool gotTP=iEvent.getByLabel("mergedtruth","MergedTrackTruth",TPCollectionH);
  bool gotTV=iEvent.getByLabel("mergedtruth","MergedTrackTruth",TVCollectionH);
  MC_ |= gotTP | gotTV;
  trkidx2tp_.clear();
  z2tp_.clear();
  trkidx2simevt_.clear();

  iSetup.get<TransientTrackRecord>().get("TransientTrackBuilder",theB_);
  fBfield_=((*theB_).field()->inTesla(GlobalPoint(0.,0.,0.))).z();



  vector<SimEvent> simEvt;
  if (gotTP && gotTV && !(trackAssociatorLabel_=="None") ){

   // need hits (raw or fevt or recodebug ) for this to work
    edm::ESHandle<TrackAssociatorBase> theAssociator;
    iSetup.get<TrackAssociatorRecord>().get(trackAssociatorLabel_, theAssociator);
    associator_ = (TrackAssociatorBase *) theAssociator.product();
    r2s_ = associator_->associateRecoToSim (trackCollectionH, TPCollectionH, &iEvent ); 
    //s2r_ = associator_->associateSimToReco(trackCollectionH, TPCollectionH, &iEvent );
    simEvt=getSimEvents(TPCollectionH, TVCollectionH, trackCollectionH);
  }

  if(gotTV){

    // use Tracking Vertices

    MC_=true;
    if(verbose_){   cout << "Found Tracking Vertices " << endl;    }
    simpv=getSimPVs(TVCollectionH);
    

  }else if(iEvent.getByLabel(mcproduct,evtMC)){

    // fill simEvt from hepMC (and, if available, pileupinfosummary)
    MC_=true;
    simpv=getSimPVs(evtMC);

    if (bPuInfo) {
      //for(int i=0; i<puInfo.getPU_NumInteractions(); i++){
      for(unsigned int i=0; i<puInfo.getPU_zpositions().size(); i++){
	if(false){
  	cout << "pile-up " << i << ")"
  	     << " z= " << puInfo.getPU_zpositions()[i]
  	     << " nlo=" << puInfo.getPU_ntrks_lowpT()[i]
  	     << " nhi=" << puInfo.getPU_ntrks_highpT()[i] <<  endl;
	}

 	PrimaryVertexAnalyzer4PU::simPrimaryVertex v(vertexBeamSpot_.x0(),vertexBeamSpot_.y0(),puInfo.getPU_zpositions()[i]);
 	v.type=2;  // partial
 	// nlo cut-off is 0.1 GeV  (includes the high pt tracks)
 	// hi cut-off is 0.5 GeV
	v.nGenTrk=puInfo.getPU_ntrks_lowpT()[i];
	v.sumpT=puInfo.getPU_sumpT_lowpT()[i];
 	//v.eventId=puInfo.getPU_EventID()[i];
 	simpv.push_back(v);
      }
    }else{
      if(verbose_){cout << "no PU info found " << endl;}
      cout << "no PU info found !!!!!!!!!!!!!!!!" << endl;
    }

    if(verbose_){
      cout << "Using HepMCProduct " << endl;
      std::cout << "simtrks " << simTrks->size() << std::endl;
    }
    tsim = PrimaryVertexAnalyzer4PU::getSimTrkParameters(simTrks, simVtxs, simUnit_);
    
  }else if(bSimTrks&&bSimVtxs){
    simpv=getSimPVs(simVtxs, simTrks);
    tsim=getSimTrkParameters(simTrks, simVtxs, simUnit_);
    MC_=true;
  }else if(bgenParticles){
    simpv=getSimPVs(genParticles);
    tsim=getSimTrkParameters(genParticles);
    MC_=true;
    if(verbose_) {cout << "Signal vertex  z=" << simpv[0].z << "  n=" << simpv[0].nGenTrk << endl;}
    if (bPuInfo) {
      if(verbose_) {cout << "PileupSummaryInfo  nPU=" << puInfo.getPU_NumInteractions() << endl;}
      //for(int i=0; i<puInfo.getPU_NumInteractions(); i++){
      for(unsigned int i=0; i<puInfo.getPU_zpositions().size(); i++){
	cout << " i="<< i << endl;
	cout << " z= " << puInfo.getPU_zpositions()[i] << endl;
	cout << " nlo=" << puInfo.getPU_ntrks_lowpT()[i] << endl;
	cout << " nhi=" << puInfo.getPU_ntrks_highpT()[i] << endl;
 	PrimaryVertexAnalyzer4PU::simPrimaryVertex v(vertexBeamSpot_.x0(),vertexBeamSpot_.y0(),puInfo.getPU_zpositions()[i]);
 	v.type=2;  // partial
 	// nlo cut-off is 0.1 GeV  (includes the high pt tracks)
 	// hi cut-off is 0.5 GeV
	v.nGenTrk=puInfo.getPU_ntrks_lowpT()[i];
	v.sumpT=puInfo.getPU_sumpT_lowpT()[i];
 	//v.eventId=puInfo.getPU_EventID()[i];
 	simpv.push_back(v);
      }
    }


  }else{
    MC_=false;
    // if(verbose_({cout << "No MC info at all" << endl;}
  }



  


  hsimPV["nsimvtx"]->Fill(simpv.size());
  //cerr << "Timecheck> "<< event_ << " " << run_ <<" "   << simpv.size() << endl;
  for(std::vector<simPrimaryVertex>::iterator vsim=simpv.begin();
       vsim!=simpv.end(); vsim++){
    if(vsim->type==1){
      if(doMatching_ ){  
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
    }//type==1
  }



  // analyze the vertex collections
  for(std::vector<std::string>::const_iterator vCollection=vertexCollectionLabels_.begin(); 
      vCollection!=vertexCollectionLabels_.end(); vCollection++){

    if(recVtxs_[*vCollection] !=NULL){

      // set up the track-> vertex map
      trkidx2recvtx_.clear();
      unsigned int iv=0;
      for(reco::VertexCollection::const_iterator v=recVtxs_[*vCollection]->begin();v!=recVtxs_[*vCollection]->end(); ++v){
	for(trackit_t t=v->tracks_begin(); t!=v->tracks_end(); t++){ 
	  trkidx2recvtx_[t->key()]=iv;
	}
	iv++;
      }
      analyzeVertexCollection(histograms_[*vCollection], recVtxs_[*vCollection], recTrks, simpv, *vCollection);
      recvmatch_[*vCollection]=tpmatch(histograms_[*vCollection],recVtxs_[*vCollection], recTrks, simEvt, *vCollection);
      analyzeVertexCollectionTP(histograms_[*vCollection], recVtxs_[*vCollection], recTrks, simEvt, recvmatch_[*vCollection], *vCollection);
    }
  }

  if( (nCompareCollections_<0) || (eventcounter_< nCompareCollections_) ){
    compareCollections(simEvt, simpv );
  }

  // print summary info 
  if((dumpThisEvent_ && (dumpcounter_<ndump_)) ||(verbose_ && (eventcounter_<ndump_)) ||(autoDumpCounter_-->0)){
    cout << endl << "Event dump" << dumpcounter_ << endl
	 << "event counter=" << eventcounter_
	 << " Run=" << run_ << "  LumiBlock " << luminosityBlock_ << "  event  " << event_
	  << " bx=" << bunchCrossing_ <<  " orbit=" << orbitNumber_ 
	 << std::endl;
    dumpcounter_++;

    bool trksdumped=false;
    for(std::vector<std::string>::const_iterator vCollection=vertexCollectionLabels_.begin(); vCollection!=vertexCollectionLabels_.end(); vCollection++){
      if( recVtxs_[*vCollection] !=NULL ){
	printRecVtxs(recVtxs_[*vCollection]);
	if(! trksdumped){
	  printPVTrks(recTrks, recVtxs_[*vCollection], tsim, simEvt, false);
	  cout << "---" << endl;
	  trksdumped=true; // only dump one track list per event
	}
      }else{
	cout << "Vertex collection not found !!!  This should not happen!"<<endl;
      }
    }

    if (dumpcounter_<2){cout << "beamspot " << vertexBeamSpot_ << endl;}
  }


  // clean up
  for(std::vector<std::string>::const_iterator vCollection=vertexCollectionLabels_.begin(); vCollection!=vertexCollectionLabels_.end(); vCollection++){
    delete recVtxs_[*vCollection];
  }


  if(verbose_){
    std::cout << std::endl << " End of PrimaryVertexAnalyzer4PU " << std::endl;
  }
}


/***************************************************************************************/

// helper for z-sorting
namespace {
bool lt(const std::pair<double,unsigned int>& a,const std::pair<double,unsigned int>& b ){
  return a.first<b.first;
}
}

/***************************************************************************************/
void PrimaryVertexAnalyzer4PU::printEventSummary(std::map<std::string, TH1*> & h,
						 const reco::VertexCollection * recVtxs,
						 const edm::Handle<reco::TrackCollection> recTrks, 
						 std::vector<simPrimaryVertex> & simpv,
						 const string message){
  // make a readable summary using simpv (no TrackingParticles, use simparticles or genparticles etc)
  if (simpv.size()==0) return;
  vector< pair<double,unsigned int> >  zrecv;
  for(unsigned int idx=0; idx<recVtxs->size(); idx++){
    zrecv.push_back( make_pair(recVtxs->at(idx).z(),idx) );
  }
  stable_sort(zrecv.begin(),zrecv.end(),lt);

  // same for simulated vertices
  vector< pair<double,unsigned int> >  zsimv;
  for(unsigned int idx=0; idx<simpv.size(); idx++){
    zsimv.push_back(make_pair(simpv[idx].z, idx));
  }
  stable_sort(zsimv.begin(), zsimv.end(),lt);


  cout << "---------------------------" << endl;
  cout << "event counter = " << eventcounter_ << "   " << message << endl;
  cout << "run " << run_ << "    event " << event_ << endl;
  cout << "---------------------------" << endl;

  unsigned int idxrec=0;
  unsigned int idxsim=0;
  double zmatch=0.05;
  cout.precision(4);

  cout << "  rec " <<  "             "  <<  " sim " << endl;
  while((idxrec<recVtxs->size())||(idxsim<simpv.size())){

    string signal=" ";
    string tag=" ";
    if ((idxsim<simpv.size()) && (zsimv[idxsim].second==0)){
      signal="*";
    }
    if ((idxrec<recVtxs->size()) && (zrecv[idxrec].second==0)){
      tag="*";
    }

    double ndof=0;
    if(idxrec<recVtxs->size()){
      ndof=recVtxs->at(zrecv[idxrec].second).ndof();
    }

    if( (idxrec<recVtxs->size()) && (idxsim<simpv.size()) 
	&& (abs(zrecv[idxrec].first-zsimv[idxsim].first)<(zmatch+recVtxs->at(zrecv[idxrec].second).zError()))
	&& (((idxsim+1)==simpv.size())||(fabs(zrecv[idxrec].first-zsimv[idxsim].first)<fabs(zrecv[idxrec].first-zsimv[idxsim+1].first)))
	&& (((idxrec+1)==recVtxs->size())||(fabs(zrecv[idxrec].first-zsimv[idxsim].first)<fabs(zrecv[idxrec+1].first-zsimv[idxsim].first)))
	){
      cout <<  setw(8) <<  setprecision(4) << fixed << zrecv[idxrec].first << tag
	   <<"   <->    " <<setw(8) << fixed <<  zsimv[idxsim].first
	   << signal
	   << setw(4) <<  simpv[zsimv[idxsim].second].nGenTrk
      << " (ndof=" << fixed << setw(5)  << setprecision(1) << ndof  << ")" ;
      if(zsimv[idxsim].second==0){
	if(tag==" "){ 
	  cout << "  signal vertex not tagged" << endl;
	}else{
	  cout << "  signal vertex found and tagged" << endl;
	}
      }else{
	cout << endl;
      }


      idxrec++;
      idxsim++;

    }else if (   ((idxrec<recVtxs->size()) && (idxsim<simpv.size())&& (zrecv[idxrec].first<zsimv[idxsim].first))
	      || ((idxrec<recVtxs->size()) && (idxsim==simpv.size())) ){
      //if ((ndof>4)&&(zrecv[idxrec].first<-0.5)&&(zrecv[idxrec].first>-1.)){ cout << "FFFake " << message << endl; dumpThisEvent_=true;}
      //if (ndof>4){ dumpThisEvent_=true;}
      cout <<  setw(8) << setprecision(4) << fixed << zrecv[idxrec].first << tag
	   << "                      "
	   << "  (ndof=" << fixed << setw(5)  << setprecision(1) << ndof  << ")" 
	   << "   fake " << endl;
      idxrec++;

    }else if (    ((idxrec<recVtxs->size()) && (idxsim<simpv.size()) && (zrecv[idxrec].first>zsimv[idxsim].first))
	       || ((idxrec==recVtxs->size()) && (idxsim<simpv.size())) ){
      cout << "         " <<  "   <->    " <<  setw(8) << setprecision(4) << fixed << zsimv[idxsim].first 
	   << signal
	   << setw(4) <<  simpv[zsimv[idxsim].second].nGenTrk;
      if (simpv[zsimv[idxsim].second].nGenTrk>2){
	if(zsimv[idxsim].second==0){
	   cout << "                lost signal vertex" << endl;
	}else{
	   cout << "                lost PU" << endl;
	}
      }else{
	cout<< endl;
      }
      idxsim++;
    }else{
		    cout << "what else?" << endl;	
		    break;
    }
  }
}
/***************************************************************************************/




/***************************************************************************************/
void PrimaryVertexAnalyzer4PU::compareCollections(vector<SimEvent> & simEvt, std::vector<simPrimaryVertex> & simpv){
/***************************************************************************************
prints a side-by-side comparison of the selected vertex collections
simulated vertices are shown if they are available
***************************************************************************************/
  

  bool debug=false;

  unsigned int ncoll=vertexCollectionLabels_.size();
 
  cout << "----compareCollections------------------------" << endl;
  cout << "event counter = " << eventcounter_ << "   ";
  cout << "run " << run_ << "    event " << event_ << endl;
  cout << "----------------------------------------------" << endl;
  
  cout << setw(15) <<  "   simulated    ";

  
  for(unsigned int icol=0; icol<ncoll; icol++){
    string h=vertexCollectionLabels_[icol];
    h.resize(15);
    cout << " " << setw(18) << h ;
  }
  cout << endl;

  int differences=0;

  // build the table
  
  //typedef pair< double, vector< int > > row_t;
  vector< pair< double,  int  * > > row;  


  // if we don't have simEvt with truth matching (TrackingParticles)
  // put in sim vertices
  if(simEvt.size()==0){
    for(unsigned int idx=0; idx<simpv.size(); idx++){
      int * v = new int[ncoll+1];
      for(unsigned int j=0; j<ncoll+1; j++){v[j]=-1;}
      v[ncoll]=idx;
      row.push_back( make_pair(simpv[idx].z, v ) );
    }
  }

  // unmatched rec vertices from all collections
  for(unsigned int icol=0; icol<ncoll; icol++){
    reco::VertexCollection * recVtxs = recVtxs_[vertexCollectionLabels_[icol]];
    if (recVtxs){
      for(unsigned int idx=0; idx<recVtxs->size(); idx++){
	if(recvmatch_[vertexCollectionLabels_[icol]][idx].sim<0){
	  int * v = new int[ncoll+1];
	  for(unsigned int j=0; j<ncoll+1; j++){v[j]=-1;}
	  v[icol]=idx;
	  row.push_back( make_pair( recVtxs->at(idx).z(), v));
	}
      }
    }
  }


  if(row.size()>1){

    if(debug){
      cout << "dump    size="<< row.size() <<endl;
      for(unsigned int irow=0; irow<row.size(); irow++){
	cout << setw(2) << irow << ")";
	cout << setw(8)<< setprecision(4)  << fixed << row[irow].first;
	if(row[irow].second==NULL) continue;
	for(unsigned int i=0; i<ncoll+1; i++){
	  cout << setw(6) << row[irow].second[i];
	}
	cout << endl;
      }
      cout <<endl;
    }
    

    // join rows
    int join=0;
    while(join>=0){
      if(row.size()>1) {stable_sort(row.begin(), row.end());}
      
      if(debug){
	cout << "dump before joining  size="<< row.size() <<endl;
	for(unsigned int irow=0; irow<row.size(); irow++){
	  cout << setw(2) << irow << ")";
	  cout << setw(8)<< setprecision(4)  << fixed << row[irow].first;
	  if( !(row[irow].second==NULL)){
	    for(unsigned int i=0; i<ncoll+1; i++){
	      cout << setw(6) << row[irow].second[i];
	    }
	  }
	  cout << endl;
	}
	cout <<endl;
      }
      
      double dzmin=0.1;
      join=-1;
      for(unsigned int irow=0; irow<row.size()-1; irow++){
	if( (row[irow].second==NULL) || (row[irow+1].second==NULL) )continue;
	if( (row[irow+1].first-row[irow].first)<dzmin){
	  bool joinable=true;
	  for(unsigned int i=0; i<ncoll+1; i++){
	    if( (row[irow].second[i]>=0) && (row[irow+1].second[i]>=0) ) joinable=false;
	  }
	  if (joinable){
	    join=irow;
	    dzmin=fabs(row[irow+1].first-row[irow].first);
	  }
	}
      }
      
      if(join>=0){
	if(debug) cout << "joining " << join << endl;
	if( (row[join].second==NULL) || (row[join+1].second==NULL) ){cout << " bad join=" << join << endl;}
	if(join>=int(row.size())){ cout << " row pointers screwed up " << join << "   " << row.size() << endl;}
	//join
	for(unsigned int i=0; i<ncoll+1; i++){
	  if( (row[join].second[i]<0) && (row[join+1].second[i]>=0) ){ 
	    row[join].second[i] = row[join+1].second[i];
	    if(i==ncoll) row[join].first=row[join+1].first;
	  }
	}
	
	//row z
	if (row[join].second[ncoll]<0){
	  double zrow=0; int nv=0;
	  for(unsigned int i=0; i<ncoll; i++){
	    int iv=row[join].second[i];
	    if(iv>int(recVtxs_[vertexCollectionLabels_[i]]->size())) {cout << "illegal vertex index "<< iv << "    join=" << join << endl; }
	    if (iv>=0){
	      reco::VertexCollection * recVtxs = recVtxs_[vertexCollectionLabels_[i]];
	      zrow+=recVtxs->at(iv).z();
	      nv++;
	    }
	  }
	  if(nv>0){
	    row[join].first=zrow/nv;
	  }else{
	    // hmmm
	  }
	}
	//delete swallowed row
	if(debug) cout  << "deleting row " << join+1 << "  row size= " << row.size() <<  "  ncoll= " << ncoll << endl;
	delete [] row[join+1].second;
	row[join+1].second=NULL;
	row.erase(row.begin()+(join+1));

	if(debug){
	  cout << "dump after joining  "<< join << " with " << join+1 <<endl;
	  for(unsigned int irow=0; irow<row.size(); irow++){
	    cout << setw(2) << irow << ")";
	    cout << setw(8)<< setprecision(4)  << fixed << row[irow].first;
	    if( !(row[irow].second==NULL)){
	      for(unsigned int i=0; i<ncoll+1; i++){
		cout << setw(6) << row[irow].second[i];
	    }
	    }
	    cout << endl;
	  }
	  cout <<endl;
	}

      }
    }
    
    if(debug){
      cout << "dump after joining  size="<< row.size() <<endl;
      for(unsigned int irow=0; irow<row.size(); irow++){
	cout << setw(2) << irow << ")";
	cout << setw(8)<< setprecision(4)  << fixed << row[irow].first;
	if( !(row[irow].second==NULL)){
	  for(unsigned int i=0; i<ncoll+1; i++){
	    cout << setw(6) << row[irow].second[i];
	  }
	}
	cout << endl;
      }
      cout <<endl;
    }
    
  }// handle un-matched vertices and simpv's



  // fill in sim vertices and matched rec vertices
  unsigned int suppressed=0;
  for(unsigned int idx=0; idx<simEvt.size(); idx++){
    if(simEvt[idx].nChTP>0){
      int * v = new int[ncoll+1];
      for(unsigned int j=0; j<ncoll+1; j++){v[j]=-1;}
      for( unsigned int jcol=0; jcol<ncoll; jcol++){
	reco::VertexCollection * recVtxs = recVtxs_[vertexCollectionLabels_[jcol]];
	if ( !(recVtxs==NULL)){
	  int i=-1;
	  for(unsigned int j=0; j<recVtxs->size(); j++){
	    if(recvmatch_[vertexCollectionLabels_[jcol]][j].sim==int(idx)){
	      i=j;
	    }
	  }
	  v[jcol]=i;
	}
      }
      v[ncoll]=idx; // the sim vertex
      row.push_back(make_pair(simEvt[idx].z, v));
    }else{
      suppressed++;
    }
  }


  if(row.size()>1){  stable_sort(row.begin(), row.end()); }

  if(debug){
    cout << "final dump  size="<< row.size() <<endl;
    for(unsigned int irow=0; irow<row.size(); irow++){
      cout << setw(2) << irow << ")";
      cout << setw(8)<< setprecision(4)  << fixed << row[irow].first;
      if(!(row[irow].second==NULL)){
	for(unsigned int i=0; i<ncoll+1; i++){
	  cout << setw(6) << row[irow].second[i];
	}
      }
      cout << endl;
    }
    cout <<endl;
    cout << "done" << endl;
  }

  // readable dump
  for(unsigned int irow=0; irow<row.size(); irow++){
    
    if(row[irow].second==NULL) continue;

    double z=row[irow].first;
    //bool fake=false;

    int * v=row[irow].second;
    int idx0=v[ncoll];
    //    cout << idx0 << endl;

    if(idx0<0){
      // no sim vertex
      cout << "%                    ";
      //fake=true;
    }else{

      if(simEvt.size()>0){
	// sim vertex
	cout << fixed << setw(10) << setprecision(4) << z << " [" 
	     << setw(3) << simEvt[idx0].nChTP << ","
	     << setw(3) << simEvt[idx0].tk.size() << "]";
	if(idx0==0){ cout << "*" ;}else{ cout << " ";}
      }else{
	cout << fixed << setw(10) << setprecision(4) << z << " [" 
	     << setw(3) << simpv[idx0].nGenTrk << "]    ";
	if(idx0==0){ cout << "*" ;}else{ cout << " ";}
      }

    }

    // count collections that  have a rec vertex for this sim vertex (for reporting)
    unsigned int nrec=0;
    for( unsigned int jcol=0; jcol<ncoll; jcol++){
      if(v[jcol]>=0){nrec++;}
    }
    if((nrec>0)&&(nrec<ncoll)){ differences++;}


    for( unsigned int jcol=0; jcol<ncoll; jcol++){
      int idx=v[jcol];
      if(idx>=0){
	reco::VertexCollection * recVtxs = recVtxs_[vertexCollectionLabels_[jcol]];
	if (! (recVtxs==NULL)){
	  cout << fixed << setw(10) << setprecision(4) << recVtxs->at(idx).z()
	       << " (" << setw(5)<< setprecision(1) << recVtxs->at(idx).ndof()  << ")";
	}else{
	  cout << "                  ";
	}
      }else{
	if(idx0<0){
	  // no sim vertex, "fake not not found" (found by others)
	  cout << "       +          ";
	}else{
	  if(nrec==0){
		cout << "       -          ";
	  }else{
	    cout << "      ---         ";// missed one (found by others)
	  }
	}
      }
    }
    cout << endl;
  }


  for(unsigned int irow=0; irow<row.size(); irow++){
    
    if( !(row[irow].second==NULL)){
      delete [] row[irow].second;
      row[irow].second=NULL;
    }
  }


  if(differences>0){cout << "the collections differ,  " << differences << "  differences " << endl;}
  if(suppressed>0){cout << suppressed << "  sim vertices without tracker hits suppressed " << endl;}
  
}
/***************************************************************************************/






/***************************************************************************************/
void PrimaryVertexAnalyzer4PU::printEventSummary(std::map<std::string, TH1*> & h,
						 const reco::VertexCollection * recVtxs,
						 const edm::Handle<reco::TrackCollection> recTrks, 
						 vector<SimEvent> & simEvt,
						 std::vector<RSmatch>& recvmatch,
						 const string message){
  // make a readable summary of the vertex finding if the TrackingParticles are availabe
  if (simEvt.size()==0){return;}


  // sort vertices in z ... for nicer printout

  vector< pair<double,unsigned int> >  zrecv;
  for(unsigned int idx=0; idx<recVtxs->size(); idx++){
    if ( (recVtxs->at(idx).ndof()<-1) || (recVtxs->at(idx).chi2()<=0) ) continue;  // skip clusters 
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
  cout << "run " << run_ << "    event " << event_ << endl;
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
  // truthMatchedVertexTracks[irecvtx]=list of rec tracks that vertex for which we have truth matched simtracks
  // (not necessarily all from the same simvertex)
  map<unsigned int, int> truthMatchedVertexTracks; 

  for(vector< pair<double,unsigned int> >::iterator itrec=zrecv.begin(); itrec!=zrecv.end(); itrec++){
    truthMatchedVertexTracks[itrec->second]=getTruthMatchedVertexTracks(recVtxs->at(itrec->second)).size();
    cout << setw(7) << fixed << truthMatchedVertexTracks[itrec->second];
    if (itrec->second==0){cout << "*" ;}else{cout << " " ;}
  }
  cout << "   truth matched " << endl;

  cout << "sim ------- trk  prim ----" << endl;



  map<unsigned int, unsigned int> rvmatch; // reco vertex matched to sim vertex  (sim to rec)
  map<unsigned int, unsigned int> svmatch; // sim vertex matched to rec vertex  (rec to sim)
  map<unsigned int, double > nmatch;  // highest number of truth-matched tracks of ev found in a recvtx
  map<unsigned int, double > wnmatch;   // highest sum of weights of truth-matched tracks of ev found in a recvtx
  map<unsigned int, double > purity;  // highest purity of a rec vtx (i.e. highest number of tracks from the same simvtx)
  map<unsigned int, double > wpurity;  // same for the sum of weights

  for(vector< pair<double,unsigned int> >::iterator itrec=zrecv.begin(); itrec!=zrecv.end(); itrec++){
    svmatch[itrec->second]=10000;
    purity[itrec->second]=0.;
    wpurity[itrec->second]=0.;
  }

  for(vector< pair<double,unsigned int> >::iterator itsim=zsimv.begin(); itsim!=zsimv.end(); itsim++){
    // itsim->first = z of the simvx
    // itsim->second= index of the simvtx
    SimEvent* ev =&(simEvt[itsim->second]);
    rvmatch[itsim->second]=10000;

    cout.precision(4);
    if (itsim->second==0){
      cout << setw(8) << fixed << ev->z << ")*" << setw(5) << ev->tk.size() << setw(5) << ev->tkprim.size() << "  | ";
    }else{
      cout << setw(8) << fixed << ev->z << ") " << setw(5) << ev->tk.size() << setw(5) << ev->tkprim.size() << "  | ";
    }

    nmatch[itsim->second]=0;  // highest number of truth-matched tracks of ev found in a recvtx
    wnmatch[itsim->second]=0;  // highest sum of weights of truth-matched tracks of ev found in a recvtx
    double matchpurity=0;//,matchwpurity=0;//,matchpurity2=0;


    // compare this sim vertex to all recvertices:
    for(vector< pair<double,unsigned int> >::iterator itrec=zrecv.begin(); itrec!=zrecv.end(); itrec++){
      // itrec->first  = z coordinate of the recvtx
      // itrec->second = index of the recvtx
      unsigned int irecvtx=itrec->second;
      const reco::Vertex *v = &(recVtxs->at(irecvtx));

      // count tracks found in both, sim and rec
      double n=0,wt=0;
      for(vector<TransientTrack>::iterator te=ev->tk.begin(); te!=ev->tk.end(); te++){
	const reco::Track&  RTe=te->track();
	 for(trackit_t tv=v->tracks_begin(); tv!=v->tracks_end(); tv++){
	   const reco::Track & RTv=*(tv->get());  
	   if(RTe.vz()==RTv.vz()) {n++; wt+=v->trackWeight(*tv);}
	}
      }
      if(n>0){
	cout << setw(7) << int(n)<< " ";
      }else{
	cout << "        ";
      }

      // consider for matching if reasonably close in z
      double deltaz=fabs(v->z()-itsim->first);
      if( (deltaz<(5*v->zError()))&&(deltaz<0.5) ){
	// match by number of tracks
	if (n > nmatch[itsim->second]){
	  nmatch[itsim->second]=n;
	  rvmatch[itsim->second]=itrec->second;
	  //matchpurity2=matchpurity;
	    matchpurity=n/truthMatchedVertexTracks[itrec->second];
	    //matchwpurity=wt/truthMatchedVertexTracks[itrec->second];
	}
	
	if(n > purity[itrec->second]){
	  purity[itrec->second]=n;
	  svmatch[itrec->second]=itsim->second;
	}
      }


      // match by weight
      if(wt > wnmatch[itrec->second]){
	wnmatch[itrec->second]=wt;
      }

      if(wt > wpurity[itrec->second]){
	wpurity[itrec->second]=wt;
      }

    }// end of reco vertex loop

    cout << "  | " << setw(1) << ev->nwosmatch <<"," << setw(1) << ev->nntmatch;
    cout << " | ";
    if  (nmatch[itsim->second]>0 ){
      if(matchpurity>=0.5){
	cout << "found  ";
      }else if(matchpurity>=0.3){
	cout << "ugly   ";
      }else{
	cout << "merged ";
      }	
      //cout << "  max eff. = "  << setw(8) << nmatch[itsim->second]/ev->tk.size() << " p=" << matchpurity << " w=" << matchwpurity;
      cout <<  endl;
    }else{
      if(ev->tk.size()==0){
	cout  << "invisible" << endl;
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
    if (itrec->second==0){cout << "*" ;}else{cout << " " ;}  // flag the tagged vertex
  }
  cout << endl;

  // rec vertex classification: the good, the bad, and the ugly
  cout << "                     |   ";
  for(vector< pair<double,unsigned int> >::iterator itrec=zrecv.begin(); itrec!=zrecv.end(); itrec++){
    if ((svmatch[itrec->second]<1000) && (rvmatch[svmatch[itrec->second]]==itrec->second) ){
      if ( (purity[itrec->second]/truthMatchedVertexTracks[itrec->second])>=0.5){
	cout << "   ok   ";
      }else{
	cout << "  ugly  ";
      }
    }else{
      cout << "  junk  ";
    }
  }
  cout << endl;
  // wos matchting
  cout << "                     |  ";
  for(vector< pair<double,unsigned int> >::iterator itrec=zrecv.begin(); itrec!=zrecv.end(); itrec++){
    if (recvmatch[itrec->second].maxwos>0){
      cout << setw(7) << fixed << simEvt[recvmatch[itrec->second].wosmatch].z <<" " ;
    }else if(recvmatch[itrec->second].maxnt>0){
      cout << setw(7) << fixed << simEvt[recvmatch[itrec->second].ntmatch].z <<" " ;
    }else{
      cout << "   -    ";
    }
  }
  cout << endl;
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
      
      //if(ivassign==(int)rvmatch[itsim->second]){
      if((ivassign == ev->rec)&&(ev->matchQuality>0)){
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

	cout << "  track z=" << setw(8) << fixed  << RTe.vz() << "+/-" << RTe.dzError() 
	     << "  pt=" <<  setw(6) << fixed<< setprecision(2) << RTe.pt()
	     << "  eta=" << setw(6) << fixed << setprecision(2) << RTe.eta();
	Measurement1D IP=te->stateAtBeamLine().transverseImpactParameter();
	cout << " ipsig=" << setw (8) << setprecision(4) << fixed << IP.value() << "+/-" << setw (6) << setprecision(4) << fixed << IP.error();
	// cout << " sel=" <<theTrackFilter(*te);

	//
	if(z2tp_.find(te->track().vz())==z2tp_.end()){
	  cout << " unmatched";
	}else{
	  TrackingParticleRef tpr = z2tp_[te->track().vz()];
	  double zparent=tpr->parentVertex().get()->position().z();
	  unsigned int nparent=tpr->parentVertex().get()->sourceTracks().size();
	  if(zparent==ev->z) {
	    cout << " prim"<< setw(3) << nparent;
	  }else{
	    cout << " sec "<< setw(3) << nparent;
	  }
	  cout << "  id=" << setw(5) << tpr->pdgId();
// 	}else{
	}
	cout << endl;

	//
      }
    }// next simvertex-track

  }//next simvertex

  cout << "---------------------------" << endl;

}
/***************************************************************************************/





/***************************************************************************************/
std::vector<PrimaryVertexAnalyzer4PU::RSmatch>  PrimaryVertexAnalyzer4PU::tpmatch(std::map<std::string, TH1*> & h,
	     const reco::VertexCollection * recVtxs,
	     const edm::Handle<reco::TrackCollection> recTrks, 
	     std::vector<SimEvent> & simEvt,
	     const std::string message){
/***************************************************************************************/

  std::vector<RSmatch> recvmatch; // vector of RSmatch objects, parallel to recvertices
  recvmatch.clear(); 
  // clear old sim->rec pointers, too
  for(vector<SimEvent>::iterator ev=simEvt.begin(); ev!=simEvt.end(); ev++){
    ev->wos.clear();
    ev->recvnt.clear();
    ev->rec=-1;
    ev->matchQuality=0;
    ev->nwosmatch=0;
    ev->nntmatch=0;
  }


  for(unsigned int iv=0; iv<recVtxs->size(); iv++){
    const reco::Vertex * v= &(recVtxs->at(iv));
    RSmatch M;
    
    unsigned int iev=0;
    for(vector<SimEvent>::iterator ev=simEvt.begin(); ev!=simEvt.end(); ev++){

      double evwos=0;       // wos of simEvt ev in the current recvtx
      unsigned int evnt=0;  // number of tracks from simEvt ev in the current recvtx
      for(trackit_t tv=v->tracks_begin(); tv!=v->tracks_end(); tv++){
	const reco::Track & RTv=*(tv->get());  
	if (trkidx2simevt_[tv->key()]==iev){
	  double tantheta=tan(RTv.theta());
	  double phi=RTv.phi();
	  double dz2=pow(RTv.dzError(),2)
	    +(pow(wx_*cos(phi),2)+pow(wy_*sin(phi),2))/pow(tantheta,2);
	  double wos=v->trackWeight(*tv)/dz2;
	  M.addTrack(iev, wos);    // fill track(wos)  rec vtx <- sim vtx
	  ev->addTrack(iv, wos);   // fill track(wos)  sim vtx -> rec vtx
	  evwos+=wos;
	  evnt++;
	}
      }     
      
      // require 2 tracks for a wos-match
      if( (evwos>0) && (evwos>M.maxwos) && (evnt>1) ){
	M.wosmatch=iev;
	M.maxwos=evwos;
      }

      if( (evnt>0) && (evnt>M.maxnt) ){
	M.ntmatch=iev;
	M.maxnt=evnt;
      }

      iev++;
    }


    // now wosmatch/maxwos holds the index of the dominant simvertex for this recvtx
    if(M.maxwos>0){
      simEvt.at(M.wosmatch).nwosmatch++;
      //Fill(h,"wosfrac",maxwos/sumwos);
    }else{
      //Fill(h,"wosfrac",0.);  // found no sim event whatsoever
    }

    if(M.maxnt>0){
      simEvt.at(M.ntmatch).nntmatch++;
    }


 
    recvmatch.push_back(M);
  }// end of recvertex loop


  // matching, start with the unambigous
  for(unsigned int iv=0; iv<recVtxs->size(); iv++){
    if ((recvmatch[iv].sim<0)&&(recvmatch[iv].maxwos>0)){
      unsigned int cand=recvmatch[iv].wosmatch;
      if((simEvt.at(cand).rec<0)&&(simEvt.at(cand).nwosmatch==1)
	 &&(fabs(simEvt.at(cand).z-recVtxs->at(iv).z())<1.0 ))  {  // reject insanely misreconstructed  vertices
	recvmatch[iv].matchQuality=1;
	recvmatch[iv].sim=cand;
	simEvt.at(cand).rec=iv;
	simEvt.at(cand).matchQuality=1;
      }
    }
  }


  // when two or more rec vertices are dominated by one sim vertex, assign the rec vertex with more tracks (from that sim vertex) to the sim
  for(unsigned int iev=0; iev<simEvt.size(); iev++){
    if((simEvt.at(iev).rec<0)&&(simEvt.at(iev).nwosmatch>1)){
      unsigned int nt=0, iv=0;
      for(map<unsigned int,unsigned int>::iterator rv=simEvt.at(iev).recvnt.begin(); rv!=simEvt.at(iev).recvnt.end(); rv++){
	if( ((*rv).second > nt) && (fabs(simEvt.at(iev).z-recVtxs->at((*rv).first).z())<1.0 )) {
	  nt=(*rv).second;
	  iv=(*rv).first;
	}
      }
      if(nt>0){
	recvmatch[iv].matchQuality=2;
	recvmatch[iv].sim=iev;
	simEvt.at(iev).rec=iv;
	simEvt.at(iev).matchQuality=2;
      }else{
	cout << "BUG" << endl;
      }
    }
  }


  // second chance, track counting match
  for(unsigned int iv=0; iv<recVtxs->size(); iv++){
    if ((recvmatch[iv].sim<0)&&(recvmatch[iv].maxnt>0)){
      unsigned int cand=recvmatch[iv].ntmatch;
      if((simEvt.at(cand).rec<0)&&(simEvt.at(cand).nntmatch==1)
	 &&(fabs(simEvt.at(cand).z-recVtxs->at(iv).z())<1.0 ))  {  // reject insanely misreconstructed  vertices
	recvmatch[iv].matchQuality=5;
	recvmatch[iv].sim=cand;
	simEvt.at(cand).rec=iv;
	simEvt.at(cand).matchQuality=5;
      }
    }
  }


  // when two or more rec vertices are dominated by one sim vertex, assign the rec vertex with more tracks (from that sim vertex) to the sim
  for(unsigned int iev=0; iev<simEvt.size(); iev++){
    if((simEvt.at(iev).rec<0)&&(simEvt.at(iev).nntmatch>1)){
      unsigned int nt=0, iv=0;
      for(map<unsigned int,unsigned int>::iterator rv=simEvt.at(iev).recvnt.begin(); rv!=simEvt.at(iev).recvnt.end(); rv++){
	if( (*rv).second > nt)  {
	  nt=(*rv).second;
	  iv=(*rv).first;
	}
      }
      if(nt>0){
	recvmatch[iv].matchQuality=6;
	recvmatch[iv].sim=iev;
	simEvt.at(iev).rec=iv;
	simEvt.at(iev).matchQuality=6;
      }else{
	cout << "BUG" << endl;
      }
    }
  }


//   // second chance, track counting match
//   for(unsigned int iv=0; iv<recVtxs->size(); iv++){
//     if ((recvmatch[iv].sim<0)&&(recvmatch[iv].maxnt>0)){
//       unsigned int cand=recvmatch[iv].ntmatch;
//       if((simEvt.at(cand).rec<0)&&(simEvt.at(cand).nntmatch==1)){
// 	recvmatch[iv].matchQuality=5;
// 	recvmatch[iv].sim=cand;
// 	simEvt.at(cand).rec=iv;
// 	simEvt.at(cand).matchQuality=5;
//       }else if((simEvt.at(cand).rec<0)&&(simEvt.at(cand).nntmatch>1)){
//  	// two rec vertices are dominated by one sim vertex, assign the one with more tracks
// 	for(unsigned int jv=0; jv<recVtxs->size(); jv++){
// 	  if((recvmatch[jv].sim<0)&&(recvmatch[jv].maxnt>0)&&(recvmatch[jv].ntmatch==cand)){
// 	    if((recvmatch[jv].maxnt>recvmatch[iv].maxnt)){ 
// 	      recvmatch[jv].matchQuality=6;
// 	      recvmatch[iv].matchQuality=99;
// 	      recvmatch[jv].sim=cand;
// 	      simEvt.at(cand).rec=iv;
// 	      simEvt.at(cand).matchQuality=6;
// 	    }else{
// 	      recvmatch[iv].matchQuality=6;
// 	      recvmatch[jv].matchQuality=99;
// 	      recvmatch[iv].sim=cand;
// 	      simEvt.at(cand).rec=iv;
// 	      simEvt.at(cand).matchQuality=6;
//  	    }
// 	  }
// 	}
//       }
//     }
//   }




  // last chance for vertices that are not matched by tracks
  // but are reasonably close
  for(unsigned int iev=0; iev<simEvt.size(); iev++){
    if( simEvt[iev].rec<0 ){
      for(unsigned int iv=0; iv<recVtxs->size(); iv++){
	if (recvmatch[iv].sim<0){
	  if( ( fabs(simEvt.at(iev).z-recVtxs->at(iv).z())<0.1 )
	      ||( fabs(simEvt.at(iev).z-recVtxs->at(iv).z())< 5*recVtxs->at(iv).zError() )
	      ){
	      recvmatch[iv].matchQuality=10;
	      recvmatch[iv].sim=iev;
	      simEvt.at(iev).rec=iv;
	      simEvt.at(iev).matchQuality=10;
	  }
	}
      }
    }
  }


  
  return recvmatch;
}
/***************************************************************************************/







/***************************************************************************************/
void PrimaryVertexAnalyzer4PU::analyzeVertexCollectionTP(std::map<std::string, TH1*> & h,
							 //const edm::Handle<reco::VertexCollection> recVtxs,
			       const reco::VertexCollection * recVtxs,
			       const edm::Handle<reco::TrackCollection> recTrks, 
			       vector<SimEvent> & simEvt,
			       std::vector<RSmatch> & recvmatch,				 
			       const string message){

  
  if(simEvt.size()==0)return;

  EncodedEventId iSignal=simEvt[0].eventId;
  Fill(h,"npu0",simEvt.size());




  if(eventSummaryCounter_++ < nEventSummary_){
    printEventSummary(h, recVtxs,recTrks,simEvt, recvmatch, message);
  }



  //
  unsigned int ntpfake=0;
  unsigned int ntpfake4=0;
  unsigned int ntpfound=0, ntpfound4=0;
  for(unsigned int iv=0; iv<recVtxs->size(); iv++){
    unsigned int q=recvmatch[iv].matchQuality;
    double z=recVtxs->at(iv).z();
    double bx=vertexBeamSpot_.x(z);
    double by=vertexBeamSpot_.y(z);
    double vxx=recVtxs->at(iv).covariance(1,1)+pow(vertexBeamSpot_.BeamWidthX(),2);
    double vyy=recVtxs->at(iv).covariance(2,2)+pow(vertexBeamSpot_.BeamWidthY(),2);;
    double vxy=recVtxs->at(iv).covariance(1,2);
    double dx=recVtxs->at(iv).x()-bx;
    double dy=recVtxs->at(iv).y()-by;
    double D=vxx*vyy-vxy*vxy;
    double c2xy=pow(dx,2)*vyy/D +  pow(dy,2)*vxx/D -2*dx*dy*vxy/D;
    Fill(h, "vtxcxy2", c2xy);
    if((q==0)||(q==99)){
      ntpfake++;
      if(recVtxs->at(iv).ndof()>4){ ntpfake4++;}
      Fill(h, "vtxcxy2Fake", c2xy);
      Fill(h, "ndofFake", recVtxs->at(iv).ndof());
            std::cout 
	      << "fake "<<std::setw(2) << q << " "
	      << "vtx "<< std::setw(3) << std::setfill(' ')<<iv
	      << " #trk " << std::fixed << std::setprecision(4) << std::setw(3) << recVtxs->at(iv).tracksSize() 
	      << " chi2 " << std::fixed << std::setw(5) << std::setprecision(1) << recVtxs->at(iv).chi2() 
	      << " ndof " << std::fixed << std::setw(6) << std::setprecision(2) << recVtxs->at(iv).ndof()
	      << " x "  << std::setw(8) <<std::fixed << std::setprecision(4) << recVtxs->at(iv).x() 
	      << " dx " << std::setw(8) << recVtxs->at(iv).xError()// <<  std::endl 
	      << " y "  << std::setw(8) << recVtxs->at(iv).y() 
	      << " dy " << std::setw(8) << recVtxs->at(iv).yError()//<< std::endl
	      << " z "  << std::setw(8) << recVtxs->at(iv).z() 
	      << " dz " << std::setw(8) << recVtxs->at(iv).zError()
	      << " r  " << std::setw(8) << sqrt(dx*dx+dy*dy)
	      << " c2 " << std::setw(8) << c2xy
	      << std::endl;
    }else{
      Fill(h, "vtxcxy2Matched", c2xy);
      ntpfound++;
      if(recVtxs->at(iv).ndof()>4){ ntpfound4++;}
    }

    for(unsigned int jv=0; jv<recVtxs->size(); jv++){
      if(!(jv==iv)){
	unsigned int q2=recvmatch[jv].matchQuality;
	if((q2>0)&&(q2<99)&&(q2>0)&&(q2<99)){
	  Fill(h, "zdiffrec4found", recVtxs->at(iv).z()- recVtxs->at(jv).z());
	}else{
	  Fill(h, "zdiffrec4fake", recVtxs->at(iv).z()- recVtxs->at(jv).z());
	}
      }
    }

  }
  Fill(h, "ntpfake", float(ntpfake));
  Fill(h, "ntpfake4", float(ntpfake4));
  Fill(h, "ntpfound", float(ntpfound));
  Fill(h, "ntpfound4", float(ntpfound4));

  for(unsigned int iev=0; iev<simEvt.size(); iev++){
    float w=0.;
    if(simEvt.at(iev).rec>=0){
      w=1.;
    }
    Fill(h, "effvsnrectp", float(simEvt.at(iev).tk.size()), w);
    Fill(h, "effvsngentp", float(simEvt.at(iev).nChTP), w);
    if((w==0)&&(simEvt.at(iev).tk.size()>20)){
      cout << "Houston " << iev << "  z=" << simEvt.at(iev).z << " ntk=" << simEvt.at(iev).tk.size() << "  nChTP=" <<  simEvt.at(iev).tk.size();
      cout << message << endl;
	dumpThisEvent_=true;
    }
    if(iev==0){
      Fill(h, "effsigvsnrectp", float(simEvt.at(iev).tk.size()), w);
      Fill(h, "effsigvsngentp", float(simEvt.at(iev).nChTP), w);
    }
  }


  for(vector<SimEvent>::iterator ev=simEvt.begin(); ev!=simEvt.end(); ev++){
    Fill(h, "nwosmatch", float(ev->nwosmatch));
  }

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
  int nrecvtxs4=0;
  int nrecndof[10]={0,0,0,0,0,0,0,0,0,0};
  for(reco::VertexCollection::const_iterator v=recVtxs->begin(); 
      v!=recVtxs->end(); ++v){
    if ( (v->isFake()) || (v->ndof()<-1) || (v->chi2()<=0) ) continue;  // skip clusters 
    nrecvtxs++;
    if(v->ndof()>4) nrecvtxs4++;
    for(int ndof=0; ndof<10; ndof++){
      if (v->ndof()>ndof) nrecndof[ndof]++;
    }
    
  }


  /*****************  this code is obsolete *************************
      use recvmatch[iv]    to find the sim vertex matched to a rec vertex
      use simEvt.rec       to find the rec vertex matched to a sim vertex
  */
 
  // --------------------------------------- fill the track assignment matrix ----------------------
  for(vector<SimEvent>::iterator ev=simEvt.begin(); ev!=simEvt.end(); ev++){
    ev->ntInRecVz.clear();  // just in case
    ev->zmatchn=-99.;       // z-position of the matched vertex (vertex with the highest number of tracks from this event)
    ev->zmatchn2=-99.;      // z-position of the 2nd matched vertex (vertex with the 2nd highest...)
    ev->zmatchw=-99.;       // z-position of the matched vertex (vertex with the highest sum of weights from this event)
    ev->nmatch=0;           // the highest number of tracks from this event found in a single rec vertex
    ev->nmatch2=0;          // the 2nd highest number of tracks from this event found in a single rec vertex
    ev->wmatch=0;           // the highest sum of weights of tracks from this event found in a single rec vertex
    ev->pmatchn=0;         
    ev->pmatchn2=0;
    

    //unsigned int recvidx=0;
    for(reco::VertexCollection::const_iterator v=recVtxs->begin(); 
	v!=recVtxs->end(); ++v){
      double n=0, wt=0;
      for(vector<TransientTrack>::iterator te=ev->tk.begin(); te!=ev->tk.end(); te++){
	const reco::Track&  RTe=te->track();
	for(trackit_t tv=v->tracks_begin(); tv!=v->tracks_end(); tv++){
	  const reco::Track & RTv=*(tv->get());  
	  if(RTe.vz()==RTv.vz()){ n++; }
	}
      }
      
      ev->ntInRecVz[v->z()]=n;


      if (n  > ev->nmatch){ 
	ev->nmatch2=ev->nmatch; ev->zmatchn2=ev->zmatchn; ev->pmatchn2=ev->pmatchn;
	ev->nmatch=n ; ev->zmatchn=v->z(); ev->pmatchn=n/v->tracksSize();
      }else if(n  > ev->nmatch2){ 
	ev->nmatch2=n ; ev->zmatchn2=v->z(); ev->pmatchn2=n/v->tracksSize();
      }
      if (wt > ev->wmatch){ ev->wmatch=wt; ev->zmatchw=v->z(); }
    }
  }
  
  

  // call it a vertex a fake if for every sim vertex there is another recvertex containing more tracks
  // from that sim vertex than the current recvertex
  double nfake=0;
  double nfake4=0;
  double nfakew4=0;
  for(reco::VertexCollection::const_iterator v=recVtxs->begin(); 
      v!=recVtxs->end(); ++v){
    bool matchedn=false;
    bool matchedw=false;
    for(vector<SimEvent>::iterator ev=simEvt.begin(); ev!=simEvt.end(); ev++){
      if ((ev->nmatch>0)&&(ev->zmatchn==v->z())){ matchedn=true; }//zmatchn=ev->z; }
      if ((ev->wmatch>0)&&(ev->zmatchw==v->z())){ matchedw=true; }//zmatchw=ev->z; }
    }
    // distance to nearest recvtx
    double deltaz=1e10;
    for(reco::VertexCollection::const_iterator v1=recVtxs->begin();v1!=recVtxs->end(); ++v1){
      if(v->z()==v1->z()) continue;
      if( !(v->z()==v1->z()) && (fabs(v1->z()-v->z())<fabs(deltaz)) ){ deltaz=v1->z()-v->z();}
    }

    if(!matchedw && !v->isFake()) {
      if( v->ndof()>4) {nfakew4++;}
    }

    if(!matchedn && !v->isFake()) {
      // note: sometimes fakes are not really fakes because the "main" vertex was swallowed by a neighbor
      nfake++;
      if( v->ndof()>4) {nfake4++;}
      cout << " fake rec vertex at z=" <<  setprecision(4) << v->z() << "+/-" << v->zError() << setprecision(2)  <<  "   chi2 ="<< v->chi2() << "   ndof="<< v->ndof() << " w=";
      for(trackit_t t=v->tracks_begin(); t!=v->tracks_end(); t++){ 
	cout << setw(4) << setprecision(2) << v->trackWeight(*t) << " ";
	fillTrackHistos(h,"unmatchedVtx",*(t->get()),&(*v));
      }
      cout << " " << message << endl;

      // some histograms of fake vertex properties here

      Fill(h,"unmatchedVtxZ",v->z());
      Fill(h,"unmatchedVtxNdof",v->ndof());
      if(fabs(deltaz)<100){
	if(fabs(deltaz)>1) Fill(h,"unmatchedVtxNdof1",v->ndof());
	if(fabs(deltaz)>2) Fill(h,"unmatchedVtxNdof2",v->ndof());
	Fill(h,"unmatchedVtxDeltaZ",deltaz);
      }
      if(fabs(deltaz)>15){
	cout << "fake vertex deltaz="<<deltaz << " " << message << endl;
	dumpThisEvent_=true;
      }
      Fill(h,"unmatchedVtxNtrk",v->tracksSize());
      if(v->tracksSize()==2){  Fill(h,"unmatchedVtx2trkchi2vsndof",v->ndof(),v->chi2());   }
      if(v->tracksSize()==3){  Fill(h,"unmatchedVtx3trkchi2vsndof",v->ndof(),v->chi2());   }
      if(v->tracksSize()==4){  Fill(h,"unmatchedVtx4trkchi2vsndof",v->ndof(),v->chi2());   }
      if(v->tracksSize()==5){  Fill(h,"unmatchedVtx5trkchi2vsndof",v->ndof(),v->chi2());   }

    }
  }
  if(nrecvtxs>0){
    Fill(h,"unmatchedVtx",nfake);
    Fill(h,"unmatchedVtx4",nfake4);
    Fill(h,"unmatchedVtxW4",nfakew4);
    Fill(h,"unmatchedVtxFrac",nfake/nrecvtxs);
  }



  // --------------------------------------- match rec to sim ---------------------------------------
  //for(reco::VertexCollection::const_iterator v=recVtxs->begin(); v!=recVtxs->end(); ++v){

  for(unsigned int iv=0; iv<recVtxs->size(); iv++){
    const reco::Vertex * v= &(recVtxs->at(iv));

    double  nmatch=-1;      // highest number of tracks in recvtx v found in any event
    EncodedEventId evmatchId;
    unsigned int ematch=0;
    double dzmatch=1e10;
    bool matchFound=false;  // found a "matching" sim vertex (i.e. this vertex contains at least on track from that sim vertex)
    double nmatchvtx=0;     // number of simvtcs contributing to recvtx v
    double nmatch30vtx=0;   // number of simvtcs contributing >30% of their tracks to recvtx v
    double nmatch50vtx=0;   // number of simvtcs contributing >50% of their tracks to recvtx v

    double testany=0;
    unsigned int iev=0;
    for(vector<SimEvent>::iterator ev=simEvt.begin(); ev!=simEvt.end(); ev++){
      double n=0;  // number of tracks that are in both, the recvtx v and the event ev
      for(vector<TransientTrack>::iterator te=ev->tk.begin(); te!=ev->tk.end(); te++){
	

	const reco::Track&  RTe=te->track();
	for(trackit_t tv=v->tracks_begin(); tv!=v->tracks_end(); tv++){
	  const reco::Track & RTv=*(tv->get());  
	  if(RTe.vz()==RTv.vz()){ n++;}
	}
      }     
      
      testany+=n;

      // find the best match in terms of the highest number of tracks 
      // from a simvertex in this rec vertex
      if ( (n > nmatch) ||( (n==nmatch) &&(fabs(v->z()-ev->z)<dzmatch)) ){
	nmatch=n;
	evmatchId=ev->eventId;
	ematch=iev; // == ev->key 
	dzmatch=fabs(v->z()-ev->z);
	matchFound=true;
      }
      if(n>0){
	nmatchvtx++;
	if (n>0.3*ev->tk.size()) nmatch30vtx++;
	if (n>0.5*ev->tk.size()) nmatch50vtx++;
      }

      iev++;
    }

    double nmatchany=getTruthMatchedVertexTracks(*v).size();  // includes tracks from any sim vertex

    if ( (v->ndof()>4) && matchFound && (nmatchany>0)){

      if( !(ematch==recvmatch[iv].wosmatch)&&(recvmatch[iv].maxwos>0) ){
	cout << "Matchers disagree " << v->z() << " ematch=" << ematch << "  wosmatch=" << recvmatch[iv].wosmatch << endl;
	cout << " ematch  z=" << simEvt.at(ematch).z <<   "   wosmatch z=" <<  simEvt.at(recvmatch[iv].wosmatch).z << endl;
      }

      double sumw=0.5*(v->ndof()+2);
      Fill(h,"sumwoverntkvsztp",v->position().z(), sumw/v->tracksSize());
      //           highest number of tracks in recvtx matched to (the same) sim vertex
      // purity := -----------------------------------------------------------------
      //                  number of truth matched tracks in this recvtx

      double purity =nmatch/nmatchany; 
      Fill(h,"vtxpurityvsz",v->z(),purity); // average purity vs z
      Fill(h,"recmatchPurity",purity);
      if(iv==0){
	Fill(h,"recmatchPurityTag",purity, (bool)(evmatchId==iSignal));
      }else{
	Fill(h,"recmatchPuritynoTag",purity,(bool)(evmatchId==iSignal));
      }

      for(trackit_t t=v->tracks_begin(); t!=v->tracks_end(); t++){ 
	Float_t wt=v->trackWeight(*t);
	Fill(h,"allweight",wt);
	Fill(h,"allweightvsz",v->position().z(),wt);
	unsigned int trkev=trkidx2simevt_[t->key()];
	if(trkev == ematch){
	  // majority
	  Fill(h,"majorityweight",wt);
	  Fill(h,"majorityweightvsz",v->z(),wt);
	  Fill(h,"majorityfractionvsz",v->z(),1.);
	  Fill(h,"unmatchedfractionvsz",v->z(),0.);
	  Fill(h,"minorityfractionvsz",v->z(),0.);
	  Fill(h,"minorityafractionvsz",v->z(),0.);
	  Fill(h,"minoritybfractionvsz",v->z(),0.);
	  Fill(h,"unmatchedvtxtrkfractionvsz",v->z(),0.);
	}else{
	  if(trkev == 666){
	    //unmatchted
	    Fill(h,"unmatchedweight",wt);
	    Fill(h,"unmatchedweightvsz",v->z(),wt);
	    Fill(h,"unmatchedfractionvsz",v->z(),1.);
	    Fill(h,"majorityfractionvsz",v->z(),0.);
	    Fill(h,"minorityfractionvsz",v->z(),0.);
	    Fill(h,"minorityafractionvsz",v->z(),0.);
	    Fill(h,"minoritybfractionvsz",v->z(),0.);
	    Fill(h,"unmatchedvtxtrkfractionvsz",v->z(),0.);
	  }else{
	    //minority
	    Fill(h,"minorityweight",wt);
	    Fill(h,"minorityweightvsz",v->z(),wt);
	    Fill(h,"minorityfractionvsz",v->z(),1.);
	    if( simEvt.at(trkev).nwosmatch==0){
	      Fill(h,"minorityaweight",wt);
	      Fill(h,"minorityaweightvsz",v->z(),wt);
	      Fill(h,"minorityafractionvsz",v->z(),1.);
	      Fill(h,"minoritybfractionvsz",v->z(),0.);
	    }else{
	      Fill(h,"minoritybweight",wt);
	      Fill(h,"minoritybweightvsz",v->z(),wt);
	      Fill(h,"minoritybfractionvsz",v->z(),1.);
	      Fill(h,"minorityafractionvsz",v->z(),0.);
	    }
	    Fill(h,"unmatchedfractionvsz",v->z(),0.);
	    Fill(h,"majorityfractionvsz",v->z(),0.);
	    Fill(h,"unmatchedvtxtrkfractionvsz",v->z(),0.);
	  }
	}
      }

    }else if( v->ndof()>4){
      // for vertices that we have not matched to anything, can't separate minority/majority
      for(trackit_t t=v->tracks_begin(); t!=v->tracks_end(); t++){ 
	Float_t wt=v->trackWeight(*t);
	Fill(h,"unmatchedvtxtrkweight",v->z(),wt);
	Fill(h,"unmatchedvtxtrkweightvsz",v->z(),wt);

	Fill(h,"unmatchedvtxtrkfractionvsz",v->z(),1.);
	Fill(h,"unmatchedfractionvsz",v->z(),0.);
	Fill(h,"majorityfractionvsz",v->z(),0.);
	Fill(h,"minorityfractionvsz",v->z(),0.);
      }
    }

    Fill(h,"recmatchvtxs",nmatchvtx);
    Fill(h,"recmatch30vtxs",nmatch30vtx);
    Fill(h,"recmatch50vtxs",nmatch30vtx);
    if(iv==0){
      Fill(h,"recmatchvtxsTag",nmatchvtx);
      Fill(h,"recmatch30vtxsTag",nmatch30vtx);
      Fill(h,"recmatch50vtxsTag",nmatch30vtx);
    }else{
      Fill(h,"recmatchvtxsnoTag",nmatchvtx);
      Fill(h,"recmatch30vtxsnoTag",nmatch30vtx);
      Fill(h,"recmatch50vtxsnoTag",nmatch30vtx);
    }


     
  } // recvtx loop
  Fill(h,"nrecv",nrecvtxs);
  Fill(h,"nrecv4",nrecvtxs4);


  // --------------------------------------- match sim to rec  ---------------------------------------

  int npu1=0, npu2=0,npu3=0,npu4=0,npu5=0;

  for(vector<SimEvent>::iterator ev=simEvt.begin(); ev!=simEvt.end(); ev++){

    if(ev->tk.size()>0) npu1++;
    if(ev->tk.size()>1) npu2++;
    if(ev->tk.size()>2) npu3++;
    if(ev->tk.size()>3) npu4++;
    if(ev->tk.size()>4) npu5++;

    bool isSignal= ev->eventId==iSignal;
    
    Fill(h,"nRecTrkInSimVtx",(double) ev->tk.size(),isSignal);
    Fill(h,"nPrimRecTrkInSimVtx",(double) ev->tkprim.size(),isSignal);
    Fill(h,"sumpt2rec",sqrt(ev->sumpt2rec),isSignal);
    Fill(h,"sumpt2",sqrt(ev->sumpt2),isSignal);
    Fill(h,"sumpt",sqrt(ev->sumpt),isSignal);

    double nRecVWithTrk=0;  // vertices with tracks from this simvertex
    double  nmatch=0, ntmatch=0, zmatch=-99;
    double n50=0; // number of rec vertices which have >=50% of their tracks from this simvertex
    double n30=0; // number of rec vertices which have >=30% of their tracks from this simvertex

    for(reco::VertexCollection::const_iterator v=recVtxs->begin(); 
	v!=recVtxs->end(); ++v){
      // count tracks found in both, sim and rec
      double n=0, wt=0;
      for(vector<TransientTrack>::iterator te=ev->tk.begin(); te!=ev->tk.end(); te++){
	const reco::Track&  RTe=te->track();
	for(trackit_t tv=v->tracks_begin(); tv!=v->tracks_end(); tv++){
	   const reco::Track & RTv=*(tv->get());  
	   if(RTe.vz()==RTv.vz()) {n++; wt+=v->trackWeight(*tv);}
	}
      }

      ev->ntInRecVz[v->position().z()]=n;
      if(n>0){	nRecVWithTrk++; }
      if(n>=0.5*v->tracksSize()){ n50++;}
      if(n>=0.3*v->tracksSize()){ n30++;}
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
	  if(verbose_) cout << "Signal vertex not matched " <<  message << "  event=" << eventcounter_ << " nmatch=" << nmatch << "  ntsim=" << ntsim << endl;
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
    Fill(h,"vtxMultiplicity50",n50,isSignal);
    Fill(h,"vtxMultiplicity30",n30,isSignal);

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

    
  }//simevt loop
  
  Fill(h,"npu1",npu1);
  Fill(h,"npu2",npu2);
  Fill(h,"npu3",npu3);
  Fill(h,"npu4",npu4);
  Fill(h,"npu5",npu5);

  Fill(h,"nrecvsnpu",npu1,float(nrecvtxs));
  Fill(h,"nrec2vsnpu",npu1,float(nrecndof[2]));
  Fill(h,"nrec4vsnpu",npu1,float(nrecndof[4]));
  Fill(h,"nrecvsnpu2",npu2,float(nrecvtxs));
  Fill(h,"nrec2vsnpu2",npu2,float(nrecndof[2]));
  Fill(h,"nrec4vsnpu2",npu2,float(nrecndof[4]));

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


  double ntruthmatched=getTruthMatchedVertexTracks(*v).size();
  if (ntruthmatched>0) Fill(h,"TagVtxTrkPurity",n/ntruthmatched);
  if (ev->tk.size()>0) Fill(h,"TagVtxTrkEfficiency",n/ev->tk.size());


  if(dumpSignalVsTag_){
    cout << "Number of tracks in reco tagvtx " << v->tracksSize() << endl;
    cout << "Number of selected tracks in sim event vtx " << ev->tk.size() << "    (prim=" << ev->tkprim.size() << ")"<<endl;
    cout << "Number of tracks in both         " << n << endl;
    if (ntruthmatched>0){
      cout << "TrackPurity = "<< n/ntruthmatched <<endl;
    }
    if (ev->tk.size()>0){
      cout << "TrackEfficiency = "<< n/ev->tk.size() <<endl;
    }
  }


  // vertex pairs with >=4 
  for(vector<SimEvent>::iterator ev1=simEvt.begin(); ev1!=simEvt.end(); ev1++){
    if (ev1->tk.size()<4) continue;
    for(vector<SimEvent>::iterator ev2=ev1+1; ev2!=simEvt.end(); ev2++){
      if (ev2->tk.size()<4) continue;
      double deltazsim = ev2->z-ev1->z;
      Fill(h,"zdiffsimallTP",deltazsim);
      
      if( (ev1->nmatch>0)&&(ev2->nmatch>0) && ( !(ev1->zmatchn==ev2->zmatchn))){
	// both sim vertices of this pair were found
	double deltazrec=ev2->zmatchn-ev1->zmatchn;
	Fill(h,"zdiffsimfoundTP",deltazsim);        
	Fill(h,"zdiffrecvssimTP",deltazsim,deltazrec);
	Fill(h,"zdiffrecvsdsimTP",deltazsim,deltazrec-deltazsim);
	Fill(h,"zdiffrecvsdsimTPprof",deltazsim,deltazrec-deltazsim);
      }
      if( (ev1->nmatch>0)&&(ev2->nmatch>0)&&(ev1->zmatchn==ev2->zmatchn) ){
	// possible merger  
	cout << " possible merger " << endl;
	cout << "zsim = " << ev1->z  << " zm=" << ev1->zmatchn << "  n=" <<ev1->nmatch<< "  p=" << ev1->pmatchn
	     << " z(2)=" << ev1->zmatchn2 << "  n(2)=" << ev1->nmatch2<< "  p(2)=" << ev1->pmatchn2 << endl;
	cout << "zsim = " << ev2->z  << " zm=" << ev2->zmatchn << "  n=" << ev2->nmatch<< "  p=" << ev2->pmatchn
	     << " z(2)=" << ev2->zmatchn2 << "  n(2)=" <<ev2->nmatch2<< "  p(2)=" << ev2->pmatchn2 << endl;
 	if( ((ev1->nmatch2>4)&&(ev1->pmatchn2>0.5)) || ((ev2->nmatch2>4)&&(ev2->pmatchn2>0.5)) ){
	  Fill(h,"zdiffsimfoundTP2",deltazsim);
	}
      }
    }
  }

}

/***************************************************************************************/






/***************************************************************************************/

void PrimaryVertexAnalyzer4PU::analyzeVertexCollection(std::map<std::string, TH1*> & h,
						       const reco::VertexCollection * recVtxs,
						       const Handle<reco::TrackCollection> recTrks, 
						       std::vector<simPrimaryVertex> & simpv,
						       const std::string message)
{


  int nrectrks=recTrks->size();
  int nrecvtxs=recVtxs->size();
  int nseltrks=-1; 
  reco::TrackCollection selTrks;   // selected tracks
  reco::TrackCollection lostTrks;  // selected but lost tracks (part of dropped clusters)

  // extract dummy vertices representing clusters
  reco::VertexCollection clusters;
  reco::Vertex allSelected;
  

  
  if(simpv.size()>0){//this is mc
    double dsimrecx=0.;
    double dsimrecy=0.;//0.0011;
    double dsimrecz=0.;//0.0012;
    
    if(eventSummaryCounter_++ < nEventSummary_){
      printEventSummary(h, recVtxs,recTrks,simpv,message);
    }

    // vertex matching and efficiency bookkeeping
    int nsimtrk=0;
    int npu[5]={0,0,0,0,0};  // count pile-up vertices with > n tracks

    for(std::vector<simPrimaryVertex>::iterator vsim=simpv.begin();
	vsim!=simpv.end(); vsim++){
      
      for(int nt=0; nt<5; nt++){if (vsim->nGenTrk>=nt) npu[nt]++;  }

      nsimtrk+=vsim->nGenTrk;
      // look for a matching reconstructed vertex
      vsim->recVtx=NULL;
      //vsim->cluster=-1;
      

      // find the nearest recvertex  (multiple sims may be mapped to the same rec)
      for(reco::VertexCollection::const_iterator vrec=recVtxs->begin();  vrec!=recVtxs->end(); ++vrec){
	if( !(vrec->isFake()) ) {
	  if( (vsim->recVtx==NULL) ||  
	      ( (fabs(vsim->recVtx->position().z()-vsim->z) > fabs(vrec->z()-vsim->z)))){
	    vsim->recVtx=&(*vrec);
	  }
	}
      }
      //cout << Form("sim  %10.4f  ", vsim->z);      if( vsim->recVtx==NULL ){	cout << "---"<<endl;;}else{ cout << Form(" %10.4f +/- %8.4f\n",vsim->recVtx->position().z(),vsim->recVtx->zError());}


      Fill(h,"nsimtrk",static_cast<double>(nsimtrk));
      Fill(h,"nsimtrk",static_cast<double>(nsimtrk),vsim==simpv.begin());
      
      // histogram properties of matched vertices
      if (vsim->recVtx && ( fabs(vsim->recVtx->z()-vsim->z*simUnit_)<zmatch_ )){
	
	if(veryverbose_){std::cout <<"primary matched " << message << " " << setw(8) << setprecision(4) << vsim->x << " " << vsim->y << " " << vsim->z << std:: endl;}
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



	
	Fill(h,"resxvsNdofprof", vsim->recVtx->ndof(), vsim->recVtx->x()-vsim->x*simUnit_ );
	Fill(h,"resyvsNdofprof", vsim->recVtx->ndof(), vsim->recVtx->y()-vsim->y*simUnit_ );
	Fill(h,"resxvsNdofSpread", vsim->recVtx->ndof(), vsim->recVtx->x()-vsim->x*simUnit_ );
	Fill(h,"resyvsNdofSpread", vsim->recVtx->ndof(), vsim->recVtx->y()-vsim->y*simUnit_ );
	if(vsim->recVtx->ndof()>50){
	  Fill(h,"resx50", vsim->recVtx->x()-vsim->x*simUnit_ );
	  Fill(h,"resy50", vsim->recVtx->y()-vsim->y*simUnit_ );
	  Fill(h,"resz50", vsim->recVtx->z()-vsim->z*simUnit_ );
	}

	// efficiency with zmatch within 500 um (or whatever zmatch is)
	Fill(h,"eff", 1.);
	if(simpv.size()==1){
	  if (vsim->recVtx==&(*recVtxs->begin())){
	    Fill(h,"efftag", 1.); 
	  }else{
	    Fill(h,"efftag", 0.); 
	    cout << "signal vertex not tagged " << message << " " << eventcounter_ << endl;
	    dumpThisEvent_=true;
	  }
	}
	
	Fill(h,"effvsnrectrk",nrectrks,1.);
	Fill(h,"effvsnseltrk",nseltrks,1.);
	Fill(h,"effvsz",vsim->z*simUnit_,1.);
	Fill(h,"effvsz2",vsim->z*simUnit_,1.);
	if(vsim->type==1){ // full (i.e. not just PUInfo)
	  Fill(h,"effvsnsimtrk",vsim->nGenTrk,1.);
	  Fill(h,"effvsptsq",vsim->ptsq,1.);
	  Fill(h,"effvsr",sqrt(vsim->x*vsim->x+vsim->y*vsim->y)*simUnit_,1.);
	}

      }else{  // no matching rec vertex found for this simvertex
	
	bool plapper=veryverbose_ && vsim->nGenTrk;
	if(plapper){
	  // be quiet about invisble vertices
	  std::cout << "primary not found "  << message << " " << eventcounter_ << "  x=" <<vsim->x*simUnit_ << "  y=" << vsim->y*simUnit_  << " z=" << vsim->z*simUnit_  << " nGenTrk=" << vsim->nGenTrk << std::endl;
	}
	//int mistype=0;
	if (vsim->recVtx){
	  if(plapper){
	    std::cout << "nearest recvertex at " << vsim->recVtx->z() << "   dz=" << vsim->recVtx->z()-vsim->z*simUnit_ << std::endl;
	  }
	  
	  if (fabs(vsim->recVtx->z()-vsim->z*simUnit_)<0.2 ){
	    Fill(h,"effvsz2",vsim->z*simUnit_,1.);
	  }
	  
	  if (fabs(vsim->recVtx->z()-vsim->z*simUnit_)<0.5 ){
	    if(plapper){std::cout << "type 1, lousy z vertex" << std::endl;}
	    Fill(h,"zlost1", vsim->z*simUnit_,1.);
	    //mistype=1;
	  }else{
	    if(plapper){std::cout << "type 2a no vertex anywhere near" << std::endl;}
	    //mistype=2;
	  }
	}else{// no recVtx at all
	  //mistype=2;
	  if(plapper){std::cout << "type 2b, no vertex at all" << std::endl;}
	}
	
	
	Fill(h,"eff", 0.);
	if(simpv.size()==1){ Fill(h,"efftag", 0.); }
	
	Fill(h,"effvsnsimtrk",float(vsim->nGenTrk),0.);
	Fill(h,"effvsnrectrk",nrectrks,0.);
	Fill(h,"effvsnseltrk",nseltrks,0.);
	Fill(h,"effvsz",vsim->z*simUnit_,0.);
	if(vsim->type==1){
	  Fill(h,"effvsptsq",vsim->ptsq,0.);
	  Fill(h,"effvsr",sqrt(vsim->x*vsim->x+vsim->y*vsim->y)*simUnit_,0.);
	}
	
      } // no recvertex for this simvertex

    } // vsim loop


    int nrecvtxs4=0;
    for(reco::VertexCollection::const_iterator vrec=recVtxs->begin();  vrec!=recVtxs->end(); ++vrec){
      if( !(vrec->isFake()) && (vrec->ndof()>4) ) {nrecvtxs4++;}
    }
    for(int nt=0; nt<5; nt++){
      Fill(h,Form("nrecvsnpus%d",nt), float(npu[nt]), float(nrecvtxs));
      Fill(h,Form("nrec4vsnpus%d",nt), float(npu[nt]), float(nrecvtxs4));
      Fill(h,Form("nrec4vsnpus%dprof",nt), float(npu[nt]), float(nrecvtxs4));
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



    // look for rec vertices with no matching sim vertex
    for(reco::VertexCollection::const_iterator vrec=recVtxs->begin();  vrec!=recVtxs->end(); ++vrec){
      simPrimaryVertex * match=NULL;
      double zmax=zmatch_;      if ((3*vrec->zError())>zmatch_) zmax=3*vrec->zError();
      if( !(vrec->isFake())){

	for(std::vector<simPrimaryVertex>::iterator vsim=simpv.begin(); vsim!=simpv.end(); vsim++){
	  if(  (vsim->recVtx==&(*vrec)) && 
	    ( (match==NULL) || ( fabs(vrec->position().z()-vsim->z) < fabs(vrec->position().z()-match->z) ) )
	       ){ 
	  match=&(*vsim);}
	}
	
	//cout << Form("rec %10.4f +/- %8.4f",vrec->position().z(),vrec->zError());}  if( match==NULL ){cout << "---"<<endl;}else{cout << Form(" sim  %10.4f       %10.4f  -> ", match->z,zmax)<<  (fabs(vrec->position().z()-match->z)>zmax) << endl;;


	if ( (match==NULL) || ( fabs(vrec->position().z()-match->z)>zmax)){
	    Fill(h,"fakeVtxZ",vrec->z());
	    if (vrec->ndof()>=0.5) Fill(h,"fakeVtxZNdofgt05",vrec->z());
	    if (vrec->ndof()>=2.0) Fill(h,"fakeVtxZNdofgt2",vrec->z());
	    if (vrec->ndof()>=4.0) Fill(h,"fakeVtxZNdofgt4",vrec->z());
	    if (vrec->ndof()>=8.0) Fill(h,"fakeVtxZNdofgt8",vrec->z());
	    Fill(h,"fakeVtxNdof",vrec->ndof());
	    Fill(h,"fakeVtxNtrk",vrec->tracksSize());
	    if(vrec->tracksSize()==2){  Fill(h,"fake2trkchi2vsndof",vrec->ndof(),vrec->chi2());   }
	    if(vrec->tracksSize()==3){  Fill(h,"fake3trkchi2vsndof",vrec->ndof(),vrec->chi2());   }
	    if(vrec->tracksSize()==4){  Fill(h,"fake4trkchi2vsndof",vrec->ndof(),vrec->chi2());   }
	    if(vrec->tracksSize()==5){  Fill(h,"fake5trkchi2vsndof",vrec->ndof(),vrec->chi2());   }
	}
      }
    }



    


  // compare the signal vertex with the nearest rec vertex
  double deltaznearest=9999.;
  int indexnearest=-1,  idx=0;
  for(reco::VertexCollection::const_iterator vrec=recVtxs->begin();  vrec!=recVtxs->end(); ++vrec){
    //if( !(vrec->isFake()) && (vrec->ndof()>4) ) {
    if( !(vrec->isFake()) && (vrec->ndof()>0) ) {
      Double_t dz=vrec->z() - (*simpv.begin()).z*simUnit_;
      if (abs(dz)<abs(deltaznearest)){ deltaznearest=dz; indexnearest=idx;}
    }
    idx++;
  }

  Fill(h,"zdistancenearest",deltaznearest);
  Fill(h,"abszdistancenearest",fabs(deltaznearest));
  Fill(h,"indexnearest",float(indexnearest));
  
  } // simulated vertices in the event


  // isolated simulated vertices and vertex pairs
  if(simpv.size()>0){ 
    // sort sim vertices in z
    vector< pair<double,unsigned int> >  zsimv;
    for(unsigned int idx=0; idx<simpv.size(); idx++){
      zsimv.push_back(make_pair(simpv[idx].z, idx));
    }
    stable_sort(zsimv.begin(), zsimv.end(),lt);


  // pairs of simulated vertices vs pairs of reconstructed vertices
  if(simpv.size()>1){ 
    for(std::vector<simPrimaryVertex>::iterator vsim1=simpv.begin(); vsim1!=(simpv.end()-1); vsim1++){
      if(vsim1->nGenTrk<4) continue;
      for(std::vector<simPrimaryVertex>::iterator vsim2=vsim1+1; vsim2!=simpv.end(); vsim2++){
	if((vsim2->nGenTrk<4)) continue;
	double deltazsim=vsim2->z - vsim1->z;
	Fill(h,"zdiffsimall",deltazsim);
	if( (vsim1->recVtx==NULL) ||( vsim2->recVtx==NULL) ) continue;
	double deltazrec=vsim2->recVtx->position().z()-vsim1->recVtx->position().z();
	if(vsim2->recVtx==vsim1->recVtx){
	  // merged or lost for some other reason
	  Fill(h,"zdiffsimmerge",deltazsim);
	}else{
	  // separated
	  Fill(h,"zdiffrecvssim",deltazsim,deltazrec);
	  if ((vsim1->recVtx->ndof()>4)&&(vsim2->recVtx->ndof()>4)) Fill(h,"zdiffrec4vssim",deltazsim,deltazrec);
	  if ((vsim1->recVtx->ndof()>12)&&(vsim2->recVtx->ndof()>12)) Fill(h,"zdiffrec12vssim",deltazsim,deltazrec);
	  Fill(h,"zdiffsimfound",deltazsim);
	  Fill(h,"zdiffrecvsdsim",deltazsim,deltazrec-deltazsim);
	  Fill(h,"zdiffrecvsdsimprof",deltazsim,deltazrec-deltazsim);
	}
	
      }
    }


    // look for isolated pairs of simvertices, then count rec vertices inside the isolated interval



    double ziso=0.5;
    for(unsigned int idxsim=0; idxsim< simpv.size()-1; idxsim++){
      if ( 
	  ((idxsim==0) || ( (idxsim>0)&&(zsimv[idxsim].first-zsimv[idxsim-1].first>ziso)) )
	  &&((idxsim+1>=zsimv.size()-1) || ((idxsim+1<zsimv.size()-1)&&(zsimv[idxsim+2].first-zsimv[idxsim+1].first>ziso)))
	  ){
	if ( (simpv[zsimv[idxsim].second].nGenTrk>4)&&(simpv[zsimv[idxsim+1].second].nGenTrk>4)){
	  double dzsim=zsimv[idxsim+1].first-zsimv[idxsim].first;
	  double zmin=zsimv[idxsim  ].first-0.5*ziso;
	  double zmax=zsimv[idxsim+1].first+0.5*ziso;
	  int nreciso=0;
	  std::vector<double> zrec;
	  for(reco::VertexCollection::const_iterator vrec=recVtxs->begin();  vrec!=recVtxs->end(); ++vrec){
	    if (!(vrec->isFake()) && (vrec->ndof()>4) && (vrec->z()>zmin)&&(vrec->z()<zmax) ){
	      nreciso++;
	      zrec.push_back(vrec->z());
	    }
	  }
	  Fill(h, "zdiffsimisoall",  dzsim);
	  if (nreciso==0) Fill(h, "zdiffsimiso0", dzsim);
	  if (nreciso==1) Fill(h, "zdiffsimiso1", dzsim);
	  if (nreciso==2) Fill(h, "zdiffsimiso2", dzsim);
	  if (nreciso==3) Fill(h, "zdiffsimiso3", dzsim);
	  if (nreciso==2){
	    double dzrec=fabs(zrec[1]-zrec[2]);
	    Fill(h,"zdiffreciso2",dzrec);
	    Fill(h,"dzrecvssimiso2",fabs(dzsim),dzrec);
	  }
	}
	
      }
    }

  // single isolated
    for(unsigned int idxsim=0; idxsim< simpv.size(); idxsim++){
      if ( 
	  (simpv[zsimv[idxsim].second].nGenTrk>4) 
	  && ((idxsim==0) || ( (idxsim>0)&&(zsimv[idxsim].first-zsimv[idxsim-1].first>ziso)) )
	  &&((idxsim>=zsimv.size()-1) || ((idxsim<zsimv.size()-1)&&(zsimv[idxsim+1].first-zsimv[idxsim].first>ziso)))
	  ){
	double zmin=zsimv[idxsim].first-0.5*ziso;
	double zmax=zsimv[idxsim].first+0.5*ziso;
	int nreciso=0;
	for(reco::VertexCollection::const_iterator vrec=recVtxs->begin();  vrec!=recVtxs->end(); ++vrec){
	  if (!(vrec->isFake()) && (vrec->ndof()>4) && (vrec->z()>zmin)&&(vrec->z()<zmax) ){
	    nreciso++;
	    Fill(h, "zreciso", vrec->z()-zsimv[idxsim].first);
	  }
	  Fill(h, "nreciso",  float(nreciso));
	}
      }
    }


  }//simpv>1
  }//>simpv>0

  

  //******* the following code does not require MC and will/should work for data **********


  Fill(h,"bunchCrossing",bunchCrossing_);
  if(recTrks->size()>0)  Fill(h,"bunchCrossingLogNtk",bunchCrossing_,log(recTrks->size())/log(10.));
  
  // -----------------  reconstructed tracks  ------------------------
  // the list of selected tracks can only be correct if the selection parameterset  and track collection
  // is the same that was used for the reconstruction

  int nt=0;
  unsigned int trkidx=0;
  for(reco::TrackCollection::const_iterator t=recTrks->begin();
      t!=recTrks->end(); ++t){
    if((recVtxs->size()>0) && (recVtxs->begin()->isValid())){
      fillTrackHistos(h,"all",*t,&(*recVtxs->begin()));
    }else{
      fillTrackHistos(h,"all",*t);
    }
    if(MC_){
      if (trkidx2tp_.count(trkidx)>0){
	Fill(h, "matchedallfractionvsz", t->vz(), 1.);
	Fill(h, "unmatchedallfractionvsz", t->vz(), 0.);
      }else{
	Fill(h, "unmatchedallfractionvsz", t->vz(), 1.);
	Fill(h, "matchedallfractionvsz", t->vz(), 0.);
      }
    }

    TransientTrack  tt = theB_->build(&(*t));  tt.setBeamSpot(vertexBeamSpot_);

    if(   (t->hitPattern().pixelLayersWithMeasurement()>1)
	  && (t->hitPattern().trackerLayersWithMeasurement()>5)
          && (t->trackerExpectedHitsInner().numberOfHits()==0)
          && (t->trackerExpectedHitsOuter().numberOfHits()<2)
          && (t->hitPattern().numberOfLostTrackerHits()<2)
	  && (fabs(t->eta())<1.5)
	  && (t->pt()>0.5)
	  )
      {
	if(tt.stateAtBeamLine().isValid()){
	  double z=(tt.stateAtBeamLine().trackStateAtPCA()).position().z();
	  double tantheta=tan((tt.stateAtBeamLine().trackStateAtPCA()).momentum().theta());
	  //double dz2= pow(tt.track().dzError(),2)+wxy2_/pow(tantheta,2);
	  double phi=(tt.stateAtBeamLine().trackStateAtPCA()).momentum().phi();
	  double dz2= pow(tt.track().dzError(),2)+(pow(wx_*cos(phi),2)+pow(wy_*sin(phi),2))/pow(tantheta,2);
	  if ((dz2<0.01)&&(tt.stateAtBeamLine().transverseImpactParameter().significance()<2.)){
	    Fill(h,"z0trk",z);  
	  }
	} else {
	  cout << "PrimaryVertexAnalyzer4PU::analyzeVertexCollection : invalid stateAtBeamLine" << endl;
	  cout << "track   z "<< t->parameters() << endl;
	  dumpThisEvent_=true;
	}
      }

    if  (theTrackFilter(tt)){
      selTrks.push_back(*t);
      fillTrackHistos(h,"sel",*t);
      if(MC_){
	if (trkidx2tp_.count(trkidx)){
	  fillTrackHistos(h, "seltpmatched", *t);
	  Fill(h, "matchedselfractionvsz", t->vz(), 1.);
	  Fill(h, "unmatchedselfractionvsz", t->vz(), 0.);
	}else{
	  Fill(h, "matchedselfractionvsz", t->vz(), 0.);
	  Fill(h, "unmatchedselfractionvsz", t->vz(), 1.);
	  fillTrackHistos(h, "seltpunmatched", *t);
	}
      }
      int foundinvtx=0;
      int nvtemp=-1;
      for(reco::VertexCollection::const_iterator v=recVtxs->begin(); v!=recVtxs->end(); ++v){
	nvtemp++;
	if(( v->isFake()) || (v->ndof()<-2) ) break;
	for(trackit_t tv=v->tracks_begin(); tv!=v->tracks_end(); tv++ ){
	  if( ((**tv).vz()==t->vz()&&((**tv).phi()==t->phi())) ) {
	    foundinvtx++;
	    //fillTrackHistos(h,"vtx",*t, &(*v));
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
    trkidx++;
  }


  if (nseltrks<0){
    nseltrks=selTrks.size();
  }else if( ! (nseltrks==(int)selTrks.size()) ){
    std::cout << "Warning: inconsistent track selection !" << std::endl;
  }



  // count vertices above some ndof thresholds
  int nrec=0,  nrec0=0, nrec2=0, nrec3=0, nrec4=0, nrec5=0, nrec6=0, nrec7=0, nrec8=0;
  for(reco::VertexCollection::const_iterator v=recVtxs->begin(); v!=recVtxs->end(); ++v){
    if (! (v->isFake()) && v->ndof()>0 && v->chi2()>0 ){
      nrec++;
      if (v->ndof()>0) nrec0++;
      if (v->ndof()>2) nrec2++;
      if (v->ndof()>3) nrec3++;
      if (v->ndof()>4) nrec4++;
      if (v->ndof()>5) nrec5++;
      if (v->ndof()>6) nrec6++;
      if (v->ndof()>7) nrec7++;
      if (v->ndof()>8) nrec8++;
    }
  }
  Fill(h,"nrecvtx",nrec);
  Fill(h,"nrecvtx2",nrec2);
  Fill(h,"nrecvtx3",nrec3);
  Fill(h,"nrecvtx4",nrec4);
  Fill(h,"nrecvtx5",nrec5);
  Fill(h,"nrecvtx6",nrec6);
  Fill(h,"nrecvtx7",nrec7);
  Fill(h,"nrecvtx8",nrec8);
  
  if (nrec4>35){
    dumpThisEvent_=true;
  }

//   Fill(h,"nrecvtx4vsNpix",float(nDigiPix_), float(nrec4));
//   Fill(h,"nrecvtx4vsNpixprof",float(nDigiPix_), float(nrec4));
  Fill(h,"Npixvsnrecvtx4", float(nrec4), float(nDigiPix_));
  Fill(h,"Npixvsnrecvtx4prof", float(nrec4),float(nDigiPix_));
  if(instBXLumi_>0){
    Fill(h,"nrecvtxvsL",instBXLumi_, float(nrec));
    Fill(h,"nrecvtxvsLprof",instBXLumi_, float(nrec));
    Fill(h,"nrecvtx4vsL",instBXLumi_, float(nrec4));
    Fill(h,"nrecvtx4vsLprof",instBXLumi_, float(nrec4));
  }

  // fill track histos
  for(reco::VertexCollection::const_iterator v=recVtxs->begin(); 
      v!=recVtxs->end(); ++v){
    
    if (! (v->isFake()) && v->ndof()>0 && v->chi2()>0 ){
      for(trackit_t t=v->tracks_begin(); t!=v->tracks_end(); t++){
	if(v==recVtxs->begin()){
	  fillTrackHistos(h,"tagged",**t,  &(*v));
	}else{
	  fillTrackHistos(h,"untagged",**t,  &(*v));
	}
	if(v->ndof()>4) fillTrackHistos(h,"ndof4",**t, &(*v));

	Float_t wt=v->trackWeight(*t);
	Fill(h,"trackWt",wt);
	if(wt>0.5){
	  fillTrackHistos(h,"wgt05",**t, &(*v));
	}else{
	  fillTrackHistos(h,"wlt05",**t, &(*v));
	}

	if(fabs(v->position().z())<2.){
	  fillTrackHistos(h,"|z|<2",**t, &(*v));
	}else if (fabs(v->position().z())>10.){
	  fillTrackHistos(h,"|z|>10",**t, &(*v));
	}
      }
    }
  }



  // -----------------  reconstructed vertices  ------------------------

  // event 
  Fill(h,"szRecVtx",recVtxs->size());
  Fill(h,"nseltrk",nseltrks);
  Fill(h,"nrectrk",nrectrks);

  if(nrec>0){
    Fill(h,"eff0vsntrec",nrectrks,1.);
    Fill(h,"eff0vsntsel",nseltrks,1.);
  }else{
    Fill(h,"eff0vsntrec",nrectrks,0.);
    Fill(h,"eff0vsntsel",nseltrks,0.);
  }
  if(nrec0>0) { Fill(h,"eff0ndof0vsntsel",nseltrks,1.);}else{ Fill(h,"eff0ndof0vsntsel",nseltrks,0.);}
  if(nrec2>0) { Fill(h,"eff0ndof2vsntsel",nseltrks,1.);}else{ Fill(h,"eff0ndof2vsntsel",nseltrks,0.);}
  if(nrec4>0) { Fill(h,"eff0ndof4vsntsel",nseltrks,1.);}else{ Fill(h,"eff0ndof4vsntsel",nseltrks,0.);}
  if(nrec8>0) { Fill(h,"eff0ndof8vsntsel",nseltrks,1.);}else{ Fill(h,"eff0ndof8vsntsel",nseltrks,0.);}



//   if((nrectrks>10)&&(nseltrks<3)){
//     cout << "small fraction of selected tracks "  << endl;
//     dumpThisEvent_=true;
//   }

  // properties of events without a vertex
  if((nrec==0)||(recVtxs->begin()->isFake())){
    Fill(h,"nrectrk0vtx",nrectrks);
    Fill(h,"nseltrk0vtx",nseltrks);
  }


  //  properties of (valid) vertices
  double ndof2=-10,ndof1=-10, zndof1=0, zndof2=0;
  unsigned int zcentral4[400]={0};
  unsigned int zcentral0[400]={0};
  
  for(reco::VertexCollection::const_iterator v=recVtxs->begin(); 
      v!=recVtxs->end(); ++v){
    if(v->isFake()){ Fill(h,"isFake",1.);}else{ Fill(h,"isFake",0.);}
    if(v->isFake()||((v->ndof()<-1)&&(v->ndof()>-3))){ Fill(h,"isFake1",1.);}else{ Fill(h,"isFake1",0.);}

    if((v->isFake())||(v->ndof()<-1)) continue;

    if     (v->ndof()>ndof1){ ndof2=ndof1; zndof2=zndof1; ndof1=v->ndof(); zndof1=v->position().z();}
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

    }// two track vertices


    // count vertex tracks
    double npt1=0, ntrkwgt05=0;
    for(trackit_t t = v->tracks_begin(); t!=v->tracks_end(); t++) {
      if (v->trackWeight(*t)>0.5) {
	ntrkwgt05++;
	Fill(h, "trackwtgt05vsz", v->position().z(), v->trackWeight(*t)); 
      }
      if ( (**t).pt()>1.0) npt1++;
      Fill(h, "trackwtvsz", v->position().z(),v->trackWeight(*t)); 
    }
    if(v->ndof()>4){
      Fill(h,"ntrkpt1vsz", v->position().z(), npt1);
      Fill(h,"ntrkwgt05vsz", v->position().z(), ntrkwgt05);
      Fill(h,"ftrkwgt05vsz", v->position().z(), ntrkwgt05/v->tracksSize());
    }

    Fill(h,"trkchi2vsndof",v->ndof(),v->chi2());
    if(v->ndof()>0){    Fill(h,"trkchi2overndof",v->chi2()/v->ndof()); }
    if(v->tracksSize()==2){  Fill(h,"2trkchi2vsndof",v->ndof(),v->chi2());   }
    if(v->tracksSize()==3){  Fill(h,"3trkchi2vsndof",v->ndof(),v->chi2());   }
    if(v->tracksSize()==4){  Fill(h,"4trkchi2vsndof",v->ndof(),v->chi2());   }
    if(v->tracksSize()==5){  Fill(h,"5trkchi2vsndof",v->ndof(),v->chi2());   }

    Fill(h,"nbtksinvtx",v->tracksSize());
    if(instBXLumi_>0){
      Fill(h,"nbtksinvtxvsL",instBXLumi_,float(v->tracksSize()));
      if(v->ndof()>4) Fill(h,"nbtksinvtx4vsL",instBXLumi_,float(v->tracksSize()));
    }
    Fill(h,"nbtksinvtx2",v->tracksSize());
    Fill(h,"vtxchi2",v->chi2());
    Fill(h,"vtxndf",v->ndof());
    Fill(h,"vtxprob",ChiSquaredProbability(v->chi2() ,v->ndof()));
    Fill(h,"vtxndfvsntk",v->tracksSize(), v->ndof());
    double sumw=0.5*(v->ndof()+2);
    Fill(h,"sumwoverntk0",sumw/v->tracksSize());
    Fill(h,"sumwoverntkvsz0",v->position().z(), sumw/v->tracksSize());
    if(v->ndof()>4){
      Fill(h,"sumwvsz",v->position().z(),sumw);
      Fill(h,"sumntkvsz",v->position().z(),(double) v->tracksSize());
      Fill(h,"sumwoverntk",sumw/v->tracksSize());
      Fill(h,"vtxndfoverntk",v->ndof()/v->tracksSize());
      Fill(h,"vtxndf2overntk",(v->ndof()+2)/v->tracksSize());
      Fill(h,"sumwoverntkvsz",v->position().z(), sumw/v->tracksSize());
      Fill(h,"sumwoverntkvsz4",v->position().z(), sumw/v->tracksSize());
      if(ntrkwgt05>0){
	Fill(h,"sumwoverntkwgt05vsz",v->position().z(),sumw/ntrkwgt05);
      }
    }
    if((v->ndof()>4)&&(v->ndof()<10)){
      Fill(h,"sumwoverntkvszlo",v->position().z(), sumw/v->tracksSize());
    }
    if((v->ndof()>4)&&(v->ndof()<10)){ Fill(h,"sumwoverntkvszlo",v->position().z(), sumw/v->tracksSize());   }
    if(v->ndof()>20){ Fill(h,"sumwoverntkvszhi",v->position().z(), sumw/v->tracksSize());   }
    Fill(h,"ntrkvsz", v->position().z(), float(v->tracksSize()));
    Fill(h,"ndofvsz", v->position().z(),v->ndof());
    double log10ndof=v->ndof()>0 ? log(v->ndof())/log(10.) : -10;
    Fill(h,"log10ndofvsz", v->position().z(), log10ndof);





    if(v->ndof()>5.0){  // enter only vertices that really contain tracks
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
      Fill(h,"zrec3a",v->position().z());
      Fill(h,"xrecb",v->position().x()-vertexBeamSpot_.x0());
      Fill(h,"yrecb",v->position().y()-vertexBeamSpot_.y0());
      Fill(h,"zrecb",v->position().z()-vertexBeamSpot_.z0());
      Fill(h,"xrecBeam",v->position().x()-vertexBeamSpot_.x0());
      Fill(h,"yrecBeam",v->position().y()-vertexBeamSpot_.y0());
      Fill(h,"zrecBeam",v->position().z()-vertexBeamSpot_.z0());
      //Fill(h,"zrecBeamvsL", instBXLumi_, v->position().z()-vertexBeamSpot_.z0());
      //Fill(h,"zrecBeamBX",float(bunchCrossing_), v->position().z()-vertexBeamSpot_.z0());
      Fill(h,"xrecBeamPull",(v->position().x()-vertexBeamSpot_.x0())/sqrt(pow(v->xError(),2)+pow(vertexBeamSpot_.BeamWidthX(),2)));
      Fill(h,"yrecBeamPull",(v->position().y()-vertexBeamSpot_.y0())/sqrt(pow(v->yError(),2)+pow(vertexBeamSpot_.BeamWidthY(),2)));
      Fill(h,"zrecBeamPull",(v->position().z()-vertexBeamSpot_.z0())/sqrt(pow(v->zError(),2)+pow(sigmaZ_,2)));
      Fill(h,"zrecBeamPull0",(v->position().z()-vertexBeamSpot_.z0())/sigmaZ_);
      if(v->ndof()>12.){
	Fill(h,"zrecBeamPull12",(v->position().z()-vertexBeamSpot_.z0())/sigmaZ_);
      }
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

      Fill(h,"xrecBeamvsNdofprof",v->ndof(),v->position().x()-vertexBeamSpot_.x0());
      Fill(h,"yrecBeamvsNdofprof",v->ndof(),v->position().y()-vertexBeamSpot_.y0());
      

      if(bunchCrossing_>0){
	//Fill(h,Form("zrecBX_%04d", bunchCrossing_), v->position().z());
	Fill(h,"zvsls",float(luminosityBlock_),v->position().z());
	Fill(h,"zbeamvsls",float(luminosityBlock_),vertexBeamSpot_.z0());
	Fill(h,"zrecLS",float(luminosityBlock_),v->position().z());
      }
      
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
    }// ndof>5



    if(v->ndof()>8){
      Fill(h,"xrec8",v->position().x());
      Fill(h,"yrec8",v->position().y());
      Fill(h,"zrec8",v->position().z());
      Fill(h,"zrec8r",(v->position().z()-vertexBeamSpot_.position().z())*sqrt(2.));
    }

    if(v->ndof()>12){
      Fill(h,"xrec12",v->position().x());
      Fill(h,"yrec12",v->position().y());
      Fill(h,"zrec12",v->position().z());
      Fill(h,"zrec12r",(v->position().z()-vertexBeamSpot_.position().z())*sqrt(2.));
      Fill(h,"zrec12q",(v->position().z()-vertexBeamSpot_.position().z())/sqrt(2.));
      if (v==recVtxs->begin()){
      Fill(h,"zrec12tag",v->position().z());
      }
    }

      
    if(v->ndof()>2.0){ 
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
      
    }// ndof>2
    


//     // more vertex properties
//     for(trackit_t t = v->tracks_begin(); 
// 	t!=v->tracks_end(); t++) {
//       tt.setBeamSpot(vertexBeamSpot_);   
//       TransientTrack  tt = theB_->build(*t);
//     }

    double z0=v->position().z()-vertexBeamSpot_.z0();
    if (fabs(z0)<2.){
      unsigned int vbin=int(100.*(z0+2.));
      if(vbin<400){
	zcentral0[vbin]++;
	if(v->ndof()>4)zcentral4[vbin]++;
      }
    }
        
    //  properties of (valid) neighbour vertices
    reco::VertexCollection::const_iterator v1=v;     v1++;
    for(; v1!=recVtxs->end(); ++v1){
      if((v1->isFake())||(v1->ndof()<-1)) continue;
      npair_++;
      Fill(h,"zdiffrec",v->position().z()-v1->position().z());
      Fill(h,"zdiffrechr",v->position().z()-v1->position().z());


      double z1=v1->position().z()-vertexBeamSpot_.z0();

      // lower ndof of the pair
      double ndoflow=v1->ndof();
      double ndofhi=v->ndof();
      if(v1->ndof()>v->ndof()){
	ndofhi=v1->ndof();
	ndoflow=v->ndof();
      }else{
	ndofhi=v->ndof();
	ndoflow=v1->ndof();
      }
      
      // central vertices, avoid acceptance issues
      if ((ndoflow>4)&&(fabs(z0)<1.)){
	Fill(h, "dzreccentral", fabs(z0-z1));
	Fill(h, "ndofcentral",   fabs(z0-z1), v->ndof());
	Fill(h, "ndoflocentral", fabs(z0-z1), ndoflow);
	Fill(h, "ndofhicentral", fabs(z0-z1), ndofhi);
	Fill(h, "ndofsumcentral", fabs(z0-z1), ndofhi+ndoflow);
      }


      double zbar=0.5*(z1+z0);
      double zbarp=(0.5*(z1+z0)-vertexBeamSpot_.z0())/sigmaZ_;

      Fill(h,"ndofnr2",ndoflow); 
      if(fabs(zndof1-zndof2)>1) Fill(h,"ndofnr2d1cm",ndoflow); 
      if(fabs(zndof1-zndof2)>2) Fill(h,"ndofnr2d2cm",ndoflow); 
      
      
      Fill(h,"zdiffvsz",z1-z0,zbar);
      Fill(h,"zdiffvszp",z1-z0,zbarp);

      if (nrec==2) Fill(h,"zdiffvszNv2",z1-z0,zbar);
      //if((fabs(z1-z0)<2.0)&&((v->ndof()<20)||(v1->ndof()<20))){
      if(fabs(z1-z0)<0.2){
	Fill(h,"zbarFakeEnriched",zbar);
	if(ndoflow>5) Fill(h,"zbarFakeEnriched5",zbar);
	if(ndoflow>2) Fill(h,"zbarFakeEnriched2",zbar);
      }
      if((fabs(z1-z0)>2.0)&&(v->ndof()>10)&&(v1->ndof()>10)){Fill(h,"zbarFakeDepleted",zbar);}   // just for comparison , pure real


      if ((v->ndof()>2) && (v1->ndof()>2)){
	Fill(h,"zdiffrec2",z1-z0);
	Fill(h,"zdiffvsz2",z1-z0,zbar);
	Fill(h,"zdiffvszp2",z1-z0,zbarp);
	if (nrec2==2) Fill(h,"zdiffvsz2Nv2",z1-z0,zbar);
	Fill(h,"zvszrec2",z0, z1);
	Fill(h,"pzvspz2",TMath::Freq(z0/sigmaZ_),TMath::Freq(z1/sigmaZ_) );
      }
      
      if ((v->ndof()>4) && (v1->ndof()>4)){
	Fill(h,"zdiffvsz4",z1-z0,zbar);
	Fill(h,"zdiffvszp4",z1-z0,zbarp);
	if (nrec4==2) Fill(h,"zdiffvsz4Nv2",z1-z0,zbar);
	Fill(h,"zdiffrec4",z1-z0);
	Fill(h,"zdiffrec4hr",z1-z0);
	Fill(h,"zdiffrec4p",(z1-z0)/sigmaZ_);
	Fill(h,"zvszrec4",z0, z1);
	Fill(h,"pzvspz4",TMath::Freq(z0/sigmaZ_),TMath::Freq(z1/sigmaZ_));
	if(v==recVtxs->begin()){
	  Fill(h,"zdiffrec4tag",z1-z0);
	  Fill(h,"zdiffrec4taghr",z1-z0);
	}

      }
      
      if ((v->ndof()>5) && (v1->ndof()>5)){
	Fill(h,"zdiffvsz5",z1-z0,zbar);
	Fill(h,"zdiffrec5",z1-z0);
	Fill(h,"zdiffvszp5",z1-z0,zbarp);
	if (nrec5==2) Fill(h,"zdiffvsz5Nv2",z1-z0,zbar);
      }

      
      if ((v->ndof()>6) && (v1->ndof()>6)){
	Fill(h,"zdiffvsz6",z1-z0,zbar);
	Fill(h,"zdiffrec6",z1-z0);
	Fill(h,"zdiffvszp6",z1-z0,zbarp);
	if (nrec6==2) Fill(h,"zdiffvsz6Nv2",z1-z0,zbar);
      }

      if ((v->ndof()>7) && (v1->ndof()>7)){
	Fill(h,"zdiffvsz7",z1-z0,zbar);
	Fill(h,"zdiffrec7",z1-z0);
	Fill(h,"zdiffvszp7",z1-z0,zbarp);
	if (nrec7==2) Fill(h,"zdiffvsz7Nv2",z1-z0,zbar);
      }

      if ((v->ndof()>8) && (v1->ndof()>8)){
	Fill(h,"zdiffvsz8",z1-z0,zbar);
	Fill(h,"zdiffrec8",z1-z0);
	Fill(h,"zdiffvszp8",z1-z0,zbarp);
	if (nrec8==2) Fill(h,"zdiffvsz8Nv2",z1-z0,zbar);
      }
      if ((v->ndof()>12) && (v1->ndof()>12)){
	Fill(h,"zdiffvsz12",z1-z0,zbar);
	Fill(h,"zdiffrec12",z1-z0);
	Fill(h,"zdiffvszp12",z1-z0,zbarp);
      }
      if ((v->ndof()>20) && (v1->ndof()>20)){
	Fill(h,"zdiffrec20",z1-z0);
      }

    }

    // is it isolated?
    double deltaz=1e10;
    for(reco::VertexCollection::const_iterator v1=recVtxs->begin(); v1!=recVtxs->end(); ++v1){
      if (v->position().z()==v1->position().z()) continue;
      if (fabs(v->position().z()-v1->position().z())<fabs(deltaz)) deltaz=v->position().z()-v1->position().z();
    }						
    if(fabs(deltaz)>2.0) Fill(h,"vtxndfIso",v->ndof());
    


      
  }  // vertex loop (v)



  for(unsigned int i=1; i<200; i++){
    double dz=0.01*i;
    unsigned int offs=99-int(i/2);
    for(unsigned int j=0; j<200; j++){
      //cout <<  setprecision(3) << dz << "  " << offs+j << " " << offs+j+i << " n=" << zcentral4[offs+j]+zcentral4[offs+j+i] << endl;
      unsigned int n=zcentral4[offs+j]+zcentral4[offs+j+i];
      if(n==0){ Fill(h, "n0dz",dz,1.); }else{ Fill(h, "n0dz",dz,0.); }
      if(n==1){ Fill(h, "n1dz",dz,1.); }else{ Fill(h, "n1dz",dz,0.); }
      if(n==2){ Fill(h, "n2dz",dz,1.); }else{ Fill(h, "n2dz",dz,0.); }
      if(n==3){ Fill(h, "n3dz",dz,1.); }else{ Fill(h, "n3dz",dz,0.); }
      if(n>=4){ Fill(h, "n4dz",dz,1.); }else{ Fill(h, "n4dz",dz,0.); }

      n=zcentral0[offs+j]+zcentral0[offs+j+i];
      if(n==0){ Fill(h, "n0dz0",dz,1.); }else{ Fill(h, "n0dz0",dz,0.); }
      if(n==1){ Fill(h, "n1dz0",dz,1.); }else{ Fill(h, "n1dz0",dz,0.); }
      if(n==2){ Fill(h, "n2dz0",dz,1.); }else{ Fill(h, "n2dz0",dz,0.); }
      if(n==3){ Fill(h, "n3dz0",dz,1.); }else{ Fill(h, "n3dz0",dz,0.); }
      if(n>=4){ Fill(h, "n4dz0",dz,1.); }else{ Fill(h, "n4dz0",dz,0.); }
    }
  }

  for(unsigned int i=0; i<400; i++){
    for(unsigned int j=0; j<400; j++){
      if(zcentral4[i]>0){
	if(zcentral4[j]==0){
	  Fill(h, "nc10", (int(i)-200)*0.01+0.001, (int(j)-200)*0.01+0.001);
	}else{
	  Fill(h, "nc11", (int(i)-200)*0.01+0.001, (int(j)-200)*0.01+0.001);
	}
      }else{
	if(zcentral4[j]==0){
	  Fill(h, "nc00", (int(i)-200)*0.01+0.001, (int(j)-200)*0.01+0.001);
	}else{
	  Fill(h, "nc01", (int(i)-200)*0.01+0.001, (int(j)-200)*0.01+0.001);
	}
      }
    }
  }


}

