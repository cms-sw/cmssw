#include "Validation/RecoVertex/interface/PrimaryVertexAnalyzer4PU.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

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

// temporary, test of IPTools
#include "TrackingTools/IPTools/interface/IPTools.h"

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
  //  vtxSample_   = iConfig.getUntrackedParameter<std::vector< std::string > >("vtxSample");
  rootFile_ = TFile::Open(outputFile_.c_str(),"RECREATE");
  verbose_= iConfig.getUntrackedParameter<bool>("verbose", false);
  doMatching_= iConfig.getUntrackedParameter<bool>("matching", false);
  simUnit_= 1.0;  // starting with CMSSW_1_2_x ??
  if ( (edm::getReleaseVersion()).find("CMSSW_1_1_",0)!=std::string::npos){
    simUnit_=0.1;  // for use in  CMSSW_1_1_1 tutorial
  }

  //zmatch_=0.0500; // 500 um
  zmatch_=iConfig.getUntrackedParameter<double>("zmatch", 0.0500);
  cout << "PrimaryVertexAnalyzer4PU: zmatch=" << zmatch_ << endl;
  eventcounter_=0;
  ndump_=100;
  DEBUG_=false;
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
  // temporary
  h["IPToolsLxy"] = new TH1F("IPToolsLxy","IPToolsLxy",100,-0.2,0.2);

  // release validation histograms used in DoCompare.C
  h["nbtksinvtx"]   = new TH1F("nbtksinvtx","reconstructed tracks in vertex",40,-0.5,39.5); 
  h["nbtksinvtxPU"]   = new TH1F("nbtksinvtxPU","reconstructed tracks in vertex",40,-0.5,39.5); 
  h["nbtksinvtxTag"]   = new TH1F("nbtksinvtxTag","reconstructed tracks in vertex",40,-0.5,39.5); 
  h["resx"]         = new TH1F("resx","residual x",100,-0.04,0.04);
  h["resy"]         = new TH1F("resy","residual y",100,-0.04,0.04);
  h["resz"]         = new TH1F("resz","residual z",100,-0.1,0.1);
  h["resz10"]       = new TH1F("resz10","residual z",100,-1.0,1.);
  h["pullx"]        = new TH1F("pullx","pull x",100,-25.,25.);
  h["pully"]        = new TH1F("pully","pull y",100,-25.,25.);
  h["pullz"]        = new TH1F("pullz","pull z",100,-25.,25.);
  h["vtxchi2"]      = new TH1F("vtxchi2","chi squared",100,0.,100.);
  h["vtxndf"]       = new TH1F("vtxndf","degrees of freedom",100,0.,100.);
  h["vtxndfvsntk"]  = new TH2F("vtxndfvsntk","ndof vs #tracks",20,0.,100, 20, 0., 200.);
  h["vtxndfoverntk"]= new TH1F("vtxndfoverntk","ndof / #tracks",40,0.,2.);
  h["tklinks"]      = new TH1F("tklinks","Usable track links",2,-0.5,1.5);
  h["nans"]         = new TH1F("nans","Illegal values for x,y,z,xx,xy,xz,yy,yz,zz",9,0.5,9.5);
  // more histograms
  h["resxr"]         = new TH1F("resxr","relative residual x",100,-0.04,0.04);
  h["resyr"]         = new TH1F("resyr","relative residual y",100,-0.04,0.04);
  h["reszr"]         = new TH1F("reszr","relative residual z",100,-0.1,0.1);
  h["pullxr"]        = new TH1F("pullxr","relative pull x",100,-25.,25.);
  h["pullyr"]        = new TH1F("pullyr","relative pull y",100,-25.,25.);
  h["pullzr"]        = new TH1F("pullzr","relative pull z",100,-25.,25.);
  h["vtxprob"]      = new TH1F("vtxprob","chisquared probability",100,0.,1.);
  h["eff"]          = new TH1F("eff","efficiency",2, -0.5, 1.5);
  h["efftag"]       = new TH1F("efftag","efficiency tagged vertex",2, -0.5, 1.5);
  h["zdistancetag"] = new TH1F("zdistancetag","z-distance between tagged and generated",100, -0.1, 0.1);
  h["abszdistancetag"] = new TH1F("abszdistancetag","z-distance between tagged and generated",1000, 0., 1.0);
  h["abszdistancetagcum"] = new TH1F("abszdistancetagcum","z-distance between tagged and generated",1000, 0., 1.0);
  h["puritytag"]    = new TH1F("puritytag","purity of primary vertex tags",2, -0.5, 1.5);
  h["effvseta"]     = new TProfile("effvseta","efficiency vs eta",20, -2.5, 2.5, 0, 1.);
  h["effvsptsq"]    = new TProfile("effvsptsq","efficiency vs ptsq",20, 0., 10000., 0, 1.);
  h["effvsnsimtrk"]    = new TProfile("effvsnsimtrk","efficiency vs # simtracks",50, 0., 50., 0, 1.);
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
  h["xrec"]         = new TH1F("xrec","reconstructed x",100,-0.01,0.01);
  h["yrec"]         = new TH1F("yrec","reconstructed y",100,-0.01,0.01);
  h["zrec"]         = new TH1F("zrec","reconstructed z",100,-20.,20.);
  h["xrec2"]         = new TH1F("xrec2","reconstructed x",100,-0.1,0.1);
  h["yrec2"]         = new TH1F("yrec2","reconstructed y",100,-0.1,0.1);
  h["zrec2"]         = new TH1F("zrec2","reconstructed z",100,-20.,20.);
  h["xrec3"]         = new TH1F("xrec3","reconstructed x",100,-0.01,0.01);
  h["yrec3"]         = new TH1F("yrec3","reconstructed y",100,-0.01,0.01);
  h["zrec3"]         = new TH1F("zrec3","reconstructed z",100,-20.,20.);
  h["nrecvtx"]      = new TH1F("nrecvtx","# of reconstructed vertices", 50, -0.5, 49.5);
  //  h["nsimvtx"]      = new TH1F("nsimvtx","# of simulated vertices", 50, -0.5, 49.5);
  h["nrectrk"]      = new TH1F("nrectrk","# of reconstructed tracks", 100, -0.5, 99.5);
  h["nsimtrk"]      = new TH1F("nsimtrk","# of simulated tracks", 100, -0.5, 99.5);
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
  
  // properties of fake vertices
  h["fakeVtxZNdofgt05"] = new TH1F("fakeVtxZndofgt05","z of fake vertices with ndof>0.5", 100, -20., 20.);
  h["fakeVtxZ"]    = new TH1F("fakeVtxZ","z of fake vertices", 100, -20., 20.);
  h["fakeVtxNdof"] = new TH1F("fakeVtxNdof","ndof of fake vertices", 100,0., 40.);
  h["fakeVtxNtrk"] = new TH1F("fakeVtxNtrk","number of tracks in fake vertex",20,-0.5, 19.5);


  //  histograms pertaining to track quality
  h["recRapidity"] = new TH1F("recRapidity","reconstructed rapidity ",100,-10., 10.);
  string types[] = {"all","bachelor"};
  for(int t=0; t<2; t++){
    h["rapidity_"+types[t]] = new TH1F(("rapidity_"+types[t]).c_str(),"rapidity ",100,-3., 3.);
    h["pt_"+types[t]] = new TH1F(("pt_"+types[t]).c_str(),"pt ",200,0., 20.);
    h["found_"+types[t]]     = new TH1F(("found_"+types[t]).c_str(),"found hits",20, 0., 20.);
    h["lost_"+types[t]]      = new TH1F(("lost_"+types[t]).c_str(),"lost hits",20, 0., 20.);
    h["nchi2_"+types[t]]     = new TH1F(("nchi2_"+types[t]).c_str(),"normalized track chi2",100, 0., 20.);
    h["rstart_"+types[t]]    = new TH1F(("rstart_"+types[t]).c_str(),"start radius",100, 0., 100.);
    h["tfom_"+types[t]]      = new TH1F(("tfom_"+types[t]).c_str(),"track figure of merit",100, 0., 100.);
    h["logtresxy_"+types[t]] = new TH1F(("logtresxy_"+types[t]).c_str(),"log10(track r-phi resolution/um)",100, 0., 5.);
    h["logtresz_"+types[t]] = new TH1F(("logtresz_"+types[t]).c_str(),"log10(track z resolution/um)",100, 0., 5.);
    h["tpullxy_"+types[t]]   = new TH1F(("tpullxy_"+types[t]).c_str(),"track r-phi pull",100, -10., 10.);
  }
   

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

  // new histograms
  h["npu0"]       =new TH1F("npu0","Number of simulated vertices",40,0.,40.);
  h["npu1"]       =new TH1F("npu1","Number of simulated vertices with >0 track",40,0.,40.);
  h["npu2"]       =new TH1F("npu2","Number of simulated vertices with >1 track",40,0.,40.);
  h["nrecv"]      =new TH1F("nrecv","# of reconstructed vertices", 40, 0, 40);
  add(h,new TH2F("nrecvsnpu","#rec vertices vs number of sim vertices with >0 tracks", 40,  0., 40, 40,  0, 40));
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
  add(h,new TH1F("recPurity","track purity of reconstructed vertices", 101, 0., 1.01));
  add(h,new TH1F("recPuritySignal","track purity of reconstructed Signal vertices", 101, 0., 1.01));
  add(h,new TH1F("recPurityPU","track purity of reconstructed PU vertices", 101, 0., 1.01));
  add(h,new TH1F("recmatchPurityTag","track purity of tagged vertices", 101, 0., 1.01));
  add(h,new TH1F("recmatchPurityTagSignal","track purity of tagged Signal vertices", 101, 0., 1.01));
  add(h,new TH1F("recmatchPurityTagPU","track purity of tagged PU vertices", 101, 0., 1.01));
  add(h,new TH1F("recmatchPuritynoTag","track purity of untagged vertices", 101, 0., 1.01));
  add(h,new TH1F("recmatchPuritynoTagSignal","track purity of untagged Signal vertices", 101, 0., 1.01));
  add(h,new TH1F("recmatchPuritynoTagPU","track purity of untagged PU vertices", 101, 0., 1.01));
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
  
  add(h,new TH1F("matchVtxFraction","fraction of sim vertex track found in a recvertex",101,0,1.01));
  add(h,new TH1F("matchVtxFractionSignal","fraction of sim vertex track found in a recvertex",101,0,1.01));
  add(h,new TH1F("matchVtxFractionPU","fraction of sim vertex track found in a recvertex",101,0,1.01));
  add(h,new TH1F("matchVtxFractionCum","fraction of sim vertex track found in a recvertex",101,0,1.01));
  add(h,new TH1F("matchVtxFractionCumSignal","fraction of sim vertex track found in a recvertex",101,0,1.01));
  add(h,new TH1F("matchVtxFractionCumPU","fraction of sim vertex track found in a recvertex",101,0,1.01));
  add(h,new TH1F("matchVtxEfficiency","efficiency for finding matching rec vertex",2,-0.5,1.5));
  add(h,new TH1F("matchVtxEfficiencySignal","efficiency for finding matching rec vertex",2,-0.5,1.5));
  add(h,new TH1F("matchVtxEfficiencyPU","efficiency for finding matching rec vertex",2,-0.5,1.5));
  add(h,new TH1F("matchVtxEfficiency2","efficiency for finding matching rec vertex (nt>1)",2,-0.5,1.5));
  add(h,new TH1F("matchVtxEfficiency2Signal","efficiency for finding matching rec vertex (nt>1)",2,-0.5,1.5));
  add(h,new TH1F("matchVtxEfficiency2PU","efficiency for finding matching rec vertex (nt>1)",2,-0.5,1.5));
  add(h,new TH1F("matchVtxEfficiencyZ","efficiency for finding matching rec vertex within 0.5mm",2,-0.5,1.5));
  add(h,new TH1F("matchVtxEfficiencyZSignal","efficiency for finding matching rec vertex within 0.5mm",2,-0.5,1.5));
  add(h,new TH1F("matchVtxEfficiencyZPU","efficiency for finding matching rec vertex within 0.5mm",2,-0.5,1.5));


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
void PrimaryVertexAnalyzer4PU::beginJob(edm::EventSetup const& iSetup){
  std::cout << " PrimaryVertexAnalyzer4PU::beginJob  conversion from sim units to rec units is " << simUnit_ << std::endl;

  edm::ESHandle<TrackAssociatorBase> theHitsAssociator;
  iSetup.get<TrackAssociatorRecord>().get("TrackAssociatorByHits",theHitsAssociator);
  associatorByHits = (TrackAssociatorBase *) theHitsAssociator.product();
  

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

  TDirectory *DAF = rootFile_->mkdir("DAF");
  DAF->cd();
  hDAF=bookVertexHistograms();
  for(std::map<std::string,TH1*>::const_iterator hist=hDAF.begin(); hist!=hDAF.end(); hist++){
    hist->second->SetDirectory(DAF);
  }

  TDirectory *MVF = rootFile_->mkdir("MVF");
  MVF->cd();
  hMVF=bookVertexHistograms();
  for(std::map<std::string,TH1*>::const_iterator hist=hMVF.begin(); hist!=hMVF.end(); hist++){
    hist->second->SetDirectory(MVF);
  }

  rootFile_->cd();
  hsimPV["rapidity"] = new TH1F("rapidity","rapidity ",100,-10., 10.);
  hsimPV["chRapidity"] = new TH1F("chRapidity","charged rapidity ",100,-10., 10.);
  hsimPV["recRapidity"] = new TH1F("recRapidity","reconstructed rapidity ",100,-10., 10.);
  hsimPV["pt"] = new TH1F("pt","pt ",100,0., 20.);

  hsimPV["xsim"]         = new TH1F("xsim","simulated x",100,-0.01,0.01); // 0.01cm = 100 um
  hsimPV["ysim"]         = new TH1F("ysim","simulated y",100,-0.01,0.01);
  hsimPV["zsim"]         = new TH1F("zsim","simulated z",100,-20.,20.);
  hsimPV["xsim2"]        = new TH1F("xsim2","simulated x",100,-0.1,0.1); // 0.01cm = 100 um
  hsimPV["ysim2"]        = new TH1F("ysim2","simulated y",100,-0.1,0.1);
  hsimPV["zsim2"]        = new TH1F("zsim2","simulated z",100,-20.,20.);
  hsimPV["xsim3"]        = new TH1F("xsim3","simulated x",100,-0.01,0.01); // 0.01cm = 100 um
  hsimPV["ysim3"]        = new TH1F("ysim3","simulated y",100,-0.01,0.01);
  hsimPV["zsim3"]        = new TH1F("zsim3","simulated z",100,-20.,20.);
  hsimPV["nsimvtx"]      = new TH1F("nsimvtx","# of simulated vertices", 50, -0.5, 49.5);
  hsimPV["nsimtrk"]      = new TH1F("nsimtrk","# of simulated tracks", 100, -0.5, 99.5); //  not filled right now, also exists in hBS..
  hsimPV["nsimtrk"]->StatOverflows(kTRUE);
  hsimPV["nbsimtksinvtx"]= new TH1F("nbsimtksinvtx","simulated tracks in vertex",100,-0.5,99.5); 
  hsimPV["nbsimtksinvtx"]->StatOverflows(kTRUE);

}


void PrimaryVertexAnalyzer4PU::endJob() {
  std::cout << "this is void PrimaryVertexAnalyzer4PU::endJob() " << std::endl;
  //cumulate some histos
  double sumDA=0,sumBS=0,sumnoBS=0, sumDAF=0,sumMVF=0;
  for(int i=101; i>0; i--){
    sumDA+=hDA["matchVtxFractionSignal"]->GetBinContent(i)/hDA["matchVtxFractionSignal"]->Integral();
    hDA["matchVtxFractionCumSignal"]->SetBinContent(i,sumDA);
    sumBS+=hBS["matchVtxFractionSignal"]->GetBinContent(i)/hBS["matchVtxFractionSignal"]->Integral();
    hBS["matchVtxFractionCumSignal"]->SetBinContent(i,sumBS);
    sumnoBS+=hnoBS["matchVtxFractionSignal"]->GetBinContent(i)/hnoBS["matchVtxFractionSignal"]->Integral();
    hnoBS["matchVtxFractionCumSignal"]->SetBinContent(i,sumnoBS);
    sumDAF+=hDAF["matchVtxFractionSignal"]->GetBinContent(i)/hDAF["matchVtxFractionSignal"]->Integral();
    hDAF["matchVtxFractionCumSignal"]->SetBinContent(i,sumDAF);
    sumMVF+=hMVF["matchVtxFractionSignal"]->GetBinContent(i)/hMVF["matchVtxFractionSignal"]->Integral();
    hMVF["matchVtxFractionCumSignal"]->SetBinContent(i,sumMVF);
  }
  sumDA=0,sumBS=0,sumnoBS=0,sumDAF=0,sumMVF=0;
  for(int i=1; i<1001; i++){
    sumDA+=hDA["abszdistancetag"]->GetBinContent(i);
    hDA["abszdistancetagcum"]->SetBinContent(i,sumDA/float(hDA["abszdistancetag"]->GetEntries()));
    sumBS+=hBS["abszdistancetag"]->GetBinContent(i);
    hBS["abszdistancetagcum"]->SetBinContent(i,sumBS/float(hBS["abszdistancetag"]->GetEntries()));
    sumnoBS+=hnoBS["abszdistancetag"]->GetBinContent(i);
    hnoBS["abszdistancetagcum"]->SetBinContent(i,sumnoBS/float(hnoBS["abszdistancetag"]->GetEntries()));
    sumDAF+=hDAF["abszdistancetag"]->GetBinContent(i);
    hDAF["abszdistancetagcum"]->SetBinContent(i,sumDAF/float(hDAF["abszdistancetag"]->GetEntries()));
    sumMVF+=hMVF["abszdistancetag"]->GetBinContent(i);
    hMVF["abszdistancetagcum"]->SetBinContent(i,sumMVF/float(hMVF["abszdistancetag"]->GetEntries()));
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
std::vector<PrimaryVertexAnalyzer4PU::SimPart> PrimaryVertexAnalyzer4PU::getSimTrkParameters(edm::Handle<edm::SimTrackContainer> & simTrks,
								 edm::Handle<edm::SimVertexContainer> & simVtcs,
								 double simUnit)
{
   std::vector<SimPart > tsim;
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
         if ( (Q != 0) && (p.pt()>0.2)  && (fabs(t->momentum().eta())<2.5)
              && fabs(v.z()*simUnit<20) && (sqrt(v.x()*v.x()+v.y()*v.y())<2.)){
           double x0=v.x()*simUnit;
           double y0=v.y()*simUnit;
           double z0=v.z()*simUnit;
           double kappa=-Q*0.002998*fBfield/p.pt();
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
           sp.par[reco::TrackBase::i_qoverp] = Q/p.mag();
           sp.par[reco::TrackBase::i_lambda] = M_PI/2.-p.theta();
           sp.par[reco::TrackBase::i_phi] = p.phi()-asin(kappa*s0);
           sp.par[reco::TrackBase::i_dxy] = -2.*D0/(1.+q);
           sp.par[reco::TrackBase::i_dsz] = z0*sin(p.theta())-s1*cos(p.theta());

           if (v.t()-t0<1e-15){
             sp.type=0;  // primary
           }else{
             sp.type=1;  //secondary
           }

           // now get zpca  (get perigee wrt beam)
           double x1=x0-0.033; double y1=y0-0.;
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

           tsim.push_back(sp);
         }
       }
     }// has vertex
   }//for loop
   return tsim;
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

// bool PrimaryVertexAnalyzer4PU::isFinalstateParticle(const TrackingParticle &p){
//     return ( p.decayVertices.size()==0 && p.status()==1 );
// }

bool PrimaryVertexAnalyzer4PU::isCharged(const HepMC::GenParticle * p){
  const ParticleData * part = pdt_->particle( p->pdg_id() );
  if (part){
    return part->charge()!=0;
  }else{
    // the new/improved particle table doesn't know anti-particles
    return  pdt_->particle( -p->pdg_id() )!=0;
  }
}

void PrimaryVertexAnalyzer4PU::fillTrackHistos(std::map<std::string, TH1*> & h, const std::string & ttype, const reco::Track & t){
    h["rapidity_"+ttype]->Fill(t.eta());
    h["pt_"+ttype]->Fill(t.pt());
    h["found_"+ttype]->Fill(t.found());
    h["lost_"+ttype]->Fill(t.lost());
    h["nchi2_"+ttype]->Fill(t.normalizedChi2());
    //std::cout << "track inner position = " << t->innerPosition() << std::endl;
    h["rstart_"+ttype]->Fill((t.innerPosition()).Rho());
    double d0Error=t.d0Error();
    double d0=t.dxy(myBeamSpot);
    if (d0Error>0){ 
      h["logtresxy_"+ttype]->Fill(log(d0Error/0.0001)/log(10.));
      h["tpullxy_"+ttype]->Fill(d0/d0Error);
    }
    //double z0=t.vz();
    double dzError=t.dzError();
    if(dzError>0){
      h["logtresz_"+ttype]->Fill(log(dzError/0.0001)/log(10.));
    }
}


void PrimaryVertexAnalyzer4PU::printRecVtxs(const Handle<reco::VertexCollection> recVtxs, std::string title){
    int ivtx=0;
    std::cout << std::endl << title << std::endl;
    for(reco::VertexCollection::const_iterator v=recVtxs->begin(); 
	v!=recVtxs->end(); ++v){
      string vtype=" recvtx  ";
      if( v->isFake()){
	vtype=" fake   ";
      }else if (v->ndof()<-1){
	vtype=" cluster "; // pos=selector[iclu],cputime[iclu],clusterz[iclu]
      }else if(v->ndof()<0){
	vtype=" event   ";
      }
      std::cout << "vtx "<< std::setw(3) << std::setfill(' ')<<ivtx++
	        << vtype
		<< " #trk " << std::fixed << std::setprecision(4) << std::setw(3) << v->tracksSize() 
		<< " chi2 " << std::setw(4) << v->chi2() 
		<< " ndof " << std::setw(3) << v->ndof() //<< std::endl 
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
    edm::RefToBase<reco::Track> track=*tv;// halleluja
    if (truthMatchedTrack(track, tpr)){
      b.push_back(*tv);
    }
  }

//     try{
//       std::vector<std::pair<TrackingParticleRef, double> > tp = r2s_[track];
//       const TransientTrack & tt = theB_->build(*tv->get());
//       if (theTrackFilter(tt)){  //  count only tracks that are selected by the filter
// 	b.push_back(*tv);
//       }
//     } catch (Exception event) {
//       // silly way of testing whether track is in r2s_
//     }
//    }

  return b;
}
/********************************************************************************************************/





/********************************************************************************************************/
std::vector<PrimaryVertexAnalyzer4PU::SimEvent> PrimaryVertexAnalyzer4PU::getSimEvents
(
 const Event& iEvent, const EventSetup& iSetup,
 edm::Handle<TrackingParticleCollection>  TPCollectionH,
 edm::Handle<TrackingVertexCollection>  TVCollectionH,
 edm::Handle<View<Track> > trackCollectionH
 ){

  const TrackingParticleCollection* simTracks = TPCollectionH.product();
  const View<Track>  tC = *(trackCollectionH.product());

  reco::BeamSpot vertexBeamSpot;
  edm::Handle<reco::BeamSpot> recoBeamSpotHandle;
  edm::ESHandle<TransientTrackBuilder> theB;
  iSetup.get<TransientTrackRecord>().get("TransientTrackBuilder",theB);

  if(iEvent.getByType(recoBeamSpotHandle)){
    vertexBeamSpot = *recoBeamSpotHandle;
  }else{
    cout << "got no beamspot " << endl;
  }
  

  vector<SimEvent> simEvt;
  map<EncodedEventId, unsigned int> eventIdToEventMap;
  map<EncodedEventId, unsigned int>::iterator id;
  for(TrackingParticleCollection::const_iterator it=simTracks->begin(); it!=simTracks->end(); it++){
    
    unsigned int event=0;  //note, this is no longer the same as eventId().event()
    id=eventIdToEventMap.find(it->eventId());
    if (id==eventIdToEventMap.end()){
      // new event here
      SimEvent e;
      e.eventId=it->eventId();
      event=simEvt.size();
      const TrackingVertex *parentVertex= it->parentVertex().get();
      cout << it->eventId().bunchCrossing() << "," <<  it->eventId().event() << " z="<< it->vz() << " " << parentVertex->eventId().bunchCrossing() << ","  <<parentVertex->eventId().event() << " z=" << parentVertex->position().z() << endl;
      if (it->eventId()==parentVertex->eventId()){
	e.x=it->vx(); e.y=it->vy(); e.z=it->vz();
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
  

  //RECOTOSIM  association by hits
  r2s_ =   associatorByHits->associateRecoToSim (trackCollectionH,TPCollectionH, &iEvent );  // move this to "analyze"
  for(View<Track>::size_type i=0; i<tC.size(); ++i) {
    RefToBase<Track> track(trackCollectionH, i);
    TrackingParticleRef tpr;
    if( truthMatchedTrack(track,tpr)){
      if( eventIdToEventMap.find(tpr->eventId())==eventIdToEventMap.end() ){ cout << "Bug in getSimEvents" << endl; break; }
      unsigned int event=eventIdToEventMap[tpr->eventId()];
      double ipsig=0,ipdist=0;
      if (event==0){
	double d=sqrt(pow(simEvt[event].x-tpr->vx(),2)+pow(simEvt[event].y-tpr->vy(),2)+pow(simEvt[event].z-tpr->vz(),2))*1.e4;
	ipdist=d;
      }else{
	double dxy=track->dxy(vertexBeamSpot.position());
	ipsig=dxy/track->dxyError();
	ipdist=0;
      }

      const TransientTrack & t = theB->build(tC[i]);    
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
      TransientVertex v = theFitter.vertex(simEvt[i].tkprim, vertexBeamSpot);
      if (v.isValid()){
	simEvt[i].xfit=v.position().x();
	simEvt[i].yfit=v.position().y();
	simEvt[i].zfit=v.position().z();
	//	if (simEvt[i].z<-80.){simEvt[i].z=v.position().z();} // for now
      }
    }

//     // at some point, we should get the vertex from the tracking vertex position
//     if(simEvt[i].tp.size()>0){
//       //simEvt[i].z=simEvt[i].tp[0]->vz();
//        if( (i==0) || ((i>0)&&(simEvt[i].tp[0]->vz()!=simEvt[0].z)) ){  // kludge !!!
// 	 simEvt[i].z=simEvt[i].tp[0]->vz();
//        }
//     }

    
    cout << i <<"  )   nTP="  << simEvt[i].tp.size()
	 << "   z=" <<  simEvt[i].z
	 << "    recTrks="  << simEvt[i].tk.size() 
	 << "    recTrksPrim="  << simEvt[i].tkprim.size() 
	 << "   zfit=" << simEvt[i].zfit
	 << endl;
  }
 
  return simEvt;
}


std::vector<PrimaryVertexAnalyzer4PU::simPrimaryVertex> PrimaryVertexAnalyzer4PU::getSimPVs(
				      const Handle<HepMCProduct> evtMC)
{
  std::vector<PrimaryVertexAnalyzer4PU::simPrimaryVertex> simpv;
  const HepMC::GenEvent* evt=evtMC->GetEvent();
  if (evt) {
    std::cout << "process id " <<evt->signal_process_id()<<std::endl;
    std::cout <<"signal process vertex "<< ( evt->signal_process_vertex() ?
					     evt->signal_process_vertex()->barcode() : 0 )   <<std::endl;
    std::cout <<"number of vertices " << evt->vertices_size() << std::endl;


    //int idx=0;
    for(HepMC::GenEvent::vertex_const_iterator vitr= evt->vertices_begin();
	vitr != evt->vertices_end(); ++vitr ) 
      { // loop for vertex ...

	HepMC::FourVector pos = (*vitr)->position();
	//	if (pos.t()>0) { continue;} // skip secondary vertices, doesn't work for some samples


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
	      if ( (m.perp()>0.8) && (fabs(m.pseudoRapidity())<2.0) && isCharged( *daughter ) ){
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
  cout << "------- PrimaryVertexAnalyzer4PU simPVs -------" <<  endl;
  for(std::vector<simPrimaryVertex>::iterator v0=simpv.begin();
      v0!=simpv.end(); v0++){
    cout << "z=" << v0->z << "  eta=" << v0->ptot.pseudoRapidity() 
	 << "  px=" << v0->ptot.px()
	 << "  py=" << v0->ptot.py()
	 << "  pz=" << v0->ptot.pz() << endl;
  }
  cout << "-----------------------------------------------" << endl;
  return simpv;
}








/* get sim pv from TrackingParticles/TrackingVertex */
std::vector<PrimaryVertexAnalyzer4PU::simPrimaryVertex> PrimaryVertexAnalyzer4PU::getSimPVs(
											  const edm::Handle<TrackingVertexCollection> tVC
											  )
{
  std::vector<PrimaryVertexAnalyzer4PU::simPrimaryVertex> simpv;
  std::cout <<"number of vertices " << tVC->size() << std::endl;

  std::cout << "TrackingVertexCollection " << std::endl;

  for (TrackingVertexCollection::const_iterator v = tVC -> begin(); v != tVC -> end(); ++v) {

    std::cout << (v->eventId()).event() << v -> position() << v->g4Vertices().size() <<" "  <<v->genVertices().size() <<  "   t=" <<v->position().t()*1.e12 <<"    ==0:" <<(v->position().t()>0) << std::endl;
    //    std::cout << "g4Vertices " << v->ng4Vertices() << std::endl;
    for( TrackingVertex::g4v_iterator gv=v->g4Vertices_begin(); gv!=v->g4Vertices_end(); gv++){
      std::cout << *gv << std::endl;
    }
    std::cout << "----------" << std::endl;
 
    //    bool hasMotherVertex=false;

    if (v->position().t()>0) { continue;} // skip secondary vertices
    
    // could be a new vertex, check  all primaries found so far to avoid multiple entries
    const double mm=0.1;
    simPrimaryVertex sv(v->position().x()*mm,v->position().y()*mm,v->position().z()*mm);

    cout << "sv: " << (v->eventId()).event() << endl;
    //sv.event=(v->eventId()).event();
    sv.eventId=v->eventId();

    for (TrackingParticleRefVector::iterator iTrack = v->daughterTracks_begin(); iTrack != v->daughterTracks_end(); ++iTrack){
      //cout <<((**iTrack).eventId()).event() << " ";  // an iterator of Refs, dereference twice. Cool eyh?
      sv.eventId=(**iTrack).eventId();
    }
    cout <<endl;
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
      std::cout << "this is a new vertex " << sv.eventId.event() << "   "  << sv.x << " " << sv.y << " " << sv.z <<std::endl;
      simpv.push_back(sv);
      vp=&simpv.back();
    }else{
      std::cout << "this is not a new vertex"  << sv.x << " " << sv.y << " " << sv.z <<std::endl;
    }


    // Loop over daughter track(s)
    //simpv.p
    for (TrackingVertex::tp_iterator iTP = v -> daughterTracks_begin(); iTP != v -> daughterTracks_end(); ++iTP) {
      std::cout << "  Daughter momentum:      " << (*(*iTP)).momentum();
      std::cout << "  Daughter type     " << (*(*iTP)).pdgId();
      std::cout << std::endl;
      //if ( find(v->sourceTracks_begin(), v-> sourceTracks_end(), *iTP)==v-> sourceTracks_end())
      //if (isFinalstateParticle(*iTP)){ };
    }
  }

// 	// collect final state descendants and sum up momenta etc
// 	for ( HepMC::GenVertex::particle_iterator
// 	      daughter  = (*vitr)->particles_begin(HepMC::descendants);
// 	      daughter != (*vitr)->particles_end(HepMC::descendants);
//               ++daughter ) {
// 	  if (isFinalstateParticle(*daughter)){ 
// 	    if ( find(vp->finalstateParticles.begin(), vp->finalstateParticles.end(),(*daughter)->barcode())
// 		 == vp->finalstateParticles.end()){
// 	      vp->finalstateParticles.push_back((*daughter)->barcode());
// 	      HepMC::FourVector m=(*daughter)->momentum();
// 	      vp->ptot.setPx(vp->ptot.px()+m.px());
// 	      vp->ptot.setPy(vp->ptot.py()+m.py());
// 	      vp->ptot.setPz(vp->ptot.pz()+m.pz());
// 	      vp->ptot.setE(vp->ptot.e()+m.e());
// 	      vp->ptsq+=(m.perp())*(m.perp());
// 	      // count relevant particles
// 	      if ( (m.perp()>0.8) && (fabs(m.pseudoRapidity())<2.5) && isCharged( *daughter ) ){
// 		vp->nGenTrk++;
// 	      }
	      
// 	      h["rapidity"]->Fill(m.pseudoRapidity());
// 	      h["pt"]->Fill(m.perp());
// 	    }//new final state particle for this vertex
// 	  }//daughter is a final state particle
// 	}


// 	//idx++;
//       }
//  }
  cout << "------- PrimaryVertexAnalyzer4PU simPVs from TrackingVertices -------" <<  endl;
  for(std::vector<simPrimaryVertex>::iterator v0=simpv.begin();
      v0!=simpv.end(); v0++){
    cout << "z=" << v0->z << "  event=" << v0->eventId.event() << endl;
  }
  cout << "-----------------------------------------------" << endl;
  return simpv;
}



// ------------ method called to produce the data  ------------
void
PrimaryVertexAnalyzer4PU::analyze(const Event& iEvent, const EventSetup& iSetup)
{
  
  bool MC=false;
  std::vector<simPrimaryVertex> simpv;  //  a list of primary MC vertices
  std::vector<SimPart> tsim;
  std::string mcproduct="generator";  // starting with 3_1_0 pre something

  eventcounter_++;
  if(verbose_){
    std::cout << "PrimaryVertexAnalyzer4PU::analyze   event counter=" << eventcounter_ << std::endl;
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

  Handle<reco::VertexCollection> recVtxsDAF;
  bool bDAF=iEvent.getByLabel("offlinePrimaryVerticesDAF", recVtxsDAF);

  Handle<reco::VertexCollection> recVtxsMVF;
  bool bMVF=iEvent.getByLabel("offlinePrimaryVerticesMVF", recVtxsMVF);

  Handle<reco::TrackCollection> recTrks;
  iEvent.getByLabel(recoTrackProducer_, recTrks);
  
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
  bool gotTP=iEvent.getByLabel("mergedtruth","MergedTrackTruth",TPCollectionH);
  bool gotTV=iEvent.getByLabel("mergedtruth","MergedTrackTruth",TVCollectionH);

  
  vector<SimEvent> simEvt;
  if (gotTP && gotTV){
    simEvt=getSimEvents(iEvent, iSetup, TPCollectionH, TVCollectionH, trackCollectionH);
  }

  fBfield=3.8; // fix this

  //if(gotTV){
  if(false){

    MC=true;
    cout << "Found Tracking Vertices " << endl;
    simpv=getSimPVs(TVCollectionH);
    

  }else if(iEvent.getByLabel(mcproduct,evtMC)){

    MC=true;
    cout << "Using HepMCProduct " << endl;
    simpv=getSimPVs(evtMC);

    std::cout << "simtrks " << simTrks->size() << std::endl;
    tsim = PrimaryVertexAnalyzer4PU::getSimTrkParameters(simTrks, simVtxs, simUnit_);

  }else{
    cout << "No MC info at all" << endl;
  }





  // get the beam spot from the appropriate dummy vertex
  for(reco::VertexCollection::const_iterator v=recVtxs->begin(); 
      v!=recVtxs->end(); ++v){
    if ( (fabs(v->ndof()+1.)<0.0001) && (v->chi2()==0) ){ 
      myBeamSpot=math::XYZPoint(v->x(), v->y(), v->z());
    }
  }

  


  hsimPV["nsimvtx"]->Fill(simpv.size());
  for(std::vector<simPrimaryVertex>::iterator vsim=simpv.begin();
       vsim!=simpv.end(); vsim++){
     if(doMatching_){
      matchRecTracksToVertex(*vsim, tsim, recTrks);
     }

     hsimPV["nbsimtksinvtx"]->Fill(vsim->nGenTrk);
     hsimPV["xsim"]->Fill(vsim->x*simUnit_);
     hsimPV["ysim"]->Fill(vsim->y*simUnit_);
     hsimPV["zsim"]->Fill(vsim->z*simUnit_);
     hsimPV["xsim2"]->Fill(vsim->x*simUnit_);
     hsimPV["ysim2"]->Fill(vsim->y*simUnit_);
     hsimPV["zsim2"]->Fill(vsim->z*simUnit_);
     hsimPV["xsim3"]->Fill(vsim->x*simUnit_-myBeamSpot.x());
     hsimPV["ysim3"]->Fill(vsim->y*simUnit_-myBeamSpot.y());
     hsimPV["zsim3"]->Fill(vsim->z*simUnit_-myBeamSpot.z());
  }



   if(verbose_ && (eventcounter_<ndump_)){
     //evtMC->GetEvent()->print();
    if (bnoBS) printRecVtxs(recVtxs,"Offline without Beamspot");
    if (bBS) printRecVtxs(recVtxsBS,"Offline with Beamspot");
    if (bDA) printRecVtxs(recVtxsDA,"Offline DA");
    if (bDAF) printRecVtxs(recVtxsDAF,"Offline DAF");
    if (bMVF) printRecVtxs(recVtxsMVF,"Offline MVF");
   }

   
   if(bnoBS){
     std::cout << "PrimaryVertexAnalyzer4PU::analyze: doing no BS" << std::endl;
     analyzeVertexCollection(hnoBS, recVtxs, recTrks, simpv,"noBS");
     analyzeVertexCollectionTP(hnoBS, recVtxs, recTrks, simEvt,"noBS");
   }
   if(bBS){
     std::cout << "PrimaryVertexAnalyzer4PU::analyze: doing BS" << std::endl;
     analyzeVertexCollection(hBS, recVtxsBS, recTrks, simpv,"BS");
     analyzeVertexCollectionTP(hBS, recVtxsBS, recTrks, simEvt,"BS");
   }
   if(bDA){
     std::cout << "PrimaryVertexAnalyzer4PU::analyze: doing DA" << std::endl;
     analyzeVertexCollection(hDA, recVtxsDA, recTrks, simpv,"DA");
     analyzeVertexCollectionTP(hDA, recVtxsDA, recTrks, simEvt,"DA");
   }
   if(bDAF){
     std::cout << "PrimaryVertexAnalyzer4PU::analyze: doing DAF" << std::endl;
     analyzeVertexCollection(hDAF, recVtxsDAF, recTrks, simpv,"DAF");
     analyzeVertexCollectionTP(hDAF, recVtxsDAF, recTrks, simEvt,"DAF");
   }
   if(bMVF){
     std::cout << "PrimaryVertexAnalyzer4PU::analyze: doing MVF" << std::endl;
     analyzeVertexCollection(hMVF, recVtxsMVF, recTrks, simpv,"MVF");
     analyzeVertexCollectionTP(hMVF, recVtxsMVF, recTrks, simEvt,"MVF");
   }

  if(verbose_){
    std::cout << "PrimaryVertexAnalyzer4PU::analyze: done " << std::endl << std::endl;
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
						vector<SimEvent> & simEvt){
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
  cout << "event counter = " << eventcounter_ << endl;
  cout << "---------------------------" << endl;
  cout << " z[cm]                  ";
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

  cout << "---------------------------" << endl;



  map<unsigned int, unsigned int> rvmatch; // reco vertex matched to sim vertex  (sim to rec)
  map<unsigned int, double > nmatch;  // highest number of truth-matched tracks of ev found in a recvtx
  map<unsigned int, double > purity;  // highest purity of a rec vtx (i.e. highest number of tracks from the same simvtx)

  for(vector< pair<double,unsigned int> >::iterator itrec=zrecv.begin(); itrec!=zrecv.end(); itrec++){purity[itrec->second]=0.;}

  for(vector< pair<double,unsigned int> >::iterator itsim=zsimv.begin(); itsim!=zsimv.end(); itsim++){
    SimEvent* ev =&(simEvt[itsim->second]);


    cout.precision(4);
    if (itsim->second==0){
      cout << setw(8) << fixed << ev->z << ")*" << setw(5) << ev->tk.size() << setw(5) << ev->tkprim.size() << "  | ";
    }else{
      cout << setw(8) << fixed << ev->z << ") " << setw(5) << ev->tk.size() << setw(5) << ev->tkprim.size() << "  | ";
    }

    nmatch[itsim->second]=0;  // highest number of truth-matched tracks of ev found in a recvtx
    
    for(vector< pair<double,unsigned int> >::iterator itrec=zrecv.begin(); itrec!=zrecv.end(); itrec++){
      const reco::Vertex *v = &(recVtxs->at(itrec->second));

      // count tracks found in both, sim and rec
      double n=0;
      for(vector<TransientTrack>::iterator te=ev->tk.begin(); te!=ev->tk.end(); te++){
	const reco::Track&  RTe=te->track();
	 for(trackit_t tv=v->tracks_begin(); tv!=v->tracks_end(); tv++){
	   const reco::Track & RTv=*(tv->get());  
	   if(RTe.vz()==RTv.vz()) {n++;}
	}
      }
      cout << setw(7) << int(n)<< " ";

      if (n > nmatch[itsim->second]){
	nmatch[itsim->second]=n;
	rvmatch[itsim->second]=itrec->second;
      }

      if(n > purity[itrec->second]){
	purity[itrec->second]=n;
      }

    }// end of reco vertex loop
    cout << "  | " << "  max eff. = "  << setw(8) << nmatch[itsim->second]/ev->tk.size() << endl;
  }
  cout << "---------------------------" << endl;

  //  the purity of the reconstructed vertex
  cout << "               purity   ";
  for(vector< pair<double,unsigned int> >::iterator itrec=zrecv.begin(); itrec!=zrecv.end(); itrec++){
    cout << setw(7) << fixed << purity[itrec->second]/truthMatchedVertexTracks[itrec->second];
    if (itrec->second==0){cout << "*" ;}else{cout << " " ;}
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
      double z=(te->stateAtBeamLine().trackStateAtPCA()).position().z();
      double dz2= pow(RTe.dzError(),2)+pow(beamspot.BeamWidthX()/tantheta,2);
      
      if(ivassign==(int)rvmatch[itsim->second]){
	h["correctlyassigned"]->Fill(RTe.eta(),RTe.pt());
	h["ptcat"]->Fill(RTe.pt());
	h["etacat"]->Fill(RTe.eta());
	h["phicat"]->Fill(RTe.phi());
	h["dzcat"]->Fill(sqrt(dz2));
      }else{
	h["misassigned"]->Fill(RTe.eta(),RTe.pt());
	h["ptmis"]->Fill(RTe.pt());
	h["etamis"]->Fill(RTe.eta());
	h["phimis"]->Fill(RTe.phi());
	h["dzmis"]->Fill(sqrt(dz2));
	cout << "vertex " << setw(8) << fixed << ev->z;

	if (ivassign<0){
	  cout << " track lost                ";
	  // for some clusterizers there shouldn't be any lost tracks,
	  // are there differences in the track selection?
	}else{
	  cout << " track misassigned " << setw(8) << fixed << recVtxs->at(ivassign).z();
	}

	cout << "  track z=" << setw(8) << fixed  << RTe.vz() << "+/-" << RTe.dzError() << "  pt=" <<  setw(8) << fixed<< RTe.pt() << "    eta=" << setw(8) << fixed << RTe.eta()<< "  z'" << z << " sel=" <<theTrackFilter(*te) << endl;
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
  
  cout <<" PrimaryVertexAnalyzer4PU::analyzeVertexCollectionTP size=" << simEvt.size() << endl;
  if(simEvt.size()==0)return;

  printEventSummary(h, recVtxs,recTrks,simEvt);

  int nrecvtxs=0;
  //const int iSignal=0;  
  EncodedEventId iSignal=simEvt[0].eventId;
  h["npu0"]->Fill(simEvt.size());

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
  
  // --------------------------------------- match rec to sim ---------------------------------------
  for(reco::VertexCollection::const_iterator v=recVtxs->begin(); 
      v!=recVtxs->end(); ++v){
    if ( (v->ndof()<0) || (v->chi2()<=0) ) continue;  // skip clusters 
    nrecvtxs++;
    double  nmatch=-1;      // highest number of tracks in recvtx v found in any event
    double  nmatchany=0;    // tracks of a recvertex matching any sim vertex (i.e. any sim track(?))
    EncodedEventId evmatch;
    bool matchFound=false;
    //int evmatch=-1;

    for(vector<SimEvent>::iterator ev=simEvt.begin(); ev!=simEvt.end(); ev++){

      double n=0;  // number of tracks that are in both, the recvtx v and the event ev
      for(vector<TransientTrack>::iterator te=ev->tk.begin(); te!=ev->tk.end(); te++){
	const reco::Track&  RTe=te->track();
	 for(trackit_t tv=v->tracks_begin(); tv!=v->tracks_end(); tv++){
	   const reco::Track & RTv=*(tv->get());  
	   if(RTe.vz()==RTv.vz()) {n++; nmatchany++;}
	}
      }     

      if (n > nmatch){
	nmatch=n;
	//	evmatch=ev->event;
	evmatch=ev->eventId;
	matchFound=true;
      }
      if(nmatchany>0){
	Fill(h,"recPurity", n/v->tracksSize(), ev->eventId==iSignal);
      }

    }
    if (matchFound){
      nmatchany=getTruthMatchedVertexTracks(*v).size();
      if(v==recVtxs->begin()){
	if(nmatchany>0){	Fill(h,"recmatchPurityTag",nmatch/nmatchany,evmatch==iSignal);      }
      }else{
	if(nmatchany>0){	Fill(h,"recmatchPuritynoTag",nmatch/nmatchany,evmatch==iSignal);      }
      }
    }
  }
  h["nrecv"]->Fill(nrecvtxs);


  // --------------------------------------- match sim to rec  ---------------------------------------

  int npu1=0, npu2=0;

  for(vector<SimEvent>::iterator ev=simEvt.begin(); ev!=simEvt.end(); ev++){
    if(ev->tk.size()>0) npu1++;
    if(ev->tk.size()>1) npu2++;
    Fill(h,"nRecTrkInSimVtx",ev->tk.size(),ev->eventId==iSignal);
    Fill(h,"nPrimRecTrkInSimVtx",ev->tkprim.size(),ev->eventId==iSignal);
    Fill(h,"sumpt2rec",sqrt(ev->sumpt2rec),ev->eventId==iSignal);
    Fill(h,"sumpt2",sqrt(ev->sumpt2),ev->eventId==iSignal);
    Fill(h,"sumpt",sqrt(ev->sumpt),ev->eventId==iSignal);
    double nRecVWithTrk=0;  // vertices with tracks from this simvertex
    double  nmatch=0, ntmatch=0, zmatch=-99;

    for(reco::VertexCollection::const_iterator v=recVtxs->begin(); 
	v!=recVtxs->end(); ++v){
      if ( (v->ndof()<0) || (v->chi2()<=0) ) continue;  // skip clusters 
      // count tracks found in both, sim and rec
      double n=0;
      for(vector<TransientTrack>::iterator te=ev->tk.begin(); te!=ev->tk.end(); te++){
	const reco::Track&  RTe=te->track();
	 for(trackit_t tv=v->tracks_begin(); tv!=v->tracks_end(); tv++){
	   const reco::Track & RTv=*(tv->get());  
	   if(RTe.vz()==RTv.vz()) {n++;}
	}
      }

      if(n>0){	nRecVWithTrk++; }
      if (n > nmatch){
	nmatch=n; ntmatch=v->tracksSize(); zmatch=v->position().z();
      }
      
    }// end of reco vertex loop


    // nmatch is the highest number of tracks from this sim vertex found in a single reco-vertex
    if(ev->tk.size()>0){ Fill(h,"trkAssignmentEfficiency", nmatch/ev->tk.size(), ev->eventId==iSignal); };
    if(ev->tkprim.size()>0){ Fill(h,"primtrkAssignmentEfficiency", nmatch/ev->tkprim.size(), ev->eventId==iSignal); };
    // matched efficiency = efficiency for finding a reco vertex with > 50% of the simvertexs reconstructed tracks
    double ntsim=ev->tk.size(); // may be better to use the number of primary tracks here
    if(ntsim>0){
      Fill(h,"matchVtxFraction",nmatch/ntsim,ev->eventId==iSignal);
      if(nmatch/ntsim>=0.5){
	Fill(h,"matchVtxEfficiency",1.,ev->eventId==iSignal);
	if(ntsim>1){Fill(h,"matchVtxEfficiency2",1.,ev->eventId==iSignal);}
      }else{
	Fill(h,"matchVtxEfficiency",0.,ev->eventId==iSignal);
	if(ntsim>1){Fill(h,"matchVtxEfficiency2",0.,ev->eventId==iSignal);}
	if(ev->eventId==iSignal){
	cout << "Signal vertex not matched " <<  message << "  event=" << eventcounter_ << " nmatch=" << nmatch << "  ntsim=" << ntsim << endl;
	}
      }
    }

    Fill(h,"vtxMultiplicity",nRecVWithTrk,ev->eventId==iSignal);
    if(nmatch>=0.5*ntmatch){  // alternative efficiency definition: a reco vertex with > 50% primary tracks
      h["vtxFindingEfficiencyVsNtrk"]->Fill(ev->tk.size(),1.);
       if(ev->eventId==iSignal){
	 h["vtxFindingEfficiencyVsNtrkSignal"]->Fill(ev->tk.size(),1.);
      }else{
	h["vtxFindingEfficiencyVsNtrkPU"]->Fill(ev->tk.size(),1.);
      }
    }else{
      h["vtxFindingEfficiencyVsNtrk"]->Fill(ev->tk.size(),0.);
      if(ev->eventId==iSignal){
	h["vtxFindingEfficiencyVsNtrkSignal"]->Fill(ev->tk.size(),1.);
      }else{
	h["vtxFindingEfficiencyVsNtrkPU"]->Fill(ev->tk.size(),1.);
      }
    }

  }
  
  h["npu1"]->Fill(npu1);
  h["npu2"]->Fill(npu2);

  h["nrecvsnpu"]->Fill(npu1,nrecvtxs);

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
    h["TagVtxTrkPurity"]->Fill(n/ntruthmatched);
  }
  if (ev->tk.size()>0){
    cout << "TrackEfficiency = "<< n/ev->tk.size() <<endl;
    h["TagVtxTrkEfficiency"]->Fill(n/ev->tk.size());
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
  cout <<" PrimaryVertexAnalyzer4PU::analyzeVertexCollection (HepMC), simpvs=" << simpv.size() << endl;
  int nrectrks=recTrks->size();
  int nrecvtxs=recVtxs->size();
  int nseltrks=0;
  // extract dummy vertices representing clusters
  reco::VertexCollection clusters;
  reco::Vertex allSelected;
  double cpufit=0;
  double cpuclu=0;
  for(reco::VertexCollection::const_iterator v=recVtxs->begin(); 
      v!=recVtxs->end(); ++v){
    if ( (fabs(v->ndof()+1.)<0.0001) && (v->chi2()<=0) ){ 
      // this dummy vertex is for the full event
      allSelected=(*v);
      nseltrks=(allSelected.tracksSize());
      nrecvtxs--;
      cpuclu=-v->chi2();
      continue;
      break;
    }else if( (fabs(v->ndof()+2.)<0.0001) && (v->chi2()==0) ){
      // this is a cluster, not a vertex
      clusters.push_back(*v);
      h["cpuvsntrk"]->Fill(v->tracksSize(),fabs(v->y()));
      cpufit+=fabs(v->y());
      h["nclutrkall"]->Fill(v->tracksSize());
      h["selstat"]->Fill(v->x());
      //h["nclutrkvtx"]->Fill();// see below
      nrecvtxs--;
    }
  }
  h["cpuclu"]->Fill(cpuclu);
  h["cpufit"]->Fill(cpufit);
  h["cpucluvsntrk"]->Fill(nrectrks, cpuclu);
  
  
  if(simpv.size()>0){//this is mc
    double dsimrecx=0.;
    double dsimrecy=0.0011;
    double dsimrecz=0.0012;
    
    // vertex matching and efficiency bookkeeping
    int nsimtrk=0;
    for(std::vector<simPrimaryVertex>::iterator vsim=simpv.begin();
	vsim!=simpv.end(); vsim++){
      
      
      nsimtrk+=vsim->nGenTrk;
      // look for a matching reconstructed vertex
      vsim->recVtx=NULL;
      vsim->cluster=-1;
      
      for(reco::VertexCollection::const_iterator vrec=recVtxs->begin(); 
	  vrec!=recVtxs->end(); ++vrec){
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
	  }else if (simpv.size()==1){// only works if there is exactly one simpv for now
	    // now we have a recvertex without a matching simvertex, I would call this fake 
	    // however, the G4 info does not contain pile-up
	    // 	 std::cout << "PrimaryVertexAnalyzer4PU> unmatched recvtx " << std::endl;
	    // 	 std::cout << "z=" << vrec->z() << std::endl;
	    // 	 std::cout << "#tracks=" << vrec->tracksSize() << std::endl;
	    h["fakeVtxZ"]->Fill(vrec->z());
	    if (vrec->ndof()>=0.5) h["fakeVtxZNdofgt05"]->Fill(vrec->z());
	    h["fakeVtxNdof"]->Fill(vrec->ndof());
	    h["fakeVtxNtrk"]->Fill(vrec->tracksSize());
	  }
      }
      

      h["nsimtrk"]->Fill(float(nsimtrk));
      h["nrecsimtrk"]->Fill(float(vsim->nMatchedTracks));
      h["nrecnosimtrk"]->Fill(float(nsimtrk-vsim->nMatchedTracks));
      
      // histogram properties of matched vertices
      if (vsim->recVtx && ( fabs(vsim->recVtx->z()-vsim->z*simUnit_)<zmatch_ )){
	
	if(verbose_){std::cout <<"primary matched " << vsim->x << " " << vsim->y << " " << vsim->z << std:: endl;}
	// residuals an pulls with respect to simulated vertex
	h["resx"]->Fill( vsim->recVtx->x()-vsim->x*simUnit_ );
	h["resy"]->Fill( vsim->recVtx->y()-vsim->y*simUnit_ );
	h["resz"]->Fill( vsim->recVtx->z()-vsim->z*simUnit_ );
	h["resz10"]->Fill( vsim->recVtx->z()-vsim->z*simUnit_ );
	h["pullx"]->Fill( (vsim->recVtx->x()-vsim->x*simUnit_)/vsim->recVtx->xError() );
	h["pully"]->Fill( (vsim->recVtx->y()-vsim->y*simUnit_)/vsim->recVtx->yError() );
	h["pullz"]->Fill( (vsim->recVtx->z()-vsim->z*simUnit_)/vsim->recVtx->zError() );
	// residuals and pulls with respect to simulated vertex + offset between true beam and reconstructed beam
	//        double dxbeam=0.0322-myBeamSpot.x();
	//        double dybeam=0.0000-myBeamSpot.y();
	//        double dzbeam=0.0000-myBeamSpot.z();
	h["resxr"]->Fill( vsim->recVtx->x()-vsim->x*simUnit_-dsimrecx);
	h["resyr"]->Fill( vsim->recVtx->y()-vsim->y*simUnit_-dsimrecy );
	h["reszr"]->Fill( vsim->recVtx->z()-vsim->z*simUnit_-dsimrecz);
	h["pullxr"]->Fill( (vsim->recVtx->x()-vsim->x*simUnit_-dsimrecx)/vsim->recVtx->xError() );
	h["pullyr"]->Fill( (vsim->recVtx->y()-vsim->y*simUnit_-dsimrecy)/vsim->recVtx->yError() );
	h["pullzr"]->Fill( (vsim->recVtx->z()-vsim->z*simUnit_-dsimrecz)/vsim->recVtx->zError() );



	// efficiency with zmatch within 500 um (or whatever zmatch is)
	h["eff"]->Fill( 1.);
	if(simpv.size()==1){
	  if (vsim->recVtx==&(*recVtxs->begin())){
	    h["efftag"]->Fill( 1.); 
	  }else{
	    h["efftag"]->Fill( 0.); 
	    cout << "signal vertex not tagged " << message << " " << eventcounter_ << endl;
	  }
	}
	
	h["effvseta"]->Fill(vsim->ptot.pseudoRapidity(),1.);
	h["effvsptsq"]->Fill(vsim->ptsq,1.);
	h["effvsnsimtrk"]->Fill(vsim->nGenTrk,1.);
	h["effvsnrectrk"]->Fill(nrectrks,1.);
	h["effvsnseltrk"]->Fill(nseltrks,1.);
	h["effvsz"]->Fill(vsim->z*simUnit_,1.);
	h["effvsz2"]->Fill(vsim->z*simUnit_,1.);
	h["effvsr"]->Fill(sqrt(vsim->x*vsim->x+vsim->y*vsim->y)*simUnit_,1.);
	

      }else{  // no matching rec vertex found for this simvertex
       
	if(verbose_){std::cout << "primary not found "  << message << " " << eventcounter_ << "  x=" <<vsim->x << "  y=" << vsim->y << " z=" << vsim->z << " nGenTrk=" << vsim->nGenTrk << std::endl;}
	int mistype=0;
	if (vsim->recVtx){
	  std::cout << "nearest recvertex at " << vsim->recVtx->z() << "   dz=" << vsim->recVtx->z()-vsim->z*simUnit_ << std::endl;
	  
	  if (fabs(vsim->recVtx->z()-vsim->z*simUnit_)<0.2 ){
	    h["effvsz2"]->Fill(vsim->z*simUnit_,1.);
	  }
	  
	  if (fabs(vsim->recVtx->z()-vsim->z*simUnit_)<0.5 ){
	    std::cout << "type 1, lousy z vertex" << std::endl;
	    h["zlost1"]->Fill( vsim->z*simUnit_,1.);
	    mistype=1;
	  }else{
	    std::cout << "type 2a no vertex anywhere near" << std::endl;
	    mistype=2;
	  }
	}else{// no recVtx at all
	  mistype=2;
	  std::cout << "type 2b, no vertex at all" << std::endl;
	}
	
	if(mistype==2){
	  int selstat=-3;
	  // no matching vertex found, is there a cluster?
	  for(unsigned int iclu=0; iclu<clusters.size(); iclu++){
	    if( fabs(clusters[iclu].position().z()-vsim->z*simUnit_) < 0.1 ){
	      selstat=int(clusters[iclu].position().x()+0.1);
	      std::cout << "matching cluster found with selstat=" << clusters[iclu].position().x() << std::endl;
	    }
	  }
	  if (selstat==0){
	    std::cout << "vertex rejected (distance to beam)" << std::endl;
	    h["zlost3"]->Fill( vsim->z*simUnit_,1.);
	  }else if(selstat==-1){
	    std::cout << "vertex invalid" << std::endl;
	    h["zlost4"]->Fill( vsim->z*simUnit_,1.);
	  }else if(selstat==1){
	    std::cout << "vertex accepted, this cannot be right!!!!!!!!!!" << std::endl;
	  }else if(selstat==-2){
	    std::cout << "dont know what this means !!!!!!!!!!" << std::endl;
	  }else if(selstat==-3){
	    std::cout << "no matching cluster found " << std::endl;
	    h["zlost2"]->Fill( vsim->z*simUnit_,1.);
	  }else{
	    std::cout << "dont know what this means either !!!!!!!!!!" << selstat << std::endl;
	  }
	}//
	
	
	h["eff"]->Fill( 0.);
	if(simpv.size()==1){ h["efftag"]->Fill( 0.); }
	
	//h["effvseta"]->Fill(vsim->ptot.pseudoRapidity(),0.);
	h["effvsptsq"]->Fill(vsim->ptsq,0.);
	h["effvsnsimtrk"]->Fill(float(vsim->nGenTrk),0.);
	h["effvsnrectrk"]->Fill(nrectrks,0.);
	h["effvsnseltrk"]->Fill(nseltrks,0.);
	h["effvsz"]->Fill(vsim->z*simUnit_,0.);
	h["effvsr"]->Fill(sqrt(vsim->x*vsim->x+vsim->y*vsim->y)*simUnit_,0.);
	
      } // no recvertex for this simvertex


    }

  // end of sim/rec matching 
   
     
   // purity of event vertex tags
    if (recVtxs->size()>0){
      Double_t dz=(*recVtxs->begin()).z() - (*simpv.begin()).z*simUnit_;
      h["zdistancetag"]->Fill(dz);
      h["abszdistancetag"]->Fill(fabs(dz));
      if( fabs(dz)<zmatch_){
	h["puritytag"]->Fill(1.);
      }else{
	// bad tag: the true primary was more than 500 um away from the tagged primary
	h["puritytag"]->Fill(0.);
      }
    }
    
  }else{
    cout << "PrimaryVertexAnalyzer4PU::analyzeVertexCollection:  simPV is empty!" << endl;
  }

  // -----------------  reconstructed tracks  ------------------------
  for(reco::TrackCollection::const_iterator t=recTrks->begin();
      t!=recTrks->end(); ++t){
    h["recRapidity"]->Fill(t->eta());
    fillTrackHistos(h,"all",*t);
  }
  // bachelor tracks
  for(unsigned int iclu=0; iclu<clusters.size(); iclu++){
    if (clusters[iclu].tracksSize()==1){
      for(trackit_t t = clusters[iclu].tracks_begin(); 
	  t!=clusters[iclu].tracks_end(); t++){
	//	trackWeight(t);
	fillTrackHistos(h,"bachelor",**t);
	
      }
    }
  }


  // -----------------  reconstructed vertices  ------------------------

  h["nclu"]->Fill(clusters.size());
  h["nseltrk"]->Fill(nseltrks);

  //properties of reconstructed vertices

  h["nrecvtx"]->Fill(nrecvtxs);//  h["nrecvtx"]->Fill(recVtxs->size());
  h["nrectrk"]->Fill(nrectrks);


  if(nrecvtxs==0) {h["nrectrk0vtx"]->Fill(nrectrks);}
  if(nrecvtxs==0) {h["nseltrk0vtx"]->Fill(nseltrks);}
  if(nrecvtxs==0) {h["nclu0vtx"]->Fill(clusters.size());}

  // test track links, use reconstructed vertices
  for(reco::VertexCollection::const_iterator v=recVtxs->begin(); 
      v!=recVtxs->end(); ++v){
    if(v->ndof()<0) continue;
    try {
      for(trackit_t t = v->tracks_begin(); 
	  t!=v->tracks_end(); t++) {
	// illegal charge
        if ( (**t).charge() < -1 || (**t).charge() > 1 ) {
	  h["tklinks"]->Fill(0.);
        }
        else {
	  h["tklinks"]->Fill(1.);
        }
      }
    } catch (...) {
      // exception thrown when trying to use linked track
      h["tklinks"]->Fill(0.);
    

      h["nbtksinvtx"]->Fill(v->tracksSize());
      h["nbtksinvtx2"]->Fill(v->tracksSize());
      h["vtxchi2"]->Fill(v->chi2());
      h["vtxndf"]->Fill(v->ndof());
      h["vtxprob"]->Fill(ChiSquaredProbability(v->chi2() ,v->ndof()));
      h["vtxndfvsntk"]->Fill(v->tracksSize(), v->ndof());
      h["vtxndfoverntk"]->Fill(v->ndof()/v->tracksSize());
      if(v->ndof()>0.5){  // enter only vertices that really contain tracks
	h["xrec"]->Fill(v->position().x());
	h["yrec"]->Fill(v->position().y());
	h["zrec"]->Fill(v->position().z());
	h["xrec2"]->Fill(v->position().x());
	h["yrec2"]->Fill(v->position().y());
	h["zrec2"]->Fill(v->position().z());
	h["xrec3"]->Fill(v->position().x()-myBeamSpot.x());
	h["yrec3"]->Fill(v->position().y()-myBeamSpot.y());
	h["zrec3"]->Fill(v->position().z()-myBeamSpot.z());
	// look at the tagged vertex separately
	if (v==recVtxs->begin()){
	  h["nbtksinvtxTag"]->Fill(v->tracksSize());
	  h["nbtksinvtxTag2"]->Fill(v->tracksSize());
	  h["xrectag"]->Fill(v->position().x());
	  h["yrectag"]->Fill(v->position().y());
	  h["zrectag"]->Fill(v->position().z());
	}else{
	  h["nbtksinvtxPU"]->Fill(v->tracksSize());
	  h["nbtksinvtxPU2"]->Fill(v->tracksSize());
	}
	
	// resolution vs number of tracks
	h["xresvsntrk"]->Fill(v->tracksSize(),v->xError());
	h["yresvsntrk"]->Fill(v->tracksSize(),v->yError());
	h["zresvsntrk"]->Fill(v->tracksSize(),v->zError());
	
      }
      
      for(unsigned int iclu=0; iclu<clusters.size(); iclu++){
	if( fabs(clusters[iclu].position().z()-v->position().z()) < 0.0001 ){
	  h["nclutrkvtx"]->Fill(clusters[iclu].tracksSize());
	}
      }
      
      
      bool problem = false;
      h["nans"]->Fill(1.,isnan(v->position().x())*1.);
      h["nans"]->Fill(2.,isnan(v->position().y())*1.);
      h["nans"]->Fill(3.,isnan(v->position().z())*1.);
      
      int index = 3;
      for (int i = 0; i != 3; i++) {
	for (int j = i; j != 3; j++) {
	  index++;
	  h["nans"]->Fill(index*1., isnan(v->covariance(i, j))*1.);
	  if (isnan(v->covariance(i, j))) problem = true;
	  // in addition, diagonal element must be positive
	  if (j == i && v->covariance(i, j) < 0) {
	    h["nans"]->Fill(index*1., 1.);
	    problem = true;
	  }
	}
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
	      gsl_vector_view evec_i 
		= gsl_matrix_column (evec, i);
	      
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
    }//catch
  }
}

