/////////////////////////////
// Track Trigger Checklist //
// L1TkCluster             //
// L1TkStub                //
//                         //
// Nicola Pozzobon - 2011  //
// Sebastien Viret         //
/////////////////////////////

#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "Geometry/Records/interface/StackedTrackerGeometryRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/StackedTrackerGeometry.h"
#include "SimDataFormats/SLHC/interface/StackedTrackerTypes.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticle.h" /* TEST PURPOSE!!! */
#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include <TH1D.h>
#include <TH2D.h>

//////////////////////////////
//                          //
//     CLASS DEFINITION     //
//                          //
//////////////////////////////

class ValidateClusterStub : public edm::EDAnalyzer
{
  /// Public methods
  public:
    /// Constructor/destructor
    explicit ValidateClusterStub(const edm::ParameterSet& iConfig);
    virtual ~ValidateClusterStub();
    // Typical methods used on Loops over events
    virtual void beginJob();
    virtual void endJob();
    virtual void analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup);

  /// Private methods and variables
  private:

    /// SimTrack and SimVertex
    TH2D* hSimVtx_XY;
    TH2D* hSimVtx_RZ;

    TH1D* hSimTrk_Pt;
    TH1D* hSimTrk_Eta_Pt10;
    TH1D* hSimTrk_Phi_Pt10;

    /// Global positions of L1TkClusters
    TH2D* hCluster_Barrel_XY;
    TH2D* hCluster_Barrel_XY_Zoom;
    TH2D* hCluster_Endcap_Fw_XY;
    TH2D* hCluster_Endcap_Bw_XY;
    TH2D* hCluster_RZ;
    TH2D* hCluster_Endcap_Fw_RZ_Zoom;
    TH2D* hCluster_Endcap_Bw_RZ_Zoom;

    TH1D* hCluster_IMem_Barrel;
    TH1D* hCluster_IMem_Endcap;
    TH1D* hCluster_OMem_Barrel;
    TH1D* hCluster_OMem_Endcap;

    TH1D* hCluster_Gen_Barrel;
    TH1D* hCluster_Unkn_Barrel;
    TH1D* hCluster_Comb_Barrel;
    TH1D* hCluster_Gen_Endcap;
    TH1D* hCluster_Unkn_Endcap;
    TH1D* hCluster_Comb_Endcap;

    TH2D* hCluster_PID;
    TH2D* hCluster_W;
    
    /// Global positions of L1TkStubs
    TH2D* hStub_Barrel_XY;
    TH2D* hStub_Barrel_XY_Zoom;
    TH2D* hStub_Endcap_Fw_XY;
    TH2D* hStub_Endcap_Bw_XY;
    TH2D* hStub_RZ;
    TH2D* hStub_Endcap_Fw_RZ_Zoom;
    TH2D* hStub_Endcap_Bw_RZ_Zoom;

    TH1D* hStub_Barrel;
    TH1D* hStub_Endcap;

    TH1D* hStub_Gen_Barrel;
    TH1D* hStub_Unkn_Barrel;
    TH1D* hStub_Comb_Barrel;
    TH1D* hStub_Gen_Endcap;
    TH1D* hStub_Unkn_Endcap;
    TH1D* hStub_Comb_Endcap;

    TH1D* hStub_PID;
    TH2D* hStub_Barrel_W;
    TH2D* hStub_Barrel_O;
    TH2D* hStub_Endcap_W;
    TH2D* hStub_Endcap_O;

    /// Denominator for Stub Prod Eff
    std::map< unsigned int, TH1D* > mapCluLayer_hSimTrk_Pt;
    std::map< unsigned int, TH1D* > mapCluLayer_hSimTrk_Eta_Pt10;
    std::map< unsigned int, TH1D* > mapCluLayer_hSimTrk_Phi_Pt10;
    std::map< unsigned int, TH1D* > mapCluDisk_hSimTrk_Pt;
    std::map< unsigned int, TH1D* > mapCluDisk_hSimTrk_Eta_Pt10;
    std::map< unsigned int, TH1D* > mapCluDisk_hSimTrk_Phi_Pt10;
    /// Numerator for Stub Prod Eff
    std::map< unsigned int, TH1D* > mapStubLayer_hSimTrk_Pt;
    std::map< unsigned int, TH1D* > mapStubLayer_hSimTrk_Eta_Pt10;
    std::map< unsigned int, TH1D* > mapStubLayer_hSimTrk_Phi_Pt10;
    std::map< unsigned int, TH1D* > mapStubDisk_hSimTrk_Pt;
    std::map< unsigned int, TH1D* > mapStubDisk_hSimTrk_Eta_Pt10;
    std::map< unsigned int, TH1D* > mapStubDisk_hSimTrk_Phi_Pt10;

    /// Comparison of Stubs to SimTracks
    std::map< unsigned int, TH2D* > mapStubLayer_hStub_InvPt_SimTrk_InvPt;
    std::map< unsigned int, TH2D* > mapStubLayer_hStub_Pt_SimTrk_Pt;
    std::map< unsigned int, TH2D* > mapStubLayer_hStub_Eta_SimTrk_Eta;
    std::map< unsigned int, TH2D* > mapStubLayer_hStub_Phi_SimTrk_Phi;
    std::map< unsigned int, TH2D* > mapStubDisk_hStub_InvPt_SimTrk_InvPt;
    std::map< unsigned int, TH2D* > mapStubDisk_hStub_Pt_SimTrk_Pt;
    std::map< unsigned int, TH2D* > mapStubDisk_hStub_Eta_SimTrk_Eta;
    std::map< unsigned int, TH2D* > mapStubDisk_hStub_Phi_SimTrk_Phi;

    /// Residuals
    std::map< unsigned int, TH2D* > mapStubLayer_hStub_InvPtRes_SimTrk_Eta;
    std::map< unsigned int, TH2D* > mapStubLayer_hStub_PtRes_SimTrk_Eta;
    std::map< unsigned int, TH2D* > mapStubLayer_hStub_EtaRes_SimTrk_Eta;
    std::map< unsigned int, TH2D* > mapStubLayer_hStub_PhiRes_SimTrk_Eta;
    std::map< unsigned int, TH2D* > mapStubDisk_hStub_InvPtRes_SimTrk_Eta;
    std::map< unsigned int, TH2D* > mapStubDisk_hStub_PtRes_SimTrk_Eta;
    std::map< unsigned int, TH2D* > mapStubDisk_hStub_EtaRes_SimTrk_Eta;
    std::map< unsigned int, TH2D* > mapStubDisk_hStub_PhiRes_SimTrk_Eta;

    /// Stub Width vs Pt
    std::map< unsigned int, TH2D* > mapStubLayer_hStub_W_SimTrk_Pt;
    std::map< unsigned int, TH2D* > mapStubLayer_hStub_W_SimTrk_InvPt;
    std::map< unsigned int, TH2D* > mapStubDisk_hStub_W_SimTrk_Pt;
    std::map< unsigned int, TH2D* > mapStubDisk_hStub_W_SimTrk_InvPt;

    /// Containers of parameters passed by python
    /// configuration file
    edm::ParameterSet config;

    bool testedGeometry;
    bool DebugMode;
};

//////////////////////////////////
//                              //
//     CLASS IMPLEMENTATION     //
//                              //
//////////////////////////////////

//////////////
// CONSTRUCTOR
ValidateClusterStub::ValidateClusterStub(edm::ParameterSet const& iConfig) : 
  config(iConfig)
{
  /// Insert here what you need to initialize
  DebugMode = iConfig.getParameter< bool >("DebugMode");
}

/////////////
// DESTRUCTOR
ValidateClusterStub::~ValidateClusterStub()
{
  /// Insert here what you need to delete
  /// when you close the class instance
}  

//////////
// END JOB
void ValidateClusterStub::endJob()//edm::Run& run, const edm::EventSetup& iSetup
{
  /// Things to be done at the exit of the event Loop
  std::cerr << " ValidateClusterStub::endJob" << std::endl;
  /// End of things to be done at the exit from the event Loop
}

////////////
// BEGIN JOB
void ValidateClusterStub::beginJob()
{
  /// Initialize all slave variables
  /// mainly histogram ranges and resolution
  testedGeometry = false;

  std::ostringstream histoName;
  std::ostringstream histoTitle;

  /// Things to be done before entering the event Loop
  std::cerr << " ValidateClusterStub::beginJob" << std::endl;

  /// Book histograms etc
  edm::Service<TFileService> fs;

  /// Prepare for LogXY Plots
  int NumBins = 200;
  double MinPt = 0.0;
  double MaxPt = 100.0;

  double* BinVec = new double[NumBins+1];
  for ( int iBin = 0; iBin < NumBins + 1; iBin++ )
  {
    double temp = pow( 10, (- NumBins + iBin)/(MaxPt - MinPt)  );
    BinVec[ iBin ] = temp;
  }

  /// SimTrack and SimVertex
  hSimVtx_XY      = fs->make<TH2D>( "hSimVtx_XY", "SimVtx y vs. x",    200, -0.4, 0.4, 200, -0.4, 0.4 );
  hSimVtx_RZ      = fs->make<TH2D>( "hSimVtx_RZ", "SimVtx #rho vs. z", 200,  -50,  50, 200,    0, 0.4 );
  hSimVtx_XY->Sumw2();
  hSimVtx_RZ->Sumw2();

  hSimTrk_Pt       = fs->make<TH1D>( "hSimTrk_Pt",       "SimTrk p_{T}",                   100,     0,   50 );
  hSimTrk_Eta_Pt10 = fs->make<TH1D>( "hSimTrk_Eta_Pt10", "SimTrk #eta (p_{T} > 10 GeV/c)", 180, -M_PI, M_PI );
  hSimTrk_Phi_Pt10 = fs->make<TH1D>( "hSimTrk_Phi_Pt10", "SimTrk #phi (p_{T} > 10 GeV/c)", 180, -M_PI, M_PI );
  hSimTrk_Pt->Sumw2();
  hSimTrk_Eta_Pt10->Sumw2();
  hSimTrk_Phi_Pt10->Sumw2();

  /// Global position of L1TkCluster
  hCluster_Barrel_XY          = fs->make<TH2D>( "hCluster_Barrel_XY",         "L1TkCluster Barrel y vs. x",              960, -120,  120, 960, -120, 120 );
  hCluster_Barrel_XY_Zoom     = fs->make<TH2D>( "hCluster_Barrel_XY_Zoom",    "L1TkCluster Barrel y vs. x",              960,   30,   60, 960,  -15,  15 );
  hCluster_Endcap_Fw_XY       = fs->make<TH2D>( "hCluster_Endcap_Fw_XY",      "L1TkCluster Forward Endcap y vs. x",      960, -120,  120, 960, -120, 120 );
  hCluster_Endcap_Bw_XY       = fs->make<TH2D>( "hCluster_Endcap_Bw_XY",      "L1TkCluster Backward Endcap y vs. x",     960, -120,  120, 960, -120, 120 );
  hCluster_RZ                 = fs->make<TH2D>( "hCluster_RZ",                "L1TkCluster #rho vs. z",                  900, -300,  300, 480,    0, 120 );
  hCluster_Endcap_Fw_RZ_Zoom  = fs->make<TH2D>( "hCluster_Endcap_Fw_RZ_Zoom", "L1TkCluster Forward Endcap #rho vs. z",   960,  140,  170, 960,   30,  60 );
  hCluster_Endcap_Bw_RZ_Zoom  = fs->make<TH2D>( "hCluster_Endcap_Bw_RZ_Zoom", "L1TkCluster Backward Endcap #rho vs. z",  960, -170, -140, 960,   70, 100 );
  hCluster_Barrel_XY->Sumw2();
  hCluster_Barrel_XY_Zoom->Sumw2();
  hCluster_Endcap_Fw_XY->Sumw2();
  hCluster_Endcap_Bw_XY->Sumw2();
  hCluster_RZ->Sumw2();
  hCluster_Endcap_Fw_RZ_Zoom->Sumw2();
  hCluster_Endcap_Bw_RZ_Zoom->Sumw2();

  hCluster_IMem_Barrel = fs->make<TH1D>("hCluster_IMem_Barrel", "Inner L1TkCluster Stack", 12, -0.5, 11.5 );
  hCluster_IMem_Endcap = fs->make<TH1D>("hCluster_IMem_Endcap", "Inner L1TkCluster Stack", 12, -0.5, 11.5 );
  hCluster_OMem_Barrel = fs->make<TH1D>("hCluster_OMem_Barrel", "Outer L1TkCluster Stack", 12, -0.5, 11.5 );
  hCluster_OMem_Endcap = fs->make<TH1D>("hCluster_OMem_Endcap", "Outer L1TkCluster Stack", 12, -0.5, 11.5 );
  hCluster_IMem_Barrel->Sumw2();
  hCluster_IMem_Endcap->Sumw2();
  hCluster_OMem_Barrel->Sumw2();
  hCluster_OMem_Endcap->Sumw2();

  hCluster_Gen_Barrel  = fs->make<TH1D>("hCluster_Gen_Barrel",  "Genuine L1TkCluster Stack",       12, -0.5, 11.5 ); 
  hCluster_Unkn_Barrel = fs->make<TH1D>("hCluster_Unkn_Barrel", "Unknown  L1TkCluster Stack",      12, -0.5, 11.5 ); 
  hCluster_Comb_Barrel = fs->make<TH1D>("hCluster_Comb_Barrel", "Combinatorial L1TkCluster Stack", 12, -0.5, 11.5 ); 
  hCluster_Gen_Endcap  = fs->make<TH1D>("hCluster_Gen_Endcap",  "Genuine L1TkCluster Stack",       12, -0.5, 11.5 ); 
  hCluster_Unkn_Endcap = fs->make<TH1D>("hCluster_Unkn_Endcap", "Unknown  L1TkCluster Stack",      12, -0.5, 11.5 ); 
  hCluster_Comb_Endcap = fs->make<TH1D>("hCluster_Comb_Endcap", "Combinatorial L1TkCluster Stack", 12, -0.5, 11.5 ); 
  hCluster_Gen_Barrel->Sumw2();
  hCluster_Unkn_Barrel->Sumw2();
  hCluster_Comb_Barrel->Sumw2();
  hCluster_Gen_Endcap->Sumw2();
  hCluster_Unkn_Endcap->Sumw2();
  hCluster_Comb_Endcap->Sumw2();

  hCluster_PID   = fs->make<TH2D>("hCluster_PID", "L1TkCluster PID (Member)", 501, -250.5, 250.5, 2, -0.5, 1.5 );
  hCluster_W     = fs->make<TH2D>("hCluster_W", "L1TkCluster Width (Member)",  10,   -0.5,   9.5, 2, -0.5, 1.5 );
  hCluster_PID->Sumw2();
  hCluster_W->Sumw2();

  /// Global position of L1TkStub
  hStub_Barrel_XY          = fs->make<TH2D>( "hStub_Barrel_XY",         "L1TkStub Barrel y vs. x",              960, -120,  120, 960, -120, 120 );
  hStub_Barrel_XY_Zoom     = fs->make<TH2D>( "hStub_Barrel_XY_Zoom",    "L1TkStub Barrel y vs. x",              960,   30,   60, 960,  -15,  15 );
  hStub_Endcap_Fw_XY       = fs->make<TH2D>( "hStub_Endcap_Fw_XY",      "L1TkStub Forward Endcap y vs. x",      960, -120,  120, 960, -120, 120 );
  hStub_Endcap_Bw_XY       = fs->make<TH2D>( "hStub_Endcap_Bw_XY",      "L1TkStub Backward Endcap y vs. x",     960, -120,  120, 960, -120, 120 );
  hStub_RZ                 = fs->make<TH2D>( "hStub_RZ",                "L1TkStub #rho vs. z",                  900, -300,  300, 480,    0, 120 );
  hStub_Endcap_Fw_RZ_Zoom  = fs->make<TH2D>( "hStub_Endcap_Fw_RZ_Zoom", "L1TkStub Forward Endcap #rho vs. z",   960,  140,  170, 960,   30,  60 );
  hStub_Endcap_Bw_RZ_Zoom  = fs->make<TH2D>( "hStub_Endcap_Bw_RZ_Zoom", "L1TkStub Backward Endcap #rho vs. z",  960, -170, -140, 960,   70, 100 );
  hStub_Barrel_XY->Sumw2();
  hStub_Barrel_XY_Zoom->Sumw2();
  hStub_Endcap_Fw_XY->Sumw2();
  hStub_Endcap_Bw_XY->Sumw2();
  hStub_RZ->Sumw2();
  hStub_Endcap_Fw_RZ_Zoom->Sumw2();
  hStub_Endcap_Bw_RZ_Zoom->Sumw2();

  hStub_Barrel     = fs->make<TH1D>("hStub_Barrel", "L1TkStub Stack", 12, -0.5, 11.5 );
  hStub_Endcap     = fs->make<TH1D>("hStub_Endcap", "L1TkStub Stack", 12, -0.5, 11.5 );
  hStub_Barrel->Sumw2();
  hStub_Endcap->Sumw2();

  hStub_Gen_Barrel  = fs->make<TH1D>("hStub_Gen_Barrel",  "Genuine L1TkStub Stack",       12, -0.5, 11.5 ); 
  hStub_Unkn_Barrel = fs->make<TH1D>("hStub_Unkn_Barrel", "Unknown  L1TkStub Stack",      12, -0.5, 11.5 ); 
  hStub_Comb_Barrel = fs->make<TH1D>("hStub_Comb_Barrel", "Combinatorial L1TkStub Stack", 12, -0.5, 11.5 ); 
  hStub_Gen_Endcap  = fs->make<TH1D>("hStub_Gen_Endcap",  "Genuine L1TkStub Stack",       12, -0.5, 11.5 ); 
  hStub_Unkn_Endcap = fs->make<TH1D>("hStub_Unkn_Endcap", "Unknown  L1TkStub Stack",      12, -0.5, 11.5 ); 
  hStub_Comb_Endcap = fs->make<TH1D>("hStub_Comb_Endcap", "Combinatorial L1TkStub Stack", 12, -0.5, 11.5 ); 
  hStub_Gen_Barrel->Sumw2();
  hStub_Unkn_Barrel->Sumw2();
  hStub_Comb_Barrel->Sumw2();
  hStub_Gen_Endcap->Sumw2();
  hStub_Unkn_Endcap->Sumw2();
  hStub_Comb_Endcap->Sumw2();

  hStub_PID      = fs->make<TH1D>("hStub_PID",      "L1TkStub PID",                            501, -250.5, 250.5 );
  hStub_Barrel_W = fs->make<TH2D>("hStub_Barrel_W", "L1TkStub Post-Corr Displacement (Layer)",  12, -0.5, 11.5, 43, -10.75, 10.75 );
  hStub_Barrel_O = fs->make<TH2D>("hStub_Barrel_O", "L1TkStub Offset (Layer)",                  12, -0.5, 11.5, 43, -10.75, 10.75 );
  hStub_Endcap_W = fs->make<TH2D>("hStub_Endcap_W", "L1TkStub Post-Corr Displacement (Layer)",  12, -0.5, 11.5, 43, -10.75, 10.75 );
  hStub_Endcap_O = fs->make<TH2D>("hStub_Endcap_O", "L1TkStub Offset (Layer)",                  12, -0.5, 11.5, 43, -10.75, 10.75 );

  hStub_PID->Sumw2();
  hStub_Barrel_W->Sumw2();
  hStub_Barrel_O->Sumw2();
  hStub_Endcap_W->Sumw2();
  hStub_Endcap_O->Sumw2();

  /// Stub Production Efficiency and comparison to SimTrack
  for ( unsigned int stackIdx = 0; stackIdx < 12; stackIdx++ )
  {
    /// BARREL

    /// Denominators
    histoName.str("");  histoName << "hSimTrk_Pt_Clu_L" << stackIdx;
    histoTitle.str(""); histoTitle << "SimTrk p_{T}, Cluster, Barrel Stack " << stackIdx;
    mapCluLayer_hSimTrk_Pt[ stackIdx ] = fs->make<TH1D>( histoName.str().c_str(),  histoTitle.str().c_str(),
                                                         100, 0, 50 );
    histoName.str("");  histoName << "hSimTrk_Eta_Pt10_Clu_L" << stackIdx;
    histoTitle.str(""); histoTitle << "SimTrk #eta (p_{T} > 10 GeV/c), Cluster, Barrel Stack " << stackIdx;
    mapCluLayer_hSimTrk_Eta_Pt10[ stackIdx ] = fs->make<TH1D>( histoName.str().c_str(),  histoTitle.str().c_str(),
                                                               180, -M_PI, M_PI );
    histoName.str("");  histoName << "hSimTrk_Phi_Pt10_Clu_L" << stackIdx;
    histoTitle.str(""); histoTitle << "SimTrk #phi (p_{T} > 10 GeV/c), Cluster, Barrel Stack " << stackIdx;
    mapCluLayer_hSimTrk_Phi_Pt10[ stackIdx ] = fs->make<TH1D>( histoName.str().c_str(),  histoTitle.str().c_str(),
                                                               180, -M_PI, M_PI );
    mapCluLayer_hSimTrk_Pt[ stackIdx ]->Sumw2();
    mapCluLayer_hSimTrk_Eta_Pt10[ stackIdx ]->Sumw2();
    mapCluLayer_hSimTrk_Phi_Pt10[ stackIdx ]->Sumw2();

    /// Numerators GeV/c
    histoName.str("");  histoName << "hSimTrk_Pt_Stub_L" << stackIdx;
    histoTitle.str(""); histoTitle << "SimTrk p_{T}, Stub, Barrel Stack " << stackIdx;
    mapStubLayer_hSimTrk_Pt[ stackIdx ] = fs->make<TH1D>( histoName.str().c_str(),  histoTitle.str().c_str(),
                                                          100, 0, 50 );
    histoName.str("");  histoName << "hSimTrk_Eta_Pt10_Stub_L" << stackIdx;
    histoTitle.str(""); histoTitle << "SimTrk #eta (p_{T} > 10 GeV/c), Stub, Barrel Stack " << stackIdx;
    mapStubLayer_hSimTrk_Eta_Pt10[ stackIdx ] = fs->make<TH1D>( histoName.str().c_str(),  histoTitle.str().c_str(),
                                                                180, -M_PI, M_PI );
    histoName.str("");  histoName << "hSimTrk_Phi_Pt10_Stub_L" << stackIdx;
    histoTitle.str(""); histoTitle << "SimTrk #phi (p_{T} > 10 GeV/c), Stub, Barrel Stack " << stackIdx;
    mapStubLayer_hSimTrk_Phi_Pt10[ stackIdx ] = fs->make<TH1D>( histoName.str().c_str(),  histoTitle.str().c_str(),
                                                                180, -M_PI, M_PI );
    mapStubLayer_hSimTrk_Pt[ stackIdx ]->Sumw2();
    mapStubLayer_hSimTrk_Eta_Pt10[ stackIdx ]->Sumw2();
    mapStubLayer_hSimTrk_Phi_Pt10[ stackIdx ]->Sumw2();

    /// Comparison to SimTrack
    histoName.str("");  histoName << "hStub_InvPt_SimTrk_InvPt_L" << stackIdx;
    histoTitle.str(""); histoTitle << "Stub p_{T}^{-1} vs. SimTrk p_{T}^{-1}, Barrel Stack " << stackIdx;
    mapStubLayer_hStub_InvPt_SimTrk_InvPt[ stackIdx ] = fs->make<TH2D>( histoName.str().c_str(),  histoTitle.str().c_str(),
                                                                        200, 0.0, 0.8,
                                                                        200, 0.0, 0.8 );
    mapStubLayer_hStub_InvPt_SimTrk_InvPt[ stackIdx ]->GetXaxis()->Set( NumBins, BinVec );
    mapStubLayer_hStub_InvPt_SimTrk_InvPt[ stackIdx ]->GetYaxis()->Set( NumBins, BinVec );
    mapStubLayer_hStub_InvPt_SimTrk_InvPt[ stackIdx ]->Sumw2();

    histoName.str("");  histoName << "hStub_Pt_SimTrk_Pt_L" << stackIdx;
    histoTitle.str(""); histoTitle << "Stub p_{T} vs. SimTrk p_{T}, Barrel Stack " << stackIdx;
    mapStubLayer_hStub_Pt_SimTrk_Pt[ stackIdx ] = fs->make<TH2D>( histoName.str().c_str(),  histoTitle.str().c_str(),
                                                                  100, 0, 50,
                                                                  100, 0, 50 );
    mapStubLayer_hStub_Pt_SimTrk_Pt[ stackIdx ]->Sumw2();

    histoName.str("");  histoName << "hStub_Eta_SimTrk_Eta_L" << stackIdx;
    histoTitle.str(""); histoTitle << "Stub #eta vs. SimTrk #eta, Barrel Stack " << stackIdx;
    mapStubLayer_hStub_Eta_SimTrk_Eta[ stackIdx ] = fs->make<TH2D>( histoName.str().c_str(),  histoTitle.str().c_str(),
                                                                    180, -M_PI, M_PI,
                                                                    180, -M_PI, M_PI );
    mapStubLayer_hStub_Eta_SimTrk_Eta[ stackIdx ]->Sumw2();

    histoName.str("");  histoName << "hStub_Phi_SimTrk_Phi_L" << stackIdx;
    histoTitle.str(""); histoTitle << "Stub #phi vs. SimTrk #phi, Barrel Stack " << stackIdx;
    mapStubLayer_hStub_Phi_SimTrk_Phi[ stackIdx ] = fs->make<TH2D>( histoName.str().c_str(),  histoTitle.str().c_str(),
                                                                    180, -M_PI, M_PI,
                                                                    180, -M_PI, M_PI );
    mapStubLayer_hStub_Phi_SimTrk_Phi[ stackIdx ]->Sumw2();

    /// Residuals
    histoName.str("");  histoName << "hStub_InvPtRes_SimTrk_Eta_L" << stackIdx;
    histoTitle.str(""); histoTitle << "Stub p_{T}^{-1} - SimTrk p_{T}^{-1} vs. SimTrk #eta, Barrel Stack " << stackIdx;
    mapStubLayer_hStub_InvPtRes_SimTrk_Eta[ stackIdx ] = fs->make<TH2D>( histoName.str().c_str(),  histoTitle.str().c_str(),
                                                                         180, -M_PI, M_PI,
                                                                         100, -2.0, 2.0 );
    mapStubLayer_hStub_InvPtRes_SimTrk_Eta[ stackIdx ]->Sumw2();

    histoName.str("");  histoName << "hStub_PtRes_SimTrk_Eta_L" << stackIdx;
    histoTitle.str(""); histoTitle << "Stub p_{T} - SimTrk p_{T} vs. SimTrk #eta, Barrel Stack " << stackIdx;
    mapStubLayer_hStub_PtRes_SimTrk_Eta[ stackIdx ] = fs->make<TH2D>( histoName.str().c_str(),  histoTitle.str().c_str(),
                                                                      180, -M_PI, M_PI,
                                                                      100, -40, 40 );
    mapStubLayer_hStub_PtRes_SimTrk_Eta[ stackIdx ]->Sumw2();

    histoName.str("");  histoName << "hStub_EtaRes_SimTrk_Eta_L" << stackIdx;
    histoTitle.str(""); histoTitle << "Stub #eta - SimTrk #eta vs. SimTrk #eta, Barrel Stack " << stackIdx;
    mapStubLayer_hStub_EtaRes_SimTrk_Eta[ stackIdx ] = fs->make<TH2D>( histoName.str().c_str(),  histoTitle.str().c_str(),
                                                                       180, -M_PI, M_PI,
                                                                       100, -2, 2 );
    mapStubLayer_hStub_EtaRes_SimTrk_Eta[ stackIdx ]->Sumw2();

    histoName.str("");  histoName << "hStub_PhiRes_SimTrk_Eta_L" << stackIdx;
    histoTitle.str(""); histoTitle << "Stub #phi - SimTrk #phi vs. SimTrk #eta, Barrel Stack " << stackIdx;
    mapStubLayer_hStub_PhiRes_SimTrk_Eta[ stackIdx ] = fs->make<TH2D>( histoName.str().c_str(),  histoTitle.str().c_str(),
                                                                       180, -M_PI, M_PI,
                                                                       100, -0.5, 0.5 );
    mapStubLayer_hStub_PhiRes_SimTrk_Eta[ stackIdx ]->Sumw2();

    /// Stub Width vs. Pt
    histoName.str("");  histoName << "hStub_W_SimTrk_Pt_L" << stackIdx;
    histoTitle.str(""); histoTitle << "Stub Width vs. SimTrk p_{T}, Barrel Stack " << stackIdx;
    mapStubLayer_hStub_W_SimTrk_Pt[ stackIdx ] = fs->make<TH2D>( histoName.str().c_str(),  histoTitle.str().c_str(),
                                                                 200, 0, 50,
                                                                 41, -10.25, 10.25 );
    mapStubLayer_hStub_W_SimTrk_Pt[ stackIdx ]->Sumw2();

    histoName.str("");  histoName << "hStub_W_SimTrk_InvPt_L" << stackIdx;
    histoTitle.str(""); histoTitle << "Stub Width vs. SimTrk p_{T}^{-1}, Barrel Stack " << stackIdx;
    mapStubLayer_hStub_W_SimTrk_InvPt[ stackIdx ] = fs->make<TH2D>( histoName.str().c_str(),  histoTitle.str().c_str(),
                                                                    200, 0, 0.8,
                                                                    41, -10.25, 10.25 );
    mapStubLayer_hStub_W_SimTrk_InvPt[ stackIdx ]->GetXaxis()->Set( NumBins, BinVec );
    mapStubLayer_hStub_W_SimTrk_InvPt[ stackIdx ]->Sumw2();

    /// ENDCAP

    /// Denominators
    histoName.str("");  histoName << "hSimTrk_Pt_Clu_D" << stackIdx;
    histoTitle.str(""); histoTitle << "SimTrk p_{T}, Cluster, Endcap Stack " << stackIdx;
    mapCluDisk_hSimTrk_Pt[ stackIdx ] = fs->make<TH1D>( histoName.str().c_str(),  histoTitle.str().c_str(),
                                                        100, 0, 50 );
    histoName.str("");  histoName << "hSimTrk_Eta_Pt10_Clu_D" << stackIdx;
    histoTitle.str(""); histoTitle << "SimTrk #eta (p_{T} > 10 GeV/c), Cluster, Endcap Stack " << stackIdx;
    mapCluDisk_hSimTrk_Eta_Pt10[ stackIdx ] = fs->make<TH1D>( histoName.str().c_str(),  histoTitle.str().c_str(),
                                                              180, -M_PI, M_PI );
    histoName.str("");  histoName << "hSimTrk_Phi_Pt10_Clu_D" << stackIdx;
    histoTitle.str(""); histoTitle << "SimTrk #phi (p_{T} > 10 GeV/c), Cluster, Endcap Stack " << stackIdx;
    mapCluDisk_hSimTrk_Phi_Pt10[ stackIdx ] = fs->make<TH1D>( histoName.str().c_str(),  histoTitle.str().c_str(),
                                                              180, -M_PI, M_PI );
    mapCluDisk_hSimTrk_Pt[ stackIdx ]->Sumw2();
    mapCluDisk_hSimTrk_Eta_Pt10[ stackIdx ]->Sumw2();
    mapCluDisk_hSimTrk_Phi_Pt10[ stackIdx ]->Sumw2();

    /// Numerators GeV/c
    histoName.str("");  histoName << "hSimTrk_Pt_Stub_D" << stackIdx;
    histoTitle.str(""); histoTitle << "SimTrk p_{T}, Stub, Endcap Stack " << stackIdx;
    mapStubDisk_hSimTrk_Pt[ stackIdx ] = fs->make<TH1D>( histoName.str().c_str(),  histoTitle.str().c_str(),
                                                         100, 0, 50 );
    histoName.str("");  histoName << "hSimTrk_Eta_Pt10_Stub_D" << stackIdx;
    histoTitle.str(""); histoTitle << "SimTrk #eta (p_{T} > 10 GeV/c), Stub, Endcap Stack " << stackIdx;
    mapStubDisk_hSimTrk_Eta_Pt10[ stackIdx ] = fs->make<TH1D>( histoName.str().c_str(),  histoTitle.str().c_str(),
                                                               180, -M_PI, M_PI );
    histoName.str("");  histoName << "hSimTrk_Phi_Pt10_Stub_D" << stackIdx;
    histoTitle.str(""); histoTitle << "SimTrk #phi (p_{T} > 10 GeV/c), Stub, Endcap Stack " << stackIdx;
    mapStubDisk_hSimTrk_Phi_Pt10[ stackIdx ] = fs->make<TH1D>( histoName.str().c_str(),  histoTitle.str().c_str(),
                                                               180, -M_PI, M_PI );
    mapStubDisk_hSimTrk_Pt[ stackIdx ]->Sumw2();
    mapStubDisk_hSimTrk_Eta_Pt10[ stackIdx ]->Sumw2();
    mapStubDisk_hSimTrk_Phi_Pt10[ stackIdx ]->Sumw2();

    /// Comparison to SimTrack
    histoName.str("");  histoName << "hStub_InvPt_SimTrk_InvPt_D" << stackIdx;
    histoTitle.str(""); histoTitle << "Stub p_{T}^{-1} vs. SimTrk p_{T}^{-1}, Endcap Stack " << stackIdx;
    mapStubDisk_hStub_InvPt_SimTrk_InvPt[ stackIdx ] = fs->make<TH2D>( histoName.str().c_str(),  histoTitle.str().c_str(),
                                                                       200, 0.0, 0.8,
                                                                       200, 0.0, 0.8 );
    mapStubDisk_hStub_InvPt_SimTrk_InvPt[ stackIdx ]->GetXaxis()->Set( NumBins, BinVec );
    mapStubDisk_hStub_InvPt_SimTrk_InvPt[ stackIdx ]->GetYaxis()->Set( NumBins, BinVec );
    mapStubDisk_hStub_InvPt_SimTrk_InvPt[ stackIdx ]->Sumw2();

    histoName.str("");  histoName << "hStub_Pt_SimTrk_Pt_D" << stackIdx;
    histoTitle.str(""); histoTitle << "Stub p_{T} vs. SimTrk p_{T}, Endcap Stack " << stackIdx;
    mapStubDisk_hStub_Pt_SimTrk_Pt[ stackIdx ] = fs->make<TH2D>( histoName.str().c_str(),  histoTitle.str().c_str(),
                                                                 100, 0, 50,
                                                                 100, 0, 50 );
    mapStubDisk_hStub_Pt_SimTrk_Pt[ stackIdx ]->Sumw2();

    histoName.str("");  histoName << "hStub_Eta_SimTrk_Eta_D" << stackIdx;
    histoTitle.str(""); histoTitle << "Stub #eta vs. SimTrk #eta, Endcap Stack " << stackIdx;
    mapStubDisk_hStub_Eta_SimTrk_Eta[ stackIdx ] = fs->make<TH2D>( histoName.str().c_str(),  histoTitle.str().c_str(),
                                                                   180, -M_PI, M_PI,
                                                                   180, -M_PI, M_PI );
    mapStubDisk_hStub_Eta_SimTrk_Eta[ stackIdx ]->Sumw2();

    histoName.str("");  histoName << "hStub_Phi_SimTrk_Phi_D" << stackIdx;
    histoTitle.str(""); histoTitle << "Stub #phi vs. SimTrk #phi, Endcap Stack " << stackIdx;
    mapStubDisk_hStub_Phi_SimTrk_Phi[ stackIdx ] = fs->make<TH2D>( histoName.str().c_str(),  histoTitle.str().c_str(),
                                                                   180, -M_PI, M_PI,
                                                                   180, -M_PI, M_PI );
    mapStubDisk_hStub_Phi_SimTrk_Phi[ stackIdx ]->Sumw2();

    /// Residuals
    histoName.str("");  histoName << "hStub_InvPtRes_SimTrk_Eta_D" << stackIdx;
    histoTitle.str(""); histoTitle << "Stub p_{T}^{-1} - SimTrk p_{T}^{-1} vs. SimTrk #eta, Endcap Stack " << stackIdx;
    mapStubDisk_hStub_InvPtRes_SimTrk_Eta[ stackIdx ] = fs->make<TH2D>( histoName.str().c_str(),  histoTitle.str().c_str(),
                                                                        180, -M_PI, M_PI,
                                                                        100, -2.0, 2.0 );
    mapStubDisk_hStub_InvPtRes_SimTrk_Eta[ stackIdx ]->Sumw2();

    histoName.str("");  histoName << "hStub_PtRes_SimTrk_Eta_D" << stackIdx;
    histoTitle.str(""); histoTitle << "Stub p_{T} - SimTrk p_{T} vs. SimTrk #eta, Endcap Stack " << stackIdx;
    mapStubDisk_hStub_PtRes_SimTrk_Eta[ stackIdx ] = fs->make<TH2D>( histoName.str().c_str(),  histoTitle.str().c_str(),
                                                                     180, -M_PI, M_PI,
                                                                     100, -40, 40 );
    mapStubDisk_hStub_PtRes_SimTrk_Eta[ stackIdx ]->Sumw2();

    histoName.str("");  histoName << "hStub_EtaRes_SimTrk_Eta_D" << stackIdx;
    histoTitle.str(""); histoTitle << "Stub #eta - SimTrk #eta vs. SimTrk #eta, Endcap Stack " << stackIdx;
    mapStubDisk_hStub_EtaRes_SimTrk_Eta[ stackIdx ] = fs->make<TH2D>( histoName.str().c_str(),  histoTitle.str().c_str(),
                                                                      180, -M_PI, M_PI,
                                                                      100, -2, 2 );
    mapStubDisk_hStub_EtaRes_SimTrk_Eta[ stackIdx ]->Sumw2();

    histoName.str("");  histoName << "hStub_PhiRes_SimTrk_Eta_D" << stackIdx;
    histoTitle.str(""); histoTitle << "Stub #phi - SimTrk #phi vs. SimTrk #eta, Endcap Stack " << stackIdx;
    mapStubDisk_hStub_PhiRes_SimTrk_Eta[ stackIdx ] = fs->make<TH2D>( histoName.str().c_str(),  histoTitle.str().c_str(),
                                                                      180, -M_PI, M_PI,
                                                                      100, -0.5, 0.5 );
    mapStubDisk_hStub_PhiRes_SimTrk_Eta[ stackIdx ]->Sumw2();

    /// Stub Width vs. Pt
    histoName.str("");  histoName << "hStub_W_SimTrk_Pt_D" << stackIdx;
    histoTitle.str(""); histoTitle << "Stub Width vs. SimTrk p_{T}, Endcap Stack " << stackIdx;
    mapStubDisk_hStub_W_SimTrk_Pt[ stackIdx ] = fs->make<TH2D>( histoName.str().c_str(),  histoTitle.str().c_str(),
                                                                200, 0, 50,
                                                                41, -10.25, 10.25 );
    mapStubDisk_hStub_W_SimTrk_Pt[ stackIdx ]->Sumw2();

    histoName.str("");  histoName << "hStub_W_SimTrk_InvPt_D" << stackIdx;
    histoTitle.str(""); histoTitle << "Stub Width vs. SimTrk p_{T}^{-1}, Endcap Stack " << stackIdx;
    mapStubDisk_hStub_W_SimTrk_InvPt[ stackIdx ] = fs->make<TH2D>( histoName.str().c_str(),  histoTitle.str().c_str(),
                                                                   200, 0, 0.8,
                                                                   41, -10.25, 10.25 );
    mapStubDisk_hStub_W_SimTrk_InvPt[ stackIdx ]->GetXaxis()->Set( NumBins, BinVec );
    mapStubDisk_hStub_W_SimTrk_InvPt[ stackIdx ]->Sumw2();
  }

  /// End of things to be done before entering the event Loop
}

//////////
// ANALYZE
void ValidateClusterStub::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  /// Geometry handles etc
  edm::ESHandle< TrackerGeometry >                GeometryHandle;
  edm::ESHandle< StackedTrackerGeometry >         StackedGeometryHandle;
  const StackedTrackerGeometry*                   theStackedGeometry;
  StackedTrackerGeometry::StackContainerIterator  StackedTrackerIterator;

  /// Geometry setup
  /// Set pointers to Geometry
  iSetup.get< TrackerDigiGeometryRecord >().get(GeometryHandle);
  /// Set pointers to Stacked Modules
  iSetup.get< StackedTrackerGeometryRecord >().get(StackedGeometryHandle);
  theStackedGeometry = StackedGeometryHandle.product(); /// Note this is different 
                                                        /// from the "global" geometry

  /// Magnetic Field
  edm::ESHandle< MagneticField > magneticFieldHandle;
  iSetup.get< IdealMagneticFieldRecord >().get(magneticFieldHandle);
  const MagneticField* theMagneticField = magneticFieldHandle.product();
  double mMagneticFieldStrength = theMagneticField->inTesla(GlobalPoint(0,0,0)).z();

  /// Sim Tracks and Vtx
  edm::Handle< edm::SimTrackContainer >  SimTrackHandle;
  edm::Handle< edm::SimVertexContainer > SimVtxHandle;
  //iEvent.getByLabel( "famosSimHits", SimTrackHandle );
  //iEvent.getByLabel( "famosSimHits", SimVtxHandle );
  iEvent.getByLabel( "g4SimHits", SimTrackHandle );
  iEvent.getByLabel( "g4SimHits", SimVtxHandle );

  /// Track Trigger
  edm::Handle< L1TkCluster_PixelDigi_Collection > PixelDigiL1TkClusterHandle;
  edm::Handle< L1TkStub_PixelDigi_Collection >    PixelDigiL1TkStubHandle;
  edm::Handle< L1TkStub_PixelDigi_Collection >    PixelDigiL1TkFailedStubHandle;
  iEvent.getByLabel( "L1TkClustersFromPixelDigis",             PixelDigiL1TkClusterHandle );
  iEvent.getByLabel( "L1TkStubsFromPixelDigis", "StubsPass",   PixelDigiL1TkStubHandle );
  iEvent.getByLabel( "L1TkStubsFromPixelDigis", "StubsFail",   PixelDigiL1TkFailedStubHandle );

  ///////////////////////////////////
  /// COLLECT CLUSTER INFORMATION ///
  ///////////////////////////////////

  /// Maps to store SimTrack information
  std::map< unsigned int, std::vector< edm::Ptr< SimTrack > > > simTrackPerLayer;
  std::map< unsigned int, std::vector< edm::Ptr< SimTrack > > > simTrackPerDisk;

  /// Go on only if there are L1TkCluster from PixelDigis
  if ( PixelDigiL1TkClusterHandle->size() > 0 )
  {
    /// Loop over L1TkClusters
    L1TkCluster_PixelDigi_Collection::const_iterator iterL1TkCluster;
    for ( iterL1TkCluster = PixelDigiL1TkClusterHandle->begin();
          iterL1TkCluster != PixelDigiL1TkClusterHandle->end();
          ++iterL1TkCluster )
    {
      StackedTrackerDetId detIdClu( iterL1TkCluster->getDetId() );
      unsigned int memberClu = iterL1TkCluster->getStackMember();
      bool genuineClu     = iterL1TkCluster->isGenuine();
      bool combinClu      = iterL1TkCluster->isCombinatoric();
      //bool unknownClu     = iterL1TkCluster->isUnknown();
      int partClu         = iterL1TkCluster->findType();
      unsigned int widClu = iterL1TkCluster->findWidth();
      GlobalPoint posClu  = theStackedGeometry->findAverageGlobalPosition( &(*iterL1TkCluster) );
      
      hCluster_RZ->Fill( posClu.z(), posClu.perp() );

      if ( detIdClu.isBarrel() )
      {
        if ( memberClu == 0 )
        {
          hCluster_IMem_Barrel->Fill( detIdClu.iLayer() );
        }
        else
        {
          hCluster_OMem_Barrel->Fill( detIdClu.iLayer() );
        }

        if ( genuineClu )
        {
          hCluster_Gen_Barrel->Fill( detIdClu.iLayer() );
        }
        else if ( combinClu )
        {
          hCluster_Comb_Barrel->Fill( detIdClu.iLayer() );
        }
        else
        {
          hCluster_Unkn_Barrel->Fill( detIdClu.iLayer() );
        }

        hCluster_Barrel_XY->Fill( posClu.x(), posClu.y() );
        hCluster_Barrel_XY_Zoom->Fill( posClu.x(), posClu.y() );
      }
      else if ( detIdClu.isEndcap() )
      {
        if ( memberClu == 0 )
        {
          hCluster_IMem_Endcap->Fill( detIdClu.iDisk() );
        }
        else
        {
          hCluster_OMem_Endcap->Fill( detIdClu.iDisk() );
        }

        if ( genuineClu )
        {
          hCluster_Gen_Endcap->Fill( detIdClu.iDisk() );
        }
        else if ( combinClu )
        {
          hCluster_Comb_Endcap->Fill( detIdClu.iDisk() );
        }
        else
        {
          hCluster_Unkn_Endcap->Fill( detIdClu.iDisk() );
        }

        if ( posClu.z() > 0 )
        {
          hCluster_Endcap_Fw_XY->Fill( posClu.x(), posClu.y() );
          hCluster_Endcap_Fw_RZ_Zoom->Fill( posClu.z(), posClu.perp() );
        }
        else
        {
          hCluster_Endcap_Bw_XY->Fill( posClu.x(), posClu.y() );
          hCluster_Endcap_Bw_RZ_Zoom->Fill( posClu.z(), posClu.perp() );
        }
      }

      hCluster_PID->Fill( partClu, memberClu );
      hCluster_W->Fill( widClu, memberClu );

      /// Store Track information in maps, skip if the Cluster is not good
      if ( !genuineClu && !combinClu ) continue;

      for ( unsigned int i = 0; i < iterL1TkCluster->getSimTrackPtrs().size(); i++ )
      {
        edm::Ptr< SimTrack > simTrackPtr = iterL1TkCluster->getSimTrackPtrs().at(i);

        if ( simTrackPtr.isNull() )
          continue;

        /// Get the corresponding vertex and reject the track
        /// if its vertex is outside the beampipe
        int vertexIndex = simTrackPtr->vertIndex();
        const SimVertex& theSimVertex = (*SimVtxHandle)[vertexIndex];
        math::XYZTLorentzVectorD trkVtxPos = theSimVertex.position();

        if ( trkVtxPos.rho() >= 2 )
          continue;
{}

      if ( detIdClu.isBarrel() )
      {
        if ( simTrackPerLayer.find( detIdClu.iLayer() ) == simTrackPerLayer.end() )
        {
          std::vector< edm::Ptr< SimTrack > > tempVec;
          simTrackPerLayer.insert( make_pair( detIdClu.iLayer(), tempVec ) );
        }
        simTrackPerLayer[detIdClu.iLayer()].push_back( simTrackPtr );
      }
      else if ( detIdClu.isEndcap() )
      {
        if ( simTrackPerDisk.find( detIdClu.iDisk() ) == simTrackPerDisk.end() )
        {
          std::vector< edm::Ptr< SimTrack > > tempVec;
          simTrackPerDisk.insert( make_pair( detIdClu.iDisk(), tempVec ) );
        }
        simTrackPerDisk[detIdClu.iDisk()].push_back( simTrackPtr );
      }
      }
    } /// End of Loop over L1TkClusters
  } /// End of if ( PixelDigiL1TkClusterHandle->size() > 0 )

  /// Clean the maps for SimTracks and fill histograms
  std::map< unsigned int, std::vector< edm::Ptr< SimTrack > > >::iterator iterSimTrackPerLayer;
  std::map< unsigned int, std::vector< edm::Ptr< SimTrack > > >::iterator iterSimTrackPerDisk;

  for ( iterSimTrackPerLayer = simTrackPerLayer.begin();
        iterSimTrackPerLayer != simTrackPerLayer.end();
        ++iterSimTrackPerLayer )
  {
    /// Remove duplicates, if any
    std::vector< edm::Ptr< SimTrack > > tempVec = iterSimTrackPerLayer->second;
    std::sort( tempVec.begin(), tempVec.end() );
    tempVec.erase( std::unique( tempVec.begin(), tempVec.end() ), tempVec.end() );

    /// Loop over the SimTracks in this piece of the map
    for ( unsigned int i = 0; i < tempVec.size(); i++ )
    {
      if ( tempVec.at(i).isNull() ) continue;
      SimTrack thisSimTrack = *(tempVec.at(i));
      mapCluLayer_hSimTrk_Pt[ iterSimTrackPerLayer->first ]->Fill( thisSimTrack.momentum().pt() );
      if ( thisSimTrack.momentum().pt() > 10.0 )
      {
        mapCluLayer_hSimTrk_Eta_Pt10[ iterSimTrackPerLayer->first ]->Fill( thisSimTrack.momentum().eta() );
        mapCluLayer_hSimTrk_Phi_Pt10[ iterSimTrackPerLayer->first ]->Fill( thisSimTrack.momentum().phi() > M_PI ?
                                                                           thisSimTrack.momentum().phi() - 2*M_PI :
                                                                           thisSimTrack.momentum().phi() );    
      }
    }
  }

  for ( iterSimTrackPerDisk = simTrackPerDisk.begin();
        iterSimTrackPerDisk != simTrackPerDisk.end();
        ++iterSimTrackPerDisk )
  {
    /// Remove duplicates, if any
    std::vector< edm::Ptr< SimTrack > > tempVec = iterSimTrackPerDisk->second;
    std::sort( tempVec.begin(), tempVec.end() );
    tempVec.erase( std::unique( tempVec.begin(), tempVec.end() ), tempVec.end() );

    /// Loop over the SimTracks in this piece of the map
    for ( unsigned int i = 0; i < tempVec.size(); i++ )
    {
      if ( tempVec.at(i).isNull() ) continue;
      SimTrack thisSimTrack = *(tempVec.at(i));
      mapCluDisk_hSimTrk_Pt[ iterSimTrackPerDisk->first ]->Fill( thisSimTrack.momentum().pt() );
      if ( thisSimTrack.momentum().pt() > 10.0 )
      {
        mapCluDisk_hSimTrk_Eta_Pt10[ iterSimTrackPerDisk->first ]->Fill( thisSimTrack.momentum().eta() );
        mapCluDisk_hSimTrk_Phi_Pt10[ iterSimTrackPerDisk->first ]->Fill( thisSimTrack.momentum().phi() > M_PI ?
                                                                         thisSimTrack.momentum().phi() - 2*M_PI :
                                                                         thisSimTrack.momentum().phi() );    
      }
    }
  }

  ////////////////////////////////
  /// COLLECT STUB INFORMATION ///
  ////////////////////////////////

  /// Maps to store SimTrack information
  std::map< unsigned int, std::vector< edm::Ptr< SimTrack > > > simTrackPerStubLayer;
  std::map< unsigned int, std::vector< edm::Ptr< SimTrack > > > simTrackPerStubDisk;

  /// Go on only if there are L1TkStub from PixelDigis
  if ( PixelDigiL1TkStubHandle->size() > 0 )
  {
    /// Loop over L1TkStubs
    L1TkStub_PixelDigi_Collection::const_iterator iterL1TkStub;
    for ( iterL1TkStub = PixelDigiL1TkStubHandle->begin();
          iterL1TkStub != PixelDigiL1TkStubHandle->end();
          ++iterL1TkStub )
    {
      StackedTrackerDetId detIdStub( iterL1TkStub->getDetId() );

      bool genuineStub    = iterL1TkStub->isGenuine();
      bool combinStub     = iterL1TkStub->isCombinatoric();
      //bool unknownStub    = iterL1TkStub->isUnknown();
      int partStub        = iterL1TkStub->findType();
      double displStub    = iterL1TkStub->getTriggerDisplacement();
      double offsetStub   = iterL1TkStub->getTriggerOffset();
      GlobalPoint posStub = theStackedGeometry->findGlobalPosition( &(*iterL1TkStub) );
      
      hStub_RZ->Fill( posStub.z(), posStub.perp() );

      if ( detIdStub.isBarrel() )
      {
        hStub_Barrel->Fill( detIdStub.iLayer() );

        if ( genuineStub )
        {
          hStub_Gen_Barrel->Fill( detIdStub.iLayer() );
        }
        else if ( combinStub )
        {
          hStub_Comb_Barrel->Fill( detIdStub.iLayer() );
        }
        else
        {
          hStub_Unkn_Barrel->Fill( detIdStub.iLayer() );
        }

        hStub_Barrel_XY->Fill( posStub.x(), posStub.y() );
        hStub_Barrel_XY_Zoom->Fill( posStub.x(), posStub.y() );
      }
      else if ( detIdStub.isEndcap() )
      {
        hStub_Endcap->Fill( detIdStub.iDisk() );

        if ( genuineStub )
        {
          hStub_Gen_Endcap->Fill( detIdStub.iDisk() );
        }
        else if ( combinStub )
        {
          hStub_Comb_Endcap->Fill( detIdStub.iDisk() );
        }
        else
        {
          hStub_Unkn_Endcap->Fill( detIdStub.iDisk() );
        }

        if ( posStub.z() > 0 ) 
        {
          hStub_Endcap_Fw_XY->Fill( posStub.x(), posStub.y() );
          hStub_Endcap_Fw_RZ_Zoom->Fill( posStub.z(), posStub.perp() );
        }
        else
        {
          hStub_Endcap_Bw_XY->Fill( posStub.x(), posStub.y() );
          hStub_Endcap_Bw_RZ_Zoom->Fill( posStub.z(), posStub.perp() );
        }
      }

      hStub_PID->Fill( partStub );
/*
if ( iterL1TkStub->isCombinatoric() )
{
std::cerr<< "CLUSTERS ARE"<<std::endl;
std::cerr<< iterL1TkStub->getClusterPtr(0)->isUnknown() << " " << iterL1TkStub->getClusterPtr(1)->isUnknown() << std::endl;
std::cerr<< iterL1TkStub->getClusterPtr(0)->isCombinatoric() << " " << iterL1TkStub->getClusterPtr(1)->isCombinatoric() << std::endl;
std::cerr<< iterL1TkStub->getClusterPtr(0)->isGenuine() << " " << iterL1TkStub->getClusterPtr(1)->isGenuine() << std::endl;

for (unsigned int iclu = 0; iclu<2; iclu++)
{
for (unsigned int ist = 0; ist < iterL1TkStub->getClusterPtr(iclu)->getSimTrackPtrs().size(); ist++ )
{
  int a, b;
  if (iterL1TkStub->getClusterPtr(iclu)->getSimTrackPtrs().at(ist).isNull())
  { a=9999; b=0 ;}
  else
  { a = iterL1TkStub->getClusterPtr(iclu)->getSimTrackPtrs().at(ist)->type();
    b = iterL1TkStub->getClusterPtr(iclu)->getSimTrackPtrs().at(ist)->trackId();
  }
  std::cerr << iclu << ">> " << a;
  std::cerr << " (";
  std::cerr << iclu << ">> " << b;
  std::cerr << ") ";
}
std::cerr << std::endl;
}

}
*/
      /// Store Track information in maps, skip if the Cluster is not good
      if ( !genuineStub ) continue;
/*
{

std::cerr<< "CLUSTERS ARE"<<std::endl;
std::cerr<< iterL1TkStub->getClusterPtr(0)->isUnknown() << " " << iterL1TkStub->getClusterPtr(1)->isUnknown() << std::endl;
std::cerr<< iterL1TkStub->getClusterPtr(0)->isCombinatoric() << " " << iterL1TkStub->getClusterPtr(1)->isCombinatoric() << std::endl;
std::cerr<< iterL1TkStub->getClusterPtr(0)->isGenuine() << " " << iterL1TkStub->getClusterPtr(1)->isGenuine() << std::endl;



        int prevTrack = -99999; // SimTrackId storage
        unsigned int whichSimTrack = 0;
        std::vector< edm::Ptr< SimTrack > > innerSimTracks = iterL1TkStub->getClusterPtr(0)->getSimTrackPtrs();
        std::vector< edm::Ptr< SimTrack > > outerSimTracks = iterL1TkStub->getClusterPtr(1)->getSimTrackPtrs();
        std::vector< uint32_t >             innerEventIds = iterL1TkStub->getClusterPtr(0)->getEventIds();
        std::vector< uint32_t >             outerEventIds = iterL1TkStub->getClusterPtr(1)->getEventIds();
std::cerr<<"ready?"<<std::endl;

        for ( unsigned int i = 0; i < iterL1TkStub->getClusterPtr(0)->getSimTrackPtrs().size(); i++ )
        {
std::cerr<<"- "<< iterL1TkStub->getClusterPtr(0)->getSimTrackPtrs().at(i).isNull() << std::endl;

          /// Skip NULL pointers
          if ( iterL1TkStub->getClusterPtr(0)->getSimTrackPtrs().at(i).isNull() )
            continue;
std::cerr<<"a"<<std::endl;
          for ( unsigned int j = 0; j < iterL1TkStub->getClusterPtr(1)->getSimTrackPtrs().size(); j++ )
          {
std::cerr<<"a1 "<< iterL1TkStub->getClusterPtr(1)->getSimTrackPtrs().at(j).isNull() <<std::endl;

            /// Skip NULL pointers
            if ( iterL1TkStub->getClusterPtr(1)->getSimTrackPtrs().at(j).isNull() )
              continue;
std::cerr<<"b"<<std::endl;

            /// Skip pairs from different EventId
            if ( innerEventIds.at(i) != outerEventIds.at(j) )
              continue;
std::cerr<<"c"<<std::endl;

            if ( iterL1TkStub->getClusterPtr(0)->getSimTrackPtrs().at(i)->trackId() == iterL1TkStub->getClusterPtr(1)->getSimTrackPtrs().at(j)->trackId() )
            {
std::cerr<<"d"<<std::endl;
              /// Same SimTrack is present in both clusters
              if ( prevTrack < 0 )
              {
std::cerr<<"e"<<std::endl;
                prevTrack = iterL1TkStub->getClusterPtr(0)->getSimTrackPtrs().at(j)->trackId();
                whichSimTrack = j;
              }
std::cerr<<"f"<<std::endl;

            }
          }
        }

std::cerr<< " ---------- " << prevTrack << " " << whichSimTrack << std::endl;


for (unsigned int iclu = 0; iclu<2; iclu++)
{

for (unsigned int ist = 0; ist < iterL1TkStub->getClusterPtr(iclu)->getSimTrackPtrs().size(); ist++ )
{

  int a, b;
  if (iterL1TkStub->getClusterPtr(iclu)->getSimTrackPtrs().at(ist).isNull())
  { a=9999; b=0 ;}
  else
  { a = iterL1TkStub->getClusterPtr(iclu)->getSimTrackPtrs().at(ist)->type();
    b = iterL1TkStub->getClusterPtr(iclu)->getSimTrackPtrs().at(ist)->trackId();
  }
  std::cerr << iclu << ">> " << a;
  std::cerr << " (";
  std::cerr << iclu << ">> " << b;
  std::cerr << ") ";
}
std::cerr << std::endl;
}

continue;
}
*/

      edm::Ptr< SimTrack > simTrackPtr = iterL1TkStub->getSimTrackPtr();

      /// Get the corresponding vertex and reject the track
      /// if its vertex is outside the beampipe
      int vertexIndex = simTrackPtr->vertIndex();
      const SimVertex& theSimVertex = (*SimVtxHandle)[vertexIndex];
      math::XYZTLorentzVectorD trkVtxPos = theSimVertex.position();

      if ( trkVtxPos.rho() >= 2 )
        continue;
{}

      if ( detIdStub.isBarrel() )
      {
        if ( simTrackPerStubLayer.find( detIdStub.iLayer() ) == simTrackPerStubLayer.end() )
        {
          std::vector< edm::Ptr< SimTrack > > tempVec;
          simTrackPerStubLayer.insert( make_pair( detIdStub.iLayer(), tempVec ) );
        }
        simTrackPerStubLayer[detIdStub.iLayer()].push_back( simTrackPtr );

        hStub_Barrel_W->Fill( detIdStub.iLayer(), displStub - offsetStub );
        hStub_Barrel_O->Fill( detIdStub.iLayer(), offsetStub );
      }
      else if ( detIdStub.isEndcap() )
      {
        if ( simTrackPerStubDisk.find( detIdStub.iDisk() ) == simTrackPerStubDisk.end() )
        {
          std::vector< edm::Ptr< SimTrack > > tempVec;
          simTrackPerStubDisk.insert( make_pair( detIdStub.iDisk(), tempVec ) );
        }
        simTrackPerStubDisk[detIdStub.iDisk()].push_back( simTrackPtr );

        hStub_Endcap_W->Fill( detIdStub.iDisk(), displStub - offsetStub );
        hStub_Endcap_O->Fill( detIdStub.iDisk(), offsetStub );
      }
      
      /// Compare to SimTrack

      if ( simTrackPtr.isNull() ) continue; /// This prevents to fill the vector if the SimTrack is not found
      SimTrack thisSimTrack = *simTrackPtr;

      double simPt = thisSimTrack.momentum().pt();
      double simEta = thisSimTrack.momentum().eta();
      double simPhi = thisSimTrack.momentum().phi();
      double recPt = theStackedGeometry->findRoughPt( mMagneticFieldStrength, &(*iterL1TkStub) );
      double recEta = theStackedGeometry->findGlobalDirection( &(*iterL1TkStub) ).eta();
      double recPhi = theStackedGeometry->findGlobalDirection( &(*iterL1TkStub) ).phi();

      if ( simPhi > M_PI )
      {
        simPhi -= 2*M_PI;
      }
      if ( recPhi > M_PI )
      {
        recPhi -= 2*M_PI;
      }

      if ( detIdStub.isBarrel() )
      {
        mapStubLayer_hStub_InvPt_SimTrk_InvPt[ detIdStub.iLayer() ]->Fill( 1./simPt, 1./recPt );
        mapStubLayer_hStub_Pt_SimTrk_Pt[ detIdStub.iLayer() ]->Fill( simPt, recPt );
        mapStubLayer_hStub_Eta_SimTrk_Eta[ detIdStub.iLayer() ]->Fill( simEta, recEta );
        mapStubLayer_hStub_Phi_SimTrk_Phi[ detIdStub.iLayer() ]->Fill( simPhi, recPhi );

        mapStubLayer_hStub_InvPtRes_SimTrk_Eta[ detIdStub.iLayer() ]->Fill( simEta, 1./recPt - 1./simPt );
        mapStubLayer_hStub_PtRes_SimTrk_Eta[ detIdStub.iLayer() ]->Fill( simEta, recPt - simPt );
        mapStubLayer_hStub_EtaRes_SimTrk_Eta[ detIdStub.iLayer() ]->Fill( simEta, recEta - simEta );
        mapStubLayer_hStub_PhiRes_SimTrk_Eta[ detIdStub.iLayer() ]->Fill( simEta, recPhi - simPhi );

        mapStubLayer_hStub_W_SimTrk_Pt[ detIdStub.iLayer() ]->Fill( simPt, displStub - offsetStub );
        mapStubLayer_hStub_W_SimTrk_InvPt[ detIdStub.iLayer() ]->Fill( 1./simPt, displStub - offsetStub );
      }
      else if ( detIdStub.isEndcap() )
      {
        mapStubDisk_hStub_InvPt_SimTrk_InvPt[ detIdStub.iDisk() ]->Fill( 1./simPt, 1./recPt );
        mapStubDisk_hStub_Pt_SimTrk_Pt[ detIdStub.iDisk() ]->Fill( simPt, recPt );
        mapStubDisk_hStub_Eta_SimTrk_Eta[ detIdStub.iDisk() ]->Fill( simEta, recEta );
        mapStubDisk_hStub_Phi_SimTrk_Phi[ detIdStub.iDisk() ]->Fill( simPhi, recPhi );

        mapStubDisk_hStub_InvPtRes_SimTrk_Eta[ detIdStub.iDisk() ]->Fill( simEta, 1./recPt - 1./simPt );
        mapStubDisk_hStub_PtRes_SimTrk_Eta[ detIdStub.iDisk() ]->Fill( simEta, recPt - simPt );
        mapStubDisk_hStub_EtaRes_SimTrk_Eta[ detIdStub.iDisk() ]->Fill( simEta, recEta - simEta );
        mapStubDisk_hStub_PhiRes_SimTrk_Eta[ detIdStub.iDisk() ]->Fill( simEta, recPhi - simPhi );

        mapStubDisk_hStub_W_SimTrk_Pt[ detIdStub.iDisk() ]->Fill( simPt, displStub - offsetStub );
        mapStubDisk_hStub_W_SimTrk_InvPt[ detIdStub.iDisk() ]->Fill( 1./simPt, displStub - offsetStub );
      }
    } /// End of loop over L1TkStubs
  } /// End of if ( PixelDigiL1TkStubHandle->size() > 0 )

  /// Clean the maps for SimTracks and fill histograms
  std::map< unsigned int, std::vector< edm::Ptr< SimTrack > > >::iterator iterSimTrackPerStubLayer;
  std::map< unsigned int, std::vector< edm::Ptr< SimTrack > > >::iterator iterSimTrackPerStubDisk;

  for ( iterSimTrackPerStubLayer = simTrackPerStubLayer.begin();
        iterSimTrackPerStubLayer != simTrackPerStubLayer.end();
        ++iterSimTrackPerStubLayer ) 
  {
    /// Remove duplicates, if any
    std::vector< edm::Ptr< SimTrack > > tempVec = iterSimTrackPerStubLayer->second;
    std::sort( tempVec.begin(), tempVec.end() );
    tempVec.erase( std::unique( tempVec.begin(), tempVec.end() ), tempVec.end() );

    /// Loop over the SimTracks in this piece of the map
    for ( unsigned int i = 0; i < tempVec.size(); i++ )
    {
      if ( tempVec.at(i).isNull() ) continue;
      SimTrack thisSimTrack = *(tempVec.at(i));
      mapStubLayer_hSimTrk_Pt[ iterSimTrackPerStubLayer->first ]->Fill( thisSimTrack.momentum().pt() );
      if ( thisSimTrack.momentum().pt() > 10.0 )
      {
        mapStubLayer_hSimTrk_Eta_Pt10[ iterSimTrackPerStubLayer->first ]->Fill( thisSimTrack.momentum().eta() );
        mapStubLayer_hSimTrk_Phi_Pt10[ iterSimTrackPerStubLayer->first ]->Fill( thisSimTrack.momentum().phi() > M_PI ?
                                                                                thisSimTrack.momentum().phi() - 2*M_PI :
                                                                                thisSimTrack.momentum().phi() );    
      }
    }
  }

  for ( iterSimTrackPerStubDisk = simTrackPerStubDisk.begin();
        iterSimTrackPerStubDisk != simTrackPerStubDisk.end();
        ++iterSimTrackPerStubDisk ) 
  {
    /// Remove duplicates, if any
    std::vector< edm::Ptr< SimTrack > > tempVec = iterSimTrackPerStubDisk->second;
    std::sort( tempVec.begin(), tempVec.end() );
    tempVec.erase( std::unique( tempVec.begin(), tempVec.end() ), tempVec.end() );

    /// Loop over the SimTracks in this piece of the map
    for ( unsigned int i = 0; i < tempVec.size(); i++ )
    {
      if ( tempVec.at(i).isNull() ) continue;
      SimTrack thisSimTrack = *(tempVec.at(i));
      mapStubDisk_hSimTrk_Pt[ iterSimTrackPerStubDisk->first ]->Fill( thisSimTrack.momentum().pt() );
      if ( thisSimTrack.momentum().pt() > 10.0 )
      {
        mapStubDisk_hSimTrk_Eta_Pt10[ iterSimTrackPerStubDisk->first ]->Fill( thisSimTrack.momentum().eta() );
        mapStubDisk_hSimTrk_Phi_Pt10[ iterSimTrackPerStubDisk->first ]->Fill( thisSimTrack.momentum().phi() > M_PI ?
                                                                              thisSimTrack.momentum().phi() - 2*M_PI :
                                                                              thisSimTrack.momentum().phi() );    
      }
    }
  }

  /// //////////////////////////
  /// SPECTRUM OF SIM TRACKS ///
  /// WITHIN PRIMARY VERTEX  ///
  /// CONSTRAINTS            ///
  /// //////////////////////////

  /// Go on only if there are SimTracks
  if ( SimTrackHandle->size() != 0 )
  {
    /// Loop over SimTracks
    edm::SimTrackContainer::const_iterator iterSimTracks;
    for ( iterSimTracks = SimTrackHandle->begin();
          iterSimTracks != SimTrackHandle->end();
          ++iterSimTracks )
    {
      /// Get the corresponding vertex
      int vertexIndex = iterSimTracks->vertIndex();
      const SimVertex& theSimVertex = (*SimVtxHandle)[vertexIndex];

      /// Assume perfectly round beamspot
      /// Correct and get the correct SimTrack Vertex position wrt beam center
      math::XYZTLorentzVectorD trkVtxPos = theSimVertex.position();

      if ( trkVtxPos.rho() >= 2 )
        continue;
{}

      /// First of all, check beamspot and correction
      hSimVtx_XY->Fill( trkVtxPos.x(), trkVtxPos.y() );
      hSimVtx_RZ->Fill( trkVtxPos.z(), trkVtxPos.rho() );

      /// Here we have only tracks form primary vertices
      /// Check Pt spectrum and pseudorapidity for over-threshold tracks
      hSimTrk_Pt->Fill( iterSimTracks->momentum().pt() );
      if ( iterSimTracks->momentum().pt() > 10.0 )
      {
        hSimTrk_Eta_Pt10->Fill( iterSimTracks->momentum().eta() );
        hSimTrk_Phi_Pt10->Fill( iterSimTracks->momentum().phi() > M_PI ?
                                iterSimTracks->momentum().phi() - 2*M_PI :
                                iterSimTracks->momentum().phi() );
      }
    } /// End of Loop over SimTracks
  } /// End of if ( SimTrackHandle->size() != 0 )

/*
// TEST
edm::Handle< std::vector<TrackingParticle> > TrackingParticleHandle;
iEvent.getByLabel( "mergedtruth", "MergedTrackTruth", TrackingParticleHandle );

  if ( TrackingParticleHandle->size() != 0 )
  {
    /// Loop over SimTracks
    std::vector<TrackingParticle>::const_iterator iterTPart;
    for ( iterTPart = TrackingParticleHandle->begin();
          iterTPart != TrackingParticleHandle->end();
          ++iterTPart )
    {
//      std::cerr<<"PDG   "<<iterTPart->pdgId()<<std::endl;
      std::vector< SimTrack > theseTracks = iterTPart->g4Tracks();
      for ( unsigned int j = 0; j < theseTracks.size(); j++ )
      {
        std::cerr<<theseTracks.at(j).trackId()<<"\t";
      }
    }
std::cerr<<std::endl;
  }
*/

} /// End of analyze()

///////////////////////////
// DEFINE THIS AS A PLUG-IN
DEFINE_FWK_MODULE(ValidateClusterStub);

