/////////////////////////////
// Track Trigger Checklist //
// L1TkTrack               //
//                         //
// Nicola Pozzobon - 2012  //
// Anders Ryd              //
/////////////////////////////

#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "DataFormats/GeometryCommonDetAlgo/interface/MeasurementPoint.h"
#include "Geometry/CommonTopologies/interface/Topology.h"
#include "Geometry/TrackerGeometryBuilder/interface/PixelGeomDetUnit.h"
#include "Geometry/CommonTopologies/interface/PixelTopology.h"

#include "Geometry/Records/interface/StackedTrackerGeometryRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/StackedTrackerGeometry.h"

#include "SimDataFormats/SLHC/interface/StackedTrackerTypes.h"
#include "SimDataFormats/Vertex/interface/SimVertexContainer.h"
//#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticle.h" /* TEST PURPOSE!!! */
#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include <TH1D.h>
#include <TH2D.h>

//////////////////////////////
//                          //
//     CLASS DEFINITION     //
//                          //
//////////////////////////////

class ValidateL1Track : public edm::EDAnalyzer
{
  /// Public methods
  public:
    /// Constructor/destructor
    explicit ValidateL1Track(const edm::ParameterSet& iConfig);
    virtual ~ValidateL1Track();
    // Typical methods used on Loops over events
    virtual void beginJob();
    virtual void endJob();
    virtual void analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup);

  /// Private methods and variables
  private:
    TH2D* hTrack_NStubs_Sector;
    TH2D* hTrack_NStubs_Wedge;
    TH2D* hTrack_Sector_Phi;
    TH2D* hTrack_Wedge_Eta;
    TH2D* hTrack_RInv_Seed_RInv;
    TH2D* hTrack_RInvRes_Track_Eta;
    TH2D* hTrack_InvPt_Seed_InvPt;
    TH2D* hTrack_InvPtRes_Track_Eta;
    TH2D* hTrack_Pt_Seed_Pt;
    TH2D* hTrack_PtRes_Track_Eta;
    TH2D* hTrack_Phi_Seed_Phi;
    TH2D* hTrack_PhiRes_Track_Eta;
    TH2D* hTrack_Eta_Seed_Eta;
    TH2D* hTrack_EtaRes_Track_Eta;
    TH2D* hTrack_VtxZ0_Seed_VtxZ0;
    TH2D* hTrack_VtxZ0Res_Track_Eta;

    std::map< std::pair< unsigned int, unsigned int >, TH2D* > mapTrackPropBB_deltaRhoPhi_Eta;
    std::map< std::pair< unsigned int, unsigned int >, TH1D* > mapTrackPropBB_deltaRhoPhi;
    std::map< std::pair< unsigned int, unsigned int >, TH2D* > mapTrackPropBB_deltaZ_Eta;
    std::map< std::pair< unsigned int, unsigned int >, TH1D* > mapTrackPropBB_deltaZ;

    std::map< std::pair< unsigned int, unsigned int >, TH2D* > mapTrackPropBE_deltaRhoPhi_Eta;
    std::map< std::pair< unsigned int, unsigned int >, TH1D* > mapTrackPropBE_deltaRhoPhi;
    std::map< std::pair< unsigned int, unsigned int >, TH2D* > mapTrackPropBE_deltaR_Eta;
    std::map< std::pair< unsigned int, unsigned int >, TH1D* > mapTrackPropBE_deltaR;

    std::map< std::pair< unsigned int, unsigned int >, TH2D* > mapTrackPropEB_deltaRhoPhi_Eta;
    std::map< std::pair< unsigned int, unsigned int >, TH1D* > mapTrackPropEB_deltaRhoPhi;
    std::map< std::pair< unsigned int, unsigned int >, TH2D* > mapTrackPropEB_deltaZ_Eta;
    std::map< std::pair< unsigned int, unsigned int >, TH1D* > mapTrackPropEB_deltaZ;

    std::map< std::pair< unsigned int, unsigned int >, TH2D* > mapTrackPropEE_deltaRhoPhi_Eta;
    std::map< std::pair< unsigned int, unsigned int >, TH1D* > mapTrackPropEE_deltaRhoPhi;
    std::map< std::pair< unsigned int, unsigned int >, TH2D* > mapTrackPropEE_deltaR_Eta;
    std::map< std::pair< unsigned int, unsigned int >, TH1D* > mapTrackPropEE_deltaR;

    TH1D* hTrack_3Stubs_Pt;
    TH1D* hTrack_3Stubs_Phi;
    TH1D* hTrack_3Stubs_Eta;

    TH1D* hTrack_2Stubs_Pt;
    TH1D* hTrack_2Stubs_Phi;
    TH1D* hTrack_2Stubs_Eta;

    TH1D* hTrack_Seed_Pt;
    TH1D* hTrack_Seed_Phi;
    TH1D* hTrack_Seed_Eta;

    TH1D* hSeed_Pt;
    TH1D* hSeed_Phi;
    TH1D* hSeed_Eta;

    TH1D* hSimTrack_Track_3Stubs_Pt;
    TH1D* hSimTrack_Track_3Stubs_Phi_Pt5;
    TH1D* hSimTrack_Track_3Stubs_Eta_Pt5;

    TH1D* hSimTrack_Track_2Stubs_Pt;
    TH1D* hSimTrack_Track_2Stubs_Phi_Pt5;
    TH1D* hSimTrack_Track_2Stubs_Eta_Pt5;

    TH1D* hSimTrack_Seed_Pt;
    TH1D* hSimTrack_Seed_Phi_Pt5;
    TH1D* hSimTrack_Seed_Eta_Pt5;

    TH1D* hSimTrack_Cluster_Pt;
    TH1D* hSimTrack_Cluster_Phi_Pt5;
    TH1D* hSimTrack_Cluster_Eta_Pt5;

    TH1D* hSimTrack_Stub_Pt;
    TH1D* hSimTrack_Stub_Phi_Pt5;
    TH1D* hSimTrack_Stub_Eta_Pt5;

    TH1D* hTrack_3Stubs_N;
    TH2D* hTrack_3Stubs_Pt_SimTrack_Pt;
    TH2D* hTrack_3Stubs_PtRes_SimTrack_Eta;
    TH2D* hTrack_3Stubs_InvPt_SimTrack_InvPt;
    TH2D* hTrack_3Stubs_InvPtRes_SimTrack_Eta;
    TH2D* hTrack_3Stubs_Phi_SimTrack_Phi;
    TH2D* hTrack_3Stubs_PhiRes_SimTrack_Eta;
    TH2D* hTrack_3Stubs_Eta_SimTrack_Eta;
    TH2D* hTrack_3Stubs_EtaRes_SimTrack_Eta;
    TH2D* hTrack_3Stubs_VtxZ0_SimTrack_VtxZ0;
    TH2D* hTrack_3Stubs_VtxZ0Res_SimTrack_Eta;
    TH2D* hTrack_3Stubs_Chi2_NStubs;
    TH2D* hTrack_3Stubs_Chi2_SimTrack_Eta;
    TH2D* hTrack_3Stubs_Chi2Red_NStubs;
    TH2D* hTrack_3Stubs_Chi2Red_SimTrack_Eta;

    TH1D* hTrack_2Stubs_N;
    TH2D* hTrack_2Stubs_Pt_SimTrack_Pt;
    TH2D* hTrack_2Stubs_PtRes_SimTrack_Eta;
    TH2D* hTrack_2Stubs_InvPt_SimTrack_InvPt;
    TH2D* hTrack_2Stubs_InvPtRes_SimTrack_Eta;
    TH2D* hTrack_2Stubs_Phi_SimTrack_Phi;
    TH2D* hTrack_2Stubs_PhiRes_SimTrack_Eta;
    TH2D* hTrack_2Stubs_Eta_SimTrack_Eta;
    TH2D* hTrack_2Stubs_EtaRes_SimTrack_Eta;
    TH2D* hTrack_2Stubs_VtxZ0_SimTrack_VtxZ0;
    TH2D* hTrack_2Stubs_VtxZ0Res_SimTrack_Eta;
    TH2D* hTrack_2Stubs_Chi2_NStubs;
    TH2D* hTrack_2Stubs_Chi2_SimTrack_Eta;
    TH2D* hTrack_2Stubs_Chi2Red_NStubs;
    TH2D* hTrack_2Stubs_Chi2Red_SimTrack_Eta;

    TH1D* hSeed_N;
    TH2D* hSeed_Pt_SimTrack_Pt;
    TH2D* hSeed_PtRes_SimTrack_Eta;
    TH2D* hSeed_InvPt_SimTrack_InvPt;
    TH2D* hSeed_InvPtRes_SimTrack_Eta;
    TH2D* hSeed_Phi_SimTrack_Phi;
    TH2D* hSeed_PhiRes_SimTrack_Eta;
    TH2D* hSeed_Eta_SimTrack_Eta;
    TH2D* hSeed_EtaRes_SimTrack_Eta;
    TH2D* hSeed_VtxZ0_SimTrack_VtxZ0;
    TH2D* hSeed_VtxZ0Res_SimTrack_Eta;
    TH2D* hSeed_Chi2_NStubs;
    TH2D* hSeed_Chi2_SimTrack_Eta;
    TH2D* hSeed_Chi2Red_NStubs;
    TH2D* hSeed_Chi2Red_SimTrack_Eta;

};

//////////////////////////////////
//                              //
//     CLASS IMPLEMENTATION     //
//                              //
//////////////////////////////////

//////////////
// CONSTRUCTOR
ValidateL1Track::ValidateL1Track(edm::ParameterSet const& iConfig) 
{
  /// Insert here what you need to initialize
}

/////////////
// DESTRUCTOR
ValidateL1Track::~ValidateL1Track()
{
  /// Insert here what you need to delete
  /// when you close the class instance
}  

//////////
// END JOB
void ValidateL1Track::endJob()
{
  /// Things to be done at the exit of the event Loop
  std::cerr << " ValidateL1Track::endJob" << std::endl;
  /// End of things to be done at the exit from the event Loop


  std::cerr<<"DeltaRhoPhi BB"<<std::endl;
  for ( unsigned int seed = 1; seed < 11; seed++ )
  {
    for ( unsigned int targ = 1; targ < 11; targ++ )
    {
      std::pair< unsigned int, unsigned int > mapKey0 = std::make_pair( seed, targ );
      mapTrackPropBB_deltaRhoPhi[ mapKey0 ]->GetXaxis()->SetRangeUser(-0.6, 0.6);
      std::cerr<< mapTrackPropBB_deltaRhoPhi[ mapKey0 ]->GetRMS() * 3.0 <<", ";
    }
    std::cerr<<std::endl;
  }

  std::cerr<<"DeltaRhoPhi BE"<<std::endl;
  for ( unsigned int seed = 1; seed < 11; seed++ )
  {
    for ( unsigned int targ = 1; targ < 11; targ++ )
    {
      std::pair< unsigned int, unsigned int > mapKey0 = std::make_pair( seed, targ );
      mapTrackPropBE_deltaRhoPhi[ mapKey0 ]->GetXaxis()->SetRangeUser(-0.6, 0.6);
      std::cerr<< mapTrackPropBE_deltaRhoPhi[ mapKey0 ]->GetRMS() * 3.0 <<", ";
    }
    std::cerr<<std::endl;
  }

  std::cerr<<"DeltaRhoPhi EB"<<std::endl;
  for ( unsigned int seed = 1; seed < 11; seed++ )
  {
    for ( unsigned int targ = 1; targ < 11; targ++ )
    {
      std::pair< unsigned int, unsigned int > mapKey0 = std::make_pair( seed, targ );
      mapTrackPropEB_deltaRhoPhi[ mapKey0 ]->GetXaxis()->SetRangeUser(-0.6, 0.6);
      std::cerr<< mapTrackPropEB_deltaRhoPhi[ mapKey0 ]->GetRMS() * 3.0 <<", ";
    }
    std::cerr<<std::endl;
  }

  std::cerr<<"DeltaRhoPhi EE"<<std::endl;
  for ( unsigned int seed = 1; seed < 11; seed++ )
  {
    for ( unsigned int targ = 1; targ < 11; targ++ )
    {
      std::pair< unsigned int, unsigned int > mapKey0 = std::make_pair( seed, targ );
      mapTrackPropEE_deltaRhoPhi[ mapKey0 ]->GetXaxis()->SetRangeUser(-0.6, 0.6);
      std::cerr<< mapTrackPropEE_deltaRhoPhi[ mapKey0 ]->GetRMS() * 3.0 <<", ";
    }
    std::cerr<<std::endl;
  }

  std::cerr<<"DeltaZ BB"<<std::endl;
  for ( unsigned int seed = 1; seed < 11; seed++ )
  {
    for ( unsigned int targ = 1; targ < 11; targ++ )
    {
      std::pair< unsigned int, unsigned int > mapKey0 = std::make_pair( seed, targ );
      std::cerr<< mapTrackPropBB_deltaZ[ mapKey0 ]->GetRMS() * 3.0 <<", ";
    }
    std::cerr<<std::endl;
  }

  std::cerr<<"DeltaZ BE"<<std::endl;
  for ( unsigned int seed = 1; seed < 11; seed++ )
  {
    for ( unsigned int targ = 1; targ < 11; targ++ )
    {
      std::pair< unsigned int, unsigned int > mapKey0 = std::make_pair( seed, targ );
      std::cerr<< mapTrackPropBE_deltaR[ mapKey0 ]->GetRMS() * 3.0 <<", ";
    }
    std::cerr<<std::endl;
  }

  std::cerr<<"DeltaZ EB"<<std::endl;
  for ( unsigned int seed = 1; seed < 11; seed++ )
  {
    for ( unsigned int targ = 1; targ < 11; targ++ )
    {
      std::pair< unsigned int, unsigned int > mapKey0 = std::make_pair( seed, targ );
      std::cerr<< mapTrackPropEB_deltaZ[ mapKey0 ]->GetRMS() * 3.0 <<", ";
    }
    std::cerr<<std::endl;
  }

  std::cerr<<"DeltaZ EE"<<std::endl;
  for ( unsigned int seed = 1; seed < 11; seed++ )
  {
    for ( unsigned int targ = 1; targ < 11; targ++ )
    {
      std::pair< unsigned int, unsigned int > mapKey0 = std::make_pair( seed, targ );
      std::cerr<< mapTrackPropEE_deltaR[ mapKey0 ]->GetRMS() * 3.0 <<", ";
    }
    std::cerr<<std::endl;
  }


}

////////////
// BEGIN JOB
void ValidateL1Track::beginJob()
{
  /// Initialize all slave variables
  /// mainly histogram ranges and resolution
  std::ostringstream histoName;
  std::ostringstream histoTitle;

  /// Things to be done before entering the event Loop
  std::cerr << " ValidateL1Track::beginJob" << std::endl;

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

  hTrack_NStubs_Sector      = fs->make<TH2D>( "hTrack_NStubs_Sector",      "L1TkTrack number of Stubs vs. Seed sector", 35, -0.5, 34.5, 20, -0.5, 19.5 );
  hTrack_NStubs_Wedge       = fs->make<TH2D>( "hTrack_NStubs_Wedge",       "L1TkTrack number of Stubs vs. Seed wedge",  35, -0.5, 34.5, 20, -0.5, 19.5 );
  hTrack_Sector_Phi         = fs->make<TH2D>( "hTrack_Sector_Phi",         "Seed sector vs. L1TkTrack #phi",            180, -M_PI, M_PI, 35, -0.5, 34.5 );
  hTrack_Wedge_Eta          = fs->make<TH2D>( "hTrack_Wedge_Eta",          "Seed wedge vs. L1TkTrack #eta",            180, -M_PI, M_PI, 35, -0.5, 34.5 );
  hTrack_RInv_Seed_RInv     = fs->make<TH2D>( "hTrack_RInv_Seed_RInv",     "L1TkTrack radius vs. Seed radius",          200, -0.01, 0.01, 200, -0.01, 0.01 );
  hTrack_RInvRes_Track_Eta  = fs->make<TH2D>( "hTrack_RInvRes_Track_Eta",  "L1TkTrack radius res. vs #eta",             180, -M_PI, M_PI, 100, -0.005, 0.005 );
  hTrack_Pt_Seed_Pt         = fs->make<TH2D>( "hTrack_Pt_Seed_Pt",         "L1TkTrack p_{T} vs. Seed p_{T}",            100, 0, 50, 100, 0, 50 );
  hTrack_PtRes_Track_Eta    = fs->make<TH2D>( "hTrack_PtRes_Track_Eta",    "L1TkTrack p_{T} res. vs #eta",              180, -M_PI, M_PI, 100, -4.0, 4.0 );
  hTrack_InvPt_Seed_InvPt   = fs->make<TH2D>( "hTrack_InvPt_Seed_InvPt",   "L1TkTrack p_{T}^{-1} vs. Seed p_{T}^{-1}",  200, 0, 0.8, 200, 0, 0.8 );
  hTrack_InvPt_Seed_InvPt->GetXaxis()->Set( NumBins, BinVec );
  hTrack_InvPt_Seed_InvPt->GetYaxis()->Set( NumBins, BinVec );
  hTrack_InvPtRes_Track_Eta = fs->make<TH2D>( "hTrack_InvPtRes_Track_Eta", "L1TkTrack p_{T}^{-1} res. vs #eta",         180, -M_PI, M_PI, 100, -0.4, 0.4 );
  hTrack_Phi_Seed_Phi       = fs->make<TH2D>( "hTrack_Phi_Seed_Phi",       "L1TkTrack #phi vs. Seed #phi",             180, -M_PI, M_PI, 180, -M_PI, M_PI );
  hTrack_PhiRes_Track_Eta   = fs->make<TH2D>( "hTrack_PhiRes_Track_Eta",   "L1TkTrack #phi res. vs #eta",               180, -M_PI, M_PI, 100, -0.1, 0.1 );
  hTrack_Eta_Seed_Eta       = fs->make<TH2D>( "hTrack_Eta_Seed_Eta",       "L1TkTrack #eta vs. Seed #eta",              180, -M_PI, M_PI, 180, -M_PI, M_PI );
  hTrack_EtaRes_Track_Eta   = fs->make<TH2D>( "hTrack_EtaRes_Track_Eta",   "L1TkTrack #eta res. vs #eta",               180, -M_PI, M_PI, 100, -0.5, 0.5 );
  hTrack_VtxZ0_Seed_VtxZ0   = fs->make<TH2D>( "hTrack_VtxZ0_Seed_VtxZ0",   "L1TkTrack z_{vtx} vs. Seed z_{vtx}",        180, -30, 30, 180, -30, 30 );
  hTrack_VtxZ0Res_Track_Eta = fs->make<TH2D>( "hTrack_VtxZ0Res_Track_Eta", "L1TkTrack z_{vtx} res. vs #eta",            180, -M_PI, M_PI, 100, -20, 20 );

  hTrack_NStubs_Sector->Sumw2();
  hTrack_NStubs_Wedge->Sumw2();
  hTrack_Sector_Phi->Sumw2();
  hTrack_Wedge_Eta->Sumw2();
  hTrack_RInv_Seed_RInv->Sumw2();
  hTrack_RInvRes_Track_Eta->Sumw2();
  hTrack_Pt_Seed_Pt->Sumw2();
  hTrack_PtRes_Track_Eta->Sumw2();
  hTrack_Phi_Seed_Phi->Sumw2();
  hTrack_PhiRes_Track_Eta->Sumw2();
  hTrack_Eta_Seed_Eta->Sumw2();
  hTrack_EtaRes_Track_Eta->Sumw2();
  hTrack_VtxZ0_Seed_VtxZ0->Sumw2();
  hTrack_VtxZ0Res_Track_Eta->Sumw2();

  for ( unsigned int seed = 1; seed < 11; seed++ )
  {
    for ( unsigned int targ = 1; targ < 11; targ++ )
    {
      std::pair< unsigned int, unsigned int > mapKey = std::make_pair( seed, targ );

      histoName.str("");  histoName << "hTrackProp_deltaRhoPhi_Eta_L" << seed << "_L" << targ;
      histoTitle.str(""); histoTitle << "#Delta#rho#phi vs seed #eta, Propagation from L " << seed << " to L " << targ;
      mapTrackPropBB_deltaRhoPhi_Eta[ mapKey ] = fs->make<TH2D>( histoName.str().c_str(),  histoTitle.str().c_str(),
                                                                 180, -M_PI, M_PI, 100, -1, 1 );

      histoName.str("");  histoName << "hTrackProp_deltaRhoPhi_L" << seed << "_L" << targ;
      histoTitle.str(""); histoTitle << "#Delta#rho#phi, Propagation from L " << seed << " to L " << targ;
      mapTrackPropBB_deltaRhoPhi[ mapKey ] = fs->make<TH1D>( histoName.str().c_str(),  histoTitle.str().c_str(),
                                                             100, -1, 1 );

      histoName.str("");  histoName << "hTrackProp_deltaZ_Eta_L" << seed << "_L" << targ;
      histoTitle.str(""); histoTitle << "#Deltaz vs seed #eta, Propagation from L " << seed << " to L " << targ;
      mapTrackPropBB_deltaZ_Eta[ mapKey ] = fs->make<TH2D>( histoName.str().c_str(),  histoTitle.str().c_str(),
                                                            180, -M_PI, M_PI, 200, -4, 4 );

      histoName.str("");  histoName << "hTrackProp_deltaZ_L" << seed << "_L" << targ;
      histoTitle.str(""); histoTitle << "#Deltaz, Propagation from L " << seed << " to L " << targ;
      mapTrackPropBB_deltaZ[ mapKey ] = fs->make<TH1D>( histoName.str().c_str(),  histoTitle.str().c_str(),
                                                        200, -4, 4 );

      mapTrackPropBB_deltaRhoPhi_Eta[ mapKey ]->Sumw2();
      mapTrackPropBB_deltaRhoPhi[ mapKey ]->Sumw2();
      mapTrackPropBB_deltaZ_Eta[ mapKey ]->Sumw2();
      mapTrackPropBB_deltaZ[ mapKey ]->Sumw2();

      histoName.str("");  histoName << "hTrackProp_deltaRhoPhi_Eta_L" << seed << "_D" << targ;
      histoTitle.str(""); histoTitle << "#Delta#rho#phi vs seed #eta, Propagation from L " << seed << " to D " << targ;
      mapTrackPropBE_deltaRhoPhi_Eta[ mapKey ] = fs->make<TH2D>( histoName.str().c_str(),  histoTitle.str().c_str(),
                                                                 180, -M_PI, M_PI, 100, -1, 1 );

      histoName.str("");  histoName << "hTrackProp_deltaRhoPhi_L" << seed << "_D" << targ;
      histoTitle.str(""); histoTitle << "#Delta#rho#phi, Propagation from L " << seed << " to D " << targ;
      mapTrackPropBE_deltaRhoPhi[ mapKey ] = fs->make<TH1D>( histoName.str().c_str(),  histoTitle.str().c_str(),
                                                             100, -1, 1 );

      histoName.str("");  histoName << "hTrackProp_deltaR_Eta_L" << seed << "_D" << targ;
      histoTitle.str(""); histoTitle << "#Delta#rho vs seed #eta, Propagation from L " << seed << " to D " << targ;
      mapTrackPropBE_deltaR_Eta[ mapKey ] = fs->make<TH2D>( histoName.str().c_str(),  histoTitle.str().c_str(),
                                                            180, -M_PI, M_PI, 200, -4, 4 );

      histoName.str("");  histoName << "hTrackProp_deltaR_L" << seed << "_D" << targ;
      histoTitle.str(""); histoTitle << "#Delta#rho, Propagation from L " << seed << " to D " << targ;
      mapTrackPropBE_deltaR[ mapKey ] = fs->make<TH1D>( histoName.str().c_str(),  histoTitle.str().c_str(),
                                                        200, -4, 4 );

      mapTrackPropBE_deltaRhoPhi_Eta[ mapKey ]->Sumw2();
      mapTrackPropBE_deltaRhoPhi[ mapKey ]->Sumw2();
      mapTrackPropBE_deltaR_Eta[ mapKey ]->Sumw2();
      mapTrackPropBE_deltaR[ mapKey ]->Sumw2();

      histoName.str("");  histoName << "hTrackProp_deltaRhoPhi_Eta_D" << seed << "_L" << targ;
      histoTitle.str(""); histoTitle << "#Delta#rho#phi vs seed #eta, Propagation from D " << seed << " to L " << targ;
      mapTrackPropEB_deltaRhoPhi_Eta[ mapKey ] = fs->make<TH2D>( histoName.str().c_str(),  histoTitle.str().c_str(),
                                                                 180, -M_PI, M_PI, 100, -1, 1 );

      histoName.str("");  histoName << "hTrackProp_deltaRhoPhi_D" << seed << "_L" << targ;
      histoTitle.str(""); histoTitle << "#Delta#rho#phi, Propagation from D " << seed << " to L " << targ;
      mapTrackPropEB_deltaRhoPhi[ mapKey ] = fs->make<TH1D>( histoName.str().c_str(),  histoTitle.str().c_str(),
                                                             100, -1, 1 );

      histoName.str("");  histoName << "hTrackProp_deltaZ_Eta_D" << seed << "_L" << targ;
      histoTitle.str(""); histoTitle << "#Deltaz vs seed #eta, Propagation from D " << seed << " to L " << targ;
      mapTrackPropEB_deltaZ_Eta[ mapKey ] = fs->make<TH2D>( histoName.str().c_str(),  histoTitle.str().c_str(),
                                                            180, -M_PI, M_PI, 400, -8, 8 );

      histoName.str("");  histoName << "hTrackProp_deltaZ_D" << seed << "_L" << targ;
      histoTitle.str(""); histoTitle << "#Deltaz, Propagation from D " << seed << " to L " << targ;
      mapTrackPropEB_deltaZ[ mapKey ] = fs->make<TH1D>( histoName.str().c_str(),  histoTitle.str().c_str(),
                                                        400, -8, 8 );

      mapTrackPropEB_deltaRhoPhi_Eta[ mapKey ]->Sumw2();
      mapTrackPropEB_deltaRhoPhi[ mapKey ]->Sumw2();
      mapTrackPropEB_deltaZ_Eta[ mapKey ]->Sumw2();
      mapTrackPropEB_deltaZ[ mapKey ]->Sumw2();

      histoName.str("");  histoName << "hTrackProp_deltaRhoPhi_Eta_D" << seed << "_D" << targ;
      histoTitle.str(""); histoTitle << "#Delta#rho#phi vs seed #eta, Propagation from D " << seed << " to D " << targ;
      mapTrackPropEE_deltaRhoPhi_Eta[ mapKey ] = fs->make<TH2D>( histoName.str().c_str(),  histoTitle.str().c_str(),
                                                                 180, -M_PI, M_PI, 100, -1, 1 );

      histoName.str("");  histoName << "hTrackProp_deltaRhoPhi_D" << seed << "_D" << targ;
      histoTitle.str(""); histoTitle << "#Delta#rho#phi, Propagation from D " << seed << " to D " << targ;
      mapTrackPropEE_deltaRhoPhi[ mapKey ] = fs->make<TH1D>( histoName.str().c_str(),  histoTitle.str().c_str(),
                                                             100, -1, 1 );

      histoName.str("");  histoName << "hTrackProp_deltaR_Eta_D" << seed << "_D" << targ;
      histoTitle.str(""); histoTitle << "#Delta#rho vs seed #eta, Propagation from D " << seed << " to D " << targ;
      mapTrackPropEE_deltaR_Eta[ mapKey ] = fs->make<TH2D>( histoName.str().c_str(),  histoTitle.str().c_str(),
                                                            180, -M_PI, M_PI, 200, -4, 4 );

      histoName.str("");  histoName << "hTrackProp_deltaR_D" << seed << "_D" << targ;
      histoTitle.str(""); histoTitle << "#Delta#rho, Propagation from D " << seed << " to D " << targ;
      mapTrackPropEE_deltaR[ mapKey ] = fs->make<TH1D>( histoName.str().c_str(),  histoTitle.str().c_str(),
                                                        200, -4, 4 );

      mapTrackPropEE_deltaRhoPhi_Eta[ mapKey ]->Sumw2();
      mapTrackPropEE_deltaRhoPhi[ mapKey ]->Sumw2();
      mapTrackPropEE_deltaR_Eta[ mapKey ]->Sumw2();
      mapTrackPropEE_deltaR[ mapKey ]->Sumw2();
    }
  }

  hTrack_3Stubs_Pt   = fs->make<TH1D>( "hTrack_3Stubs_Pt",  "L1TkTrack 3Stubs p_{T}",    100, 0, 50 );
  hTrack_3Stubs_Phi  = fs->make<TH1D>( "hTrack_3Stubs_Phi", "L1TkTrack 3Stubs #phi",     180, -M_PI, M_PI );
  hTrack_3Stubs_Eta  = fs->make<TH1D>( "hTrack_3Stubs_Eta", "L1TkTrack 3Stubs #eta",     180, -M_PI, M_PI );
  hTrack_3Stubs_Pt->Sumw2();
  hTrack_3Stubs_Phi->Sumw2();
  hTrack_3Stubs_Eta->Sumw2();

  hTrack_2Stubs_Pt   = fs->make<TH1D>( "hTrack_2Stubs_Pt",  "L1TkTrack 2Stubs p_{T}",    100, 0, 50 );
  hTrack_2Stubs_Phi  = fs->make<TH1D>( "hTrack_2Stubs_Phi", "L1TkTrack 2Stubs #phi",     180, -M_PI, M_PI );
  hTrack_2Stubs_Eta  = fs->make<TH1D>( "hTrack_2Stubs_Eta", "L1TkTrack 2Stubs #eta",     180, -M_PI, M_PI );
  hTrack_2Stubs_Pt->Sumw2();
  hTrack_2Stubs_Phi->Sumw2();
  hTrack_2Stubs_Eta->Sumw2();

  hTrack_Seed_Pt   = fs->make<TH1D>( "hTrack_Seed_Pt",  "L1TkTrack Seed p_{T}",    100, 0, 50 );
  hTrack_Seed_Phi  = fs->make<TH1D>( "hTrack_Seed_Phi", "L1TkTrack Seed #phi",     180, -M_PI, M_PI );
  hTrack_Seed_Eta  = fs->make<TH1D>( "hTrack_Seed_Eta", "L1TkTrack Seed #eta",     180, -M_PI, M_PI );
  hTrack_Seed_Pt->Sumw2();
  hTrack_Seed_Phi->Sumw2();
  hTrack_Seed_Eta->Sumw2();

  hSeed_Pt         = fs->make<TH1D>( "hSeed_Pt",        "Seed p_{T}",              100, 0, 50 );
  hSeed_Phi        = fs->make<TH1D>( "hSeed_Phi",       "Seed #phi",               180, -M_PI, M_PI );
  hSeed_Eta        = fs->make<TH1D>( "hSeed_Eta",       "Seed #eta",               180, -M_PI, M_PI );
  hSeed_Pt->Sumw2();
  hSeed_Phi->Sumw2();
  hSeed_Eta->Sumw2();

  hSimTrack_Track_3Stubs_Pt       = fs->make<TH1D>( "hSimTrack_Track_3Stubs_Pt",      "L1TkTrack 3Stubs SimTrack p_{T}",    100, 0, 50 );
  hSimTrack_Track_3Stubs_Phi_Pt5  = fs->make<TH1D>( "hSimTrack_Track_3Stubs_Phi_Pt5", "L1TkTrack 3Stubs SimTrack #phi",     180, -M_PI, M_PI );
  hSimTrack_Track_3Stubs_Eta_Pt5  = fs->make<TH1D>( "hSimTrack_Track_3Stubs_Eta_Pt5", "L1TkTrack 3Stubs SimTrack  #eta",    180, -M_PI, M_PI );
  hSimTrack_Track_3Stubs_Pt->Sumw2();
  hSimTrack_Track_3Stubs_Phi_Pt5->Sumw2();
  hSimTrack_Track_3Stubs_Eta_Pt5->Sumw2();

  hSimTrack_Track_2Stubs_Pt       = fs->make<TH1D>( "hSimTrack_Track_2Stubs_Pt",      "L1TkTrack 2Stubs SimTrack p_{T}",    100, 0, 50 );
  hSimTrack_Track_2Stubs_Phi_Pt5  = fs->make<TH1D>( "hSimTrack_Track_2Stubs_Phi_Pt5", "L1TkTrack 2Stubs SimTrack #phi",     180, -M_PI, M_PI );
  hSimTrack_Track_2Stubs_Eta_Pt5  = fs->make<TH1D>( "hSimTrack_Track_2Stubs_Eta_Pt5", "L1TkTrack 2Stubs SimTrack  #eta",    180, -M_PI, M_PI );
  hSimTrack_Track_2Stubs_Pt->Sumw2();
  hSimTrack_Track_2Stubs_Phi_Pt5->Sumw2();
  hSimTrack_Track_2Stubs_Eta_Pt5->Sumw2();

  hSimTrack_Seed_Pt           = fs->make<TH1D>( "hSimTrack_Seed_Pt",          "Seed SimTrack p_{T}",         100, 0, 50 );
  hSimTrack_Seed_Phi_Pt5      = fs->make<TH1D>( "hSimTrack_Seed_Phi_Pt5",     "Seed SimTrack #phi",          180, -M_PI, M_PI );
  hSimTrack_Seed_Eta_Pt5      = fs->make<TH1D>( "hSimTrack_Seed_Eta_Pt5",     "Seed SimTrack #eta",          180, -M_PI, M_PI );
  hSimTrack_Seed_Pt->Sumw2();
  hSimTrack_Seed_Phi_Pt5->Sumw2();
  hSimTrack_Seed_Eta_Pt5->Sumw2();

  hSimTrack_Cluster_Pt           = fs->make<TH1D>( "hSimTrack_Cluster_Pt",          "Cluster SimTrack p_{T}",         100, 0, 50 );
  hSimTrack_Cluster_Phi_Pt5      = fs->make<TH1D>( "hSimTrack_Cluster_Phi_Pt5",     "Cluster SimTrack #phi",          180, -M_PI, M_PI );
  hSimTrack_Cluster_Eta_Pt5      = fs->make<TH1D>( "hSimTrack_Cluster_Eta_Pt5",     "Cluster SimTrack #eta",          180, -M_PI, M_PI );
  hSimTrack_Cluster_Pt->Sumw2();
  hSimTrack_Cluster_Phi_Pt5->Sumw2();
  hSimTrack_Cluster_Eta_Pt5->Sumw2();

  hSimTrack_Stub_Pt           = fs->make<TH1D>( "hSimTrack_Stub_Pt",          "Stub SimTrack p_{T}",         100, 0, 50 );
  hSimTrack_Stub_Phi_Pt5      = fs->make<TH1D>( "hSimTrack_Stub_Phi_Pt5",     "Stub SimTrack #phi",          180, -M_PI, M_PI );
  hSimTrack_Stub_Eta_Pt5      = fs->make<TH1D>( "hSimTrack_Stub_Eta_Pt5",     "Stub SimTrack #eta",          180, -M_PI, M_PI );
  hSimTrack_Stub_Pt->Sumw2();
  hSimTrack_Stub_Phi_Pt5->Sumw2();
  hSimTrack_Stub_Eta_Pt5->Sumw2();

  hTrack_3Stubs_N                     = fs->make<TH1D>( "hTrack_3Stubs_N",                      "Number of L1TkTrack 3Stubs",                                            100, -0.5, 99.5 );
  hTrack_3Stubs_Pt_SimTrack_Pt        = fs->make<TH2D>( "hTrack_3Stubs_Pt_SimTrack_Pt",         "L1TkTrack 3Stubs p_{T} vs. SimTrack p_{T}",                             100, 0, 50, 100, 0, 50 );
  hTrack_3Stubs_PtRes_SimTrack_Eta    = fs->make<TH2D>( "hTrack_3Stubs_PtRes_SimTrack_Eta",     "L1TkTrack 3Stubs p_{T} - SimTrack p_{T} vs. SimTrack #eta",             180, -M_PI, M_PI, 200, -4.0, 4.0 );
  hTrack_3Stubs_InvPt_SimTrack_InvPt  = fs->make<TH2D>( "hTrack_3Stubs_InvPt_SimTrack_InvPt",   "L1TkTrack 3Stubs p_{T}^{-1} vs. SimTrack p_{T}^{-1}",                   200, 0, 0.8, 200, 0, 0.8 );
  hTrack_3Stubs_InvPt_SimTrack_InvPt->GetXaxis()->Set( NumBins, BinVec );
  hTrack_3Stubs_InvPt_SimTrack_InvPt->GetYaxis()->Set( NumBins, BinVec );
  hTrack_3Stubs_InvPtRes_SimTrack_Eta = fs->make<TH2D>( "hTrack_3Stubs_InvPtRes_SimTrack_Eta",  "L1TkTrack 3Stubs p_{T}^{-1} - SimTrack p_{T}^{-1}  vs. SimTrack #eta",  180, -M_PI, M_PI, 100, -1.0, 1.0 );
  hTrack_3Stubs_Phi_SimTrack_Phi      = fs->make<TH2D>( "hTrack_3Stubs_Phi_SimTrack_Phi",       "L1TkTrack 3Stubs #phi vs. SimTrack #phi",                               180, -M_PI, M_PI, 180, -M_PI, M_PI );
  hTrack_3Stubs_PhiRes_SimTrack_Eta   = fs->make<TH2D>( "hTrack_3Stubs_PhiRes_SimTrack_Eta",    "L1TkTrack 3Stubs #phi - SimTrack #phi vs. SimTrack #eta",               180, -M_PI, M_PI, 100, -0.5, 0.5 );
  hTrack_3Stubs_Eta_SimTrack_Eta      = fs->make<TH2D>( "hTrack_3Stubs_Eta_SimTrack_Eta",       "L1TkTrack 3Stubs #eta vs. SimTrack #eta",                               180, -M_PI, M_PI, 180, -M_PI, M_PI );
  hTrack_3Stubs_EtaRes_SimTrack_Eta   = fs->make<TH2D>( "hTrack_3Stubs_EtaRes_SimTrack_Eta",    "L1TkTrack 3Stubs #eta - SimTrack #eta vs. SimTrack #eta",               180, -M_PI, M_PI, 100, -0.5, 0.5 );
  hTrack_3Stubs_VtxZ0_SimTrack_VtxZ0  = fs->make<TH2D>( "hTrack_3Stubs_VtxZ0_SimTrack_VtxZ0",   "L1TkTrack 3Stubs z_{vtx} vs. SimTrack z_{vtx}",                         180, -30, 30, 180, -30, 30 );
  hTrack_3Stubs_VtxZ0Res_SimTrack_Eta = fs->make<TH2D>( "hTrack_3Stubs_VtxZ0Res_SimTrack_Eta",  "L1TkTrack 3Stubs z_{vtx} - SimTrack z_{vtx} vs. SimTrack #eta",         180, -M_PI, M_PI, 100, -5, 5 );
  hTrack_3Stubs_Chi2_NStubs           = fs->make<TH2D>( "hTrack_3Stubs_Chi2_NStubs",            "L1TkTrack 3Stubs #chi^{2} vs. number of Stubs",                         20, -0.5, 19.5, 200, 0, 50 );
  hTrack_3Stubs_Chi2_SimTrack_Eta     = fs->make<TH2D>( "hTrack_3Stubs_Chi2_SimTrack_Eta",      "L1TkTrack 3Stubs #chi^{2} vs. SimTrack #eta",                           180, -M_PI, M_PI, 200, 0, 50 );
  hTrack_3Stubs_Chi2Red_NStubs        = fs->make<TH2D>( "hTrack_3Stubs_Chi2Red_NStubs",         "L1TkTrack 3Stubs #chi^{2}/dof vs. number of Stubs",                     20, -0.5, 19.5, 200, 0, 10 );
  hTrack_3Stubs_Chi2Red_SimTrack_Eta  = fs->make<TH2D>( "hTrack_3Stubs_Chi2Red_SimTrack_Eta",   "L1TkTrack 3Stubs #chi^{2}/dof vs. SimTrack #eta",                       180, -M_PI, M_PI, 200, 0, 10 );
  hTrack_3Stubs_N->Sumw2();
  hTrack_3Stubs_Pt_SimTrack_Pt->Sumw2();
  hTrack_3Stubs_PtRes_SimTrack_Eta->Sumw2();
  hTrack_3Stubs_InvPt_SimTrack_InvPt->Sumw2();
  hTrack_3Stubs_InvPtRes_SimTrack_Eta->Sumw2();
  hTrack_3Stubs_Phi_SimTrack_Phi->Sumw2();
  hTrack_3Stubs_PhiRes_SimTrack_Eta->Sumw2();
  hTrack_3Stubs_Eta_SimTrack_Eta->Sumw2();
  hTrack_3Stubs_EtaRes_SimTrack_Eta->Sumw2();
  hTrack_3Stubs_VtxZ0_SimTrack_VtxZ0->Sumw2();
  hTrack_3Stubs_VtxZ0Res_SimTrack_Eta->Sumw2();
  hTrack_3Stubs_Chi2_NStubs->Sumw2();
  hTrack_3Stubs_Chi2_SimTrack_Eta->Sumw2();
  hTrack_3Stubs_Chi2Red_NStubs->Sumw2();
  hTrack_3Stubs_Chi2Red_SimTrack_Eta->Sumw2();

  hTrack_2Stubs_N                     = fs->make<TH1D>( "hTrack_2Stubs_N",                      "Number of L1TkTrack 2Stubs",                                            100, -0.5, 99.5 );
  hTrack_2Stubs_Pt_SimTrack_Pt        = fs->make<TH2D>( "hTrack_2Stubs_Pt_SimTrack_Pt",         "L1TkTrack 2Stubs p_{T} vs. SimTrack p_{T}",                             100, 0, 50, 100, 0, 50 );
  hTrack_2Stubs_PtRes_SimTrack_Eta    = fs->make<TH2D>( "hTrack_2Stubs_PtRes_SimTrack_Eta",     "L1TkTrack 2Stubs p_{T} - SimTrack p_{T} vs. SimTrack #eta",             180, -M_PI, M_PI, 200, -4.0, 4.0 );
  hTrack_2Stubs_InvPt_SimTrack_InvPt  = fs->make<TH2D>( "hTrack_2Stubs_InvPt_SimTrack_InvPt",   "L1TkTrack 2Stubs p_{T}^{-1} vs. SimTrack p_{T}^{-1}",                   200, 0, 0.8, 200, 0, 0.8 );
  hTrack_2Stubs_InvPt_SimTrack_InvPt->GetXaxis()->Set( NumBins, BinVec );
  hTrack_2Stubs_InvPt_SimTrack_InvPt->GetYaxis()->Set( NumBins, BinVec );
  hTrack_2Stubs_InvPtRes_SimTrack_Eta = fs->make<TH2D>( "hTrack_2Stubs_InvPtRes_SimTrack_Eta",  "L1TkTrack 2Stubs p_{T}^{-1} - SimTrack p_{T}^{-1}  vs. SimTrack #eta",  180, -M_PI, M_PI, 100, -1.0, 1.0 );
  hTrack_2Stubs_Phi_SimTrack_Phi      = fs->make<TH2D>( "hTrack_2Stubs_Phi_SimTrack_Phi",       "L1TkTrack 2Stubs #phi vs. SimTrack #phi",                               180, -M_PI, M_PI, 180, -M_PI, M_PI );
  hTrack_2Stubs_PhiRes_SimTrack_Eta   = fs->make<TH2D>( "hTrack_2Stubs_PhiRes_SimTrack_Eta",    "L1TkTrack 2Stubs #phi - SimTrack #phi vs. SimTrack #eta",               180, -M_PI, M_PI, 100, -0.5, 0.5 );
  hTrack_2Stubs_Eta_SimTrack_Eta      = fs->make<TH2D>( "hTrack_2Stubs_Eta_SimTrack_Eta",       "L1TkTrack 2Stubs #eta vs. SimTrack #eta",                               180, -M_PI, M_PI, 180, -M_PI, M_PI );
  hTrack_2Stubs_EtaRes_SimTrack_Eta   = fs->make<TH2D>( "hTrack_2Stubs_EtaRes_SimTrack_Eta",    "L1TkTrack 2Stubs #eta - SimTrack #eta vs. SimTrack #eta",               180, -M_PI, M_PI, 100, -0.5, 0.5 );
  hTrack_2Stubs_VtxZ0_SimTrack_VtxZ0  = fs->make<TH2D>( "hTrack_2Stubs_VtxZ0_SimTrack_VtxZ0",   "L1TkTrack 2Stubs z_{vtx} vs. SimTrack z_{vtx}",                         180, -30, 30, 180, -30, 30 );
  hTrack_2Stubs_VtxZ0Res_SimTrack_Eta = fs->make<TH2D>( "hTrack_2Stubs_VtxZ0Res_SimTrack_Eta",  "L1TkTrack 2Stubs z_{vtx} - SimTrack z_{vtx} vs. SimTrack #eta",         180, -M_PI, M_PI, 100, -5, 5 );
  hTrack_2Stubs_Chi2_NStubs           = fs->make<TH2D>( "hTrack_2Stubs_Chi2_NStubs",            "L1TkTrack 2Stubs #chi^{2} vs. number of Stubs",                         20, -0.5, 19.5, 200, 0, 50 );
  hTrack_2Stubs_Chi2_SimTrack_Eta     = fs->make<TH2D>( "hTrack_2Stubs_Chi2_SimTrack_Eta",      "L1TkTrack 2Stubs #chi^{2} vs. SimTrack #eta",                           180, -M_PI, M_PI, 200, 0, 50 );
  hTrack_2Stubs_Chi2Red_NStubs        = fs->make<TH2D>( "hTrack_2Stubs_Chi2Red_NStubs",         "L1TkTrack 2Stubs #chi^{2}/dof vs. number of Stubs",                     20, -0.5, 19.5, 200, 0, 10 );
  hTrack_2Stubs_Chi2Red_SimTrack_Eta  = fs->make<TH2D>( "hTrack_2Stubs_Chi2Red_SimTrack_Eta",   "L1TkTrack 2Stubs #chi^{2}/dof vs. SimTrack #eta",                       180, -M_PI, M_PI, 200, 0, 10 );

  hTrack_2Stubs_N->Sumw2();
  hTrack_2Stubs_Pt_SimTrack_Pt->Sumw2();
  hTrack_2Stubs_PtRes_SimTrack_Eta->Sumw2();
  hTrack_2Stubs_InvPt_SimTrack_InvPt->Sumw2();
  hTrack_2Stubs_InvPtRes_SimTrack_Eta->Sumw2();
  hTrack_2Stubs_Phi_SimTrack_Phi->Sumw2();
  hTrack_2Stubs_PhiRes_SimTrack_Eta->Sumw2();
  hTrack_2Stubs_Eta_SimTrack_Eta->Sumw2();
  hTrack_2Stubs_EtaRes_SimTrack_Eta->Sumw2();
  hTrack_2Stubs_VtxZ0_SimTrack_VtxZ0->Sumw2();
  hTrack_2Stubs_VtxZ0Res_SimTrack_Eta->Sumw2();
  hTrack_2Stubs_Chi2_NStubs->Sumw2();
  hTrack_2Stubs_Chi2_SimTrack_Eta->Sumw2();
  hTrack_2Stubs_Chi2Red_NStubs->Sumw2();
  hTrack_2Stubs_Chi2Red_SimTrack_Eta->Sumw2();

  hSeed_N                     = fs->make<TH1D>( "hSeed_N",                      "Number of Seed",                                            100, -0.5, 99.5 );
  hSeed_Pt_SimTrack_Pt        = fs->make<TH2D>( "hSeed_Pt_SimTrack_Pt",         "Seed p_{T} vs. SimTrack p_{T}",                             100, 0, 50, 100, 0, 50 );
  hSeed_PtRes_SimTrack_Eta    = fs->make<TH2D>( "hSeed_PtRes_SimTrack_Eta",     "Seed p_{T} - SimTrack p_{T} vs. SimTrack #eta",             180, -M_PI, M_PI, 200, -4.0, 4.0 );
  hSeed_InvPt_SimTrack_InvPt  = fs->make<TH2D>( "hSeed_InvPt_SimTrack_InvPt",   "Seed p_{T}^{-1} vs. SimTrack p_{T}^{-1}",                   200, 0, 0.8, 200, 0, 0.8 );
  hSeed_InvPt_SimTrack_InvPt->GetXaxis()->Set( NumBins, BinVec );
  hSeed_InvPt_SimTrack_InvPt->GetYaxis()->Set( NumBins, BinVec );
  hSeed_InvPtRes_SimTrack_Eta = fs->make<TH2D>( "hSeed_InvPtRes_SimTrack_Eta",  "Seed p_{T}^{-1} - SimTrack p_{T}^{-1}  vs. SimTrack #eta",  180, -M_PI, M_PI, 100, -1.0, 1.0 );
  hSeed_Phi_SimTrack_Phi      = fs->make<TH2D>( "hSeed_Phi_SimTrack_Phi",       "Seed #phi vs. SimTrack #phi",                               180, -M_PI, M_PI, 180, -M_PI, M_PI );
  hSeed_PhiRes_SimTrack_Eta   = fs->make<TH2D>( "hSeed_PhiRes_SimTrack_Eta",    "Seed #phi - SimTrack #phi vs. SimTrack #eta",               180, -M_PI, M_PI, 100, -0.5, 0.5 );
  hSeed_Eta_SimTrack_Eta      = fs->make<TH2D>( "hSeed_Eta_SimTrack_Eta",       "Seed #eta vs. SimTrack #eta",                               180, -M_PI, M_PI, 180, -M_PI, M_PI );
  hSeed_EtaRes_SimTrack_Eta   = fs->make<TH2D>( "hSeed_EtaRes_SimTrack_Eta",    "Seed #eta - SimTrack #eta vs. SimTrack #eta",               180, -M_PI, M_PI, 100, -0.5, 0.5 );
  hSeed_VtxZ0_SimTrack_VtxZ0  = fs->make<TH2D>( "hSeed_VtxZ0_SimTrack_VtxZ0",   "Seed z_{vtx} vs. SimTrack z_{vtx}",                         180, -30, 30, 180, -30, 30 );
  hSeed_VtxZ0Res_SimTrack_Eta = fs->make<TH2D>( "hSeed_VtxZ0Res_SimTrack_Eta",  "Seed z_{vtx} - SimTrack z_{vtx} vs. SimTrack #eta",         180, -M_PI, M_PI, 100, -5, 5 );
  hSeed_Chi2_NStubs           = fs->make<TH2D>( "hSeed_Chi2_NStubs",            "Seed #chi^{2} vs. number of Stubs",                         20, -0.5, 19.5, 200, 0, 50 );
  hSeed_Chi2_SimTrack_Eta     = fs->make<TH2D>( "hSeed_Chi2_SimTrack_Eta",      "Seed #chi^{2} vs. SimTrack #eta",                           180, -M_PI, M_PI, 200, 0, 50 );
  hSeed_Chi2Red_NStubs        = fs->make<TH2D>( "hSeed_Chi2Red_NStubs",         "Seed #chi^{2}/dof vs. number of Stubs",                     20, -0.5, 19.5, 200, 0, 10 );
  hSeed_Chi2Red_SimTrack_Eta  = fs->make<TH2D>( "hSeed_Chi2Red_SimTrack_Eta",   "Seed #chi^{2}/dof vs. SimTrack #eta",                       180, -M_PI, M_PI, 200, 0, 10 );
  hSeed_N->Sumw2();
  hSeed_Pt_SimTrack_Pt->Sumw2();
  hSeed_PtRes_SimTrack_Eta->Sumw2();
  hSeed_InvPt_SimTrack_InvPt->Sumw2();
  hSeed_InvPtRes_SimTrack_Eta->Sumw2();
  hSeed_Phi_SimTrack_Phi->Sumw2();
  hSeed_PhiRes_SimTrack_Eta->Sumw2();
  hSeed_Eta_SimTrack_Eta->Sumw2();
  hSeed_EtaRes_SimTrack_Eta->Sumw2();
  hSeed_VtxZ0_SimTrack_VtxZ0->Sumw2();
  hSeed_VtxZ0Res_SimTrack_Eta->Sumw2();
  hSeed_Chi2_NStubs->Sumw2();
  hSeed_Chi2_SimTrack_Eta->Sumw2();
  hSeed_Chi2Red_NStubs->Sumw2();
  hSeed_Chi2Red_SimTrack_Eta->Sumw2();

  /// End of things to be done before entering the event Loop
}

//////////
// ANALYZE
void ValidateL1Track::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  /// Geometry handles etc
//  edm::ESHandle< TrackerGeometry >         geometryHandle;
//  const TrackerGeometry*                   theGeometry;
//  edm::ESHandle< StackedTrackerGeometry >  stackedGeometryHandle;

  /// Geometry setup
  /// Set pointers to Geometry
//  iSetup.get<TrackerDigiGeometryRecord>().get(geometryHandle);
//  theGeometry = &(*geometryHandle);


  /// Sim Tracks and Vtx
  //edm::Handle< edm::SimTrackContainer >  SimTrackHandle;
  edm::Handle< edm::SimVertexContainer > SimVtxHandle;
  //iEvent.getByLabel( "famosSimHits", SimTrackHandle );
  //iEvent.getByLabel( "famosSimHits", SimVtxHandle );
  //iEvent.getByLabel( "g4SimHits", SimTrackHandle );
  iEvent.getByLabel( "g4SimHits", SimVtxHandle );

  /// Get geometry
  edm::ESHandle< StackedTrackerGeometry >  StackedGeometryHandle;
  const StackedTrackerGeometry*            theStackedGeometry;
  iSetup.get< StackedTrackerGeometryRecord >().get(StackedGeometryHandle);
  theStackedGeometry = StackedGeometryHandle.product();

  /// Track Trigger
  edm::Handle< L1TkCluster_PixelDigi_Collection > PixelDigiL1TkClusterHandle;
  edm::Handle< L1TkStub_PixelDigi_Collection >    PixelDigiL1TkStubHandle;
  iEvent.getByLabel( "L1TkClustersFromPixelDigis",             PixelDigiL1TkClusterHandle );
  iEvent.getByLabel( "L1TkStubsFromPixelDigis", "StubsPass",   PixelDigiL1TkStubHandle );

  /// Maps to store SimTrack information
  std::map< unsigned int, double > mapSimTrackCluPt;
  std::map< unsigned int, double > mapSimTrackCluPhi;
  std::map< unsigned int, double > mapSimTrackCluEta;

  /// Go on only if there are L1TkCluster from PixelDigis
  if ( PixelDigiL1TkClusterHandle->size() > 0 )
  {
    /// Loop over L1TkClusters
    L1TkCluster_PixelDigi_Collection::const_iterator iterL1TkCluster;
    for ( iterL1TkCluster = PixelDigiL1TkClusterHandle->begin();
          iterL1TkCluster != PixelDigiL1TkClusterHandle->end();
          ++iterL1TkCluster )
    {
      bool genuineClu     = iterL1TkCluster->isGenuine();
      bool combinClu      = iterL1TkCluster->isCombinatoric();

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

      if ( mapSimTrackCluPt.find( simTrackPtr->trackId() ) == mapSimTrackCluPt.end() )
      {
        /// New map entry
        mapSimTrackCluPt.insert( std::make_pair( simTrackPtr->trackId(), simTrackPtr->momentum().pt() ) );
        mapSimTrackCluPhi.insert( std::make_pair( simTrackPtr->trackId(), simTrackPtr->momentum().phi() ) );
        mapSimTrackCluEta.insert( std::make_pair( simTrackPtr->trackId(), simTrackPtr->momentum().eta() ) );
      }
    }
  }
  }

  std::map< unsigned int, double >::iterator iterMapSimTrack;

  for ( iterMapSimTrack = mapSimTrackCluPt.begin();
        iterMapSimTrack != mapSimTrackCluPt.end();
        ++iterMapSimTrack )
  {
    hSimTrack_Cluster_Pt->Fill( iterMapSimTrack->second );
  }
  for ( iterMapSimTrack = mapSimTrackCluPhi.begin();
        iterMapSimTrack != mapSimTrackCluPhi.end();
        ++iterMapSimTrack )
  {
    if ( mapSimTrackCluPt.find( iterMapSimTrack->first )->second > 5 )
      hSimTrack_Cluster_Phi_Pt5->Fill( iterMapSimTrack->second );
  }
  for ( iterMapSimTrack = mapSimTrackCluEta.begin();
        iterMapSimTrack != mapSimTrackCluEta.end();
        ++iterMapSimTrack )
  {
    if ( mapSimTrackCluPt.find( iterMapSimTrack->first )->second > 5 )
      hSimTrack_Cluster_Eta_Pt5->Fill( iterMapSimTrack->second );
  }

  /// Maps to store SimTrack information
  std::map< unsigned int, double > mapSimTrackStubPt;
  std::map< unsigned int, double > mapSimTrackStubPhi;
  std::map< unsigned int, double > mapSimTrackStubEta;

  /// Go on only if there are L1TkStubs from PixelDigis
  if ( PixelDigiL1TkStubHandle->size() > 0 )
  {
    /// Loop over L1TkStubs
    L1TkStub_PixelDigi_Collection::const_iterator iterL1TkStub;
    for ( iterL1TkStub = PixelDigiL1TkStubHandle->begin();
          iterL1TkStub != PixelDigiL1TkStubHandle->end();
          ++iterL1TkStub )
    {
      bool genuineStub     = iterL1TkStub->isGenuine();

      /// Store Track information in maps, skip if the Stub is not good
      if ( !genuineStub ) continue;

      edm::Ptr< SimTrack > simTrackPtr = iterL1TkStub->getSimTrackPtr();

      /// Get the corresponding vertex and reject the track
      /// if its vertex is outside the beampipe
      int vertexIndex = simTrackPtr->vertIndex();
      const SimVertex& theSimVertex = (*SimVtxHandle)[vertexIndex];
      math::XYZTLorentzVectorD trkVtxPos = theSimVertex.position();

      if ( trkVtxPos.rho() >= 2 )
        continue;

      if ( mapSimTrackStubPt.find( simTrackPtr->trackId() ) == mapSimTrackStubPt.end() )
      {
        /// New map entry
        mapSimTrackStubPt.insert( std::make_pair( simTrackPtr->trackId(), simTrackPtr->momentum().pt() ) );
        mapSimTrackStubPhi.insert( std::make_pair( simTrackPtr->trackId(), simTrackPtr->momentum().phi() ) );
        mapSimTrackStubEta.insert( std::make_pair( simTrackPtr->trackId(), simTrackPtr->momentum().eta() ) );
      }
    }
  }

  for ( iterMapSimTrack = mapSimTrackStubPt.begin();
        iterMapSimTrack != mapSimTrackStubPt.end();
        ++iterMapSimTrack )
  {
    hSimTrack_Stub_Pt->Fill( iterMapSimTrack->second );
  }
  for ( iterMapSimTrack = mapSimTrackStubPhi.begin();
        iterMapSimTrack != mapSimTrackStubPhi.end();
        ++iterMapSimTrack )
  {
    if ( mapSimTrackStubPt.find( iterMapSimTrack->first )->second > 5 )
      hSimTrack_Stub_Phi_Pt5->Fill( iterMapSimTrack->second );
  }
  for ( iterMapSimTrack = mapSimTrackStubEta.begin();
        iterMapSimTrack != mapSimTrackStubEta.end();
        ++iterMapSimTrack )
  {
    if ( mapSimTrackStubPt.find( iterMapSimTrack->first )->second > 5 )
      hSimTrack_Stub_Eta_Pt5->Fill( iterMapSimTrack->second );
  }

  /// Track Trigger
  edm::Handle< L1TkTrack_PixelDigi_Collection >    PixelDigiL1TkSeedHandle;
  edm::Handle< L1TkTrack_PixelDigi_Collection >    PixelDigiL1TkTrackHandle;
  iEvent.getByLabel( "L1TkTracksFromPixelDigis", "Seeds",   PixelDigiL1TkSeedHandle );
  iEvent.getByLabel( "L1TkTracksFromPixelDigis", "NoDup",   PixelDigiL1TkTrackHandle );

  /// Maps to store SimTrack information
  std::map< unsigned int, double > mapSimTrackTrack2Pt;
  std::map< unsigned int, double > mapSimTrackTrack2Phi;
  std::map< unsigned int, double > mapSimTrackTrack2Eta;
  std::map< unsigned int, double > mapSimTrackTrack3Pt;
  std::map< unsigned int, double > mapSimTrackTrack3Phi;
  std::map< unsigned int, double > mapSimTrackTrack3Eta;
  unsigned int num3Stubs = 0;
  unsigned int num2Stubs = 0;

  /// Go on only if there are L1TkTracks from PixelDigis
  if ( PixelDigiL1TkTrackHandle->size() > 0 )
  {
    /// Loop over L1TkTracks
    L1TkTrack_PixelDigi_Collection::const_iterator iterL1TkTrack;
    for ( iterL1TkTrack = PixelDigiL1TkTrackHandle->begin();
          iterL1TkTrack != PixelDigiL1TkTrackHandle->end();
          ++iterL1TkTrack )
    {
      /// Get everything is relevant
      unsigned int nStubs     = iterL1TkTrack->getStubPtrs().size();
      unsigned int seedSector = iterL1TkTrack->getSector();
      unsigned int seedWedge = iterL1TkTrack->getWedge();

      hTrack_NStubs_Sector->Fill( seedSector, nStubs );
      hTrack_NStubs_Wedge->Fill( seedWedge, nStubs );

      double trackRInv  = iterL1TkTrack->getRInv();
      double trackPt    = iterL1TkTrack->getMomentum().perp();
      double trackPhi   = iterL1TkTrack->getMomentum().phi();
      double trackEta   = iterL1TkTrack->getMomentum().eta();
      double trackVtxZ0 = iterL1TkTrack->getVertex().z();
      double trackChi2  = iterL1TkTrack->getChi2();
      double trackChi2R = iterL1TkTrack->getChi2Red();


      hTrack_Sector_Phi->Fill( trackPhi, seedSector );
      hTrack_Wedge_Eta->Fill( trackEta, seedWedge );

      bool genuineTrack = iterL1TkTrack->isGenuine();
      
      if ( !genuineTrack ) continue;

      edm::Ptr< SimTrack > simTrackPtr = iterL1TkTrack->getSimTrackPtr();

      /// Get the corresponding vertex and reject the track
      /// if its vertex is outside the beampipe
      int vertexIndex = simTrackPtr->vertIndex();
      const SimVertex& theSimVertex = (*SimVtxHandle)[vertexIndex];
      math::XYZTLorentzVectorD trkVtxPos = theSimVertex.position();

      if ( trkVtxPos.rho() >= 2 )
        continue;

      double simTrackPt = simTrackPtr->momentum().pt();
      double simTrackEta = simTrackPtr->momentum().eta();
      double simTrackPhi = simTrackPtr->momentum().phi();
      double simTrackVtxZ0 = ((*SimVtxHandle)[simTrackPtr->vertIndex()]).position().z();

      if ( nStubs > 2 )
      {
        hTrack_3Stubs_Pt->Fill( trackPt );
        hTrack_3Stubs_Eta->Fill( trackEta );
        hTrack_3Stubs_Phi->Fill( trackPhi );

        if ( mapSimTrackTrack3Pt.find( simTrackPtr->trackId() ) == mapSimTrackTrack3Pt.end() )
        {
          /// New map entry
          mapSimTrackTrack3Pt.insert( std::make_pair( simTrackPtr->trackId(), simTrackPt ) );
          mapSimTrackTrack3Phi.insert( std::make_pair( simTrackPtr->trackId(), simTrackPhi ) );
          mapSimTrackTrack3Eta.insert( std::make_pair( simTrackPtr->trackId(), simTrackEta ) );
        }

        num3Stubs++;
        hTrack_3Stubs_Pt_SimTrack_Pt->Fill( simTrackPt, trackPt );
        hTrack_3Stubs_PtRes_SimTrack_Eta->Fill( simTrackEta, trackPt - simTrackPt);
        hTrack_3Stubs_InvPt_SimTrack_InvPt->Fill( 1./simTrackPt, 1./trackPt );
        hTrack_3Stubs_InvPtRes_SimTrack_Eta->Fill( simTrackEta, 1./trackPt - 1./simTrackPt);
        hTrack_3Stubs_Phi_SimTrack_Phi->Fill( simTrackPhi, trackPhi );
        hTrack_3Stubs_PhiRes_SimTrack_Eta->Fill( simTrackEta, trackPhi - simTrackPhi);
        hTrack_3Stubs_Eta_SimTrack_Eta->Fill( simTrackEta, trackEta );
        hTrack_3Stubs_EtaRes_SimTrack_Eta->Fill( simTrackEta, trackEta - simTrackEta);
        hTrack_3Stubs_VtxZ0_SimTrack_VtxZ0->Fill( simTrackVtxZ0, trackVtxZ0);
        hTrack_3Stubs_VtxZ0Res_SimTrack_Eta->Fill( simTrackEta, trackVtxZ0 - simTrackVtxZ0 );
        hTrack_3Stubs_Chi2_NStubs->Fill( nStubs, trackChi2 );
        hTrack_3Stubs_Chi2_SimTrack_Eta->Fill( simTrackEta, trackChi2 );
        hTrack_3Stubs_Chi2Red_NStubs->Fill( nStubs, trackChi2R );
        hTrack_3Stubs_Chi2Red_SimTrack_Eta->Fill( simTrackEta, trackChi2R );
      }
      else
      {
        hTrack_2Stubs_Pt->Fill( trackPt );
        hTrack_2Stubs_Eta->Fill( trackEta );
        hTrack_2Stubs_Phi->Fill( trackPhi );

        if ( mapSimTrackTrack2Pt.find( simTrackPtr->trackId() ) == mapSimTrackTrack2Pt.end() )
        {
          /// New map entry
          mapSimTrackTrack2Pt.insert( std::make_pair( simTrackPtr->trackId(), simTrackPt ) );
          mapSimTrackTrack2Phi.insert( std::make_pair( simTrackPtr->trackId(), simTrackPhi ) );
          mapSimTrackTrack2Eta.insert( std::make_pair( simTrackPtr->trackId(), simTrackEta ) );
        }

        num2Stubs++;
        hTrack_2Stubs_Pt_SimTrack_Pt->Fill( simTrackPt, trackPt );
        hTrack_2Stubs_PtRes_SimTrack_Eta->Fill( simTrackEta, trackPt - simTrackPt);
        hTrack_2Stubs_InvPt_SimTrack_InvPt->Fill( 1./simTrackPt, 1./trackPt );
        hTrack_2Stubs_InvPtRes_SimTrack_Eta->Fill( simTrackEta, 1./trackPt - 1./simTrackPt);
        hTrack_2Stubs_Phi_SimTrack_Phi->Fill( simTrackPhi, trackPhi );
        hTrack_2Stubs_PhiRes_SimTrack_Eta->Fill( simTrackEta, trackPhi - simTrackPhi);
        hTrack_2Stubs_Eta_SimTrack_Eta->Fill( simTrackEta, trackEta );
        hTrack_2Stubs_EtaRes_SimTrack_Eta->Fill( simTrackEta, trackEta - simTrackEta);
        hTrack_2Stubs_VtxZ0_SimTrack_VtxZ0->Fill( simTrackVtxZ0, trackVtxZ0);
        hTrack_2Stubs_VtxZ0Res_SimTrack_Eta->Fill( simTrackEta, trackVtxZ0 - simTrackVtxZ0 );
        hTrack_2Stubs_Chi2_NStubs->Fill( nStubs, trackChi2 );
        hTrack_2Stubs_Chi2_SimTrack_Eta->Fill( simTrackEta, trackChi2 );
        hTrack_2Stubs_Chi2Red_NStubs->Fill( nStubs, trackChi2R );
        hTrack_2Stubs_Chi2Red_SimTrack_Eta->Fill( simTrackEta, trackChi2R );      
      }

      /// Go on only if there are L1TkTracks from PixelDigis
      if ( PixelDigiL1TkSeedHandle->size() > 0 )
      {
        /// Loop over L1TkTrack seeds
        L1TkTrack_PixelDigi_Collection::const_iterator iterSeed;
        for ( iterSeed = PixelDigiL1TkSeedHandle->begin();
              iterSeed != PixelDigiL1TkSeedHandle->end();
              ++iterSeed )
        {
          /// Check the track is the same
          bool dontSkip = iterL1TkTrack->isTheSameAs( *iterSeed );
          if ( !dontSkip ) continue;

          /// Get everything is relevant
          double seedRInv  = iterSeed->getRInv();
          double seedPt    = iterSeed->getMomentum().perp();
          double seedPhi   = iterSeed->getMomentum().phi();
          double seedEta   = iterSeed->getMomentum().eta();
          double seedVtxZ0 = iterSeed->getVertex().z();

          hTrack_RInv_Seed_RInv->Fill( seedRInv, trackRInv );
          hTrack_RInvRes_Track_Eta->Fill( trackEta, trackRInv - seedRInv );
          hTrack_Pt_Seed_Pt->Fill( seedPt, trackPt );
          hTrack_PtRes_Track_Eta->Fill( trackEta, trackPt - seedPt );
          hTrack_InvPt_Seed_InvPt->Fill( 1./seedPt, 1./trackPt );
          hTrack_InvPtRes_Track_Eta->Fill( trackEta, 1./trackPt - 1./seedPt );
          hTrack_Phi_Seed_Phi->Fill( seedPhi, trackPhi );
          hTrack_PhiRes_Track_Eta->Fill( trackEta, trackPhi - seedPhi );
          hTrack_Eta_Seed_Eta->Fill( seedEta, trackEta );
          hTrack_EtaRes_Track_Eta->Fill( trackEta, trackEta - seedEta );
          hTrack_VtxZ0_Seed_VtxZ0->Fill( seedVtxZ0, trackVtxZ0 );
          hTrack_VtxZ0Res_Track_Eta->Fill( trackEta, trackVtxZ0 - seedVtxZ0 );

          /// Propagate seed and check distances
          StackedTrackerDetId detIdInner( iterSeed->getStubPtrs().at(0)->getDetId() );
//          unsigned int seedSL = (unsigned int)((detIdInner.iLayer() + 1)/2);
//          seedSL = ( seedSL > 3 ) ? 3 : seedSL; /// Renormalize 1-3

          unsigned int seedBarrel0 = 0;
          unsigned int seedEndcap0 = 0;
          if ( detIdInner.isBarrel() ) seedBarrel0 = detIdInner.iLayer();
          else if ( detIdInner.isEndcap() ) seedEndcap0 = detIdInner.iDisk();

          /// Loop over track stubs
          for ( unsigned int js = 0; js < iterL1TkTrack->getStubPtrs().size(); js++ )
          {
            /// Skip Stubs in the Seed
            bool isInSeed = false;
            for ( unsigned int ks = 0; ks < iterSeed->getStubPtrs().size(); ks++ )
            {
              if ( iterL1TkTrack->getStubPtrs().at(js) == iterSeed->getStubPtrs().at(ks) )
                isInSeed = true;
            }
            if ( isInSeed ) continue;

            /// Candidate SL
            StackedTrackerDetId detIdCand( iterL1TkTrack->getStubPtrs().at(js)->getDetId() );
//            unsigned int candSL = (unsigned int)((detIdCand.iLayer() + 1)/2);
//            candSL = ( candSL > 3 ) ? 3 : candSL; /// Renormalize 1-3

            unsigned int candBarrel = 0;
            unsigned int candEndcap = 0;
            if ( detIdCand.isBarrel() ) candBarrel = detIdCand.iLayer();
            else if ( detIdCand.isEndcap() ) candEndcap = detIdCand.iDisk();

//            if ( candSL == seedSL ) continue;

            GlobalPoint posStub = theStackedGeometry->findGlobalPosition( &(*iterL1TkTrack->getStubPtrs().at(js)) );

            if ( candBarrel )
            {
              /// Propagation
              double propPsi = asin( posStub.perp() * 0.5 * seedRInv );
              double propPhi = seedPhi - propPsi;
              double propRhoPsi = 2 * propPsi / seedRInv;
              double propZ = seedVtxZ0 + propRhoPsi * tan( M_PI_2 - iterSeed->getMomentum().theta() );

              /// Calculate displacement
              /// Perform standard trigonometric operations
              double deltaPhi = posStub.phi() - propPhi;
              if ( fabs(deltaPhi) >= M_PI )
              {
                if ( deltaPhi>0 )
                  deltaPhi = deltaPhi - 2*M_PI;
                else
                  deltaPhi = 2*M_PI + deltaPhi;
              }
              double deltaRPhi = deltaPhi * posStub.perp();
              double deltaZ = propZ - posStub.z();

              if ( seedBarrel0 )
              {
                std::pair< unsigned int, unsigned int > mapKey0 = std::make_pair( seedBarrel0, candBarrel );
                mapTrackPropBB_deltaRhoPhi_Eta[ mapKey0 ]->Fill( seedEta, deltaRPhi );
                mapTrackPropBB_deltaRhoPhi[ mapKey0 ]->Fill( deltaRPhi );
                mapTrackPropBB_deltaZ_Eta[ mapKey0 ]->Fill( seedEta, deltaZ );
                mapTrackPropBB_deltaZ[ mapKey0 ]->Fill( deltaZ );
              }
              else if ( seedEndcap0 )
              {
                std::pair< unsigned int, unsigned int > mapKey0 = std::make_pair( seedEndcap0, candBarrel );
                mapTrackPropEB_deltaRhoPhi_Eta[ mapKey0 ]->Fill( seedEta, deltaRPhi );
                mapTrackPropEB_deltaRhoPhi[ mapKey0 ]->Fill( deltaRPhi );
                mapTrackPropEB_deltaZ_Eta[ mapKey0 ]->Fill( seedEta, deltaZ );
                mapTrackPropEB_deltaZ[ mapKey0 ]->Fill( deltaZ );
              }
            }
            else if ( candEndcap )
            {
              /// Propagation
              double propPsi = 0.5*( posStub.z() - seedVtxZ0 ) * seedRInv / tan( M_PI_2 - iterSeed->getMomentum().theta() );
              double propPhi = seedPhi - propPsi;
              double propRho = 2 * sin( propPsi ) / seedRInv;
              double deltaPhi = posStub.phi() - propPhi;

              /// Calculate displacement
              if ( fabs(deltaPhi) >= M_PI )
              {
                if ( deltaPhi>0 )
                  deltaPhi = deltaPhi - 2*M_PI;
                else
                  deltaPhi = 2*M_PI + deltaPhi;
              }
              double deltaRPhi = deltaPhi * posStub.perp(); /// OLD VERSION (updated few lines below)
              double deltaR = posStub.perp() - propRho;

              /// NEW VERSION - non-pointing strips correction
              double rhoTrack = 2.0 * sin( 0.5 * seedRInv * ( posStub.z() - seedVtxZ0 ) / tan( M_PI_2 - iterSeed->getMomentum().theta() ) ) / seedRInv;
              double phiTrack = iterSeed->getMomentum().phi() - 0.5 * seedRInv * ( posStub.z() - seedVtxZ0 ) / tan( M_PI_2 - iterSeed->getMomentum().theta() );

              /// Calculate a correction for non-pointing-strips in square modules
              /// Relevant angle is the one between hit and module center, with
              /// vertex at (0, 0). Take snippet from HitMatchingAlgorithm_window201*
              /// POSITION IN TERMS OF PITCH MULTIPLES:
              ///       0 1 2 3 4 5 5 6 8 9 ...
              /// COORD: 0 1 2 3 4 5 6 7 8 9 ...
              /// OUT   | | | | | |x| | | | | | | | | |
              ///
              /// IN    | | | |x|x| | | | | | | | | | |
              ///             THIS is 3.5 (COORD) and 4.0 (POS)
              /// The center of the module is at NROWS/2 (position) and NROWS-0.5 (coordinates)
              const GeomDetUnit* det0 = theStackedGeometry->idToDetUnit( detIdCand, 0 );
              const PixelGeomDetUnit* pix0 = dynamic_cast< const PixelGeomDetUnit* >( det0 );
              const PixelTopology* top0 = dynamic_cast< const PixelTopology* >( &(pix0->specificTopology()) );
              std::pair< float, float > pitch0 = top0->pitch();
              MeasurementPoint stubCoord = iterL1TkTrack->getStubPtrs().at(js)->getClusterPtr(0)->findAverageLocalCoordinates();
              double stubTransvDispl = pitch0.first * ( stubCoord.x() - (top0->nrows()/2 - 0.5) ); /// Difference in coordinates is the same as difference in position
              if ( posStub.z() > 0 )
              {
                stubTransvDispl = - stubTransvDispl;
              }
              double stubPhiCorr = asin( stubTransvDispl / posStub.perp() );
              deltaRPhi = stubTransvDispl - rhoTrack * sin( stubPhiCorr - phiTrack + posStub.phi() );

              if ( seedBarrel0 )
              {
                std::pair< unsigned int, unsigned int > mapKey0 = std::make_pair( seedBarrel0, candEndcap );
                mapTrackPropBE_deltaRhoPhi_Eta[ mapKey0 ]->Fill( seedEta, deltaRPhi );
                mapTrackPropBE_deltaRhoPhi[ mapKey0 ]->Fill( deltaRPhi );
                mapTrackPropBE_deltaR_Eta[ mapKey0 ]->Fill( seedEta, deltaR);
                mapTrackPropBE_deltaR[ mapKey0 ]->Fill( deltaR );
              }
              else if ( seedEndcap0 )
              {
                std::pair< unsigned int, unsigned int > mapKey0 = std::make_pair( seedEndcap0, candEndcap );
                mapTrackPropEE_deltaRhoPhi_Eta[ mapKey0 ]->Fill( seedEta, deltaRPhi );
                mapTrackPropEE_deltaRhoPhi[ mapKey0 ]->Fill( deltaRPhi );
                mapTrackPropEE_deltaR_Eta[ mapKey0 ]->Fill( seedEta, deltaR );
                mapTrackPropEE_deltaR[ mapKey0 ]->Fill( deltaR );
              }
            }

          } /// End of loop over track stubs
        } /// End of loop over L1TkTrack seeds
      }
    } /// End of loop over L1TkTracks
  }

  hTrack_2Stubs_N->Fill( num2Stubs );
  hTrack_3Stubs_N->Fill( num3Stubs );

  for ( iterMapSimTrack = mapSimTrackTrack2Pt.begin();
        iterMapSimTrack != mapSimTrackTrack2Pt.end();
        ++iterMapSimTrack )
  {
    hSimTrack_Track_2Stubs_Pt->Fill( iterMapSimTrack->second );
  }
  for ( iterMapSimTrack = mapSimTrackTrack2Phi.begin();
        iterMapSimTrack != mapSimTrackTrack2Phi.end();
        ++iterMapSimTrack )
  {
    if ( mapSimTrackTrack2Pt.find( iterMapSimTrack->first )->second > 5 )
      hSimTrack_Track_2Stubs_Phi_Pt5->Fill( iterMapSimTrack->second );
  }
  for ( iterMapSimTrack = mapSimTrackTrack2Eta.begin();
        iterMapSimTrack != mapSimTrackTrack2Eta.end();
        ++iterMapSimTrack )
  {
    if ( mapSimTrackTrack2Pt.find( iterMapSimTrack->first )->second > 5 )
      hSimTrack_Track_2Stubs_Eta_Pt5->Fill( iterMapSimTrack->second );
  }

  for ( iterMapSimTrack = mapSimTrackTrack3Pt.begin();
        iterMapSimTrack != mapSimTrackTrack3Pt.end();
        ++iterMapSimTrack )
  {
    hSimTrack_Track_3Stubs_Pt->Fill( iterMapSimTrack->second );
  }
  for ( iterMapSimTrack = mapSimTrackTrack3Phi.begin();
        iterMapSimTrack != mapSimTrackTrack3Phi.end();
        ++iterMapSimTrack )
  {
    if ( mapSimTrackTrack3Pt.find( iterMapSimTrack->first )->second > 5 )
      hSimTrack_Track_3Stubs_Phi_Pt5->Fill( iterMapSimTrack->second );
  }
  for ( iterMapSimTrack = mapSimTrackTrack3Eta.begin();
        iterMapSimTrack != mapSimTrackTrack3Eta.end();
        ++iterMapSimTrack )
  {
    if ( mapSimTrackTrack3Pt.find( iterMapSimTrack->first )->second > 5 )
      hSimTrack_Track_3Stubs_Eta_Pt5->Fill( iterMapSimTrack->second );
  }

  /// Operations needing reversed-nesting
  std::map< unsigned int, double > mapSimTrackSeedPt;
  std::map< unsigned int, double > mapSimTrackSeedPhi;
  std::map< unsigned int, double > mapSimTrackSeedEta;
  unsigned int numSeeds = 0;


  if ( PixelDigiL1TkSeedHandle->size() > 0 )
  {

    L1TkTrack_PixelDigi_Collection::const_iterator iterSeed;
    for ( iterSeed = PixelDigiL1TkSeedHandle->begin();
          iterSeed != PixelDigiL1TkSeedHandle->end();
          ++iterSeed )
    {
      bool genuineSeed = iterSeed->isGenuine();

      if ( !genuineSeed ) continue;
      
      double seedPt    = iterSeed->getMomentum().perp();
      double seedPhi   = iterSeed->getMomentum().phi();
      double seedEta   = iterSeed->getMomentum().eta();
      double seedVtxZ0 = iterSeed->getVertex().z();
      double seedChi2  = iterSeed->getChi2();
      double seedChi2R = iterSeed->getChi2Red();

      hSeed_Pt->Fill( seedPt );
      hSeed_Phi->Fill( seedPhi );
      hSeed_Eta->Fill( seedEta );

      edm::Ptr< SimTrack > simTrackPtr = iterSeed->getSimTrackPtr();

      /// Get the corresponding vertex and reject the track
      /// if its vertex is outside the beampipe
      int vertexIndex = simTrackPtr->vertIndex();
      const SimVertex& theSimVertex = (*SimVtxHandle)[vertexIndex];
      math::XYZTLorentzVectorD trkVtxPos = theSimVertex.position();

      if ( trkVtxPos.rho() >= 2 )
        continue;

      double simTrackPt = simTrackPtr->momentum().pt();
      double simTrackEta = simTrackPtr->momentum().eta();
      double simTrackPhi = simTrackPtr->momentum().phi();
      double simTrackVtxZ0 = ((*SimVtxHandle)[simTrackPtr->vertIndex()]).position().z();

      if ( mapSimTrackSeedPt.find( simTrackPtr->trackId() ) == mapSimTrackSeedPt.end() )
      {
        /// New map entry
        mapSimTrackSeedPt.insert( std::make_pair( simTrackPtr->trackId(), simTrackPt ) );
        mapSimTrackSeedPhi.insert( std::make_pair( simTrackPtr->trackId(), simTrackPhi ) );
        mapSimTrackSeedEta.insert( std::make_pair( simTrackPtr->trackId(), simTrackEta ) );
      }

      numSeeds++;
      hSeed_Pt_SimTrack_Pt->Fill( simTrackPt, seedPt );
      hSeed_PtRes_SimTrack_Eta->Fill( simTrackEta, seedPt - simTrackPt);
      hSeed_InvPt_SimTrack_InvPt->Fill( 1./simTrackPt, 1./seedPt );
      hSeed_InvPtRes_SimTrack_Eta->Fill( simTrackEta, 1./seedPt - 1./simTrackPt);
      hSeed_Phi_SimTrack_Phi->Fill( simTrackPhi, seedPhi );
      hSeed_PhiRes_SimTrack_Eta->Fill( simTrackEta, seedPhi - simTrackPhi);
      hSeed_Eta_SimTrack_Eta->Fill( simTrackEta, seedEta );
      hSeed_EtaRes_SimTrack_Eta->Fill( simTrackEta, seedEta - simTrackEta);
      hSeed_VtxZ0_SimTrack_VtxZ0->Fill( simTrackVtxZ0, seedVtxZ0);
      hSeed_VtxZ0Res_SimTrack_Eta->Fill( simTrackEta, seedVtxZ0 - simTrackVtxZ0 );
      hSeed_Chi2_NStubs->Fill( iterSeed->getStubPtrs().size(), seedChi2 );
      hSeed_Chi2_SimTrack_Eta->Fill( simTrackEta, seedChi2 );
      hSeed_Chi2Red_NStubs->Fill( iterSeed->getStubPtrs().size(), seedChi2R );
      hSeed_Chi2Red_SimTrack_Eta->Fill( simTrackEta, seedChi2R );

      unsigned int q = 0;

      if ( PixelDigiL1TkTrackHandle->size() > 0 )
      {
        L1TkTrack_PixelDigi_Collection::const_iterator iterL1TkTrack;
        for ( iterL1TkTrack = PixelDigiL1TkTrackHandle->begin();
              iterL1TkTrack != PixelDigiL1TkTrackHandle->end();
              ++iterL1TkTrack )
        {
          unsigned int nStubs = iterL1TkTrack->getStubPtrs().size();
          if ( nStubs < 3 ) continue;

          bool dontSkip = iterL1TkTrack->isTheSameAs( *iterSeed );
          if ( !dontSkip ) continue;

          q++;
          hTrack_Seed_Pt->Fill( seedPt );
          hTrack_Seed_Phi->Fill( seedPhi );
          hTrack_Seed_Eta->Fill( seedEta );
        }
      }

      if ( q > 1 ) std::cerr << "q is " << q << std::endl;
    }
  }

  hSeed_N->Fill( numSeeds );

  for ( iterMapSimTrack = mapSimTrackSeedPt.begin();
        iterMapSimTrack != mapSimTrackSeedPt.end();
        ++iterMapSimTrack )
  {
    hSimTrack_Seed_Pt->Fill( iterMapSimTrack->second );
  }
  for ( iterMapSimTrack = mapSimTrackSeedPhi.begin();
        iterMapSimTrack != mapSimTrackSeedPhi.end();
        ++iterMapSimTrack )
  {
    if ( mapSimTrackSeedPt.find( iterMapSimTrack->first )->second > 5 )
      hSimTrack_Seed_Phi_Pt5->Fill( iterMapSimTrack->second );
  }
  for ( iterMapSimTrack = mapSimTrackSeedEta.begin();
        iterMapSimTrack != mapSimTrackSeedEta.end();
        ++iterMapSimTrack )
  {
    if ( mapSimTrackSeedPt.find( iterMapSimTrack->first )->second > 5 )
      hSimTrack_Seed_Eta_Pt5->Fill( iterMapSimTrack->second );
  }

} /// End of analyze()

///////////////////////////
// DEFINE THIS AS A PLUG-IN
DEFINE_FWK_MODULE(ValidateL1Track);

