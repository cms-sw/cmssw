#include "Validation/MuonHits/src/MuonSimHitsValidAnalyzer.h"

#include "TFile.h"
#include "TTree.h"
#include "TBranch.h"
#include "TH1F.h"

#include <iostream>
#include <string>

using namespace edm;
using namespace std;


MuonSimHitsValidAnalyzer::MuonSimHitsValidAnalyzer(const edm::ParameterSet& iPSet) :
  fName(""), verbosity(0), label(""), getAllProvenances(false),
  printProvenanceInfo(false), nRawGenPart(0), count(0)

{
  /// get information from parameter set
  fName = iPSet.getUntrackedParameter<std::string>("Name");
  verbosity = iPSet.getUntrackedParameter<int>("Verbosity");
  label = iPSet.getParameter<std::string>("Label");
  edm::ParameterSet m_Prov =
    iPSet.getParameter<edm::ParameterSet>("ProvenanceLookup");
  getAllProvenances =
    m_Prov.getUntrackedParameter<bool>("GetAllProvenances");
  printProvenanceInfo =
    m_Prov.getUntrackedParameter<bool>("PrintProvenanceInfo");

   nRawGenPart = 0;
 // ROOT Histos output files
   DToutputFile_ =  iPSet.getUntrackedParameter<std::string>("DT_outputFile", "");
   //  CSCoutputFile_ =  iPSet.getUntrackedParameter<std::string>("CSC_outputFile", "");
   //  RPCoutputFile_ =  iPSet.getUntrackedParameter<std::string>("RPC_outputFile", "");


   /// get labels for input tags
   CSCHitsToken_ = consumes<edm::PSimHitContainer>(
       iPSet.getParameter<edm::InputTag>("CSCHitsSrc"));
   DTHitsToken_  = consumes<edm::PSimHitContainer>(
       iPSet.getParameter<edm::InputTag>("DTHitsSrc"));
   RPCHitsToken_ = consumes<edm::PSimHitContainer>(
       iPSet.getParameter<edm::InputTag>("RPCHitsSrc"));

   /// print out Parameter Set information being used

   if (verbosity) {
     Labels l;
     labelsForToken(DTHitsToken_, l);
     edm::LogInfo ("MuonSimHitsValidAnalyzer::MuonSimHitsValidAnalyzer")
       << "\n===============================\n"
       << "Initialized as EDAnalyzer with parameter values:\n"
       << "    Name      = " << fName << "\n"
       << "    Verbosity = " << verbosity << "\n"
       << "    Label     = " << label << "\n"
       << "    GetProv   = " << getAllProvenances << "\n"
       << "    PrintProv = " << printProvenanceInfo << "\n"
       //    << "    CSCHitsSrc=  " <<CSCHitsSrc_.label()
       //    << ":" << CSCHitsSrc_.instance() << "\n"
       << "    DTHitsSrc =  " << l.module
       << ":" << l.productInstance << "\n"
       //     << "    RPCHitsSrc=  " <<RPCHitsSrc_.label()
       //     << ":" << RPCHitsSrc_.instance() << "\n"
       << "===============================\n";
   }

   // ----------------------
   // get hold of back-end interface DT
   dbeDT_ = 0;
   dbeDT_ = Service<DQMStore>().operator->();
   if ( dbeDT_ ) {
     if ( verbosity ) {
       dbeDT_->setVerbose(1);
     } else {
       dbeDT_->setVerbose(0);
     }
   }
   if ( dbeDT_ ) {
     if ( verbosity ) dbeDT_->showDirStructure();
   }

   // ----------------------

   bookHistos_DT();

   /*
   // get hold of back-end interface CSC
   dbeCSC_ = 0;
   dbeCSC_ = Service<DQMStore>().operator->();
   if ( dbeCSC_ ) {
   if ( verbosity ) {
   dbeCSC_->setVerbose(1);
   } else {
   dbeCSC_->setVerbose(0);
   }
   }
   if ( dbeCSC_ ) {
   if ( verbosity ) dbeCSC_->showDirStructure();
   }

   // ----------------------

   bookHistos_CSC();

   // get hold of back-end interface RPC
   dbeRPC_ = 0;
   dbeRPC_ = Service<DQMStore>().operator->();
   if ( dbeRPC_ ) {
   if ( verbosity ) {
   dbeRPC_->setVerbose(1);
   } else {
   dbeRPC_->setVerbose(0);
   }
   }
   if ( dbeRPC_ ) {
   if ( verbosity ) dbeRPC_->showDirStructure();
   }

   // ----------------------

   bookHistos_RPC();
   */

   pow6=1000000.0;
   mom4 =0.;
   mom1 = 0;
   costeta = 0.;
   radius = 0;
   sinteta = 0.;
   globposx = 0.;
   globposy = 0;
   nummu_DT = 0;
   nummu_CSC =0;
   nummu_RPC=0;

}

MuonSimHitsValidAnalyzer::~MuonSimHitsValidAnalyzer()
{
 if ( DToutputFile_.size() != 0 )
   {
    LogInfo("OutputInfo") << " DT MuonHits histos file is closed " ;
    theDTFile->Close();
   }

// theCSCFile->Close();
// theRPCFile->Close();
}

void MuonSimHitsValidAnalyzer::beginJob()
{
  return;
}

void MuonSimHitsValidAnalyzer::bookHistos_DT()
{
  meAllDTHits =0 ;
  meMuDTHits =0 ;
  meToF =0 ;
  meEnergyLoss =0 ;
  meMomentumMB1 =0 ;
  meMomentumMB4 =0 ;
  meLossMomIron =0 ;
  meLocalXvsZ =0 ;
  meLocalXvsY =0 ;
  meGlobalXvsZ =0 ;
  meGlobalXvsY =0 ;
  meGlobalXvsZWm2 =0 ;
  meGlobalXvsZWm1 =0 ;
  meGlobalXvsZW0 =0 ;
  meGlobalXvsZWp1 =0 ;
  meGlobalXvsZWp2 =0 ;
  meGlobalXvsYWm2 =0 ;
  meGlobalXvsYWm1 =0 ;
  meGlobalXvsYW0 =0 ;
  meGlobalXvsYWp1 =0 ;
  meGlobalXvsYWp2 =0 ;
  meWheelOccup =0 ;
  meStationOccup =0 ;
  meSectorOccup =0 ;
  meSuperLOccup =0 ;
  meLayerOccup =0 ;
  meWireOccup =0 ;
  mePathMuon =0 ;
  meChamberOccup =0 ;
  meHitRadius =0 ;
  meCosTheta =0 ;
  meGlobalEta =0 ;
  meGlobalPhi =0 ;

  if ( DToutputFile_.size() != 0 ) {
   theDTFile = new TFile(DToutputFile_.c_str(),"RECREATE");
   theDTFile->cd();
   LogInfo("OutputInfo") << " DT MuonHits histograms will be saved to '" << DToutputFile_.c_str() << "'";
  } else {
   LogInfo("OutputInfo") << " DT MuonHits histograms will NOT be saved";
  }


  Char_t histo_n[100];
  Char_t histo_t[100];

  if ( dbeDT_ ) {
    dbeDT_->setCurrentFolder("MuonDTHitsV/DTHitsValidationTask");

    sprintf (histo_n, "Number_of_all_DT_hits" );
    sprintf (histo_t, "Number_of_all_DT_hits" );
    meAllDTHits = dbeDT_->book1D(histo_n, histo_t,  200, 1.0, 201.0) ;

    sprintf (histo_n, "Number_of_muon_DT_hits" );
    sprintf (histo_t, "Number_of_muon_DT_hits" );
    meMuDTHits  = dbeDT_->book1D(histo_n, histo_t, 150, 1.0, 151.0);

    sprintf (histo_n, "Tof_of_hits " );
    sprintf (histo_t, "Tof_of_hits " );
    meToF = dbeDT_->book1D(histo_n, histo_t, 100, -0.5, 50.) ;

    sprintf (histo_n, "DT_energy_loss_keV" );
    sprintf (histo_t, "DT_energy_loss_keV" );
    meEnergyLoss  = dbeDT_->book1D(histo_n, histo_t, 100, 0.0, 10.0);

    sprintf (histo_n, "Momentum_at_MB1" );
    sprintf (histo_t, "Momentum_at_MB1" );
    meMomentumMB1 = dbeDT_->book1D(histo_n, histo_t, 100, 10.0, 200.0);

    sprintf (histo_n, "Momentum_at_MB4" );
    sprintf (histo_t, "Momentum_at_MB4" );
    meMomentumMB4 = dbeDT_->book1D(histo_n, histo_t, 100, 10.0, 200.0) ;

    sprintf (histo_n, "Loss_of_muon_Momentum_in_Iron" );
    sprintf (histo_t, "Loss_of_muon_Momentum_in_Iron" );
    meLossMomIron  = dbeDT_->book1D(histo_n, histo_t, 80, 0.0, 40.0) ;

    sprintf (histo_n, "Local_x-coord_vs_local_z-coord_of_muon_hit" );
    sprintf (histo_t, "Local_x-coord_vs_local_z-coord_of_muon_hit" );
    meLocalXvsZ = dbeDT_->book2D(histo_n, histo_t,100, -150., 150., 100, -0.8, 0.8 ) ;

    sprintf (histo_n, "local_x-coord_vs_local_y-coord_of_muon_hit" );
    sprintf (histo_t, "local_x-coord_vs_local_y-coord_of_muon_hit" );
    meLocalXvsY = dbeDT_->book2D(histo_n, histo_t, 100, -150., 150., 100, -150., 150. );

    sprintf (histo_n, "Global_x-coord_vs_global_z-coord_of_muon_hit" );
    sprintf (histo_t, "Global_x-coord_vs_global_z-coord_of_muon_hit" );
    meGlobalXvsZ = dbeDT_->book2D(histo_n, histo_t, 100, -800., 800., 100, -800., 800. ) ;

    sprintf (histo_n, "Global_x-coord_vs_global_y-coord_of_muon_hit" );
    sprintf (histo_t, "Global_x-coord_vs_global_y-coord_of_muon_hit" );
    meGlobalXvsY = dbeDT_->book2D(histo_n, histo_t, 100, -800., 800., 100, -800., 800. ) ;

//   New histos

    sprintf (histo_n, "Global_x-coord_vs_global_z-coord_of_muon_hit_w-2" );
    sprintf (histo_t, "Global_x-coord_vs_global_z-coord_of_muon_hit_w-2" );
    meGlobalXvsZWm2 = dbeDT_->book2D(histo_n, histo_t, 100, -800., 800., 100, -800., 800. );

    sprintf (histo_n, "Global_x-coord_vs_global_y-coord_of_muon_hit_w-2" );
    sprintf (histo_t, "Global_x-coord_vs_global_y-coord_of_muon_hit_w-2" );
    meGlobalXvsYWm2 = dbeDT_->book2D(histo_n, histo_t,  100, -800., 800., 100, -800., 800. );

    sprintf (histo_n, "Global_x-coord_vs_global_z-coord_of_muon_hit_w-1" );
    sprintf (histo_t, "Global_x-coord_vs_global_z-coord_of_muon_hit_w-1" );
    meGlobalXvsZWm1 = dbeDT_->book2D(histo_n, histo_t, 100, -800., 800., 100, -800., 800. );

    sprintf (histo_n, "Global_x-coord_vs_global_y-coord_of_muon_hit_w-1" );
    sprintf (histo_t, "Global_x-coord_vs_global_y-coord_of_muon_hit_w-1" );
    meGlobalXvsYWm1 = dbeDT_->book2D(histo_n, histo_t,  100, -800., 800., 100, -800., 800. );

    sprintf (histo_n, "Global_x-coord_vs_global_z-coord_of_muon_hit_w0" );
    sprintf (histo_t, "Global_x-coord_vs_global_z-coord_of_muon_hit_w0" );
    meGlobalXvsZW0 = dbeDT_->book2D(histo_n, histo_t, 100, -800., 800., 100, -800., 800. );

    sprintf (histo_n, "Global_x-coord_vs_global_y-coord_of_muon_hit_w0" );
    sprintf (histo_t, "Global_x-coord_vs_global_y-coord_of_muon_hit_w0" );
    meGlobalXvsYW0 = dbeDT_->book2D(histo_n, histo_t,  100, -800., 800., 100, -800., 800. );

    sprintf (histo_n, "Global_x-coord_vs_global_z-coord_of_muon_hit_w1" );
    sprintf (histo_t, "Global_x-coord_vs_global_z-coord_of_muon_hit_w1" );
    meGlobalXvsZWp1 = dbeDT_->book2D(histo_n, histo_t, 100, -800., 800., 100, -800., 800. );

    sprintf (histo_n, "Global_x-coord_vs_global_y-coord_of_muon_hit_w1" );
    sprintf (histo_t, "Global_x-coord_vs_global_y-coord_of_muon_hit_w1" );
    meGlobalXvsYWp1 = dbeDT_->book2D(histo_n, histo_t,  100, -800., 800., 100, -800., 800. );

    sprintf (histo_n, "Global_x-coord_vs_global_z-coord_of_muon_hit_w2" );
    sprintf (histo_t, "Global_x-coord_vs_global_z-coord_of_muon_hit_w2" );
    meGlobalXvsZWp2 = dbeDT_->book2D(histo_n, histo_t, 100, -800., 800., 100, -800., 800. );

    sprintf (histo_n, "Global_x-coord_vs_global_y-coord_of_muon_hit_w2" );
    sprintf (histo_t, "Global_x-coord_vs_global_y-coord_of_muon_hit_w2" );
    meGlobalXvsYWp2 = dbeDT_->book2D(histo_n, histo_t,  100, -800., 800., 100, -800., 800. );

//

    sprintf (histo_n, "Wheel_occupancy" );
    sprintf (histo_t, "Wheel_occupancy" );
    meWheelOccup = dbeDT_->book1D(histo_n, histo_t, 10, -5.0, 5.0) ;

    sprintf (histo_n, "Station_occupancy" );
    sprintf (histo_t, "Station_occupancy" );
    meStationOccup = dbeDT_->book1D(histo_n, histo_t, 6, 0., 6.0) ;

    sprintf (histo_n, "Sector_occupancy" );
    sprintf (histo_t, "Sector_occupancy" );
    meSectorOccup = dbeDT_->book1D(histo_n, histo_t, 20, 0., 20.) ;

    sprintf (histo_n, "SuperLayer_occupancy" );
    sprintf (histo_t, "SuperLayer_occupancy" );
    meSuperLOccup = dbeDT_->book1D(histo_n, histo_t, 5, 0., 5.) ;

    sprintf (histo_n, "Layer_occupancy" );
    sprintf (histo_t, "Layer_occupancy" );
    meLayerOccup = dbeDT_->book1D(histo_n, histo_t,6, 0., 6.) ;

    sprintf (histo_n, "Wire_occupancy" );
    sprintf (histo_t, "Wire_occupancy" );
    meWireOccup = dbeDT_->book1D(histo_n, histo_t, 100, 0., 100.) ;

    sprintf (histo_n, "path_followed_by_muon" );
    sprintf (histo_t, "path_followed_by_muon" );
    mePathMuon = dbeDT_->book1D(histo_n, histo_t, 160, 0., 160.) ;

    sprintf (histo_n, "chamber_occupancy" );
    sprintf (histo_t, "chamber_occupancy" );
    meChamberOccup = dbeDT_->book1D(histo_n, histo_t,  251, 0., 251.) ;

    sprintf (histo_n, "radius_of_hit");
    sprintf (histo_t, "radius_of_hit");
    meHitRadius = dbeDT_->book1D(histo_n, histo_t, 100, 0., 1200. );

    sprintf (histo_n, "costheta_of_hit" );
    sprintf (histo_t, "costheta_of_hit" );
    meCosTheta = dbeDT_->book1D(histo_n, histo_t,  100, -1., 1.) ;

    sprintf (histo_n, "global_eta_of_hit" );
    sprintf (histo_t, "global_eta_of_hit" );
    meGlobalEta = dbeDT_->book1D(histo_n, histo_t, 60, -2.7, 2.7 );

    sprintf (histo_n, "global_phi_of_hit" );
    sprintf (histo_t, "global_phi_of_hit" );
    meGlobalPhi = dbeDT_->book1D(histo_n, histo_t, 60, -3.14, 3.14);

  }

}

void MuonSimHitsValidAnalyzer::bookHistos_RPC()
{
  meAllRPCHits = 0 ;
  meMuRPCHits = 0 ;
  meRegionOccup = 0 ;
  meRingOccBar = 0 ;
  meRingOccEndc = 0 ;
  meStatOccBar = 0 ;
  meStatOccEndc = 0 ;
  meSectorOccBar = 0 ;
  meSectorOccEndc = 0 ;
  meLayerOccBar = 0 ;
  meLayerOccEndc = 0 ;
  meSubSectOccBar = 0 ;
  meSubSectOccEndc = 0 ;
  meRollOccBar = 0 ;
  meRollOccEndc = 0 ;
  meElossBar = 0 ;
  meElossEndc = 0 ;
  mepathRPC = 0 ;
  meMomRB1 = 0 ;
  meMomRB4 = 0 ;
  meLossMomBar = 0 ;
  meMomRE1 = 0 ;
  meMomRE4 = 0 ;
  meLossMomEndc = 0 ;
  meLocalXvsYBar = 0 ;
  meGlobalXvsZBar = 0 ;
  meGlobalXvsYBar = 0 ;
  meLocalXvsYEndc = 0 ;
  meGlobalXvsZEndc = 0 ;
  meGlobalXvsYEndc = 0 ;
  meHitRadiusBar = 0 ;
  meCosThetaBar = 0 ;
  meHitRadiusEndc = 0 ;
  meCosThetaEndc = 0 ;

  theRPCFile = new TFile(RPCoutputFile_.c_str(),"RECREATE");
  theRPCFile->cd();

  Char_t histo_n[100];
  Char_t histo_t[100];

   if ( dbeRPC_ ) {
    dbeRPC_->setCurrentFolder("MuonRPCHitsV/RPCHitsValidationTask");

    sprintf (histo_n, "Number_of_all_RPC_hits" );
    sprintf (histo_t, "Number_of_all_RPC_hits" );
    meAllRPCHits = dbeRPC_->book1D(histo_n, histo_t,  100, 1.0, 101.0) ;

    sprintf (histo_n, "Number_of_muon_RPC_hits" );
    sprintf (histo_t, "Number_of_muon_RPC_hits" );
    meMuRPCHits = dbeRPC_->book1D(histo_n, histo_t,  50, 1., 51.);

    sprintf (histo_n, "Region_occupancy");
    sprintf (histo_t, "Region_occupancy");
    meRegionOccup  = dbeRPC_->book1D(histo_n, histo_t, 6, -3.0, 3.0) ;

    sprintf (histo_n, "Ring_occupancy_barrel");
    sprintf (histo_t, "Ring_occupancy_barrel");
    meRingOccBar = dbeRPC_->book1D(histo_n, histo_t, 8, -3., 5.0) ;

    sprintf (histo_n, "Ring_occupancy_endcaps");
    sprintf (histo_t, "Ring_occupancy_endcaps");
    meRingOccEndc = dbeRPC_->book1D(histo_n, histo_t, 8, -3., 5.0) ;

    sprintf (histo_n, "Station_occupancy_barrel");
    sprintf (histo_t, "Station_occupancy_barrel");
    meStatOccBar = dbeRPC_->book1D(histo_n, histo_t, 8, 0., 8.);

    sprintf (histo_n, "Station_occupancy_endcaps" );
    sprintf (histo_t, "Station_occupancy_endcaps" );
    meStatOccEndc = dbeRPC_->book1D(histo_n, histo_t, 8, 0., 8.);

    sprintf (histo_n, "Sector_occupancy_barrel" );
    sprintf (histo_t, "Sector_occupancy_barrel" );
    meSectorOccBar = dbeRPC_->book1D(histo_n, histo_t, 16, 0., 16.) ;

    sprintf (histo_n, "Sector_occupancy_endcaps" );
    sprintf (histo_t, "Sector_occupancy_endcaps" );
    meSectorOccEndc = dbeRPC_->book1D(histo_n, histo_t, 16, 0., 16.) ;

    sprintf (histo_n, "Layer_occupancy_barrel" );
    sprintf (histo_t, "Layer_occupancy_barrel" );
    meLayerOccBar = dbeRPC_->book1D(histo_n, histo_t,4, 0., 4.) ;

    sprintf (histo_n, "Layer_occupancy_endcaps" );
    sprintf (histo_t, "Layer_occupancy_endcaps" );
    meLayerOccEndc = dbeRPC_->book1D(histo_n, histo_t,4, 0., 4.) ;

    sprintf (histo_n, "Subsector_occupancy_barrel" );
    sprintf (histo_t, "Subsector_occupancy_barrel" );
    meSubSectOccBar = dbeRPC_->book1D(histo_n, histo_t, 10, 0., 10.) ;

    sprintf (histo_n, "Subsector_occupancy_endcaps" );
    sprintf (histo_t, "Subsector_occupancy_endcaps" );
    meSubSectOccEndc = dbeRPC_->book1D(histo_n, histo_t, 10, 0., 10.) ;

    sprintf (histo_n, "Roll_occupancy_barrel" );
    sprintf (histo_t, "Roll_occupancy_barrel" );
    meRollOccBar = dbeRPC_->book1D(histo_n, histo_t,  6, 0., 6.) ;

    sprintf (histo_n, "Roll_occupancy_endcaps" );
    sprintf (histo_t, "Roll_occupancy_endcaps" );
    meRollOccEndc = dbeRPC_->book1D(histo_n, histo_t,  6, 0., 6.) ;

    sprintf (histo_n, "RPC_energy_loss_barrel" );
    sprintf (histo_t, "RPC_energy_loss_barrel" );
    meElossBar = dbeRPC_->book1D(histo_n, histo_t, 50, 0.0, 10.0) ;

    sprintf (histo_n, "RPC_energy_loss_endcaps" );
    sprintf (histo_t, "RPC_energy_loss_endcaps" );
    meElossEndc = dbeRPC_->book1D(histo_n, histo_t, 50, 0.0, 10.0) ;

    sprintf (histo_n, "path_followed_by_muon" );
    sprintf (histo_t, "path_followed_by_muon" );
    mepathRPC = dbeRPC_->book1D(histo_n, histo_t, 160, 0., 160.) ;

    sprintf (histo_n, "Momentum_at_RB1") ;
    sprintf (histo_t, "Momentum_at_RB1") ;
    meMomRB1 = dbeRPC_->book1D(histo_n, histo_t, 80, 10.0, 200.0) ;

    sprintf (histo_n, "Momentum_at_RB4") ;
    sprintf (histo_t, "Momentum_at_RB4") ;
    meMomRB4 = dbeRPC_->book1D(histo_n, histo_t, 80, 10.0, 200.0) ;

    sprintf (histo_n, "Loss_of_muon_Momentum_in_Iron_barrel" );
    sprintf (histo_t, "Loss_of_muon_Momentum_in_Iron_barrel" );
    meLossMomBar = dbeRPC_->book1D(histo_n, histo_t,  80, 0.0, 40.0) ;

    sprintf (histo_n, "Momentum_at_RE1");
    sprintf (histo_t, "Momentum_at_RE1");
    meMomRE1 = dbeRPC_->book1D(histo_n, histo_t,  100, 10.0, 300.0);

    sprintf (histo_n, "Momentum_at_RE4");
    sprintf (histo_t, "Momentum_at_RE4");
    meMomRE4 = dbeRPC_->book1D(histo_n, histo_t,  100, 10.0, 300.0);

    sprintf (histo_n, "Loss_of_muon_Momentum_in_Iron_endcap" );
    sprintf (histo_t, "Loss_of_muon_Momentum_in_Iron_endcap" );
    meLossMomEndc = dbeRPC_->book1D(histo_n, histo_t, 80, 0.0, 40.0) ;

    sprintf (histo_n, "local_x-coord_vs_local_y-coord_of_muon_hit") ;
    sprintf (histo_t, "local_x-coord_vs_local_y-coord_of_muon_hit") ;
    meLocalXvsYBar = dbeRPC_->book2D(histo_n, histo_t, 100, -150., 150., 100, -100., 100. );

    sprintf (histo_n, "Global_z-coord_vs_global_x-coord_of_muon_hit_barrel" );
    sprintf (histo_t, "Global_z-coord_vs_global_x-coord_of_muon_hit_barrel" );
    meGlobalXvsZBar = dbeRPC_->book2D(histo_n, histo_t, 100, -800., 800., 100, -800., 800. );

    sprintf (histo_n, "Global_x-coord_vs_global_y-coord_of_muon_hit_barrel" );
    sprintf (histo_t, "Global_x-coord_vs_global_y-coord_of_muon_hit_barrel" );
    meGlobalXvsYBar = dbeRPC_->book2D(histo_n, histo_t, 100, -800., 800., 100, -800., 800. );

    sprintf (histo_n, "radius_of_hit_barrel" );
    sprintf (histo_t, "radius_of_hit_barrel" );
    meHitRadiusBar = dbeRPC_->book1D(histo_n, histo_t, 100, 0., 1200.) ;

    sprintf (histo_n, "radius_of_hit_endcaps" );
    sprintf (histo_t, "radius_of_hit_endcaps" );
    meHitRadiusEndc = dbeRPC_->book1D(histo_n, histo_t, 100, 0., 1300.) ;

    sprintf (histo_n, "costheta_of_hit_barrel" ) ;
    sprintf (histo_t, "costheta_of_hit_barrel" ) ;
    meCosThetaBar = dbeRPC_->book1D(histo_n, histo_t,  100, -1., 1.);

    sprintf (histo_n, "costheta_of_hit_endcaps" );
    sprintf (histo_t, "costheta_of_hit_endcaps" );
    meCosThetaEndc = dbeRPC_->book1D(histo_n, histo_t,  100, -1., 1.);

    sprintf (histo_n, "Global_z-coord_vs_global_x-coord_of_muon_hit_endcaps" );
    sprintf (histo_t, "Global_z-coord_vs_global_x-coord_of_muon_hit_endcaps" );
    meGlobalXvsZEndc = dbeRPC_->book2D(histo_n, histo_t,  100, -1200., 1200., 100, -800., 800. ) ;

    sprintf (histo_n, "Global_x-coord_vs_global_y-coord_of_muon_hit_endcaps" );
    sprintf (histo_t, "Global_x-coord_vs_global_y-coord_of_muon_hit_endcaps" );
    meGlobalXvsYEndc = dbeRPC_->book2D(histo_n, histo_t, 100, -800., 800., 100, -800., 800. );

  }

}

void MuonSimHitsValidAnalyzer::bookHistos_CSC()
{
  meAllCSCHits =0 ;
  meMuCSCHits =0 ;
  meEnergyLoss_111 =0 ;
  meToF_311 =0 ;
  meEnergyLoss_112 =0 ;
  meToF_312 =0 ;
  meEnergyLoss_113 =0 ;
  meToF_313 =0 ;
  meEnergyLoss_114 =0 ;
  meToF_314 =0 ;
  meEnergyLoss_121 =0 ;
  meToF_321 =0 ;
  meEnergyLoss_122 =0 ;
  meToF_322 =0 ;
  meEnergyLoss_131 =0 ;
  meToF_331 =0 ;
  meEnergyLoss_132 =0 ;
  meToF_332 =0 ;
  meEnergyLoss_141 =0 ;
  meToF_341 =0 ;
  meEnergyLoss_211 =0 ;
  meToF_411 =0 ;
  meEnergyLoss_212 =0 ;
  meToF_412 =0 ;
  meEnergyLoss_213 =0 ;
  meToF_413 =0 ;
  meEnergyLoss_214 =0 ;
  meToF_414 =0 ;
  meEnergyLoss_221 =0 ;
  meToF_421 =0 ;
  meEnergyLoss_222 =0 ;
  meToF_422 =0 ;
  meEnergyLoss_231 =0 ;
  meToF_431 =0 ;
  meEnergyLoss_232 =0 ;
  meToF_432 =0 ;
  meEnergyLoss_241 =0 ;
  meToF_441 =0 ;


   theCSCFile = new TFile(CSCoutputFile_.c_str(),"RECREATE");
   theCSCFile->cd();

   Char_t histo_n[100];
   Char_t histo_t[100];

   if ( dbeCSC_ ) {
    dbeCSC_->setCurrentFolder("MuonCSCHitsV/CSCHitsValidationTask");

    sprintf (histo_n, "Number_of_all_CSC_hits " );
    sprintf (histo_t, "Number_of_all_CSC_hits " );
    meAllCSCHits = dbeCSC_->book1D(histo_n, histo_t,  100, 1.0, 101.0) ;

    sprintf (histo_n, "Number_of_muon_CSC_hits" );
    sprintf (histo_t, "Number_of_muon_CSC_hits" );
    meMuCSCHits = dbeCSC_->book1D(histo_n, histo_t, 50, 1.0, 51.0);

    sprintf (histo_n, "111__energy_loss");
    sprintf (histo_t, "111__energy_loss");
    meEnergyLoss_111 = dbeCSC_->book1D(histo_n, histo_t,50, 0.0, 50.0) ;

    sprintf (histo_n, "311_tof");
    sprintf (histo_t, "311_tof");
    meToF_311 = dbeCSC_->book1D(histo_n, histo_t, 60, 0.0, 60.0) ;

    sprintf (histo_n, "112__energy_loss");
    sprintf (histo_t, "112__energy_loss");
    meEnergyLoss_112 = dbeCSC_->book1D(histo_n, histo_t,50, 0.0, 50.0) ;

    sprintf (histo_n, "312_tof");
    sprintf (histo_t, "312_tof");
    meToF_312 = dbeCSC_->book1D(histo_n, histo_t, 60, 0.0, 60.0) ;

    sprintf (histo_n, "113__energy_loss");
    sprintf (histo_t, "113__energy_loss");
    meEnergyLoss_111 = dbeCSC_->book1D(histo_n, histo_t,50, 0.0, 50.0) ;

    sprintf (histo_n, "313_tof");
    sprintf (histo_t, "313_tof");
    meToF_313 = dbeCSC_->book1D(histo_n, histo_t, 60, 0.0, 60.0) ;

    sprintf (histo_n, "114__energy_loss");
    sprintf (histo_t, "114__energy_loss");
    meEnergyLoss_114 = dbeCSC_->book1D(histo_n, histo_t,50, 0.0, 50.0) ;

    sprintf (histo_n, "314_tof");
    sprintf (histo_t, "314_tof");
    meToF_314 = dbeCSC_->book1D(histo_n, histo_t, 60, 0.0, 60.0) ;

    sprintf (histo_n, "121__energy_loss");
    sprintf (histo_t, "121__energy_loss");
    meEnergyLoss_121 = dbeCSC_->book1D(histo_n, histo_t,50, 0.0, 50.0) ;

    sprintf (histo_n, "321_tof");
    sprintf (histo_t, "321_tof");
    meToF_321 = dbeCSC_->book1D(histo_n, histo_t, 60, 0.0, 60.0) ;

    sprintf (histo_n, "122__energy_loss");
    sprintf (histo_t, "122__energy_loss");
    meEnergyLoss_122 = dbeCSC_->book1D(histo_n, histo_t,50, 0.0, 50.0) ;

    sprintf (histo_n, "322_tof");
    sprintf (histo_t, "322_tof");
    meToF_322 = dbeCSC_->book1D(histo_n, histo_t, 60, 0.0, 60.0) ;

    sprintf (histo_n, "131__energy_loss");
    sprintf (histo_t, "131__energy_loss");
    meEnergyLoss_131 = dbeCSC_->book1D(histo_n, histo_t,50, 0.0, 50.0) ;

    sprintf (histo_n, "331_tof");
    sprintf (histo_t, "331_tof");
    meToF_331 = dbeCSC_->book1D(histo_n, histo_t, 60, 0.0, 60.0) ;

    sprintf (histo_n, "132__energy_loss");
    sprintf (histo_t, "132__energy_loss");
    meEnergyLoss_132 = dbeCSC_->book1D(histo_n, histo_t,50, 0.0, 50.0) ;

    sprintf (histo_n, "332_tof");
    sprintf (histo_t, "332_tof");
    meToF_332 = dbeCSC_->book1D(histo_n, histo_t, 60, 0.0, 60.0) ;

    sprintf (histo_n, "141__energy_loss");
    sprintf (histo_t, "141__energy_loss");
    meEnergyLoss_141 = dbeCSC_->book1D(histo_n, histo_t,50, 0.0, 50.0) ;

    sprintf (histo_n, "341_tof");
    sprintf (histo_t, "341_tof");
    meToF_341 = dbeCSC_->book1D(histo_n, histo_t, 60, 0.0, 60.0) ;



    sprintf (histo_n, "211__energy_loss");
    sprintf (histo_t, "211__energy_loss");
    meEnergyLoss_211 = dbeCSC_->book1D(histo_n, histo_t,50, 0.0, 50.0) ;

    sprintf (histo_n, "411_tof");
    sprintf (histo_t, "411_tof");
    meToF_411 = dbeCSC_->book1D(histo_n, histo_t, 60, 0.0, 60.0) ;

    sprintf (histo_n, "212__energy_loss");
    sprintf (histo_t, "212__energy_loss");
    meEnergyLoss_212 = dbeCSC_->book1D(histo_n, histo_t,50, 0.0, 50.0) ;

    sprintf (histo_n, "412_tof");
    sprintf (histo_t, "412_tof");
    meToF_412 = dbeCSC_->book1D(histo_n, histo_t, 60, 0.0, 60.0) ;

    sprintf (histo_n, "213__energy_loss");
    sprintf (histo_t, "213__energy_loss");
    meEnergyLoss_211 = dbeCSC_->book1D(histo_n, histo_t,50, 0.0, 50.0) ;

    sprintf (histo_n, "413_tof");
    sprintf (histo_t, "413_tof");
    meToF_413 = dbeCSC_->book1D(histo_n, histo_t, 60, 0.0, 60.0) ;

    sprintf (histo_n, "214__energy_loss");
    sprintf (histo_t, "214__energy_loss");
    meEnergyLoss_214 = dbeCSC_->book1D(histo_n, histo_t,50, 0.0, 50.0) ;

    sprintf (histo_n, "414_tof");
    sprintf (histo_t, "414_tof");
    meToF_414 = dbeCSC_->book1D(histo_n, histo_t, 60, 0.0, 60.0) ;

    sprintf (histo_n, "221__energy_loss");
    sprintf (histo_t, "221__energy_loss");
    meEnergyLoss_221 = dbeCSC_->book1D(histo_n, histo_t,50, 0.0, 50.0) ;

    sprintf (histo_n, "421_tof");
    sprintf (histo_t, "421_tof");
    meToF_421 = dbeCSC_->book1D(histo_n, histo_t, 60, 0.0, 60.0) ;

    sprintf (histo_n, "222__energy_loss");
    sprintf (histo_t, "222__energy_loss");
    meEnergyLoss_222 = dbeCSC_->book1D(histo_n, histo_t,50, 0.0, 50.0) ;

    sprintf (histo_n, "422_tof");
    sprintf (histo_t, "422_tof");
    meToF_422 = dbeCSC_->book1D(histo_n, histo_t, 60, 0.0, 60.0) ;

    sprintf (histo_n, "231__energy_loss");
    sprintf (histo_t, "231__energy_loss");
    meEnergyLoss_231 = dbeCSC_->book1D(histo_n, histo_t,50, 0.0, 50.0) ;

    sprintf (histo_n, "431_tof");
    sprintf (histo_t, "431_tof");
    meToF_431 = dbeCSC_->book1D(histo_n, histo_t, 60, 0.0, 60.0) ;

    sprintf (histo_n, "232__energy_loss");
    sprintf (histo_t, "232__energy_loss");
    meEnergyLoss_232 = dbeCSC_->book1D(histo_n, histo_t,50, 0.0, 50.0) ;

    sprintf (histo_n, "432_tof");
    sprintf (histo_t, "432_tof");
    meToF_432 = dbeCSC_->book1D(histo_n, histo_t, 60, 0.0, 60.0) ;

    sprintf (histo_n, "241__energy_loss");
    sprintf (histo_t, "241__energy_loss");
    meEnergyLoss_241 = dbeCSC_->book1D(histo_n, histo_t,50, 0.0, 50.0) ;

    sprintf (histo_n, "441_tof");
    sprintf (histo_t, "441_tof");
    meToF_441 = dbeCSC_->book1D(histo_n, histo_t, 60, 0.0, 60.0) ;

   }

}

void MuonSimHitsValidAnalyzer::saveHistos_DT()
{
  //int DTHistos;
  //DTHistos = 1000;
  theDTFile->cd();

  if ( dbeDT_ ) {
    dbeDT_->setCurrentFolder("MuonDTHitsV/DTHitsValidationTask");
    //    cout << " DTFile.size " << DToutputFile_.size() << " dbeDT " << dbeDT_ << endl;
    dbeDT_->save(DToutputFile_);
  }

//  gDirectory->pwd();
//  theDTFile->ls();
// theDTFile->GetList()->ls();
//  hmgr->save(DTHistos);
}

void MuonSimHitsValidAnalyzer::saveHistos_RPC()
{
  //int RPCHistos;
  //RPCHistos = 3000;
  theRPCFile->cd();

  if ( dbeRPC_ ) {
    dbeRPC_->setCurrentFolder("MuonRPCHitsV/RPCHitsValidationTask");
    //    cout << " RPCFile.size " << RPCoutputFile_.size() << " dbeRPC " << dbeRPC_ << endl;
    dbeRPC_->save(RPCoutputFile_);
  }




//  gDirectory->pwd();
//  theRPCFile->ls();
//  theRPCFile->GetList()->ls();
//  hmgr->save(RPCHistos);
}

void MuonSimHitsValidAnalyzer::saveHistos_CSC()
{
  //int CSCHistos;
  //CSCHistos = 2000;
  theCSCFile->cd();

  if ( dbeCSC_ ) {
    dbeCSC_->setCurrentFolder("MuonCSCHitsV/CSCHitsValidationTask");
    //    cout << " CSCFile.size " << CSCoutputFile_.size() << " dbeCSC " << dbeCSC_ << endl;
    dbeCSC_->save(CSCoutputFile_);
  }



//  gDirectory->pwd();
//  theCSCFile->ls();
//  theCSCFile->GetList()->ls();
//     hmgr->save(CSCHistos);
}

void MuonSimHitsValidAnalyzer::endJob()
{

 if ( DToutputFile_.size() != 0 ) {
  saveHistos_DT();
  LogInfo("OutputInfo") << " DT MuonHits histos already saved" ;
 } else {
    LogInfo("OutputInfo") << " DT MuonHits histos NOT saved";
  }



//  saveHistos_CSC();
//  saveHistos_RPC();
  if (verbosity > 0)
    edm::LogInfo ("MuonSimHitsValidAnalyzer::endJob")
      << "Terminating having processed " << count << " events.";
  return;

}

void MuonSimHitsValidAnalyzer::analyze(const edm::Event& iEvent,
			       const edm::EventSetup& iSetup)
{
  /// keep track of number of events processed
  ++count;

  /// get event id information
  int nrun = iEvent.id().run();
  int nevt = iEvent.id().event();

  if (verbosity > 0) {
    edm::LogInfo ("MuonSimHitsValidAnalyzer::analyze")
      << "Processing run " << nrun << ", event " << nevt;
  }

  /// look at information available in the event
  if (getAllProvenances) {

    std::vector<const edm::Provenance*> AllProv;
    iEvent.getAllProvenance(AllProv);

    if (verbosity > 0)
      edm::LogInfo ("MuonSimHitsValidAnalyzer::analyze")
	<< "Number of Provenances = " << AllProv.size();

    if (printProvenanceInfo && (verbosity > 0)) {
      TString eventout("\nProvenance info:\n");

      for (unsigned int i = 0; i < AllProv.size(); ++i) {
	eventout += "\n       ******************************";
	eventout += "\n       Module       : ";
	eventout += AllProv[i]->moduleLabel();
	eventout += "\n       ProductID    : ";
	eventout += AllProv[i]->productID().id();
	eventout += "\n       ClassName    : ";
	eventout += AllProv[i]->className();
	eventout += "\n       InstanceName : ";
	eventout += AllProv[i]->productInstanceName();
	eventout += "\n       BranchName   : ";
	eventout += AllProv[i]->branchName();
      }
      eventout += "       ******************************\n";
      edm::LogInfo("MuonSimHitsValidAnalyzer::analyze") << eventout << "\n";
    }
  }

  /// call fill functions

  /// gather CSC, DT and RPC information from event
//    fillCSC(iEvent, iSetup);
    fillDT(iEvent, iSetup);
 //   fillRPC(iEvent, iSetup);

  if (verbosity > 0)
    edm::LogInfo ("MuonSimHitsValidAnalyzer::analyze")
      << "Done gathering data from event.";

  return;
}



void MuonSimHitsValidAnalyzer::fillCSC(const edm::Event& iEvent,
				 const edm::EventSetup& iSetup)
{

  TString eventout;
  if (verbosity > 0)
    eventout = "\nGathering CSC info:";

  /// iterator to access containers
  edm::PSimHitContainer::const_iterator itHit;

  /// access the CSC
  /// access the CSC geometry
  edm::ESHandle<CSCGeometry> theCSCGeometry;
  iSetup.get<MuonGeometryRecord>().get(theCSCGeometry);
  if (!theCSCGeometry.isValid()) {
    edm::LogWarning("MuonSimHitsValidAnalyzer::fillCSC")
      << "Unable to find MuonGeometryRecord for the CSCGeometry in event!";
    return;
  }
  const CSCGeometry& theCSCMuon(*theCSCGeometry);

  /// get  CSC information
  edm::Handle<edm::PSimHitContainer> MuonCSCContainer;
  iEvent.getByToken(CSCHitsToken_, MuonCSCContainer);
  if (!MuonCSCContainer.isValid()) {
    edm::LogWarning("MuonSimHitsValidAnalyzer::fillCSC")
      << "Unable to find MuonCSCHits in event!";
    return;
  }

  nummu_CSC =0;
  meAllCSCHits->Fill( MuonCSCContainer->size() );

  /// cycle through container
  int i = 0, j = 0;
  for (itHit = MuonCSCContainer->begin(); itHit != MuonCSCContainer->end();
       ++itHit) {
    ++i;


    /// create a DetId from the detUnitId
    DetId theDetUnitId(itHit->detUnitId());
    int detector = theDetUnitId.det();
    int subdetector = theDetUnitId.subdetId();

    /// check that expected detector is returned
    if ((detector == dMuon) &&
        (subdetector == sdMuonCSC)) {

      /// get the GeomDetUnit from the geometry using theDetUnitID
      const GeomDetUnit *theDet = theCSCMuon.idToDetUnit(theDetUnitId);

      if (!theDet) {
	edm::LogWarning("MuonSimHitsValidAnalyzer::fillCSC")
	  << "Unable to get GeomDetUnit from theCSCMuon for hit " << i;
	continue;
      }

      ++j;

      /// get the Surface of the hit (knows how to go from local <-> global)
   //   const BoundPlane& bsurf = theDet->surface();

      /// gather necessary information

      if ( abs(itHit->particleType()) == 13 ) {

      nummu_CSC++;

 /* Comment out for the moment
      const CSCDetId& id=CSCDetId(itHit->detUnitId());

      int cscid=id.endcap()*100000 + id.station()*10000 +
                id.ring()*1000     + id.chamber()*10 +id.layer();

      int iden = cscid/1000;

      hmgr->getHisto1(iden+2000)->Fill( itHit->energyLoss()*pow6 );
      hmgr->getHisto1(iden+2200)->Fill( itHit->tof() );
 */
      }
    } else {
      edm::LogWarning("MuonSimHitsValidAnalyzer::fillCSC")
        << "MuonCsc PSimHit " << i
        << " is expected to be (det,subdet) = ("
        << dMuon << "," << sdMuonCSC
        << "); value returned is: ("
        << detector << "," << subdetector << ")";
      continue;
    }
  }

  if (verbosity > 1) {
    eventout += "\n          Number of CSC muon Hits collected:......... ";
    eventout += j;
  }

   meMuCSCHits->Fill( (float) nummu_CSC );

  if (verbosity > 0)
    edm::LogInfo("MuonSimHitsValidAnalyzer::fillCSC") << eventout << "\n";

  return;
}


void MuonSimHitsValidAnalyzer::fillDT(const edm::Event& iEvent,
				 const edm::EventSetup& iSetup)
{
 TString eventout;
  if (verbosity > 0)
    eventout = "\nGathering DT info:";

  /// iterator to access containers
  edm::PSimHitContainer::const_iterator itHit;

  /// access the DT
  /// access the DT geometry
  edm::ESHandle<DTGeometry> theDTGeometry;
  iSetup.get<MuonGeometryRecord>().get(theDTGeometry);
  if (!theDTGeometry.isValid()) {
    edm::LogWarning("MuonSimHitsValidAnalyzer::fillDT")
      << "Unable to find MuonGeometryRecord for the DTGeometry in event!";
    return;
  }
  const DTGeometry& theDTMuon(*theDTGeometry);

  /// get DT information
  edm::Handle<edm::PSimHitContainer> MuonDTContainer;
  iEvent.getByToken(DTHitsToken_, MuonDTContainer);
  if (!MuonDTContainer.isValid()) {
    edm::LogWarning("MuonSimHitsValidAnalyzer::fillDT")
      << "Unable to find MuonDTHits in event!";
    return;
  }

  touch1 = 0;
  touch4 = 0;
  nummu_DT = 0 ;

  meAllDTHits->Fill( MuonDTContainer->size() );

  /// cycle through container
  int i = 0, j = 0;
  for (itHit = MuonDTContainer->begin(); itHit != MuonDTContainer->end();
       ++itHit) {

    ++i;

    /// create a DetId from the detUnitId
    DetId theDetUnitId(itHit->detUnitId());
    int detector = theDetUnitId.det();
    int subdetector = theDetUnitId.subdetId();

    /// check that expected detector is returned
    if ((detector == dMuon) &&
        (subdetector == sdMuonDT)) {

      /// get the GeomDetUnit from the geometry using theDetUnitID
      const GeomDetUnit *theDet = theDTMuon.idToDetUnit(theDetUnitId);

      if (!theDet) {
  	edm::LogWarning("MuonSimHitsValidAnalyzer::fillDT")
	  << "Unable to get GeomDetUnit from theDTMuon for hit " << i;
	continue;
      }

      ++j;

      /// get the Surface of the hit (knows how to go from local <-> global)
      const BoundPlane& bsurf = theDet->surface();

      /// gather necessary information

      if ( abs(itHit->particleType()) == 13 ) {

       nummu_DT++;
       meToF->Fill( itHit->tof() );
       meEnergyLoss->Fill( itHit->energyLoss()*pow6 );

       iden = itHit->detUnitId();

       wheel = ((iden>>15) & 0x7 ) -3  ;
       station = ((iden>>22) & 0x7 ) ;
       sector = ((iden>>18) & 0xf ) ;
       superlayer = ((iden>>13) & 0x3 ) ;
       layer = ((iden>>10) & 0x7 ) ;
       wire = ((iden>>3) & 0x7f ) ;

       meWheelOccup->Fill((float)wheel);
       meStationOccup->Fill((float) station);
       meSectorOccup->Fill((float) sector);
       meSuperLOccup->Fill((float) superlayer);
       meLayerOccup->Fill((float) layer);
       meWireOccup->Fill((float) wire);

   // Define a quantity to take into account station, splayer and layer being hit.
       path = (station-1) * 40 + superlayer * 10 + layer;
       mePathMuon->Fill((float) path);

   // Define a quantity to take into chamber being hit.
       pathchamber = (wheel+2) * 50 + (station-1) * 12 + sector;
       meChamberOccup->Fill((float) pathchamber);

   /// Muon Momentum at MB1
       if (station == 1 )
        {
         if (touch1 == 0)
         {
          mom1=itHit->pabs();
          meMomentumMB1->Fill(mom1);
          touch1 = 1;
         }
        }

   /// Muon Momentum at MB4 & Loss of Muon Momentum in Iron (between MB1 and MB4)
       if (station == 4 )
        {
         if ( touch4 == 0)
         {
          mom4=itHit->pabs();
          touch4 = 1;
          meMomentumMB4->Fill(mom4);
          if (touch1 == 1 )
          {
           meLossMomIron->Fill(mom1-mom4);
          }
         }
        }

   /// X-Local Coordinate vs Z-Local Coordinate
       meLocalXvsZ->Fill(itHit->localPosition().x(), itHit->localPosition().z() );

   /// X-Local Coordinate vs Y-Local Coordinate
       meLocalXvsY->Fill(itHit->localPosition().x(), itHit->localPosition().y() );

   /// Global Coordinates

      globposz =  bsurf.toGlobal(itHit->localPosition()).z();
      globposeta = bsurf.toGlobal(itHit->localPosition()).eta();
      globposphi = bsurf.toGlobal(itHit->localPosition()).phi();

      radius = globposz* ( 1.+ exp(-2.* globposeta) ) / ( 1. - exp(-2.* globposeta ) ) ;

      costeta = ( 1. - exp(-2.*globposeta) ) /( 1. + exp(-2.* globposeta) ) ;
      sinteta = 2. * exp(-globposeta) /( 1. + exp(-2.*globposeta) );

    /// Z-Global Coordinate vs X-Global Coordinate
    /// Y-Global Coordinate vs X-Global Coordinate
      globposx = radius*sinteta*cos(globposphi);
      globposy = radius*sinteta*sin(globposphi);

      meGlobalXvsZ->Fill(globposz, globposx);
      meGlobalXvsY->Fill(globposx, globposy);

//  New Histos
      if (wheel == -2) {
      meGlobalXvsZWm2->Fill(globposz, globposx);
      meGlobalXvsYWm2->Fill(globposx, globposy);
      }
      if (wheel == -1) {
      meGlobalXvsZWm1->Fill(globposz, globposx);
      meGlobalXvsYWm1->Fill(globposx, globposy);
      }
      if (wheel == 0) {
      meGlobalXvsZW0->Fill(globposz, globposx);
      meGlobalXvsYW0->Fill(globposx, globposy);
      }
      if (wheel == 1) {
      meGlobalXvsZWp1->Fill(globposz, globposx);
      meGlobalXvsYWp1->Fill(globposx, globposy);
      }
      if (wheel == 2) {
      meGlobalXvsZWp2->Fill(globposz, globposx);
      meGlobalXvsYWp2->Fill(globposx, globposy);
      }
//
      meHitRadius->Fill(radius);
      meCosTheta->Fill(costeta);
      meGlobalEta->Fill(globposeta);
      meGlobalPhi->Fill(globposphi);

      }
    } else {
      edm::LogWarning("MuonSimHitsValidAnalyzer::fillDT")
        << "MuonDT PSimHit " << i
        << " is expected to be (det,subdet) = ("
        << dMuon << "," << sdMuonDT
        << "); value returned is: ("
        << detector << "," << subdetector << ")";
      continue;
    }
  }

  if (verbosity > 1) {
    eventout += "\n          Number of DT muon Hits collected:......... ";
    eventout += j;
  }
  meMuDTHits->Fill( (float) nummu_DT );

  if (verbosity > 0)
    edm::LogInfo("MuonSimHitsValidAnalyzer::fillDT") << eventout << "\n";
return;
}


void MuonSimHitsValidAnalyzer::fillRPC(const edm::Event& iEvent,
				 const edm::EventSetup& iSetup)
{
  TString eventout;
  if (verbosity > 0)
    eventout = "\nGathering RPC info:";

  /// iterator to access containers
  edm::PSimHitContainer::const_iterator itHit;

  /// access the RPC
  /// access the RPC geometry
  edm::ESHandle<RPCGeometry> theRPCGeometry;
  iSetup.get<MuonGeometryRecord>().get(theRPCGeometry);
  if (!theRPCGeometry.isValid()) {
    edm::LogWarning("MuonSimHitsValidAnalyzer::fillRPC")
      << "Unable to find MuonGeometryRecord for the RPCGeometry in event!";
    return;
  }
  const RPCGeometry& theRPCMuon(*theRPCGeometry);

  // get Muon RPC information
  edm::Handle<edm::PSimHitContainer> MuonRPCContainer;
  iEvent.getByToken(RPCHitsToken_, MuonRPCContainer);
  if (!MuonRPCContainer.isValid()) {
    edm::LogWarning("MuonSimHitsValidAnalyzer::fillRPC")
      << "Unable to find MuonRPCHits in event!";
    return;
  }

  touch1 = 0;
  touch4 = 0;
  touche1 = 0;
  touche4 = 0;
  nummu_RPC = 0 ;

  meAllRPCHits->Fill( MuonRPCContainer->size() );

  /// cycle through container
  int i = 0, j = 0;
  for (itHit = MuonRPCContainer->begin(); itHit != MuonRPCContainer->end();
       ++itHit) {

    ++i;

    /// create a DetId from the detUnitId
    DetId theDetUnitId(itHit->detUnitId());
    int detector = theDetUnitId.det();
    int subdetector = theDetUnitId.subdetId();

    /// check that expected detector is returned
    if ((detector == dMuon) &&
        (subdetector == sdMuonRPC)) {

      /// get the GeomDetUnit from the geometry using theDetUnitID
      const GeomDetUnit *theDet = theRPCMuon.idToDetUnit(theDetUnitId);

      if (!theDet) {
	edm::LogWarning("MuonSimHitsValidAnalyzer::fillRPC")
	  << "Unable to get GeomDetUnit from theRPCMuon for hit " << i;
	continue;
      }

      ++j;

      /// get the Surface of the hit (knows how to go from local <-> global)
      const BoundPlane& bsurf = theDet->surface();

      /// gather necessary information

      if ( abs(itHit->particleType()) == 13 ) {

       nummu_RPC++;

       iden = itHit->detUnitId();

       region = ( ((iden>>0) & 0X3) -1 )  ;
       ring = ((iden>>2) & 0X7 ) ;

       if ( ring < 3 )
       {
        if ( region == 0 ) cout << "Region - Ring inconsistency" << endl;
        ring += 1 ;
       } else {
        ring -= 5 ;
       }

       station =  ( ((iden>>5) & 0X3) + 1 ) ;
       sector =  ( ((iden>>7) & 0XF) + 1 ) ;
       layer = ( ((iden>>11) & 0X1) + 1 ) ;
       subsector =  ( ((iden>>12) & 0X7) + 1 ) ;   //  ! Beware: mask says 0x7 !!
       roll =  ( (iden>>15) & 0X7)  ;

       meRegionOccup->Fill((float)region);                  // Region
       if (region == 0 )   // Barrel
        {
          meRingOccBar->Fill((float) ring);
          meStatOccBar->Fill((float) station);
          meSectorOccBar->Fill((float) sector);
          meLayerOccBar->Fill((float) layer);
          meSubSectOccBar->Fill((float) subsector);
          meRollOccBar->Fill((float) roll);

          meElossBar->Fill(itHit->energyLoss()*pow6 );
        }
       if (region != 0 )   // Endcaps
        {
           meRingOccEndc->Fill((float)ring);
           meStatOccEndc->Fill((float) station);
           meSectorOccEndc->Fill((float) sector);
           meLayerOccEndc->Fill((float) layer);
           meSubSectOccEndc->Fill((float) subsector);
           meRollOccEndc->Fill((float) roll);

           meElossEndc->Fill(itHit->energyLoss()*pow6 );
        }

   // Define a quantity to take into account station, splayer and layer being hit.
        path = (region+1) * 50 + (ring+2) * 10 + (station -1) *2+ layer;
        if (region != 0) path -= 10 ;
        mepathRPC->Fill((float)path);

  /// Muon Momentum at RB1 (Barrel)
        if ( region == 0 )  //  BARREL
       {
         if (station == 1 && layer == 1 )
         {
          if (touch1 == 0)
          {
           mom1=itHit->pabs();
           meMomRB1->Fill(mom1);
           touch1 = 1;
          }
         }
   /// Muon Momentum at RB4 (Barrel)

         if (station == 4 )
         {
          if ( touch4 == 0)
          {
           mom4=itHit->pabs();
           meMomRB4->Fill(mom4);
           touch4 = 1;
/// Loss of Muon Momentum in Iron (between RB1_layer1 and RB4)
           if (touch1 == 1 )
            {
             meLossMomBar->Fill(mom1-mom4);
            }
          }
         }
       }  // End of Barrel

  /// Muon Momentum at RE1 (Endcaps)
        if ( region != 0 )  //  ENDCAPS
       {
         if (station == 1 )
         {
          if (touche1 == 0)
          {
           mome1=itHit->pabs();
           meMomRE1->Fill(mome1);
           touche1 = 1;
          }
         }
   /// Muon Momentum at RE4 (Endcaps)
         if (station == 4 )
         {
          if ( touche4 == 0)
          {
           mome4=itHit->pabs();
           meMomRE4->Fill(mome4);
           touche4 = 1;
 /// Loss of Muon Momentum in Iron (between RE1_layer1 and RE4)
           if (touche1 == 1 )
            {
             meLossMomEndc->Fill(mome1-mome4);
            }
          }
         }
       }  // End of Endcaps

  //  X-Local Coordinate vs Y-Local Coordinate
       meLocalXvsYBar->Fill(itHit->localPosition().x(), itHit->localPosition().y() );

  /// X-Global Coordinate vs Z-Global Coordinate
       globposz =  bsurf.toGlobal(itHit->localPosition()).z();
       globposeta = bsurf.toGlobal(itHit->localPosition()).eta();
       globposphi = bsurf.toGlobal(itHit->localPosition()).phi();

       radius = globposz* ( 1.+ exp(-2.* globposeta) ) / ( 1. - exp(-2.* globposeta ) ) ;
       costeta = ( 1. - exp(-2.*globposeta) ) /( 1. + exp(-2.* globposeta) ) ;
       sinteta = 2. * exp(-globposeta) /( 1. + exp(-2.*globposeta) );

       globposx = radius*sinteta*cos(globposphi);
       globposy = radius*sinteta*sin(globposphi);

       if (region == 0 ) // Barrel
        {
         meHitRadiusBar->Fill(radius);
         meCosThetaBar->Fill(costeta);
         meGlobalXvsZBar->Fill(globposz, globposx);
         meGlobalXvsYBar->Fill(globposx, globposy);
        }
       if (region != 0 ) // Endcaps
        {
         meHitRadiusEndc->Fill(radius);
         meCosThetaEndc->Fill(costeta);
         meGlobalXvsZEndc->Fill(globposz, globposx);
         meGlobalXvsYEndc->Fill(globposx, globposy);
        }

      }

    } else {
      edm::LogWarning("MuonSimHitsValidAnalyzer::fillRPC")
        << "MuonRpc PSimHit " << i
        << " is expected to be (det,subdet) = ("
        << dMuon << "," << sdMuonRPC
        << "); value returned is: ("
        << detector << "," << subdetector << ")";
      continue;
    }
  }

  if (verbosity > 1) {
    eventout += "\n          Number of RPC muon Hits collected:......... ";
    eventout += j;
  }

  meMuRPCHits->Fill( (float) nummu_RPC );

  if (verbosity > 0)
    edm::LogInfo("MuonSimHitsValidAnalyzer::fillRPC") << eventout << "\n";

return;
}

