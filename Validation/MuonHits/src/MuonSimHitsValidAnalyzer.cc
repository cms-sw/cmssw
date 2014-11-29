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

   /// get labels for input tags
   DTHitsToken_  = consumes<edm::PSimHitContainer>(
       iPSet.getParameter<edm::InputTag>("DTHitsSrc"));

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
}

void MuonSimHitsValidAnalyzer::bookHistograms(DQMStore::IBooker & iBooker, edm::Run const & iRun, edm::EventSetup const & /* iSetup */)
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

  Char_t histo_n[100];
  Char_t histo_t[100];

  iBooker.setCurrentFolder("MuonDTHitsV/DTHitsValidationTask");

  sprintf (histo_n, "Number_of_all_DT_hits" );
  sprintf (histo_t, "Number_of_all_DT_hits" );
  meAllDTHits = iBooker.book1D(histo_n, histo_t,  200, 1.0, 201.0) ;

  sprintf (histo_n, "Number_of_muon_DT_hits" );
  sprintf (histo_t, "Number_of_muon_DT_hits" );
  meMuDTHits  = iBooker.book1D(histo_n, histo_t, 150, 1.0, 151.0);

  sprintf (histo_n, "Tof_of_hits " );
  sprintf (histo_t, "Tof_of_hits " );
  meToF = iBooker.book1D(histo_n, histo_t, 100, -0.5, 50.) ;

  sprintf (histo_n, "DT_energy_loss_keV" );
  sprintf (histo_t, "DT_energy_loss_keV" );
  meEnergyLoss  = iBooker.book1D(histo_n, histo_t, 100, 0.0, 10.0);

  sprintf (histo_n, "Momentum_at_MB1" );
  sprintf (histo_t, "Momentum_at_MB1" );
  meMomentumMB1 = iBooker.book1D(histo_n, histo_t, 100, 10.0, 200.0);

  sprintf (histo_n, "Momentum_at_MB4" );
  sprintf (histo_t, "Momentum_at_MB4" );
  meMomentumMB4 = iBooker.book1D(histo_n, histo_t, 100, 10.0, 200.0) ;

  sprintf (histo_n, "Loss_of_muon_Momentum_in_Iron" );
  sprintf (histo_t, "Loss_of_muon_Momentum_in_Iron" );
  meLossMomIron  = iBooker.book1D(histo_n, histo_t, 80, 0.0, 40.0) ;

  sprintf (histo_n, "Local_x-coord_vs_local_z-coord_of_muon_hit" );
  sprintf (histo_t, "Local_x-coord_vs_local_z-coord_of_muon_hit" );
  meLocalXvsZ = iBooker.book2D(histo_n, histo_t,100, -150., 150., 100, -0.8, 0.8 ) ;

  sprintf (histo_n, "local_x-coord_vs_local_y-coord_of_muon_hit" );
  sprintf (histo_t, "local_x-coord_vs_local_y-coord_of_muon_hit" );
  meLocalXvsY = iBooker.book2D(histo_n, histo_t, 100, -150., 150., 100, -150., 150. );

  sprintf (histo_n, "Global_x-coord_vs_global_z-coord_of_muon_hit" );
  sprintf (histo_t, "Global_x-coord_vs_global_z-coord_of_muon_hit" );
  meGlobalXvsZ = iBooker.book2D(histo_n, histo_t, 100, -800., 800., 100, -800., 800. ) ;

  sprintf (histo_n, "Global_x-coord_vs_global_y-coord_of_muon_hit" );
  sprintf (histo_t, "Global_x-coord_vs_global_y-coord_of_muon_hit" );
  meGlobalXvsY = iBooker.book2D(histo_n, histo_t, 100, -800., 800., 100, -800., 800. ) ;

  // New histos

  sprintf (histo_n, "Global_x-coord_vs_global_z-coord_of_muon_hit_w-2" );
  sprintf (histo_t, "Global_x-coord_vs_global_z-coord_of_muon_hit_w-2" );
  meGlobalXvsZWm2 = iBooker.book2D(histo_n, histo_t, 100, -800., 800., 100, -800., 800. );

  sprintf (histo_n, "Global_x-coord_vs_global_y-coord_of_muon_hit_w-2" );
  sprintf (histo_t, "Global_x-coord_vs_global_y-coord_of_muon_hit_w-2" );
  meGlobalXvsYWm2 = iBooker.book2D(histo_n, histo_t,  100, -800., 800., 100, -800., 800. );

  sprintf (histo_n, "Global_x-coord_vs_global_z-coord_of_muon_hit_w-1" );
  sprintf (histo_t, "Global_x-coord_vs_global_z-coord_of_muon_hit_w-1" );
  meGlobalXvsZWm1 = iBooker.book2D(histo_n, histo_t, 100, -800., 800., 100, -800., 800. );

  sprintf (histo_n, "Global_x-coord_vs_global_y-coord_of_muon_hit_w-1" );
  sprintf (histo_t, "Global_x-coord_vs_global_y-coord_of_muon_hit_w-1" );
  meGlobalXvsYWm1 = iBooker.book2D(histo_n, histo_t,  100, -800., 800., 100, -800., 800. );

  sprintf (histo_n, "Global_x-coord_vs_global_z-coord_of_muon_hit_w0" );
  sprintf (histo_t, "Global_x-coord_vs_global_z-coord_of_muon_hit_w0" );
  meGlobalXvsZW0 = iBooker.book2D(histo_n, histo_t, 100, -800., 800., 100, -800., 800. );

  sprintf (histo_n, "Global_x-coord_vs_global_y-coord_of_muon_hit_w0" );
  sprintf (histo_t, "Global_x-coord_vs_global_y-coord_of_muon_hit_w0" );
  meGlobalXvsYW0 = iBooker.book2D(histo_n, histo_t,  100, -800., 800., 100, -800., 800. );

  sprintf (histo_n, "Global_x-coord_vs_global_z-coord_of_muon_hit_w1" );
  sprintf (histo_t, "Global_x-coord_vs_global_z-coord_of_muon_hit_w1" );
  meGlobalXvsZWp1 = iBooker.book2D(histo_n, histo_t, 100, -800., 800., 100, -800., 800. );

  sprintf (histo_n, "Global_x-coord_vs_global_y-coord_of_muon_hit_w1" );
  sprintf (histo_t, "Global_x-coord_vs_global_y-coord_of_muon_hit_w1" );
  meGlobalXvsYWp1 = iBooker.book2D(histo_n, histo_t,  100, -800., 800., 100, -800., 800. );

  sprintf (histo_n, "Global_x-coord_vs_global_z-coord_of_muon_hit_w2" );
  sprintf (histo_t, "Global_x-coord_vs_global_z-coord_of_muon_hit_w2" );
  meGlobalXvsZWp2 = iBooker.book2D(histo_n, histo_t, 100, -800., 800., 100, -800., 800. );

  sprintf (histo_n, "Global_x-coord_vs_global_y-coord_of_muon_hit_w2" );
  sprintf (histo_t, "Global_x-coord_vs_global_y-coord_of_muon_hit_w2" );
  meGlobalXvsYWp2 = iBooker.book2D(histo_n, histo_t,  100, -800., 800., 100, -800., 800. );

  //

  sprintf (histo_n, "Wheel_occupancy" );
  sprintf (histo_t, "Wheel_occupancy" );
  meWheelOccup = iBooker.book1D(histo_n, histo_t, 10, -5.0, 5.0) ;

  sprintf (histo_n, "Station_occupancy" );
  sprintf (histo_t, "Station_occupancy" );
  meStationOccup = iBooker.book1D(histo_n, histo_t, 6, 0., 6.0) ;

  sprintf (histo_n, "Sector_occupancy" );
  sprintf (histo_t, "Sector_occupancy" );
  meSectorOccup = iBooker.book1D(histo_n, histo_t, 20, 0., 20.) ;

  sprintf (histo_n, "SuperLayer_occupancy" );
  sprintf (histo_t, "SuperLayer_occupancy" );
  meSuperLOccup = iBooker.book1D(histo_n, histo_t, 5, 0., 5.) ;

  sprintf (histo_n, "Layer_occupancy" );
  sprintf (histo_t, "Layer_occupancy" );
  meLayerOccup = iBooker.book1D(histo_n, histo_t,6, 0., 6.) ;

  sprintf (histo_n, "Wire_occupancy" );
  sprintf (histo_t, "Wire_occupancy" );
  meWireOccup = iBooker.book1D(histo_n, histo_t, 100, 0., 100.) ;

  sprintf (histo_n, "path_followed_by_muon" );
  sprintf (histo_t, "path_followed_by_muon" );
  mePathMuon = iBooker.book1D(histo_n, histo_t, 160, 0., 160.) ;

  sprintf (histo_n, "chamber_occupancy" );
  sprintf (histo_t, "chamber_occupancy" );
  meChamberOccup = iBooker.book1D(histo_n, histo_t,  251, 0., 251.) ;

  sprintf (histo_n, "radius_of_hit");
  sprintf (histo_t, "radius_of_hit");
  meHitRadius = iBooker.book1D(histo_n, histo_t, 100, 0., 1200. );

  sprintf (histo_n, "costheta_of_hit" );
  sprintf (histo_t, "costheta_of_hit" );
  meCosTheta = iBooker.book1D(histo_n, histo_t,  100, -1., 1.) ;

  sprintf (histo_n, "global_eta_of_hit" );
  sprintf (histo_t, "global_eta_of_hit" );
  meGlobalEta = iBooker.book1D(histo_n, histo_t, 60, -2.7, 2.7 );

  sprintf (histo_n, "global_phi_of_hit" );
  sprintf (histo_t, "global_phi_of_hit" );
  meGlobalPhi = iBooker.book1D(histo_n, histo_t, 60, -3.14, 3.14);
}

void MuonSimHitsValidAnalyzer::analyze(const edm::Event& iEvent,
			       const edm::EventSetup& iSetup)
{
  /// keep track of number of events processed
  ++count;

  /// get event id information
  edm::RunNumber_t nrun = iEvent.id().run();
  edm::EventNumber_t nevt = iEvent.id().event();

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

  fillDT(iEvent, iSetup);

  if (verbosity > 0)
    edm::LogInfo ("MuonSimHitsValidAnalyzer::analyze")
      << "Done gathering data from event.";

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



