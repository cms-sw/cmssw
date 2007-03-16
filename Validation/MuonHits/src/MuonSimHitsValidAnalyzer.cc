#include "Validation/MuonHits/src/MuonSimHitsValidAnalyzer.h"
#include "Validation/MuonHits/interface/HistoMgr.h"

#include "TFile.h"
#include "TTree.h"
#include "TBranch.h"
#include "TH1F.h"

#include <iostream>
#include <string>


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
  CSCoutputFile_ =  iPSet.getUntrackedParameter<std::string>("CSC_outputFile", "");
  RPCoutputFile_ =  iPSet.getUntrackedParameter<std::string>("RPC_outputFile", "");


  /// get labels for input tags
   CSCHitsSrc_ = iPSet.getParameter<edm::InputTag>("CSCHitsSrc");
   DTHitsSrc_  = iPSet.getParameter<edm::InputTag>("DTHitsSrc");
   RPCHitsSrc_ = iPSet.getParameter<edm::InputTag>("RPCHitsSrc");

  /// use value of first digit to determine default output level (inclusive)
  /// 0 is none, 1 is basic, 2 is fill output, 3 is gather output
  verbosity %= 10;


  /// print out Parameter Set information being used
  if (verbosity > 0) {
    edm::LogInfo ("MuonSimHitsValidAnalyzer::MuonSimHitsValidAnalyzer") 
      << "\n===============================\n"
      << "Initialized as EDAnalyzer with parameter values:\n"
      << "    Name      = " << fName << "\n"
      << "    Verbosity = " << verbosity << "\n"
      << "    Label     = " << label << "\n"
      << "    GetProv   = " << getAllProvenances << "\n"
      << "    PrintProv = " << printProvenanceInfo << "\n"
      << "    CSCHitsSrc=  " <<CSCHitsSrc_.label() 
      << ":" << CSCHitsSrc_.instance() << "\n"
      << "    DTHitsSrc =  " <<DTHitsSrc_.label()
      << ":" << DTHitsSrc_.instance() << "\n"
      << "    RPCHitsSrc=  " <<RPCHitsSrc_.label()
      << ":" << RPCHitsSrc_.instance() << "\n"
      << "===============================\n";
  }

 bookHistos_DT();
 bookHistos_CSC();
 bookHistos_RPC();
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
 theDTFile->Close();
 theCSCFile->Close();
 theRPCFile->Close();
}

void MuonSimHitsValidAnalyzer::beginJob(const edm::EventSetup& iSetup)
{
  return;
}

void MuonSimHitsValidAnalyzer::bookHistos_DT()
{
  HistoMgr* hmgr = HistoMgr::getInstance();
  string hnam;

   theDTFile = new TFile(DToutputFile_.c_str(),"RECREATE");
   theDTFile->cd();

   hnam = ("Number_of_all_DT_hits " );
   hmgr->addHisto1( 1000, new TH1F(hnam.c_str(),hnam.c_str(), 150, 1.0, 151.0) );

   hnam = ("Number_of_muon_DT_hits" );
   hmgr->addHisto1( 1020, new TH1F(hnam.c_str(),hnam.c_str(),50, 1.0, 51.0) );

   hnam = ("Tof_of_hits " );
   hmgr->addHisto1( 1001, new TH1F(hnam.c_str(),hnam.c_str(), 100, -0.5, 50.) );
   
   hnam = ("DT_energy_loss" );
   hmgr->addHisto1( 1002, new TH1F(hnam.c_str(),hnam.c_str(), 100, 0.0, 10.0) );

   hnam = ("Momentum_at_MB1" );
   hmgr->addHisto1( 1003, new TH1F(hnam.c_str(),hnam.c_str(), 100, 10.0, 200.0) );

   hnam = ("Momentum_at_MB4" );
   hmgr->addHisto1( 1004, new TH1F(hnam.c_str(),hnam.c_str(), 100, 10.0, 200.0) );

   hnam = ("Loss_of_muon_Momentum_in_Iron" );
   hmgr->addHisto1( 1005, new TH1F(hnam.c_str(),hnam.c_str(), 80, 0.0, 40.0) );

   hnam = ("Local_x-coord_vs_local_z-coord_of_muon_hit" );
   hmgr->addHisto2(1006, new TH2F(hnam.c_str(),hnam.c_str(), 100, -150., 150., 100, -0.8, 0.8 ) );

   hnam = ("local_x-coord_vs_local_y-coord_of_muon_hit" );
   hmgr->addHisto2(1007, new TH2F(hnam.c_str(),hnam.c_str(), 100, -150., 150., 100, -150., 150. ) );

   hnam = ("Global_x-coord_vs_global_z-coord_of_muon_hit" );
   hmgr->addHisto2(1008, new TH2F(hnam.c_str(),hnam.c_str(), 100, -800., 800., 100, -800., 800. ) );

   hnam = ("Global_x-coord_vs_global_y-coord_of_muon_hit" );
   hmgr->addHisto2(1009, new TH2F(hnam.c_str(),hnam.c_str(), 100, -800., 800., 100, -800., 800. ) ) ;

   hnam = ("Wheel_occupancy" );
   hmgr->addHisto1( 1010, new TH1F(hnam.c_str(),hnam.c_str(), 10, -5.0, 5.0) );

   hnam = ("Station_occupancy" );
   hmgr->addHisto1( 1011,  new TH1F(hnam.c_str(),hnam.c_str(), 6, 0., 6.0) );
   
   hnam = ("Sector_occupancy" );
   hmgr->addHisto1( 1012,  new TH1F(hnam.c_str(),hnam.c_str(), 20, 0., 20.) );
 
   hnam = ("SuperLayer_occupancy" );
   hmgr->addHisto1( 1013, new TH1F(hnam.c_str(),hnam.c_str(), 5, 0., 5.) );

   hnam = ("Layer_occupancy" );
   hmgr->addHisto1( 1014, new TH1F(hnam.c_str(),hnam.c_str(), 6, 0., 6.) );

   hnam = ("Wire_occupancy" );
   hmgr->addHisto1( 1015, new TH1F(hnam.c_str(),hnam.c_str(), 100, 0., 100.) );

   hnam = ("path_followed_by_muon" );
   hmgr->addHisto1( 1016, new TH1F(hnam.c_str(),hnam.c_str(), 160, 0., 160.) );
 
   hnam = ("chamber_occupancy" );
   hmgr->addHisto1( 1017, new TH1F(hnam.c_str(),hnam.c_str(), 251, 0., 251.) );

   hnam = ("radius_of_hit");
   hmgr->addHisto1( 1018,new TH1F(hnam.c_str(),hnam.c_str(), 100, 0., 1200. ));

   hnam = ("costheta_of_hit" );
   hmgr->addHisto1( 1019,new TH1F(hnam.c_str(),hnam.c_str(), 100, -1., 1.) );

   hnam = ("global_eta_of_hit" );
   hmgr->addHisto1( 1022,new TH1F(hnam.c_str(),hnam.c_str(), 60, -2.7, 2.7 ) );

   hnam = ("global_phi_of_hit" );
   hmgr->addHisto1( 1021, new TH1F(hnam.c_str(),hnam.c_str(), 60, -3.14, 3.14) ) ;

//   theDTFile->ls();
}

void MuonSimHitsValidAnalyzer::bookHistos_RPC()
{
   HistoMgr* hmgr = HistoMgr::getInstance();
   string hnam;

   theRPCFile = new TFile(RPCoutputFile_.c_str(),"RECREATE");
   theRPCFile->cd();


   hnam = ("Number_of_all_RPC_hits" );
   hmgr->addHisto1( 3000, new TH1F(hnam.c_str(),hnam.c_str(), 100, 1.0, 101.0) );

   hnam = ("Region occupancy");
   hmgr->addHisto1( 3001, new TH1F(hnam.c_str(),hnam.c_str(), 6, -3.0, 3.0) );
 
   hnam = ("Ring occupancy (barrel)");
   hmgr->addHisto1( 3002, new TH1F(hnam.c_str(),hnam.c_str(),8, -3., 5.0) );

   hnam = ("Ring occupancy (endcaps)");
   hmgr->addHisto1( 3003, new TH1F(hnam.c_str(),hnam.c_str(),8, -3., 5.0) );

   hnam = ("Station occupancy (barrel)");
   hmgr->addHisto1( 3004, new TH1F(hnam.c_str(),hnam.c_str(), 8, 0., 8.) );

   hnam = ("Station occupancy (endcaps)" );
   hmgr->addHisto1( 3005, new TH1F(hnam.c_str(),hnam.c_str(), 8, 0., 8.) );

   hnam = ("Sector occupancy (barrel)" );
   hmgr->addHisto1( 3006, new TH1F(hnam.c_str(),hnam.c_str(), 16, 0., 16.) );

   hnam = ("Sector occupancy (endcaps)" );
   hmgr->addHisto1( 3007, new TH1F(hnam.c_str(),hnam.c_str(), 16, 0., 16.) ); 

   hnam = ("Layer occupancy (barrel)" );
   hmgr->addHisto1( 3008, new TH1F(hnam.c_str(),hnam.c_str(), 4, 0., 4.) );

   hnam = ("Layer occupancy (endcaps)" );
   hmgr->addHisto1( 3009, new TH1F(hnam.c_str(),hnam.c_str(), 4, 0., 4.) );

   hnam = ("Subsector occupancy (barrel)" );
   hmgr->addHisto1( 3010, new TH1F(hnam.c_str(),hnam.c_str(), 10, 0., 10.) );

   hnam = ("Subsector occupancy (endcaps)" );
   hmgr->addHisto1( 3011, new TH1F(hnam.c_str(),hnam.c_str(), 10, 0., 10.) );

   hnam = ("Roll occupancy (barrel)" );
   hmgr->addHisto1( 3012, new TH1F(hnam.c_str(),hnam.c_str(), 6, 0., 6.) );

   hnam = ("Roll occupancy (endcaps)" );
   hmgr->addHisto1( 3013, new TH1F(hnam.c_str(),hnam.c_str(), 6, 0., 6.) );

   hnam = ("RPC energy_loss (barrel)" );
   hmgr->addHisto1( 3014, new TH1F(hnam.c_str(),hnam.c_str(), 50, 0.0, 10.0) );

   hnam = ("RPC energy_loss (endcaps)" );
   hmgr->addHisto1( 3015, new TH1F(hnam.c_str(),hnam.c_str(), 50, 0.0, 10.0)  );

   hnam = ("path followed by muon" );
   hmgr->addHisto1( 3016, new TH1F(hnam.c_str(),hnam.c_str(), 160, 0., 160.) );

   hnam = ("Momentum at RB1") ;
   hmgr->addHisto1( 3017, new TH1F(hnam.c_str(),hnam.c_str(), 80, 10.0, 200.0) );

   hnam = ("Momentum at RB4") ;
   hmgr->addHisto1( 3018, new TH1F(hnam.c_str(),hnam.c_str(), 80, 10.0, 200.0) );

   hnam = ("Loss of muon Momentum in Iron (barrel)" );
   hmgr->addHisto1( 3019, new TH1F(hnam.c_str(),hnam.c_str(), 80, 0.0, 40.0) );

   hnam = ("Momentum at RE1");
   hmgr->addHisto1( 3020, new TH1F(hnam.c_str(),hnam.c_str(), 100, 10.0, 300.0) );

   hnam = ("Momentum at RE4") ;
   hmgr->addHisto1( 3021, new TH1F(hnam.c_str(),hnam.c_str(), 100, 10.0, 300.0) );

   hnam = ("Loss of muon Momentum in Iron (endcap)" );
   hmgr->addHisto1( 3022, new TH1F(hnam.c_str(),hnam.c_str(), 80, 0.0, 40.0) );

   hnam = ("local x-coord. vs local y-coord of muon hit") ;
   hmgr->addHisto2( 3023, new TH2F(hnam.c_str(),hnam.c_str(), 100, -150., 150., 100, -100., 100. ) );

   hnam = ("Global z-coord. vs global x-coord of muon hit (barrel)" );
   hmgr->addHisto2( 3024, new TH2F(hnam.c_str(),hnam.c_str(), 100, -800., 800., 100, -800., 800. ) );

   hnam = ("Global x-coord. vs global y-coord of muon hit (barrel)" );
   hmgr->addHisto2( 3025, new TH2F(hnam.c_str(),hnam.c_str(), 100, -800., 800., 100, -800., 800. ) );

   hnam = ("radius of hit (barrel)" );
   hmgr->addHisto1( 3026, new TH1F(hnam.c_str(),hnam.c_str(), 100, 0., 1200.) );

   hnam = ("radius of hit (endcaps)" );
   hmgr->addHisto1( 3027, new TH1F(hnam.c_str(),hnam.c_str(), 100, 0., 1300.) );

   hnam = ("costheta of hit (barrel)" ) ;
   hmgr->addHisto1( 3028, new TH1F(hnam.c_str(),hnam.c_str(), 100, -1., 1.) );

   hnam = ("costheta of hit (endcaps)" );
   hmgr->addHisto1( 3029, new TH1F(hnam.c_str(),hnam.c_str(), 100, -1., 1.) );

   hnam = ("Global z-coord. vs global x-coord of muon hit (endcaps)" );
   hmgr->addHisto2( 3030, new TH2F(hnam.c_str(),hnam.c_str(), 100, -1200., 1200., 100, -800., 800. ) );

   hnam = ("Global x-coord. vs global y-coord of muon hit (endcaps)" );
   hmgr->addHisto2( 3031, new TH2F(hnam.c_str(),hnam.c_str(), 100, -800., 800., 100, -800., 800. ) ); 

   hnam = ("Number of muon RPC hits" );
   hmgr->addHisto1( 3032, new TH1F(hnam.c_str(),hnam.c_str(), 50, 1.0, 51.0) );
  
//   theRPCFile->ls();
}

void MuonSimHitsValidAnalyzer::bookHistos_CSC()
{
   HistoMgr* hmgr = HistoMgr::getInstance();
   string hnam;

   theCSCFile = new TFile(CSCoutputFile_.c_str(),"RECREATE");
   theCSCFile->cd();
 
   hnam = ("Number_of_all_CSC_hits " );
   hmgr->addHisto1( 2000, new TH1F(hnam.c_str(),hnam.c_str(), 100, 1.0, 101.0) );

   hnam = ("Number_of_muon_CSC_hits" );
   hmgr->addHisto1( 2001, new TH1F(hnam.c_str(),hnam.c_str(), 50, 1.0, 51.0) );

   char labelh[10];
   char histoName[40];
   char histoNametof[40];

   for ( int k = 1; k <3 ; ++k) {
     for ( int i = 1; i<5 ; ++i)  {
       for ( int j = 1; j<5 ; ++j)  {
              
           if (i != 1 && j>2 ) continue;
           if (i == 4 && j>1 ) continue;  

           int idhisto_ener = 2000+ k*100+ i*10+j;
           int idhisto = idhisto_ener - 2000;
           sprintf(labelh,"ME%d", idhisto) ; 
           strcpy(histoName, labelh);
           strcat(histoName,"_energy loss");
           hnam = (histoName);
           hmgr->addHisto1(idhisto_ener, new TH1F(hnam.c_str(),hnam.c_str(),50, 0.0, 50.0) );

           int idhisto_tof = idhisto_ener+200; 
           strcpy(histoNametof, labelh);
           strcat(histoNametof,"_tof"); 

           hnam = (histoNametof);
           hmgr->addHisto1(idhisto_tof,new TH1F(hnam.c_str(),hnam.c_str(), 60, 0.0, 60.0) ); 
       }
     }
   }
//  theCSCFile->ls();
}




void MuonSimHitsValidAnalyzer::saveHistos_DT()
{
  int DTHistos;

  DTHistos = 1000;
  theDTFile->cd();
  
  HistoMgr* hmgr = HistoMgr::getInstance();
//  gDirectory->pwd();
//  theDTFile->ls();
// theDTFile->GetList()->ls();
  hmgr->save(DTHistos); 
}

void MuonSimHitsValidAnalyzer::saveHistos_RPC()
{
  int RPCHistos;
  RPCHistos = 3000;
  theRPCFile->cd();

  HistoMgr* hmgr = HistoMgr::getInstance();
//  gDirectory->pwd();
//  theRPCFile->ls();
//  theRPCFile->GetList()->ls();
  hmgr->save(RPCHistos);
}

void MuonSimHitsValidAnalyzer::saveHistos_CSC()
{
  int CSCHistos;
  CSCHistos = 2000;
  theCSCFile->cd();
 
  HistoMgr* hmgr = HistoMgr::getInstance();
//  gDirectory->pwd();
//  theCSCFile->ls();
//  theCSCFile->GetList()->ls();
     hmgr->save(CSCHistos);
}

void MuonSimHitsValidAnalyzer::endJob()
{
  saveHistos_DT();
  saveHistos_CSC();
  saveHistos_RPC(); 
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
	eventout += (AllProv[i]->product).moduleLabel();
	eventout += "\n       ProductID    : ";
	eventout += (AllProv[i]->product).productID_.id_;
	eventout += "\n       ClassName    : ";
	eventout += (AllProv[i]->product).fullClassName_;
	eventout += "\n       InstanceName : ";
	eventout += (AllProv[i]->product).productInstanceName_;
	eventout += "\n       BranchName   : ";
	eventout += (AllProv[i]->product).branchName_;
      }
      eventout += "       ******************************\n";
      edm::LogInfo("MuonSimHitsValidAnalyzer::analyze") << eventout << "\n";
    }
  }
 
  /// call fill functions

  /// gather CSC, DT and RPC information from event
    fillCSC(iEvent, iSetup);
    fillDT(iEvent, iSetup);
    fillRPC(iEvent, iSetup);

  if (verbosity > 0)
    edm::LogInfo ("MuonSimHitsValidAnalyzer::analyze")
      << "Done gathering data from event.";

  return;
}



void MuonSimHitsValidAnalyzer::fillCSC(const edm::Event& iEvent, 
				 const edm::EventSetup& iSetup)
{
  HistoMgr* hmgr = HistoMgr::getInstance();

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
  iEvent.getByLabel(CSCHitsSrc_,MuonCSCContainer);
//  iEvent.getByLabel("g4SimHits","MuonCSCHits",MuonCSCContainer);
  if (!MuonCSCContainer.isValid()) {
    edm::LogWarning("MuonSimHitsValidAnalyzer::fillCSC")
      << "Unable to find MuonCSCHits in event!";
    return;
  }

  nummu_CSC =0;
  hmgr->getHisto1(2000)->Fill( MuonCSCContainer->size() );

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

      const CSCDetId& id=CSCDetId(itHit->detUnitId());

      int cscid=id.endcap()*100000 + id.station()*10000 +
                id.ring()*1000     + id.chamber()*10 +id.layer(); 

      int iden = cscid/1000;

      hmgr->getHisto1(iden+2000)->Fill( itHit->energyLoss()*pow6 );
      hmgr->getHisto1(iden+2200)->Fill( itHit->tof() );

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

   hmgr->getHisto1(2001)->Fill( (float) nummu_CSC );

  if (verbosity > 0)
    edm::LogInfo("MuonSimHitsValidAnalyzer::fillCSC") << eventout << "\n";

  return;
}


void MuonSimHitsValidAnalyzer::fillDT(const edm::Event& iEvent, 
				 const edm::EventSetup& iSetup)
{
 HistoMgr* hmgr = HistoMgr::getInstance();
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
  iEvent.getByLabel(DTHitsSrc_,MuonDTContainer);
//  iEvent.getByLabel("g4SimHits","MuonDTHits",MuonDTContainer);
  if (!MuonDTContainer.isValid()) {
    edm::LogWarning("MuonSimHitsValidAnalyzer::fillDT")
      << "Unable to find MuonDTHits in event!";
    return;
  }

  touch1 = 0;
  touch4 = 0;
  nummu_DT = 0 ;
  hmgr->getHisto1(1000)->Fill( MuonDTContainer->size() );

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
       hmgr->getHisto1(1001)->Fill( itHit->tof() );
       hmgr->getHisto1(1002)->Fill( itHit->energyLoss()*pow6 );     

       iden = itHit->detUnitId();
       
       wheel = ((iden>>15) & 0x7 ) -3  ;
       station = ((iden>>22) & 0x7 ) ;
       sector = ((iden>>18) & 0xf ) ;
       superlayer = ((iden>>13) & 0x3 ) ;
       layer = ((iden>>10) & 0x7 ) ;
       wire = ((iden>>3) & 0x7f ) ;

       hmgr->getHisto1(1010)->Fill((float)wheel);
       hmgr->getHisto1(1011)->Fill((float) station);
       hmgr->getHisto1(1012)->Fill((float) sector);
       hmgr->getHisto1(1013)->Fill((float) superlayer);
       hmgr->getHisto1(1014)->Fill((float) layer);
       hmgr->getHisto1(1015)->Fill((float) wire);

   // Define a quantity to take into account station, splayer and layer being hit.
       path = (station-1) * 40 + superlayer * 10 + layer;
       hmgr->getHisto1(1016)->Fill((float) path); 

   // Define a quantity to take into chamber being hit.
       pathchamber = (wheel+2) * 50 + (station-1) * 12 + sector;
       hmgr->getHisto1(1017)->Fill((float) pathchamber);

   /// Muon Momentum at MB1
       if (station == 1 )
        {
         if (touch1 == 0)
         {
          mom1=itHit->pabs();  
          hmgr->getHisto1(1003)->Fill(mom1);
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
          hmgr->getHisto1(1004)->Fill(mom4);
          if (touch1 == 1 )
          {
           hmgr->getHisto1(1005)->Fill(mom1-mom4);
          }
         } 
        }

   /// X-Local Coordinate vs Z-Local Coordinate
       hmgr->getHisto2(1006)->Fill(itHit->localPosition().x(), itHit->localPosition().z() ); 
   
   /// X-Local Coordinate vs Y-Local Coordinate
       hmgr->getHisto2(1007)->Fill(itHit->localPosition().x(), itHit->localPosition().y() );

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

      hmgr->getHisto2(1008)->Fill(globposz, globposx);
      hmgr->getHisto2(1009)->Fill(globposx, globposy); 

      hmgr->getHisto1(1018)->Fill(radius);
      hmgr->getHisto1(1019)->Fill(costeta);
      hmgr->getHisto1(1022)->Fill(globposeta);
      hmgr->getHisto1(1021)->Fill(globposphi);

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
  hmgr->getHisto1(1020)->Fill( (float) nummu_DT );
   
  if (verbosity > 0)
    edm::LogInfo("MuonSimHitsValidAnalyzer::fillDT") << eventout << "\n";
return;
}


void MuonSimHitsValidAnalyzer::fillRPC(const edm::Event& iEvent, 
				 const edm::EventSetup& iSetup)
{
  HistoMgr* hmgr = HistoMgr::getInstance();
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
  iEvent.getByLabel(RPCHitsSrc_,MuonRPCContainer);
//  iEvent.getByLabel("g4SimHits","MuonRPCHits",MuonRPCContainer);
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

  hmgr->getHisto1(3000)->Fill( MuonRPCContainer->size() );

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
  
       hmgr->getHisto1(3001)->Fill((float)region);                  // Region
       if (region == 0 )   // Barrel
        {  
          hmgr->getHisto1(3002)->Fill((float) ring);  
          hmgr->getHisto1(3004)->Fill((float) station); 
          hmgr->getHisto1(3006)->Fill((float) sector);
          hmgr->getHisto1(3008)->Fill((float) layer);
          hmgr->getHisto1(3010)->Fill((float) subsector);
          hmgr->getHisto1(3012)->Fill((float) roll);

          hmgr->getHisto1(3014)->Fill(itHit->energyLoss()*pow6 );
        }
       if (region != 0 )   // Endcaps
        {
           hmgr->getHisto1(3003)->Fill((float)ring);
           hmgr->getHisto1(3005)->Fill((float) station);
           hmgr->getHisto1(3007)->Fill((float) sector);
           hmgr->getHisto1(3009)->Fill((float) layer);
           hmgr->getHisto1(3011)->Fill((float) subsector);
           hmgr->getHisto1(3013)->Fill((float) roll);  

           hmgr->getHisto1(3015)->Fill(itHit->energyLoss()*pow6 );
        } 

   // Define a quantity to take into account station, splayer and layer being hit.
        path = (region+1) * 50 + (ring+2) * 10 + (station -1) *2+ layer;
        if (region != 0) path -= 10 ;
        hmgr->getHisto1(3016)->Fill((float)path);

  /// Muon Momentum at RB1 (Barrel)
        if ( region == 0 )  //  BARREL
       {
         if (station == 1 && layer == 1 ) 
         {
          if (touch1 == 0)
          {
           mom1=itHit->pabs();
           hmgr->getHisto1(3017)->Fill(mom1);
           touch1 = 1;
          }
         }
   /// Muon Momentum at RB4 (Barrel)

         if (station == 4 )
         {
          if ( touch4 == 0)
          { 
           mom4=itHit->pabs();
           hmgr->getHisto1(3018)->Fill(mom4);
           touch4 = 1;
/// Loss of Muon Momentum in Iron (between RB1_layer1 and RB4)
           if (touch1 == 1 )
            {
             hmgr->getHisto1(3019)->Fill(mom1-mom4);
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
           hmgr->getHisto1(3020)->Fill(mome1);
           touche1 = 1;
          }
         }  
   /// Muon Momentum at RE4 (Endcaps)
         if (station == 4 )
         {
          if ( touche4 == 0)
          {
           mome4=itHit->pabs();
           hmgr->getHisto1(3021)->Fill(mome4);
           touche4 = 1;
 /// Loss of Muon Momentum in Iron (between RE1_layer1 and RE4) 
           if (touche1 == 1 )
            {
             hmgr->getHisto1(3022)->Fill(mome1-mome4); 
            } 
          }
         }
       }  // End of Endcaps 

  //  X-Local Coordinate vs Y-Local Coordinate
       hmgr->getHisto2(3023)->Fill(itHit->localPosition().x(), itHit->localPosition().y() );

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
         hmgr->getHisto1(3026)->Fill(radius);
         hmgr->getHisto1(3028)->Fill(costeta);
         hmgr->getHisto2(3024)->Fill(globposz, globposx);
         hmgr->getHisto2(3025)->Fill(globposx, globposy);
        } 
       if (region != 0 ) // Endcaps
        {
         hmgr->getHisto1(3027)->Fill(radius); 
         hmgr->getHisto1(3029)->Fill(costeta);
         hmgr->getHisto2(3030)->Fill(globposz, globposx);
         hmgr->getHisto2(3031)->Fill(globposx, globposy);
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

  hmgr->getHisto1(3032)->Fill( (float) nummu_RPC );

  if (verbosity > 0)
    edm::LogInfo("MuonSimHitsValidAnalyzer::fillRPC") << eventout << "\n";

return;
}

