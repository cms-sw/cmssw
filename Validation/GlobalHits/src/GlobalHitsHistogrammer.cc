/** \file GlobalHitsHistogrammer.cc
 *  
 *  See header file for description of class
 *
 *  $Date: 2010/01/06 14:24:50 $
 *  $Revision: 1.8 $
 *  \author M. Strang SUNY-Buffalo
 */

#include "Validation/GlobalHits/interface/GlobalHitsHistogrammer.h"
#include "DQMServices/Core/interface/DQMStore.h"

GlobalHitsHistogrammer::GlobalHitsHistogrammer(const edm::ParameterSet& iPSet) 
  : fName(""), verbosity(0), frequency(0), vtxunit(0), label(""), 
  getAllProvenances(false), printProvenanceInfo(false), count(0)
{
  std::string MsgLoggerCat = "GlobalHitsHistogrammer_GlobalHitsHistogrammer";

  // get information from parameter set
  fName = iPSet.getUntrackedParameter<std::string>("Name");
  verbosity = iPSet.getUntrackedParameter<int>("Verbosity");
  frequency = iPSet.getUntrackedParameter<int>("Frequency");
  vtxunit = iPSet.getUntrackedParameter<int>("VtxUnit");
  outputfile = iPSet.getParameter<std::string>("OutputFile");
  doOutput = iPSet.getParameter<bool>("DoOutput");
  edm::ParameterSet m_Prov =
    iPSet.getParameter<edm::ParameterSet>("ProvenanceLookup");
  getAllProvenances = 
    m_Prov.getUntrackedParameter<bool>("GetAllProvenances");
  printProvenanceInfo = 
    m_Prov.getUntrackedParameter<bool>("PrintProvenanceInfo");

  //get Labels to use to extract information
  GlobalHitSrc_ = iPSet.getParameter<edm::InputTag>("GlobalHitSrc");

  // use value of first digit to determine default output level (inclusive)
  // 0 is none, 1 is basic, 2 is fill output, 3 is gather output
  verbosity %= 10;

  // print out Parameter Set information being used
  if (verbosity >= 0) {
    edm::LogInfo(MsgLoggerCat) 
      << "\n===============================\n"
      << "Initialized as EDAnalyzer with parameter values:\n"
      << "    Name          = " << fName << "\n"
      << "    Verbosity     = " << verbosity << "\n"
      << "    Frequency     = " << frequency << "\n"
      << "    VtxUnit       = " << vtxunit << "\n"
      << "    OutputFile    = " << outputfile << "\n"
      << "    DoOutput      = " << doOutput << "\n"
      << "    GetProv       = " << getAllProvenances << "\n"
      << "    PrintProv     = " << printProvenanceInfo << "\n"
      << "    GlobalHitSrc  = " << GlobalHitSrc_.label() 
      << ":" << GlobalHitSrc_.instance() << "\n"
      << "===============================\n";
  }

  // get dqm info
  dbe = 0;
  dbe = edm::Service<DQMStore>().operator->();
  if (dbe) {
    if (verbosity > 0 ) {
      dbe->setVerbose(1);
    } else {
      dbe->setVerbose(0);
    }
  }

  if (dbe) {
    if (verbosity > 0 ) dbe->showDirStructure();
  }

  // initialize monitor elements
  for (Int_t i = 0; i < 2; ++i) {
    meMCRGP[i] = 0;
    meMCG4Vtx[i] = 0;
    meGeantVtxX[i] = 0;
    meGeantVtxY[i] = 0;
    meGeantVtxZ[i] = 0; 
    meMCG4Trk[i] = 0;
    meCaloEcal[i] = 0;
    meCaloEcalE[i] = 0;
    meCaloEcalToF[i] = 0;
    meCaloPreSh[i] = 0;
    meCaloPreShE[i] = 0;
    meCaloPreShToF[i] = 0;
    meCaloHcal[i] = 0;
    meCaloHcalE[i] = 0;
    meCaloHcalToF[i] = 0;
    meTrackerPx[i] = 0;
    meTrackerSi[i] = 0;
    meMuon[i] = 0;
    meMuonDtToF[i] = 0;
    meMuonCscToF[i] = 0;
    meMuonRpcFToF[i] = 0;
    meMuonRpcBToF[i] = 0;
  }
  meGeantTrkPt = 0;
  meGeantTrkE = 0;
  meCaloEcalPhi = 0;
  meCaloEcalEta = 0;
  meCaloPreShPhi = 0;
  meCaloPreShEta = 0;
  meCaloHcalPhi = 0;
  meCaloHcalEta = 0;
  meTrackerPxPhi = 0;
  meTrackerPxEta = 0;
  meTrackerPxBToF = 0;
  meTrackerPxBR = 0;
  meTrackerPxFToF = 0;
  meTrackerPxFZ = 0;
  meTrackerSiPhi = 0;
  meTrackerSiEta = 0;
  meTrackerSiBToF = 0;
  meTrackerSiBR = 0;
  meTrackerSiFToF = 0;
  meTrackerSiFZ = 0;
  meMuonPhi = 0;
  meMuonEta = 0;
  meMuonDtR = 0;
  meMuonCscZ = 0;
  meMuonRpcBR = 0;
  meMuonRpcFZ = 0;

  //create histograms
  Char_t hname[200];
  Char_t htitle[200];
  if (dbe) {

    // MCGeant
    dbe->setCurrentFolder("GlobalHitsV/MCGeant");
    sprintf(hname,"hMCRGP1");
    sprintf(htitle,"RawGenParticles");
    meMCRGP[0] = dbe->book1D(hname,htitle,100,0.,5000.);
    sprintf(hname,"hMCRGP2");
    meMCRGP[1] = dbe->book1D(hname,htitle,100,0.,500.);  
    for (Int_t i = 0; i < 2; ++i) {
      meMCRGP[i]->setAxisTitle("Number of Raw Generated Particles",1);
      meMCRGP[i]->setAxisTitle("Count",2);
    }

    sprintf(hname,"hMCG4Vtx1");
    sprintf(htitle,"G4 Vertices");
    meMCG4Vtx[0] = dbe->book1D(hname,htitle,100,0.,50000.);
    sprintf(hname,"hMCG4Vtx2");
    meMCG4Vtx[1] = dbe->book1D(hname,htitle,100,-0.5,99.5); 
    for (Int_t i = 0; i < 2; ++i) {
      meMCG4Vtx[i]->setAxisTitle("Number of Vertices",1);
      meMCG4Vtx[i]->setAxisTitle("Count",2);
    }

    sprintf(hname,"hMCG4Trk1");
    sprintf(htitle,"G4 Tracks");
    meMCG4Trk[0] = dbe->book1D(hname,htitle,150,0.,15000.);
    sprintf(hname,"hMCG4Trk2");
    meMCG4Trk[1] = dbe->book1D(hname,htitle,150,-0.5,99.5);    
    for (Int_t i = 0; i < 2; ++i) {
      meMCG4Trk[i]->setAxisTitle("Number of Tracks",1);
      meMCG4Trk[i]->setAxisTitle("Count",2);
    }

    sprintf(hname,"hGeantVtxX1");
    sprintf(htitle,"Geant vertex x/micrometer");
    meGeantVtxX[0] = dbe->book1D(hname,htitle,100,-8000000.,8000000.);
    sprintf(hname,"hGeantVtxX2");
    meGeantVtxX[1] = dbe->book1D(hname,htitle,100,-50.,50.); 
    for (Int_t i = 0; i < 2; ++i) {
      meGeantVtxX[i]->setAxisTitle("x of Vertex (um)",1);
      meGeantVtxX[i]->setAxisTitle("Count",2);
    }

    sprintf(hname,"hGeantVtxY1");
    sprintf(htitle,"Geant vertex y/micrometer");
    meGeantVtxY[0] = dbe->book1D(hname,htitle,100,-8000000,8000000.);
    sprintf(hname,"hGeantVtxY2");
    meGeantVtxY[1] = dbe->book1D(hname,htitle,100,-50.,50.); 
    for (Int_t i = 0; i < 2; ++i) {
      meGeantVtxY[i]->setAxisTitle("y of Vertex (um)",1);
      meGeantVtxY[i]->setAxisTitle("Count",2);
    }

    sprintf(hname,"hGeantVtxZ1");
    sprintf(htitle,"Geant vertex z/millimeter");
    meGeantVtxZ[0] = dbe->book1D(hname,htitle,100,-11000.,11000.);
    sprintf(hname,"hGeantVtxZ2");
    meGeantVtxZ[1] = dbe->book1D(hname,htitle,100,-250.,250.);
    for (Int_t i = 0; i < 2; ++i) {
      meGeantVtxZ[i]->setAxisTitle("z of Vertex (mm)",1);
      meGeantVtxZ[i]->setAxisTitle("Count",2);
    }

    sprintf(hname,"hGeantTrkPt");
    sprintf(htitle,"Geant track pt/GeV");
    meGeantTrkPt = dbe->book1D(hname,htitle,100,0.,200.);
    meGeantTrkPt->setAxisTitle("pT of Track (GeV)",1);
    meGeantTrkPt->setAxisTitle("Count",2);

    sprintf(hname,"hGeantTrkE");
    sprintf(htitle,"Geant track E/GeV");
    meGeantTrkE = dbe->book1D(hname,htitle,100,0.,5000.);
    meGeantTrkE->setAxisTitle("E of Track (GeV)",1);
    meGeantTrkE->setAxisTitle("Count",2);

    // ECal
    dbe->setCurrentFolder("GlobalHitsV/ECals");
    sprintf(hname,"hCaloEcal1");
    sprintf(htitle,"Ecal hits");
    meCaloEcal[0] = dbe->book1D(hname,htitle,100,0.,10000.);
    sprintf(hname,"hCaloEcal2");
    meCaloEcal[1] = dbe->book1D(hname,htitle,100,-0.5,99.5);

    sprintf(hname,"hCaloEcalE1");
    sprintf(htitle,"Ecal hits, energy/GeV");
    meCaloEcalE[0] = dbe->book1D(hname,htitle,100,0.,10.);
    sprintf(hname,"hCaloEcalE2");
    meCaloEcalE[1] = dbe->book1D(hname,htitle,100,0.,0.1);

    sprintf(hname,"hCaloEcalToF1");
    sprintf(htitle,"Ecal hits, ToF/ns");
    meCaloEcalToF[0] = dbe->book1D(hname,htitle,100,0.,1000.);
    sprintf(hname,"hCaloEcalToF2");
    meCaloEcalToF[1] = dbe->book1D(hname,htitle,100,0.,100.);
 
    for (Int_t i = 0; i < 2; ++i) {
      meCaloEcal[i]->setAxisTitle("Number of Hits",1);
      meCaloEcal[i]->setAxisTitle("Count",2);
      meCaloEcalE[i]->setAxisTitle("Energy of Hits (GeV)",1);
      meCaloEcalE[i]->setAxisTitle("Count",2);
      meCaloEcalToF[i]->setAxisTitle("Time of Flight of Hits (ns)",1);
      meCaloEcalToF[i]->setAxisTitle("Count",2);
    }

    sprintf(hname,"hCaloEcalPhi");
    sprintf(htitle,"Ecal hits, phi/rad");
    meCaloEcalPhi = dbe->book1D(hname,htitle,100,-3.2,3.2);
    meCaloEcalPhi->setAxisTitle("Phi of Hits (rad)",1);
    meCaloEcalPhi->setAxisTitle("Count",2);

    sprintf(hname,"hCaloEcalEta");
    sprintf(htitle,"Ecal hits, eta");
    meCaloEcalEta = dbe->book1D(hname,htitle,100,-5.5,5.5);
    meCaloEcalEta->setAxisTitle("Eta of Hits",1);
    meCaloEcalEta->setAxisTitle("Count",2);

    sprintf(hname,"hCaloPreSh1");
    sprintf(htitle,"PreSh hits");
    meCaloPreSh[0] = dbe->book1D(hname,htitle,100,0.,10000.);
    sprintf(hname,"hCaloPreSh2");
    meCaloPreSh[1] = dbe->book1D(hname,htitle,100,-0.5,99.5);

    sprintf(hname,"hCaloPreShE1");
    sprintf(htitle,"PreSh hits, energy/GeV");
    meCaloPreShE[0] = dbe->book1D(hname,htitle,100,0.,10.);
    sprintf(hname,"hCaloPreShE2");
    meCaloPreShE[1] = dbe->book1D(hname,htitle,100,0.,0.1);

    sprintf(hname,"hCaloPreShToF1");
    sprintf(htitle,"PreSh hits, ToF/ns");
    meCaloPreShToF[0] = dbe->book1D(hname,htitle,100,0.,1000.);
    sprintf(hname,"hCaloPreShToF2");
    meCaloPreShToF[1] = dbe->book1D(hname,htitle,100,0.,100.);

    for (Int_t i = 0; i < 2; ++i) {
      meCaloPreSh[i]->setAxisTitle("Number of Hits",1);
      meCaloPreSh[i]->setAxisTitle("Count",2);
      meCaloPreShE[i]->setAxisTitle("Energy of Hits (GeV)",1);
      meCaloPreShE[i]->setAxisTitle("Count",2);
      meCaloPreShToF[i]->setAxisTitle("Time of Flight of Hits (ns)",1);
      meCaloPreShToF[i]->setAxisTitle("Count",2);
    }

    sprintf(hname,"hCaloPreShPhi");
    sprintf(htitle,"PreSh hits, phi/rad");
    meCaloPreShPhi = dbe->book1D(hname,htitle,100,-3.2,3.2);
    meCaloPreShPhi->setAxisTitle("Phi of Hits (rad)",1);
    meCaloPreShPhi->setAxisTitle("Count",2);

    sprintf(hname,"hCaloPreShEta");
    sprintf(htitle,"PreSh hits, eta");
    meCaloPreShEta = dbe->book1D(hname,htitle,100,-5.5,5.5);
    meCaloPreShEta->setAxisTitle("Eta of Hits",1);
    meCaloPreShEta->setAxisTitle("Count",2);

    // Hcal
    dbe->setCurrentFolder("GlobalHitsV/HCals");
    sprintf(hname,"hCaloHcal1");
    sprintf(htitle,"Hcal hits");
    meCaloHcal[0] = dbe->book1D(hname,htitle,100,0.,10000.);
    sprintf(hname,"hCaloHcal2");
    meCaloHcal[1] = dbe->book1D(hname,htitle,100,-0.5,99.5);

    sprintf(hname,"hCaloHcalE1");
    sprintf(htitle,"Hcal hits, energy/GeV");
    meCaloHcalE[0] = dbe->book1D(hname,htitle,100,0.,10.);
    sprintf(hname,"hCaloHcalE2");
    meCaloHcalE[1] = dbe->book1D(hname,htitle,100,0.,0.1);

    sprintf(hname,"hCaloHcalToF1");
    sprintf(htitle,"Hcal hits, ToF/ns");
    meCaloHcalToF[0] = dbe->book1D(hname,htitle,100,0.,1000.);
    sprintf(hname,"hCaloHcalToF2");
    meCaloHcalToF[1] = dbe->book1D(hname,htitle,100,0.,100.);

    for (Int_t i = 0; i < 2; ++i) {
      meCaloHcal[i]->setAxisTitle("Number of Hits",1);
      meCaloHcal[i]->setAxisTitle("Count",2);
      meCaloHcalE[i]->setAxisTitle("Energy of Hits (GeV)",1);
      meCaloHcalE[i]->setAxisTitle("Count",2);
      meCaloHcalToF[i]->setAxisTitle("Time of Flight of Hits (ns)",1);
      meCaloHcalToF[i]->setAxisTitle("Count",2);
    }

    sprintf(hname,"hCaloHcalPhi");
    sprintf(htitle,"Hcal hits, phi/rad");
    meCaloHcalPhi = dbe->book1D(hname,htitle,100,-3.2,3.2);
    meCaloHcalPhi->setAxisTitle("Phi of Hits (rad)",1);
    meCaloHcalPhi->setAxisTitle("Count",2);

    sprintf(hname,"hCaloHcalEta");
    sprintf(htitle,"Hcal hits, eta");
    meCaloHcalEta = dbe->book1D(hname,htitle,100,-5.5,5.5);
    meCaloHcalEta->setAxisTitle("Eta of Hits",1);
    meCaloHcalEta->setAxisTitle("Count",2);
    
    // SiPixels
    dbe->setCurrentFolder("GlobalHitsV/SiPixels");
    sprintf(hname,"hTrackerPx1");
    sprintf(htitle,"Pixel hits");
    meTrackerPx[0] = dbe->book1D(hname,htitle,100,0.,10000.);
    sprintf(hname,"hTrackerPx2");
    meTrackerPx[1] = dbe->book1D(hname,htitle,100,-0.5,99.5);
    for (Int_t i = 0; i < 2; ++i) {
      meTrackerPx[i]->setAxisTitle("Number of Pixel Hits",1);
      meTrackerPx[i]->setAxisTitle("Count",2);
    }

    sprintf(hname,"hTrackerPxPhi");
    sprintf(htitle,"Pixel hits phi/rad");
    meTrackerPxPhi = dbe->book1D(hname,htitle,100,-3.2,3.2);
    meTrackerPxPhi->setAxisTitle("Phi of Hits (rad)",1);
    meTrackerPxPhi->setAxisTitle("Count",2);

    sprintf(hname,"hTrackerPxEta");
    sprintf(htitle,"Pixel hits eta");
    meTrackerPxEta = dbe->book1D(hname,htitle,100,-3.5,3.5);
    meTrackerPxEta->setAxisTitle("Eta of Hits",1);
    meTrackerPxEta->setAxisTitle("Count",2);

    sprintf(hname,"hTrackerPxBToF");
    sprintf(htitle,"Pixel barrel hits, ToF/ns");
    meTrackerPxBToF = dbe->book1D(hname,htitle,100,0.,40.);
    meTrackerPxBToF->setAxisTitle("Time of Flight of Hits (ns)",1);
    meTrackerPxBToF->setAxisTitle("Count",2);

    sprintf(hname,"hTrackerPxBR");
    sprintf(htitle,"Pixel barrel hits, R/cm");
    meTrackerPxBR = dbe->book1D(hname,htitle,100,0.,50.);
    meTrackerPxBR->setAxisTitle("R of Hits (cm)",1);
    meTrackerPxBR->setAxisTitle("Count",2);

    sprintf(hname,"hTrackerPxFToF");
    sprintf(htitle,"Pixel forward hits, ToF/ns");
    meTrackerPxFToF = dbe->book1D(hname,htitle,100,0.,50.);
    meTrackerPxFToF->setAxisTitle("Time of Flight of Hits (ns)",1);
    meTrackerPxFToF->setAxisTitle("Count",2);

    sprintf(hname,"hTrackerPxFZ");
    sprintf(htitle,"Pixel forward hits, Z/cm");
    meTrackerPxFZ = dbe->book1D(hname,htitle,200,-100.,100.);
    meTrackerPxFZ->setAxisTitle("Z of Hits (cm)",1);
    meTrackerPxFZ->setAxisTitle("Count",2);

    // SiStrips
    dbe->setCurrentFolder("GlobalHitsV/SiPixels");
    sprintf(hname,"hTrackerSi1");
    sprintf(htitle,"Silicon hits");
    meTrackerSi[0] = dbe->book1D(hname,htitle,100,0.,10000.);
    sprintf(hname,"hTrackerSi2");
    meTrackerSi[1] = dbe->book1D(hname,htitle,100,-0.5,99.5);
    for (Int_t i = 0; i < 2; ++i) { 
      meTrackerSi[i]->setAxisTitle("Number of Silicon Hits",1);
      meTrackerSi[i]->setAxisTitle("Count",2);
    }

    sprintf(hname,"hTrackerSiPhi");
    sprintf(htitle,"Silicon hits phi/rad");
    meTrackerSiPhi = dbe->book1D(hname,htitle,100,-3.2,3.2);
    meTrackerSiPhi->setAxisTitle("Phi of Hits (rad)",1);
    meTrackerSiPhi->setAxisTitle("Count",2);

    sprintf(hname,"hTrackerSiEta");
    sprintf(htitle,"Silicon hits eta");
    meTrackerSiEta = dbe->book1D(hname,htitle,100,-3.5,3.5);
    meTrackerSiEta->setAxisTitle("Eta of Hits",1);
    meTrackerSiEta->setAxisTitle("Count",2);

    sprintf(hname,"hTrackerSiBToF");
    sprintf(htitle,"Silicon barrel hits, ToF/ns");
    meTrackerSiBToF = dbe->book1D(hname,htitle,100,0.,50.);
    meTrackerSiBToF->setAxisTitle("Time of Flight of Hits (ns)",1);
    meTrackerSiBToF->setAxisTitle("Count",2);

    sprintf(hname,"hTrackerSiBR");
    sprintf(htitle,"Silicon barrel hits, R/cm");
    meTrackerSiBR = dbe->book1D(hname,htitle,100,0.,200.);
    meTrackerSiBR->setAxisTitle("R of Hits (cm)",1);
    meTrackerSiBR->setAxisTitle("Count",2);

    sprintf(hname,"hTrackerSiFToF");
    sprintf(htitle,"Silicon forward hits, ToF/ns");
    meTrackerSiFToF = dbe->book1D(hname,htitle,100,0.,75.);
    meTrackerSiFToF->setAxisTitle("Time of Flight of Hits (ns)",1);
    meTrackerSiFToF->setAxisTitle("Count",2);

    sprintf(hname,"hTrackerSiFZ");
    sprintf(htitle,"Silicon forward hits, Z/cm");
    meTrackerSiFZ = dbe->book1D(hname,htitle,200,-300.,300.);
    meTrackerSiFZ->setAxisTitle("Z of Hits (cm)",1);
    meTrackerSiFZ->setAxisTitle("Count",2);

    // muon
    dbe->setCurrentFolder("GlobalHitsV/Muons");
    sprintf(hname,"hMuon1");
    sprintf(htitle,"Muon hits");
    meMuon[0] = dbe->book1D(hname,htitle,100,0.,10000.);
    sprintf(hname,"hMuon2");
    meMuon[1] = dbe->book1D(hname,htitle,100,-0.5,99.5);
    for (Int_t i = 0; i < 2; ++i) { 
      meMuon[i]->setAxisTitle("Number of Muon Hits",1);
      meMuon[i]->setAxisTitle("Count",2);
    }  

    sprintf(hname,"hMuonPhi");
    sprintf(htitle,"Muon hits phi/rad");
    meMuonPhi = dbe->book1D(hname,htitle,100,-3.2,3.2);
    meMuonPhi->setAxisTitle("Phi of Hits (rad)",1);
    meMuonPhi->setAxisTitle("Count",2);

    sprintf(hname,"hMuonEta");
    sprintf(htitle,"Muon hits eta");
    meMuonEta = dbe->book1D(hname,htitle,100,-3.5,3.5);
    meMuonEta->setAxisTitle("Eta of Hits",1);
    meMuonEta->setAxisTitle("Count",2);

    sprintf(hname,"hMuonCscToF1");
    sprintf(htitle,"Muon CSC hits, ToF/ns");
    meMuonCscToF[0] = dbe->book1D(hname,htitle,100,0.,250.);
    sprintf(hname,"hMuonCscToF2");
    meMuonCscToF[1] = dbe->book1D(hname,htitle,100,0.,50.);
    for (Int_t i = 0; i < 2; ++i) {   
      meMuonCscToF[i]->setAxisTitle("Time of Flight of Hits (ns)",1);
      meMuonCscToF[i]->setAxisTitle("Count",2);
    }  

    sprintf(hname,"hMuonCscZ");
    sprintf(htitle,"Muon CSC hits, Z/cm");
    meMuonCscZ = dbe->book1D(hname,htitle,200,-1500.,1500.);
    meMuonCscZ->setAxisTitle("Z of Hits (cm)",1);
    meMuonCscZ->setAxisTitle("Count",2);

    sprintf(hname,"hMuonDtToF1");
    sprintf(htitle,"Muon DT hits, ToF/ns");
    meMuonDtToF[0] = dbe->book1D(hname,htitle,100,0.,250.);
    sprintf(hname,"hMuonDtToF2");
    meMuonDtToF[1] = dbe->book1D(hname,htitle,100,0.,50.);
    for (Int_t i = 0; i < 2; ++i) {   
      meMuonDtToF[i]->setAxisTitle("Time of Flight of Hits (ns)",1);
      meMuonDtToF[i]->setAxisTitle("Count",2);
    } 

    sprintf(hname,"hMuonDtR");
    sprintf(htitle,"Muon DT hits, R/cm");
    meMuonDtR = dbe->book1D(hname,htitle,100,0.,1500.); 
    meMuonDtR->setAxisTitle("R of Hits (cm)",1);
    meMuonDtR->setAxisTitle("Count",2);

    sprintf(hname,"hMuonRpcFToF1");
    sprintf(htitle,"Muon RPC forward hits, ToF/ns");
    meMuonRpcFToF[0] = dbe->book1D(hname,htitle,100,0.,250.);
    sprintf(hname,"hMuonRpcFToF2_4305");
    meMuonRpcFToF[1] = dbe->book1D(hname,htitle,100,0.,50.);
    for (Int_t i = 0; i < 2; ++i) {   
      meMuonRpcFToF[i]->setAxisTitle("Time of Flight of Hits (ns)",1);
      meMuonRpcFToF[i]->setAxisTitle("Count",2);
    }  

    sprintf(hname,"hMuonRpcFZ");
    sprintf(htitle,"Muon RPC forward hits, Z/cm");
    meMuonRpcFZ = dbe->book1D(hname,htitle,201,-1500.,1500.);
    meMuonRpcFZ->setAxisTitle("Z of Hits (cm)",1);
    meMuonRpcFZ->setAxisTitle("Count",2);

    sprintf(hname,"hMuonRpcBToF1");
    sprintf(htitle,"Muon RPC barrel hits, ToF/ns");
    meMuonRpcBToF[0] = dbe->book1D(hname,htitle,100,0.,250.);
    sprintf(hname,"hMuonRpcBToF2");
    meMuonRpcBToF[1] = dbe->book1D(hname,htitle,100,0.,50.);
    for (Int_t i = 0; i < 2; ++i) {   
      meMuonRpcBToF[i]->setAxisTitle("Time of Flight of Hits (ns)",1);
      meMuonRpcBToF[i]->setAxisTitle("Count",2);
    }

    sprintf(hname,"hMuonRpcBR");
    sprintf(htitle,"Muon RPC barrel hits, R/cm");
    meMuonRpcBR = dbe->book1D(hname,htitle,100,0.,1500.);
    meMuonRpcBR->setAxisTitle("R of Hits (cm)",1);
    meMuonRpcBR->setAxisTitle("Count",2); 
  }
}

GlobalHitsHistogrammer::~GlobalHitsHistogrammer() 
{
  if (doOutput)
    if (outputfile.size() != 0 && dbe) dbe->save(outputfile);
}

void GlobalHitsHistogrammer::beginJob( void )
{
  return;
}

void GlobalHitsHistogrammer::endJob()
{
  std::string MsgLoggerCat = "GlobalHitsHistogrammer_endJob";
  if (verbosity >= 0)
    edm::LogInfo(MsgLoggerCat) 
      << "Terminating having processed " << count << " events.";
  return;
}

void GlobalHitsHistogrammer::analyze(const edm::Event& iEvent, 
				 const edm::EventSetup& iSetup)
{
  std::string MsgLoggerCat = "GlobalHitsHistogrammer_analyze";

  // keep track of number of events processed
  ++count;

  // get event id information
  int nrun = iEvent.id().run();
  int nevt = iEvent.id().event();

  if (verbosity > 0) {
    edm::LogInfo(MsgLoggerCat)
      << "Processing run " << nrun << ", event " << nevt
      << " (" << count << " events total)";
  } else if (verbosity == 0) {
    if (nevt%frequency == 0 || nevt == 1) {
      edm::LogInfo(MsgLoggerCat)
	<< "Processing run " << nrun << ", event " << nevt
	<< " (" << count << " events total)";
    }
  }

  // look at information available in the event
  if (getAllProvenances) {

    std::vector<const edm::Provenance*> AllProv;
    iEvent.getAllProvenance(AllProv);

    if (verbosity >= 0)
      edm::LogInfo(MsgLoggerCat)
	<< "Number of Provenances = " << AllProv.size();

    if (printProvenanceInfo && (verbosity >= 0)) {
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
      eventout += "\n       ******************************\n";
      edm::LogInfo(MsgLoggerCat) << eventout << "\n";
      printProvenanceInfo = false;
    }
    getAllProvenances = false;
  }

  // fill histograms
  edm::Handle<PGlobalSimHit> srcGlobalHits;
  iEvent.getByLabel(GlobalHitSrc_,srcGlobalHits);
  if (!srcGlobalHits.isValid()) {
    edm::LogWarning(MsgLoggerCat)
      << "Unable to find PGlobalSimHit in event!";
    return;
  }

  nPxlBrlHits = srcGlobalHits->getnPxlBrlHits();
  nPxlFwdHits = srcGlobalHits->getnPxlFwdHits();
  nPxlHits = nPxlBrlHits + nPxlFwdHits;
  nSiBrlHits = srcGlobalHits->getnSiBrlHits();
  nSiFwdHits = srcGlobalHits->getnSiFwdHits();
  nSiHits = nSiBrlHits + nSiFwdHits;    
  nMuonDtHits = srcGlobalHits->getnMuonDtHits();
  nMuonCscHits = srcGlobalHits->getnMuonCscHits();
  nMuonRpcBrlHits = srcGlobalHits->getnMuonRpcBrlHits();
  nMuonRpcFwdHits = srcGlobalHits->getnMuonRpcFwdHits();
  nMuonHits = nMuonDtHits + nMuonCscHits + nMuonRpcBrlHits + nMuonRpcFwdHits;

  for (Int_t i = 0; i < 2; ++i) {
    meMCRGP[i]->Fill((float)srcGlobalHits->getnRawGenPart());
    meMCG4Vtx[i]->Fill((float)srcGlobalHits->getnG4Vtx());
    meMCG4Trk[i]->Fill((float)srcGlobalHits->getnG4Trk());
    meCaloEcal[i]->Fill((float)srcGlobalHits->getnECalHits());
    meCaloPreSh[i]->Fill((float)srcGlobalHits->getnPreShHits());
    meCaloHcal[i]->Fill((float)srcGlobalHits->getnHCalHits());
    meTrackerPx[i]->Fill((float)nPxlHits);
    meTrackerSi[i]->Fill((float)nSiHits);
    meMuon[i]->Fill((float)nMuonHits);
  }

  // get G4Vertex info
  std::vector<PGlobalSimHit::Vtx> G4Vtx = srcGlobalHits->getG4Vtx();
  for (unsigned int i = 0; i < G4Vtx.size(); ++i) {
    for (int j = 0; j < 2; ++j) {
      meGeantVtxX[j]->Fill(G4Vtx[i].x);
      meGeantVtxY[j]->Fill(G4Vtx[i].y);
      meGeantVtxZ[j]->Fill(G4Vtx[i].z);
    }
  }
  
  // get G4Track info
  std::vector<PGlobalSimHit::Trk> G4Trk = srcGlobalHits->getG4Trk();
  for (unsigned int i = 0; i < G4Trk.size(); ++i) {
    meGeantTrkPt->Fill(G4Trk[i].pt);
    meGeantTrkE->Fill(G4Trk[i].e);
  }
  
  // get Ecal info
  std::vector<PGlobalSimHit::CalHit> ECalHits = 
    srcGlobalHits->getECalHits();
  for (unsigned int i = 0; i < ECalHits.size(); ++i) {
    for (Int_t j = 0; j < 2; ++j) {
	meCaloEcalE[j]->Fill(ECalHits[i].e);
	meCaloEcalToF[j]->Fill(ECalHits[i].tof);
    }
    meCaloEcalPhi->Fill(ECalHits[i].phi);
    meCaloEcalEta->Fill(ECalHits[i].eta);
  }
  
  // get PreShower info
  std::vector<PGlobalSimHit::CalHit> PreShHits = 
    srcGlobalHits->getPreShHits();
  for (unsigned int i = 0; i < PreShHits.size(); ++i) {
    for (Int_t j = 0; j < 2; ++j) {
      meCaloPreShE[j]->Fill(PreShHits[i].e);
      meCaloPreShToF[j]->Fill(PreShHits[i].tof);
    }
    meCaloPreShPhi->Fill(PreShHits[i].phi);
    meCaloPreShEta->Fill(PreShHits[i].eta);
  }
  
  // get Hcal info
  std::vector<PGlobalSimHit::CalHit> HCalHits = 
    srcGlobalHits->getHCalHits();
  for (unsigned int i = 0; i < HCalHits.size(); ++i) {
    for (Int_t j = 0; j < 2; ++j) {
      meCaloHcalE[j]->Fill(HCalHits[i].e);
      meCaloHcalToF[j]->Fill(HCalHits[i].tof);
    }
    meCaloHcalPhi->Fill(HCalHits[i].phi);
    meCaloHcalEta->Fill(HCalHits[i].eta);
  }
  
  // get Pixel Barrel info
  std::vector<PGlobalSimHit::BrlHit> PxlBrlHits = 
    srcGlobalHits->getPxlBrlHits();
  for (unsigned int i = 0; i < PxlBrlHits.size(); ++i) {
    meTrackerPxPhi->Fill(PxlBrlHits[i].phi);
    meTrackerPxEta->Fill(PxlBrlHits[i].eta);
    meTrackerPxBToF->Fill(PxlBrlHits[i].tof);
    meTrackerPxBR->Fill(PxlBrlHits[i].r);
  }
  
  // get Pixel Forward info
  std::vector<PGlobalSimHit::FwdHit> PxlFwdHits = 
    srcGlobalHits->getPxlFwdHits();
  for (unsigned int i = 0; i < PxlFwdHits.size(); ++i) {
    meTrackerPxPhi->Fill(PxlFwdHits[i].phi);
    meTrackerPxEta->Fill(PxlFwdHits[i].eta);
    meTrackerPxFToF->Fill(PxlFwdHits[i].tof);
    meTrackerPxFZ->Fill(PxlFwdHits[i].z);
  }
  
  // get Strip Barrel info
  std::vector<PGlobalSimHit::BrlHit> SiBrlHits = 
    srcGlobalHits->getSiBrlHits();
  for (unsigned int i = 0; i < SiBrlHits.size(); ++i) {
    meTrackerSiPhi->Fill(SiBrlHits[i].phi);
    meTrackerSiEta->Fill(SiBrlHits[i].eta);
    meTrackerSiBToF->Fill(SiBrlHits[i].tof);
    meTrackerSiBR->Fill(SiBrlHits[i].r);
  }
  
  // get Strip Forward info
  std::vector<PGlobalSimHit::FwdHit> SiFwdHits = 
    srcGlobalHits->getSiFwdHits();
  for (unsigned int i = 0; i < SiFwdHits.size(); ++i) {
    meTrackerSiPhi->Fill(SiFwdHits[i].phi);
    meTrackerSiEta->Fill(SiFwdHits[i].eta);
    meTrackerSiFToF->Fill(SiFwdHits[i].tof);
    meTrackerSiFZ->Fill(SiFwdHits[i].z);
  }
  
  // get Muon CSC info
  std::vector<PGlobalSimHit::FwdHit> MuonCscHits = 
    srcGlobalHits->getMuonCscHits();
  for (unsigned int i = 0; i < MuonCscHits.size(); ++i) {
    meMuonPhi->Fill(MuonCscHits[i].phi);
    meMuonEta->Fill(MuonCscHits[i].eta);
    for (Int_t j = 0; j < 2; ++j) {
      meMuonCscToF[j]->Fill(MuonCscHits[i].tof);
    }
    meMuonCscZ->Fill(MuonCscHits[i].z);
  }    
  
  // get Muon DT info
  std::vector<PGlobalSimHit::BrlHit> MuonDtHits = 
    srcGlobalHits->getMuonDtHits();
  for (unsigned int i = 0; i < MuonDtHits.size(); ++i) {
    meMuonPhi->Fill(MuonDtHits[i].phi);
    meMuonEta->Fill(MuonDtHits[i].eta);
    for (Int_t j = 0; j < 2; ++j) {
      meMuonDtToF[j]->Fill(MuonDtHits[i].tof);
    }
    meMuonDtR->Fill(MuonDtHits[i].r);
  }
  
  // get Muon RPC forward info
  std::vector<PGlobalSimHit::FwdHit> MuonRpcFwdHits = 
    srcGlobalHits->getMuonRpcFwdHits();
  for (unsigned int i = 0; i < MuonRpcFwdHits.size(); ++i) {
    meMuonPhi->Fill(MuonRpcFwdHits[i].phi);
    meMuonEta->Fill(MuonRpcFwdHits[i].eta);
    for (Int_t j = 0; j < 2; ++j) {
      meMuonRpcFToF[j]->Fill(MuonRpcFwdHits[i].tof);
    }
    meMuonRpcFZ->Fill(MuonRpcFwdHits[i].z);
  }    
  
  // get Muon RPC barrel info
  std::vector<PGlobalSimHit::BrlHit> MuonRpcBrlHits = 
    srcGlobalHits->getMuonRpcBrlHits();
  for (unsigned int i = 0; i < MuonRpcBrlHits.size(); ++i) {
    meMuonPhi->Fill(MuonRpcBrlHits[i].phi);
    meMuonEta->Fill(MuonRpcBrlHits[i].eta);
    for (Int_t j = 0; j < 2; ++j) {
      meMuonRpcBToF[j]->Fill(MuonRpcBrlHits[i].tof);
    }
    meMuonRpcBR->Fill(MuonRpcBrlHits[i].r);
  }   
  
  return;
}


