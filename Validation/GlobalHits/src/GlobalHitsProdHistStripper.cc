/** \file GlobalHitsProdHistStripper.cc
 *  
 *  See header file for description of class
 *
 *  $Date: 2007/10/10 21:23:59 $
 *  $Revision: 1.1 $
 *  \author M. Strang SUNY-Buffalo
 */

#include "Validation/GlobalHits/interface/GlobalHitsProdHistStripper.h"

GlobalHitsProdHistStripper::GlobalHitsProdHistStripper(const 
						       edm::ParameterSet& 
						       iPSet) 
  : fName(""), verbosity(0), frequency(0), vtxunit(0), 
    getAllProvenances(false), printProvenanceInfo(false), outputfile(""),
    count(0)
{
  std::string MsgLoggerCat = 
    "GlobalHitsProdHistStripper_GlobalHitsProdHistStripper";

  // get information from parameter set
  fName = iPSet.getUntrackedParameter<std::string>("Name");
  verbosity = iPSet.getUntrackedParameter<int>("Verbosity");
  frequency = iPSet.getUntrackedParameter<int>("Frequency");
  vtxunit = iPSet.getUntrackedParameter<int>("VtxUnit");
  outputfile = iPSet.getParameter<std::string>("OutputFile");
  edm::ParameterSet m_Prov =
    iPSet.getParameter<edm::ParameterSet>("ProvenanceLookup");
  getAllProvenances = 
    m_Prov.getUntrackedParameter<bool>("GetAllProvenances");
  printProvenanceInfo = 
    m_Prov.getUntrackedParameter<bool>("PrintProvenanceInfo");


  // use value of first digit to determine default output level (inclusive)
  // 0 is none, 1 is basic, 2 is fill output, 3 is gather output
  verbosity %= 10;

  // get dqm info
  dbe = 0;
  dbe = edm::Service<DaqMonitorBEInterface>().operator->();
  if (dbe) {
    if (verbosity >= 0 ) {
      dbe->setVerbose(1);
    } else {
      dbe->setVerbose(0);
    }
  }

  if (dbe) {
    if (verbosity >= 0 ) dbe->showDirStructure();
  }

  /*
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
    dbe->setCurrentFolder("MCGeant");
    sprintf(hname,"hMCRGP_1003");
    sprintf(htitle,"RawGenParticles");
    meMCRGP[0] = dbe->book1D(hname,htitle,100,0.,5000.);
    sprintf(hname,"hMCRGP_1013");
    meMCRGP[1] = dbe->book1D(hname,htitle,100,0.,500.);  
    for (Int_t i = 0; i < 2; ++i) {
      meMCRGP[i]->setAxisTitle("Number of Raw Generated Particles",1);
      meMCRGP[i]->setAxisTitle("Count",2);
    }
    monitorElements["hMCRGP1"] = meMCRGP[0];
    monitorElements["hMCRGP2"] = meMCRGP[1];

    sprintf(hname,"hMCG4Vtx_1001");
    sprintf(htitle,"G4 Vertices");
    meMCG4Vtx[0] = dbe->book1D(hname,htitle,100,0.,50000.);
    sprintf(hname,"hMCG4Vtx2_1011");
    meMCG4Vtx[1] = dbe->book1D(hname,htitle,100,-0.5,99.5); 
    for (Int_t i = 0; i < 2; ++i) {
      meMCG4Vtx[i]->setAxisTitle("Number of Vertices",1);
      meMCG4Vtx[i]->setAxisTitle("Count",2);
    }
    monitorElements["hMCG4Vtx1"] = meMCG4Vtx[0];
    monitorElements["hMCG4Vtx2"] = meMCG4Vtx[1];

    sprintf(hname,"hMCG4Trk_1002");
    sprintf(htitle,"G4 Tracks");
    meMCG4Trk[0] = dbe->book1D(hname,htitle,150,0.,15000.);
    sprintf(hname,"hMCG4Trk2_1012");
    meMCG4Trk[1] = dbe->book1D(hname,htitle,150,-0.5,99.5);    
    for (Int_t i = 0; i < 2; ++i) {
      meMCG4Trk[i]->setAxisTitle("Number of Tracks",1);
      meMCG4Trk[i]->setAxisTitle("Count",2);
    }
    monitorElements["hMCG4Trk1"] = meMCG4Trk[0];
    monitorElements["hMCG4Trk2"] = meMCG4Trk[1];

    sprintf(hname,"hGeantVtxX2_1104");
    sprintf(htitle,"Geant vertex x/micrometer");
    meGeantVtxX[0] = dbe->book1D(hname,htitle,100,-8000000.,8000000.);
    sprintf(hname,"hGeantVtxX_1101");
    meGeantVtxX[1] = dbe->book1D(hname,htitle,100,-50.,50.); 
    for (Int_t i = 0; i < 2; ++i) {
      meGeantVtxX[i]->setAxisTitle("x of Vertex (um)",1);
      meGeantVtxX[i]->setAxisTitle("Count",2);
    }
    monitorElements["hGeantVtxX1"] = meGeantVtxX[0];
    monitorElements["hGeantVtxX2"] = meGeantVtxX[1];

    sprintf(hname,"hGeantVtxY_1105");
    sprintf(htitle,"Geant vertex y/micrometer");
    meGeantVtxY[0] = dbe->book1D(hname,htitle,100,-8000000,8000000.);
    sprintf(hname,"hGeantVtxY_1102");
    meGeantVtxY[1] = dbe->book1D(hname,htitle,100,-50.,50.); 
    for (Int_t i = 0; i < 2; ++i) {
      meGeantVtxY[i]->setAxisTitle("y of Vertex (um)",1);
      meGeantVtxY[i]->setAxisTitle("Count",2);
    }
    monitorElements["hGeantVtxY1"] = meGeantVtxY[0];
    monitorElements["hGeantVtxY2"] = meGeantVtxY[1];

    sprintf(hname,"hGeantVtxZ_1106");
    sprintf(htitle,"Geant vertex z/millimeter");
    meGeantVtxZ[0] = dbe->book1D(hname,htitle,100,-11000.,11000.);
    sprintf(hname,"hGeantVtxZ_1103");
    meGeantVtxZ[1] = dbe->book1D(hname,htitle,100,-250.,250.);
    for (Int_t i = 0; i < 2; ++i) {
      meGeantVtxZ[i]->setAxisTitle("z of Vertex (mm)",1);
      meGeantVtxZ[i]->setAxisTitle("Count",2);
    }
    monitorElements["hGeantVtxZ1"] = meGeantVtxZ[0];
    monitorElements["hGeantVtxZ2"] = meGeantVtxZ[1];

    sprintf(hname,"hGeantTrkPt_1201");
    sprintf(htitle,"Geant track pt/GeV");
    meGeantTrkPt = dbe->book1D(hname,htitle,100,0.,200.);
    meGeantTrkPt->setAxisTitle("pT of Track (GeV)",1);
    meGeantTrkPt->setAxisTitle("Count",2);
    monitorElements["hGeantTrkPt"] = meGeantTrkPt;

    sprintf(hname,"hGeantTrkE_1202");
    sprintf(htitle,"Geant track E/GeV");
    meGeantTrkE = dbe->book1D(hname,htitle,100,0.,5000.);
    meGeantTrkE->setAxisTitle("E of Track (GeV)",1);
    meGeantTrkE->setAxisTitle("Count",2);
    monitorElements["hGeantTrkE"] = meGeantTrkE;

    // ECal
    dbe->setCurrentFolder("ECal");
    sprintf(hname,"hCaloEcal_2101");
    sprintf(htitle,"Ecal hits");
    meCaloEcal[0] = dbe->book1D(hname,htitle,100,0.,10000.);
    sprintf(hname,"hCaloEcal2_2111");
    meCaloEcal[1] = dbe->book1D(hname,htitle,100,-0.5,99.5);
    monitorElements["hCaloEcal1"] = meCaloEcal[0];
    monitorElements["hCaloEcal2"] = meCaloEcal[1];

    sprintf(hname,"hCaloEcalE_2102");
    sprintf(htitle,"Ecal hits, energy/GeV");
    meCaloEcalE[0] = dbe->book1D(hname,htitle,100,0.,10.);
    sprintf(hname,"hCaloEcalE2_2112");
    meCaloEcalE[1] = dbe->book1D(hname,htitle,100,0.,0.1);
    monitorElements["hCaloEcalE1"] = meCaloEcalE[0];
    monitorElements["hCaloEcalE2"] = meCaloEcalE[1];

    sprintf(hname,"hCaloEcalToF_2103");
    sprintf(htitle,"Ecal hits, ToF/ns");
    meCaloEcalToF[0] = dbe->book1D(hname,htitle,100,0.,1000.);
    sprintf(hname,"hCaloEcalToF2_2113");
    meCaloEcalToF[1] = dbe->book1D(hname,htitle,100,0.,100.);
    monitorElements["hCaloEcalToF1"] = meCaloEcalToF[0];
    monitorElements["hCaloEcalToF2"] = meCaloEcalToF[1];
 
    for (Int_t i = 0; i < 2; ++i) {
      meCaloEcal[i]->setAxisTitle("Number of Hits",1);
      meCaloEcal[i]->setAxisTitle("Count",2);
      meCaloEcalE[i]->setAxisTitle("Energy of Hits (GeV)",1);
      meCaloEcalE[i]->setAxisTitle("Count",2);
      meCaloEcalToF[i]->setAxisTitle("Time of Flight of Hits (ns)",1);
      meCaloEcalToF[i]->setAxisTitle("Count",2);
    }

    sprintf(hname,"hCaloEcalPhi_2104");
    sprintf(htitle,"Ecal hits, phi/rad");
    meCaloEcalPhi = dbe->book1D(hname,htitle,100,-3.2,3.2);
    meCaloEcalPhi->setAxisTitle("Phi of Hits (rad)",1);
    meCaloEcalPhi->setAxisTitle("Count",2);
    monitorElements["hCaloEcalPhi"] = meCaloEcalPhi;

    sprintf(hname,"hCaloEcalEta_2105");
    sprintf(htitle,"Ecal hits, eta");
    meCaloEcalEta = dbe->book1D(hname,htitle,100,-5.5,5.5);
    meCaloEcalEta->setAxisTitle("Eta of Hits",1);
    meCaloEcalEta->setAxisTitle("Count",2);
    monitorElements["hCaloEcalEta"] = meCaloEcalEta;

    sprintf(hname,"hCaloPreSh_2201");
    sprintf(htitle,"PreSh hits");
    meCaloPreSh[0] = dbe->book1D(hname,htitle,100,0.,10000.);
    sprintf(hname,"hCaloPreSh2_2211");
    meCaloPreSh[1] = dbe->book1D(hname,htitle,100,-0.5,99.5);
    monitorElements["hCaloPreSh1"] = meCaloPreSh[0];
    monitorElements["hCaloPreSh2"] = meCaloPreSh[1];

    sprintf(hname,"hCaloPreShE_2202");
    sprintf(htitle,"PreSh hits, energy/GeV");
    meCaloPreShE[0] = dbe->book1D(hname,htitle,100,0.,10.);
    sprintf(hname,"hCaloPreShE2_2212");
    meCaloPreShE[1] = dbe->book1D(hname,htitle,100,0.,0.1);

    sprintf(hname,"hCaloPreShToF_2203");
    sprintf(htitle,"PreSh hits, ToF/ns");
    meCaloPreShToF[0] = dbe->book1D(hname,htitle,100,0.,1000.);
    sprintf(hname,"hCaloPreShToF2_2213");
    meCaloPreShToF[1] = dbe->book1D(hname,htitle,100,0.,100.);

    for (Int_t i = 0; i < 2; ++i) {
      meCaloPreSh[i]->setAxisTitle("Number of Hits",1);
      meCaloPreSh[i]->setAxisTitle("Count",2);
      meCaloPreShE[i]->setAxisTitle("Energy of Hits (GeV)",1);
      meCaloPreShE[i]->setAxisTitle("Count",2);
      meCaloPreShToF[i]->setAxisTitle("Time of Flight of Hits (ns)",1);
      meCaloPreShToF[i]->setAxisTitle("Count",2);
    }

    sprintf(hname,"hCaloPreShPhi_2204");
    sprintf(htitle,"PreSh hits, phi/rad");
    meCaloPreShPhi = dbe->book1D(hname,htitle,100,-3.2,3.2);
    meCaloPreShPhi->setAxisTitle("Phi of Hits (rad)",1);
    meCaloPreShPhi->setAxisTitle("Count",2);

    sprintf(hname,"hCaloPreShEta_2205");
    sprintf(htitle,"PreSh hits, eta");
    meCaloPreShEta = dbe->book1D(hname,htitle,100,-5.5,5.5);
    meCaloPreShEta->setAxisTitle("Eta of Hits",1);
    meCaloPreShEta->setAxisTitle("Count",2);

    // Hcal
    dbe->setCurrentFolder("HCal");
    sprintf(hname,"hCaloHcal_2301");
    sprintf(htitle,"Hcal hits");
    meCaloHcal[0] = dbe->book1D(hname,htitle,100,0.,10000.);
    sprintf(hname,"hCaloHcal2_2311");
    meCaloHcal[1] = dbe->book1D(hname,htitle,100,-0.5,99.5);

    sprintf(hname,"hCaloHcalE_2302");
    sprintf(htitle,"Hcal hits, energy/GeV");
    meCaloHcalE[0] = dbe->book1D(hname,htitle,100,0.,10.);
    sprintf(hname,"hCaloHcalE2_2312");
    meCaloHcalE[1] = dbe->book1D(hname,htitle,100,0.,0.1);

    sprintf(hname,"hCaloHcalToF_2303");
    sprintf(htitle,"Hcal hits, ToF/ns");
    meCaloHcalToF[0] = dbe->book1D(hname,htitle,100,0.,1000.);
    sprintf(hname,"hCaloHcalToF2_2313");
    meCaloHcalToF[1] = dbe->book1D(hname,htitle,100,0.,100.);

    for (Int_t i = 0; i < 2; ++i) {
      meCaloHcal[i]->setAxisTitle("Number of Hits",1);
      meCaloHcal[i]->setAxisTitle("Count",2);
      meCaloHcalE[i]->setAxisTitle("Energy of Hits (GeV)",1);
      meCaloHcalE[i]->setAxisTitle("Count",2);
      meCaloHcalToF[i]->setAxisTitle("Time of Flight of Hits (ns)",1);
      meCaloHcalToF[i]->setAxisTitle("Count",2);
    }

    sprintf(hname,"hCaloHcalPhi_2304");
    sprintf(htitle,"Hcal hits, phi/rad");
    meCaloHcalPhi = dbe->book1D(hname,htitle,100,-3.2,3.2);
    meCaloHcalPhi->setAxisTitle("Phi of Hits (rad)",1);
    meCaloHcalPhi->setAxisTitle("Count",2);

    sprintf(hname,"hCaloHcalEta_2305");
    sprintf(htitle,"Hcal hits, eta");
    meCaloHcalEta = dbe->book1D(hname,htitle,100,-5.5,5.5);
    meCaloHcalEta->setAxisTitle("Eta of Hits",1);
    meCaloHcalEta->setAxisTitle("Count",2);
    
    dbe->setCurrentFolder("Tracker");
    sprintf(hname,"hTrackerPx_3101");
    sprintf(htitle,"Pixel hits");
    meTrackerPx[0] = dbe->book1D(hname,htitle,100,0.,10000.);
    sprintf(hname,"hTrackerPx2_3111");
    meTrackerPx[1] = dbe->book1D(hname,htitle,100,-0.5,99.5);
    for (Int_t i = 0; i < 2; ++i) {
      meTrackerPx[i]->setAxisTitle("Number of Pixel Hits",1);
      meTrackerPx[i]->setAxisTitle("Count",2);
    }

    sprintf(hname,"hTrackerPxPhi_3102");
    sprintf(htitle,"Pixel hits phi/rad");
    meTrackerPxPhi = dbe->book1D(hname,htitle,100,-3.2,3.2);
    meTrackerPxPhi->setAxisTitle("Phi of Hits (rad)",1);
    meTrackerPxPhi->setAxisTitle("Count",2);

    sprintf(hname,"hTrackerPxEta_3103");
    sprintf(htitle,"Pixel hits eta");
    meTrackerPxEta = dbe->book1D(hname,htitle,100,-3.5,3.5);
    meTrackerPxEta->setAxisTitle("Eta of Hits",1);
    meTrackerPxEta->setAxisTitle("Count",2);

    sprintf(hname,"hTrackerPxBToF_3104");
    sprintf(htitle,"Pixel barrel hits, ToF/ns");
    meTrackerPxBToF = dbe->book1D(hname,htitle,100,0.,40.);
    meTrackerPxBToF->setAxisTitle("Time of Flight of Hits (ns)",1);
    meTrackerPxBToF->setAxisTitle("Count",2);

    sprintf(hname,"hTrackerPxBR_3106");
    sprintf(htitle,"Pixel barrel hits, R/cm");
    meTrackerPxBR = dbe->book1D(hname,htitle,100,0.,50.);
    meTrackerPxBR->setAxisTitle("R of Hits (cm)",1);
    meTrackerPxBR->setAxisTitle("Count",2);

    sprintf(hname,"hTrackerPxFToF_3105");
    sprintf(htitle,"Pixel forward hits, ToF/ns");
    meTrackerPxFToF = dbe->book1D(hname,htitle,100,0.,50.);
    meTrackerPxFToF->setAxisTitle("Time of Flight of Hits (ns)",1);
    meTrackerPxFToF->setAxisTitle("Count",2);

    sprintf(hname,"hTrackerPxFZ_3107");
    sprintf(htitle,"Pixel forward hits, Z/cm");
    meTrackerPxFZ = dbe->book1D(hname,htitle,200,-100.,100.);
    meTrackerPxFZ->setAxisTitle("Z of Hits (cm)",1);
    meTrackerPxFZ->setAxisTitle("Count",2);

    sprintf(hname,"hTrackerSi_3201");
    sprintf(htitle,"Silicon hits");
    meTrackerSi[0] = dbe->book1D(hname,htitle,100,0.,10000.);
    sprintf(hname,"hTrackerSi2_3211");
    meTrackerSi[1] = dbe->book1D(hname,htitle,100,-0.5,99.5);
    for (Int_t i = 0; i < 2; ++i) { 
      meTrackerSi[i]->setAxisTitle("Number of Silicon Hits",1);
      meTrackerSi[i]->setAxisTitle("Count",2);
    }

    sprintf(hname,"hTrackerSiPhi_3202");
    sprintf(htitle,"Silicon hits phi/rad");
    meTrackerSiPhi = dbe->book1D(hname,htitle,100,-3.2,3.2);
    meTrackerSiPhi->setAxisTitle("Phi of Hits (rad)",1);
    meTrackerSiPhi->setAxisTitle("Count",2);

    sprintf(hname,"hTrackerSiEta_3203");
    sprintf(htitle,"Silicon hits eta");
    meTrackerSiEta = dbe->book1D(hname,htitle,100,-3.5,3.5);
    meTrackerSiEta->setAxisTitle("Eta of Hits",1);
    meTrackerSiEta->setAxisTitle("Count",2);

    sprintf(hname,"hTrackerSiBToF_3204");
    sprintf(htitle,"Silicon barrel hits, ToF/ns");
    meTrackerSiBToF = dbe->book1D(hname,htitle,100,0.,50.);
    meTrackerSiBToF->setAxisTitle("Time of Flight of Hits (ns)",1);
    meTrackerSiBToF->setAxisTitle("Count",2);

    sprintf(hname,"hTrackerSiBR_3206");
    sprintf(htitle,"Silicon barrel hits, R/cm");
    meTrackerSiBR = dbe->book1D(hname,htitle,100,0.,200.);
    meTrackerSiBR->setAxisTitle("R of Hits (cm)",1);
    meTrackerSiBR->setAxisTitle("Count",2);

    sprintf(hname,"hTrackerSiFToF_3205");
    sprintf(htitle,"Silicon forward hits, ToF/ns");
    meTrackerSiFToF = dbe->book1D(hname,htitle,100,0.,75.);
    meTrackerSiFToF->setAxisTitle("Time of Flight of Hits (ns)",1);
    meTrackerSiFToF->setAxisTitle("Count",2);

    sprintf(hname,"hTrackerSiFZ_3207");
    sprintf(htitle,"Silicon forward hits, Z/cm");
    meTrackerSiFZ = dbe->book1D(hname,htitle,200,-300.,300.);
    meTrackerSiFZ->setAxisTitle("Z of Hits (cm)",1);
    meTrackerSiFZ->setAxisTitle("Count",2);

    dbe->setCurrentFolder("Muon");
    sprintf(hname,"hMuon_4001");
    sprintf(htitle,"Muon hits");
    meMuon[0] = dbe->book1D(hname,htitle,100,0.,10000.);
    sprintf(hname,"hMuon2_4011");
    meMuon[1] = dbe->book1D(hname,htitle,100,-0.5,99.5);
    for (Int_t i = 0; i < 2; ++i) { 
      meMuon[i]->setAxisTitle("Number of Muon Hits",1);
      meMuon[i]->setAxisTitle("Count",2);
    }  

    sprintf(hname,"hMuonPhi_4002");
    sprintf(htitle,"Muon hits phi/rad");
    meMuonPhi = dbe->book1D(hname,htitle,100,-3.2,3.2);
    meMuonPhi->setAxisTitle("Phi of Hits (rad)",1);
    meMuonPhi->setAxisTitle("Count",2);

    sprintf(hname,"hMuonEta_4003");
    sprintf(htitle,"Muon hits eta");
    meMuonEta = dbe->book1D(hname,htitle,100,-3.5,3.5);
    meMuonEta->setAxisTitle("Eta of Hits",1);
    meMuonEta->setAxisTitle("Count",2);

    sprintf(hname,"hMuonCscToF_4201");
    sprintf(htitle,"Muon CSC hits, ToF/ns");
    meMuonCscToF[0] = dbe->book1D(hname,htitle,100,0.,250.);
    sprintf(hname,"hMuonCscToF2_4202");
    meMuonCscToF[1] = dbe->book1D(hname,htitle,100,0.,50.);
    for (Int_t i = 0; i < 2; ++i) {   
      meMuonCscToF[i]->setAxisTitle("Time of Flight of Hits (ns)",1);
      meMuonCscToF[i]->setAxisTitle("Count",2);
    }  

    sprintf(hname,"hMuonCscZ_4203");
    sprintf(htitle,"Muon CSC hits, Z/cm");
    meMuonCscZ = dbe->book1D(hname,htitle,200,-1500.,1500.);
    meMuonCscZ->setAxisTitle("Z of Hits (cm)",1);
    meMuonCscZ->setAxisTitle("Count",2);

    sprintf(hname,"hMuonDtToF_4101");
    sprintf(htitle,"Muon DT hits, ToF/ns");
    meMuonDtToF[0] = dbe->book1D(hname,htitle,100,0.,250.);
    sprintf(hname,"hMuonDtToF2_4102");
    meMuonDtToF[1] = dbe->book1D(hname,htitle,100,0.,50.);
    for (Int_t i = 0; i < 2; ++i) {   
      meMuonDtToF[i]->setAxisTitle("Time of Flight of Hits (ns)",1);
      meMuonDtToF[i]->setAxisTitle("Count",2);
    } 

    sprintf(hname,"hMuonDtR_4103");
    sprintf(htitle,"Muon DT hits, R/cm");
    meMuonDtR = dbe->book1D(hname,htitle,100,0.,1500.); 
    meMuonDtR->setAxisTitle("R of Hits (cm)",1);
    meMuonDtR->setAxisTitle("Count",2);

    sprintf(hname,"hMuonRpcFToF_4304");
    sprintf(htitle,"Muon RPC forward hits, ToF/ns");
    meMuonRpcFToF[0] = dbe->book1D(hname,htitle,100,0.,250.);
    sprintf(hname,"hMuonRpcFToF2_4305");
    meMuonRpcFToF[1] = dbe->book1D(hname,htitle,100,0.,50.);
    for (Int_t i = 0; i < 2; ++i) {   
      meMuonRpcFToF[i]->setAxisTitle("Time of Flight of Hits (ns)",1);
      meMuonRpcFToF[i]->setAxisTitle("Count",2);
    }  

    sprintf(hname,"hMuonRpcFZ_4306");
    sprintf(htitle,"Muon RPC forward hits, Z/cm");
    meMuonRpcFZ = dbe->book1D(hname,htitle,201,-1500.,1500.);
    meMuonRpcFZ->setAxisTitle("Z of Hits (cm)",1);
    meMuonRpcFZ->setAxisTitle("Count",2);

    sprintf(hname,"hMuonRpcBToF_4101");
    sprintf(htitle,"Muon RPC barrel hits, ToF/ns");
    meMuonRpcBToF[0] = dbe->book1D(hname,htitle,100,0.,250.);
    sprintf(hname,"hMuonRpcBToF2_4102");
    meMuonRpcBToF[1] = dbe->book1D(hname,htitle,100,0.,50.);
    for (Int_t i = 0; i < 2; ++i) {   
      meMuonRpcBToF[i]->setAxisTitle("Time of Flight of Hits (ns)",1);
      meMuonRpcBToF[i]->setAxisTitle("Count",2);
    }

    sprintf(hname,"hMuonRpcBR_4103");
    sprintf(htitle,"Muon RPC barrel hits, R/cm");
    meMuonRpcBR = dbe->book1D(hname,htitle,100,0.,1500.);
    meMuonRpcBR->setAxisTitle("R of Hits (cm)",1);
    meMuonRpcBR->setAxisTitle("Count",2);
  }
  */

  // print out Parameter Set information being used
  if (verbosity >= 0) {
    edm::LogInfo(MsgLoggerCat) 
      << "\n===============================\n"
      << "Initialized as EDAnalyzer with parameter values:\n"
      << "    Name           = " << fName << "\n"
      << "    Verbosity      = " << verbosity << "\n"
      << "    Frequency      = " << frequency << "\n"
      << "    VtxUnit        = " << vtxunit << "\n"
      << "    OutputFile     = " << outputfile << "\n"
      << "    GetProv        = " << getAllProvenances << "\n"
      << "    PrintProv      = " << printProvenanceInfo << "\n"
      << "===============================\n";
  }

}

GlobalHitsProdHistStripper::~GlobalHitsProdHistStripper() 
{
  if (outputfile.size() != 0 && dbe) dbe->save(outputfile);
}

void GlobalHitsProdHistStripper::beginJob(const edm::EventSetup& iSetup)
{
  return;
}

void GlobalHitsProdHistStripper::endJob()
{
  std::string MsgLoggerCat = "GlobalHitsProdHistStripper_endJob";
  if (verbosity >= 0)
    edm::LogInfo(MsgLoggerCat) 
      << "Terminating having processed " << count << " runs.";
  return;
}

void GlobalHitsProdHistStripper::beginRun(const edm::Run& iRun,
					  const edm::EventSetup& iSetup)
{
  std::string MsgLoggerCat = "GlobalHitsProdHistStripper_beginRun";
  // keep track of number of runs processed
  ++count;  

  int nrun = iRun.run();

  if (verbosity > 0) {
    edm::LogInfo(MsgLoggerCat)
      << "Processing run " << nrun << " (" << count << " runs total)";
  } else if (verbosity == 0) {
    if (nrun%frequency == 0 || nrun == 0) {
      edm::LogInfo(MsgLoggerCat)
	<< "Processing run " << nrun << " (" << count << " runs total)";
    }
  }

  if (getAllProvenances) {

    std::vector<const edm::Provenance*> AllProv;
    iRun.getAllProvenance(AllProv);

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

  return;
}

void GlobalHitsProdHistStripper::endRun(const edm::Run& iRun,
					const edm::EventSetup& iSetup)
{
  std::string MsgLoggerCat = "GlobalHitsProdHistStripper_endRun";

  //std::map<std::string,MonitorElement*>::iterator iter;
  edm::Handle<TH1F> histogram1D;
  std::vector<edm::Handle<TH1F> > allhistogram1D;
  iRun.getManyByType(allhistogram1D);

  me.resize(allhistogram1D.size());

  for (uint i = 0; i < allhistogram1D.size(); ++i) {
    histogram1D = allhistogram1D[i];
    if(!histogram1D.isValid()) {
      edm::LogWarning(MsgLoggerCat)
	<< "Invalid histogram extracted from event.";
      continue;      
    }

    me[i] = 0;

    /*
    std::cout << "Extracting histogram: " << std::endl
	      << "       Module       : "
	      << (histogram1D.provenance()->product()).moduleLabel()
	      << std::endl
	      << "       ProductID    : "
	      << (histogram1D.provenance()->product()).productID().id()
	      << std::endl
	      << "       ClassName    : "
	      << (histogram1D.provenance()->product()).className()
	      << std::endl
	      << "       InstanceName : "
	      << (histogram1D.provenance()->product()).productInstanceName()
	      << std::endl
	      << "       BranchName   : "
	      << (histogram1D.provenance()->product()).branchName()
	      << std::endl;
    */

    if ((histogram1D.provenance()->product()).moduleLabel()
	!= "globalhitsprodhist") continue;
   
    std::string histname = histogram1D->GetName();

    //iter = monitorElements.find(histname);
    
    std::string subhist1 = histname.substr(1,5);
    std::string subhist2 = histname.substr(1,4);

    if (dbe) {
      if (subhist1 == "CaloE" || subhist1 == "CaloP") {
	dbe->setCurrentFolder("ECal");
      } else if (subhist1 == "CaloH") {
	dbe->setCurrentFolder("HCal");
      } else if (subhist1 == "Geant" || subhist2 == "MCG4" ||
		 subhist1 == "MCRGP") {
	dbe->setCurrentFolder("MCGeant");
      } else if (subhist2 == "Muon") {
	dbe->setCurrentFolder("Muon");
      } else if (subhist1 == "Track") {
	dbe->setCurrentFolder("Tracker");
      }
     
      me[i] = dbe->book1D(histname,histogram1D->GetTitle(),
		       histogram1D->GetXaxis()->GetNbins(),
		       histogram1D->GetXaxis()->GetXmin(),
		       histogram1D->GetXaxis()->GetXmax());
      me[i]->setAxisTitle(histogram1D->GetXaxis()->GetTitle(),1);
      me[i]->setAxisTitle(histogram1D->GetYaxis()->GetTitle(),2);
      
    }
    
    std::string mename = me[i]->getName();
    
    std::cout << "Extracting histogram " << histname
	      << " into MonitorElement " << mename
	      << std::endl;
    
    for (Int_t x = 1; x <= histogram1D->GetXaxis()->GetNbins(); ++x) {
      Double_t binx = histogram1D->GetBinCenter(x);
      Double_t value = histogram1D->GetBinContent(x);
      me[i]->Fill(binx,value);
    }
  }
  return;
}
    /*
    if (iter != monitorElements.end()) {
      
      std::string mename = iter->second->getName();

      std::cout << "Extracting histogram " << histname
		<< " into MonitorElement " << mename
		<< std::endl;
    
      if (histname == "hGeantTrkE" || histname == "hGeantTrkPt") {
	std::cout << "Information stored in histogram pointer:" 
		  << std::endl;
	std::cout << histname << ":" << std::endl;
	std::cout << "  Entries: " << histogram1D->GetEntries() 
		  << std::endl;
	std::cout << "  Mean: " << histogram1D->GetMean() << std::endl;
	std::cout << "  RMS: " << histogram1D->GetRMS() << std::endl;
      }

      for (Int_t x = 1; x <= histogram1D->GetXaxis()->GetNbins(); ++x) {
	Double_t binx = histogram1D->GetBinCenter(x);
	Double_t value = histogram1D->GetBinContent(x);
	iter->second->Fill(binx,value);
      }
     
      if (histname == "hGeantTrkE" || histname == "hGeantTrkPt") {
	std::cout << "Information stored in monitor element:" << std::endl;
	std::cout << mename << ":" << std::endl;
	std::cout << "  Entries: " 
		  << iter->second->getEntries() << std::endl;
	std::cout << "  Mean: " << iter->second->getMean() 
		  << std::endl;
	std::cout << "  RMS: " << iter->second->getRMS() 
		  << std::endl;
		  }
  } // find in map
} // loop through getManyByType

  return;
}
    */

void GlobalHitsProdHistStripper::analyze(const edm::Event& iEvent, 
					 const edm::EventSetup& iSetup)
{
  return;
}


