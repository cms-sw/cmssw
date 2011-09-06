// -*- C++ -*-
//
// Package:     Forward
// Class  :     CastorShowerLibraryMaker
//
// Implementation:
//     <Notes on implementation>
//
// Original Author: P. Katsas
//         Created: 02/2007 
//
// Adapted by W. Carvalho , 02/2009
//
//////////////////////////////////////////////////////////////

#include "SimG4Core/Notification/interface/SimG4Exception.h"
#include "SimG4CMS/Calo/interface/CaloG4Hit.h"
#include "SimG4CMS/Calo/interface/CaloG4HitCollection.h"
#include "DataFormats/Math/interface/Point3D.h"

#include "SimG4CMS/ShowerLibraryProducer/interface/CastorShowerLibraryMaker.h"
#include "SimG4CMS/Forward/interface/CastorNumberingScheme.h"

#include "SimDataFormats/CaloHit/interface/CastorShowerEvent.h"

#include "TFile.h"
#include <cmath>
#include <iostream>
#include <iomanip>

CastorShowerLibraryMaker::CastorShowerLibraryMaker(const edm::ParameterSet &p) : 
                             NPGParticle(0),DoHadSL(false),DoEmSL(false),
                             emShower(NULL) , hadShower(NULL) {

  MapOfSecondaries.clear();
  hadInfo = NULL;
  emInfo  = NULL;
  edm::ParameterSet p_SLM   = p.getParameter<edm::ParameterSet>("CastorShowerLibraryMaker");
  verbosity                 = p_SLM.getParameter<int>("Verbosity");
  eventNtFileName           = p_SLM.getParameter<std::string>("EventNtupleFileName");
  hadSLHolder.nEvtPerBinPhi = p_SLM.getParameter<int>("nhadEvents");
  emSLHolder.nEvtPerBinPhi  = p_SLM.getParameter<int>("nemEvents");
  hadSLHolder.SLEnergyBins  = p_SLM.getParameter<std::vector<double> >("SLhadEnergyBins");
  hadSLHolder.SLEtaBins     = p_SLM.getParameter<std::vector<double> >("SLhadEtaBins");
  hadSLHolder.SLPhiBins     = p_SLM.getParameter<std::vector<double> >("SLhadPhiBins");
  emSLHolder.SLEnergyBins   = p_SLM.getParameter<std::vector<double> >("SLemEnergyBins");
  emSLHolder.SLEtaBins      = p_SLM.getParameter<std::vector<double> >("SLemEtaBins");
  emSLHolder.SLPhiBins      = p_SLM.getParameter<std::vector<double> >("SLemPhiBins");
  PGParticleIDs             = p_SLM.getParameter<std::vector<int> >("PartID");
  NPGParticle               = PGParticleIDs.size(); 
//
  for(unsigned int i=0;i<PGParticleIDs.size();i++) {
     switch (int(fabs(PGParticleIDs.at(i)))) {
        case 11:
        case 22:
               DoEmSL = true;
               break;
        default:
               DoHadSL = true;
     }
  }
  hadSLHolder.nEvtPerBinEta = (hadSLHolder.nEvtPerBinPhi)*(hadSLHolder.SLPhiBins.size());
  hadSLHolder.nEvtPerBinE   = (hadSLHolder.nEvtPerBinEta)*(hadSLHolder.SLEtaBins.size());
  emSLHolder.nEvtPerBinEta = (emSLHolder.nEvtPerBinPhi)*(emSLHolder.SLPhiBins.size());
  emSLHolder.nEvtPerBinE   = (emSLHolder.nEvtPerBinEta)*(emSLHolder.SLEtaBins.size());

  std::cout << "============================================================================"<<std::endl;
  std::cout << "CastorShowerLibraryMaker:: Initialized as observer" << std::endl;
  std::cout << " Event Ntuple will be created" << std::endl;
  std::cout << " Event Ntuple file: " << eventNtFileName << std::endl;
  std::cout << " Number of Hadronic events in E   bins: " << hadSLHolder.nEvtPerBinE << std::endl;
  std::cout << " Number of Hadronic events in Eta bins: " << hadSLHolder.nEvtPerBinEta << std::endl;
  std::cout << " Number of Hadronic events in Phi bins: " << hadSLHolder.nEvtPerBinPhi << std::endl;
  std::cout << " Number of Electromag. events in E   bins: " << emSLHolder.nEvtPerBinE << std::endl;
  std::cout << " Number of Electromag. events in Eta bins: " << emSLHolder.nEvtPerBinEta << std::endl;
  std::cout << " Number of Electromag. events in Phi bins: " << emSLHolder.nEvtPerBinPhi << std::endl;
  std::cout << "============================================================================"<<std::endl;
  std::cout << std::endl;

  // Initializing the SL collections
  InitSLHolder(hadSLHolder);
  InitSLHolder(emSLHolder);
}
void CastorShowerLibraryMaker::InitSLHolder(ShowerLib& showerholder)
{
  int nBinsE,nBinsEta,nBinsPhi,nEvtPerBinPhi;
  nBinsE      = showerholder.SLEnergyBins.size();
  nBinsEta    = showerholder.SLEtaBins.size();
  nBinsPhi    = showerholder.SLPhiBins.size();
  nEvtPerBinPhi=showerholder.nEvtPerBinPhi;
//
// Info
//
  showerholder.SLInfo.Energy.setNEvts(nEvtPerBinPhi*nBinsPhi*nBinsEta*nBinsE);
  showerholder.SLInfo.Energy.setNEvtPerBin(nEvtPerBinPhi*nBinsPhi*nBinsEta);
  showerholder.SLInfo.Energy.setNBins(nBinsE);
  showerholder.SLInfo.Energy.setBin(showerholder.SLEnergyBins);
//
  showerholder.SLInfo.Eta.setNEvts(nEvtPerBinPhi*nBinsPhi*nBinsEta);
  showerholder.SLInfo.Eta.setNEvtPerBin(nEvtPerBinPhi*nBinsPhi);
  showerholder.SLInfo.Eta.setNBins(nBinsEta);
  showerholder.SLInfo.Eta.setBin(showerholder.SLEtaBins);
//
  showerholder.SLInfo.Phi.setNEvts(nEvtPerBinPhi*nBinsPhi);
  showerholder.SLInfo.Phi.setNEvtPerBin(nEvtPerBinPhi);
  showerholder.SLInfo.Phi.setNBins(nBinsPhi);
  showerholder.SLInfo.Phi.setBin(showerholder.SLPhiBins);
//
// Shower
  showerholder.SLCollection.assign(nBinsE,std::vector<std::vector<std::vector<CastorShowerEvent> > >());
  showerholder.nEvtInBinE.assign(nBinsE,0);
  showerholder.nEvtInBinEta.assign(nBinsE,std::vector<int>(0));
  showerholder.nEvtInBinPhi.assign(nBinsE,std::vector<std::vector<int> >());
  for(int i=0;i<nBinsE;i++) {
     showerholder.SLCollection.at(i).assign(nBinsEta,std::vector<std::vector<CastorShowerEvent> >());
     showerholder.nEvtInBinEta.at(i).assign(nBinsEta,0);
     showerholder.nEvtInBinPhi.at(i).assign(nBinsEta,std::vector<int>(0));
     for(int j=0;j<nBinsEta;j++) {
        showerholder.SLCollection.at(i).at(j).assign(nBinsPhi,std::vector<CastorShowerEvent>());
        showerholder.nEvtInBinPhi.at(i).at(j).assign(nBinsPhi,0);
        for(int k=0;k<nBinsPhi;k++) 
           showerholder.SLCollection.at(i).at(j).at(k).assign(nEvtPerBinPhi,CastorShowerEvent());
     }
  }
}

//===============================================================================================

CastorShowerLibraryMaker::~CastorShowerLibraryMaker() {

  Finish();

  std::cout << "CastorShowerLibraryMaker: End of process" << std::endl;

}

//=================================================================== per EVENT
void CastorShowerLibraryMaker::update(const BeginOfJob * job) {

  std::cout << " CastorShowerLibraryMaker::Starting new job " << std::endl;
}

//==================================================================== per RUN
void CastorShowerLibraryMaker::update(const BeginOfRun * run) {

  std::cout << std::endl << "CastorShowerLibraryMaker: Starting Run"<< std::endl; 

  std::cout << "CastorShowerLibraryMaker: output event root file created" << std::endl;

  TString eventfilename = eventNtFileName;
  theFile = new TFile(eventfilename,"RECREATE");
  theTree = new TTree("CastorCherenkovPhotons", "Cherenkov Photons");

  Int_t split = 1;
  Int_t bsize = 64000;
  emInfo      = new CastorShowerLibraryInfo();
  emShower    = new CastorShowerEvent();
  hadInfo     = new CastorShowerLibraryInfo();
  hadShower   = new CastorShowerEvent();
  // Create Branchs
  theTree->Branch("emShowerLibInfo.", "CastorShowerLibraryInfo", &emInfo, bsize, split);
  theTree->Branch("emParticles.", "CastorShowerEvent", &emShower, bsize, split);
  theTree->Branch("hadShowerLibInfo.", "CastorShowerLibraryInfo", &hadInfo, bsize, split);
  theTree->Branch("hadParticles.", "CastorShowerEvent", &hadShower, bsize, split);

// set the Info for electromagnetic shower
// set the energy bins info
  emInfo->Energy.setNEvts(emSLHolder.nEvtPerBinE*emSLHolder.SLEnergyBins.size());
  emInfo->Energy.setNBins(emSLHolder.SLEnergyBins.size());
  emInfo->Energy.setNEvtPerBin(emSLHolder.nEvtPerBinE);
  emInfo->Energy.setBin(emSLHolder.SLEnergyBins);
// set the eta bins info
  emInfo->Eta.setNEvts(emSLHolder.nEvtPerBinEta*emSLHolder.SLEtaBins.size());
  emInfo->Eta.setNBins(emSLHolder.SLEtaBins.size());
  emInfo->Eta.setNEvtPerBin(emSLHolder.nEvtPerBinEta);
  emInfo->Eta.setBin(emSLHolder.SLEtaBins);
// set the eta bins info
  emInfo->Phi.setNEvts(emSLHolder.nEvtPerBinPhi*emSLHolder.SLPhiBins.size());
  emInfo->Phi.setNBins(emSLHolder.SLPhiBins.size());
  emInfo->Phi.setNEvtPerBin(emSLHolder.nEvtPerBinPhi);
  emInfo->Phi.setBin(emSLHolder.SLPhiBins);
// The same for the hadronic shower
// set the energy bins info
  hadInfo->Energy.setNEvts(hadSLHolder.nEvtPerBinE*hadSLHolder.SLEnergyBins.size());
  hadInfo->Energy.setNBins(hadSLHolder.SLEnergyBins.size());
  hadInfo->Energy.setNEvtPerBin(hadSLHolder.nEvtPerBinE);
  hadInfo->Energy.setBin(hadSLHolder.SLEnergyBins);
// set the eta bins info
  hadInfo->Eta.setNEvts(hadSLHolder.nEvtPerBinEta*hadSLHolder.SLEtaBins.size());
  hadInfo->Eta.setNBins(hadSLHolder.SLEtaBins.size());
  hadInfo->Eta.setNEvtPerBin(hadSLHolder.nEvtPerBinEta);
  hadInfo->Eta.setBin(hadSLHolder.SLEtaBins);
// set the eta bins info
  hadInfo->Phi.setNEvts(hadSLHolder.nEvtPerBinPhi*hadSLHolder.SLPhiBins.size());
  hadInfo->Phi.setNBins(hadSLHolder.SLPhiBins.size());
  hadInfo->Phi.setNEvtPerBin(hadSLHolder.nEvtPerBinPhi);
  hadInfo->Phi.setBin(hadSLHolder.SLPhiBins);
  // int flag = theTree->GetBranch("CastorShowerLibInfo")->Fill();
  // Loop on all leaves of this branch to fill Basket buffer.
  // The function returns the number of bytes committed to the memory basket.
  // If a write error occurs, the number of bytes returned is -1.
  // If no data are written, because e.g. the branch is disabled,
  // the number of bytes returned is 0.
  // if(flag==-1) {
  //    edm::LogInfo("CastorAnalyzer") << " WARNING: Error writing to Branch \"CastorShowerLibInfo\" \n" ; 
  // } else 
  // if(flag==0) {
  //    edm::LogInfo("CastorAnalyzer") << " WARNING: No data written to Branch \"CastorShowerLibInfo\" \n" ; 
  // }

  // Initialize "accounting" variables

  eventIndex = 0;

}

//=================================================================== per EVENT
void CastorShowerLibraryMaker::update(const BeginOfEvent * evt) {

  eventIndex++;
  stepIndex = 0;
// reset the pointers to the shower objects
  SLShowerptr = NULL;
  MapOfSecondaries.clear();
//
  std::cout << "CastorShowerLibraryMaker: Processing Event Number: " << eventIndex << std::endl;
}

//=================================================================== per STEP
void CastorShowerLibraryMaker::update(const G4Step * aStep) {
   static int CurrentPrimary = 0;
   G4Track *trk = aStep->GetTrack();
   if (trk->GetCurrentStepNumber()==1) {
      if (trk->GetParentID()==0) CurrentPrimary = trk->GetDynamicParticle()->GetPDGcode();
      if (CurrentPrimary==0) 
         SimG4Exception("CastorShowerLibraryMaker::update(G4Step) -> Primary particle undefined");
      MapOfSecondaries[CurrentPrimary].insert((int)trk->GetTrackID());
   }
/*
  if(aStep->IsFirstStepInVolume()) { 
    edm::LogInfo("CastorShowerLibraryMaker") << "CastorShowerLibraryMaker::update(const G4Step * aStep):"
                                             << "\n IsFirstStepInVolume , " 
                                             << "time = " << aStep->GetTrack()->GetGlobalTime() ; 
  }
  stepIndex++;
*/
}

//================= End of EVENT ===============
void CastorShowerLibraryMaker::update(const EndOfEvent * evt) {

// check if the job is done!
  if (IsSLReady()) update((EndOfRun*)NULL);

  std::cout << "CastorShowerLibraryMaker: End of Event: " << eventIndex << std::endl;
// Get the pointer to the primary particle
  std::vector<G4PrimaryParticle*> thePrims = GetPrimary(evt);
  if (thePrims.size() == 0) {
     edm::LogInfo("CastorShowerLibraryMaker") << "No valid primary particle found. Skipping event" << std::endl;
     return;
  }

// Loop over primaries
  for(unsigned int i=0;i<thePrims.size();i++) {
     G4PrimaryParticle* thePrim = thePrims.at(i);
     if (!thePrim) {
        edm::LogInfo("CastorShowerLibraryMaker") << "NULL Pointer to the primary" << std::endl;
        continue;
     }
// Check primary particle type
     int particleType = thePrim->GetPDGcode();

// set the pointer to the shower collection
     std::string SLType("");
     if (particleType==11) {
        SLShowerptr = &emSLHolder;
        SLType = "Electromagnetic";
     }
     else {
        SLShowerptr = &hadSLHolder;
        SLType = "Hadronic";
     }

// Obtain primary particle's initial momentum (pInit)
     double px=0., py=0., pz=0., pInit = 0., eta = 0., phi = 0.;

     GetKinematics(thePrim,px,py,pz,pInit,eta,phi);
     edm::LogInfo("CastorShowerLibraryMaker") << "\n Primary (thePrim) trackID is " << thePrim->GetTrackID() << "\n" ;

// Check if current event falls into any bin
// first: energy
     int ebin = FindEnergyBin(pInit);
     int etabin= FindEtaBin(eta);
     int phibin = FindPhiBin(phi);
     std::cout << SLType << std::endl;
     printSLstatus(ebin,etabin,phibin);
     if (!SLacceptEvent(ebin,etabin,phibin)) {
        edm::LogInfo("CastorShowerLibraryMaker") << "Event not accepted for ebin="
             << ebin<<",etabin="<<etabin<<",phibin="<<phibin<<std::endl;
        continue;
     }
//
// event passed. Fill the vector accordingly
//  
// Look for the Hit Collection 
     edm::LogInfo("CastorShowerLibraryMaker")
        << "\n CastorShowerLibraryMaker::update(EndOfEvent * evt) - event #" 
        << (*evt)()->GetEventID() ;

  // access to the G4 hit collections 
     G4HCofThisEvent* allHC = (*evt)()->GetHCofThisEvent();
/*
     std::cout << "Number of collections : " << allHC->GetNumberOfCollections() << std::endl;
     for(int ii = 0;ii<allHC->GetNumberOfCollections();ii++) 
        std::cout << "Name of collection " << ii << " : " << allHC->GetHC(ii)->GetName() << std::endl;
*/
     edm::LogInfo("CastorShowerLibraryMaker") << " update(*evt) --> accessed all HC ";

     CastorShowerEvent* shower=NULL;
     int cur_evt_idx = SLShowerptr->nEvtInBinPhi.at(ebin).at(etabin).at(phibin);
     shower = &(SLShowerptr->SLCollection.at(ebin).at(etabin).at(phibin).at(cur_evt_idx));

// Get Hit information
     if (FillShowerEvent(allHC,shower,particleType)) { 
//  Primary particle information
        shower->setPrimE(pInit);
        shower->setPrimEta(eta);
        shower->setPrimPhi(phi);
        //shower->setPrimX(entry.x());
        //shower->setPrimY(entry.y());
        //shower->setPrimZ(entry.z());
        SLnEvtInBinE(ebin)++;
        SLnEvtInBinEta(ebin,etabin)++;
        SLnEvtInBinPhi(ebin,etabin,phibin)++;
      }
  }

  //int iEvt = (*evt)()->GetEventID();
  //double xint;
/*
  if (modf(log10(iEvt),&xint)==0) 
    std::cout << " CastorShowerLibraryMaker Event " << iEvt << std::endl;
*/
  // std::cout << std::endl << "===>>> Done writing user histograms " << std::endl;
}

//========================= End of RUN ======================
void CastorShowerLibraryMaker::update(const EndOfRun * run)
{
// Fill the tree with the collected objects
  if (!IsSLReady()) SimG4Exception("\n\nShower Library 	NOT READY.\n\n");

  unsigned int  ibine,ibineta,ibinphi,ievt; // indexes for em shower
  unsigned int  jbine,jbineta,jbinphi,jevt;// indexes for had shower

  ibine=ibineta=ibinphi=ievt=jbine=jbineta=jbinphi=jevt=0;

  int  nEvtInTree = 0;
  int  maxEvtInTree=std::max(hadSLHolder.nEvtPerBinE*hadSLHolder.SLEnergyBins.size(),
                             emSLHolder.nEvtPerBinE*emSLHolder.SLEnergyBins.size());

  emInfo = &emSLHolder.SLInfo;
  hadInfo= &hadSLHolder.SLInfo;

  while(nEvtInTree<maxEvtInTree) {
    if (emShower) emShower->Clear();
    if (hadShower) hadShower->Clear();
    while(ibine<emSLHolder.SLEnergyBins.size()){
      emShower = &(emSLHolder.SLCollection.at(ibine).at(ibineta).at(ibinphi).at(ievt));
      ievt++;
      if (ievt==emSLHolder.nEvtPerBinPhi) {ievt=0;ibinphi++;}
      if (ibinphi==emSLHolder.SLPhiBins.size()) {ibinphi=0;ibineta++;}
      if (ibineta==emSLHolder.SLEtaBins.size()) {ibineta=0;ibine++;}
      break;
    }
    while(jbine<hadSLHolder.SLEnergyBins.size()){
      hadShower = &(hadSLHolder.SLCollection.at(jbine).at(jbineta).at(jbinphi).at(jevt));
      jevt++;
      if (jevt==hadSLHolder.nEvtPerBinPhi) {jevt=0;jbinphi++;}
      if (jbinphi==hadSLHolder.SLPhiBins.size()) {jbinphi=0;jbineta++;}
      if (jbineta==hadSLHolder.SLEtaBins.size()) {jbineta=0;jbine++;}
      break;
    }
    theTree->Fill();
    nEvtInTree++;
    if (nEvtInTree==1) {
       theTree->SetBranchStatus("emShowerLibInfo.",0);
       theTree->SetBranchStatus("hadShowerLibInfo.",0);
    }
  }
// check if run is NULL and exit
  if (run==NULL) throw SimG4Exception("\n\nNumber of needed trigger events reached in CastorShowerLibraryMaker\n\n");
}

//============================================================
void CastorShowerLibraryMaker::Finish() {

  // if (doNTcastorevent) {

  theFile->cd();
  theTree->Write("",TObject::kOverwrite);
  std::cout << "CastorShowerLibraryMaker: Ntuple event written" << std::endl;   
  theFile->Close();
  std::cout << "CastorShowerLibraryMaker: Event file closed" << std::endl;

  // Delete pointers to objects, now that TTree has been written and TFile closed
//  delete      info; 
//  delete  emShower;
//  delete hadShower;
  // }
} 
int CastorShowerLibraryMaker::FindEnergyBin(double energy) {
  //
  // returns the integer index of the energy bin, taken from SLenergies vector
  // returns -1 if ouside valid range
  //
  if (!SLShowerptr) {
     edm::LogInfo("CastorShowerLibraryMaker") << "\n\nFindEnergyBin can be called only after BeginOfEvent\n\n";
     throw SimG4Exception("\n\nNULL Pointer to the shower library.\n\n");
  }
  const std::vector<double>& SLenergies = SLShowerptr->SLEnergyBins;
  if (energy >= SLenergies.back()) return SLenergies.size()-1;

  unsigned int i = 0;
  for(;i<SLenergies.size()-1;i++) 
    if (energy >= SLenergies.at(i) && energy < SLenergies.at(i+1)) return (int)i;

  // now i points to the last but 1 bin
  if (energy>=SLenergies.at(i)) return (int)i;
  // energy outside bin range
  return -1;
}
int CastorShowerLibraryMaker::FindEtaBin(double eta) {
  //
  // returns the integer index of the eta bin, taken from SLetas vector
  // returns -1 if ouside valid range
  //
  if (!SLShowerptr) {
     edm::LogInfo("CastorShowerLibraryMaker") << "\n\nFindEtaBin can be called only after BeginOfEvent\n\n";
     throw SimG4Exception("\n\nNULL Pointer to the shower library.\n\n");
  }
  const std::vector<double>& SLetas = SLShowerptr->SLEtaBins;
  if (eta>=SLetas.back()) return SLetas.size()-1;
  unsigned int i = 0;
  for(;i<SLetas.size()-1;i++)
     if (eta >= SLetas.at(i) && eta < SLetas.at(i+1)) return (int)i;
  // now i points to the last but 1 bin
  if (eta>=SLetas.at(i)) return (int)i;
  // eta outside bin range
  return -1;
}
int CastorShowerLibraryMaker::FindPhiBin(double phi) {
  //
  // returns the integer index of the phi bin, taken from SLphis vector
  // returns -1 if ouside valid range
  //
  // needs protection in case phi is outside range -pi,pi
  //
  if (!SLShowerptr) {
     edm::LogInfo("CastorShowerLibraryMaker") << "\n\nFindPhiBin can be called only after BeginOfEvent\n\n";
     throw SimG4Exception("\n\nNULL Pointer to the shower library.\n\n");
  }
  const std::vector<double>& SLphis = SLShowerptr->SLPhiBins;
  if (phi>=SLphis.back()) return SLphis.size()-1;
  unsigned int i = 0;
  for(;i<SLphis.size()-1;i++)
     if (phi >= SLphis.at(i) && phi < SLphis.at(i+1)) return (int)i;
  // now i points to the last but 1 bin
  if (phi>=SLphis.at(i)) return (int)i;
  // phi outside bin range
  return -1;
}
bool CastorShowerLibraryMaker::IsSLReady()
{
// at this point, the pointer to the shower library should be NULL
  if (SLShowerptr) {
     edm::LogInfo("CastorShowerLibraryMaker") << "\n\nIsSLReady must be called when a new event starts.\n\n";
     throw SimG4Exception("\n\nNOT NULL Pointer to the shower library.\n\n");
  }
// it is enough to check if all the energy bin is filled
  if (DoEmSL) {
     SLShowerptr = &emSLHolder;
     for(unsigned int i=0;i<SLShowerptr->SLEnergyBins.size();i++) {
        if (!SLisEBinFilled(i)) {
           SLShowerptr=NULL;
           return false;
        }
     }
  }
  if (DoHadSL) {
     SLShowerptr = &hadSLHolder;
     for(unsigned int i=0;i<SLShowerptr->SLEnergyBins.size();i++) {
        if (!SLisEBinFilled(i)) {
           SLShowerptr=NULL;
           return false;
        }
     }
  }
  SLShowerptr=NULL;
  return true;
}
void CastorShowerLibraryMaker::GetKinematics(G4PrimaryParticle* thePrim,double& px, double& py, double& pz, double& pInit, double& eta, double& phi)
{
    px=py=pz=phi=eta=0.0;
    if (thePrim==0) return;
    px = thePrim->GetPx()/GeV;
    py = thePrim->GetPy()/GeV;
    pz = thePrim->GetPz()/GeV;
    pInit = sqrt(pow(px,2.)+pow(py,2.)+pow(pz,2.));
    if (pInit==0) {
      std::cout << "CastorShowerLibraryMaker::GetKinematics: ERROR: primary has p=0 " << std::endl;
      return;
    }
    double costheta = pz/pInit;
    double theta = acos(std::min(std::max(costheta,double(-1.)),double(1.)));
    eta = -log(tan(theta/2.0));
    phi = (px==0 && py==0) ? 0 : atan2(py,px); // the recommended way of calculating phi
    //if (px!=0) phi=atan(py/px);
}
std::vector<G4PrimaryParticle*> CastorShowerLibraryMaker::GetPrimary(const EndOfEvent * evt)
{
  // Find Primary info:
  int trackID = 0;
  std::vector<G4PrimaryParticle*> thePrims;
  G4PrimaryParticle* thePrim = 0;
  G4int nvertex = (*evt)()->GetNumberOfPrimaryVertex();
  edm::LogInfo("CastorShowerLibraryMaker")  << "Event has " << nvertex << " vertex";   
  if (nvertex!=1) {
     edm::LogInfo("CastorShowerLibraryMaker") << "CastorShowerLibraryMaker::GetPrimary ERROR: no vertex";
     return thePrims;
  }

  for (int i = 0 ; i<nvertex; i++) {
    G4PrimaryVertex* avertex = (*evt)()->GetPrimaryVertex(i);
    if (avertex == 0) {
       edm::LogInfo("CastorShowerLibraryMaker")
          << "CastorShowerLibraryMaker::GetPrimary ERROR: pointer to vertex = 0";
       continue;
    }
    unsigned int npart = avertex->GetNumberOfParticle();
    if (npart!=NPGParticle) continue;
    for (unsigned int j=0;j<npart;j++) {
        unsigned int k = 0;
        //int test_pID = 0;
        trackID = j;
        thePrim=avertex->GetPrimary(trackID);
        while(k<NPGParticle&&PGParticleIDs.at(k++)!=thePrim->GetPDGcode()){;};
        if (k>NPGParticle) continue; // ID not found in the requested particles
        thePrims.push_back(thePrim);
    }
  }
  return thePrims;
}
void CastorShowerLibraryMaker::printSLstatus(int ebin,int etabin,int phibin)
{
  int nBinsE  =SLShowerptr->SLEnergyBins.size();
  int nBinsEta=SLShowerptr->SLEtaBins.size();
  int nBinsPhi=SLShowerptr->SLPhiBins.size();
  std::vector<double> SLenergies = SLShowerptr->SLEnergyBins;
  for(int n=0;n<11+(nBinsEta*nBinsPhi);n++) std::cout << "=";
  std::cout << std::endl;
  for(int i=0;i<nBinsE;i++) {
     std::cout << "E bin " << SLenergies.at(i) << " : ";
     for(int j=0;j<nBinsEta;j++) {
        for(int k=0;k<nBinsPhi;k++) {
           (SLisPhiBinFilled(i,j,k))?std::cout << "1":std::cout << "-";
        }
        if (j<nBinsEta-1) std::cout << "|";
     }
     std::cout << " (" << SLnEvtInBinE(i) << " events)";
     std::cout << std::endl;
     if (ebin!=i) continue;
     std::cout << "          ";
     for(int j=0;j<nBinsEta;j++) {
        for(int k=0;k<nBinsPhi;k++) {
           (ebin==i&&etabin==j&&phibin==k)?std::cout <<  "^":std::cout << " ";
        }
        if (j<nBinsEta-1) std::cout << " ";
     }
     std::cout << std::endl;
  }
  for(int n=0;n<11+(nBinsEta*nBinsPhi);n++) std::cout << "=";
  std::cout << std::endl;
}
bool CastorShowerLibraryMaker::SLacceptEvent(int ebin, int etabin, int phibin)
{
     if (ebin<0) return false;
     if (SLisEBinFilled(ebin)) return false;

     if (etabin<0) return false;
     if (SLisEtaBinFilled(ebin,etabin)) return false;

     if (phibin<0) return false;
     if (SLisPhiBinFilled(ebin,etabin,phibin)) return false;
     return true;
}
bool CastorShowerLibraryMaker::FillShowerEvent(G4HCofThisEvent* allHC, CastorShowerEvent* shower,int ipart)
{
  //   int CAFIid = G4SDManager::GetSDMpointer()->GetCollectionID("CastorPL"); // Trick to get CASTPL
     int CAFIid = G4SDManager::GetSDMpointer()->GetCollectionID("CastorFI");

     CaloG4HitCollection* theCAFI = (CaloG4HitCollection*) allHC->GetHC(CAFIid);

     CastorNumberingScheme *theCastorNumScheme = new CastorNumberingScheme();

     unsigned int volumeID=0;
     double en_in_fi = 0.;
     //double totalEnergy = 0;

     int nentries = theCAFI->entries();
     edm::LogInfo("CastorShowerLibraryMaker") << "Found "<<nentries << " hits in G4HitCollection";
     if (nentries == 0) {
       edm::LogInfo("CastorShowerLibraryMaker") << "\n Empty G4HitCollection";
       return false;
     }

// Compute Total Energy in CastorFI volume
/*
     for(int ihit = 0; ihit < nentries; ihit++) {
       CaloG4Hit* aHit = (*theCAFI)[ihit];
       totalEnergy += aHit->getEnergyDeposit();
     }
*/
     if (!shower) {
        edm::LogInfo("CastorShowerLibraryMaker") << "Error. NULL pointer to CastorShowerEvent";
        return false;
     }

// Hit position
     math::XYZPoint entry;
     math::XYZPoint position;
     int nHits;
     nHits=0;
     for (int ihit = 0; ihit < nentries; ihit++) {
       CaloG4Hit* aHit  = (*theCAFI)[ihit];
       int hit_particleID = aHit->getTrackID();
       if (MapOfSecondaries[ipart].find(hit_particleID)==MapOfSecondaries[ipart].end()) {
          if (verbosity) edm::LogInfo("CastorShowerLibraryMaker") << "Skipping hit from trackID " << hit_particleID;
          continue;
       }
       volumeID         = aHit->getUnitID();
       double hitEnergy = aHit->getEnergyDeposit();
       en_in_fi        += aHit->getEnergyDeposit();
       float time       = aHit->getTimeSlice();
       int zside, sector, zmodule;
       theCastorNumScheme->unpackIndex(volumeID, zside, sector,zmodule);
       entry    = aHit->getEntry();
       position = aHit->getPosition();
       if (verbosity) edm::LogInfo("CastorShowerLibraryMaker")
          << "\n side , sector , module = " << zside << " , " 
          << sector << " , " << zmodule 
          << "\n nphotons = " << hitEnergy ; 

       if (verbosity) edm::LogInfo("CastorShowerLibraryMaker")
          << "\n packIndex = " 
          << theCastorNumScheme->packIndex(zside, sector,zmodule);

       if(time>100.) {
         edm::LogInfo("CastorShowerLibraryMaker")
            << "\n nentries = " << nentries 
            << "\n     time[" << ihit << "] = " << time
            << "\n  trackID[" << ihit << "] = " << aHit->getTrackID()
            << "\n volumeID[" << ihit << "] = " << volumeID 
            << "\n nphotons[" << ihit << "] = " << hitEnergy 
            << "\n side, sector, module  = " << zside <<", " << sector<<", " << zmodule
            << "\n packIndex " << theCastorNumScheme->packIndex(zside,sector,zmodule)
            << "\n X,Y,Z = " << entry.x() << ","<< entry.y() << "," << entry.z();
       }
       if(nHits==0) {
/*
         edm::LogInfo("CastorShowerLibraryMaker")
            << "\n    entry(x,y,z) = (" << entry.x() << "," 
            << entry.y() << "," << entry.z() << ") \n" 
            << "\n    entry(eta,phi,z) = (" << entry.eta() << "," 
            << entry.phi() << "," << entry.z() << ") \n" 
            << "\n    eta , phi = "  
            << eta << " , " << phi << " \n" ;
*/
          shower->setPrimX(entry.x());
          shower->setPrimY(entry.y());
          shower->setPrimZ(entry.z());
       }
       if (verbosity) edm::LogInfo("CastorShowerLibraryMaker") << "\n    Incident Energy = "  
                                                << aHit->getIncidentEnergy() << " \n" ;


//  CaloG4Hit information 
       shower->setDetID(volumeID);
       shower->setHitPosition(position);
       shower->setNphotons(hitEnergy);
       shower->setTime(time);
       nHits++;
     }
// Write number of hits to CastorShowerEvent instance
     if (nHits==0) {
        edm::LogInfo("CastorShowerLibraryMaker") << "No hits found for this track (trackID=" << ipart << ")." << std::endl;
        return false;
     }
     shower->setNhit(nHits);

     edm::LogInfo("CastorShowerLibraryMaker") << "Filling the SL vector with new element ("<<nHits<<" hits)";
// update the event counters
     return true;
}
int& CastorShowerLibraryMaker::SLnEvtInBinE(int ebin)
{
   if (!SLShowerptr) {
      edm::LogInfo("CastorShowerLibraryMaker") << "\n\nSLnEvtInBinE can be called only after BeginOfEvent\n\n";
      throw SimG4Exception("\n\nNULL Pointer to the shower library.");
   }
   return SLShowerptr->nEvtInBinE.at(ebin);
}

int& CastorShowerLibraryMaker::SLnEvtInBinEta(int ebin, int etabin)
{
   if (!SLShowerptr) {
      edm::LogInfo("CastorShowerLibraryMaker") << "\n\nSLnEvtInBinEta can be called only after BeginOfEvent\n\n";
      throw SimG4Exception("\n\nNULL Pointer to the shower library.");
   }
   return SLShowerptr->nEvtInBinEta.at(ebin).at(etabin);
}

int& CastorShowerLibraryMaker::SLnEvtInBinPhi(int ebin, int etabin, int phibin)
{
   if (!SLShowerptr) {
      edm::LogInfo("CastorShowerLibraryMaker") << "\n\nSLnEvtInBinPhi can be called only after BeginOfEvent\n\n";
      throw SimG4Exception("\n\nNULL Pointer to the shower library.");
   }
   return SLShowerptr->nEvtInBinPhi.at(ebin).at(etabin).at(phibin);
}
bool CastorShowerLibraryMaker::SLisEBinFilled(int ebin)
{
   if (!SLShowerptr) {
      edm::LogInfo("CastorShowerLibraryMaker") << "\n\nSLisEBinFilled can be called only after BeginOfEvent\n\n";
      throw SimG4Exception("\n\nNULL Pointer to the shower library.");
   }
   if (SLShowerptr->nEvtInBinE.at(ebin)<(int)SLShowerptr->nEvtPerBinE) return false;
   return true;
}
bool CastorShowerLibraryMaker::SLisEtaBinFilled(int ebin,int etabin)
{
   if (!SLShowerptr) {
      edm::LogInfo("CastorShowerLibraryMaker") << "\n\nSLisEtaBinFilled can be called only after BeginOfEvent\n\n";
      throw SimG4Exception("\n\nNULL Pointer to the shower library.");
   }
   if (SLShowerptr->nEvtInBinEta.at(ebin).at(etabin)<(int)SLShowerptr->nEvtPerBinEta) return false;
   return true;
}
bool CastorShowerLibraryMaker::SLisPhiBinFilled(int ebin,int etabin,int phibin)
{
   if (!SLShowerptr) {
      edm::LogInfo("CastorShowerLibraryMaker") << "\n\nSLisPhiBinFilled can be called only after BeginOfEvent\n\n";
      throw SimG4Exception("\n\nNULL Pointer to the shower library.");
   }
   if (SLShowerptr->nEvtInBinPhi.at(ebin).at(etabin).at(phibin)<(int)SLShowerptr->nEvtPerBinPhi) return false;
   return true;
}
