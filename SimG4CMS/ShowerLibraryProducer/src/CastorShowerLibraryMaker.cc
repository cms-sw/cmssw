// -*- C++ -*-
//
// Package:     ShowerLibraryProducer
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
#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "SimG4Core/Notification/interface/SimG4Exception.h"
#include "SimG4CMS/Calo/interface/CaloG4Hit.h"
#include "SimG4CMS/Calo/interface/CaloG4HitCollection.h"
#include "DataFormats/Math/interface/Point3D.h"

#include "SimG4CMS/ShowerLibraryProducer/interface/CastorShowerLibraryMaker.h"
#include "SimG4CMS/Forward/interface/CastorNumberingScheme.h"

#include "SimDataFormats/CaloHit/interface/CastorShowerEvent.h"

#include "CLHEP/Units/GlobalSystemOfUnits.h"
#include "CLHEP/Units/GlobalPhysicalConstants.h"

#include "TFile.h"
#include <cmath>
#include <iostream>
#include <sstream>
#include <iomanip>
#include <cstdlib>

CastorShowerLibraryMaker::CastorShowerLibraryMaker(const edm::ParameterSet& p)
    : NPGParticle(0),
      DoHadSL(false),
      DoEmSL(false),
      DeActivatePhysicsProcess(false),
      emShower(nullptr),
      hadShower(nullptr) {
  MapOfSecondaries.clear();
  hadInfo = nullptr;
  emInfo = nullptr;
  edm::ParameterSet p_SLM = p.getParameter<edm::ParameterSet>("CastorShowerLibraryMaker");
  verbosity = p_SLM.getParameter<int>("Verbosity");
  eventNtFileName = p_SLM.getParameter<std::string>("EventNtupleFileName");
  hadSLHolder.nEvtPerBinPhi = p_SLM.getParameter<int>("nhadEvents");
  emSLHolder.nEvtPerBinPhi = p_SLM.getParameter<int>("nemEvents");
  hadSLHolder.SLEnergyBins = p_SLM.getParameter<std::vector<double> >("SLhadEnergyBins");
  hadSLHolder.SLEtaBins = p_SLM.getParameter<std::vector<double> >("SLhadEtaBins");
  hadSLHolder.SLPhiBins = p_SLM.getParameter<std::vector<double> >("SLhadPhiBins");
  emSLHolder.SLEnergyBins = p_SLM.getParameter<std::vector<double> >("SLemEnergyBins");
  emSLHolder.SLEtaBins = p_SLM.getParameter<std::vector<double> >("SLemEtaBins");
  emSLHolder.SLPhiBins = p_SLM.getParameter<std::vector<double> >("SLemPhiBins");
  PGParticleIDs = p_SLM.getParameter<std::vector<int> >("PartID");
  DeActivatePhysicsProcess = p_SLM.getParameter<bool>("DeActivatePhysicsProcess");
  MaxPhi = p_SLM.getParameter<double>("SLMaxPhi");
  MaxEta = p_SLM.getParameter<double>("SLMaxEta");
  //
  NPGParticle = PGParticleIDs.size();
  for (unsigned int i = 0; i < PGParticleIDs.size(); i++) {
    switch (std::abs(PGParticleIDs.at(i))) {
      case 11:
      case 22:
        DoEmSL = true;
        break;
      default:
        DoHadSL = true;
    }
  }
  hadSLHolder.nEvtPerBinEta = (hadSLHolder.nEvtPerBinPhi) * (hadSLHolder.SLPhiBins.size());
  hadSLHolder.nEvtPerBinE = (hadSLHolder.nEvtPerBinEta) * (hadSLHolder.SLEtaBins.size());
  emSLHolder.nEvtPerBinEta = (emSLHolder.nEvtPerBinPhi) * (emSLHolder.SLPhiBins.size());
  emSLHolder.nEvtPerBinE = (emSLHolder.nEvtPerBinEta) * (emSLHolder.SLEtaBins.size());

  std::cout << "============================================================================" << std::endl;
  std::cout << "CastorShowerLibraryMaker:: Initialized as observer" << std::endl;
  std::cout << " Event Ntuple will be created" << std::endl;
  std::cout << " Event Ntuple file: " << eventNtFileName << std::endl;
  std::cout << " Number of Hadronic events in E   bins: " << hadSLHolder.nEvtPerBinE << std::endl;
  std::cout << " Number of Hadronic events in Eta bins: " << hadSLHolder.nEvtPerBinEta << std::endl;
  std::cout << " Number of Hadronic events in Phi bins: " << hadSLHolder.nEvtPerBinPhi << std::endl;
  std::cout << " Number of Electromag. events in E   bins: " << emSLHolder.nEvtPerBinE << std::endl;
  std::cout << " Number of Electromag. events in Eta bins: " << emSLHolder.nEvtPerBinEta << std::endl;
  std::cout << " Number of Electromag. events in Phi bins: " << emSLHolder.nEvtPerBinPhi << std::endl;
  std::cout << "============================================================================" << std::endl;
  std::cout << std::endl;

  // Initializing the SL collections
  InitSLHolder(hadSLHolder);
  InitSLHolder(emSLHolder);
}
void CastorShowerLibraryMaker::InitSLHolder(ShowerLib& showerholder) {
  int nBinsE, nBinsEta, nBinsPhi, nEvtPerBinPhi;
  nBinsE = showerholder.SLEnergyBins.size();
  nBinsEta = showerholder.SLEtaBins.size();
  nBinsPhi = showerholder.SLPhiBins.size();
  nEvtPerBinPhi = showerholder.nEvtPerBinPhi;
  //
  // Info
  //
  showerholder.SLInfo.Energy.setNEvts(nEvtPerBinPhi * nBinsPhi * nBinsEta * nBinsE);
  showerholder.SLInfo.Energy.setNEvtPerBin(nEvtPerBinPhi * nBinsPhi * nBinsEta);
  showerholder.SLInfo.Energy.setNBins(nBinsE);
  showerholder.SLInfo.Energy.setBin(showerholder.SLEnergyBins);
  //
  showerholder.SLInfo.Eta.setNEvts(nEvtPerBinPhi * nBinsPhi * nBinsEta);
  showerholder.SLInfo.Eta.setNEvtPerBin(nEvtPerBinPhi * nBinsPhi);
  showerholder.SLInfo.Eta.setNBins(nBinsEta);
  showerholder.SLInfo.Eta.setBin(showerholder.SLEtaBins);
  //
  showerholder.SLInfo.Phi.setNEvts(nEvtPerBinPhi * nBinsPhi);
  showerholder.SLInfo.Phi.setNEvtPerBin(nEvtPerBinPhi);
  showerholder.SLInfo.Phi.setNBins(nBinsPhi);
  showerholder.SLInfo.Phi.setBin(showerholder.SLPhiBins);
  //
  // Shower
  showerholder.SLCollection.assign(nBinsE, std::vector<std::vector<std::vector<CastorShowerEvent> > >());
  showerholder.nEvtInBinE.assign(nBinsE, 0);
  showerholder.nEvtInBinEta.assign(nBinsE, std::vector<int>(0));
  showerholder.nEvtInBinPhi.assign(nBinsE, std::vector<std::vector<int> >());
  for (int i = 0; i < nBinsE; i++) {
    showerholder.SLCollection.at(i).assign(nBinsEta, std::vector<std::vector<CastorShowerEvent> >());
    showerholder.nEvtInBinEta.at(i).assign(nBinsEta, 0);
    showerholder.nEvtInBinPhi.at(i).assign(nBinsEta, std::vector<int>(0));
    for (int j = 0; j < nBinsEta; j++) {
      showerholder.SLCollection.at(i).at(j).assign(nBinsPhi, std::vector<CastorShowerEvent>());
      showerholder.nEvtInBinPhi.at(i).at(j).assign(nBinsPhi, 0);
      for (int k = 0; k < nBinsPhi; k++)
        showerholder.SLCollection.at(i).at(j).at(k).assign(nEvtPerBinPhi, CastorShowerEvent());
    }
  }
}

//===============================================================================================

CastorShowerLibraryMaker::~CastorShowerLibraryMaker() {
  Finish();

  std::cout << "CastorShowerLibraryMaker: End of process" << std::endl;
}

//=================================================================== per EVENT
void CastorShowerLibraryMaker::update(const BeginOfJob* job) {
  std::cout << " CastorShowerLibraryMaker::Starting new job " << std::endl;
}

//==================================================================== per RUN
void CastorShowerLibraryMaker::update(const BeginOfRun* run) {
  std::cout << std::endl << "CastorShowerLibraryMaker: Starting Run" << std::endl;

  std::cout << "CastorShowerLibraryMaker: output event root file created" << std::endl;

  TString eventfilename = eventNtFileName;
  theFile = new TFile(eventfilename, "RECREATE");
  theTree = new TTree("CastorCherenkovPhotons", "Cherenkov Photons");

  Int_t split = 1;
  Int_t bsize = 64000;
  emInfo = new CastorShowerLibraryInfo();
  emShower = new CastorShowerEvent();
  hadInfo = new CastorShowerLibraryInfo();
  hadShower = new CastorShowerEvent();
  // Create Branchs
  theTree->Branch("emShowerLibInfo.", "CastorShowerLibraryInfo", &emInfo, bsize, split);
  theTree->Branch("emParticles.", "CastorShowerEvent", &emShower, bsize, split);
  theTree->Branch("hadShowerLibInfo.", "CastorShowerLibraryInfo", &hadInfo, bsize, split);
  theTree->Branch("hadParticles.", "CastorShowerEvent", &hadShower, bsize, split);

  // set the Info for electromagnetic shower
  // set the energy bins info
  emInfo->Energy.setNEvts(emSLHolder.nEvtPerBinE * emSLHolder.SLEnergyBins.size());
  emInfo->Energy.setNBins(emSLHolder.SLEnergyBins.size());
  emInfo->Energy.setNEvtPerBin(emSLHolder.nEvtPerBinE);
  emInfo->Energy.setBin(emSLHolder.SLEnergyBins);
  // set the eta bins info
  emInfo->Eta.setNEvts(emSLHolder.nEvtPerBinEta * emSLHolder.SLEtaBins.size());
  emInfo->Eta.setNBins(emSLHolder.SLEtaBins.size());
  emInfo->Eta.setNEvtPerBin(emSLHolder.nEvtPerBinEta);
  emInfo->Eta.setBin(emSLHolder.SLEtaBins);
  // set the eta bins info
  emInfo->Phi.setNEvts(emSLHolder.nEvtPerBinPhi * emSLHolder.SLPhiBins.size());
  emInfo->Phi.setNBins(emSLHolder.SLPhiBins.size());
  emInfo->Phi.setNEvtPerBin(emSLHolder.nEvtPerBinPhi);
  emInfo->Phi.setBin(emSLHolder.SLPhiBins);
  // The same for the hadronic shower
  // set the energy bins info
  hadInfo->Energy.setNEvts(hadSLHolder.nEvtPerBinE * hadSLHolder.SLEnergyBins.size());
  hadInfo->Energy.setNBins(hadSLHolder.SLEnergyBins.size());
  hadInfo->Energy.setNEvtPerBin(hadSLHolder.nEvtPerBinE);
  hadInfo->Energy.setBin(hadSLHolder.SLEnergyBins);
  // set the eta bins info
  hadInfo->Eta.setNEvts(hadSLHolder.nEvtPerBinEta * hadSLHolder.SLEtaBins.size());
  hadInfo->Eta.setNBins(hadSLHolder.SLEtaBins.size());
  hadInfo->Eta.setNEvtPerBin(hadSLHolder.nEvtPerBinEta);
  hadInfo->Eta.setBin(hadSLHolder.SLEtaBins);
  // set the eta bins info
  hadInfo->Phi.setNEvts(hadSLHolder.nEvtPerBinPhi * hadSLHolder.SLPhiBins.size());
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
void CastorShowerLibraryMaker::update(const BeginOfEvent* evt) {
  eventIndex++;
  stepIndex = 0;
  InsideCastor = false;
  PrimaryMomentum.clear();
  PrimaryPosition.clear();
  int NAccepted = 0;
  // reset the pointers to the shower objects
  SLShowerptr = nullptr;
  MapOfSecondaries.clear();
  thePrims.clear();
  G4EventManager* e_mgr = G4EventManager::GetEventManager();
  if (IsSLReady()) {
    printSLstatus(-1, -1, -1);
    update((EndOfRun*)nullptr);
    return;
  }

  thePrims = GetPrimary((*evt)());
  for (unsigned int i = 0; i < thePrims.size(); i++) {
    G4PrimaryParticle* thePrim = thePrims.at(i);
    int particleType = thePrim->GetPDGcode();

    std::string SLType("");
    if (particleType == 11) {
      SLShowerptr = &emSLHolder;
      SLType = "Electromagnetic";
    } else {
      SLShowerptr = &hadSLHolder;
      SLType = "Hadronic";
    }
    double px = 0., py = 0., pz = 0., pInit = 0., eta = 0., phi = 0.;
    GetKinematics(thePrim, px, py, pz, pInit, eta, phi);
    int ebin = FindEnergyBin(pInit);
    int etabin = FindEtaBin(eta);
    int phibin = FindPhiBin(phi);
    if (verbosity)
      std::cout << "\n========================================================================"
                << "\nBeginOfEvent: E   : " << pInit << "\t  bin : " << ebin << "\n              Eta : " << eta
                << "\t  bin : " << etabin << "\n              Phi : " << phi << "\t  bin : " << phibin
                << "\n========================================================================" << std::endl;

    if (ebin < 0 || etabin < 0 || phibin < 0)
      continue;
    bool accept = false;
    if (!(SLacceptEvent(ebin, etabin, phibin))) {
      /*
// To increase the chance of a particle arriving at CASTOR inside a not full bin,
// check if there is available phase space in the neighboring bins
        unsigned int ebin_min = std::max(0,ebin-3);
        unsigned int eta_bin_min = std::max(0,etabin-2);
        unsigned int eta_bin_max = std::min(etabin,etabin+2);
        unsigned int phi_bin_min = std::max(0,phibin-2);
        unsigned int phi_bin_max = std::min(phibin,phibin+2);
        for(unsigned int i_ebin=ebin_min;i_ebin<=(unsigned int)ebin;i_ebin++) {
           for (unsigned int i_etabin=eta_bin_min;i_etabin<=eta_bin_max;i_etabin++) {
               for (unsigned int i_phibin=phi_bin_min;i_phibin<=phi_bin_max;i_phibin++) {
                   if (SLacceptEvent((int)i_ebin,(int)i_etabin,(int)i_phibin)) {accept=true;break;}
               }
               if (accept) break;
           }
           if (accept) break;
        }
*/
      if (!accept)
        edm::LogInfo("CastorShowerLibraryMaker")
            << "Event not accepted for ebin=" << ebin << ",etabin=" << etabin << ",phibin=" << phibin << std::endl;
    } else {
      accept = true;
    }
    if (accept)
      NAccepted++;
  }

  if (NAccepted == 0) {
    const_cast<G4Event*>((*evt)())->SetEventAborted();
    const_cast<G4Event*>((*evt)())->KeepTheEvent((G4bool) false);
    e_mgr->AbortCurrentEvent();
  }
  SLShowerptr = nullptr;
  //
  std::cout << "CastorShowerLibraryMaker: Processing Event Number: " << eventIndex << std::endl;
}

//=================================================================== per STEP
void CastorShowerLibraryMaker::update(const G4Step* aStep) {
  static thread_local int CurrentPrimary = 0;
  G4Track* trk = aStep->GetTrack();
  int pvec_size;
  if (trk->GetCurrentStepNumber() == 1) {
    if (trk->GetParentID() == 0) {
      CurrentPrimary = (int)trk->GetDynamicParticle()->GetPDGcode();
      if (CurrentPrimary == 0)
        SimG4Exception("CastorShowerLibraryMaker::update(G4Step) -> Primary particle undefined");
      InsideCastor = false;
      // Deactivate the physics process
      if (DeActivatePhysicsProcess) {
        G4ProcessManager* p_mgr = trk->GetDefinition()->GetProcessManager();
        G4ProcessVector* pvec = p_mgr->GetProcessList();
        pvec_size = pvec->size();
        for (int i = 0; i < pvec_size; i++) {
          G4VProcess* proc = (*pvec)(i);
          if (proc->GetProcessName() != "Transportation" && proc->GetProcessName() != "Decay") {
            std::cout << "DeActivating process: " << proc->GetProcessName() << std::endl;
            p_mgr->SetProcessActivation(proc, false);
          }
        }
      }
      // move track to z of CASTOR
      G4ThreeVector pos;
      pos.setZ(-14390);
      double t = std::abs((pos.z() - trk->GetPosition().z())) / trk->GetVelocity();
      double r = (pos.z() - trk->GetPosition().z()) / trk->GetMomentum().cosTheta();
      pos.setX(r * sin(trk->GetMomentum().theta()) * cos(trk->GetMomentum().phi()) + trk->GetPosition().x());
      pos.setY(r * sin(trk->GetMomentum().theta()) * sin(trk->GetMomentum().phi()) + trk->GetPosition().y());
      trk->SetPosition(pos);
      trk->SetGlobalTime(trk->GetGlobalTime() + t);
      trk->AddTrackLength(r);
    } else if (!InsideCastor) {
      std::cout << "CastorShowerLibraryMaker::update(G4Step) -> Killing spurious track" << std::endl;
      trk->SetTrackStatus(fKillTrackAndSecondaries);
      return;
    }
    MapOfSecondaries[CurrentPrimary].insert((int)trk->GetTrackID());
  }
  // Checks if primary already inside CASTOR
  std::string CurVolume = trk->GetVolume()->GetName();
  if (!InsideCastor && (
                           //CurVolume=="C3EF"||CurVolume=="C4EF"||CurVolume=="CAEL"||
                           //CurVolume=="CAHL"||CurVolume=="C3HF"||CurVolume=="C4HF")) {
                           //CurVolume=="CastorB"||
                           CurVolume == "CAST")) {
    //CurVolume=="CAIR")) {
    InsideCastor = true;
    // Activate the physics process
    if (trk->GetParentID() == 0 && DeActivatePhysicsProcess) {
      G4ProcessManager* p_mgr = trk->GetDefinition()->GetProcessManager();
      G4ProcessVector* pvec = p_mgr->GetProcessList();
      pvec_size = pvec->size();
      for (int i = 0; i < pvec_size; i++) {
        G4VProcess* proc = (*pvec)(i);
        if (proc->GetProcessName() != "Transportation" && proc->GetProcessName() != "Decay") {
          std::cout << "  Activating process: " << proc->GetProcessName() << std::endl;
          p_mgr->SetProcessActivation(proc, true);
        }
      }
    }
    //PrimaryMomentum[CurrentPrimary]=aStep->GetPreStepPoint()->GetMomentum();
    // check fiducial eta and phi
    if (trk->GetMomentum().phi() > MaxPhi || trk->GetMomentum().eta() > MaxEta) {
      trk->SetTrackStatus(fKillTrackAndSecondaries);
      InsideCastor = false;
      return;
    }
    PrimaryMomentum[CurrentPrimary] = trk->GetMomentum();
    PrimaryPosition[CurrentPrimary] = trk->GetPosition();
    KillSecondaries(aStep);
    return;
  }
  // Kill the secondaries if they have been produced before entering castor
  if (CurrentPrimary != 0 && trk->GetParentID() == 0 && !InsideCastor) {
    KillSecondaries(aStep);
    if (verbosity) {
      double pre_phi = aStep->GetPreStepPoint()->GetMomentum().phi();
      double cur_phi = trk->GetMomentum().phi();
      if (pre_phi != cur_phi) {
        std::cout << "Primary track phi :  " << pre_phi << " changed in current step: " << cur_phi
                  << " by processes:" << std::endl;
        const G4VProcess* proc = aStep->GetPreStepPoint()->GetProcessDefinedStep();
        std::cout << "           " << proc->GetProcessName() << "  In volume " << CurVolume << std::endl;
      }
    }
  }

  //==============================================
  /*
*/
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
void CastorShowerLibraryMaker::update(const EndOfEvent* evt) {
  // check if the job is done!
  if ((*evt)()->IsAborted()) {
    std::cout << "\n========================================================================"
              << "\nEndOfEvent: EVENT ABORTED"
              << "\n========================================================================" << std::endl;
    return;
  }
  //DynamicRangeFlatRandomEGunProducer* pgun = edm::DynamicRangeFlatRandomEGunKernel::get_instance();
  //std::cout << pgun->EGunMaxE() << std::endl;
  /*
  std::cout << "Minimum energy in Particle Gun : " << pgun->EGunMinE() << "\n"
            << "Maximum energy in Particle Gun : " << pgun->EGunMaxE() << std::endl;
*/
  if (verbosity)
    std::cout << "CastorShowerLibraryMaker: End of Event: " << eventIndex << std::endl;
  // Get the pointer to the primary particle
  if (thePrims.empty()) {
    edm::LogInfo("CastorShowerLibraryMaker") << "No valid primary particle found. Skipping event" << std::endl;
    return;
  }
  // access to the G4 hit collections
  G4HCofThisEvent* allHC = (*evt)()->GetHCofThisEvent();
  int CAFIid = G4SDManager::GetSDMpointer()->GetCollectionID("CastorFI");
  CaloG4HitCollection* theCAFI = (CaloG4HitCollection*)allHC->GetHC(CAFIid);
  if (verbosity)
    edm::LogInfo("CastorShowerLibraryMaker") << " update(*evt) --> accessed all HC ";
  edm::LogInfo("CastorShowerLibraryMaker") << "Found " << theCAFI->entries() << " hits in G4HitCollection";
  if (theCAFI->entries() == 0) {
    edm::LogInfo("CastorShowerLibraryMaker") << "\n Empty G4HitCollection";
    return;
  }

  // Loop over primaries
  int NEvtAccepted = 0;
  int NHitInEvent = 0;
  for (unsigned int i = 0; i < thePrims.size(); i++) {
    G4PrimaryParticle* thePrim = thePrims.at(i);
    if (!thePrim) {
      edm::LogInfo("CastorShowerLibraryMaker") << "nullptr Pointer to the primary" << std::endl;
      continue;
    }
    // Check primary particle type
    int particleType = thePrim->GetPDGcode();

    // set the pointer to the shower collection
    std::string SLType("");
    if (particleType == 11) {
      SLShowerptr = &emSLHolder;
      SLType = "Electromagnetic";
    } else {
      SLShowerptr = &hadSLHolder;
      SLType = "Hadronic";
    }
    edm::LogInfo("CastorShowerLibraryMaker") << "\n Primary (thePrim) trackID is " << thePrim->GetTrackID() << "\n";

    // Obtain primary particle's initial momentum (pInit)
    double px = 0., py = 0., pz = 0., pInit = 0., eta = 0., phi = 0.;
    GetKinematics(particleType, px, py, pz, pInit, eta, phi);
    // Check if current event falls into any bin
    // first: energy
    if (pInit == 0) {
      edm::LogInfo("CastorShowerLibraryMaker") << "Primary did not hit CASTOR" << std::endl;
      continue;
    }
    int ebin = FindEnergyBin(pInit);
    int etabin = FindEtaBin(eta);
    int phibin = FindPhiBin(phi);
    std::cout << SLType << std::endl;
    printSLstatus(ebin, etabin, phibin);
    if (!SLacceptEvent(ebin, etabin, phibin)) {
      edm::LogInfo("CastorShowerLibraryMaker")
          << "Event not accepted for ebin=" << ebin << ",etabin=" << etabin << ",phibin=" << phibin << "(" << pInit
          << "," << eta << "," << phi << ")" << std::endl;
      continue;
    }
    //
    // event passed. Fill the vector accordingly
    //
    // Look for the Hit Collection
    edm::LogInfo("CastorShowerLibraryMaker")
        << "\n CastorShowerLibraryMaker::update(EndOfEvent * evt) - event #" << (*evt)()->GetEventID();

    /*
     std::cout << "Number of collections : " << allHC->GetNumberOfCollections() << std::endl;
     for(int ii = 0;ii<allHC->GetNumberOfCollections();ii++) 
        std::cout << "Name of collection " << ii << " : " << allHC->GetHC(ii)->GetName() << std::endl;
*/

    CastorShowerEvent* shower = nullptr;
    int cur_evt_idx = SLShowerptr->nEvtInBinPhi.at(ebin).at(etabin).at(phibin);
    shower = &(SLShowerptr->SLCollection.at(ebin).at(etabin).at(phibin).at(cur_evt_idx));

    // Get Hit information
    if (FillShowerEvent(theCAFI, shower, particleType)) {
      //  Primary particle information
      /*
        edm::LogInfo("CastorShowerLibraryMaker") << "New SL event: Primary = " << particleType
             << "; Energy = " << pInit << "; Eta = " << eta << "; Phi = " << phi
             << "; Nhits = " << shower->getNhit() << std::endl;
*/
      shower->setPrimE(pInit);
      shower->setPrimEta(eta);
      shower->setPrimPhi(phi);
      shower->setPrimX(PrimaryPosition[particleType].x());
      shower->setPrimY(PrimaryPosition[particleType].y());
      shower->setPrimZ(PrimaryPosition[particleType].z());
      SLnEvtInBinE(ebin)++;
      SLnEvtInBinEta(ebin, etabin)++;
      SLnEvtInBinPhi(ebin, etabin, phibin)++;
      NHitInEvent += shower->getNhit();
      NEvtAccepted++;
    } else {
      shower->Clear();
    }
  }
  // Check for unassociated energy
  int thecafi_entries = theCAFI->entries();
  if (NEvtAccepted == int(thePrims.size()) && thecafi_entries != NHitInEvent) {
    std::cout << "WARNING: Inconsistent Number of Hits -> Hits in collection: " << theCAFI->entries()
              << "   Hits in the showers: " << NHitInEvent << std::endl;
    double miss_energy = 0;
    double tot_energy = 0;
    GetMissingEnergy(theCAFI, miss_energy, tot_energy);
    if (miss_energy > 0) {
      std::cout << "Total missing energy: " << miss_energy << " for an incident energy: " << tot_energy << std::endl;
    }
  }

  /*
  for (int i=emSLHolder.SLEnergyBins.size()-1;i>0;i--) {
      if (emSLHolder.nEvtInBinE.at(i)==(int)emSLHolder.nEvtPerBinE) {
         std::ostringstream out;
         out << emSLHolder.SLEnergyBins.at(i);
         cout << "Bin Limit: " << out.str() << std::endl;
         setenv("CASTOR_SL_PG_MAXE",out.str().c_str(),1);
       }
       break;
   }
*/
  //int iEvt = (*evt)()->GetEventID();
  //double xint;
  /*
  if (modf(log10(iEvt),&xint)==0) 
    std::cout << " CastorShowerLibraryMaker Event " << iEvt << std::endl;
*/
  // std::cout << std::endl << "===>>> Done writing user histograms " << std::endl;
}

//========================= End of RUN ======================
void CastorShowerLibraryMaker::update(const EndOfRun* run) {
  // Fill the tree with the collected objects
  if (!IsSLReady())
    SimG4Exception("\n\nShower Library 	NOT READY.\n\n");

  unsigned int ibine, ibineta, ibinphi, ievt;  // indexes for em shower
  unsigned int jbine, jbineta, jbinphi, jevt;  // indexes for had shower

  ibine = ibineta = ibinphi = ievt = jbine = jbineta = jbinphi = jevt = 0;

  int nEvtInTree = 0;
  int nEMevt = emSLHolder.nEvtPerBinE * emSLHolder.SLEnergyBins.size();
  int nHadevt = hadSLHolder.nEvtPerBinE * hadSLHolder.SLEnergyBins.size();
  int maxEvtInTree = std::max(nEMevt, nHadevt);

  emInfo = &emSLHolder.SLInfo;
  hadInfo = &hadSLHolder.SLInfo;

  while (nEvtInTree < maxEvtInTree) {
    if (emShower)
      emShower->Clear();
    if (hadShower)
      hadShower->Clear();
    while (ibine < emSLHolder.SLEnergyBins.size() && nEMevt > 0) {
      emShower = &(emSLHolder.SLCollection.at(ibine).at(ibineta).at(ibinphi).at(ievt));
      ievt++;
      if (ievt == emSLHolder.nEvtPerBinPhi) {
        ievt = 0;
        ibinphi++;
      }
      if (ibinphi == emSLHolder.SLPhiBins.size()) {
        ibinphi = 0;
        ibineta++;
      }
      if (ibineta == emSLHolder.SLEtaBins.size()) {
        ibineta = 0;
        ibine++;
      }
      break;
    }
    while (jbine < hadSLHolder.SLEnergyBins.size() && nHadevt > 0) {
      hadShower = &(hadSLHolder.SLCollection.at(jbine).at(jbineta).at(jbinphi).at(jevt));
      jevt++;
      if (jevt == hadSLHolder.nEvtPerBinPhi) {
        jevt = 0;
        jbinphi++;
      }
      if (jbinphi == hadSLHolder.SLPhiBins.size()) {
        jbinphi = 0;
        jbineta++;
      }
      if (jbineta == hadSLHolder.SLEtaBins.size()) {
        jbineta = 0;
        jbine++;
      }
      break;
    }
    theTree->Fill();
    nEvtInTree++;
    if (nEvtInTree == 1) {
      theTree->SetBranchStatus("emShowerLibInfo.", false);
      theTree->SetBranchStatus("hadShowerLibInfo.", false);
    }
  }
  // check if run is nullptr and exit
  if (run == nullptr)
    throw SimG4Exception("\n\nNumber of needed trigger events reached in CastorShowerLibraryMaker\n\n");
}

//============================================================
void CastorShowerLibraryMaker::Finish() {
  // if (doNTcastorevent) {

  theFile->cd();
  theTree->Write("", TObject::kOverwrite);
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
    throw SimG4Exception("\n\nnullptr Pointer to the shower library.\n\n");
  }
  const std::vector<double>& SLenergies = SLShowerptr->SLEnergyBins;
  if (energy >= SLenergies.back())
    return SLenergies.size() - 1;

  unsigned int i = 0;
  for (; i < SLenergies.size() - 1; i++)
    if (energy >= SLenergies.at(i) && energy < SLenergies.at(i + 1))
      return (int)i;

  // now i points to the last but 1 bin
  if (energy >= SLenergies.at(i))
    return (int)i;
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
    throw SimG4Exception("\n\nnullptr Pointer to the shower library.\n\n");
  }
  const std::vector<double>& SLetas = SLShowerptr->SLEtaBins;
  if (eta >= SLetas.back())
    return SLetas.size() - 1;
  unsigned int i = 0;
  for (; i < SLetas.size() - 1; i++)
    if (eta >= SLetas.at(i) && eta < SLetas.at(i + 1))
      return (int)i;
  // now i points to the last but 1 bin
  if (eta >= SLetas.at(i))
    return (int)i;
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
    throw SimG4Exception("\n\nnullptr Pointer to the shower library.\n\n");
  }
  const std::vector<double>& SLphis = SLShowerptr->SLPhiBins;
  if (phi >= SLphis.back())
    return SLphis.size() - 1;
  unsigned int i = 0;
  for (; i < SLphis.size() - 1; i++)
    if (phi >= SLphis.at(i) && phi < SLphis.at(i + 1))
      return (int)i;
  // now i points to the last but 1 bin
  if (phi >= SLphis.at(i))
    return (int)i;
  // phi outside bin range
  return -1;
}
bool CastorShowerLibraryMaker::IsSLReady() {
  // at this point, the pointer to the shower library should be nullptr
  if (SLShowerptr) {
    edm::LogInfo("CastorShowerLibraryMaker") << "\n\nIsSLReady must be called when a new event starts.\n\n";
    throw SimG4Exception("\n\nNOT nullptr Pointer to the shower library.\n\n");
  }
  // it is enough to check if all the energy bin is filled
  if (DoEmSL) {
    SLShowerptr = &emSLHolder;
    for (unsigned int i = 0; i < SLShowerptr->SLEnergyBins.size(); i++) {
      if (!SLisEBinFilled(i)) {
        SLShowerptr = nullptr;
        return false;
      }
    }
  }
  if (DoHadSL) {
    SLShowerptr = &hadSLHolder;
    for (unsigned int i = 0; i < SLShowerptr->SLEnergyBins.size(); i++) {
      if (!SLisEBinFilled(i)) {
        SLShowerptr = nullptr;
        return false;
      }
    }
  }
  SLShowerptr = nullptr;
  return true;
}
void CastorShowerLibraryMaker::GetKinematics(
    int thePrim, double& px, double& py, double& pz, double& pInit, double& eta, double& phi) {
  if (thePrim == 0)
    return;
  if (PrimaryMomentum.find(thePrim) == PrimaryMomentum.end())
    return;
  px = PrimaryMomentum[thePrim].x() / GeV;
  py = PrimaryMomentum[thePrim].y() / GeV;
  pz = PrimaryMomentum[thePrim].z() / GeV;
  pInit = PrimaryMomentum[thePrim].mag() / GeV;
  if (pInit == 0)
    return;
  double costheta = pz / pInit;
  double theta = acos(std::min(std::max(costheta, double(-1.)), double(1.)));
  eta = -log(tan(theta / 2.0));
  phi = (px == 0 && py == 0) ? 0 : atan2(py, px);  // the recommended way of calculating phi
  phi = PrimaryMomentum[thePrim].phi();
}
void CastorShowerLibraryMaker::GetKinematics(
    G4PrimaryParticle* thePrim, double& px, double& py, double& pz, double& pInit, double& eta, double& phi) {
  px = py = pz = phi = eta = 0.0;
  if (thePrim == nullptr)
    return;
  px = thePrim->GetMomentum().x() / GeV;
  py = thePrim->GetMomentum().y() / GeV;
  pz = thePrim->GetMomentum().z() / GeV;
  pInit = thePrim->GetMomentum().mag() / GeV;
  //pInit = sqrt(pow(px,2.)+pow(py,2.)+pow(pz,2.));
  if (pInit == 0)
    return;
  double costheta = pz / pInit;
  double theta = acos(std::min(std::max(costheta, double(-1.)), double(1.)));
  eta = -log(tan(theta / 2.0));
  phi = (px == 0 && py == 0) ? 0 : atan2(py, px);  // the recommended way of calculating phi
  phi = thePrim->GetMomentum().phi();
  //if (px!=0) phi=atan(py/px);
}
std::vector<G4PrimaryParticle*> CastorShowerLibraryMaker::GetPrimary(const G4Event* evt) {
  // Find Primary info:
  int trackID = 0;
  std::vector<G4PrimaryParticle*> thePrims;
  G4PrimaryParticle* thePrim = nullptr;
  G4int nvertex = evt->GetNumberOfPrimaryVertex();
  edm::LogInfo("CastorShowerLibraryMaker") << "Event has " << nvertex << " vertex";
  if (nvertex != 1) {
    edm::LogInfo("CastorShowerLibraryMaker") << "CastorShowerLibraryMaker::GetPrimary ERROR: no vertex";
    return thePrims;
  }

  for (int i = 0; i < nvertex; i++) {
    G4PrimaryVertex* avertex = evt->GetPrimaryVertex(i);
    if (avertex == nullptr) {
      edm::LogInfo("CastorShowerLibraryMaker") << "CastorShowerLibraryMaker::GetPrimary ERROR: pointer to vertex = 0";
      continue;
    }
    unsigned int npart = avertex->GetNumberOfParticle();
    if (npart != NPGParticle)
      continue;
    for (unsigned int j = 0; j < npart; j++) {
      unsigned int k = 0;
      //int test_pID = 0;
      trackID = j;
      thePrim = avertex->GetPrimary(trackID);
      while (k < NPGParticle && PGParticleIDs.at(k++) != thePrim->GetPDGcode()) {
        ;
      };
      if (k > NPGParticle)
        continue;  // ID not found in the requested particles
      thePrims.push_back(thePrim);
    }
  }
  return thePrims;
}
void CastorShowerLibraryMaker::printSLstatus(int ebin, int etabin, int phibin) {
  if (!SLShowerptr) {
    edm::LogInfo("CastorShowerLibraryInfo") << "nullptr shower pointer. Printing both";
    std::cout << "Electromagnetic" << std::endl;
    SLShowerptr = &emSLHolder;
    this->printSLstatus(ebin, etabin, phibin);
    std::cout << "Hadronic" << std::endl;
    SLShowerptr = &hadSLHolder;
    this->printSLstatus(ebin, etabin, phibin);
    SLShowerptr = nullptr;
    return;
  }
  int nBinsE = SLShowerptr->SLEnergyBins.size();
  int nBinsEta = SLShowerptr->SLEtaBins.size();
  int nBinsPhi = SLShowerptr->SLPhiBins.size();
  std::vector<double> SLenergies = SLShowerptr->SLEnergyBins;
  for (int n = 0; n < 11 + (nBinsEta * nBinsPhi); n++)
    std::cout << "=";
  std::cout << std::endl;
  for (int i = 0; i < nBinsE; i++) {
    std::cout << "E bin " << std::setw(6) << SLenergies.at(i) << " : ";
    for (int j = 0; j < nBinsEta; j++) {
      for (int k = 0; k < nBinsPhi; k++) {
        (SLisPhiBinFilled(i, j, k)) ? std::cout << "1" : std::cout << "-";
      }
      if (j < nBinsEta - 1)
        std::cout << "|";
    }
    std::cout << " (" << SLnEvtInBinE(i) << " events)";
    std::cout << std::endl;
    if (ebin != i)
      continue;
    std::cout << "               ";
    for (int j = 0; j < nBinsEta; j++) {
      for (int k = 0; k < nBinsPhi; k++) {
        (ebin == i && etabin == j && phibin == k) ? std::cout << "^" : std::cout << " ";
      }
      if (j < nBinsEta - 1)
        std::cout << " ";
    }
    std::cout << std::endl;
  }
  for (int n = 0; n < 11 + (nBinsEta * nBinsPhi); n++)
    std::cout << "=";
  std::cout << std::endl;
}
bool CastorShowerLibraryMaker::SLacceptEvent(int ebin, int etabin, int phibin) {
  if (SLShowerptr == nullptr) {
    edm::LogInfo("CastorShowerLibraryMaker::SLacceptEvent:") << "Error. nullptr pointer to CastorShowerEvent";
    return false;
  }
  if (ebin < 0 || ebin >= int(SLShowerptr->SLEnergyBins.size()))
    return false;
  if (SLisEBinFilled(ebin))
    return false;

  if (etabin < 0 || etabin >= int(SLShowerptr->SLEtaBins.size()))
    return false;
  if (SLisEtaBinFilled(ebin, etabin))
    return false;

  if (phibin < 0 || phibin >= int(SLShowerptr->SLPhiBins.size()))
    return false;
  if (SLisPhiBinFilled(ebin, etabin, phibin))
    return false;
  return true;
}
bool CastorShowerLibraryMaker::FillShowerEvent(CaloG4HitCollection* theCAFI, CastorShowerEvent* shower, int ipart) {
  unsigned int volumeID = 0;
  double en_in_fi = 0.;
  //double totalEnergy = 0;

  int nentries = theCAFI->entries();

  // Compute Total Energy in CastorFI volume
  /*
     for(int ihit = 0; ihit < nentries; ihit++) {
       CaloG4Hit* aHit = (*theCAFI)[ihit];
       totalEnergy += aHit->getEnergyDeposit();
     }
*/
  if (!shower) {
    edm::LogInfo("CastorShowerLibraryMaker") << "Error. nullptr pointer to CastorShowerEvent";
    return false;
  }

  CastorNumberingScheme* theCastorNumScheme = new CastorNumberingScheme();
  // Hit position
  math::XYZPoint entry;
  math::XYZPoint position;
  int nHits;
  nHits = 0;
  for (int ihit = 0; ihit < nentries; ihit++) {
    CaloG4Hit* aHit = (*theCAFI)[ihit];
    int hit_particleID = aHit->getTrackID();
    if (MapOfSecondaries[ipart].find(hit_particleID) == MapOfSecondaries[ipart].end()) {
      if (verbosity)
        edm::LogInfo("CastorShowerLibraryMaker") << "Skipping hit from trackID " << hit_particleID;
      continue;
    }
    volumeID = aHit->getUnitID();
    double hitEnergy = aHit->getEnergyDeposit();
    en_in_fi += aHit->getEnergyDeposit();
    float time = aHit->getTimeSlice();
    int zside, sector, zmodule;
    theCastorNumScheme->unpackIndex(volumeID, zside, sector, zmodule);
    entry = aHit->getEntry();
    position = aHit->getPosition();
    if (verbosity)
      edm::LogInfo("CastorShowerLibraryMaker") << "\n side , sector , module = " << zside << " , " << sector << " , "
                                               << zmodule << "\n nphotons = " << hitEnergy;

    if (verbosity)
      edm::LogInfo("CastorShowerLibraryMaker")
          << "\n packIndex = " << theCastorNumScheme->packIndex(zside, sector, zmodule);

    if (verbosity && time > 100.) {
      edm::LogInfo("CastorShowerLibraryMaker")
          << "\n nentries = " << nentries << "\n     time[" << ihit << "] = " << time << "\n  trackID[" << ihit
          << "] = " << aHit->getTrackID() << "\n volumeID[" << ihit << "] = " << volumeID << "\n nphotons[" << ihit
          << "] = " << hitEnergy << "\n side, sector, module  = " << zside << ", " << sector << ", " << zmodule
          << "\n packIndex " << theCastorNumScheme->packIndex(zside, sector, zmodule) << "\n X,Y,Z = " << entry.x()
          << "," << entry.y() << "," << entry.z();
    }
    if (verbosity)
      edm::LogInfo("CastorShowerLibraryMaker") << "\n    Incident Energy = " << aHit->getIncidentEnergy() << " \n";

    //  CaloG4Hit information
    shower->setDetID(volumeID);
    shower->setHitPosition(position);
    shower->setNphotons(hitEnergy);
    shower->setTime(time);
    nHits++;
  }
  // Write number of hits to CastorShowerEvent instance
  if (nHits == 0) {
    edm::LogInfo("CastorShowerLibraryMaker") << "No hits found for this track (trackID=" << ipart << ")." << std::endl;
    if (theCastorNumScheme)
      delete theCastorNumScheme;
    return false;
  }
  shower->setNhit(nHits);

  // update the event counters
  if (theCastorNumScheme)
    delete theCastorNumScheme;
  return true;
}
int& CastorShowerLibraryMaker::SLnEvtInBinE(int ebin) {
  if (!SLShowerptr) {
    edm::LogInfo("CastorShowerLibraryMaker") << "\n\nSLnEvtInBinE can be called only after BeginOfEvent\n\n";
    throw SimG4Exception("\n\nnullptr Pointer to the shower library.");
  }
  return SLShowerptr->nEvtInBinE.at(ebin);
}

int& CastorShowerLibraryMaker::SLnEvtInBinEta(int ebin, int etabin) {
  if (!SLShowerptr) {
    edm::LogInfo("CastorShowerLibraryMaker") << "\n\nSLnEvtInBinEta can be called only after BeginOfEvent\n\n";
    throw SimG4Exception("\n\nnullptr Pointer to the shower library.");
  }
  return SLShowerptr->nEvtInBinEta.at(ebin).at(etabin);
}

int& CastorShowerLibraryMaker::SLnEvtInBinPhi(int ebin, int etabin, int phibin) {
  if (!SLShowerptr) {
    edm::LogInfo("CastorShowerLibraryMaker") << "\n\nSLnEvtInBinPhi can be called only after BeginOfEvent\n\n";
    throw SimG4Exception("\n\nnullptr Pointer to the shower library.");
  }
  return SLShowerptr->nEvtInBinPhi.at(ebin).at(etabin).at(phibin);
}
bool CastorShowerLibraryMaker::SLisEBinFilled(int ebin) {
  if (!SLShowerptr) {
    edm::LogInfo("CastorShowerLibraryMaker") << "\n\nSLisEBinFilled can be called only after BeginOfEvent\n\n";
    throw SimG4Exception("\n\nnullptr Pointer to the shower library.");
  }
  if (SLShowerptr->nEvtInBinE.at(ebin) < (int)SLShowerptr->nEvtPerBinE)
    return false;
  return true;
}
bool CastorShowerLibraryMaker::SLisEtaBinFilled(int ebin, int etabin) {
  if (!SLShowerptr) {
    edm::LogInfo("CastorShowerLibraryMaker") << "\n\nSLisEtaBinFilled can be called only after BeginOfEvent\n\n";
    throw SimG4Exception("\n\nnullptr Pointer to the shower library.");
  }
  if (SLShowerptr->nEvtInBinEta.at(ebin).at(etabin) < (int)SLShowerptr->nEvtPerBinEta)
    return false;
  return true;
}
bool CastorShowerLibraryMaker::SLisPhiBinFilled(int ebin, int etabin, int phibin) {
  if (!SLShowerptr) {
    edm::LogInfo("CastorShowerLibraryMaker") << "\n\nSLisPhiBinFilled can be called only after BeginOfEvent\n\n";
    throw SimG4Exception("\n\nnullptr Pointer to the shower library.");
  }
  if (SLShowerptr->nEvtInBinPhi.at(ebin).at(etabin).at(phibin) < (int)SLShowerptr->nEvtPerBinPhi)
    return false;
  return true;
}
void CastorShowerLibraryMaker::KillSecondaries(const G4Step* aStep) {
  const G4TrackVector* p_sec = aStep->GetSecondary();
  for (int i = 0; i < int(p_sec->size()); i++) {
    /*if (verbosity)*/ std::cout << "Killing track ID " << p_sec->at(i)->GetTrackID() << " and its secondaries"
                                 << " Produced by Process " << p_sec->at(i)->GetCreatorProcess()->GetProcessName()
                                 << " in the volume " << aStep->GetTrack()->GetVolume()->GetName() << std::endl;
    p_sec->at(i)->SetTrackStatus(fKillTrackAndSecondaries);
  }
}

void CastorShowerLibraryMaker::GetMissingEnergy(CaloG4HitCollection* theCAFI, double& miss_energy, double& tot_energy) {
  // Get the total deposited energy and the one from hit not associated to a primary
  miss_energy = 0;
  tot_energy = 0;
  int nhits = theCAFI->entries();
  for (int ihit = 0; ihit < nhits; ihit++) {
    int id = (*theCAFI)[ihit]->getTrackID();
    tot_energy += (*theCAFI)[ihit]->getEnergyDeposit();
    int hit_prim = 0;
    for (unsigned int i = 0; i < thePrims.size(); i++) {
      int particleType = thePrims.at(i)->GetPDGcode();
      if (MapOfSecondaries[particleType].find(id) != MapOfSecondaries[particleType].end())
        hit_prim = particleType;
    }
    if (hit_prim == 0) {
      std::cout << "Track ID " << id << " produced a hit which is not associated with a primary." << std::endl;
      miss_energy += (*theCAFI)[ihit]->getEnergyDeposit();
    }
  }
}
