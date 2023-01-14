// -*- C++ -*-
//
// Package:     Forward
// Class  :     CastorTestAnalysis
//
// Implementation:
//     <Notes on implementation>
//
// Original Author: P. Katsas
//         Created: 02/2007
//

#include "SimG4Core/Notification/interface/Observer.h"
#include "SimG4Core/Notification/interface/BeginOfJob.h"
#include "SimG4Core/Notification/interface/BeginOfRun.h"
#include "SimG4Core/Notification/interface/EndOfRun.h"
#include "SimG4Core/Notification/interface/BeginOfEvent.h"
#include "SimG4Core/Notification/interface/EndOfEvent.h"
#include "SimG4Core/Watcher/interface/SimWatcher.h"

#include "SimG4CMS/Calo/interface/CaloG4Hit.h"
#include "SimG4CMS/Calo/interface/CaloG4HitCollection.h"
#include "SimG4CMS/Forward/interface/CastorNumberingScheme.h"

#include "DataFormats/Math/interface/Point3D.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "G4SDManager.hh"
#include "G4Step.hh"
#include "G4Track.hh"
#include "G4Event.hh"
#include "G4PrimaryVertex.hh"
#include "G4VProcess.hh"
#include "G4HCofThisEvent.hh"
#include "G4UserEventAction.hh"

#include <CLHEP/Units/GlobalSystemOfUnits.h>
#include <CLHEP/Units/GlobalPhysicalConstants.h>
#include <CLHEP/Random/Randomize.h>

#include "TROOT.h"
#include "TFile.h"
#include "TH1.h"
#include "TH2.h"
#include "TProfile.h"
#include "TNtuple.h"
#include "TRandom.h"
#include "TLorentzVector.h"
#include "TUnixSystem.h"
#include "TSystem.h"
#include "TMath.h"
#include "TF1.h"

#include <cassert>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <map>
#include <string>
#include <vector>

//#define EDM_ML_DEBUG

class CastorTestAnalysis : public SimWatcher,
                           public Observer<const BeginOfJob *>,
                           public Observer<const BeginOfRun *>,
                           public Observer<const EndOfRun *>,
                           public Observer<const BeginOfEvent *>,
                           public Observer<const EndOfEvent *>,
                           public Observer<const G4Step *> {
public:
  CastorTestAnalysis(const edm::ParameterSet &p);
  ~CastorTestAnalysis() override;

private:
  // observer classes
  void update(const BeginOfJob *run) override;
  void update(const BeginOfRun *run) override;
  void update(const EndOfRun *run) override;
  void update(const BeginOfEvent *evt) override;
  void update(const EndOfEvent *evt) override;
  void update(const G4Step *step) override;

private:
  void getCastorBranchData(const CaloG4HitCollection *hc);
  void Finish();

  int verbosity;
  int doNTcastorstep;
  int doNTcastorevent;
  std::string stepNtFileName;
  std::string eventNtFileName;

  TFile *castorOutputEventFile;
  TFile *castorOutputStepFile;

  TNtuple *castorstepntuple;
  TNtuple *castoreventntuple;

  CastorNumberingScheme *theCastorNumScheme;

  int eventIndex;
  int stepIndex;
  int eventGlobalHit;

  Float_t castorsteparray[14];
  Float_t castoreventarray[11];
};

enum ntcastors_elements {
  ntcastors_evt,
  ntcastors_trackid,
  ntcastors_charge,
  ntcastors_pdgcode,
  ntcastors_x,
  ntcastors_y,
  ntcastors_z,
  ntcastors_stepl,
  ntcastors_stepe,
  ntcastors_eta,
  ntcastors_phi,
  ntcastors_vpx,
  ntcastors_vpy,
  ntcastors_vpz
};

enum ntcastore_elements {
  ntcastore_evt,
  ntcastore_ihit,
  ntcastore_detector,
  ntcastore_sector,
  ntcastore_module,
  ntcastore_enem,
  ntcastore_enhad,
  ntcastore_hitenergy,
  ntcastore_x,
  ntcastore_y,
  ntcastore_z
};

CastorTestAnalysis::CastorTestAnalysis(const edm::ParameterSet &p) {
  edm::ParameterSet m_Anal = p.getParameter<edm::ParameterSet>("CastorTestAnalysis");
  verbosity = m_Anal.getParameter<int>("Verbosity");
  doNTcastorstep = m_Anal.getParameter<int>("StepNtupleFlag");
  doNTcastorevent = m_Anal.getParameter<int>("EventNtupleFlag");
  stepNtFileName = m_Anal.getParameter<std::string>("StepNtupleFileName");
  eventNtFileName = m_Anal.getParameter<std::string>("EventNtupleFileName");

  if (verbosity > 0) {
    edm::LogVerbatim("ForwardSim") << std::endl;
    edm::LogVerbatim("ForwardSim") << "============================================================================";
    edm::LogVerbatim("ForwardSim") << "CastorTestAnalysis:: Initialized as observer";
    if (doNTcastorstep > 0) {
      edm::LogVerbatim("ForwardSim") << " Step Ntuple will be created";
      edm::LogVerbatim("ForwardSim") << " Step Ntuple file: " << stepNtFileName;
    }
    if (doNTcastorevent > 0) {
      edm::LogVerbatim("ForwardSim") << " Event Ntuple will be created";
      edm::LogVerbatim("ForwardSim") << " Step Ntuple file: " << stepNtFileName;
    }
    edm::LogVerbatim("ForwardSim") << "============================================================================";
    edm::LogVerbatim("ForwardSim") << std::endl;
  }
  if (doNTcastorstep > 0)
    castorstepntuple =
        new TNtuple("NTcastorstep", "NTcastorstep", "evt:trackid:charge:pdgcode:x:y:z:stepl:stepe:eta:phi:vpx:vpy:vpz");

  if (doNTcastorevent > 0)
    castoreventntuple = new TNtuple(
        "NTcastorevent", "NTcastorevent", "evt:ihit:detector:sector:module:enem:totalenergy:hitenergy:x:y:z");
}

CastorTestAnalysis::~CastorTestAnalysis() {
  //destructor of CastorTestAnalysis

  Finish();
  if (verbosity > 0) {
    edm::LogVerbatim("ForwardSim") << std::endl << "End of CastorTestAnalysis";
  }

  edm::LogVerbatim("ForwardSim") << "CastorTestAnalysis: End of process";
}

//=================================================================== per EVENT
void CastorTestAnalysis::update(const BeginOfJob *job) { edm::LogVerbatim("ForwardSim") << " Starting new job "; }

//==================================================================== per RUN
void CastorTestAnalysis::update(const BeginOfRun *run) {
  edm::LogVerbatim("ForwardSim") << std::endl << "CastorTestAnalysis: Starting Run";
  if (doNTcastorstep) {
    edm::LogVerbatim("ForwardSim") << "CastorTestAnalysis: output step root file created";
    TString stepfilename = stepNtFileName;
    castorOutputStepFile = new TFile(stepfilename, "RECREATE");
  }

  if (doNTcastorevent) {
    edm::LogVerbatim("ForwardSim") << "CastorTestAnalysis: output event root file created";
    TString stepfilename = eventNtFileName;
    castorOutputEventFile = new TFile(stepfilename, "RECREATE");
  }

  eventIndex = 0;
}

void CastorTestAnalysis::update(const BeginOfEvent *evt) {
  edm::LogVerbatim("ForwardSim") << "CastorTestAnalysis: Processing Event Number: " << eventIndex;
  eventIndex++;
  stepIndex = 0;
}

void CastorTestAnalysis::update(const G4Step *aStep) {
  stepIndex++;

  if (doNTcastorstep) {
    G4StepPoint *preStepPoint = aStep->GetPreStepPoint();
    //  G4StepPoint * postStepPoint= aStep->GetPostStepPoint();
    G4double stepL = aStep->GetStepLength();
    G4double stepE = aStep->GetTotalEnergyDeposit();

    if (verbosity >= 2)
      edm::LogVerbatim("ForwardSim") << "Step " << stepL << ", " << stepE;

    G4Track *theTrack = aStep->GetTrack();
    G4int theTrackID = theTrack->GetTrackID();
    G4double theCharge = theTrack->GetDynamicParticle()->GetCharge();
    //  G4String particleType = theTrack->GetDefinition()->GetParticleName();
    G4int pdgcode = theTrack->GetDefinition()->GetPDGEncoding();

    const G4ThreeVector &vert_mom = theTrack->GetVertexMomentumDirection();
    G4double vpx = vert_mom.x();
    G4double vpy = vert_mom.y();
    G4double vpz = vert_mom.z();
    double eta = 0.5 * log((1. + vpz) / (1. - vpz));
    double phi = atan2(vpy, vpx);

    const G4ThreeVector &hitPoint = preStepPoint->GetPosition();

    // Fill-in ntuple
    //  castorsteparray[ntcastors_evt] = (*evt)()->GetEventID();
    castorsteparray[ntcastors_evt] = (float)eventIndex;
    castorsteparray[ntcastors_trackid] = (float)theTrackID;
    castorsteparray[ntcastors_charge] = theCharge;
    castorsteparray[ntcastors_pdgcode] = pdgcode;
    castorsteparray[ntcastors_x] = hitPoint.x();
    castorsteparray[ntcastors_y] = hitPoint.y();
    castorsteparray[ntcastors_z] = hitPoint.z();
    castorsteparray[ntcastors_stepl] = stepL;
    castorsteparray[ntcastors_stepe] = stepE;
    castorsteparray[ntcastors_eta] = eta;
    castorsteparray[ntcastors_phi] = phi;
    castorsteparray[ntcastors_vpx] = vpx;
    castorsteparray[ntcastors_vpy] = vpy;
    castorsteparray[ntcastors_vpz] = vpz;

    /*
  edm::LogVerbatim("ForwardSim") << "TrackID: " << theTrackID;
  edm::LogVerbatim("ForwardSim") << "   StepN: "<< theTrack->GetCurrentStepNumber();
  edm::LogVerbatim("ForwardSim") << "      ParentID: " << aStep->GetTrack()->GetParentID();
  edm::LogVerbatim("ForwardSim") << "      PDG: " << pdgcode;
  edm::LogVerbatim("ForwardSim") << "      X,Y,Z (mm): " << theTrack->GetPosition().x() << "," << theTrack->GetPosition().y() << "," << theTrack->GetPosition().z();
  edm::LogVerbatim("ForwardSim") << "      KE (MeV): " << theTrack->GetKineticEnergy();
  edm::LogVerbatim("ForwardSim") << "      Total EDep (MeV): " << aStep->GetTotalEnergyDeposit();
  edm::LogVerbatim("ForwardSim") << "      StepLength (mm): " << aStep->GetStepLength();
  edm::LogVerbatim("ForwardSim") << "      TrackLength (mm): " << theTrack->GetTrackLength();

  if ( theTrack->GetNextVolume() != 0 )
      edm::LogVerbatim("ForwardSim") <<"      NextVolume: " << theTrack->GetNextVolume()->GetName();
  else 
      edm::LogVerbatim("ForwardSim") <<"      NextVolume: OutOfWorld";
  
  if(aStep->GetPostStepPoint()->GetProcessDefinedStep() != NULL)
      edm::LogVerbatim("ForwardSim") << "      ProcessName: "<< aStep->GetPostStepPoint()->GetProcessDefinedStep()->GetProcessName();
  else
      edm::LogVerbatim("ForwardSim") <<"      ProcessName: UserLimit";
  

   edm::LogVerbatim("ForwardSim") << std::endl;
  */

#ifdef EDM_ML_DEBUG
    if (theTrack->GetNextVolume() != 0)
      edm::LogVerbatim("ForwardSim") << " NextVolume: " << theTrack->GetNextVolume()->GetName();
    else
      edm::LogVerbatim("ForwardSim") << " NextVolume: OutOfWorld";
#endif

    //fill ntuple with step level information
    castorstepntuple->Fill(castorsteparray);
  }
}

//================= End of EVENT ===============
void CastorTestAnalysis::update(const EndOfEvent *evt) {
  // Look for the Hit Collection
  edm::LogVerbatim("ForwardSim") << std::endl
                                 << "CastorTest::update(EndOfEvent * evt) - event #" << (*evt)()->GetEventID();

  // access to the G4 hit collections
  G4HCofThisEvent *allHC = (*evt)()->GetHCofThisEvent();
  edm::LogVerbatim("ForwardSim") << "update(*evt) --> accessed all HC";

  int CAFIid = G4SDManager::GetSDMpointer()->GetCollectionID("CastorFI");

  CaloG4HitCollection *theCAFI = (CaloG4HitCollection *)allHC->GetHC(CAFIid);

  theCastorNumScheme = new CastorNumberingScheme();
  // CastorNumberingScheme *theCastorNumScheme = new CastorNumberingScheme();

  /*
  unsigned int volumeID=0;
  int det, zside, sector, zmodule;
  std::map<int,float,std::less<int> > themap;
  double totalEnergy = 0;
  double hitEnergy = 0;
  double en_in_fi = 0.;
  double en_in_pl = 0.;
*/
  //  double en_in_bu = 0.;
  //  double en_in_tu = 0.;

  if (doNTcastorevent) {
    eventGlobalHit = 0;
    // int eventGlobalHit = 0 ;

    //  Check FI TBranch for Hits
    if (theCAFI->entries() > 0)
      getCastorBranchData(theCAFI);

    // Find Primary info:
    int trackID = 0;
#ifdef EDM_ML_DEBUG
    int particleType = 0;
#endif
    G4PrimaryParticle *thePrim = nullptr;
    G4int nvertex = (*evt)()->GetNumberOfPrimaryVertex();
    edm::LogVerbatim("ForwardSim") << "Event has " << nvertex << " vertex";
    if (nvertex == 0)
      edm::LogVerbatim("ForwardSim") << "CASTORTest End Of Event  ERROR: no vertex";

    for (int i = 0; i < nvertex; i++) {
      G4PrimaryVertex *avertex = (*evt)()->GetPrimaryVertex(i);
      if (avertex == nullptr) {
        edm::LogVerbatim("ForwardSim") << "CASTORTest End Of Event ERR: pointer to vertex = 0";
        continue;
      }
      edm::LogVerbatim("ForwardSim") << "Vertex number :" << i;
      int npart = avertex->GetNumberOfParticle();
      if (npart == 0)
        edm::LogVerbatim("ForwardSim") << "CASTORTest End Of Event ERR: no primary!";
      if (thePrim == nullptr)
        thePrim = avertex->GetPrimary(trackID);
    }

    double px = 0., py = 0., pz = 0., pInit = 0;
#ifdef EDM_ML_DEBUG
    double eta = 0., phi = 0.;
#endif
    if (thePrim != nullptr) {
      px = thePrim->GetPx();
      py = thePrim->GetPy();
      pz = thePrim->GetPz();
      pInit = sqrt(pow(px, 2.) + pow(py, 2.) + pow(pz, 2.));
      if (pInit == 0) {
        edm::LogVerbatim("ForwardSim") << "CASTORTest End Of Event  ERR: primary has p=0 ";
#ifdef EDM_ML_DEBUG
      } else {
        float costheta = pz / pInit;
        float theta = acos(std::min(std::max(costheta, float(-1.)), float(1.)));
        eta = -log(tan(theta / 2));

        if (px != 0)
          phi = atan(py / px);
#endif
      }
#ifdef EDM_ML_DEBUG
      particleType = thePrim->GetPDGcode();
#endif
    } else {
      edm::LogVerbatim("ForwardSim") << "CASTORTest End Of Event ERR: could not find primary ";
    }
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("ForwardSim") << "CastorTestAnalysis: Particle Type " << particleType << " p/eta/phi " << pInit
                                   << ", " << eta << ", " << phi;
#endif
  }

  int iEvt = (*evt)()->GetEventID();
  if (iEvt < 10)
    edm::LogVerbatim("ForwardSim") << " CastorTest Event " << iEvt;
  else if ((iEvt < 100) && (iEvt % 10 == 0))
    edm::LogVerbatim("ForwardSim") << " CastorTest Event " << iEvt;
  else if ((iEvt < 1000) && (iEvt % 100 == 0))
    edm::LogVerbatim("ForwardSim") << " CastorTest Event " << iEvt;
  else if ((iEvt < 10000) && (iEvt % 1000 == 0))
    edm::LogVerbatim("ForwardSim") << " CastorTest Event " << iEvt;

  edm::LogVerbatim("ForwardSim") << std::endl << "===>>> Done writing user histograms ";
}

void CastorTestAnalysis::update(const EndOfRun *run) { ; }

//===================================================================
void CastorTestAnalysis::getCastorBranchData(const CaloG4HitCollection *hc) {
  int nentries = hc->entries();

  if (nentries > 0) {
    unsigned int volumeID = 0;
    int det = 0, zside, sector, zmodule;
    std::map<int, float, std::less<int> > themap;
    double totalEnergy = 0;
    double hitEnergy = 0;
    double en_in_sd = 0.;

    for (int ihit = 0; ihit < nentries; ihit++) {
      CaloG4Hit *aHit = (*hc)[ihit];
      totalEnergy += aHit->getEnergyDeposit();
    }

    for (int ihit = 0; ihit < nentries; ihit++) {
      CaloG4Hit *aHit = (*hc)[ihit];
      volumeID = aHit->getUnitID();
      hitEnergy = aHit->getEnergyDeposit();
      en_in_sd += aHit->getEnergyDeposit();
      //	double enEm = aHit->getEM();
      //	double enHad = aHit->getHadr();

      themap[volumeID] += aHit->getEnergyDeposit();
      // int det, zside, sector, zmodule;
      theCastorNumScheme->unpackIndex(volumeID, zside, sector, zmodule);

      // det = 2 ;  //  det=2/3 for CAFI/CAPL

      castoreventarray[ntcastore_evt] = (float)eventIndex;
      //	castoreventarray[ntcastore_ihit]      = (float)ihit;
      castoreventarray[ntcastore_ihit] = (float)eventGlobalHit;
      castoreventarray[ntcastore_detector] = (float)det;
      castoreventarray[ntcastore_sector] = (float)sector;
      castoreventarray[ntcastore_module] = (float)zmodule;
      castoreventarray[ntcastore_enem] = en_in_sd;
      castoreventarray[ntcastore_enhad] = totalEnergy;
      castoreventarray[ntcastore_hitenergy] = hitEnergy;
      castoreventarray[ntcastore_x] = aHit->getPosition().x();
      castoreventarray[ntcastore_y] = aHit->getPosition().y();
      castoreventarray[ntcastore_z] = aHit->getPosition().z();
      //	castoreventarray[ntcastore_x]         = aHit->getEntry().x();
      //	castoreventarray[ntcastore_y]         = aHit->getEntry().y();
      //	castoreventarray[ntcastore_z]         = aHit->getEntry().z();

      castoreventntuple->Fill(castoreventarray);

      eventGlobalHit++;
    }
  }  // nentries > 0
}

//===================================================================

void CastorTestAnalysis::Finish() {
  if (doNTcastorstep) {
    castorOutputStepFile->cd();
    castorstepntuple->Write();
    edm::LogVerbatim("ForwardSim") << "CastorTestAnalysis: Ntuple step  written";
    castorOutputStepFile->Close();
    edm::LogVerbatim("ForwardSim") << "CastorTestAnalysis: Step file closed";
  }

  if (doNTcastorevent) {
    castorOutputEventFile->cd();
    castoreventntuple->Write("", TObject::kOverwrite);
    edm::LogVerbatim("ForwardSim") << "CastorTestAnalysis: Ntuple event written";
    castorOutputEventFile->Close();
    edm::LogVerbatim("ForwardSim") << "CastorTestAnalysis: Event file closed";
  }
}

#include "SimG4Core/Watcher/interface/SimWatcherFactory.h"
#include "FWCore/PluginManager/interface/ModuleDef.h"

DEFINE_SIMWATCHER(CastorTestAnalysis);
