///////////////////////////////////////////////////////////////////////////////
// File: ZdcTestAnalysis.cc
// Date: 03.06 Edmundo Garcia
// Description: simulation analysis steering code
//
///////////////////////////////////////////////////////////////////////////////
#include "DataFormats/Math/interface/Point3D.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "SimG4CMS/Calo/interface/CaloG4Hit.h"
#include "SimG4CMS/Calo/interface/CaloG4HitCollection.h"
#include "SimG4CMS/Forward/interface/ZdcNumberingScheme.h"

#include "SimG4Core/Notification/interface/BeginOfJob.h"
#include "SimG4Core/Notification/interface/BeginOfRun.h"
#include "SimG4Core/Notification/interface/EndOfRun.h"
#include "SimG4Core/Notification/interface/BeginOfEvent.h"
#include "SimG4Core/Notification/interface/EndOfEvent.h"
#include "SimG4Core/Notification/interface/Observer.h"
#include "SimG4Core/Watcher/interface/SimWatcher.h"

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
#include "TFile.h"

#include <cassert>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <map>
#include <string>
#include <vector>

class ZdcTestAnalysis : public SimWatcher,
                        public Observer<const BeginOfJob*>,
                        public Observer<const BeginOfRun*>,
                        public Observer<const EndOfRun*>,
                        public Observer<const BeginOfEvent*>,
                        public Observer<const EndOfEvent*>,
                        public Observer<const G4Step*> {
public:
  ZdcTestAnalysis(const edm::ParameterSet& p);
  ~ZdcTestAnalysis() override;

private:
  // observer classes
  void update(const BeginOfJob* run) override;
  void update(const BeginOfRun* run) override;
  void update(const EndOfRun* run) override;
  void update(const BeginOfEvent* evt) override;
  void update(const EndOfEvent* evt) override;
  void update(const G4Step* step) override;

  void finish();

  int verbosity;
  int doNTzdcstep;
  int doNTzdcevent;
  std::string stepNtFileName;
  std::string eventNtFileName;

  TFile* zdcOutputEventFile;
  TFile* zdcOutputStepFile;

  TNtuple* zdcstepntuple;
  TNtuple* zdceventntuple;

  int eventIndex;
  int stepIndex;

  Float_t zdcsteparray[18];
  Float_t zdceventarray[16];
};

enum ntzdcs_elements {
  ntzdcs_evt,
  ntzdcs_trackid,
  ntzdcs_charge,
  ntzdcs_pdgcode,
  ntzdcs_x,
  ntzdcs_y,
  ntzdcs_z,
  ntzdcs_stepl,
  ntzdcs_stepe,
  ntzdcs_eta,
  ntzdcs_phi,
  ntzdcs_vpx,
  ntzdcs_vpy,
  ntzdcs_vpz,
  ntzdcs_idx,
  ntzdcs_idl,
  ntzdcs_pvtype,
  ntzdcs_ncherphot
};

enum ntzdce_elements {
  ntzdce_evt,
  ntzdce_ihit,
  ntzdce_fiberid,
  ntzdce_zside,
  ntzdce_subdet,
  ntzdce_layer,
  ntzdce_fiber,
  ntzdce_channel,
  ntzdce_enem,
  ntzdce_enhad,
  ntzdce_hitenergy,
  ntzdce_x,
  ntzdce_y,
  ntzdce_z,
  ntzdce_time,
  ntzdce_etot
};

ZdcTestAnalysis::ZdcTestAnalysis(const edm::ParameterSet& p) {
  //constructor
  edm::ParameterSet m_Anal = p.getParameter<edm::ParameterSet>("ZdcTestAnalysis");
  verbosity = m_Anal.getParameter<int>("Verbosity");
  doNTzdcstep = m_Anal.getParameter<int>("StepNtupleFlag");
  doNTzdcevent = m_Anal.getParameter<int>("EventNtupleFlag");
  stepNtFileName = m_Anal.getParameter<std::string>("StepNtupleFileName");
  eventNtFileName = m_Anal.getParameter<std::string>("EventNtupleFileName");

  if (verbosity > 0)
    edm::LogVerbatim("ZdcAnalysis") << "\n============================================================================";
  edm::LogVerbatim("ZdcAnalysis") << "ZdcTestAnalysis:: Initialized as observer";
  if (doNTzdcstep > 0) {
    edm::LogVerbatim("ZdcAnalysis") << " Step Ntuple will be created";
    edm::LogVerbatim("ZdcAnalysis") << " Step Ntuple file: " << stepNtFileName;
  }
  if (doNTzdcevent > 0) {
    edm::LogVerbatim("ZdcAnalysis") << " Event Ntuple will be created";
    edm::LogVerbatim("ZdcAnalysis") << " Step Ntuple file: " << stepNtFileName;
  }
  edm::LogVerbatim("ZdcAnalysis") << "============================================================================"
                                  << std::endl;

  if (doNTzdcstep > 0)
    zdcstepntuple =
        new TNtuple("NTzdcstep",
                    "NTzdcstep",
                    "evt:trackid:charge:pdgcode:x:y:z:stepl:stepe:eta:phi:vpx:vpy:vpz:idx:idl:pvtype:ncherphot");

  if (doNTzdcevent > 0)
    zdceventntuple =
        new TNtuple("NTzdcevent",
                    "NTzdcevent",
                    "evt:ihit:fiberid:zside:subdet:layer:fiber:channel:enem:enhad:hitenergy:x:y:z:time:etot");

  //theZdcSD = new ZdcSD("ZDCHITSB", new ZdcNumberingScheme());
}

ZdcTestAnalysis::~ZdcTestAnalysis() {
  // destructor
  finish();
}

void ZdcTestAnalysis::update(const BeginOfJob* job) {
  //job
  edm::LogVerbatim("ZdcAnalysis") << "beggining of job";
}

//==================================================================== per RUN
void ZdcTestAnalysis::update(const BeginOfRun* run) {
  //run

  edm::LogVerbatim("ZdcAnalysis") << "\nZdcTestAnalysis: Begining of Run";
  if (doNTzdcstep) {
    edm::LogVerbatim("ZdcAnalysis") << "ZDCTestAnalysis: output step file created";
    TString stepfilename = stepNtFileName;
    zdcOutputStepFile = new TFile(stepfilename, "RECREATE");
  }

  if (doNTzdcevent) {
    edm::LogVerbatim("ZdcAnalysis") << "ZDCTestAnalysis: output event file created";
    TString stepfilename = eventNtFileName;
    zdcOutputEventFile = new TFile(stepfilename, "RECREATE");
  }

  eventIndex = 0;
}

void ZdcTestAnalysis::update(const BeginOfEvent* evt) {
  //event
  edm::LogVerbatim("ZdcAnalysis") << "ZdcTest: Processing Event Number: " << eventIndex;
  eventIndex++;
  stepIndex = 0;
}

void ZdcTestAnalysis::update(const G4Step* aStep) {
  //step;
  stepIndex++;

  if (doNTzdcstep) {
    G4StepPoint* preStepPoint = aStep->GetPreStepPoint();
    // G4StepPoint * postStepPoint= aStep->GetPostStepPoint();
    G4double stepL = aStep->GetStepLength();
    G4double stepE = aStep->GetTotalEnergyDeposit();

    if (verbosity >= 2)
      edm::LogVerbatim("ZdcAnalysis") << "Step " << stepL << "," << stepE;

    G4Track* theTrack = aStep->GetTrack();
    G4int theTrackID = theTrack->GetTrackID();
    G4double theCharge = theTrack->GetDynamicParticle()->GetCharge();
    G4String particleType = theTrack->GetDefinition()->GetParticleName();
    G4int pdgcode = theTrack->GetDefinition()->GetPDGEncoding();

    const G4ThreeVector& vert_mom = theTrack->GetVertexMomentumDirection();
    G4double vpx = vert_mom.x();
    G4double vpy = vert_mom.y();
    G4double vpz = vert_mom.z();
    double eta = 0.5 * log((1. + vpz) / (1. - vpz));
    double phi = atan2(vpy, vpx);

    const G4ThreeVector& hitPoint = preStepPoint->GetPosition();
    G4ThreeVector localPoint = theTrack->GetTouchable()->GetHistory()->GetTopTransform().TransformPoint(hitPoint);

    const G4VTouchable* touch = aStep->GetPreStepPoint()->GetTouchable();
    int idx = touch->GetReplicaNumber(0);
    int idLayer = -1;
    int thePVtype = -1;

    int historyDepth = touch->GetHistoryDepth();

    if (historyDepth > 0) {
      std::vector<int> theReplicaNumbers(historyDepth);
      std::vector<G4VPhysicalVolume*> thePhysicalVolumes(historyDepth);
      std::vector<G4String> thePVnames(historyDepth);
      std::vector<G4LogicalVolume*> theLogicalVolumes(historyDepth);
      std::vector<G4String> theLVnames(historyDepth);
      std::vector<G4Material*> theMaterials(historyDepth);
      std::vector<G4String> theMaterialNames(historyDepth);

      for (int jj = 0; jj < historyDepth; jj++) {
        theReplicaNumbers[jj] = touch->GetReplicaNumber(jj);
        thePhysicalVolumes[jj] = touch->GetVolume(jj);
        thePVnames[jj] = thePhysicalVolumes[jj]->GetName();
        theLogicalVolumes[jj] = thePhysicalVolumes[jj]->GetLogicalVolume();
        theLVnames[jj] = theLogicalVolumes[jj]->GetName();
        theMaterials[jj] = theLogicalVolumes[jj]->GetMaterial();
        theMaterialNames[jj] = theMaterials[jj]->GetName();
        if (verbosity >= 2)
          edm::LogVerbatim("ZdcAnalysis") << " GHD " << jj << ": " << theReplicaNumbers[jj] << "," << thePVnames[jj]
                                          << "," << theLVnames[jj] << "," << theMaterialNames[jj];
      }

      idLayer = theReplicaNumbers[1];
      if (thePVnames[0] == "ZDC_EMLayer")
        thePVtype = 1;
      else if (thePVnames[0] == "ZDC_EMAbsorber")
        thePVtype = 2;
      else if (thePVnames[0] == "ZDC_EMFiber")
        thePVtype = 3;
      else if (thePVnames[0] == "ZDC_HadLayer")
        thePVtype = 7;
      else if (thePVnames[0] == "ZDC_HadAbsorber")
        thePVtype = 8;
      else if (thePVnames[0] == "ZDC_HadFiber")
        thePVtype = 9;
      else if (thePVnames[0] == "ZDC_PhobosLayer")
        thePVtype = 11;
      else if (thePVnames[0] == "ZDC_PhobosAbsorber")
        thePVtype = 12;
      else if (thePVnames[0] == "ZDC_PhobosFiber")
        thePVtype = 13;
      else {
        thePVtype = 0;
        if (verbosity >= 2)
          edm::LogVerbatim("ZdcAnalysis") << " pvtype=0 hd=" << historyDepth << " " << theReplicaNumbers[0] << ","
                                          << thePVnames[0] << "," << theLVnames[0] << "," << theMaterialNames[0];
      }
    } else if (historyDepth == 0) {
      int theReplicaNumber = touch->GetReplicaNumber(0);
      G4VPhysicalVolume* thePhysicalVolume = touch->GetVolume(0);
      const G4String& thePVname = thePhysicalVolume->GetName();
      G4LogicalVolume* theLogicalVolume = thePhysicalVolume->GetLogicalVolume();
      const G4String& theLVname = theLogicalVolume->GetName();
      G4Material* theMaterial = theLogicalVolume->GetMaterial();
      const G4String& theMaterialName = theMaterial->GetName();
      if (verbosity >= 2)
        edm::LogVerbatim("ZdcAnalysis") << " hd=0 " << theReplicaNumber << "," << thePVname << "," << theLVname << ","
                                        << theMaterialName;
    } else {
      edm::LogVerbatim("ZdcAnalysis") << " hd<0:  hd=" << historyDepth;
    }

    double NCherPhot = -1;
    zdcsteparray[ntzdcs_evt] = (float)eventIndex;
    zdcsteparray[ntzdcs_trackid] = (float)theTrackID;
    zdcsteparray[ntzdcs_charge] = theCharge;
    zdcsteparray[ntzdcs_pdgcode] = (float)pdgcode;
    zdcsteparray[ntzdcs_x] = hitPoint.x();
    zdcsteparray[ntzdcs_y] = hitPoint.y();
    zdcsteparray[ntzdcs_z] = hitPoint.z();
    zdcsteparray[ntzdcs_stepl] = stepL;
    zdcsteparray[ntzdcs_stepe] = stepE;
    zdcsteparray[ntzdcs_eta] = eta;
    zdcsteparray[ntzdcs_phi] = phi;
    zdcsteparray[ntzdcs_vpx] = vpx;
    zdcsteparray[ntzdcs_vpy] = vpy;
    zdcsteparray[ntzdcs_vpz] = vpz;
    zdcsteparray[ntzdcs_idx] = (float)idx;
    zdcsteparray[ntzdcs_idl] = (float)idLayer;
    zdcsteparray[ntzdcs_pvtype] = thePVtype;
    zdcsteparray[ntzdcs_ncherphot] = NCherPhot;
    zdcstepntuple->Fill(zdcsteparray);
  }
}

void ZdcTestAnalysis::update(const EndOfEvent* evt) {
  //end of event

  // Look for the Hit Collection
  edm::LogVerbatim("ZdcAnalysis") << "\nZdcTest::upDate(const EndOfEvent * evt) - event #" << (*evt)()->GetEventID()
                                  << "\n  # of aSteps followed in event = " << stepIndex;

  // access to the G4 hit collections
  G4HCofThisEvent* allHC = (*evt)()->GetHCofThisEvent();
  edm::LogVerbatim("ZdcAnalysis") << "  accessed all HC";

  int theZDCHCid = G4SDManager::GetSDMpointer()->GetCollectionID("ZDCHITS");
  edm::LogVerbatim("ZdcAnalysis") << " - theZDCHCid = " << theZDCHCid;

  CaloG4HitCollection* theZDCHC = (CaloG4HitCollection*)allHC->GetHC(theZDCHCid);
  edm::LogVerbatim("ZdcAnalysis") << " - theZDCHC = " << theZDCHC;

  //float ETot = 0.;
  int maxTime = 0;
  int fiberID = 0;
  unsigned int unsignedfiberID = 0;
  std::map<int, float, std::less<int> > energyInFibers;
  std::map<int, float, std::less<int> > primaries;
  float totalEnergy = 0;
  int nentries = theZDCHC->entries();
  edm::LogVerbatim("ZdcAnalysis") << "  theZDCHC has " << nentries << " entries";

  if (doNTzdcevent) {
    if (nentries > 0) {
      for (int ihit = 0; ihit < nentries; ihit++) {
        CaloG4Hit* caloHit = (*theZDCHC)[ihit];
        totalEnergy += caloHit->getEnergyDeposit();
      }

      for (int ihit = 0; ihit < nentries; ihit++) {
        CaloG4Hit* aHit = (*theZDCHC)[ihit];
        fiberID = aHit->getUnitID();
        unsignedfiberID = aHit->getUnitID();
        double enEm = aHit->getEM();
        double enHad = aHit->getHadr();
        math::XYZPoint hitPoint = aHit->getPosition();
        double hitEnergy = aHit->getEnergyDeposit();
        if (verbosity >= 1)
          edm::LogVerbatim("ZdcAnalysis")
              << " entry #" << ihit << ": fiberID=0x" << std::hex << fiberID << std::dec << "; enEm=" << enEm
              << "; enHad=" << enHad << "; hitEnergy=" << hitEnergy << "z=" << hitPoint.z();
        energyInFibers[fiberID] += enEm + enHad;
        primaries[aHit->getTrackID()] += enEm + enHad;
        float time = aHit->getTimeSliceID();
        if (time > maxTime)
          maxTime = (int)time;

        int thesubdet, thelayer, thefiber, thechannel, thez;
        ZdcNumberingScheme::unpackZdcIndex(fiberID, thesubdet, thelayer, thefiber, thechannel, thez);
        int unsignedsubdet, unsignedlayer, unsignedfiber, unsignedchannel, unsignedz;
        ZdcNumberingScheme::unpackZdcIndex(
            unsignedfiberID, unsignedsubdet, unsignedlayer, unsignedfiber, unsignedchannel, unsignedz);

        // unsigned int packidx1 = packZdcIndex(thesubdet, thelayer, thefiber, thechannel, thez);
        // unsigned int packidx1 = packZdcIndex(thesubdet, thelayer, thefiber, thechannel, thez);
        // unsigned int packidx1 = packZdcIndex(thesubdet, thelayer, thefiber, thechannel, thez);
        // unsigned int packidx1 = packZdcIndex(thesubdet, thelayer, thefiber, thechannel, thez);

        zdceventarray[ntzdce_evt] = (float)eventIndex;
        zdceventarray[ntzdce_ihit] = (float)ihit;
        zdceventarray[ntzdce_fiberid] = (float)fiberID;
        zdceventarray[ntzdce_zside] = (float)thez;
        zdceventarray[ntzdce_subdet] = (float)thesubdet;
        zdceventarray[ntzdce_layer] = (float)thelayer;
        zdceventarray[ntzdce_fiber] = (float)thefiber;
        zdceventarray[ntzdce_channel] = (float)thechannel;
        zdceventarray[ntzdce_enem] = enEm;
        zdceventarray[ntzdce_enhad] = enHad;
        zdceventarray[ntzdce_hitenergy] = hitEnergy;
        zdceventarray[ntzdce_x] = hitPoint.x();
        zdceventarray[ntzdce_y] = hitPoint.y();
        zdceventarray[ntzdce_z] = hitPoint.z();
        zdceventarray[ntzdce_time] = time;
        zdceventarray[ntzdce_etot] = totalEnergy;
        zdceventntuple->Fill(zdceventarray);
      }

      /*
      for (std::map<int, float, std::less<int> >::iterator is = energyInFibers.begin(); is != energyInFibers.end();
           is++) {
        ETot = (*is).second;
      }
      */

      // Find Primary info:
      int trackID = 0;
      G4PrimaryParticle* thePrim = nullptr;
      G4int nvertex = (*evt)()->GetNumberOfPrimaryVertex();
      edm::LogVerbatim("ZdcAnalysis") << "Event has " << nvertex << " vertex";
      if (nvertex == 0)
        edm::LogVerbatim("ZdcAnalysis") << "ZdcTest End Of Event  ERROR: no vertex";

      for (int i = 0; i < nvertex; i++) {
        G4PrimaryVertex* avertex = (*evt)()->GetPrimaryVertex(i);
        if (avertex == nullptr) {
          edm::LogVerbatim("ZdcAnalysis") << "ZdcTest End Of Event ERR: pointer to vertex = 0";
        } else {
          edm::LogVerbatim("ZdcAnalysis") << "Vertex number :" << i;
          int npart = avertex->GetNumberOfParticle();
          if (npart == 0)
            edm::LogVerbatim("ZdcAnalysis") << "ZdcTest End Of Event ERR: no primary!";
          if (thePrim == nullptr)
            thePrim = avertex->GetPrimary(trackID);
        }
      }

      double px = 0., py = 0., pz = 0.;
      double pInit = 0.;

      if (thePrim != nullptr) {
        px = thePrim->GetPx();
        py = thePrim->GetPy();
        pz = thePrim->GetPz();
        pInit = sqrt(pow(px, 2.) + pow(py, 2.) + pow(pz, 2.));
        if (pInit == 0) {
          edm::LogVerbatim("ZdcAnalysis") << "ZdcTest End Of Event  ERR: primary has p=0 ";
        }
      } else {
        edm::LogVerbatim("ZdcAnalysis") << "ZdcTest End Of Event ERR: could not find primary ";
      }

    }  // nentries > 0

  }  // createNTzdcevent

  int iEvt = (*evt)()->GetEventID();
  if (iEvt < 10)
    edm::LogVerbatim("ZdcAnalysis") << " ZdcTest Event " << iEvt;
  else if ((iEvt < 100) && (iEvt % 10 == 0))
    edm::LogVerbatim("ZdcAnalysis") << " ZdcTest Event " << iEvt;
  else if ((iEvt < 1000) && (iEvt % 100 == 0))
    edm::LogVerbatim("ZdcAnalysis") << " ZdcTest Event " << iEvt;
  else if ((iEvt < 10000) && (iEvt % 1000 == 0))
    edm::LogVerbatim("ZdcAnalysis") << " ZdcTest Event " << iEvt;
}

void ZdcTestAnalysis::update(const EndOfRun* run) {}

void ZdcTestAnalysis::finish() {
  if (doNTzdcstep) {
    zdcOutputStepFile->cd();
    zdcstepntuple->Write();
    edm::LogVerbatim("ZdcAnalysis") << "ZdcTestAnalysis: Ntuple step  written for event: " << eventIndex;
    zdcOutputStepFile->Close();
    edm::LogVerbatim("ZdcAnalysis") << "ZdcTestAnalysis: Step file closed";
  }

  if (doNTzdcevent) {
    zdcOutputEventFile->cd();
    zdceventntuple->Write("", TObject::kOverwrite);
    edm::LogVerbatim("ZdcAnalysis") << "ZdcTestAnalysis: Ntuple event written for event: " << eventIndex;
    zdcOutputEventFile->Close();
    edm::LogVerbatim("ZdcAnalysis") << "ZdcTestAnalysis: Event file closed";
  }
}

#include "SimG4Core/Watcher/interface/SimWatcherFactory.h"
#include "FWCore/PluginManager/interface/ModuleDef.h"

DEFINE_SIMWATCHER(ZdcTestAnalysis);
