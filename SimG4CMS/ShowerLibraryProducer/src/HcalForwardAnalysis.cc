// system include files
#include <cmath>
#include <iostream>
#include <iomanip>

// user include files
#include "SimG4CMS/ShowerLibraryProducer/interface/HcalForwardAnalysis.h"

#include "SimG4Core/Notification/interface/BeginOfRun.h"
#include "SimG4Core/Notification/interface/BeginOfEvent.h"
#include "SimG4Core/Notification/interface/EndOfEvent.h"

#include "DataFormats/Math/interface/Point3D.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "G4SDManager.hh"
#include "G4VProcess.hh"
#include "G4HCofThisEvent.hh"
#include "CLHEP/Units/GlobalSystemOfUnits.h"
#include "CLHEP/Units/GlobalPhysicalConstants.h"

HcalForwardAnalysis::HcalForwardAnalysis(const edm::ParameterSet& p) {
  edm::ParameterSet m_SLP = p.getParameter<edm::ParameterSet>("HFShowerLibraryProducer");
  theNames = m_SLP.getParameter<std::vector<std::string> >("Names");
  //LibVer = m_HS.getParameter<std::string> ("LibVer");
  //produces<HFShowerPhotonCollection> ();
  init();
  theEventCounter = 0;
  nphot = 0;
  for (int i = 0; i < 10000; ++i) {
    x[i] = 0.;
    y[i] = 0.;
    z[i] = 0.;
    t[i] = 0.;
    lambda[i] = 0.;
    fiberId[i] = 0;
  }
  primX = primY = primZ = primT = 0.;
  primMomX = primMomY = primMomZ = 0.;
}

HcalForwardAnalysis::~HcalForwardAnalysis() {}

//
// member functions
//

void HcalForwardAnalysis::produce(edm::Event& iEvent, const edm::EventSetup&) {
  //std::auto_ptr<HFShowerPhotonCollection> product(new HFShowerPhotonCollection);
  //edm::LogInfo("HcalForwardLib") << "HcalForwardAnalysis: =====> Filling  event";
  //fillEvent(*product);
  //iEvent.put(product);
  //std::auto_ptr<PHcalForwardLibInfo> product(new PHcalForwardLibInfo);
  //fillEvent(*product);
  fillEvent();
  //iEvent.put(product);
}

void HcalForwardAnalysis::init() {
  theTree = theFile->make<TTree>("CherenkovPhotons", "Cherenkov Photons");
  theTree->Branch("nphot", &nphot, "nphot/I");
  theTree->Branch("x", &x, "x[nphot]/F");
  theTree->Branch("y", &y, "y[nphot]/F");
  theTree->Branch("z", &z, "z[nphot]/F");
  theTree->Branch("t", &t, "t[nphot]/F");
  theTree->Branch("lambda", &lambda, "lambda[nphot]/F");
  theTree->Branch("fiberId", &fiberId, "fiberId[nphot]/I");
  theTree->Branch("primX", &primX, "primX/F");
  theTree->Branch("primY", &primY, "primY/F");
  theTree->Branch("primZ", &primZ, "primZ/F");
  theTree->Branch("primMomX", &primMomX, "primMomX/F");
  theTree->Branch("primMomY", &primMomY, "primMomY/F");
  theTree->Branch("primMomZ", &primMomZ, "primMomZ/F");
  theTree->Branch("primT", &primT, "primT/F");

  // counter
  count = 0;
  evNum = 0;
  clear();
}

void HcalForwardAnalysis::update(const BeginOfRun* run) {
  int irun = (*run)()->GetRunID();
  edm::LogInfo("HcalForwardLib") << " =====> Begin of Run = " << irun;
}

void HcalForwardAnalysis::update(const BeginOfEvent* evt) {
  evNum = (*evt)()->GetEventID();
  clear();
  edm::LogInfo("HcalForwardLib") << "HcalForwardAnalysis: =====> Begin of event = " << evNum;
}

void HcalForwardAnalysis::update(const G4Step* aStep) {}

void HcalForwardAnalysis::update(const EndOfEvent* evt) {
  count++;

  //fill the buffer
  LogDebug("HcalForwardLib") << "HcalForwardAnalysis::Fill event " << (*evt)()->GetEventID();
  setPhotons(evt);

  int iEvt = (*evt)()->GetEventID();
  if (iEvt < 10)
    edm::LogInfo("HcalForwardLib") << "HcalForwardAnalysis:: Event " << iEvt;
  else if ((iEvt < 100) && (iEvt % 10 == 0))
    edm::LogInfo("HcalForwardLib") << "HcalForwardAnalysis:: Event " << iEvt;
  else if ((iEvt < 1000) && (iEvt % 100 == 0))
    edm::LogInfo("HcalForwardLib") << "HcalForwardAnalysis:: Event " << iEvt;
  else if ((iEvt < 10000) && (iEvt % 1000 == 0))
    edm::LogInfo("HcalForwardLib") << "HcalForwardAnalysis:: Event " << iEvt;
}

void HcalForwardAnalysis::setPhotons(const EndOfEvent* evt) {
  int idHC, j;
  FiberG4HitsCollection* theHC;
  // Look for the Hit Collection of HCal
  G4HCofThisEvent* allHC = (*evt)()->GetHCofThisEvent();
  std::string sdName = theNames[0];  //name for fiber hits
  idHC = G4SDManager::GetSDMpointer()->GetCollectionID(sdName);
  theHC = (FiberG4HitsCollection*)allHC->GetHC(idHC);
  LogDebug("HcalForwardLib") << "HcalForwardAnalysis::setPhotons() Hit Collection for " << sdName << " of ID " << idHC
                             << " is obtained at " << theHC;
  std::vector<HFShowerPhoton> ShortFiberPhotons;
  std::vector<HFShowerPhoton> LongFiberPhotons;
  LongFiberPhotons.clear();
  ShortFiberPhotons.clear();
  int thehc_entries = theHC->entries();
  if (idHC >= 0 && theHC != nullptr) {
    std::cout << "FiberhitSize " << thehc_entries << std::endl;
    for (j = 0; j < thehc_entries; j++) {
      FiberG4Hit* aHit = (*theHC)[j];
      std::vector<HFShowerPhoton> thePhotonsFromHit = aHit->photon();
      std::cout << "Fiberhit " << j << " has " << thePhotonsFromHit.size() << " photons." << std::endl;
      int fTowerId = -1;
      int fCellId = -1;
      int fFiberId = -1;
      parseDetId(aHit->towerId(), fTowerId, fCellId, fFiberId);
      for (unsigned int iph = 0; iph < thePhotonsFromHit.size(); ++iph) {
        if (aHit->depth() == 1)
          LongFiberPhotons.push_back(thePhotonsFromHit[iph]);
        if (aHit->depth() == 2)
          ShortFiberPhotons.push_back(thePhotonsFromHit[iph]);
      }
      LogDebug("HcalForwardLib") << "HcalForwardAnalysis::setPhotons() NbPhotons " << thePhotonsFromHit.size()
                                 << " towerId " << fTowerId << " cellId " << fCellId << " fiberId " << fFiberId
                                 << " depth " << aHit->depth();
    }
  } else {
    LogDebug("HcalForwardLib") << "HcalForwardAnalysis::setPhotons(): No Photons!";
    return;
  }
  LogDebug("HcalForwardLib") << "HcalForwardAnalysis::setPhotons() LongFibPhotons: " << LongFiberPhotons.size()
                             << " ShortFibPhotons: " << ShortFiberPhotons.size();
  std::cout << "HcalForwardAnalysis::setPhotons() LongFibPhotons: " << LongFiberPhotons.size()
            << " ShortFibPhotons: " << ShortFiberPhotons.size() << std::endl;

  //Chamber hits to find information about primary particle on surface
  HFShowerG4HitsCollection* theChamberHC;
  G4HCofThisEvent* allChamberHC = (*evt)()->GetHCofThisEvent();
  sdName = theNames[1];
  idHC = G4SDManager::GetSDMpointer()->GetCollectionID(sdName);
  theChamberHC = (HFShowerG4HitsCollection*)allChamberHC->GetHC(idHC);
  math::XYZPoint primPosOnSurf(0, 0, 0);
  math::XYZPoint primMomDirOnSurf(0, 0, 0);
  float primTimeOnSurf = 0;
  //	the chamber hit is for primary particle, but step size can be small
  //	(in newer Geant4 versions) and as a result primary particle may have
  //	multiple hits. We want to take last one which is close the HF absorber
  if (idHC >= 0 && theChamberHC != nullptr) {
    LogDebug("HcalForwardLib") << "HcalForwardAnalysis::setPhotons() Chamber Hits size: " << theChamberHC->entries();
    int thec_hc_entries = theChamberHC->entries();
    for (j = 0; j < thec_hc_entries; ++j) {
      HFShowerG4Hit* aHit = (*theChamberHC)[j];
      LogDebug("HcalForwardLib") << "HcalForwardAnalysis::setPhotons() Chamber Hit id " << aHit->hitId() << " track id "
                                 << aHit->trackId() << " prim. pos. " << aHit->globalPosition() << " prom mom. dir. "
                                 << aHit->primaryMomDir() << " time " << aHit->time();
      primPosOnSurf.SetXYZ(aHit->globalPosition().x(), aHit->globalPosition().y(), aHit->globalPosition().z());
      primMomDirOnSurf.SetXYZ(aHit->primaryMomDir().x(), aHit->primaryMomDir().y(), aHit->primaryMomDir().z());
      primTimeOnSurf = aHit->time();
    }
  } else {
    LogDebug("HcalForwardLib") << "HcalForwardAnalysis::setPhotons(): No Chamber hits. Something is wrong!";
    return;
  }
  primX = primPosOnSurf.x();
  primY = primPosOnSurf.y();
  primZ = primPosOnSurf.z();
  primT = primTimeOnSurf;
  primMomX = primMomDirOnSurf.x();
  primMomY = primMomDirOnSurf.y();
  primMomZ = primMomDirOnSurf.z();
  //production point of Cherenkov photons in long fibers w.r.t local coordinates
  float photonProdX = 0.;
  float photonProdY = 0.;
  float photonProdZ = 0.;
  float photonProdTime = 0.;
  //angles for rotation matrices
  double theta = primMomDirOnSurf.theta();
  double phi = primMomDirOnSurf.phi();

  for (unsigned int k = 0; k < LongFiberPhotons.size(); ++k) {
    HFShowerPhoton aPhoton = LongFiberPhotons[k];
    photonProdX = aPhoton.x() * cos(theta) * cos(phi) + aPhoton.y() * cos(theta) * sin(phi) - aPhoton.z() * sin(theta) -
                  primPosOnSurf.x();
    photonProdY = -aPhoton.x() * sin(phi) + aPhoton.y() * cos(phi) - primPosOnSurf.y();
    photonProdZ = aPhoton.x() * sin(theta) * cos(phi) + aPhoton.y() * sin(theta) * sin(phi) + aPhoton.z() * cos(theta) -
                  primPosOnSurf.z();
    photonProdTime = aPhoton.t() - primTimeOnSurf;
    thePhotons.push_back(Photon(1, photonProdX, photonProdY, photonProdZ, photonProdTime, aPhoton.lambda()));
  }
  for (unsigned int k = 0; k < ShortFiberPhotons.size(); ++k) {
    HFShowerPhoton aPhoton = ShortFiberPhotons[k];
    photonProdX = aPhoton.x() * cos(theta) * cos(phi) + aPhoton.y() * cos(theta) * sin(phi) - aPhoton.z() * sin(theta) -
                  primPosOnSurf.x();
    photonProdY = -aPhoton.x() * sin(phi) + aPhoton.y() * cos(phi) - primPosOnSurf.y();
    photonProdZ = aPhoton.x() * sin(theta) * cos(phi) + aPhoton.y() * sin(theta) * sin(phi) + aPhoton.z() * cos(theta) -
                  primPosOnSurf.z();
    photonProdTime = aPhoton.t() - primTimeOnSurf;
    thePhotons.push_back(Photon(2, photonProdX, photonProdY, photonProdZ, photonProdTime, aPhoton.lambda()));
  }
}
void HcalForwardAnalysis::fillEvent() {
  /*
	 edm::LogInfo("HcalForwardLib") << "HcalForwardAnalysis: =====> filledEvent";
	 */
  nphot = int(thePhotons.size());
  for (int i = 0; i < nphot; ++i) {
    x[i] = thePhotons[i].x;
    y[i] = thePhotons[i].y;
    z[i] = thePhotons[i].z;
    t[i] = thePhotons[i].t;
    lambda[i] = thePhotons[i].lambda;
    fiberId[i] = thePhotons[i].fiberId;
  }
  theTree->Fill();
}
void HcalForwardAnalysis::parseDetId(int id, int& tower, int& cell, int& fiber) {
  tower = id / 10000;
  cell = id / 10 - tower * 10;
  fiber = id - tower * 10000 - cell * 10;
}

void HcalForwardAnalysis::clear() {
  nphot = 0;
  for (int i = 0; i < 10000; ++i) {
    x[i] = 0.;
    y[i] = 0.;
    z[i] = 0.;
    t[i] = 0.;
    lambda[i] = 0.;
    fiberId[i] = 0;
  }
  primX = primY = primZ = primT = 0.;
  primMomX = primMomY = primMomZ = 0.;

  thePhotons.clear();
}
