// system include files
#include <cmath>
#include <iostream>
#include <iomanip>
#include <string>
#include <vector>

// user include files
#include "DataFormats/Math/interface/Point3D.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "SimG4Core/Notification/interface/BeginOfRun.h"
#include "SimG4Core/Notification/interface/BeginOfEvent.h"
#include "SimG4Core/Notification/interface/EndOfEvent.h"
#include "SimG4Core/Watcher/interface/SimProducer.h"
#include "SimG4Core/Notification/interface/Observer.h"

#include "SimG4CMS/ShowerLibraryProducer/interface/FiberG4Hit.h"
#include "SimG4CMS/ShowerLibraryProducer/interface/HFShowerG4Hit.h"

#include "G4HCofThisEvent.hh"
#include "G4SDManager.hh"
#include "G4Step.hh"
#include "G4Track.hh"
#include "G4ThreeVector.hh"
#include "G4VProcess.hh"

#include <CLHEP/Units/SystemOfUnits.h>
#include <CLHEP/Units/GlobalPhysicalConstants.h>

#include "TFile.h"
#include "TTree.h"

//#define EDM_ML_DEBUG

class HcalForwardAnalysis : public SimProducer,
                            public Observer<const BeginOfRun*>,
                            public Observer<const BeginOfEvent*>,
                            public Observer<const EndOfEvent*>,
                            public Observer<const G4Step*> {
public:
  struct Photon {
    Photon(int id, float X, float Y, float Z, float T, float Lambda)
        : fiberId(id), x(X), y(Y), z(Z), t(T), lambda(Lambda) {}
    int fiberId;
    float x;
    float y;
    float z;
    float t;
    float lambda;
  };

  HcalForwardAnalysis(const edm::ParameterSet& p);
  HcalForwardAnalysis(const HcalForwardAnalysis&) = delete;  // stop default
  const HcalForwardAnalysis& operator=(const HcalForwardAnalysis&) = delete;
  ~HcalForwardAnalysis() override;

  void produce(edm::Event&, const edm::EventSetup&) override;

private:
  void init();

  // observer methods
  void update(const BeginOfRun* run) override;
  void update(const BeginOfEvent* evt) override;
  void update(const G4Step* step) override;
  void update(const EndOfEvent* evt) override;
  //  void write(const EndOfRun * run);

  //User methods
  void setPhotons(const EndOfEvent* evt);
  //void fillEvent(PHcalForwardLibInfo&);
  void fillEvent();
  void parseDetId(int id, int& tower, int& cell, int& fiber);
  void clear();

  TTree* theTree;
  int theEventCounter;
  int count;
  int evNum;
  float x[10000], y[10000], z[10000], t[10000], lambda[10000];
  float primX, primY, primZ, primT;
  float primMomX, primMomY, primMomZ;
  int nphot;
  int fiberId[10000];
  std::vector<Photon> thePhotons;
  std::vector<std::string> theNames;
  bool fillt;
};

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
  if (fillt)
    fillEvent();
}

void HcalForwardAnalysis::init() {
  edm::Service<TFileService> theFile;
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
  edm::LogVerbatim("HcalForwardLib") << " =====> Begin of Run = " << irun;
}

void HcalForwardAnalysis::update(const BeginOfEvent* evt) {
  evNum = (*evt)()->GetEventID();
  clear();
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HcalForwardLib") << "HcalForwardAnalysis: =====> Begin of event = " << evNum;
#endif
}

void HcalForwardAnalysis::update(const G4Step* aStep) {}

void HcalForwardAnalysis::update(const EndOfEvent* evt) {
  count++;

  //fill the buffer
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HcalForwardLib") << "HcalForwardAnalysis::Fill event " << (*evt)()->GetEventID();
#endif
  setPhotons(evt);

  int iEvt = (*evt)()->GetEventID();
  if (iEvt < 10)
    edm::LogVerbatim("HcalForwardLib") << "HcalForwardAnalysis:: Event " << iEvt;
  else if ((iEvt < 100) && (iEvt % 10 == 0))
    edm::LogVerbatim("HcalForwardLib") << "HcalForwardAnalysis:: Event " << iEvt;
  else if ((iEvt < 1000) && (iEvt % 100 == 0))
    edm::LogVerbatim("HcalForwardLib") << "HcalForwardAnalysis:: Event " << iEvt;
  else if ((iEvt < 10000) && (iEvt % 1000 == 0))
    edm::LogVerbatim("HcalForwardLib") << "HcalForwardAnalysis:: Event " << iEvt;
}

void HcalForwardAnalysis::setPhotons(const EndOfEvent* evt) {
  fillt = true;
  int idHC, j;
  FiberG4HitsCollection* theHC;
  // Look for the Hit Collection of HCal
  G4HCofThisEvent* allHC = (*evt)()->GetHCofThisEvent();
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HcalForwardLib") << "HcalForwardAnalysis:: Has " << allHC->GetNumberOfCollections()
                                     << " collections";
  for (int k = 0; k < allHC->GetNumberOfCollections(); ++k) {
    G4String name = (allHC->GetHC(k) == nullptr) ? "Unknown" : allHC->GetHC(k)->GetName();
    G4String nameSD = (allHC->GetHC(k) == nullptr) ? "Unknown" : allHC->GetHC(k)->GetSDname();
    edm::LogVerbatim("HcalForwardLib") << "Collecttion[" << k << "] " << allHC->GetHC(k) << "  " << name << ":"
                                       << nameSD;
  }
#endif
  std::string sdName = theNames[0];  //name for fiber hits
  idHC = G4SDManager::GetSDMpointer()->GetCollectionID(sdName);
  theHC = (FiberG4HitsCollection*)allHC->GetHC(idHC);
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HcalForwardLib") << "HcalForwardAnalysis::setPhotons() Hit Collection for " << sdName << " of ID "
                                     << idHC << " is obtained at " << theHC;
#endif
  std::vector<HFShowerPhoton> ShortFiberPhotons;
  std::vector<HFShowerPhoton> LongFiberPhotons;
  LongFiberPhotons.clear();
  ShortFiberPhotons.clear();
  if (idHC >= 0 && theHC != nullptr) {
    int thehc_entries = theHC->entries();
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HcalForwardLib") << "FiberhitSize " << thehc_entries;
#endif
    for (j = 0; j < thehc_entries; j++) {
      FiberG4Hit* aHit = (*theHC)[j];
      std::vector<HFShowerPhoton> thePhotonsFromHit = aHit->photon();
#ifdef EDM_ML_DEBUG
      edm::LogVerbatim("HcalForwardLib") << "Fiberhit " << j << " has " << thePhotonsFromHit.size() << " photons.";
#endif
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
#ifdef EDM_ML_DEBUG
      edm::LogVerbatim("HcalForwardLib") << "HcalForwardAnalysis::setPhotons() NbPhotons " << thePhotonsFromHit.size()
                                         << " towerId " << fTowerId << " cellId " << fCellId << " fiberId " << fFiberId
                                         << " depth " << aHit->depth();
#endif
    }
  } else {
    fillt = false;
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HcalForwardLib") << "HcalForwardAnalysis::setPhotons(): No Photons!";
#endif
    return;
  }
  edm::LogVerbatim("HcalForwardLib") << "HcalForwardAnalysis::setPhotons() LongFibPhotons: " << LongFiberPhotons.size()
                                     << " ShortFibPhotons: " << ShortFiberPhotons.size();
  edm::LogVerbatim("HcalForwardLib") << "HcalForwardAnalysis::setPhotons() LongFibPhotons: " << LongFiberPhotons.size()
                                     << " ShortFibPhotons: " << ShortFiberPhotons.size();

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
  //  if (idHC >= 0 && theChamberHC != nullptr && theChamberHC->entries()>0) {
  if (idHC >= 0 && theChamberHC != nullptr) {
    edm::LogVerbatim("HcalForwardLib") << "HcalForwardAnalysis::setPhotons() Chamber Hits size: "
                                       << theChamberHC->entries();
    int thec_hc_entries = theChamberHC->entries();
    for (j = 0; j < thec_hc_entries; ++j) {
      HFShowerG4Hit* aHit = (*theChamberHC)[j];
#ifdef EDM_ML_DEBUG
      edm::LogVerbatim("HcalForwardLib") << "HcalForwardAnalysis::setPhotons() Chamber Hit id " << aHit->hitId()
                                         << " track id " << aHit->trackId() << " prim. pos. " << aHit->globalPosition()
                                         << " prom mom. dir. " << aHit->primaryMomDir() << " time " << aHit->time();
#endif
      primPosOnSurf.SetXYZ(aHit->globalPosition().x(), aHit->globalPosition().y(), aHit->globalPosition().z());
      primMomDirOnSurf.SetXYZ(aHit->primaryMomDir().x(), aHit->primaryMomDir().y(), aHit->primaryMomDir().z());
      primTimeOnSurf = aHit->time();
    }
  } else {
    edm::LogVerbatim("HcalForwardLib") << "HcalForwardAnalysis::setPhotons(): No Chamber hits are stored";
    fillt = false;
    return;
  }
  primX = primPosOnSurf.x();
  primY = primPosOnSurf.y();
  primZ = primPosOnSurf.z();
  if (primZ < 990) {  // there were interactions before HF
    edm::LogVerbatim("HcalForwardLib") << "HcalForwardAnalysis::setPhotons(): First interaction before HF";
    fillt = false;
    return;
  }
  primT = primTimeOnSurf;
  primMomX = primMomDirOnSurf.x();
  primMomY = primMomDirOnSurf.y();
  primMomZ = primMomDirOnSurf.z();
  //angles for rotation matrices
  double theta = primMomDirOnSurf.theta();
  double phi = primMomDirOnSurf.phi();

  // my insert ----------------------------------------------------------------
  double sphi = sin(phi);
  double cphi = cos(phi);
  double ctheta = cos(theta);
  double stheta = sin(theta);

  double pex = 0, pey = 0, zv = 0;
  double xx, yy, zz;

  for (unsigned int k = 0; k < LongFiberPhotons.size(); ++k) {
    HFShowerPhoton aPhoton = LongFiberPhotons[k];
    // global coordinates
    xx = aPhoton.x();
    yy = aPhoton.y();
    zz = aPhoton.z();

    // local coordinates in rotated to shower axis system and vs shower origin
    pex = xx * ctheta * cphi + yy * ctheta * sphi - zz * stheta;
    pey = -xx * sphi + yy * cphi;
    zv = xx * stheta * cphi + yy * stheta * sphi + zz * ctheta - primZ / ctheta;

    double photonProdTime = aPhoton.t() - primTimeOnSurf;
    thePhotons.push_back(Photon(1, pex, pey, zv, photonProdTime, aPhoton.lambda()));
  }
  for (unsigned int k = 0; k < ShortFiberPhotons.size(); ++k) {
    HFShowerPhoton aPhoton = ShortFiberPhotons[k];
    // global coordinates
    xx = aPhoton.x();
    yy = aPhoton.y();
    zz = aPhoton.z();

    // local coordinates in rotated to shower axis system and vs shower origin
    pex = xx * ctheta * cphi + yy * ctheta * sphi - zz * stheta;
    pey = -xx * sphi + yy * cphi;
    zv = xx * stheta * cphi + yy * stheta * sphi + zz * ctheta - primZ / ctheta;

    double photonProdTime = aPhoton.t() - primTimeOnSurf;
    thePhotons.push_back(Photon(2, pex, pey, zv, photonProdTime, aPhoton.lambda()));
  }
}

void HcalForwardAnalysis::fillEvent() {
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HcalForwardLib") << "HcalForwardAnalysis: =====> filledEvent";
#endif
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

#include "SimG4Core/Watcher/interface/SimWatcherFactory.h"
#include "FWCore/PluginManager/interface/ModuleDef.h"

DEFINE_SIMWATCHER(HcalForwardAnalysis);
