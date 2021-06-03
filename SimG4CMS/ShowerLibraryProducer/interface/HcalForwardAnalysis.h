#ifndef ShowerLibraryProducer_HcalForwardAnalysis_h
#define ShowerLibraryProducer_HcalForwardAnalysis_h

#include <iostream>
#include <memory>
#include <vector>
#include <string>

// user include files
#include "SimG4Core/Watcher/interface/SimProducer.h"
#include "SimG4Core/Notification/interface/Observer.h"

#include "SimG4CMS/ShowerLibraryProducer/interface/FiberG4Hit.h"
#include "SimG4CMS/ShowerLibraryProducer/interface/HFShowerG4Hit.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"

#include "G4Step.hh"
#include "G4Track.hh"
#include "G4ThreeVector.hh"

#include "TFile.h"
#include "TTree.h"

#include <vector>
#include <string>

class BeginOfRun;
class BeginOfEvent;
class EndOfEvent;

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
  ~HcalForwardAnalysis() override;

  void produce(edm::Event&, const edm::EventSetup&) override;

private:
  HcalForwardAnalysis(const HcalForwardAnalysis&) = delete;  // stop default
  const HcalForwardAnalysis& operator=(const HcalForwardAnalysis&) = delete;

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

  edm::Service<TFileService> theFile;
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
#endif
