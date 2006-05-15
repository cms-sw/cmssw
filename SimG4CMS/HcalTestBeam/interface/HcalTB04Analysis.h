///////////////////////////////////////////////////////////////////////////////
// File: HcalTB04Analysis.h
// Analysis of 2004 Hcal Test beam simulation
///////////////////////////////////////////////////////////////////////////////
#ifndef HcalTB04Analysis_H
#define HcalTB04Analysis_H
 
#include "SimG4Core/Watcher/interface/SimProducer.h"
#include "SimG4Core/Notification/interface/Observer.h"
 
#include "SimG4CMS/Calo/interface/CaloHit.h"
#include "SimG4CMS/Calo/interface/HcalQie.h"

#include "SimDataFormats/HcalTestBeam/interface/PHcalTB04Info.h"

#include "G4Step.hh"
#include "G4Track.hh"
#include "G4ThreeVector.hh"

#include <boost/cstdint.hpp>
#include <iostream>
#include <memory>
#include <vector>
#include <string>

class BeginOfRun;
class BeginOfEvent;
class EndOfEvent;

class PHcalTB04Info;

class HcalTB04Analysis : public SimProducer,
			 public Observer<const BeginOfRun *>,
			 public Observer<const BeginOfEvent *>,
			 public Observer<const EndOfEvent *>,
			 public Observer<const G4Step *> {

public:

  HcalTB04Analysis(const edm::ParameterSet &p);
  virtual ~HcalTB04Analysis();

  virtual void produce(edm::Event&, const edm::EventSetup&);

private:

  HcalTB04Analysis(const HcalTB04Analysis&); // stop default
  const HcalTB04Analysis& operator=(const HcalTB04Analysis&);
 
  void  init();
 
  // observer methods
  void update(const BeginOfRun * run);
  void update(const BeginOfEvent * evt);
  void update(const G4Step * step);
  void update(const EndOfEvent * evt);

  //User methods
  void fillBuffer(const EndOfEvent * evt);
  void qieAnalysis();
  void xtalAnalysis();
  void finalAnalysis(PHcalTB04Info&);

  void   clear();
  int    unitID(uint32_t id);
  double scale(int det, int layer);
  double timeOfFlight(int det, int layer, double eta);

private:

  HcalQie*               myQie;   

  // to read from parameter set
  bool                     hcalOnly;
  int                      mode, type;
  double                   ecalNoise, beamOffset;
  int                      iceta, icphi;
  double                   scaleHB0, scaleHB16, scaleHO, scaleHE0;
  std::vector<std::string> names;
  G4RotationMatrix*        beamline_RM;

  // Constants for the run
  int                       count;
  int                       nTower, nCrystal;
  std::vector<int>          idHcal, idXtal;
  std::vector<uint32_t>     idTower, idEcal;
    
  // Constants for the event
  int                       nPrimary, particleType;
  double                    pInit, etaInit, phiInit;
  std::vector<CaloHit>      ecalHitCache;
  std::vector<CaloHit>      hcalHitCache, hcalHitLayer;
  std::vector<double>       esimh, eqie, esime, enois;

  bool                       pvFound;
  int                        evNum, pvType;
  G4ThreeVector              pvPosition, pvMomentum, pvUVW;
  std::vector<int>           secTrackID, secPartID;
  std::vector<G4ThreeVector> secMomentum;
  std::vector<double>        secEkin;
  std::vector<int>           shortLivedSecondaries;
};

#endif
