#ifndef _EcalSimHitsValidProducer_h
#define _EcalSimHitsValidProducer_h
#include <map>
#include <string>
#include <vector>

#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"
#include "DataFormats/EcalDetId/interface/ESDetId.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "SimG4Core/Notification/interface/Observer.h"
#include "SimG4Core/Watcher/interface/SimProducer.h"

#include "DataFormats/Math/interface/LorentzVector.h"

class BeginOfEvent;
class G4Step;
class EndOfEvent;
class PEcalValidInfo;

namespace edm {
  class ParameterSet;
}

class EcalSimHitsValidProducer : public SimProducer,
                                 public Observer<const BeginOfEvent *>,
                                 public Observer<const G4Step *>,
                                 public Observer<const EndOfEvent *> {
  typedef std::vector<float> FloatVector;
  typedef std::map<uint32_t, float, std::less<uint32_t>> MapType;

public:
  EcalSimHitsValidProducer(const edm::ParameterSet &);
  ~EcalSimHitsValidProducer() override;

  void produce(edm::Event &, const edm::EventSetup &) override;

private:
  EcalSimHitsValidProducer(const EcalSimHitsValidProducer &) = delete;                   // stop default
  const EcalSimHitsValidProducer &operator=(const EcalSimHitsValidProducer &) = delete;  // stop default

  void update(const BeginOfEvent *) override;
  void update(const G4Step *) override;
  void update(const EndOfEvent *) override;

  void fillEventInfo(PEcalValidInfo &);

private:
  uint32_t getUnitWithMaxEnergy(MapType &themap);

  float energyInEEMatrix(int nCellInX, int nCellInY, int centralX, int centralY, int centralZ, MapType &themap);
  float energyInEBMatrix(int nCellInX, int nCellInY, int centralX, int centralY, int centralZ, MapType &themap);

  bool fillEEMatrix(
      int nCellInEta, int nCellInPhi, int CentralEta, int CentralPhi, int CentralZ, MapType &fillmap, MapType &themap);

  bool fillEBMatrix(
      int nCellInEta, int nCellInPhi, int CentralEta, int CentralPhi, int CentralZ, MapType &fillmap, MapType &themap);

  float eCluster2x2(MapType &themap);
  float eCluster4x4(float e33, MapType &themap);

private:
  float ee1;
  float ee4;
  float ee9;
  float ee16;
  float ee25;

  float eb1;
  float eb4;
  float eb9;
  float eb16;
  float eb25;

  float totalEInEE;
  float totalEInEB;
  float totalEInES;

  float totalEInEEzp;
  float totalEInEEzm;
  float totalEInESzp;
  float totalEInESzm;

  int totalHits;
  int nHitsInEE;
  int nHitsInEB;
  int nHitsInES;
  int nHitsIn1ES;
  int nHitsIn2ES;
  int nCrystalInEB;
  int nCrystalInEEzp;
  int nCrystalInEEzm;

  int nHitsIn1ESzp;
  int nHitsIn1ESzm;
  int nHitsIn2ESzp;
  int nHitsIn2ESzm;

  float eBX0[26];
  float eEX0[26];

  FloatVector eOf1ES;
  FloatVector eOf2ES;
  FloatVector eOf1ESzp;
  FloatVector eOf1ESzm;
  FloatVector eOf2ESzp;
  FloatVector eOf2ESzm;

  FloatVector zOfES;
  FloatVector phiOfEECaloG4Hit;
  FloatVector etaOfEECaloG4Hit;
  FloatVector tOfEECaloG4Hit;
  FloatVector eOfEECaloG4Hit;
  FloatVector eOfEEPlusCaloG4Hit;
  FloatVector eOfEEMinusCaloG4Hit;

  FloatVector phiOfEBCaloG4Hit;
  FloatVector etaOfEBCaloG4Hit;
  FloatVector tOfEBCaloG4Hit;
  FloatVector eOfEBCaloG4Hit;

  FloatVector phiOfESCaloG4Hit;
  FloatVector etaOfESCaloG4Hit;
  FloatVector tOfESCaloG4Hit;
  FloatVector eOfESCaloG4Hit;

  math::XYZTLorentzVector theMomentum;
  math::XYZTLorentzVector theVertex;

  int thePID;
  std::string label;
};

#endif
