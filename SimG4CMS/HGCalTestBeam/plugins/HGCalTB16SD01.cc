#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "SimG4CMS/Calo/interface/CaloSD.h"
#include "SimG4Core/Notification/interface/TrackInformation.h"

#include "CLHEP/Units/GlobalSystemOfUnits.h"
#include "G4LogicalVolumeStore.hh"
#include "G4Material.hh"
#include "G4Step.hh"
#include "G4Track.hh"

#include <string>

//#define EDM_ML_DEBUG

class HGCalTB16SD01 : public CaloSD {
public:
  HGCalTB16SD01(const std::string&,
                const edm::EventSetup&,
                const SensitiveDetectorCatalog&,
                edm::ParameterSet const&,
                const SimTrackManager*);
  ~HGCalTB16SD01() override = default;
  uint32_t setDetUnitId(const G4Step* step) override;
  static uint32_t packIndex(int det, int lay, int x, int y);
  static void unpackIndex(const uint32_t& idx, int& det, int& lay, int& x, int& y);

protected:
  double getEnergyDeposit(const G4Step*) override;

private:
  void initialize(const G4StepPoint* point);

  std::string matName_;
  bool useBirk_;
  double birk1_, birk2_, birk3_;
  bool initialize_;
  G4Material* matScin_;
};

HGCalTB16SD01::HGCalTB16SD01(const std::string& name,
                             const edm::EventSetup& es,
                             const SensitiveDetectorCatalog& clg,
                             edm::ParameterSet const& p,
                             const SimTrackManager* manager)
    : CaloSD(name, clg, p, manager), initialize_(true) {
  // Values from NIM 80 (1970) 239-244: as implemented in Geant3
  edm::ParameterSet m_HC = p.getParameter<edm::ParameterSet>("HGCalTestBeamSD");
  matName_ = m_HC.getParameter<std::string>("Material");
  useBirk_ = m_HC.getParameter<bool>("UseBirkLaw");
  birk1_ = m_HC.getParameter<double>("BirkC1") * (g / (MeV * cm2));
  birk2_ = m_HC.getParameter<double>("BirkC2");
  birk3_ = m_HC.getParameter<double>("BirkC3");
  matScin_ = nullptr;

  edm::LogVerbatim("HGCSim") << "HGCalTB16SD01:: Use of Birks law is set to " << useBirk_ << " for " << matName_
                             << " with three constants kB = " << birk1_ << ", C1 = " << birk2_ << ", C2 = " << birk3_;
}

double HGCalTB16SD01::getEnergyDeposit(const G4Step* aStep) {
  auto const point = aStep->GetPreStepPoint();
  if (initialize_)
    initialize(point);
  double destep = aStep->GetTotalEnergyDeposit();
  double wt2 = aStep->GetTrack()->GetWeight();
  double weight = (wt2 > 0.0) ? wt2 : 1.0;
  if (useBirk_ && matScin_ == point->GetMaterial()) {
    weight *= getAttenuation(aStep, birk1_, birk2_, birk3_);
  }
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCSim") << "HGCalTB16SD01: Detector " << point->GetTouchable()->GetVolume()->GetName() << " with "
                             << point->GetMaterial()->GetName() << " weight " << weight << ":" << wt2;
#endif
  return weight * destep;
}

uint32_t HGCalTB16SD01::setDetUnitId(const G4Step* aStep) {
  const G4StepPoint* preStepPoint = aStep->GetPreStepPoint();
  const G4VTouchable* touch = preStepPoint->GetTouchable();

  int det(1), x(0), y(0);
  int lay = (touch->GetReplicaNumber(0));

  return packIndex(det, lay, x, y);
}

uint32_t HGCalTB16SD01::packIndex(int det, int lay, int x, int y) {
  int ix = 0, ixx = x;
  if (x < 0) {
    ix = 1;
    ixx = -x;
  }
  int iy = 0, iyy = y;
  if (y < 0) {
    iy = 1;
    iyy = -y;
  }
  uint32_t idx = (det & 15) << 28;  // bits 28-31
  idx += (lay & 127) << 21;         // bits 21-27
  idx += (iy & 1) << 19;            // bit  19
  idx += (iyy & 511) << 10;         // bits 10-18
  idx += (ix & 1) << 9;             // bit   9
  idx += (ixx & 511);               // bits  0-8

#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCSim") << "HGCalTB16SD01: Detector " << det << " Layer " << lay << " x " << x << " " << ix << " "
                             << ixx << " y " << y << " " << iy << " " << iyy << " ID " << std::hex << idx << std::dec;
#endif
  return idx;
}

void HGCalTB16SD01::unpackIndex(const uint32_t& idx, int& det, int& lay, int& x, int& y) {
  det = (idx >> 28) & 15;
  lay = (idx >> 21) & 127;
  y = (idx >> 10) & 511;
  if (((idx >> 19) & 1) == 1)
    y = -y;
  x = (idx)&511;
  if (((idx >> 9) & 1) == 1)
    x = -x;
}

void HGCalTB16SD01::initialize(const G4StepPoint* point) {
  if (matName_ == point->GetMaterial()->GetName()) {
    matScin_ = point->GetMaterial();
    initialize_ = false;
  }
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCSim") << "HGCalTB16SD01: Material pointer for " << matName_
                             << " is initialized to : " << matScin_;
#endif
}

#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "SimG4Core/SensitiveDetector/interface/SensitiveDetectorPluginFactory.h"

typedef HGCalTB16SD01 HGCalTB1601SensitiveDetector;
DEFINE_SENSITIVEDETECTOR(HGCalTB1601SensitiveDetector);
