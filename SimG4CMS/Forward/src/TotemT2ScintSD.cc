#include "SimG4CMS/Forward/interface/TotemT2ScintSD.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "SimG4Core/Notification/interface/TrackInformation.h"

#include "G4SDManager.hh"
#include "G4Step.hh"
#include "G4Track.hh"
#include "G4VProcess.hh"
#include "G4ios.hh"
#include "G4Cerenkov.hh"
#include "G4ParticleTable.hh"
#include "CLHEP/Units/GlobalSystemOfUnits.h"
#include "CLHEP/Units/GlobalPhysicalConstants.h"
#include "Randomize.hh"
#include "G4Poisson.hh"

//#define EDM_ML_DEBUG

TotemT2ScintSD::TotemT2ScintSD(const std::string& name,
                               const edm::EventSetup& es,
                               const SensitiveDetectorCatalog& clg,
                               edm::ParameterSet const& p,
                               const SimTrackManager* manager)
    : CaloSD(name,
             clg,
             p,
             manager,
             (float)(p.getParameter<edm::ParameterSet>("TotemT2ScintSD").getParameter<double>("TimeSliceUnit")),
             p.getParameter<edm::ParameterSet>("TotemT2ScintSD").getParameter<bool>("IgnoreTrackID")) {
  edm::ParameterSet m_T2SD = p.getParameter<edm::ParameterSet>("TotemT2ScintSD");
  useBirk_ = m_T2SD.getParameter<bool>("UseBirkLaw");
  birk1_ = m_T2SD.getParameter<double>("BirkC1") * (g / (MeV * cm2));
  birk2_ = m_T2SD.getParameter<double>("BirkC2");
  birk3_ = m_T2SD.getParameter<double>("BirkC3");
  setNumberingScheme(new TotemT2ScintNumberingScheme());

  edm::LogVerbatim("ForwardSim") << "***************************************************\n"
                                 << "*                                                 *\n"
                                 << "* Constructing a TotemT2ScintSD with name " << name << " *\n"
                                 << "*                                                 *\n"
                                 << "***************************************************";

  edm::LogVerbatim("ForwardSim") << "\nUse of Birks law is set to      " << useBirk_
                                 << "  with three constants kB = " << birk1_ << ", C1 = " << birk2_
                                 << ", C2 = " << birk3_;
}

uint32_t TotemT2ScintSD::setDetUnitId(const G4Step* aStep) {
  auto const prePoint = aStep->GetPreStepPoint();
  auto const touch = prePoint->GetTouchable();

  int iphi = (touch->GetReplicaNumber(0)) % 10;
  int lay = (touch->GetReplicaNumber(0) / 10) % 10 + 1;
  int zside = (((touch->GetReplicaNumber(1)) == 1) ? 1 : -1);

  return setDetUnitId(zside, lay, iphi);
}

void TotemT2ScintSD::setNumberingScheme(TotemT2ScintNumberingScheme* scheme) {
  if (scheme != nullptr) {
    edm::LogVerbatim("ForwardSim") << "TotemT2ScintSD: updates numbering scheme for " << GetName();
    numberingScheme.reset(scheme);
  }
}

double TotemT2ScintSD::getEnergyDeposit(const G4Step* aStep) {
  double destep = aStep->GetTotalEnergyDeposit();
  double weight = ((useBirk_) ? getAttenuation(aStep, birk1_, birk2_, birk3_) : 1.0);
  double edep = weight * destep;
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("ForwardSim") << "TotemT2ScintSD: edep= " << destep << ":" << weight << ":" << edep;
#endif
  return edep;
}

uint32_t TotemT2ScintSD::setDetUnitId(const int& zside, const int& lay, const int& iphi) {
  uint32_t id = ((numberingScheme.get()) ? numberingScheme->packID(zside, lay, iphi) : 0);
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("ForwardSim") << "TotemT2ScintSD: zside " << zside << " layer " << lay << " phi " << iphi << " ID "
                                 << std::hex << id << std::dec;
#endif
  return id;
}
