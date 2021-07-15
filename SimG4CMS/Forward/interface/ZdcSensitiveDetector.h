///////////////////////////////////////////////////////////////////////////////
// File: ZdcSensitiveDetector.h
// Date: 02.04
// Description: Stores hits of Zdc in appropriate  container
//
///////////////////////////////////////////////////////////////////////////////
#ifndef ZdcSensitiveDetector_h
#define ZdcSensitiveDetector_h
#include "SimG4CMS/Calo/interface/CaloSD.h"
#include "SimG4CMS/Forward/interface/ZdcShowerLibrary.h"
#include "SimG4CMS/Forward/interface/ZdcNumberingScheme.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/ParameterSet/interface/ParameterSetfwd.h"

class ZdcSensitiveDetector : public CaloSD {
public:
  ZdcSensitiveDetector(const std::string &,
		       const SensitiveDetectorCatalog &,
		       edm::ParameterSet const &,
		       const SimTrackManager *);

  ~ZdcSensitiveDetector() override = default;

  uint32_t setDetUnitId(const G4Step *step) override;

  void setNumberingScheme(ZdcNumberingScheme *scheme);

protected:
  double getEnergyDeposit(const G4Step *) override;
  bool getFromLibrary(const G4Step *) override;
  void initRun() override;

private:
  int verbosity;
  bool useShowerLibrary, useShowerHits;
  double thFibDir;
  double zdcHitEnergyCut;

  std::unique_ptr<ZdcShowerLibrary> showerLibrary;
  std::unique_ptr<ZdcNumberingScheme> numberingScheme;
  std::vector<ZdcShowerLibrary::Hit> hits;
};

#endif  // ZdcSensitiveDetector_h
