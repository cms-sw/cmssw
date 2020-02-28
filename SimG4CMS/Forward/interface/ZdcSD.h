///////////////////////////////////////////////////////////////////////////////
// File: ZdcSD.h
// Date: 02.04
// Description: Stores hits of Zdc in appropriate  container
//
///////////////////////////////////////////////////////////////////////////////
#ifndef ZdcSD_h
#define ZdcSD_h
#include "SimG4CMS/Calo/interface/CaloSD.h"
#include "SimG4CMS/Forward/interface/ZdcShowerLibrary.h"
#include "SimG4CMS/Forward/interface/ZdcNumberingScheme.h"

class ZdcSD : public CaloSD {
public:
  ZdcSD(const std::string &,
        const edm::EventSetup &,
        const SensitiveDetectorCatalog &,
        edm::ParameterSet const &,
        const SimTrackManager *);

  ~ZdcSD() override = default;

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

#endif  // ZdcSD_h
