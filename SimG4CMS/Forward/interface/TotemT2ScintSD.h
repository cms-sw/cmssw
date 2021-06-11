#ifndef SimG4CMSForwardTotemT2ScintSD_h
#define SimG4CMSForwardTotemT2ScintSD_h

#include "SimG4CMS/Calo/interface/CaloSD.h"
#include "SimG4CMS/Forward/interface/TotemT2ScintNumberingScheme.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/ParameterSet/interface/ParameterSetfwd.h"

class TotemT2ScintSD : public CaloSD {
public:
  TotemT2ScintSD(const std::string &,
                 const edm::EventSetup &,
                 const SensitiveDetectorCatalog &,
                 edm::ParameterSet const &,
                 const SimTrackManager *);
  ~TotemT2ScintSD() override = default;
  uint32_t setDetUnitId(const G4Step *step) override;
  void setNumberingScheme(TotemT2ScintNumberingScheme *scheme);

protected:
  double getEnergyDeposit(const G4Step *) override;

private:
  uint32_t setDetUnitId(const int &zside, const int &lay, const int &phi);

  bool useBirk_;
  double birk1_, birk2_, birk3_;

  std::unique_ptr<TotemT2ScintNumberingScheme> numberingScheme;
};

#endif  // TotemT2ScintSD_h
