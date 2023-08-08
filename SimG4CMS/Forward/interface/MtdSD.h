#ifndef SimG4CMSForward_MtdSD_h
#define SimG4CMSForward_MtdSD_h

#include "SimG4CMS/Forward/interface/TimingSD.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/ParameterSet/interface/ParameterSetfwd.h"

#include "Geometry/MTDCommonData/interface/MTDNumberingScheme.h"
#include "Geometry/MTDCommonData/interface/MTDBaseNumber.h"

#include <string>
#include <vector>

class G4Step;
class SimTrackManager;

//-------------------------------------------------------------------

class MtdSD : public TimingSD {
public:
  MtdSD(const std::string &, const SensitiveDetectorCatalog &, edm::ParameterSet const &, const SimTrackManager *);

  ~MtdSD() override;

  uint32_t setDetUnitId(const G4Step *) override;

protected:
  int getTrackID(const G4Track *) override;

private:
  static constexpr unsigned int k_idsecOffset = 1;
  static constexpr unsigned int k_idloopOffset = 2;
  static constexpr unsigned int k_idFromCaloOffset = 3;

  double energyCut;
  double energyHistoryCut;

  void setNumberingScheme(MTDNumberingScheme *);
  void getBaseNumber(const G4Step *);

  MTDNumberingScheme *numberingScheme;
  MTDBaseNumber theBaseNumber;
  bool isBTL;
  bool isETL;
};

#endif
