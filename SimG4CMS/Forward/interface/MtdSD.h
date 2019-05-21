#ifndef SimG4CMSForward_MtdSD_h
#define SimG4CMSForward_MtdSD_h

#include "SimG4CMS/Forward/interface/TimingSD.h"

#include "DetectorDescription/Core/interface/DDsvalues.h"
#include "Geometry/MTDCommonData/interface/MTDNumberingScheme.h"
#include "Geometry/MTDCommonData/interface/MTDBaseNumber.h"

#include <string>
#include <vector>

class G4Step;
class SimTrackManager;

//-------------------------------------------------------------------

class MtdSD : public TimingSD {
public:
  MtdSD(const std::string &,
        const DDCompactView &,
        const SensitiveDetectorCatalog &,
        edm::ParameterSet const &,
        const SimTrackManager *);

  ~MtdSD() override;

  uint32_t setDetUnitId(const G4Step *) override;

private:
  std::vector<double> getDDDArray(const std::string &, const DDsvalues_type &);
  void setNumberingScheme(MTDNumberingScheme *);
  void getBaseNumber(const G4Step *);

  MTDNumberingScheme *numberingScheme;
  MTDBaseNumber theBaseNumber;
  bool isBTL;
  bool isETL;
};

#endif
