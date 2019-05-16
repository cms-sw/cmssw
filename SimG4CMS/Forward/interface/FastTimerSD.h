#ifndef SimG4CMSForward_FastTimerSD_h
#define SimG4CMSForward_FastTimerSD_h

#include "SimG4CMS/Forward/interface/TimingSD.h"

#include "DetectorDescription/Core/interface/DDsvalues.h"
#include "SimG4Core/Notification/interface/Observer.h"
#include "SimG4Core/Notification/interface/BeginOfJob.h"

#include <string>
#include <vector>

class G4Step;
class SimTrackManager;
class FastTimeDDDConstants;

//-------------------------------------------------------------------

class FastTimerSD : public TimingSD, public Observer<const BeginOfJob *> {
public:
  FastTimerSD(const std::string &,
              const DDCompactView &,
              const SensitiveDetectorCatalog &,
              edm::ParameterSet const &,
              const SimTrackManager *);

  ~FastTimerSD() override;

  uint32_t setDetUnitId(const G4Step *) override;

protected:
  using TimingSD::update;
  void update(const BeginOfJob *) override;

private:
  std::vector<double> getDDDArray(const std::string &, const DDsvalues_type &);

private:
  const FastTimeDDDConstants *ftcons;
  int type_;
};

#endif
