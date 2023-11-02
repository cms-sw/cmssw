#ifndef SimG4CMSForwardZdcSD_h
#define SimG4CMSForwardZdcSD_h
///////////////////////////////////////////////////////////////////////////////
// File: ZdcSD.h
// Date: 02.04
// Description: Stores hits of Zdc in appropriate  container
//
///////////////////////////////////////////////////////////////////////////////
#include "SimG4CMS/Calo/interface/CaloSD.h"
#include "SimG4CMS/Forward/interface/ZdcShowerLibrary.h"
#include "SimG4CMS/Forward/interface/ZdcNumberingScheme.h"

class ZdcSD : public CaloSD {
public:
  ZdcSD(const std::string &, const SensitiveDetectorCatalog &, edm::ParameterSet const &, const SimTrackManager *);

  ~ZdcSD() override = default;
  bool ProcessHits(G4Step *step, G4TouchableHistory *tHistory) override;
  uint32_t setDetUnitId(const G4Step *step) override;

  //  bool getFromLibrary(const G4Step * step) override;

  double calculateCherenkovDeposit(const G4Step *);
  double calculateMeanNumberOfPhotons(int, double, double);
  double photonEnergyDist(int, double, double);
  double generatePhotonEnergy(int, double, double);
  double pmtEfficiency(double);
  double convertEnergyToWavelength(double);

  double calculateN2InvIntegral(double);
  double evaluateFunction(const std::vector<double> &, const std::vector<double> &, double);
  double linearInterpolation(double, double, double, double, double);

protected:
  void initRun() override;

private:
  int verbosity;
  bool useShowerLibrary, useShowerHits;
  int setTrackID(const G4Step *step) override;
  double thFibDir;
  double zdcHitEnergyCut;
  std::unique_ptr<ZdcShowerLibrary> showerLibrary;
  std::unique_ptr<ZdcNumberingScheme> numberingScheme;

  std::vector<ZdcShowerLibrary::Hit> hits;
};

#endif  // ZdcSD_h
