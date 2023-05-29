/**
 * @file G4muDarkBremsstrahlungModel.h
 * @brief Class provided to simulate the dark brem cross section and interaction.
 * @author Michael Revering, University of Minnesota
 */

#ifndef G4muDarkBremsstrahlungModel_h
#define G4muDarkBremsstrahlungModel_h

// Geant
#include "G4VEmModel.hh"
#include "G4Material.hh"
#include "G4Element.hh"
#include "G4DataVector.hh"
#include "G4ParticleChangeForLoss.hh"

// ROOT
#include "TLorentzVector.h"

struct ParamsForChi {
  double AA;
  double ZZ;
  double MMA;
  double MMu;
  double EE0;
};
struct frame {
  TLorentzVector* fEl;
  TLorentzVector* cm;
  G4double E;
};

class G4Element;
class G4ParticleChangeForLoss;

class G4muDarkBremsstrahlungModel : public G4VEmModel {
public:
  G4muDarkBremsstrahlungModel(const G4String& scalefile,
                              const G4double biasFactor,
                              const G4ParticleDefinition* p = nullptr,
                              const G4String& nam = "eDBrem");

  ~G4muDarkBremsstrahlungModel() override;

  void Initialise(const G4ParticleDefinition*, const G4DataVector&) override;

  G4double ComputeCrossSectionPerAtom(
      const G4ParticleDefinition*, G4double tkin, G4double Z, G4double, G4double cut, G4double maxE = DBL_MAX) override;

  G4DataVector* ComputePartialSumSigma(const G4Material* material, G4double tkin, G4double cut);

  void SampleSecondaries(std::vector<G4DynamicParticle*>*,
                         const G4MaterialCutsCouple*,
                         const G4DynamicParticle*,
                         G4double tmin,
                         G4double maxEnergy) override;

  void LoadMG();

  void MakePlaceholders();

  void SetMethod(std::string);

  frame GetMadgraphData(double E0);
  G4muDarkBremsstrahlungModel& operator=(const G4muDarkBremsstrahlungModel& right) = delete;
  G4muDarkBremsstrahlungModel(const G4muDarkBremsstrahlungModel&) = delete;

protected:
  const G4Element* SelectRandomAtom(const G4MaterialCutsCouple* couple);

private:
  void SetParticle(const G4ParticleDefinition* p);

  static G4double chi(double t, void* pp);

  static G4double DsigmaDx(double x, void* pp);

protected:
  const G4String& mgfile;
  const G4double cxBias;
  const G4ParticleDefinition* particle;
  G4ParticleDefinition* theAPrime;
  G4ParticleChangeForLoss* fParticleChange;
  G4double MA;
  G4double muonMass;
  G4bool isMuon;

private:
  G4double highKinEnergy;
  G4double lowKinEnergy;
  G4double probsup;
  G4bool isInitialised;
  std::string method;
  G4bool mg_loaded;
  std::map<double, std::vector<frame> > mgdata;
  std::vector<std::pair<double, int> > energies;
  std::vector<G4DataVector*> partialSumSigma;
};

#endif
