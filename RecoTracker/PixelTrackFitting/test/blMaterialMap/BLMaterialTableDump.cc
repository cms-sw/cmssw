// Write one text line per Geant4 material that the rays step through:
//   index name density[g/cm3] X0[cm] rhoE[mol/cm3] I[eV] hwp[eV] Cbar x0 x1 a m delta0
// rhoE = rho <Z/A> is the electron density over N_A, and I the mean excitation energy. The last six numbers are the
// Sternheimer density-effect parameters. blMaterialMapBuild finds each step's material by its (density, X0) pair.
// Read the materials from the G4Step objects, which work when Geant4 is linked statically.
// blMaterialMapRays_cfg.py configures it as a second watcher next to MaterialBudgetAction.
#include <cmath>
#include <cstdio>
#include <set>
#include <string>

#include <CLHEP/Units/PhysicalConstants.h>
#include <CLHEP/Units/SystemOfUnits.h>
#include <G4IonisParamMat.hh>
#include <G4Material.hh>
#include <G4Step.hh>
#include <G4StepPoint.hh>

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "SimG4Core/Notification/interface/Observer.h"
#include "SimG4Core/Watcher/interface/SimWatcher.h"
#include "SimG4Core/Watcher/interface/SimWatcherFactory.h"

class BLMaterialTableDump : public SimWatcher, public Observer<const G4Step*> {
public:
  explicit BLMaterialTableDump(const edm::ParameterSet& p)
      : file_(p.getParameter<edm::ParameterSet>("BLMaterialTableDump").getParameter<std::string>("file")) {
    FILE* o = std::fopen(file_.c_str(), "w");
    if (o == nullptr) {
      throw cms::Exception("BLMaterialTableDump") << "cannot open \"" << file_ << "\" for writing";
    }
    std::fprintf(o, "# index name density[g/cm3] X0[cm] rhoE[mol/cm3] I[eV] hwp[eV] Cbar x0 x1 a m delta0\n");
    std::fclose(o);
  }
  ~BLMaterialTableDump() override = default;

  void update(const G4Step* step) override {
    if (step == nullptr || step->GetPreStepPoint() == nullptr) {
      return;
    }
    const G4Material* m = step->GetPreStepPoint()->GetMaterial();
    if (m == nullptr || !seen_.insert(m).second) {
      return;
    }
    const G4IonisParamMat* ion = m->GetIonisation();
    if (ion == nullptr) {
      return;
    }
    FILE* o = std::fopen(file_.c_str(), "a");
    if (o == nullptr) {
      throw cms::Exception("BLMaterialTableDump") << "cannot open \"" << file_ << "\" for appending";
    }
    const double density = m->GetDensity() / (CLHEP::g / CLHEP::cm3);
    const double x0 = m->GetRadlen() / CLHEP::cm;
    // electrons per cm^3 over Avogadro's number: rho * Z/A in mol/cm^3
    const double rhoE = m->GetElectronDensity() * CLHEP::cm3 / CLHEP::Avogadro;
    std::fprintf(o,
                 "%zu %s %.9g %.9g %.9g %.9g %.9g %.9g %.9g %.9g %.9g %.9g %.9g\n",
                 seen_.size() - 1,
                 m->GetName().c_str(),
                 density,
                 x0,
                 rhoE,
                 ion->GetMeanExcitationEnergy() / CLHEP::eV,
                 ion->GetPlasmaEnergy() / CLHEP::eV,
                 ion->GetCdensity(),
                 ion->GetX0density(),
                 ion->GetX1density(),
                 ion->GetAdensity(),
                 ion->GetMdensity(),
                 ion->GetD0density());
    std::fclose(o);
  }

private:
  std::string file_;
  std::set<const G4Material*> seen_;
};

DEFINE_SIMWATCHER(BLMaterialTableDump);
