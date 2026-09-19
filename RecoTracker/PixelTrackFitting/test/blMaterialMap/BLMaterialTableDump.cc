// Geant4 material table dump for the BL-fit material map: one text line per material the rays step through,
//   index name density[g/cm3] X0[cm] rhoE[mol/cm3] I[eV] hwp[eV] Cbar x0 x1 a m delta0
// with rhoE = electron density / N_A = rho * <Z/A> (the Landau xi per cm of path), I the mean excitation
// energy (Bragg additivity as Geant4 evaluates it) and the Sternheimer density-effect parameters Geant4
// uses for the material. blMaterialMapBuild identifies each step's material by its (density, X0) pair,
// the only material identity the MaterialBudgetAction step tree carries, and reads the dE/dx weights
// from this table.
//
// The materials are taken from the G4Step objects the framework hands the watcher, not from the global
// G4Material table: the simulation runs from the BigProducts/Simulation bundle, which links Geant4
// statically, so a plugin linked against the dynamic Geant4 sees its own, empty, copy of every Geant4
// static. The accessors used below read the material object's own fields. A line is appended the first
// time a material is met (every job of a run writes the same set; the builder de-duplicates).
//
// Configured as a second watcher next to MaterialBudgetAction in blMaterialMapRays_cfg.py:
//   cms.PSet(type = cms.string('BLMaterialTableDump'), BLMaterialTableDump = cms.PSet(file = cms.string(...)))
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
#include "SimG4Core/Notification/interface/Observer.h"
#include "SimG4Core/Watcher/interface/SimWatcher.h"
#include "SimG4Core/Watcher/interface/SimWatcherFactory.h"

class BLMaterialTableDump : public SimWatcher, public Observer<const G4Step*> {
public:
  explicit BLMaterialTableDump(const edm::ParameterSet& p)
      : file_(p.getParameter<edm::ParameterSet>("BLMaterialTableDump").getParameter<std::string>("file")) {
    FILE* o = std::fopen(file_.c_str(), "w");
    if (o != nullptr) {
      std::fprintf(o, "# index name density[g/cm3] X0[cm] rhoE[mol/cm3] I[eV] hwp[eV] Cbar x0 x1 a m delta0\n");
      std::fclose(o);
    }
  }
  ~BLMaterialTableDump() override = default;

  void update(const G4Step* step) override {
    if (step == nullptr || step->GetPreStepPoint() == nullptr)
      return;
    const G4Material* m = step->GetPreStepPoint()->GetMaterial();
    if (m == nullptr || !seen_.insert(m).second)
      return;
    const G4IonisParamMat* ion = m->GetIonisation();
    if (ion == nullptr)
      return;
    FILE* o = std::fopen(file_.c_str(), "a");
    if (o == nullptr)
      return;
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
