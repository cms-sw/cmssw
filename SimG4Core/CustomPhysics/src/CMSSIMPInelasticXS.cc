
#include "SimG4Core/CustomPhysics/interface/CMSSIMPInelasticXS.h"
#include "SimG4Core/CustomPhysics/interface/CMSSIMP.h"
#include "G4DynamicParticle.hh"
#include "G4Element.hh"
#include "G4ElementTable.hh"
#include "G4PhysicsLogVector.hh"
#include "G4PhysicsVector.hh"
#include "G4NeutronInelasticXS.hh"
#include "G4Proton.hh"
#include "G4Neutron.hh"

CMSSIMPInelasticXS::CMSSIMPInelasticXS() : G4VCrossSectionDataSet("CMSSIMPInelasticXS"), neutron(G4Neutron::Neutron()) {
  verboseLevel = 1;
  nXsection = new G4NeutronInelasticXS();
  isInitialized = false;
}

CMSSIMPInelasticXS::~CMSSIMPInelasticXS() {}

G4bool CMSSIMPInelasticXS::IsElementApplicable(const G4DynamicParticle*, G4int, const G4Material*) { return true; }

G4bool CMSSIMPInelasticXS::IsIsoApplicable(const G4DynamicParticle*, G4int, G4int, const G4Element*, const G4Material*) {
  return true;
}

G4double CMSSIMPInelasticXS::GetElementCrossSection(const G4DynamicParticle* aParticle,
                                                    G4int Z,
                                                    const G4Material* mat) {
  return nXsection->GetElementCrossSection(aParticle, Z, mat);
}

G4double CMSSIMPInelasticXS::GetIsoCrossSection(const G4DynamicParticle* aParticle,
                                                G4int Z,
                                                G4int A,
                                                const G4Isotope* iso,
                                                const G4Element* elm,
                                                const G4Material* mat) {
  return nXsection->GetIsoCrossSection(aParticle, Z, A, iso, elm, mat);
}

void CMSSIMPInelasticXS::BuildPhysicsTable(const G4ParticleDefinition& p) {
  if (isInitialized) {
    return;
  }
  nXsection->BuildPhysicsTable(p);
  if (verboseLevel > 0) {
    G4cout << "CMSSIMPInelasticXS::BuildPhysicsTable for " << p.GetParticleName() << G4endl;
  }
  if (p.GetParticleName() != "chi" && p.GetParticleName() != "anti_chi" && p.GetParticleName() != "chibar") {
    G4ExceptionDescription ed;
    ed << p.GetParticleName() << " is a wrong particle type -"
       << " only simp is allowed";
    G4Exception("CMSSIMPInelasticXS::BuildPhysicsTable(..)", "had012", FatalException, ed, "");
    return;
  }
  isInitialized = true;
}
