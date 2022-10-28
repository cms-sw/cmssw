#include "SimG4Core/CustomPhysics/interface/G4muDarkBremsstrahlungModel.h"
#include "SimG4Core/CustomPhysics/interface/G4muDarkBremsstrahlung.h"
#include "SimG4Core/CustomPhysics/interface/G4APrime.h"

//Geant 4
#include "G4ProcessTable.hh"
#include "G4ProcTblElement.hh"
#include "G4MuonMinus.hh"
#include "G4MuonPlus.hh"
#include "G4ProductionCutsTable.hh"
#include "G4SystemOfUnits.hh"
//Root
#include "TFile.h"
#include "TTree.h"
//gsl
#include <gsl/gsl_math.h>
#include <gsl/gsl_integration.h>

using namespace std;

G4muDarkBremsstrahlungModel::G4muDarkBremsstrahlungModel(const G4String& scalefile,
                                                         const G4double biasFactor,
                                                         const G4ParticleDefinition* p,
                                                         const G4String& nam)
    : G4VEmModel(nam),
      mgfile(scalefile),
      cxBias(biasFactor),
      particle(nullptr),
      isMuon(true),
      probsup(1.0),
      isInitialised(false),
      method("forward_only"),
      mg_loaded(false) {
  if (p) {
    SetParticle(p);
  }  //Verify that the particle is a muon.
  theAPrime = G4APrime::APrime();
  MA = G4APrime::APrime()->GetPDGMass() / CLHEP::GeV;                        //Get the A' mass.
  muonMass = G4MuonMinus::MuonMinusDefinition()->GetPDGMass() / CLHEP::GeV;  //Get the muon mass
  highKinEnergy = HighEnergyLimit();
  lowKinEnergy = LowEnergyLimit();
  fParticleChange = nullptr;
}

G4muDarkBremsstrahlungModel::~G4muDarkBremsstrahlungModel() {
  size_t n = partialSumSigma.size();
  if (n > 0) {
    for (size_t i = 0; i < n; i++) {
      delete partialSumSigma[i];
    }
  }
}

void G4muDarkBremsstrahlungModel::SetParticle(const G4ParticleDefinition* p) {
  particle = p;

  if ((p == G4MuonPlus::MuonPlus()) || (p == G4MuonMinus::MuonMinus())) {
    isMuon = true;
  } else {
    isMuon = false;
  }
}

void G4muDarkBremsstrahlungModel::Initialise(const G4ParticleDefinition* p, const G4DataVector& cuts) {
  if (p) {
    SetParticle(p);
  }

  highKinEnergy = HighEnergyLimit();
  lowKinEnergy = LowEnergyLimit();
  const G4ProductionCutsTable* theCoupleTable = G4ProductionCutsTable::GetProductionCutsTable();

  if (theCoupleTable) {
    G4int numOfCouples = theCoupleTable->GetTableSize();
    G4int nn = partialSumSigma.size();
    G4int nc = cuts.size();
    if (nn > 0) {
      for (G4int ii = 0; ii < nn; ii++) {
        G4DataVector* a = partialSumSigma[ii];
        if (a)
          delete a;
      }
      partialSumSigma.clear();
    }
    if (numOfCouples > 0) {
      for (G4int i = 0; i < numOfCouples; i++) {
        G4double cute = DBL_MAX;
        if (i < nc)
          cute = cuts[i];
        const G4MaterialCutsCouple* couple = theCoupleTable->GetMaterialCutsCouple(i);
        const G4Material* material = couple->GetMaterial();
        G4DataVector* dv = ComputePartialSumSigma(material, 0.5 * highKinEnergy, std::min(cute, 0.25 * highKinEnergy));
        partialSumSigma.push_back(dv);
      }
    }
  }

  if (isInitialised)
    return;
  fParticleChange = GetParticleChangeForLoss();
  isInitialised = true;
}

void G4muDarkBremsstrahlungModel::LoadMG()
//Parses a Madgraph root file to extract the kinetic energy fraction and pt of the outgoing electron in each event. Loads the two numbers from every event into a map of vectors of pairs (mgdata). Map is keyed by energy, vector pairs are energy fraction + pt. Also creates an list of energies and placeholders (energies), so that different energies can be looped separately.
{
  TFile* f = TFile::Open(mgfile.c_str());
  TTree* tree = (TTree*)f->Get("Events");
  TLorentzVector* mvec = new TLorentzVector();
  TLorentzVector* avec = new TLorentzVector();
  TLorentzVector* nvec = new TLorentzVector();
  tree->SetBranchAddress("IncidentParticle", &mvec);
  tree->SetBranchAddress("APrime", &avec);
  tree->SetBranchAddress("Nucleus", &nvec);
  int entries = tree->GetEntries();
  int start = int(G4UniformRand() * entries);
  for (int i = start; i < int(start + entries / 10.); i++) {
    if (i < entries) {
      tree->GetEntry(i);
    } else {
      tree->GetEntry(i - entries);
    }
    frame evnt;
    evnt.fEl = (TLorentzVector*)mvec->Clone();
    evnt.cm = (TLorentzVector*)avec->Clone();
    *evnt.cm = *evnt.cm + *evnt.fEl;
    TLorentzVector* ebeam = (TLorentzVector*)nvec->Clone();
    *ebeam = *ebeam + *evnt.cm;
    evnt.E = round(ebeam->Z() * 10.) / 10.;
    if (mgdata.count(evnt.E) == 0) {
      mgdata[evnt.E];
    }
    mgdata[evnt.E].push_back(evnt);
  }
  f->Close();
}

void G4muDarkBremsstrahlungModel::MakePlaceholders() {
  //Need to do this to set up random sampling of mg distributions
  for (const auto& iter : mgdata) {
    energies.push_back(std::make_pair(iter.first, iter.second.size()));
  }

  for (uint64_t i = 0; i < energies.size(); i++) {
    energies[i].second = int(G4UniformRand() * mgdata[energies[i].first].size());
  }
}

frame G4muDarkBremsstrahlungModel::GetMadgraphData(double E0)
//Gets the energy fraction and Pt from the imported LHE data. E0 should be in GeV, returns the total energy and Pt in GeV. Scales from the closest imported beam energy above the given value (scales down to avoid biasing issues).
{
  double samplingE = energies[0].first;
  frame cmdata;
  uint64_t i = 0;
  bool pass = false;
  //G4double Mel = 5.1E-04;
  //Cycle through imported beam energies until the closest one above is found, or the max is reached.
  while (!pass) {
    i++;
    samplingE = energies[i].first;
    if ((E0 <= samplingE) || (i >= energies.size())) {
      pass = true;
    }
  }

  if (i == energies.size()) {
    i--;
  }  //Decrement if the loop hit the maximum, to prevent a segfault.
  //energies is a vector of pairs. energies[i].first holds the imported beam energy,
  //energies[i].second holds the place as we loop through different energy sampling files.
  //Need to loop around if we hit the end, when the size of mgdata is smaller than our index
  //on energies[i].second.
  if (energies[i].second >= double(mgdata[energies[i].first].size())) {
    energies[i].second = 0;
  }

  //Get the lorentz vectors from the index given by the placeholder.
  cmdata = mgdata[energies[i].first].at(energies[i].second);

  //Increment the placeholder.
  energies[i].second = energies[i].second + 1;

  return cmdata;
}

G4double G4muDarkBremsstrahlungModel::DsigmaDx(double x, void* pp) {
  ParamsForChi* params = (ParamsForChi*)pp;

  G4double beta = sqrt(1 - (params->MMA) * (params->MMA) / (params->EE0) / (params->EE0));
  G4double num = 1. - x + x * x / 3.;
  G4double denom = (params->MMA) * (params->MMA) * (1. - x) / x + (params->MMu) * (params->MMu) * x;
  G4double DsDx = beta * num / denom;

  return DsDx;
}

G4double G4muDarkBremsstrahlungModel::chi(double t, void* pp) {
  ParamsForChi* params = (ParamsForChi*)pp;
  /* Reminder II:
 * params->AA;
 * params->ZZ;
 * params->MMA;
 * params->MMu;
 * params->EE0;
 */
  G4double Mel = 5.1E-04;
  G4double MUp = 2.79;
  G4double Mpr = 0.938;
  G4double d = 0.164 / pow((params->AA), 2. / 3.);
  G4double ap = 773.0 / Mel / pow((params->ZZ), 2. / 3.);
  G4double a = 111.0 / Mel / pow((params->ZZ), 1. / 3.);
  G4double G2el = (params->ZZ) * (params->ZZ) * a * a * a * a * t * t / (1.0 + a * a * t) / (1.0 + a * a * t) /
                  (1.0 + t / d) / (1.0 + t / d);
  G4double G2in = (params->ZZ) * ap * ap * ap * ap * t * t / (1.0 + ap * ap * t) / (1.0 + ap * ap * t) /
                  (1.0 + t / 0.71) / (1.0 + t / 0.71) / (1.0 + t / 0.71) / (1.0 + t / 0.71) / (1.0 + t / 0.71) /
                  (1.0 + t / 0.71) / (1.0 + t / 0.71) / (1.0 + t / 0.71) *
                  (1.0 + t * (MUp * MUp - 1.0) / 4.0 / Mpr / Mpr) * (1.0 + t * (MUp * MUp - 1.0) / 4.0 / Mpr / Mpr);
  G4double G2 = G2el + G2in;
  G4double ttmin = (params->MMA) * (params->MMA) * (params->MMA) * (params->MMA) / 4.0 / (params->EE0) / (params->EE0);
  //   G4double ttmin = lowerLimit(x,theta,p);
  G4double Under = G2 * (t - ttmin) / t / t;

  return Under;
}

G4double G4muDarkBremsstrahlungModel::ComputeCrossSectionPerAtom(
    const G4ParticleDefinition*, G4double E0, G4double Z, G4double A, G4double cut, G4double)
// Calculates the cross section per atom in GEANT4 internal units. Uses WW approximation to find the total cross section, performing numerical integrals over x and theta.
{
  G4double cross = 0.0;
  if (E0 < keV || E0 < cut) {
    return cross;
  }

  E0 = E0 / CLHEP::GeV;  //Change energy to GeV.
  if (E0 < 2. * MA)
    return 0.;

  //begin: chi-formfactor calculation
  gsl_integration_workspace* w = gsl_integration_workspace_alloc(1000);

  G4double result, error;
  G4double tmin = MA * MA * MA * MA / (4. * E0 * E0);
  G4double tmax = MA * MA;

  gsl_function F;
  ParamsForChi alpha = {1.0, 1.0, 1.0, 1.0, 1.0};
  F.function = &G4muDarkBremsstrahlungModel::chi;
  F.params = &alpha;
  alpha.AA = A;
  alpha.ZZ = Z;
  alpha.MMA = MA;
  alpha.MMu = muonMass;
  alpha.EE0 = E0;

  //Integrate over chi.
  gsl_integration_qags(&F, tmin, tmax, 0, 1e-7, 1000, w, &result, &error);

  G4double ChiRes = result;
  gsl_integration_workspace_free(w);

  //Integrate over x. Can use log approximation instead, which falls off at high A' mass.
  gsl_integration_workspace* dxspace = gsl_integration_workspace_alloc(1000);
  gsl_function G;
  G.function = &DsigmaDx;
  G.params = &alpha;
  G4double xmin = 0;
  G4double xmax = 1;
  if ((muonMass / E0) > (MA / E0))
    xmax = 1 - muonMass / E0;
  else
    xmax = 1 - MA / E0;
  G4double res, err;

  gsl_integration_qags(&G, xmin, xmax, 0, 1e-7, 1000, dxspace, &res, &err);

  G4double DsDx = res;
  gsl_integration_workspace_free(dxspace);

  G4double GeVtoPb = 3.894E08;
  G4double alphaEW = 1.0 / 137.0;
  G4double epsilBench = 1;

  cross = GeVtoPb * 4. * alphaEW * alphaEW * alphaEW * epsilBench * epsilBench * ChiRes * DsDx * CLHEP::picobarn;
  if (cross < 0.) {
    cross = 0.;
  }
  E0 = E0 * CLHEP::GeV;

  cross = cross * cxBias;
  return cross;
}

G4DataVector* G4muDarkBremsstrahlungModel::ComputePartialSumSigma(const G4Material* material,
                                                                  G4double kineticEnergy,
                                                                  G4double cut)

// Build the table of cross section per element.
//The table is built for MATERIALS.
// This table is used by DoIt to select randomly an element in the material.
{
  G4int nElements = material->GetNumberOfElements();
  const G4ElementVector* theElementVector = material->GetElementVector();
  const G4double* theAtomNumDensityVector = material->GetAtomicNumDensityVector();
  G4DataVector* dv = new G4DataVector();
  G4double cross = 0.0;

  for (G4int i = 0; i < nElements; i++) {
    cross += theAtomNumDensityVector[i] *
             ComputeCrossSectionPerAtom(
                 particle, kineticEnergy, (*theElementVector)[i]->GetZ(), (*theElementVector)[i]->GetN(), cut);
    dv->push_back(cross);
  }
  return dv;
}

void G4muDarkBremsstrahlungModel::SampleSecondaries(std::vector<G4DynamicParticle*>* vdp,
                                                    const G4MaterialCutsCouple* couple,
                                                    const G4DynamicParticle* dp,
                                                    G4double tmin,
                                                    G4double maxEnergy)
//Simulates the emission of a dark photon + electron. Gets an energy fraction and Pt from madgraph files. Scales the energy so that the fraction of kinectic energy is constant, keeps the Pt constant. If the Pt is larger than the new energy, that event is skipped, and a new one is taken from the file.
{
  //Deactivate the process after one dark brem. Needs to be reactivated in the end of event action. If this is in the stepping action instead, more than one brem can occur within each step.
  G4bool state = false;
  G4String pname = "muDBrem";
  G4ProcessTable* ptable = G4ProcessTable::GetProcessTable();
  ptable->SetProcessActivation(pname, state);
  if (mg_loaded == false) {
    LoadMG();
    MakePlaceholders();  //Setup the placeholder offsets for getting data.
    mg_loaded = true;
  }
  G4double E0 = dp->GetTotalEnergy();
  G4double tmax = min(maxEnergy, E0);
  if (tmin >= tmax) {
    return;
  }  // limits of the energy sampling

  E0 = E0 / CLHEP::GeV;  //Convert the energy to GeV, the units used in the sampling files.

  frame data = GetMadgraphData(E0);
  double EAcc, Pt, P, PhiAcc;
  if (method == "forward_only") {
    EAcc = (data.fEl->E() - muonMass) / (data.E - muonMass - MA) * (E0 - muonMass - MA);
    EAcc = muonMass + EAcc;
    Pt = data.fEl->Pt();
    P = sqrt(EAcc * EAcc - muonMass * muonMass);
    PhiAcc = data.fEl->Phi();
    int i = 0;
    while (Pt * Pt + muonMass * muonMass > EAcc * EAcc)  //Skip events until the Pt is less than the energy.
    {
      i++;
      data = GetMadgraphData(E0);
      EAcc = (data.fEl->E() - muonMass) / (data.E - muonMass - MA) * (E0 - muonMass - MA);
      EAcc = muonMass + EAcc;
      Pt = data.fEl->Pt();
      P = sqrt(EAcc * EAcc - muonMass * muonMass);
      PhiAcc = data.fEl->Phi();

      if (i > 10000) {
        break;
      }
    }
  } else if (method == "cm_scaling") {
    TLorentzVector* el = new TLorentzVector(data.fEl->X(), data.fEl->Y(), data.fEl->Z(), data.fEl->E());
    double ediff = data.E - E0;
    TLorentzVector* newcm = new TLorentzVector(data.cm->X(), data.cm->Y(), data.cm->Z() - ediff, data.cm->E() - ediff);
    el->Boost(-1. * data.cm->BoostVector());
    el->Boost(newcm->BoostVector());
    double newE = (data.fEl->E() - muonMass) / (data.E - muonMass - MA) * (E0 - muonMass - MA);
    el->SetE(newE);
    EAcc = el->E();
    Pt = el->Pt();
    P = el->P();
    PhiAcc = el->Phi();
  } else {
    EAcc = E0;
    P = dp->GetTotalMomentum();
    Pt = sqrt(dp->Get4Momentum().px() * dp->Get4Momentum().px() + dp->Get4Momentum().py() * dp->Get4Momentum().py());
    PhiAcc = dp->Get4Momentum().phi();
  }

  EAcc = EAcc * CLHEP::GeV;  //Change the energy back to MeV, the internal GEANT unit.

  G4double muon_mass_mev = muonMass * CLHEP::GeV;
  G4double momentum = sqrt(EAcc * EAcc - muon_mass_mev * muon_mass_mev);  //Muon momentum in MeV.
  G4ThreeVector newDirection;
  double ThetaAcc = std::asin(Pt / P);
  newDirection.set(std::sin(ThetaAcc) * std::cos(PhiAcc), std::sin(ThetaAcc) * std::sin(PhiAcc), std::cos(ThetaAcc));
  newDirection.rotateUz(dp->GetMomentumDirection());
  newDirection.setMag(momentum);
  // create g4dynamicparticle object for the dark photon.
  G4ThreeVector direction = (dp->GetMomentum() - newDirection);
  G4DynamicParticle* dphoton = new G4DynamicParticle(theAPrime, direction);
  vdp->push_back(dphoton);

  // energy of primary
  G4double finalKE = EAcc - muon_mass_mev;

  // stop tracking and create new secondary instead of primary
  bool makeSecondary = true;
  if (makeSecondary) {
    fParticleChange->ProposeTrackStatus(fStopAndKill);
    fParticleChange->SetProposedKineticEnergy(0.0);
    G4DynamicParticle* mu =
        new G4DynamicParticle(const_cast<G4ParticleDefinition*>(particle), newDirection.unit(), finalKE);
    vdp->push_back(mu);
    // continue tracking
  } else {
    fParticleChange->SetProposedMomentumDirection(newDirection.unit());
    fParticleChange->SetProposedKineticEnergy(finalKE);
  }
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....

const G4Element* G4muDarkBremsstrahlungModel::SelectRandomAtom(const G4MaterialCutsCouple* couple) {
  // select randomly 1 element within the material

  const G4Material* material = couple->GetMaterial();
  G4int nElements = material->GetNumberOfElements();
  const G4ElementVector* theElementVector = material->GetElementVector();

  const G4Element* elm = nullptr;

  if (1 < nElements) {
    --nElements;
    G4DataVector* dv = partialSumSigma[couple->GetIndex()];
    G4double rval = G4UniformRand() * ((*dv)[nElements]);

    elm = (*theElementVector)[nElements];
    for (G4int i = 0; i < nElements; ++i) {
      if (rval <= (*dv)[i]) {
        elm = (*theElementVector)[i];
        break;
      }
    }
  } else {
    elm = (*theElementVector)[0];
  }

  SetCurrentElement(elm);
  return elm;
}

void G4muDarkBremsstrahlungModel::SetMethod(std::string method_in) {
  method = method_in;
  return;
}
