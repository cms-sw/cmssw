
#include "G4SIMPInelasticXS.hh"
#include "G4SIMP.hh"
#include "G4DynamicParticle.hh"
#include "G4Element.hh"
#include "G4ElementTable.hh"
#include "G4PhysicsLogVector.hh"
#include "G4PhysicsVector.hh"
#include "G4GlauberGribovCrossSection.hh"
#include "G4HadronNucleonXsc.hh"
#include "G4NistManager.hh"
#include "G4Proton.hh"
#include "Randomize.hh"

#include <iostream>
#include <fstream>
#include <sstream>

using namespace std;

const G4int G4SIMPInelasticXS::amin[] = {0,
					  0, 0, 6, 0,10,12,14,16, 0, 0, //1-10
					  0, 0, 0,28, 0, 0, 0,36, 0,40, //11-20
					  0, 0, 0, 0, 0,54, 0,58,63,64, //21-30
					  0,70, 0, 0, 0, 0, 0, 0, 0,90, //31-40
					  0, 0, 0, 0, 0, 0,107,106, 0,112, //41-50
					  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, //51-60
					  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, //61-70
					  0, 0, 0,180, 0, 0, 0, 0, 0, 0, //71-80
					  0,204, 0, 0, 0, 0, 0, 0, 0, 0, //81-90
					  0,235};
const G4int G4SIMPInelasticXS::amax[] = {0,
					  0, 0, 7, 0,11,13,15,18, 0, 0, //1-10
					  0, 0, 0,30, 0, 0, 0,40, 0,48, //11-20
					  0, 0, 0, 0, 0,58, 0,64,65,70, //21-30
					  0,76, 0, 0, 0, 0, 0, 0, 0,96, //31-40
					  0, 0, 0, 0, 0, 0,109,116, 0,124, //41-50
					  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, //51-60
					  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, //61-70
					  0, 0, 0,186, 0, 0, 0, 0, 0, 0, //71-80
					  0,208, 0, 0, 0, 0, 0, 0, 0, 0, //81-90
					  0,238};

G4SIMPInelasticXS::G4SIMPInelasticXS() 
  : G4VCrossSectionDataSet("G4SIMPInelasticXS"),
    proton(G4Proton::Proton())
{
  verboseLevel = 1;
  if(verboseLevel > 0){
    G4cout  << "G4SIMPInelasticXS::G4SIMPInelasticXS Initialise for Z < " 
	    << MAXZINEL << G4endl;
  }
  data.SetName("SIMPInelastic");
  work.resize(13,0);
  temp.resize(13,0.0);
  coeff.resize(MAXZINEL, 1.0);
  ggXsection = new G4GlauberGribovCrossSection();
  fNucleon = new G4HadronNucleonXsc();
  isInitialized = false;
}

G4SIMPInelasticXS::~G4SIMPInelasticXS()
{
  delete fNucleon;
}

void G4SIMPInelasticXS::CrossSectionDescription(std::ostream& outFile) const
{
  outFile << "G4SIMPInelasticXS calculates the SIMP inelastic scattering\n"
          << "cross section on nuclei using data from the high precision\n"
          << "neutron database.  These data are simplified and smoothed over\n"
          << "the resonance region in order to reduce CPU time.\n"
          << "G4SIMPInelasticXS is valid for energies up to 20 MeV, for\n"
          << "nuclei through U.\n";
}

G4bool 
G4SIMPInelasticXS::IsElementApplicable(const G4DynamicParticle*, 
					  G4int, const G4Material*)
{
  return true;
}

G4bool 
G4SIMPInelasticXS::IsIsoApplicable(const G4DynamicParticle*,
				      G4int /*ZZ*/, G4int /*AA*/,
				      const G4Element*, const G4Material*)
{
  return true;
}

G4double 
G4SIMPInelasticXS::GetElementCrossSection(const G4DynamicParticle* aParticle,
					     G4int Z, const G4Material*)
{
  G4double xs = 0.0;
  G4double ekin = aParticle->GetKineticEnergy();

  if(Z < 1 || Z >= MAXZINEL) { return xs; }
  G4int Amean = G4lrint(G4NistManager::Instance()->GetAtomicMassAmu(Z));

  G4PhysicsVector* pv = data.GetElementData(Z);
  if(verboseLevel > 0){
    G4cout  << "G4SIMPInelasticXS::GetCrossSection e= " << ekin << " Z= " 
            << Z << G4endl;
  }

  // element was not initialised
  if(!pv) {
    Initialise(Z);
    pv = data.GetElementData(Z);
    if(!pv) { return xs; }
  }

  G4double e1 = pv->Energy(0);
  if(ekin <= e1) { return xs; }

  G4double e2 = pv->GetMaxEnergy();

  if(ekin <= e2) { 
    xs = pv->Value(ekin); 
  } else if(1 == Z) { 
    fNucleon->GetHadronNucleonXscPDG(aParticle, proton);
    xs = coeff[1]*fNucleon->GetInelasticHadronNucleonXsc();
  } else {          
    ggXsection->GetIsoCrossSection(aParticle, Z, Amean);
    xs = coeff[Z]*ggXsection->GetInelasticGlauberGribovXsc();
  }

  if(verboseLevel > 0) {
    G4cout  << "ekin= " << ekin << ",  XSinel= " << xs << G4endl;
  }
  return xs;
}

G4double 
G4SIMPInelasticXS::GetIsoCrossSection(const G4DynamicParticle* aParticle, 
					 G4int Z, G4int A,
					 const G4Isotope*, const G4Element*,
					 const G4Material*)
{
  G4double xs = 0.0;
  G4double ekin = aParticle->GetKineticEnergy();
  if(Z > 0 && Z < MAXZINEL) { xs = IsoCrossSection(ekin, Z, A); }
  return xs;
}

G4double 
G4SIMPInelasticXS::IsoCrossSection(G4double ekin, G4int Z, G4int A)
{
  G4double xs = 0.0;

  G4PhysicsVector* pv = data.GetElementData(Z);

  // element was not initialised
  if(!pv) {
    Initialise(Z);
    pv = data.GetElementData(Z);
    if(!pv) { return xs; }
  }
 G4PhysicsVector* pviso = data.GetComponentDataByID(Z, A);
  if(pviso) { pv = pviso; }

  xs = pv->Value(ekin); 

  if(verboseLevel > 0){
    G4cout  << "ekin= " << ekin << ",  xs= " << xs << G4endl;
  }
  return xs;
}

G4Isotope* G4SIMPInelasticXS::SelectIsotope(const G4Element* anElement,
					       G4double kinEnergy)
{
  G4int nIso = anElement->GetNumberOfIsotopes();
  G4IsotopeVector* isoVector = anElement->GetIsotopeVector();
  G4Isotope* iso = (*isoVector)[0];

  // more than 1 isotope
  if(1 < nIso) {
    G4int Z = G4lrint(anElement->GetZ());
    if(Z >= MAXZINEL) { Z = MAXZINEL - 1; }
    G4double* abundVector = anElement->GetRelativeAbundanceVector();
    G4double q = G4UniformRand();
    G4double sum = 0.0;

    // is there isotope wise cross section?
    if(0 == amin[Z]) {
      for (G4int j = 0; j<nIso; ++j) {
	sum += abundVector[j];
	if(q <= sum) {
	  iso = (*isoVector)[j];
	  break;
	}
      }
    } else {
      size_t nmax = data.GetNumberOfComponents(Z);
      if(temp.size() < nmax) { temp.resize(nmax,0.0); }
      for (size_t i=0; i<nmax; ++i) {
	G4int A = (*isoVector)[i]->GetN();
        sum += abundVector[i]*IsoCrossSection(kinEnergy, Z, A);
        temp[i] = sum;
      }
      sum *= q;
      for (size_t j = 0; j<nmax; ++j) {
        if(temp[j] >= sum) {
          iso = (*isoVector)[j];
          break;
	}
      }
    }
  }
  return iso;
}

void 
G4SIMPInelasticXS::BuildPhysicsTable(const G4ParticleDefinition& p)
{
  if(isInitialized) { return; }
  if(verboseLevel > 0){
    G4cout << "G4SIMPInelasticXS::BuildPhysicsTable for " 
	   << p.GetParticleName() << G4endl;
  }
  if(p.GetParticleName() != "chi" && 
     p.GetParticleName() != "anti_chi" &&
     p.GetParticleName() != "chibar") { 
    G4ExceptionDescription ed;
    ed << p.GetParticleName() << " is a wrong particle type -"
       << " only simp is allowed";
    G4Exception("G4SIMPInelasticXS::BuildPhysicsTable(..)","had012",
		FatalException, ed, "");
    return; 
  }
  isInitialized = true;

  // check environment variable 
  // Build the complete string identifying the file with the data set
  char* path = getenv("G4NEUTRONXSDATA");

  G4DynamicParticle* dynParticle = 
    new G4DynamicParticle(G4SIMP::SIMP(),G4ThreeVector(1,0,0),1);

  // Access to elements
  const G4ElementTable* theElmTable = G4Element::GetElementTable();
  size_t numOfElm = G4Element::GetNumberOfElements();
  if(numOfElm > 0) {
    for(size_t i=0; i<numOfElm; ++i) {
      G4int Z = G4lrint(((*theElmTable)[i])->GetZ());
      if(Z < 1)         { Z = 1; }
      else if(Z >= MAXZINEL) { Z = MAXZINEL-1; }
      //G4cout << "Z= " << Z << G4endl;
      // Initialisation 
      if(!data.GetElementData(Z)) { Initialise(Z, dynParticle, path); }
    }
  }
  delete dynParticle;
}

void 
G4SIMPInelasticXS::Initialise(G4int Z, G4DynamicParticle* dp, 
				 const char* p)
{
  if(data.GetElementData(Z)) { return; }
  const char* path = p;
  if(!p) {
  // check environment variable 
  // Build the complete string identifying the file with the data set
    path = getenv("G4NEUTRONXSDATA");
    if (!path) {
      G4Exception("G4SIMPInelasticXS::Initialise(..)","had013",
		  FatalException,
                  "Environment variable G4NEUTRONXSDATA is not defined");
      return;
    }
  }
  G4DynamicParticle* dynParticle = dp;
  if(!dp) {
    dynParticle = 
      new G4DynamicParticle(G4Neutron::Neutron(),G4ThreeVector(1,0,0),1);
  }

  G4int Amean = G4lrint(G4NistManager::Instance()->GetAtomicMassAmu(Z));

  // upload element data 
  std::ostringstream ost;
  ost << path << "/inelast" << Z ;
  G4PhysicsVector* v = RetrieveVector(ost, true);
  data.InitialiseForElement(Z, v);
    
  // upload isotope data
  if(amin[Z] > 0) {
    size_t n = 0;
    size_t i = 0;
    size_t nmax = (size_t)(amax[Z]-amin[Z]+1);
    if(work.size() < nmax) { work.resize(nmax,0); }
    for(G4int A=amin[Z]; A<=amax[Z]; ++A) {
      std::ostringstream ost1;
      ost1 << path << "/cap" << Z << "_" << A;
      G4PhysicsVector* v1 = RetrieveVector(ost1, false);
      if(v1) { ++n; }
      work[i] = v1;
      ++i;
    }
    data.InitialiseForComponent(Z, n);
    for(size_t j=0; j<i; ++j) {
      if(work[j]) { data.AddComponent(Z, amin[Z]+j, work[j]); }
    }
  }

  // smooth transition 
  G4double emax = v->GetMaxEnergy();
  G4double sig1 = (*v)[v->GetVectorLength() - 1];
  dynParticle->SetKineticEnergy(emax);
  G4double sig2 = 0.0;
  if(1 == Z) {
    fNucleon->GetHadronNucleonXscPDG(dynParticle, proton);
    sig2 = fNucleon->GetInelasticHadronNucleonXsc();
  } else {
    ggXsection->GetIsoCrossSection(dynParticle, Z, Amean);
    sig2 = ggXsection->GetInelasticGlauberGribovXsc();
  }
  if(sig2 > 0.) { coeff[Z] = sig1/sig2; } 
  if(!dp) { delete dynParticle; }
}

G4PhysicsVector* 
G4SIMPInelasticXS::RetrieveVector(std::ostringstream& ost, G4bool warn)
{
  G4PhysicsLogVector* v = 0;
  std::ifstream filein(ost.str().c_str());
  if (!(filein)) {
    if(!warn) { return v; }
    G4ExceptionDescription ed;
    ed << "Data file <" << ost.str().c_str()
       << "> is not opened!";
    G4Exception("G4SIMPInelasticXS::RetrieveVector(..)","had014",
                FatalException, ed, "Check G4NEUTRONXSDATA");
  }else{
    if(verboseLevel > 1) {
      G4cout << "File " << ost.str() 
	     << " is opened by G4SIMPInelasticXS" << G4endl;
    }
    // retrieve data from DB
    v = new G4PhysicsLogVector();
    if(!v->Retrieve(filein, true)) {
      G4ExceptionDescription ed;
      ed << "Data file <" << ost.str().c_str()
	 << "> is not retrieved!";
      G4Exception("G4SIMPInelasticXS::RetrieveVector(..)","had015",
		  FatalException, ed, "Check G4NEUTRONXSDATA");
    }
  }
  return v;
}
