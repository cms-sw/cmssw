#ifndef SimG4CMS_CFCShowerLibrary_h
#define SimG4CMS_CFCShowerLibrary_h 1
///////////////////////////////////////////////////////////////////////////////
// File: CFCShowerLibrary.h
// Description: Gets information from a shower library
///////////////////////////////////////////////////////////////////////////////

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "SimG4CMS/Calo/interface/HFFibre.h"

#include "G4ParticleTable.hh"
#include "G4ThreeVector.hh"
 
//ROOT
#include "TROOT.h"
#include "TFile.h"
#include "TTree.h"

#include <vector>
#include <memory>

class G4Step;

class CFCShowerLibrary {
  
public:
  
  //Constructor and Destructor
  CFCShowerLibrary(edm::ParameterSet const & p, std::vector<double>& gpar);
  ~CFCShowerLibrary();

public:

  struct Hit {
    Hit() {}
    G4ThreeVector     position;  // local coordinate
    int               type;
    double            lambda;
    double            time;
  };

  void                initRun(G4ParticleTable * theParticleTable);
  std::vector<Hit>    getHits(G4Step * aStep, bool &ok);

private:

  std::vector<double> gpar;
  TFile*              hfile;

  int                 emPDG, epPDG, gammaPDG, mumPDG, mupPDG;
  int                 pi0PDG, etaPDG, nuePDG, numuPDG, nutauPDG;
  int                 anuePDG, anumuPDG, anutauPDG, geantinoPDG;

};
#endif
