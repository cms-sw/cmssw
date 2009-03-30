#ifndef SimG4CMS_CastorShowerLibrary_h
#define SimG4CMS_CastorShowerLibrary_h
///////////////////////////////////////////////////////////////////////////////
// File: CastorShowerLibrary.h
// Description: Gets information from a shower library
//              Adapted from HFShowerLibrary class
//
//              Wagner Carvalho (wcarvalh@cern.ch)  
//
///////////////////////////////////////////////////////////////////////////////

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "SimDataFormats/CaloHit/interface/CastorShowerLibraryInfo.h"
#include "SimDataFormats/CaloHit/interface/CastorShowerEvent.h"
#include "DetectorDescription/Core/interface/DDsvalues.h"

#include "G4ParticleTable.hh"
#include "G4ThreeVector.hh"
 
//ROOT
#include "TROOT.h"
#include "TFile.h"
#include "TTree.h"
#include "TBranchObject.h"

#include <string>
#include <memory>

class G4Step;

class CastorShowerLibrary {
  
public:
  
  //Constructor and Destructor
  CastorShowerLibrary(std::string & name, edm::ParameterSet const & p);
  ~CastorShowerLibrary();

public:

  void                initParticleTable(G4ParticleTable *);
  CastorShowerEvent   getShowerHits(G4Step*, bool&);

protected:

  void                initFile(edm::ParameterSet const & );
  void                getRecord(int, int);
  void                loadEventInfo(TBranchObject *);
  void                select(int, double);               // Replaces interpolate / extrapolate
  // void                interpolate(int, double);
  // void                extrapolate(int, double);

private:

  TFile               *hf;                      
  TBranchObject       *evtInfo;                 // pointer to CastorShowerLibraryInfo-type branch 
  TBranchObject       *emBranch, *hadBranch;    // pointer to CastorShowerEvent-type branch

  // User defined classes in Root Dictionary 
  CastorShowerLibraryInfo  *eventInfo;
  CastorShowerEvent        *showerEvent;

  bool                verbose;
  unsigned int        nMomBin, totEvents, evtPerBin;
  std::vector<double> pmom;

  int                 emPDG, epPDG, gammaPDG;
  int                 pi0PDG, etaPDG, nuePDG, numuPDG, nutauPDG;
  int                 anuePDG, anumuPDG, anutauPDG, geantinoPDG;
  int                 mumPDG, mupPDG;

};
#endif
