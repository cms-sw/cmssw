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

#include "G4ParticleTable.hh"
#include "G4ThreeVector.hh"

#include <string>
#include <memory>

class G4Step;
class TFile;
class TBranchObject;

class CastorShowerLibrary {
public:
  //Constructor and Destructor
  CastorShowerLibrary(const std::string &name, edm::ParameterSet const &p);
  ~CastorShowerLibrary();

public:
  CastorShowerEvent getShowerHits(const G4Step *, bool &);
  int FindEnergyBin(double);
  int FindEtaBin(double);
  int FindPhiBin(double);

protected:
  void initFile(edm::ParameterSet const &);
  void getRecord(int, int);
  void loadEventInfo(TBranchObject *);
  // if eta or phi is not given, take into account only the binning in energy
  void select(int, double, double = 0, double = 9);

private:
  TFile *hf;
  TBranchObject *evtInfo;               // pointer to CastorShowerLibraryInfo-type branch
  TBranchObject *emBranch, *hadBranch;  // pointer to CastorShowerEvent-type branch

  // User defined classes in Root Dictionary
  CastorShowerLibraryInfo *eventInfo;
  CastorShowerEvent *showerEvent;

  bool verbose;
  unsigned int nMomBin, totEvents, evtPerBin;

  std::vector<double> pmom;

  // new variables (bins in eta and phi)
  unsigned int nBinsE, nBinsEta, nBinsPhi;
  unsigned int nEvtPerBinE, nEvtPerBinEta, nEvtPerBinPhi;
  std::vector<double> SLenergies;
  std::vector<double> SLetas;
  std::vector<double> SLphis;
};
#endif
