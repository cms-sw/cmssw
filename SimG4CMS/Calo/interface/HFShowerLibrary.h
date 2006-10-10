#ifndef SimG4CMS_HFShowerLibrary_h
#define SimG4CMS_HFShowerLibrary_h 1
///////////////////////////////////////////////////////////////////////////////
// File: HFShowerLibrary.h
// Description: Gets information from a shower library
///////////////////////////////////////////////////////////////////////////////

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "SimG4CMS/Calo/interface/HFFibre.h"
#include "DetectorDescription/Core/interface/DDsvalues.h"

#include "G4ThreeVector.hh"
 
//ROOT
#include "TROOT.h"
#include "TFile.h"
#include "TTree.h"

#include <string>
#include <memory>

class DDCompactView;    
class G4Step;

class HFShowerLibrary : public TObject {
  
public:
  
  //Constructor and Destructor
  HFShowerLibrary(std::string & name, const DDCompactView & cpv,
		  edm::ParameterSet const & p);
  ~HFShowerLibrary();

public:

  int                 getHits(G4Step * aStep);
  G4ThreeVector       getPosHit(int i);
  int                 getDepth(int i);
  double              getTSlice(int i);

protected:

  bool                rInside(double r);
  int                 getPhoton(TTree *, int);
  void                getRecord(TTree *, int);
  void                loadPacking(TTree *);
  void                loadEventInfo(TTree *, bool);
  void                interpolate(TTree *, double);
  void                extrapolate(TTree *, double);
  void                storePhoton(int j);
  std::vector<double> getDDDArray(const std::string&, const DDsvalues_type&,
				  int&);

  struct Photon {
    Photon() {}
    int               xyz;
    int               lambda;
    int               time;
  };

  struct Hit {
    Hit() {}
    G4ThreeVector     position;
    int               depth;
    double            time;
  };

  struct PhotoElectron {
    PhotoElectron() {}
    double            x;
    double            y;
    double            z;
    double            lambda;
    double            time;
  };

private:

  HFFibre *           fibre;
  TFile *             hf;
  TTree *             emTree;
  TTree *             hadTree;

  int                 xOffset, xMultiplier, xScale;
  int                 yOffset, yMultiplier, yScale;
  int                 zOffset, zMultiplier, zScale;
  int                 nMomBin, totEvents, evtPerBin;
  float               libVers, listVersion; 
  std::vector<double> pmom;

  double              probMax;
  double              dphi, rMin, rMax;
  std::vector<double> gpar;

  int                 nPhoton;
  std::vector<Photon> photon;

  int                 nHit;
  std::vector<Hit>    hit;

  int                 npe;
  std::vector<PhotoElectron> pe;

};
#endif
