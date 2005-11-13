///////////////////////////////////////////////////////////////////////////////
// File: HFShowerLibrary.h
// Description: Gets information from a shower library
///////////////////////////////////////////////////////////////////////////////
#ifndef HFShowerLibrary_h
#define HFShowerLibrary_h 1

#include "FWCore/ParameterSet/interface/ParameterSet.h"
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
  typedef G4ThreeVector* ptrThreeVector;
  
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

  int                 verbosity;
  double              probMax;
  double              dphi, rMin, rMax;
  std::vector<double> gpar;

  int                 nPhoton;
  int *               ixyz;
  int *               l;
  int *               it;

  int                 nHit;
  ptrThreeVector *    posHit;
  int *               depHit;
  double *            timHit;

  int                 npe;
  double *            xpe;
  double *            ype;
  double *            zpe;
  double *            lpe;
  double *            tpe;

};
#endif
