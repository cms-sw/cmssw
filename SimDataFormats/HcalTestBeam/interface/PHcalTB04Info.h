#ifndef HcalTestBeam_PHcalTB04Info_h
#define HcalTestBeam_PHcalTB04Info_h
// -*- C++ -*-
//
// Package:     HcalTestBeam
// Class  :     PHcalTB04Info
// 
/**\class PHcalTB04Info PHcalTB04Info.h SimDataFormats/HcalTestBeam/interface/PHcalTB04Info.h

 Description: Histogram handling class for analysis

 Usage:
    Simulation information for test beam studies of 2004 Test Beam
    Contains beam information, hits and digitised results

*/
//
// Original Author:  Sunanda Banerjee
//         Created:  Sun May 14 10:14:34 CEST 2006
// $Id: PHcalTB04Info.h,v 1.3 2013/04/22 22:30:15 wmtan Exp $
//

// system include files
#include <string>
#include <vector>
#include <memory>
#include <cmath>

// user include files

// forward declarations
class HcalTB04Analysis;

class PHcalTB04Info {

  typedef std::vector<float>  FloatVector;
  typedef std::vector<int>    IntVector;
  friend class HcalTB04Analysis;

public:
  PHcalTB04Info();
  virtual ~PHcalTB04Info();

  // ---------- const member functions ---------------------
  int         primary()     const {return nPrimary;}
  int         partID()      const {return idBeam;}
  float       initE()       const {return eBeam;}
  float       eta()         const {return etaBeam;}
  float       phi()         const {return phiBeam;}
  int         crystal()     const {return nCrystal;}
  IntVector   idsEcal()     const {return idEcal;}
  int         tower()       const {return nTower;}
  IntVector   idsHcal()     const {return idHcal;}
  float       simEtotal()   const {return simEtot;}
  float       simEcE()      const {return simEEc;}
  float       simHcE()      const {return simEHc;}
  float       digEtotal()   const {return digEtot;}
  float       digEcE()      const {return digEEc;}
  float       digHcE()      const {return digEHc;}
  FloatVector simEEcal()    const {return esime;}
  FloatVector digEEcal()    const {return edige;}
  FloatVector simEHcal()    const {return esimh;}
  FloatVector digEHcal()    const {return edigh;}

  int         nHit()        const {return hit;}
  IntVector   detectorHit() const {return detHit;}
  IntVector   etaIndexHit() const {return etaHit;}
  IntVector   phiIndexHit() const {return phiHit;}
  IntVector   layerHit()    const {return layHit;}
  FloatVector energyHit()   const {return eHit;}
  FloatVector timeHit()     const {return tHit;}

  int         evNum()       const {return v1EvNum;}
  int         vtxType()     const {return v1Type;}
  int         vtxSec()      const {return v1Nsec;}
  IntVector   vtxTrkID()    const {return v1secTrackID;}
  IntVector   vtxPartID()   const {return v1secPartID;}
  float       vtxPrimX()    const {return v1X;}
  float       vtxPrimY()    const {return v1Y;}
  float       vtxPrimZ()    const {return v1Z;}
  float       vtxPrimR()    const {return v1R;}
  float       vtxPrimU()    const {return v1U;}
  float       vtxPrimV()    const {return v1V;}
  float       vtxPrimW()    const {return v1W;}
  float       vtxPrimPx()   const {return v1Px;}
  float       vtxPrimPy()   const {return v1Py;}
  float       vtxPrimPz()   const {return v1Pz;}
  FloatVector vtxSecPx()    const {return v1secPx;}
  FloatVector vtxSecPy()    const {return v1secPy;}
  FloatVector vtxSecPz()    const {return v1secPz;}
  FloatVector vtxSecEk()    const {return v1secEk;}

  // ---------- static member functions --------------------

  // ---------- member functions ---------------------------
  void clear();
  void setIDs(const std::vector<int>&, const std::vector<int>&);
  void setPrimary(int primary, int id, double energy, double eta, double phi);
  void setEdep(double simtot, double sime, double simh, 
	       double digtot, double dige, double digh);
  void setEdepEcal(const std::vector<double>& esim, const std::vector<double>& edig);
  void setEdepHcal(const std::vector<double>& esim, const std::vector<double>& edig);

  void setTrnsProf(const std::vector<double>& es1, const std::vector<double>& eq1, 
		   const std::vector<double>& es2, const std::vector<double>& eq2);
  void setLongProf(const std::vector<double>& es, const std::vector<double>& eq);
  void saveHit(int det, int lay, int eta, int phi, double e, double t);

  //Vertex associated methods
  void setVtxPrim(int evNum, int type, double x, double y, double z, double u,
		  double v, double w, double px, double py, double pz); 
  void setVtxSec(int id, int pdg, double px, double py, double pz, double ek); 

private:
  //  PHcalTB04Info(const PHcalTB04Info&); 
  //  const PHcalTB04Info& operator=(const PHcalTB04Info&); 

  // ---------- member data --------------------------------

  //Beam parameters
  int          nPrimary, idBeam;
  float        eBeam, etaBeam, phiBeam;

  //Deposited energies
  float        simEtot, simEEc, simEHc;
  float        digEtot, digEEc, digEHc;
  FloatVector  esime, edige;
  FloatVector  esimh, edigh;
  FloatVector  latsimEta, latdigEta;
  FloatVector  latsimPhi, latdigPhi;
  FloatVector  longsim,   longdig;

  //Tower Members
  int          nCrystal;
  IntVector    idEcal;
  int          nTower;
  IntVector    idHcal;

  //Hit Members
  int         hit;
  IntVector   detHit, etaHit, phiHit, layHit;
  FloatVector eHit, tHit;

  //Vertex members
  int          v1EvNum, v1Type, v1Nsec;
  IntVector    v1secTrackID, v1secPartID;
  float        v1X, v1Y, v1Z, v1R, v1U, v1V, v1W;
  float        v1Px, v1Py, v1Pz;
  FloatVector  v1secPx, v1secPy, v1secPz, v1secEk;
};


#endif
