#ifndef HcalTestBeam_PHcalTB06Info_h
#define HcalTestBeam_PHcalTB06Info_h
// -*- C++ -*-
//
// Package:     HcalTestBeam
// Class  :     PHcalTB06Info
// 
/**\class PHcalTB06Info PHcalTB06Info.h SimDataFormats/HcalTestBeam/interface/PHcalTB06Info.h

 Description: Histogram handling class for analysis

 Usage:
    Simulation information for test beam studies of 2004 Test Beam
    Contains beam information, hits and digitised results

*/
//
// Original Author:  Sunanda Banerjee
//         Created:  Tue Oct 10 10:14:34 CEST 2006
//

// system include files
#include <string>
#include <vector>
#include <memory>
#include <cmath>

// user include files

class PHcalTB06Info {

public:
  PHcalTB06Info();
  virtual ~PHcalTB06Info();

  struct Vtx {
    Vtx(): trackID(0), partID(0), px(0), py(0), pz(0), eKin(0) {}
    int   trackID;
    int   partID;
    float px;
    float py;
    float pz;
    float eKin;
  };

  struct Hit {
    Hit(): id(0), eta(0), phi(0), e(0), t(0) {}
    unsigned int id;
    float    eta;
    float    phi;
    float    e;
    float    t;
  };

  typedef std::vector<float>  FloatVector;
  typedef std::vector<int>    IntVector;
  typedef std::vector<Vtx>    VtxVector;
  typedef std::vector<Hit>    HitVector;

  // ---------- const member functions ---------------------
  int          primary()                    const {return nPrimary;}
  int          partID()                     const {return idBeam;}
  float        initE()                      const {return eBeam;}
  float        eta()                        const {return etaBeam;}
  float        phi()                        const {return phiBeam;}
  float        simEtotal()                  const {return simEtot;}
  float        simEcE()                     const {return simEEc;}
  float        simHcE()                     const {return simEHc;}

  HitVector    simHits()                    const {return hits;}
  Hit          simHit(unsigned int i)       const {return hits[i];}
  unsigned int simHitID(unsigned int i)     const {return hits[i].id;}
  float        simHitEta(unsigned int i)    const {return hits[i].eta;}
  float        simHitPhi(unsigned int i)    const {return hits[i].phi;}
  float        simHitE(unsigned int i)      const {return hits[i].e;}
  float        simHitT(unsigned int i)      const {return hits[i].t;}

  int          evNum()                      const {return v1EvNum;}
  int          vtxType()                    const {return v1Type;}
  float        vtxPrimX()                   const {return v1X;}
  float        vtxPrimY()                   const {return v1Y;}
  float        vtxPrimZ()                   const {return v1Z;}
  float        vtxPrimR()                   const {return v1R;}
  float        vtxPrimU()                   const {return v1U;}
  float        vtxPrimV()                   const {return v1V;}
  float        vtxPrimW()                   const {return v1W;}
  float        vtxPrimPx()                  const {return v1Px;}
  float        vtxPrimPy()                  const {return v1Py;}
  float        vtxPrimPz()                  const {return v1Pz;}
  int          vtxSec()                     const {return v1Nsec;}
  VtxVector    vtxSecondaries()             const {return v1Sec;}
  Vtx          vtxSecondary(unsigned int i) const {return v1Sec[i];}
  int          vtxTrackID(unsigned int i)   const {return v1Sec[i].trackID;}
  int          vtxPartID(unsigned int i)    const {return v1Sec[i].partID;}
  float        vtxSecPx(unsigned int i)     const {return v1Sec[i].px;}
  float        vtxSecPy(unsigned int i)     const {return v1Sec[i].py;}
  float        vtxSecPz(unsigned int i)     const {return v1Sec[i].pz;}
  float        vtxSecEKin(unsigned int i)   const {return v1Sec[i].eKin;}

  // ---------- static member functions --------------------

  // ---------- member functions ---------------------------
  void clear();
  void setPrimary(int primary, int id, double energy, double eta, double phi);
  void setEdep(double simtot, double sime, double simh);
  void saveHit(unsigned int det, double eta, double phi, double e, double t);

  //Vertex associated methods
  void setVtxPrim(int evNum, int type, double x, double y, double z, double u,
		  double v, double w, double px, double py, double pz); 
  void setVtxSec(int id, int pdg, double px, double py, double pz, double ek); 

private:
  //  PHcalTB06Info(const PHcalTB06Info&); 
  //  const PHcalTB06Info& operator=(const PHcalTB06Info&); 

  // ---------- member data --------------------------------

  //Beam parameters
  int          nPrimary, idBeam;
  float        eBeam, etaBeam, phiBeam;

  //Deposited energies
  float        simEtot, simEEc, simEHc;
  float        digEtot, digEEc, digEHc;

  //Hit Members
  int         hit;
  HitVector   hits;

  //Vertex members
  int          v1EvNum, v1Type, v1Nsec;
  float        v1X, v1Y, v1Z, v1R, v1U, v1V, v1W;
  float        v1Px, v1Py, v1Pz;
  VtxVector    v1Sec;
};


#endif
