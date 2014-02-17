#ifndef SimDataFormats_Forward_TotemTestHistoClass_h
#define SimDataFormats_Forward_TotemTestHistoClass_h 1
// -*- C++ -*-
//
// Package:     Forward
// Class  :     TotemTestHistoClass
//
/**\class TotemTestHistoClass TotemTestHistoClass.h SimG4CMS/Forward/interface/TotemTestHistoClass.h
 
 Description: Content of the Root file for Totem Tests
 
 Usage:
    Used in testing Totem simulation
 
*/
//
// Original Author: 
//         Created:  Tue May 16 10:14:34 CEST 2006
// $Id: TotemTestHistoClass.h,v 1.1 2006/11/16 16:40:31 sunanda Exp $
//
 
// system include files
#include <vector>

// user include files

class TotemTestHistoClass {

public:

  // ---------- Constructor and destructor -----------------
  explicit TotemTestHistoClass();
  virtual ~TotemTestHistoClass();

  struct Hit {
    Hit() {}
    int   UID;
    int   Ptype;
    int   TID;
    int   PID;
    float ELoss;
    float PABS;
    float x;
    float y;
    float z;
    float vx;
    float vy;
    float vz;
  };
	 
  // ---------- Member functions --------------------------- 
  int getEVT()      const {return evt;}
  int getNHit()     const {return hits;}
  std::vector<Hit> getHits() const {return hit;}

  void setEVT(int v)      {evt=v;}
  void fillHit(int uID, int pType, int tID, int pID, float eLoss, float pAbs,
	       float vX, float vY, float vZ, float x, float y, float z);

private: 
	 
  // ---------- Private Data members ----------------------- 
  int              evt, hits;
  std::vector<Hit> hit;

};

#endif
