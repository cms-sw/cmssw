#ifndef EcalTestBeam_PEcalTBInfo_h
#define EcalTestBeam_PEcalTBInfo_h
// -*- C++ -*-
//
// Package:     EcalTestBeam
// Class  :     PEcalTBInfo
// 
//
// $Id: PEcalTBInfo.h,v 1.2 2006/10/25 16:58:04 fabiocos Exp $
//

// system include files
#include <string>
#include <vector>
#include <memory>

// user include files

class PEcalTBInfo {

  typedef std::vector<float>  FloatVector;
  typedef std::vector<int>    IntVector;

public:
  PEcalTBInfo();
  virtual ~PEcalTBInfo();

  // ---------- const member functions ---------------------
  int         nCrystal()    const {return nCrystal_; }

  double      etaBeam()     const {return etaBeam_; }
  double      phiBeam()     const {return phiBeam_; }
  double      dXbeam()      const {return dXbeam_; }
  double      dYbeam()      const {return dYbeam_; }

  double      evXbeam()     const {return evXbeam_; }
  double      evYbeam()     const {return evYbeam_; }

  double      phaseShift()  const {return phaseShift_;}

  // ---------- static member functions --------------------

  // ---------- member functions ---------------------------
  void clear();

  void setCrystal(int nCrystal);
  void setBeamDirection(double etaBeam, double phiBeam);
  void setBeamOffset(double dXbeam, double dYbeam);

  void setBeamPosition(double evXbeam, double evYbeam);
  void setPhaseShift(double phaseShift);

private:
  //  PEcalTBInfo(const PEcalTBInfo&); 
  //  const PEcalTBInfo& operator=(const PEcalTBInfo&); 

  // ---------- member data --------------------------------

  //fixed run beam parameters

  int nCrystal_;

  double etaBeam_,phiBeam_;
  double dXbeam_,dYbeam_;

  //event beam parameters

  double evXbeam_,evYbeam_;

  // phase
  double phaseShift_;
};


#endif
