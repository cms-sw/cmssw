#ifndef _PPS_PixelG4Hit_h
#define _PPS_PixelG4Hit_h 1
// -*- C++ -*-
//
// Package:     PPS
// Class  :     PPSPixelG4Hit
//
/**\class PPSPixelG4Hit PPSPixelG4Hit.h SimG4CMS/PPS/interface/PPSPixelG4Hit.h
 
 Description: Transient Hit class for PPS taken from those for Calorimeters
 
 Usage: One Hit object should be created
   -for each new particle entering the calorimeter
   -for each detector unit (= cristal or fiber or scintillator layer)
   -for each nanosecond of the shower development

   This implies that all hit objects created for a given shower
   have the same value for
   - Entry (= local coordinates of the entrance point of the particle
              in the unit where the shower starts) 
   - the TrackID (= Identification number of the incident particle)
   - the IncidentEnergy (= energy of that particle)
 
*/
//
// Original Author:
//         Created:  Tue May 16 10:14:34 CEST 2006
//

// system include files

// user include files

#include "G4VHit.hh"
#include "G4ThreeVector.hh"
#include <iostream>

class PPSPixelG4Hit : public G4VHit {
public:
  // ---------- Constructor and destructor -----------------
  PPSPixelG4Hit();
  ~PPSPixelG4Hit() override = default;
  PPSPixelG4Hit(const PPSPixelG4Hit& right);

  // ---------- operators ----------------------------------
  const PPSPixelG4Hit& operator=(const PPSPixelG4Hit& right);
  int operator==(const PPSPixelG4Hit&) { return 0; }

  // ---------- member functions ---------------------------
  void Draw() override {}
  void Print() override;

  const G4ThreeVector& meanPosition() const { return MeanPosition_; };
  void setMeanPosition(const G4ThreeVector& a) { MeanPosition_ = a; };

  const G4ThreeVector& entryPoint() const;
  void setEntryPoint(const G4ThreeVector&);
  const G4ThreeVector& exitPoint() const;
  void setExitPoint(const G4ThreeVector&);

  double eM() const;
  void setEM(double e);

  double hadr() const;
  void setHadr(double e);

  double incidentEnergy() const;
  void setIncidentEnergy(double e);

  int trackID() const;
  void setTrackID(int i);

  uint32_t unitID() const;
  void setUnitID(uint32_t i);

  double timeSlice() const;
  void setTimeSlice(double d);
  int timeSliceID() const;

  void addEnergyDeposit(double em, double hd);
  void addEnergyDeposit(const PPSPixelG4Hit& aHit);

  double energyDeposit() const;

  float p() const;
  float tof() const;
  float energyLoss() const;
  int particleType() const;

  void setP(float e);
  void setTof(float e);
  void setEnergyLoss(float e);
  void setParticleType(short i);

  float thetaAtEntry() const;
  float phiAtEntry() const;

  void setThetaAtEntry(float t);
  void setPhiAtEntry(float f);
  float px() const;
  float py() const;
  float pz() const;
  float vPx() const;
  float vPy() const;
  float vPz() const;

  void setPx(float e);
  void setPy(float e);
  void setPz(float e);
  void setVPx(float e);
  void setVPy(float e);
  void setVPz(float e);
  float x() const;
  float y() const;
  float z() const;
  void setX(float t);
  void setY(float t);
  void setZ(float t);

  int parentId() const;
  float vx() const;
  float vy() const;
  float vz() const;

  void setParentId(int p);
  void setVx(float p);
  void setVy(float p);
  void setVz(float p);

private:
  G4ThreeVector MeanPosition_;
  double elem_;               //EnergyDeposit of EM particles
  double hadr_;               //EnergyDeposit of HD particles
  double theIncidentEnergy_;  //Energy of the primary particle
  int theTrackID_;            //Identification number of the primary
                              //particle
  uint32_t theUnitID_;        //PPS Unit Number
  double theTimeSlice_;       //Time Slice Identification

  float theX_;
  float theY_;
  float theZ_;
  float thePabs_;
  float theTof_;
  float theEnergyLoss_;
  int theParticleType_;

  float theThetaAtEntry_;
  float thePhiAtEntry_;
  G4ThreeVector theEntryPoint_;
  G4ThreeVector theExitPoint_;
  float thePx_, thePy_, thePz_, theVPx_, theVPy_, theVPz_;
  int theParentId_;
  float theVx_;
  float theVy_;
  float theVz_;
};

std::ostream& operator<<(std::ostream&, const PPSPixelG4Hit&);

#endif
