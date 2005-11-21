#ifndef DTDIGITIZER_H
#define DTDIGITIZER_H
//
// class decleration
//
// -*- C++ -*-
//
// Package:    DTDigitizer
// Class:      DTDigitizer
// 
/*
 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Riccardo Bellan
//         Created:  Fri Nov  4 18:56:35 CET 2005
// $Id$
//

#include "FWCore/ParameterSet/interface/ParameterSet.h"
//
#include <vector>
#include <pair>


class DTDigitizer : public edm::EDProducer {
  
 public:
  explicit DTDigitizer(const edm::ParameterSet&);
  ~DTDigitizer();
  virtual void produce(edm::Event&, const edm::EventSetup&);
  
 private:
  typedef pair<const PSimHit*,float> hitAndT; // hit & corresponding time
  typedef vector<hitAndT> TDContainer; // hits & times for one wire
  
  // Sort hits container by time.
  struct hitLessT {
    bool operator()(const  hitAndT & h1, const hitAndT & h2) {
      if (h1.second < h2.second) return true;
      return false;
    }
  };

  // Calculate the drift time for one hit. 
  // if status flag == false, hit has to be discarded.
  pair<float,bool> computeTime(DTGeomDetUnit* layer,const DTDetId &wireId, const PSimHit hit) ;

  // Calculate the drift time using the GARFIELD cell parametrization,
  // taking care of all conversions from CMSSW local coordinates
  // to the conventions used for the parametrization.
  pair<float,bool> driftTimeFromParametrization(float x, float alpha, float By,
						float Bz) const;
  
  // Calculate the drift time for the cases where it is not possible
  // to use the GARFIELD cell parametrization.
  pair<float,bool> driftTimeFromTimeMap() const;
  
  // Add all delays other than drift times (signal propagation along the wire, 
  // TOF etc.; subtract calibration time.
  float externalDelays( .. ..);  
  /* OLD
  float externalDelays(MuBarBaseReadout * stat,
		       const MuBarWireId & wireId,
		       const PSimHit * hit) const;*/ 

  // Store digis for one wire, taking into account the dead time.
  void storeDigis(... ... );
  /* OLD
  void storeDigis(MuBarBaseReadout* stat, const MuBarWireId & wireId,
		  TDContainer & hits);*/

  // Debug output
  void dumpHit(const PSimHit * hit, float xEntry, float xExit, DTWireType* wire_type);

  // Check if given point (in cell r.f.) is on cell borders.
  enum sides {zMin,zMax,xMin,xMax,yMin,yMax,none}; // sides of the cell
  sides onWhichBorder(float x, float y, float z, DTWireType* wire_type);
    
  // Double half-gaussian smearing.
  float asymGausSmear(double mean, double sigmaLeft, double sigmaRight) const;

  // Additional "synchronization" delays
  DTBaseDigiSync * theSync; // ci sara' ancora??

  // Allow debugging and testing.
  friend class DTDigitizerAnalysis;

  // Parameter Set:
  edm::ParameterSet conf_;
  // Its Atributes:
  double vPropWire;
  float deadTime;
  float smearing;
  bool debug;
  bool interpolate;
  bool onlyMuHits;
};
#endif
