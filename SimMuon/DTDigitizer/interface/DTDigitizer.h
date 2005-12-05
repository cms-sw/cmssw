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
// $Id: DTDigitizer.h,v 1.2 2005/11/30 17:33:33 bellan Exp $
//

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Handle.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "FWCore/EDProduct/interface/EDProduct.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
//
#include <vector>
#include "DataFormats/DTDigi/interface/DTDigiCollection.h"
#include "DataFormats/MuonDetId/interface/DTDetId.h"
#include "Geometry/CommonTopologies/interface/DTTopology.h"

using namespace std;

class DTGeomDetUnit;
class PSimHit;
class DTWireType;
class DTBaseDigiSync;

class DTDigitizer : public edm::EDProducer {
  
 public:
  explicit DTDigitizer(const edm::ParameterSet&);
  ~DTDigitizer();
  virtual void produce(edm::Event&, const edm::EventSetup&);
  
 private:
  typedef pair<const PSimHit*,float> hitAndT; // hit & corresponding time
  typedef vector<hitAndT> TDContainer; // hits & times for one wire

  typedef map<DTDetId, vector<PSimHit> > DTDetIdMap;
  typedef map<DTDetId, vector<PSimHit> >::iterator DTDetIdMapIter;  
  typedef map<DTDetId, vector<PSimHit> >::const_iterator DTDetIdMapConstIter;  

  // Sort hits container by time.
  struct hitLessT {
    bool operator()(const  hitAndT & h1, const hitAndT & h2) {
      if (h1.second < h2.second) return true;
      return false;
    }
  };

  // Calculate the drift time for one hit. 
  // if status flag == false, hit has to be discarded.
  pair<float,bool> computeTime(const DTGeomDetUnit* layer,const DTDetId &wireId, const PSimHit &hit) ;

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
  float externalDelays(const DTTopology &topo, 
		       const DTDetId &wireId, 
		       const PSimHit *hit) const;

  // Store digis for one wire, taking into account the dead time.
  //FiXME put alias for the map.
  void storeDigis(DTDetId &wireId, 
		  DTDetIdMapConstIter &wire,
		  DTDetIdMapIter end,
		  TDContainer &hits,
		  DTDigiCollection &output);

  void loadOutput(DTDigiCollection &output,
		  vector<DTDigi> &digis, DTDetId &layerID);

  // Debug output
  void dumpHit(const PSimHit * hit, float xEntry, float xExit, const DTTopology &topo);


  // Check if given point (in cell r.f.) is on cell borders.
  enum sides {zMin,zMax,xMin,xMax,yMin,yMax,none}; // sides of the cell
  sides onWhichBorder_old(float x, float y, float z, const DTTopology& topo);
  sides onWhichBorder(float x, float y, float z, const DTTopology& topo);

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
