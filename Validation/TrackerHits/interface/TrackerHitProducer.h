#ifndef TrackerHitProducer_h
#define TrackerHitProducer_h

// framework & common header files
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"
//#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/Common/interface/Provenance.h"
#include "FWCore/Framework/interface/Provenance.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "Geometry/CommonDetUnit/interface/GeomDetUnit.h"
#include "DataFormats/DetId/interface/DetId.h"

// tracker info
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/CommonDetUnit/interface/TrackingGeometry.h"
#include "DataFormats/SiStripDetId/interface/StripSubdetector.h"
#include "DataFormats/SiPixelDetId/interface/PixelSubdetector.h"


// data in edm::event
#include "SimDataFormats/TrackerValidation/interface/PTrackerSimHit.h"
#include "SimDataFormats/HepMCProduct/interface/HepMCProduct.h"
#include "SimDataFormats/Vertex/interface/EmbdSimVertexContainer.h"
#include "SimDataFormats/Track/interface/EmbdSimTrackContainer.h"
#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"
#include "SimDataFormats/CaloHit/interface/PCaloHitContainer.h"
//#include "SimDataFormats/CrossingFrame/interface/CrossingFrame.h"
//#include "SimDataFormats/CrossingFrame/interface/MixCollection.h"

// helper files
#include <CLHEP/Vector/LorentzVector.h>
#include <CLHEP/Units/SystemOfUnits.h>

#include <iostream>
#include <stdlib.h>
#include <string>
#include <memory>
#include <vector>

#include "TString.h"

class PGlobalSimHit;
  
class TrackerHitProducer : public edm::EDProducer
{
  
 public:

  typedef std::vector<float> FloatVector;
  typedef std::vector<int> IntegerVector;

  explicit TrackerHitProducer(const edm::ParameterSet&);
  virtual ~TrackerHitProducer();
  virtual void beginJob(const edm::EventSetup&);
  virtual void endJob();  
  virtual void produce(edm::Event&, const edm::EventSetup&);
  
 private:

  //TrackerHitidation(const TrackerHitidation&);   
  //const TrackerHitidation& operator=(const TrackerHitidation&);

  // production related methods
  void fillG4MC(edm::Event&);
  void storeG4MC(PTrackerSimHit&);
  void fillTrk(edm::Event&, const edm::EventSetup&);
  void storeTrk(PTrackerSimHit&);

  void clear();

 private:

  //  parameter information
  std::string fName;
  int verbosity;
  std::string label;
  bool getAllProvenances;
  bool printProvenanceInfo;

  // G4MC info
  int nRawGenPart;
  FloatVector G4VtxX; 
  FloatVector G4VtxY; 
  FloatVector G4VtxZ; 
  FloatVector G4TrkPt; 
  FloatVector G4TrkE;


  // Tracker info

  // Hit info
  IntegerVector HitsSysID;
  FloatVector HitsDuID;
  FloatVector HitsTkID; 
  FloatVector HitsProT; 
  FloatVector HitsParT; 
  FloatVector HitsP;
  FloatVector HitsLpX; 
  FloatVector HitsLpY; 
  FloatVector HitsLpZ; 
  FloatVector HitsLdX; 
  FloatVector HitsLdY; 
  FloatVector HitsLdZ; 
  FloatVector HitsLdTheta; 
  FloatVector HitsLdPhi;
  FloatVector HitsExPx; 
  FloatVector HitsExPy; 
  FloatVector HitsExPz;
  FloatVector HitsEnPx; 
  FloatVector HitsEnPy; 
  FloatVector HitsEnPz;
  FloatVector HitsEloss; 
  FloatVector HitsToF;


  // private statistics information
  unsigned int count;

}; // end class declaration
  

#endif
