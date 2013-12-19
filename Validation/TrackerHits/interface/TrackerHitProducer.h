#ifndef TrackerHitProducer_h
#define TrackerHitProducer_h

// framework & common header files
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"


#include <string>
#include <vector>

class PTrackerSimHit;

class TrackerHitProducer : public edm::EDProducer
{
  
 public:

  typedef std::vector<float> FloatVector;
  typedef std::vector<int> IntegerVector;

  explicit TrackerHitProducer(const edm::ParameterSet&);
  virtual ~TrackerHitProducer();
  virtual void beginJob();
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
  FloatVector G4TrkEta;
  FloatVector G4TrkPhi;


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
  
  edm::InputTag SiTIBLowSrc_;
  edm::InputTag SiTIBHighSrc_;
  edm::InputTag SiTOBLowSrc_;
  edm::InputTag SiTOBHighSrc_;
  edm::InputTag SiTIDLowSrc_;
  edm::InputTag SiTIDHighSrc_;
  edm::InputTag SiTECLowSrc_;
  edm::InputTag SiTECHighSrc_;
  edm::InputTag PxlBrlLowSrc_;
  edm::InputTag PxlBrlHighSrc_;
  edm::InputTag PxlFwdLowSrc_;
  edm::InputTag PxlFwdHighSrc_;
  edm::InputTag G4VtxSrc_;
  edm::InputTag G4TrkSrc_;

  edm::ParameterSet config_;
  std::string HepMClabel_;
  std::string HepMCinstance_;
 
  // private statistics information
  unsigned int count;

}; // end class declaration
  

#endif
