#ifndef TrackerHitProducer_h
#define TrackerHitProducer_h

// framework & common header files
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "SimDataFormats/Track/interface/SimTrackContainer.h"
#include "SimDataFormats/Vertex/interface/SimVertexContainer.h"
#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"

#include <string>
#include <vector>

namespace edm {
  class HepMCProduct;
}
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

  // production related methods
  void fillG4MC(edm::Event&);
  void storeG4MC(PTrackerSimHit&);
  void fillTrk(edm::Event&, const edm::EventSetup&);
  void storeTrk(PTrackerSimHit&);

  void clear();

 private:

  //  parameter information
  bool getAllProvenances;
  bool printProvenanceInfo;
  int verbosity;

  // private statistics information
  unsigned int count;

  int nRawGenPart;

  edm::ParameterSet config_;

  edm::EDGetTokenT<edm::HepMCProduct> edmHepMCProductToken_;
  edm::EDGetTokenT<edm::SimVertexContainer> edmSimVertexContainerToken_;
  edm::EDGetTokenT<edm::SimTrackContainer> edmSimTrackContainerToken_;
  edm::EDGetTokenT<edm::PSimHitContainer> edmPSimHitContainer_pxlBrlLow_Token_, edmPSimHitContainer_pxlBrlHigh_Token_;
  edm::EDGetTokenT<edm::PSimHitContainer> edmPSimHitContainer_pxlFwdLow_Token_, edmPSimHitContainer_pxlFwdHigh_Token_;
  edm::EDGetTokenT<edm::PSimHitContainer> edmPSimHitContainer_siTIBLow_Token_, edmPSimHitContainer_siTIBHigh_Token_;
  edm::EDGetTokenT<edm::PSimHitContainer> edmPSimHitContainer_siTOBLow_Token_, edmPSimHitContainer_siTOBHigh_Token_;
  edm::EDGetTokenT<edm::PSimHitContainer> edmPSimHitContainer_siTIDLow_Token_, edmPSimHitContainer_siTIDHigh_Token_;
  edm::EDGetTokenT<edm::PSimHitContainer> edmPSimHitContainer_siTECLow_Token_, edmPSimHitContainer_siTECHigh_Token_;


  // G4MC info
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
  
  std::string fName;
  std::string label;

}; // end class declaration
  

#endif
