#ifndef SimMuon_DTDigitizer_h
#define SimMuon_DTDigitizer_h

/** \class DTDigitizer
 *  Digitize the muon drift tubes. 
 *  The parametrisation function in DTDriftTimeParametrization 
 *  from P.G.Abia, J.Puerta is used in all cases where it is applicable. 
 *
 *  \authors: G. Bevilacqua, N. Amapane, G. Cerminara, R. Bellan
 */

#include "FWCore/Framework/interface/stream/EDProducer.h"

#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "DataFormats/DTDigi/interface/DTDigiCollection.h"
#include "SimDataFormats/DigiSimLinks/interface/DTDigiSimLinkCollection.h"
#include "DataFormats/MuonDetId/interface/DTWireId.h"

#include "DataFormats/GeometryVector/interface/LocalVector.h"
// SimHits
#include "SimDataFormats/CrossingFrame/interface/MixCollection.h"
#include "SimDataFormats/CrossingFrame/interface/CrossingFrame.h"
#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"
#include "SimDataFormats/TrackingHit/interface/PSimHit.h"



#include <vector>
#include <memory>

namespace CLHEP {
  class HepRandomEngine;
}

class DTLayer;
class PSimHit;
class DTWireType;
class DTBaseDigiSync;
class DTTopology;
class DTDigiSyncBase;


namespace edm {class ParameterSet; class Event; class EventSetup;}

class DTDigitizer : public edm::stream::EDProducer<> {
  
 public:

  explicit DTDigitizer(const edm::ParameterSet&);

  virtual void produce(edm::Event&, const edm::EventSetup&) override;
  
 private:
  typedef std::pair<const PSimHit*,float> hitAndT; // hit & corresponding time
  typedef std::vector<hitAndT> TDContainer; // hits & times for one wire

  typedef std::map<DTWireId, std::vector<const PSimHit*> > DTWireIdMap; 
  typedef DTWireIdMap::iterator DTWireIdMapIter;  
  typedef DTWireIdMap::const_iterator DTWireIdMapConstIter;  

  // Sort hits container by time.
  struct hitLessT {
    bool operator()(const  hitAndT & h1, const hitAndT & h2) {
      if (h1.second < h2.second) return true;
      return false;
    }
  };

  // Calculate the drift time for one hit. 
  // if status flag == false, hit has to be discarded.
  std::pair<float,bool> computeTime(const DTLayer* layer,const DTWireId &wireId, 
				    const PSimHit *hit, 
				    const LocalVector &BLoc,
                                    CLHEP::HepRandomEngine*); //FIXME??
  
  // Calculate the drift time using the GARFIELD cell parametrization,
  // taking care of all conversions from CMSSW local coordinates
  // to the conventions used for the parametrization.
  std::pair<float,bool> driftTimeFromParametrization(float x, float alpha, float By,
						     float Bz, CLHEP::HepRandomEngine*) const;
  
  // Calculate the drift time for the cases where it is not possible
  // to use the GARFIELD cell parametrization.
  std::pair<float,bool> driftTimeFromTimeMap() const;
  
  // Add all delays other than drift times (signal propagation along the wire, 
  // TOF etc.; subtract calibration time.
  float externalDelays(const DTLayer* layer,
		       const DTWireId &wireId, 
		       const PSimHit *hit) const;

  // Store digis for one wire, taking into account the dead time.
  //FiXME put alias for the map.
  void storeDigis(DTWireId &wireId, 
		  TDContainer &hits,
		  DTDigiCollection &output, DTDigiSimLinkCollection &outputLinks);
  
  // Debug output
  void dumpHit(const PSimHit * hit, float xEntry, float xExit, const DTTopology &topo);
  
  // Double half-gaussian smearing.
  float asymGausSmear(double mean, double sigmaLeft, double sigmaRight, CLHEP::HepRandomEngine*) const;
  
  // Allow debugging and testing.
  friend class DTDigitizerAnalysis;

  //Its Atributes:
  double vPropWire;
  float deadTime;
  float smearing;
  bool debug;
  bool interpolate;
  bool onlyMuHits;

  std::string syncName;
  std::unique_ptr<DTDigiSyncBase> theSync;

  std::string geometryType;

  // Ideal model. Used for debug
  bool IdealModel;
  float theConstVDrift;  

  // to configure the creation of Digi-Sim links
  bool MultipleLinks;
  float LinksTimeWindow;
  
  //Name of Collection use for create the XF 
  std::string mix_;
  std::string collection_for_XF;

  edm::EDGetTokenT<CrossingFrame<PSimHit> > cf_token;


};
#endif
