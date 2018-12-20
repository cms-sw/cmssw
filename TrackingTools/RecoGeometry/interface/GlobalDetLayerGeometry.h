#ifndef TT_RecoGeometry_GlobalDetLayerGeometry_h
#define TT_RecoGeometry_GlobalDetLayerGeometry_h

/**
 * Global "reconstruction" geometry. It implements the idToLayer() method for both 
 * Tracker and Muon layers.
 * \author Boris Mangano (UCSD)  1/7/2009
 */

#include "DataFormats/DetId/interface/DetId.h"
#include "RecoTracker/TkDetLayers/interface/GeometricSearchTracker.h"
#include "RecoMTD/DetLayers/interface/MTDDetLayerGeometry.h"
#include "RecoMuon/DetLayers/interface/MuonDetLayerGeometry.h"
#include "TrackingTools/DetLayers/interface/DetLayerGeometry.h"

#include <vector>

class DetLayer;

class GlobalDetLayerGeometry: public DetLayerGeometry {
 public:
  GlobalDetLayerGeometry(const GeometricSearchTracker* tracker,
			const MuonDetLayerGeometry* muon):
  tracker_(tracker),muon_(muon),mtd_(nullptr){};

  GlobalDetLayerGeometry(const GeometricSearchTracker* tracker,
			 const MuonDetLayerGeometry* muon,
			 const MTDDetLayerGeometry* mtd):
  tracker_(tracker),muon_(muon),mtd_(mtd){};
	
	~GlobalDetLayerGeometry() override {}
  
  /*
  const std::vector<DetLayer*>& allLayers() const =0;
  const std::vector<DetLayer*>& barrelLayers() const =0;
  const std::vector<DetLayer*>& negForwardLayers() const =0;
  const std::vector<DetLayer*>& posForwardLayers() const =0;
  */


  /// Give the DetId of a module, returns the pointer to the corresponding DetLayer
  const DetLayer* idToLayer(const DetId& detId) const override;
 
 private:
  const GeometricSearchTracker* tracker_;
  const MuonDetLayerGeometry* muon_;
  const MTDDetLayerGeometry* mtd_;
};


#endif
