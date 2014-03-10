#ifndef GEMValidation_BaseMatcher_h
#define GEMValidation_BaseMatcher_h

/**\class BaseMatcher

  Base for Sim and Trigger info matchers for SimTrack in CSC & GEM

 Original Author:  "Vadim Khotilovich"
 $Id: BaseMatcher.h,v 1.1 2013/02/11 07:33:06 khotilov Exp $

*/

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include <SimDataFormats/Track/interface/SimTrackContainer.h>
#include <SimDataFormats/Vertex/interface/SimVertexContainer.h>

#include "MagneticField/Engine/interface/MagneticField.h"
#include "TrackingTools/GeomPropagators/interface/Propagator.h"

//static const float AVERAGE_GEM_Z(587.5); // [cm]
static const float AVERAGE_GEM_Z(568.6); // [cm]

class BaseMatcher
{
public:
  BaseMatcher(const SimTrack& t, const SimVertex& v,
      const edm::ParameterSet& ps, const edm::Event& ev, const edm::EventSetup& es);

  ~BaseMatcher();

  // non-copyable
  BaseMatcher(const BaseMatcher&) = delete;
  BaseMatcher& operator=(const BaseMatcher&) = delete;


  const SimTrack& trk() const {return trk_;}
  const SimVertex& vtx() const {return vtx_;}

  const edm::ParameterSet& conf() const {return conf_;}

  const edm::Event& event() const {return ev_;}
  const edm::EventSetup& eventSetup() const {return es_;}

  void setVerbose(int v) { verbose_ = v; }
  int verbose() const { return verbose_; }

  /// general interface to propagation
  GlobalPoint propagateToZ(GlobalPoint &inner_point, GlobalVector &inner_vector, float z) const;

  /// propagation for a track starting from a vertex
  GlobalPoint propagateToZ(float z) const;

  /// propagate the track to average GEM z-position                                                                            
  GlobalPoint propagatedPositionGEM() const;

private:

  const SimTrack& trk_;
  const SimVertex& vtx_;

  const edm::ParameterSet& conf_;

  const edm::Event& ev_;
  const edm::EventSetup& es_;

  int verbose_;

  edm::ESHandle<MagneticField> magfield_;
  edm::ESHandle<Propagator> propagator_;
  edm::ESHandle<Propagator> propagatorOpposite_;
};

#endif
