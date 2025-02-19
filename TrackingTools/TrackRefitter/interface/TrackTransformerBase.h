#ifndef TrackingTools_TrackRefitter_TrackTransformerBase_H
#define TrackingTools_TrackRefitter_TrackTransformerBase_H

/** \class TrackTransformerBase
 *  Base class for Track transformer classes
 *
 *  $Date: 2008/11/05 11:30:25 $
 *  $Revision: 1.1 $
 *  \author R. Bellan - CERN <riccardo.bellan@cern.ch>
 */

#include "DataFormats/TrackReco/interface/TrackFwd.h"

class Trajectory;
namespace edm {class EventSetup;}

class TrackTransformerBase {
public:
  /// Constructor
  TrackTransformerBase(){};

  /// Destructor
  virtual ~TrackTransformerBase(){};

  // Operations

  /// Convert a reco::Track into Trajectory
  virtual  std::vector<Trajectory> transform(const reco::Track&) const =0;

  /// set the services needed by the TrackTransformers
  virtual void setServices(const edm::EventSetup&) = 0;

protected:

private:

};
#endif

