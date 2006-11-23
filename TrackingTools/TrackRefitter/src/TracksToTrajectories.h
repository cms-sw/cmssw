#ifndef TrackingTools_TrackRefitter_TracksToTrajectories_H
#define TrackingTools_TrackRefitter_TracksToTrajectories_H

/** \class TracksToTrajectories
 *  No description available.
 *
 *  $Date: 2006/11/22 18:36:45 $
 *  $Revision: 1.1 $
 *  \author R. Bellan - INFN Torino <riccardo.bellan@cern.ch>
 */

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/ParameterSet/interface/InputTag.h"

namespace edm {class ParameterSet; class Event; class EventSetup;}
class TrackTransformer;

class TracksToTrajectories: public edm::EDProducer{
public:

  /// Constructor
  TracksToTrajectories(const edm::ParameterSet&);

  // Operations

  /// Convert Tracks into Trajectories
  virtual void produce(edm::Event&, const edm::EventSetup&);

  /// Destructor
  virtual ~TracksToTrajectories();
  
protected:
  
private:

  edm::InputTag theTracksLabel;
  TrackTransformer *theTrackTransformer;
};
#endif

