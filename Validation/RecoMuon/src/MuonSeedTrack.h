#ifndef Validation_RecoMuon_MuonSeedTrack_H
#define Validation_RecoMuon_MuonSeedTrack_H

/** \class MuonSeedTrack 
 *
 *
 * Make a (fake) reco::Track from a TrajectorySeed.  The (fake) track
 * can be associated to a TrackingParticle (SimTrack) using
 * SimTracker/TrackAssociation.  The association can then be used in
 * Validation packages, such as Validation/RecoMuon/MuonTrackValidator.cc
 *
 *
 *  \author Adam Everett        Purdue University
 */

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"

#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"

#include <DQMServices/Core/interface/DQMEDAnalyzer.h>

namespace reco {
  class Track;
}

class MuonServiceProxy;
class TrajectorySeed;
class MuonUpdatorAtVertex;
class DQMStore;
//
// class decleration
//

class MuonSeedTrack : public edm::EDProducer {
public:
  /// constructor with config
  explicit MuonSeedTrack(const edm::ParameterSet&);

  /// destructor
  ~MuonSeedTrack() override;

private:
  /// pre-job booking
  void beginJob() override;

  /// construct proto-tracks
  void produce(edm::Event&, const edm::EventSetup&) override;

  /// post-job
  void endJob() override;

  /// get the TrajectorySeed's TrajectoryStateOnSurface
  TrajectoryStateOnSurface getSeedTSOS(const TrajectorySeed& seed) const;

  /// set the Branch alias
  void setAlias(std::string alias) {
    alias.erase(alias.size() - 1, alias.size());
    theAlias = alias;
  }

  /// compute the TrajectorySeed's degree of freedom
  double computeNDOF(const TrajectorySeed&) const;

  /// Build a track at the PCA WITHOUT any vertex constriant
  std::pair<bool, reco::Track> buildTrackAtPCA(const TrajectorySeed&) const;

  // ----------member data ---------------------------

  /// the event setup proxy, it takes care the services update
  MuonServiceProxy* theService;

  /// the class used for updating a Trajectory State at veretex
  MuonUpdatorAtVertex* theUpdatorAtVtx;

  /// the Branch alias
  std::string theAlias;

  /// the propagator used for extracting TSOS from seed
  //std::string theSeedPropagatorName;

  /// the TrajectorySeed label
  edm::InputTag theSeedsLabel;
  edm::EDGetTokenT<TrajectorySeedCollection> theSeedsToken;

  ///
  bool theAllowNoVtxFlag;
};

#endif
