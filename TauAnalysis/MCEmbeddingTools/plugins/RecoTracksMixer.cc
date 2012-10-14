
/** \class RecoTracksMixer
 *
 * Merge collections of reco::Tracks and Trajectories 
 * of original Z -> mumu events (after removing the reconstructed muons)
 * and embedded tau decay products.
 * 
 * \author Tomasz Maciej Frueboes
 *
 * \version $Revision: 1.1 $
 *
 * $Id: RecoTracksMixer.h,v 1.1 2012/10/09 09:00:03 veelken Exp $
 *
 */

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "TrackingTools/PatternTools/interface/Trajectory.h"

#include <vector>

class RecoTracksMixer : public edm::EDProducer 
{
 public:
  explicit RecoTracksMixer(const edm::ParameterSet&);
  ~RecoTracksMixer() {}

 private:
  virtual void produce(edm::Event&, const edm::EventSetup&);

  edm::InputTag srcTracks1_;
  edm::InputTag srcTracks2_;

  typedef std::vector<Trajectory> TrajectoryCollection;
};

RecoTracksMixer::RecoTracksMixer(const edm::ParameterSet& cfg) 
  : srcTracks1_(cfg.getParameter<edm::InputTag>("trackCol1")),
    srcTracks2_(cfg.getParameter<edm::InputTag>("trackCol2"))
{
  produces<reco::TrackCollection>();
  produces<TrajectoryCollection>();
}

void
RecoTracksMixer::produce(edm::Event& evt, const edm::EventSetup& es)
{
  edm::Handle<reco::TrackCollection> tracks1;
  evt.getByLabel(srcTracks1_, tracks1);
  //edm::Handle<TrajectoryCollection> trajectories1;
  //evt.getByLabel(srcTracks1_, trajectories1);

  edm::Handle<reco::TrackCollection> tracks2;
  evt.getByLabel(srcTracks2_, tracks2);
  //edm::Handle<TrajectoryCollection> trajectories2;
  //evt.getByLabel(srcTracks2_, trajectories2);

  std::auto_ptr<reco::TrackCollection> mixedTracks(new reco::TrackCollection());
  mixedTracks->insert(mixedTracks->end(), tracks1->begin(), tracks1->end());
  mixedTracks->insert(mixedTracks->end(), tracks2->begin(), tracks2->end());

  //std::auto_ptr<TrajectoryCollection> mixedTrajectories(new TrajectoryCollection());
  //mixedTrajactories->insert(mixedTrajectories->end(), trajectories1->begin(), trajectories1->end());
  //mixedTrajactories->insert(mixedTrajectories->end(), trajectories2->begin(), trajectories2->end());

  evt.put(mixedTracks);
  //evt.put(mixedTrajactories);
}

#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(RecoTracksMixer);
