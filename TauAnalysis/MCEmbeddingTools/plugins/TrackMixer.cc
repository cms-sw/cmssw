
/** \class TracksMixer
 *
 * Merge collections of reco::Tracks
 * of original Z -> mumu events (after removing the reconstructed muons)
 * and embedded tau decay products.
 * 
 * \author Tomasz Maciej Frueboes;
 *         Christian Veelken, LLR
 *
 * \version $Revision: 1.5 $
 *
 * $Id: TracksMixer.cc,v 1.5 2012/10/14 12:22:24 veelken Exp $
 *
 */

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"

#include "TauAnalysis/MCEmbeddingTools/interface/TrackMixerBase.h"

#include <vector>

class TrackMixer : public TrackMixerBase 
{
 public:
  explicit TrackMixer(const edm::ParameterSet&);
  ~TrackMixer() {}

 private:
  virtual void produceTrackExtras(edm::Event&, const edm::EventSetup&) {};
};

TrackMixer::TrackMixer(const edm::ParameterSet& cfg) 
  : TrackMixerBase(cfg)
{}

#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(TrackMixer);
