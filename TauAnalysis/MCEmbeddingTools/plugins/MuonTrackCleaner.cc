
/** \class MuonTrackCleaner
 *
 * Produce collection of reco::Tracks in Z --> mu+ mu- event
 * from which the two muons are removed 
 * (later to be replaced by simulated tau decay products) 
 * 
 * \authors Tomasz Maciej Frueboes;
 *          Christian Veelken, LLR
 *
 * \version $Revision: 1.2 $
 *
 * $Id: MuonTrackCleaner.cc,v 1.2 2013/03/29 15:55:19 veelken Exp $
 *
 */

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/Common/interface/AssociationMap.h"
#include "DataFormats/Common/interface/OneToOne.h"
#include "DataFormats/Math/interface/deltaR.h"

#include "TauAnalysis/MCEmbeddingTools/interface/MuonTrackCleanerBase.h"

#include <vector>
#include <algorithm>

class MuonTrackCleaner : public MuonTrackCleanerBase
{
 public:
  explicit MuonTrackCleaner(const edm::ParameterSet& cfg)
    : MuonTrackCleanerBase(cfg)
  {}
  ~MuonTrackCleaner() {}

 private:
  virtual void produceTrackExtras(edm::Event&, const edm::EventSetup&) {}
};

#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(MuonTrackCleaner);
