#ifndef MultiTrackValidatorBase_h
#define MultiTrackValidatorBase_h

/** \class MultiTrackValidatorBase
 *  Base class for analyzers that produces histrograms to validate Track Reconstruction performances
 *
 *  \author cerati
 */

#include <memory>

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/EDGetToken.h"

#include "MagneticField/Engine/interface/MagneticField.h" 
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h" 

#include "SimDataFormats/Associations/interface/TrackToTrackingParticleAssociator.h"

#include "FWCore/ServiceRegistry/interface/Service.h"

#include "SimTracker/Common/interface/TrackingParticleSelector.h"
#include "CommonTools/RecoAlgos/interface/CosmicTrackingParticleSelector.h"
#include "DataFormats/Common/interface/ValueMap.h"

#include <DQMServices/Core/interface/DQMStore.h>

#include "FWCore/Framework/interface/ConsumesCollector.h"

#include <iostream>
#include <sstream>
#include <string>

class PileupSummaryInfo;
namespace reco {
class DeDxData;
}

class MultiTrackValidatorBase {
 public:

  /// Constructor
  MultiTrackValidatorBase(const edm::ParameterSet& pset, edm::ConsumesCollector && iC, bool isSeed = false);
    
  /// Destructor
  virtual ~MultiTrackValidatorBase(){ }
  
  //virtual void initialize()=0;

 protected:

  //DQMStore* dbe_;

  // MTV-specific data members
  std::vector<edm::InputTag> associators;
  edm::EDGetTokenT<TrackingParticleCollection> label_tp_effic;
  edm::EDGetTokenT<TrackingParticleCollection> label_tp_fake;
  edm::EDGetTokenT<TrackingVertexCollection> label_tv;
  edm::EDGetTokenT<std::vector<PileupSummaryInfo> > label_pileupinfo;

  std::vector<edm::EDGetTokenT<std::vector<PSimHit> > > simHitTokens_;
  std::string parametersDefiner;


  std::vector<edm::InputTag> label;
  std::vector<edm::EDGetTokenT<edm::View<reco::Track> > > labelToken;
  std::vector<edm::EDGetTokenT<edm::View<TrajectorySeed> > > labelTokenSeed;
  edm::EDGetTokenT<reco::BeamSpot>  bsSrc;

  edm::EDGetTokenT<edm::ValueMap<reco::DeDxData> > m_dEdx1Tag;
  edm::EDGetTokenT<edm::ValueMap<reco::DeDxData> > m_dEdx2Tag;

  bool ignoremissingtkcollection_;
};


#endif
