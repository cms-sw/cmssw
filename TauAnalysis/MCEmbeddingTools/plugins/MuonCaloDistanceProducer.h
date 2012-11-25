#ifndef TauAnalysis_MCEmbeddingTools_MuonCaloDistanceProducer_h
#define TauAnalysis_MCEmbeddingTools_MuonCaloDistanceProducer_h

/** \class MuonCaloDistanceProducer
 *
 * Compute distance traversed by muon through calorimeter cells.
 *
 * NOTE: The output of this class is used as input for MuonCaloCleanerByDistance.
 *
 * \author Tomasz Maciej Frueboes;
 *         Christian Veelken, LLR
 *
 * \version $Revision: 1.1 $
 *
 * $Id: MuonCaloDistanceProducer.h,v 1.1 2012/10/24 09:37:14 veelken Exp $
 *
 */

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "TrackingTools/TrackAssociator/interface/TrackDetectorAssociator.h"
#include "TrackingTools/TrackAssociator/interface/TrackAssociatorParameters.h"

#include <DataFormats/MuonReco/interface/Muon.h>

class MuonCaloDistanceProducer : public edm::EDProducer 
{
 public:
  explicit MuonCaloDistanceProducer(const edm::ParameterSet&);
  ~MuonCaloDistanceProducer();

 private:
  virtual void produce(edm::Event&, const edm::EventSetup&);

  typedef std::map<uint32_t, float> detIdToFloatMap;
  void fillDistanceMap(edm::Event&, const edm::EventSetup&, const reco::Candidate*, detIdToFloatMap&);
      
  edm::InputTag srcSelectedMuons_;

  TrackDetectorAssociator trackAssociator_;
  TrackAssociatorParameters trackAssociatorParameters_;
};

#endif
