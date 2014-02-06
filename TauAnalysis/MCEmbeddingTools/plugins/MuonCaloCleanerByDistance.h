#ifndef TauAnalysis_MCEmbeddingTools_MuonCaloCleanerByDistance_h
#define TauAnalysis_MCEmbeddingTools_MuonCaloCleanerByDistance_h

/** \class MuonCaloCleanerByDistance
 *
 * Produce collections of calorimeter recHits
 * from which energy deposits of muons are subtracted.
 *
 * This module attributes to the muon energy
 * proportional to the distance traversed by the muon through each calorimeter cell.
 * 
 * \author Tomasz Maciej Frueboes;
 *         Christian Veelken, LLR
 *
 * \version $Revision: 1.4 $
 *
 * $Id: MuonCaloCleanerByDistance.h,v 1.4 2013/02/05 20:01:19 veelken Exp $
 *
 */

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "TauAnalysis/MCEmbeddingTools/interface/DetNaming.h"

#include <DataFormats/Candidate/interface/Candidate.h>
#include <DataFormats/Candidate/interface/CandidateFwd.h>

#include <boost/foreach.hpp>

#include <string>

class MuonCaloCleanerByDistance : public edm::EDProducer 
{
 public:
  explicit MuonCaloCleanerByDistance(const edm::ParameterSet&);
  ~MuonCaloCleanerByDistance();

 private:
  virtual void produce(edm::Event&, const edm::EventSetup&);

  typedef std::map<uint32_t, float> detIdToFloatMap;
  void fillEnergyDepositMap(const reco::Candidate& muon, const detIdToFloatMap&, detIdToFloatMap&, double&, double&);
      
  std::string moduleLabel_;

  edm::InputTag srcSelectedMuons_;
  edm::InputTag srcDistanceMapMuPlus_;
  edm::InputTag srcDistanceMapMuMinus_;

  DetNaming detNaming_;

  std::map<std::string, double> energyDepositCorrection_;

  int verbosity_;
};

#endif


