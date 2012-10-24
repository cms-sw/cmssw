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
 * \version $Revision: 1.1 $
 *
 * $Id: MuonCaloCleanerByDistance.h,v 1.1 2012/10/14 12:22:24 veelken Exp $
 *
 */

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "TauAnalysis/MCEmbeddingTools/interface/DetNaming.h"

#include <boost/foreach.hpp>

class MuonCaloCleanerByDistance : public edm::EDProducer 
{
 public:
  explicit MuonCaloCleanerByDistance(const edm::ParameterSet&);
  ~MuonCaloCleanerByDistance();

 private:
  virtual void produce(edm::Event&, const edm::EventSetup&);

  typedef std::map<uint32_t, float> detIdToFloatMap;
  void fillEnergyDepositMap(const detIdToFloatMap&, detIdToFloatMap&);
      
  edm::InputTag srcDistanceMapMuPlus_;
  edm::InputTag srcDistanceMapMuMinus_;

  DetNaming detNaming_;

  std::map<std::string, double> energyDepositPerDistance_;
};

#endif


