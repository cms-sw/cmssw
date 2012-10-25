#include "TauAnalysis/MCEmbeddingTools/plugins/MuonCaloCleanerByDistance.h"

#include "FWCore/Utilities/interface/Exception.h"

#include "DataFormats/Common/interface/Handle.h"

#include <string>
#include <vector>

MuonCaloCleanerByDistance::MuonCaloCleanerByDistance(const edm::ParameterSet& cfg)
  : srcDistanceMapMuPlus_(cfg.getParameter<edm::InputTag>("distanceMapMuPlus")),
    srcDistanceMapMuMinus_(cfg.getParameter<edm::InputTag>("distanceMapMuMinus"))
{
  edm::ParameterSet cfgEnergyDepositPerDistance = cfg.getParameter<edm::ParameterSet>("energyDepositPerDistance");
  typedef std::vector<std::string> vstring;
  vstring detNames = cfgEnergyDepositPerDistance.getParameterNamesForType<double>();
  for ( vstring::const_iterator detName = detNames.begin();
	detName != detNames.end(); ++detName ) {
    energyDepositPerDistance_[*detName] = cfgEnergyDepositPerDistance.getParameter<double>(*detName);
  }

  // maps of detId to expected energy deposits of muon
  produces<detIdToFloatMap>("energyDepositsMuPlus");
  produces<detIdToFloatMap>("energyDepositsMuMinus");
}

MuonCaloCleanerByDistance::~MuonCaloCleanerByDistance()
{
// nothing to be done yet...
}

void MuonCaloCleanerByDistance::produce(edm::Event& evt, const edm::EventSetup& es)
{
  std::auto_ptr<detIdToFloatMap> energyDepositsMuPlus(new detIdToFloatMap());
  std::auto_ptr<detIdToFloatMap> energyDepositsMuMinus(new detIdToFloatMap());
  
  edm::Handle<detIdToFloatMap> distanceMapMuPlus;
  evt.getByLabel(srcDistanceMapMuPlus_, distanceMapMuPlus);
  edm::Handle<detIdToFloatMap> distanceMapMuMinus;
  evt.getByLabel(srcDistanceMapMuMinus_, distanceMapMuMinus);

  fillEnergyDepositMap(*distanceMapMuPlus, *energyDepositsMuPlus);
  fillEnergyDepositMap(*distanceMapMuMinus, *energyDepositsMuMinus);

  evt.put(energyDepositsMuPlus, "energyDepositsMuPlus");
  evt.put(energyDepositsMuMinus, "energyDepositsMuMinus");
}

void MuonCaloCleanerByDistance::fillEnergyDepositMap(const detIdToFloatMap& distanceMap, detIdToFloatMap& energyDepositMap)
{
  for ( detIdToFloatMap::const_iterator rawDetId_and_distance = distanceMap.begin();
	rawDetId_and_distance != distanceMap.end(); ++rawDetId_and_distance ) {
    DetId detId(rawDetId_and_distance->first);

    std::string key = detNaming_.getKey(detId);

    if ( energyDepositPerDistance_.find(key) == energyDepositPerDistance_.end() )
      throw cms::Exception("MuonCaloCleanerByDistance") 
	<< "No mean energy deposit defined for detId = " << detId.rawId() << " (key = " << key << ") !!\n";
    
    double energyDepositPerDistance_value = energyDepositPerDistance_[key];

    energyDepositMap[rawDetId_and_distance->first] += rawDetId_and_distance->second*energyDepositPerDistance_value;
  }
}

#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(MuonCaloCleanerByDistance);




