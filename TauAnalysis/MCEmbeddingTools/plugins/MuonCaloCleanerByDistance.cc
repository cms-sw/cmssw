#include "TauAnalysis/MCEmbeddingTools/plugins/MuonCaloCleanerByDistance.h"
#include "TauAnalysis/MCEmbeddingTools/interface/embeddingAuxFunctions.h"

#include "FWCore/Utilities/interface/Exception.h"

#include "DataFormats/Common/interface/Handle.h"

#include <string>
#include <vector>

MuonCaloCleanerByDistance::MuonCaloCleanerByDistance(const edm::ParameterSet& cfg)
  : srcMuons_(cfg.getParameter<edm::InputTag>("muons")),
    srcDistanceMapMuPlus_(cfg.getParameter<edm::InputTag>("distanceMapMuPlus")),
    srcDistanceMapMuMinus_(cfg.getParameter<edm::InputTag>("distanceMapMuMinus"))
{
  edm::ParameterSet cfgEnergyDepositCorrection = cfg.getParameter<edm::ParameterSet>("energyDepositCorrection");
  typedef std::vector<std::string> vstring;
  vstring detNames = cfgEnergyDepositCorrection.getParameterNamesForType<double>();
  for ( vstring::const_iterator detName = detNames.begin();
	detName != detNames.end(); ++detName ) {
    energyDepositCorrection_[*detName] = cfgEnergyDepositCorrection.getParameter<double>(*detName);
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

  edm::Handle<reco::CandidateCollection> muons;
  evt.getByLabel(srcMuons_, muons);
  edm::Handle<detIdToFloatMap> distanceMapMuPlus;
  evt.getByLabel(srcDistanceMapMuPlus_, distanceMapMuPlus);
  edm::Handle<detIdToFloatMap> distanceMapMuMinus;
  evt.getByLabel(srcDistanceMapMuMinus_, distanceMapMuMinus);

  if(muons->size() != 2 || (*muons)[0].charge() * (*muons)[1].charge() > 0)
    throw cms::Exception("MuonCaloCleanerByDistance") << "There must be exactly two oppositely charged input muons !!";

  const reco::Candidate& muPlus = (*muons)[0].charge() > 0 ? (*muons)[0] : (*muons)[1];
  const reco::Candidate& muMinus = (*muons)[0].charge() < 0 ? (*muons)[0] : (*muons)[1];

  fillEnergyDepositMap(muPlus, *distanceMapMuPlus, *energyDepositsMuPlus);
  fillEnergyDepositMap(muMinus, *distanceMapMuMinus, *energyDepositsMuMinus);

  evt.put(energyDepositsMuPlus, "energyDepositsMuPlus");
  evt.put(energyDepositsMuMinus, "energyDepositsMuMinus");
}

void MuonCaloCleanerByDistance::fillEnergyDepositMap(const reco::Candidate& muon, const detIdToFloatMap& distanceMap, detIdToFloatMap& energyDepositMap)
{
  for ( detIdToFloatMap::const_iterator rawDetId_and_distance = distanceMap.begin();
	rawDetId_and_distance != distanceMap.end(); ++rawDetId_and_distance ) {
    DetId detId(rawDetId_and_distance->first);

    std::string key = detNaming_.getKey(detId);

    if ( energyDepositCorrection_.find(key) == energyDepositCorrection_.end() )
      throw cms::Exception("MuonCaloCleanerByDistance") 
	<< "No energy deposit correction defined for detId = " << detId.rawId() << " (key = " << key << ") !!\n";

    double distance = rawDetId_and_distance->second;
    double energyDepositCorrection_value = energyDepositCorrection_[key];

    double dedx, rho;
    switch(detId.det())
    {
    case DetId::Ecal:
      dedx = getDeDxForPbWO4(muon.p());
      rho = DENSITY_PBWO4;
      break;
    case DetId::Hcal:
      // AB: We don't have a dedx curve for the HCAL. Use the PbWO4 one as an approximation,
      // the correction factors should be determined with respect to the PbWO4 curve.
      dedx = getDeDxForPbWO4(muon.p());
      if(detId.subdetId() == HcalOuter)
      {
        rho = DENSITY_IRON; // iron coil and return yoke

        // HO uses magnet coil as additional absorber, add to flight distance:
        const double theta = muon.theta();
        distance += 31.2 / sin(theta); // 31.2cm is dr of cold mass of the magnet coil
      }
      else
      {
        rho = DENSITY_BRASS; // brass absorber
      }

      break;
    default:
      throw cms::Exception("MuonCaloCleanerByDistance") << "Unknown detector type: " << key << ", ID=" << static_cast<unsigned int>(detId);
    }

    energyDepositMap[rawDetId_and_distance->first] += distance * dedx * rho * energyDepositCorrection_value;
  }
}

#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(MuonCaloCleanerByDistance);


