#include "TauAnalysis/MCEmbeddingTools/plugins/MuonCaloCleanerByDistance.h"
#include "TauAnalysis/MCEmbeddingTools/interface/embeddingAuxFunctions.h"

#include "FWCore/Utilities/interface/Exception.h"

#include "DataFormats/Common/interface/Handle.h"

#include <string>
#include <vector>

MuonCaloCleanerByDistance::MuonCaloCleanerByDistance(const edm::ParameterSet& cfg)
  : moduleLabel_(cfg.getParameter<std::string>("@module_label")), 
    srcSelectedMuons_(cfg.getParameter<edm::InputTag>("muons")),
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

  verbosity_ = ( cfg.exists("verbosity") ) ?
    cfg.getParameter<int>("verbosity") : 0;

  // maps of detId to expected energy deposits of muon
  produces<detIdToFloatMap>("energyDepositsMuPlus");
  produces<double>("totalDistanceMuPlus");
  produces<double>("totalEnergyDepositMuPlus");
  produces<detIdToFloatMap>("energyDepositsMuMinus");
  produces<double>("totalDistanceMuMinus");
  produces<double>("totalEnergyDepositMuMinus");
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

  std::vector<reco::CandidateBaseRef> selMuons = getSelMuons(evt, srcSelectedMuons_);
  const reco::CandidateBaseRef muPlus  = getTheMuPlus(selMuons);
  const reco::CandidateBaseRef muMinus = getTheMuMinus(selMuons);

  std::auto_ptr<double> totalDistanceMuPlus(new double(0.));
  std::auto_ptr<double> totalEnergyDepositMuPlus(new double(0.));
  if ( muPlus.isNonnull() ) fillEnergyDepositMap(*muPlus, *distanceMapMuPlus, *energyDepositsMuPlus, *totalDistanceMuPlus, *totalEnergyDepositMuPlus);
  std::auto_ptr<double> totalDistanceMuMinus(new double(0.));
  std::auto_ptr<double> totalEnergyDepositMuMinus(new double(0.));
  if ( muMinus.isNonnull() ) fillEnergyDepositMap(*muMinus, *distanceMapMuMinus, *energyDepositsMuMinus, *totalDistanceMuMinus, *totalEnergyDepositMuMinus);

  if ( verbosity_ ) {
    std::cout << "<MuonCaloCleanerByDistance::produce (" << moduleLabel_ << ")>:" << std::endl;
    std::cout << " mu+: distance = " << (*totalDistanceMuPlus) << ", expected(EnergyDeposits) = " << (*totalEnergyDepositMuPlus) << std::endl;
    std::cout << " mu-: distance = " << (*totalDistanceMuMinus) << ", expected(EnergyDeposits) = " << (*totalEnergyDepositMuMinus) << std::endl;
  }

  evt.put(energyDepositsMuPlus, "energyDepositsMuPlus");
  evt.put(totalDistanceMuPlus, "totalDistanceMuPlus");
  evt.put(totalEnergyDepositMuPlus, "totalEnergyDepositMuPlus");
  evt.put(energyDepositsMuMinus, "energyDepositsMuMinus");
  evt.put(totalDistanceMuMinus, "totalDistanceMuMinus");
  evt.put(totalEnergyDepositMuMinus, "totalEnergyDepositMuMinus");
}

void MuonCaloCleanerByDistance::fillEnergyDepositMap(const reco::Candidate& muon, const detIdToFloatMap& distanceMap, detIdToFloatMap& energyDepositMap,
						     double& totalDistance, double& totalEnergyDeposit)
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

    double dEdx = 0.;
    double rho = 0.;
    switch ( detId.det() ) {
    case DetId::Ecal:
      dEdx = getDeDxForPbWO4(muon.p());
      rho = DENSITY_PBWO4;
      break;
    case DetId::Hcal:
      // AB: We don't have a dedx curve for the HCAL. Use the PbWO4 one as an approximation,
      // the correction factors should be determined with respect to the PbWO4 curve.
      dEdx = getDeDxForPbWO4(muon.p());
      if ( detId.subdetId() == HcalOuter ) {
        rho = DENSITY_IRON; // iron coil and return yoke
        // HO uses magnet coil as additional absorber, add to flight distance:
        const double theta = muon.theta();
        distance += 31.2 / sin(theta); // 31.2cm is dr of cold mass of the magnet coil
      } else {
        rho = DENSITY_BRASS; // brass absorber
      }
      break;
    default:
      throw cms::Exception("MuonCaloCleanerByDistance") 
	<< "Unknown detector type: " << key << ", detId = " << static_cast<unsigned int>(detId);
    }

    double energyDeposit = distance*dEdx*rho*energyDepositCorrection_value;
    energyDepositMap[rawDetId_and_distance->first] += energyDeposit;
    totalDistance += distance;
    totalEnergyDeposit += energyDeposit;
  }
}

#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(MuonCaloCleanerByDistance);


