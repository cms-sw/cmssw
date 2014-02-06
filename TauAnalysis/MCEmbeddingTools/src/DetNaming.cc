#include "TauAnalysis/MCEmbeddingTools/interface/DetNaming.h"

#include "FWCore/Utilities/interface/Exception.h"

#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/EcalDetId/interface/EcalSubdetector.h"
#include "DataFormats/HcalDetId/interface/HcalSubdetector.h"

#include <boost/foreach.hpp>

#include <TString.h>

#include <vector>
#include <string>
#include <map>

DetNaming::DetNaming()
{
  detMap_[DetId::Ecal]                      = "Ecal";
  subDetMap_[DetId::Ecal][EcalBarrel]       = "EcalBarrel";
  subDetMap_[DetId::Ecal][EcalEndcap]       = "EcalEndcap";
  subDetMap_[DetId::Ecal][EcalPreshower]    = "EcalPreshower";
  subDetMap_[DetId::Ecal][EcalTriggerTower] = "EcalTriggerTower";
  subDetMap_[DetId::Ecal][EcalLaserPnDiode] = "EcalLaserPnDiode";

  detMap_[DetId::Hcal]                      = "Hcal";
  subDetMap_[DetId::Hcal][HcalEmpty]        = "HcalEmpty";
  subDetMap_[DetId::Hcal][HcalBarrel]       = "HcalBarrel";
  subDetMap_[DetId::Hcal][HcalEndcap]       = "HcalEndcap";
  subDetMap_[DetId::Hcal][HcalOuter]        = "HcalOuter";
  subDetMap_[DetId::Hcal][HcalForward]      = "HcalForward";
  subDetMap_[DetId::Hcal][HcalTriggerTower] = "HcalTriggerTower";
  subDetMap_[DetId::Hcal][HcalOther]        = "HcalOther";
}

std::string DetNaming::getKey(const DetId& detId)
{
  if ( detMap_.find(detId.det())                      != detMap_.end()               && 
       subDetMap_.find(detId.det())                   != subDetMap_.end()            && 
       subDetMap_[detId.det()].find(detId.subdetId()) != subDetMap_[detId.det()].end() ) {
    return Form("H_%s_%s", detMap_[detId.det()].data(), subDetMap_[detId.det()][detId.subdetId()].data());
  } else {
    throw cms::Exception("DetNaming") 
      << "Invalid detId = " << detId.rawId() << " !!\n";
    return std::string();
  }
}

std::vector<std::string> DetNaming::getAllKeys()
{
  std::vector<std::string> keys;
  keys.push_back("H__");
  BOOST_FOREACH(TMyMainMap::value_type& entry, detMap_) {
    BOOST_FOREACH(TMySubMap::mapped_type::value_type& subEntry, subDetMap_[entry.first]) {
      std::string name = Form("H_%s_%s", entry.second.data(), subEntry.second.data());
      keys.push_back(name);
    }
  }
  
  return keys;
}



