#include "TauAnalysis/MCEmbeddingTools/interface/DetNaming.h"

#include <vector>
#include <string>
#include <map>
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/EcalDetId/interface/EcalSubdetector.h"
#include "DataFormats/HcalDetId/interface/HcalSubdetector.h"

#include <boost/foreach.hpp>


DetNaming::DetNaming(){

  detMap_[DetId::Hcal]="Hcal";
  detMap_[DetId::Ecal]="Ecal";

  subDetMap_[DetId::Ecal][EcalBarrel]="EcalBarrel";
  subDetMap_[DetId::Ecal][EcalEndcap]="EcalEndcap";
  subDetMap_[DetId::Ecal][EcalPreshower ]="EcalPreshower";
  subDetMap_[DetId::Ecal][EcalTriggerTower]="EcalTriggerTower";
  subDetMap_[DetId::Ecal][EcalLaserPnDiode]="EcalLaserPnDiode";

  subDetMap_[DetId::Hcal][HcalEmpty]="HcalEmpty";
  subDetMap_[DetId::Hcal][HcalBarrel]="HcalBarrel";
  subDetMap_[DetId::Hcal][HcalEndcap]="HcalEndcap";
  subDetMap_[DetId::Hcal][HcalOuter]="HcalOuter";
  subDetMap_[DetId::Hcal][HcalForward]="HcalForward";
  subDetMap_[DetId::Hcal][HcalTriggerTower]="HcalTriggerTower";
  subDetMap_[DetId::Hcal][HcalOther]="HcalOther";



}

std::string DetNaming::getKey(const DetId & det){
  return "H_"+detMap_[det.det()]+"_"+subDetMap_[det.det()][det.subdetId()];
}


std::vector<std::string> DetNaming::getAllKeys(){
  std::vector<std::string> ret;
  ret.push_back("H__");
  BOOST_FOREACH(TMyMainMap::value_type & entry, detMap_){
    BOOST_FOREACH(TMySubMap::mapped_type::value_type & subEntry, subDetMap_[entry.first]){
      std::string name = "H_"+entry.second+"_"+subEntry.second;
      ret.push_back(name);
    }

  }

  return ret;
}



