#include "SimG4Core/SensitiveDetector/interface/SensitiveDetectorPluginFactory.h"
#include "SimG4Core/SensitiveDetector/interface/SensitiveDetector.h"
#include "SimG4Core/SensitiveDetector/interface/SensitiveTkDetector.h"
#include "SimG4Core/SensitiveDetector/interface/SensitiveCaloDetector.h"
#include "SimG4Core/SensitiveDetector/interface/AttachSD.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <string>
#include <sstream>

AttachSD::AttachSD() {}

AttachSD::~AttachSD() {}

std::pair< std::vector<SensitiveTkDetector*>,
           std::vector<SensitiveCaloDetector*> > 
AttachSD::create(const DDCompactView & cpv,
                 const SensitiveDetectorCatalog & clg,
                 edm::ParameterSet const & p,
                 const SimTrackManager* man,
                 SimActivityRegistry& reg) const
{
  std::pair< std::vector<SensitiveTkDetector *>,std::vector<SensitiveCaloDetector*> > detList;
  const std::vector<std::string>& rouNames = clg.readoutNames();
  edm::LogVerbatim("SimG4CoreSensitiveDetector") 
    << " AttachSD: Initialising " << rouNames.size() << " SDs";
  std::unique_ptr<SensitiveDetectorMakerBase> temp; 
  for (auto & rname : rouNames) {
    std::string className = clg.className(rname);
    temp.reset(SensitiveDetectorPluginFactory::get()->create(className));

    SensitiveDetector* sd = temp.get()->make(rname,cpv,clg,p,man,reg);
    
    std::stringstream ss;
    ss << " AttachSD: created a " << className << " with name " << rname;
 
    if(sd->isCaloSD()) {
      SensitiveCaloDetector* caloDet = (SensitiveCaloDetector*)(sd);
      detList.second.push_back(caloDet);
      ss << " + calo SD"; 
    } else {
      SensitiveTkDetector* tkDet = (SensitiveTkDetector*)(sd);
      detList.first.push_back(tkDet);
      ss << " + tracking SD"; 
    }
    edm::LogVerbatim("SimG4CoreSensitiveDetector") << ss.str();
  }      
  return detList;
}
