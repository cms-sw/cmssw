#include "SimG4Core/SensitiveDetector/interface/SensitiveDetectorPluginFactory.h"
#include "SimG4Core/SensitiveDetector/interface/SensitiveDetector.h"
#include "SimG4Core/SensitiveDetector/interface/AttachSD.h"
#include <string>
#include <vector>

#include "FWCore/MessageLogger/interface/MessageLogger.h"

AttachSD::AttachSD() {}

AttachSD::~AttachSD() {}

std::pair< std::vector<SensitiveTkDetector*>,
	   std::vector<SensitiveCaloDetector*> > 
AttachSD::create(const DDDWorld & w, 
		 const DDCompactView & cpv,
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
    std::auto_ptr<SensitiveTkDetector> tkDet;
    std::auto_ptr<SensitiveCaloDetector> caloDet;
    temp.get()->make(rname,cpv,clg,p,man,reg,tkDet,caloDet);
    edm::LogVerbatim("SimG4CoreSensitiveDetector") 
      << " AttachSD: created a " << className << " with name " << rname 
      << " TkDet <" << tkDet.get() << "> caloDet<" << caloDet.get() << ">"
      << " temp= <" << temp.get() << ">";
    if(tkDet.get()){
      detList.first.push_back(tkDet.get());
      tkDet.release();
    }
    if(caloDet.get()){
      detList.second.push_back(caloDet.get());
      caloDet.release();
    }
  }      
  return detList;
}

