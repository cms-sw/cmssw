//#include "SimG4Core/SensitiveDetector/interface/SensitiveDetectorFactoryByName.h"
#include "SimG4Core/SensitiveDetector/interface/SensitiveDetectorPluginFactory.h"
#include "SimG4Core/SensitiveDetector/interface/SensitiveDetector.h"
#include "SimG4Core/SensitiveDetector/interface/AttachSD.h"
#include <string>
#include <vector>

#include "FWCore/MessageLogger/interface/MessageLogger.h"

using std::vector;
using std::string;
using std::cout;
using std::endl;

// #define DEBUG

AttachSD::AttachSD() {}

AttachSD::~AttachSD() {}

std::pair< std::vector<SensitiveTkDetector*>,
	   std::vector<SensitiveCaloDetector*> > 
AttachSD::create(const DDDWorld & w, 
		 const DDCompactView & cpv,
		 const SensitiveDetectorCatalog & clg,
		 edm::ParameterSet const & p,
		 const SimTrackManager* m,
		 SimActivityRegistry& reg) const
{
  std::pair< std::vector<SensitiveTkDetector *>,
    std::vector<SensitiveCaloDetector*> > detList;
//#ifdef DEBUG
  //cout << " Initializing AttachSD " << endl;
  LogDebug("SimG4CoreSensitiveDetector") << " AttachSD: Initializing" ;
//#endif
  const std::vector<std::string>& rouNames = clg.readoutNames();
  for (std::vector<std::string>::const_iterator it = rouNames.begin();
       it != rouNames.end(); it++) {
    std::string className = clg.className(*it);
    //std::cout<<" trying to find something for "<<className<<" " <<*it<<std::endl;
    edm::LogInfo("SimG4CoreSensitiveDetector") << " AttachSD: trying to find something for " << className << " "  << *it ;
    std::auto_ptr<SensitiveDetectorMakerBase> temp(
						   SensitiveDetectorPluginFactory::get()->create(className) );
    std::auto_ptr<SensitiveTkDetector> tkDet;
    std::auto_ptr<SensitiveCaloDetector> caloDet;
    temp->make(*it,cpv,clg,p,m,reg,tkDet,caloDet);
    if(tkDet.get()){
      detList.first.push_back(tkDet.get());
      tkDet.release();
    }
    if(caloDet.get()){
      detList.second.push_back(caloDet.get());
      caloDet.release();
    }
//#ifdef DEBUG
    // cout << " AttachSD: created a " << className << " with name " << *it << endl;
    LogDebug("SimG4CoreSensitiveDetector") << " AttachSD: created a " << className << " with name " << *it ;
    //#endif
  }      
  return detList;
}

