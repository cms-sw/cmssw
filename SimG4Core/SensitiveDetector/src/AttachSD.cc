//#include "SimG4Core/SensitiveDetector/interface/SensitiveDetectorFactoryByName.h"
#include "SimG4Core/SensitiveDetector/interface/SensitiveDetectorPluginFactory.h"
#include "SimG4Core/SensitiveDetector/interface/SensitiveDetector.h"
#include "SimG4Core/SensitiveDetector/interface/AttachSD.h"
#include "SimG4Core/Geometry/interface/SDCatalog.h"
#include <string>
#include <vector>

using std::vector;
using std::string;
using std::cout;
using std::endl;

#define DEBUG

AttachSD::AttachSD() {}

AttachSD::~AttachSD() {}

std::pair< std::vector<SensitiveTkDetector*>,
	   std::vector<SensitiveCaloDetector*> > 
AttachSD::create(const DDDWorld & w, 
		 const DDCompactView & cpv,
		 edm::ParameterSet const & p,
		 SimActivityRegistry& reg) const
{
  std::pair< std::vector<SensitiveTkDetector *>,
    std::vector<SensitiveCaloDetector*> > detList;
#ifdef DEBUG
  cout << " Initializing AttachSD " << endl;
#endif
  vector<string> rouNames = SensitiveDetectorCatalog::instance()->readoutNames();
  for (vector<string>::iterator it = rouNames.begin();  it != rouNames.end(); it++)
    {

      string className = SensitiveDetectorCatalog::instance()->className(*it);
      std::cout<<" trying to find something for "<<className<<" " <<*it<<std::endl;
      std::auto_ptr<SensitiveDetectorMakerBase> temp(
						     SensitiveDetectorPluginFactory::get()->create(className) );
      std::auto_ptr<SensitiveTkDetector> tkDet;
      std::auto_ptr<SensitiveCaloDetector> caloDet;
      temp->make(*it,cpv,p,reg,tkDet,caloDet);
      if(tkDet.get()){
	detList.first.push_back(tkDet.get());
	tkDet.release();
      }
      if(caloDet.get()){
	detList.second.push_back(caloDet.get());
	caloDet.release();
      }
#ifdef DEBUG
      cout << " AttachSD: created a " << className << " with name " << *it << endl;
#endif
    }      
  return detList;
}

