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

std::vector<SensitiveDetector*> AttachSD::create(const DDDWorld & w) const
{
    std::vector<SensitiveDetector *> detList;
#ifdef DEBUG
    cout << " Initializing AttachSD " << endl;
#endif
    // Clear the sensitive detector list
    detList.clear();
    vector<string> rouNames = SensitiveDetectorCatalog::instance()->readoutNames();
    for (vector<string>::iterator it = rouNames.begin();  it != rouNames.end(); it++)
    {

	string className = SensitiveDetectorCatalog::instance()->className(*it);
	std::cout<<" trying to find something for "<<className<<" " <<*it<<std::endl;
	SensitiveDetector * temp =
	  //(SensitiveDetectorFactoryByName::getBuilder(className))->constructComponent(*it);
	  SensitiveDetectorPluginFactory::get()->create(className,*it);
#ifdef DEBUG
	cout << " AttachSD: created a " << className << " with name " << *it << endl;
#endif
	detList.push_back(temp);
    }      
    return detList;
}

