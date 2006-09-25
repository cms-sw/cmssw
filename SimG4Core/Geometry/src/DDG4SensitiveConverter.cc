#include "SimG4Core/Geometry/interface/DDG4SensitiveConverter.h"
#include "SimG4Core/Geometry/interface/SDCatalog.h"

#include "G4LogicalVolume.hh"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

using std::string;
using std::vector;
using std::cout;
using std::endl;

DDG4SensitiveConverter::DDG4SensitiveConverter() {}

DDG4SensitiveConverter::~DDG4SensitiveConverter() {}

void DDG4SensitiveConverter::upDate(const DDG4DispContainer & ddg4s) 
{
    LogDebug("SimG4CoreGeometry") <<" DDG4SensitiveConverter::upDate() starts" ;

    for (unsigned int i=0; i<ddg4s.size(); i++)
    {
	DDG4Dispatchable * ddg4 = ddg4s[i];
	const DDLogicalPart * part   = (ddg4->getDDLogicalPart());
	G4LogicalVolume *     result = (ddg4->getG4LogicalVolume());
  
	string sClassName = getString("SensitiveDetector",part);
	string sROUName   = getString("ReadOutName",part);
	string fff        = result->GetName();
	if (sClassName != "NotFound") 
	{
	    LogDebug("SimG4CoreGeometry") << " DDG4SensitiveConverter: Sensitive " << fff
					  << " Class Name " << sClassName << " ROU Name " << sROUName ;	    
	    fff = result->GetName();
	    SensitiveDetectorCatalog::instance()->insert(sClassName,sROUName,fff);
	}
    }
}

string DDG4SensitiveConverter::getString(const string & s, const DDLogicalPart * part) 
{
    vector<string> temp;
    DDValue val(s);
    vector<const DDsvalues_type *> result = part->specifics();
    vector<const DDsvalues_type *>::iterator it = result.begin();
    bool foundIt = false;
    for (; it != result.end(); ++it) 
    {
	foundIt = DDfetch(*it,val);
	if (foundIt) break;
    }    
    if (foundIt) 
    { 
	temp = val.strings(); 
	if (temp.size() != 1) 
	{
	    // cout << " ERROR: I need 1 " << s << " tags" << endl;
	    edm::LogError("SimG4CoreGeometry") << " DDG4SensitiveConverter - ERROR: I need 1 " << s << " tags" ;
	    abort();
	}
	return temp[0]; 
    }
    return "NotFound";
}

