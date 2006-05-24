#include "SimG4Core/Geometry/interface/SDCatalog.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

//#define DEBUG

#include <iostream>

using namespace std;

void SDCatalog::insert(string& cN, string& rN, string& lvN)
{
    theClassNameMap[cN].push_back(rN);
    theROUNameMap[rN].push_back(lvN);
//#ifdef DEBUG
    //cout <<" I have inserted ("<<cN<<","<<rN<<","<<lvN<<")"<<endl;
    //cout <<" I have "<<readoutNames().size()<<" ROUs "<<readoutNames().front()<<endl;
    //cout <<" I have "<<classNames().size()<<" classes "<<classNames().front()<<endl;
    LogDebug("SimG4CoreGeometry") << " I have inserted ("<<cN<<","<<rN<<","<<lvN<<")" ;
    LogDebug("SimG4CoreGeometry") << " I have "<<readoutNames().size()<<" ROUs "<<readoutNames().front() ;
    LogDebug("SimG4CoreGeometry") << " I have "<<classNames().size()<<" classes "<<classNames().front() ;
//#endif
}

vector<string> SDCatalog::readoutNames()
{
    vector<string> temp;
    for (MapType::const_iterator it = theROUNameMap.begin();  it != theROUNameMap.end(); it++)
	temp.push_back(it->first);
    return temp;
}
vector<string> SDCatalog::readoutNames(string & className)
{ return theClassNameMap[className]; }

vector<string> SDCatalog::logicalNames(string & readoutName)
{ return theROUNameMap[readoutName]; }

vector<string> SDCatalog::logicalNamesFromClassName(string & className)
{
    vector<string> temp;
    vector<string> rous = theClassNameMap[className];
    for (vector<string>::const_iterator it = rous.begin(); it!= rous.end(); it++)
	temp.push_back(*it);
    return temp;
}

string SDCatalog::className(string & readoutName)
{
    for(MapType::const_iterator it = theClassNameMap.begin();  
	it != theClassNameMap.end(); it++)
    {
	vector<string> temp = (*it).second;
	for (vector<string>::const_iterator it2 = temp.begin(); it2!=temp.end(); it2++)
	{
	    if (*it2 == readoutName ) return (*it).first;
	}
    } 
    return "NotFound";
}

vector<string> SDCatalog::classNames()
{
    vector<string> temp;
    for (MapType::const_iterator it = theClassNameMap.begin();  
	 it != theClassNameMap.end(); it++)
	temp.push_back(it->first);
    return temp;
}
