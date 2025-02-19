#include "RecoVertex/ConfigurableVertexReco/interface/VertexRecoManager.h"

using namespace std;

void VertexRecoManager::registerReconstructor (
    const string & name, AbstractConfReconstructor * o, const string & d )
{
  theAbstractConfReconstructors[name]=o;
  theDescription[name]=d;
}

VertexRecoManager::~VertexRecoManager()
{
  // why should we delete?
  /*
  for ( map < string, AbstractConfReconstructor * >::iterator i=theAbstractConfReconstructors.begin(); 
        i!=theAbstractConfReconstructors.end() ; ++i )
  {
    delete i->second;
  }*/
}

std::string VertexRecoManager::describe ( const std::string & d )
{
  return theDescription[d];
}

VertexRecoManager * VertexRecoManager::clone() const
{
  return new VertexRecoManager ( * this );
}

VertexRecoManager::VertexRecoManager ( const VertexRecoManager & o ) 
{
  std::cout << "[VertexRecoManager] copy constructor! Error!" << std::endl;
  exit(0);
  /*
  for ( map < string, AbstractConfReconstructor * >::const_iterator i=o.theAbstractConfReconstructors.begin(); 
        i!=o.theAbstractConfReconstructors.end() ; ++i )
  {
    theAbstractConfReconstructors[ i->first ] = i->second->clone();
  }
  
  theIsEnabled=o.theIsEnabled;
  */
}

VertexRecoManager & VertexRecoManager::Instance()
{
  static VertexRecoManager singleton;
  return singleton;
}

AbstractConfReconstructor * VertexRecoManager::get ( const string & s )
{
  return theAbstractConfReconstructors[s];
}

map < string, AbstractConfReconstructor * > VertexRecoManager::get()
{
  return theAbstractConfReconstructors;
}

VertexRecoManager::VertexRecoManager()
{}
