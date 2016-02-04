#include "RecoVertex/ConfigurableVertexReco/interface/VertexFitterManager.h"
#include "RecoVertex/ConfigurableVertexReco/interface/VertexRecoManager.h"
#include "RecoVertex/ConfigurableVertexReco/interface/ReconstructorFromFitter.h"

using namespace std;

void VertexFitterManager::registerFitter (
    const string & name, AbstractConfFitter * o, const string & d )
{
  theAbstractConfFitters[name]=o;
  theDescription[name]=d;

  // every fitter registers as a reconstructor, also
  VertexRecoManager::Instance().registerReconstructor ( name, new ReconstructorFromFitter ( *o ), d);
}

VertexFitterManager::~VertexFitterManager()
{
  /*
   * Let the VertexRecoManager delete them (they all register there, as well!)
  for ( map < string, AbstractConfFitter * >::iterator i=theAbstractConfFitters.begin(); 
        i!=theAbstractConfFitters.end() ; ++i )
  {
    delete i->second;
  }*/
}

std::string VertexFitterManager::describe ( const std::string & d )
{
  return theDescription[d];
}

VertexFitterManager * VertexFitterManager::clone() const
{
  return new VertexFitterManager ( * this );
}

VertexFitterManager::VertexFitterManager ( const VertexFitterManager & o ) 
{
  std::cout << "[VertexFitterManager] copy constructor! Error!" << std::endl;
  exit(0);
  /*
  for ( map < string, AbstractConfFitter * >::const_iterator i=o.theAbstractConfFitters.begin(); 
        i!=o.theAbstractConfFitters.end() ; ++i )
  {
    theAbstractConfFitters[ i->first ] = i->second->clone();
  }
  
  theIsEnabled=o.theIsEnabled;
  */
}

VertexFitterManager & VertexFitterManager::Instance()
{
  static VertexFitterManager singleton;
  return singleton;
}

AbstractConfFitter * VertexFitterManager::get ( const string & s )
{
  return theAbstractConfFitters[s];
}

map < string, AbstractConfFitter * > VertexFitterManager::get()
{
  return theAbstractConfFitters;
}

VertexFitterManager::VertexFitterManager()
{}
