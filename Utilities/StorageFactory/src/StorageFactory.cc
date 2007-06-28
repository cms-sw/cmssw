//<<<<<< INCLUDES                                                       >>>>>>

#include "Utilities/StorageFactory/interface/StorageFactory.h"
#include "Utilities/StorageFactory/interface/StorageMakerFactory.h"
#include "Utilities/StorageFactory/interface/StorageAccount.h"
#include "Utilities/StorageFactory/interface/StorageAccountProxy.h"
#include "FWCore/PluginManager/interface/PluginManager.h"
#include "FWCore/PluginManager/interface/standard.h"
//#include "SealBase/StringOps.h"
#include <boost/shared_ptr.hpp>


//<<<<<< PRIVATE DEFINES                                                >>>>>>
//<<<<<< PRIVATE CONSTANTS                                              >>>>>>
//<<<<<< PRIVATE TYPES                                                  >>>>>>
//<<<<<< PRIVATE VARIABLE DEFINITIONS                                   >>>>>>
//<<<<<< PUBLIC VARIABLE DEFINITIONS                                    >>>>>>
//<<<<<< CLASS STRUCTURE INITIALIZATION                                 >>>>>>

StorageFactory StorageFactory::s_instance;

//<<<<<< PRIVATE FUNCTION DEFINITIONS                                   >>>>>>
//<<<<<< PUBLIC FUNCTION DEFINITIONS                                    >>>>>>
//<<<<<< MEMBER FUNCTION DEFINITIONS                                    >>>>>>

StorageFactory::StorageFactory (void)
    : m_accounting (false)
{}

StorageFactory::~StorageFactory (void)
{
    for (MakerTable::iterator i = m_makers.begin (); i != m_makers.end (); ++i)
	 delete i->second;
}

StorageFactory *
StorageFactory::get (void)
{
    return &s_instance;
}

bool
StorageFactory::enableAccounting (bool enabled)
{
    bool old = m_accounting;
    m_accounting = enabled;
    return old;
}

StorageMaker *
StorageFactory::getMaker (const std::string &proto)
{
    StorageMaker *&instance = m_makers [proto];
    // Force plug-in manager to initialise so clients don't need to know about it
    if( not edmplugin::PluginManager::isAvailable()) {
       edmplugin::PluginManager::configure(edmplugin::standard::config());
    }
    if (! instance) instance = edm::storage::StorageMakerFactory::get()->create (proto);
    return instance;
}

StorageMaker *
StorageFactory::getMaker (const std::string &url,
		          std::string &protocol,
		          std::string &rest)
{
    size_t p = url.find(':');
    if (p!=std::string::npos) {
      protocol = url.substr(0,p);
      rest = url.substr(p+1);
    } else {
      protocol = "file"; 
      rest = url;
    }

    return getMaker (protocol);
}
   
seal::Storage *
StorageFactory::open (const std::string &url, int mode, const std::string &tmpdir /* = "" */)
{ 
  std::string protocol;
  std::string rest;
  
  boost::shared_ptr<StorageAccount::Stamp> stats;
  seal::Storage * ret = 0;
  if (StorageMaker *maker = getMaker (url, protocol, rest)) {
    if (m_accounting) 
      stats.reset(new StorageAccount::Stamp(StorageAccount::counter (protocol, "open")));
    if (seal::Storage *storage = maker->open (protocol, rest, mode, tmpdir))
	ret = m_accounting ? new StorageAccountProxy (protocol, storage) : storage;
    if (stats) stats->tick();
  } 
    return ret;
}

bool
StorageFactory::check (const std::string &url, seal::IOOffset *size /* = 0 */)
{ 
    std::string protocol;
    std::string rest;

    boost::shared_ptr<StorageAccount::Stamp> stats;
    bool ret = false;
    if (StorageMaker *maker = getMaker (url, protocol, rest)) {
      if (m_accounting) 
	stats.reset(new StorageAccount::Stamp(StorageAccount::counter (protocol, "check")));
      ret = maker->check (protocol, rest, size);
      if (stats) stats->tick();
    }
 
    return ret;
}
