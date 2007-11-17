#include "Utilities/StorageFactory/interface/StorageFactory.h"
#include "Utilities/StorageFactory/interface/StorageMakerFactory.h"
#include "Utilities/StorageFactory/interface/StorageAccount.h"
#include "Utilities/StorageFactory/interface/StorageAccountProxy.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/PluginManager/interface/PluginManager.h"
#include "FWCore/PluginManager/interface/standard.h"
#include "FWCore/Utilities/interface/Exception.h"
#include <boost/shared_ptr.hpp>

StorageFactory StorageFactory::s_instance;

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
  if (! edmplugin::PluginManager::isAvailable())
    edmplugin::PluginManager::configure(edmplugin::standard::config());
  if (! instance)
    instance = StorageMakerFactory::get()->create (proto);
  return instance;
}

StorageMaker *
StorageFactory::getMaker (const std::string &url,
		          std::string &protocol,
		          std::string &rest)
{
  size_t p = url.find(':');
  if (p != std::string::npos)
  {
    protocol = url.substr(0,p);
    rest = url.substr(p+1);
  }
  else
  {
    protocol = "file"; 
    rest = url;
  }

  return getMaker (protocol);
}
   
Storage *
StorageFactory::open (const std::string &url, int mode, const std::string &tmpdir /* = "" */)
{ 
  std::string protocol;
  std::string rest;
  
  Storage *ret = 0;
  boost::shared_ptr<StorageAccount::Stamp> stats;
  if (StorageMaker *maker = getMaker (url, protocol, rest))
  {
    if (m_accounting) 
      stats.reset(new StorageAccount::Stamp(StorageAccount::counter (protocol, "open")));
    try
    {
      if (Storage *storage = maker->open (protocol, rest, mode, tmpdir))
        ret = m_accounting ? new StorageAccountProxy (protocol, storage) : storage;
      if (stats) stats->tick();
    }
    catch (cms::Exception &err)
    {
      throw cms::Exception("StorageFactory::open()")
	<< "Failed to open the file '" << url << "' because:\n"
	<< err;
    }
  } 
  return ret;
}

bool
StorageFactory::check (const std::string &url, IOOffset *size /* = 0 */)
{ 
  std::string protocol;
  std::string rest;

  bool ret = false;
  boost::shared_ptr<StorageAccount::Stamp> stats;
  if (StorageMaker *maker = getMaker (url, protocol, rest))
  {
    if (m_accounting) 
      stats.reset(new StorageAccount::Stamp(StorageAccount::counter (protocol, "check")));
    try
    {
      ret = maker->check (protocol, rest, size);
      if (stats) stats->tick();
    }
    catch (cms::Exception &err)
    {
      edm::LogWarning("StorageFactory::check()")
	<< "Existence or size check for the file '" << url << "' failed because:\n"
	<< err.explainSelf();
    }
  }
 
  return ret;
}
