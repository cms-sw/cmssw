#include "Utilities/StorageFactory/interface/StorageFactory.h"
#include "Utilities/StorageFactory/interface/StorageMakerFactory.h"
#include "Utilities/StorageFactory/interface/StorageAccount.h"
#include "Utilities/StorageFactory/interface/StorageAccountProxy.h"
#include "Utilities/StorageFactory/interface/LocalCacheFile.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/PluginManager/interface/PluginManager.h"
#include "FWCore/PluginManager/interface/standard.h"
#include "FWCore/Utilities/interface/Exception.h"
#include <boost/shared_ptr.hpp>

StorageFactory StorageFactory::s_instance;

StorageFactory::StorageFactory (void)
  : m_cacheHint(CACHE_HINT_AUTO_DETECT),
    m_readHint(READ_HINT_AUTO),
    m_accounting (false),
    m_tempdir(".")
{}

StorageFactory::~StorageFactory (void)
{
  for (MakerTable::iterator i = m_makers.begin (); i != m_makers.end (); ++i)
    delete i->second;
}

StorageFactory *
StorageFactory::get (void)
{ return &s_instance; }

bool
StorageFactory::enableAccounting (bool enabled)
{
  bool old = m_accounting;
  m_accounting = enabled;
  return old;
}

bool
StorageFactory::accounting(void) const
{ return m_accounting; }

void
StorageFactory::setCacheHint(CacheHint value)
{ m_cacheHint = value; }

StorageFactory::CacheHint
StorageFactory::cacheHint(void) const
{ return m_cacheHint; }

void
StorageFactory::setReadHint(ReadHint value)
{ m_readHint = value; }

StorageFactory::ReadHint
StorageFactory::readHint(void) const
{ return m_readHint; }

void
StorageFactory::setTempDir(const std::string &s)
{ m_tempdir = s; }

std::string
StorageFactory::tempDir(void) const
{ return m_tempdir; }

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
  const std::string &tmp = (tmpdir.empty() ? m_tempdir : tmpdir);
  
  Storage *ret = 0;
  boost::shared_ptr<StorageAccount::Stamp> stats;
  if (StorageMaker *maker = getMaker (url, protocol, rest))
  {
    if (m_accounting) 
      stats.reset(new StorageAccount::Stamp(StorageAccount::counter (protocol, "open")));
    try
    {
      if (Storage *storage = maker->open (protocol, rest, mode, tmp))
      {
	if (dynamic_cast<LocalCacheFile *>(storage))
	  protocol = "local-cache";

	if (m_accounting)
	  ret = new StorageAccountProxy(protocol, storage);
	else
	  ret = storage;

        if (stats)
	  stats->tick();
      }
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

void
StorageFactory::stagein (const std::string &url)
{ 
  std::string protocol;
  std::string rest;

  boost::shared_ptr<StorageAccount::Stamp> stats;
  if (StorageMaker *maker = getMaker (url, protocol, rest))
  {
    if (m_accounting) 
      stats.reset(new StorageAccount::Stamp(StorageAccount::counter (protocol, "stagein")));
    try
    {
      maker->stagein (protocol, rest);
      if (stats) stats->tick();
    }
    catch (cms::Exception &err)
    {
      edm::LogWarning("StorageFactory::stagein()")
	<< "Failed to stage in file '" << url << "' because:\n"
	<< err.explainSelf();
    }
  }
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
