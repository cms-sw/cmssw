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
    m_tempfree (4.), // GB
    m_temppath (".:$TMPDIR")
{
  setTempDir(m_temppath, m_tempfree);
}

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
StorageFactory::setTempDir(const std::string &s, double minFreeSpace)
{
  edm::LogInfo("StorageFactory")
    << "Considering path '" << s
    << "', min free space " << minFreeSpace
    << "GB for temp dir";

  size_t begin = 0;
  std::vector<std::string> dirs;
  dirs.reserve(std::count(s.begin(), s.end(), ':') + 1);

  while (true)
  {
    size_t end = s.find(':', begin);
    if (end == std::string::npos)
    {
      dirs.push_back(s.substr(begin, end));
      break;
    }
    else
    {
      dirs.push_back(s.substr(begin, end - begin));
      begin = end+1;
    }
  }

  m_temppath = s;
  m_tempfree = minFreeSpace;
  m_tempdir = m_lfs.findCachePath(dirs, minFreeSpace);

  edm::LogInfo("StorageFactory") << "Using '" << m_tempdir << "' for temp dir";
}

std::string
StorageFactory::tempDir(void) const
{ return m_tempdir; }

std::string
StorageFactory::tempPath(void) const
{ return m_temppath; }

double
StorageFactory::tempMinFree(void) const
{ return m_tempfree; }

StorageMaker *
StorageFactory::getMaker (const std::string &proto)
{
  StorageMaker *&instance = m_makers [proto];
  if (! edmplugin::PluginManager::isAvailable())
    edmplugin::PluginManager::configure(edmplugin::standard::config());
  if (! instance)
    instance = StorageMakerFactory::get()->tryToCreate(proto);
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
StorageFactory::open (const std::string &url, int mode /* = IOFlags::OpenRead */)
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
      if (Storage *storage = maker->open (protocol, rest, mode))
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

Storage *
StorageFactory::wrapNonLocalFile (Storage *s,
				  const std::string &proto,
				  const std::string &path,
				  int mode)
{
  StorageFactory::CacheHint hint = cacheHint();
  if ((hint == StorageFactory::CACHE_HINT_LAZY_DOWNLOAD
       || hint == StorageFactory::CACHE_HINT_AUTO_DETECT)
      && ! (mode & IOFlags::OpenWrite)
      && ! m_tempdir.empty()
      && (path.empty() || ! m_lfs.isLocalPath(path)))
  {
    if (accounting())
      s = new StorageAccountProxy(proto, s);
    s = new LocalCacheFile(s, m_tempdir);
  }

  return s;
}
