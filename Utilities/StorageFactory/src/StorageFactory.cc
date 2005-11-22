//<<<<<< INCLUDES                                                       >>>>>>

#include "Utilities/StorageFactory/interface/StorageFactory.h"
#include "Utilities/StorageFactory/interface/StorageAccount.h"
#include "Utilities/StorageFactory/interface/StorageAccountProxy.h"
#include "Utilities/StorageFactory/interface/StorageMaker.h"
#include "PluginManager/PluginManager.h"
#include "SealBase/StringOps.h"

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
    : seal::PluginFactory<StorageMaker *(void)> ("CMS Storage Maker"),
      m_accounting (false)
{}

StorageFactory::~StorageFactory (void)
{
    for (MakerTable::iterator i = m_makers.begin (); i != m_makers.end (); ++i)
	 delete i->second;
}

StorageFactory *
StorageFactory::get (void)
{
    // Force plug-in manager to initialise so clients don't need to know about it
    seal::PluginManager::get ()->initialise ();
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
    if (! instance) instance = create (proto);
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

    if (StorageMaker *maker = getMaker (url, protocol, rest))
	if (seal::Storage *storage = maker->open (protocol, rest, mode, tmpdir))
	    return m_accounting ? new StorageAccountProxy (protocol, storage) : storage;

    return 0;
}

bool
StorageFactory::check (const std::string &url, seal::IOOffset *size /* = 0 */)
{ 
    std::string protocol;
    std::string rest;

    if (StorageMaker *maker = getMaker (url, protocol, rest))
	return maker->check (protocol, rest, size);
 
    return false;
}
