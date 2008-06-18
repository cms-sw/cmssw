#ifndef STORAGE_FACTORY_STORAGE_FACTORY_H
# define STORAGE_FACTORY_STORAGE_FACTORY_H

# include "Utilities/StorageFactory/interface/StorageMaker.h"
# include "Utilities/StorageFactory/interface/IOTypes.h"
# include "Utilities/StorageFactory/interface/IOFlags.h"
# include <string>
# include <map>

class Storage;
class StorageFactory 
{
public:
  enum CacheHint
  {
    CACHE_HINT_APPLICATION,
    CACHE_HINT_STORAGE,
    CACHE_HINT_LAZY_DOWNLOAD,
    CACHE_HINT_AUTO_DETECT
  };

  enum ReadHint
  {
    READ_HINT_UNBUFFERED,
    READ_HINT_READAHEAD,
    READ_HINT_AUTO
  };

  static StorageFactory *get (void);
  ~StorageFactory (void);

  // implicit copy constructor
  // implicit assignment operator

  void		setCacheHint(CacheHint value);
  CacheHint	cacheHint(void) const;

  void		setReadHint(ReadHint value);
  ReadHint	readHint(void) const;

  bool		enableAccounting (bool enabled);
  bool		accounting (void) const;

  void		stagein (const std::string &url);
  Storage *	open (const std::string &url,
	    	      int mode = IOFlags::OpenRead,
	    	      const std::string &tmpdir = "");
  bool		check (const std::string &url,
	    	       IOOffset *size = 0);

protected:
  typedef std::map<std::string, StorageMaker *> MakerTable;

  StorageFactory (void);
  StorageMaker *getMaker (const std::string &proto);
  StorageMaker *getMaker (const std::string &url,
			  std::string &protocol,
			  std::string &rest);
  
  MakerTable	m_makers;
  CacheHint	m_cacheHint;
  ReadHint	m_readHint;
  bool		m_accounting;
  static StorageFactory s_instance;
};

#endif // STORAGE_FACTORY_STORAGE_FACTORY_H
