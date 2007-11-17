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
  static StorageFactory *get (void);
  ~StorageFactory (void);

  // implicit copy constructor
  // implicit assignment operator

  bool		enableAccounting (bool enabled);
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
  bool		m_accounting;
  static StorageFactory s_instance;
};

#endif // STORAGE_FACTORY_STORAGE_FACTORY_H
