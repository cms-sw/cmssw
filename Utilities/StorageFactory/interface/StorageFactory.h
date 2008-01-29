#ifndef STORAGE_FACTORY_STORAGE_FACTORY_H
# define STORAGE_FACTORY_STORAGE_FACTORY_H

//<<<<<< INCLUDES                                                       >>>>>>

# include "Utilities/StorageFactory/interface/StorageMaker.h"
# include "SealBase/sysapi/IOTypes.h"
# include "SealBase/IOFlags.h"
# include <string>
# include <map>

//<<<<<< PUBLIC DEFINES                                                 >>>>>>
//<<<<<< PUBLIC CONSTANTS                                               >>>>>>
//<<<<<< PUBLIC TYPES                                                   >>>>>>

namespace seal { class Storage; }

//<<<<<< PUBLIC VARIABLES                                               >>>>>>
//<<<<<< PUBLIC FUNCTIONS                                               >>>>>>
//<<<<<< CLASS DECLARATIONS                                             >>>>>>

class StorageFactory 
{
public:
    static StorageFactory *get (void);
    ~StorageFactory (void);

    // implicit copy constructor
    // implicit assignment operator

    bool		enableAccounting (bool enabled);
    seal::Storage *	open (const std::string &url,
		    	      int mode = seal::IOFlags::OpenRead,
		    	      const std::string &tmpdir = "");
    bool		check (const std::string &url,
		    	       seal::IOOffset *size = 0);

protected:
    typedef std::map<std::string, StorageMaker *> MakerTable;

    StorageFactory (void);
    StorageMaker *	getMaker (const std::string &proto);
    StorageMaker *	getMaker (const std::string &url,
		    		  std::string &protocol,
				  std::string &rest);
    
    MakerTable		m_makers;
    bool		m_accounting;
    static StorageFactory s_instance;
};

//<<<<<< INLINE PUBLIC FUNCTIONS                                        >>>>>>
//<<<<<< INLINE MEMBER FUNCTIONS                                        >>>>>>

#endif // STORAGE_FACTORY_STORAGE_FACTORY_H
