#ifndef STORAGE_FACTORY_RFIO_STORAGE_MAKER_H
# define STORAGE_FACTORY_RFIO_STORAGE_MAKER_H

//<<<<<< INCLUDES                                                       >>>>>>

# include "Utilities/StorageFactory/interface/StorageMaker.h"
# include <string>

//<<<<<< PUBLIC DEFINES                                                 >>>>>>
//<<<<<< PUBLIC CONSTANTS                                               >>>>>>
//<<<<<< PUBLIC TYPES                                                   >>>>>>
//<<<<<< PUBLIC VARIABLES                                               >>>>>>
//<<<<<< PUBLIC FUNCTIONS                                               >>>>>>
//<<<<<< CLASS DECLARATIONS                                             >>>>>>

class RFIOStorageMaker : public StorageMaker
{
public:
    // implicit constructor
   RFIOStorageMaker();
    // implicit destructor
    // implicit copy constructor
    // implicit assignment operator

    virtual seal::Storage *open_ (const std::string &proto,
		    		 const std::string &path,
				 int mode,
				 const std::string &tmpdir);
    virtual bool	  check_ (const std::string &proto,
		    		 const std::string &path,
				 seal::IOOffset *size = 0);
};

//<<<<<< INLINE PUBLIC FUNCTIONS                                        >>>>>>
//<<<<<< INLINE MEMBER FUNCTIONS                                        >>>>>>

#endif // STORAGE_FACTORY_RFIO_STORAGE_MAKER_H
