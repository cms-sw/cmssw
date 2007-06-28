#ifndef STORAGE_FACTORY_STORAGE_MAKER_H
# define STORAGE_FACTORY_STORAGE_MAKER_H

//<<<<<< INCLUDES                                                       >>>>>>

# include "SealBase/sysapi/IOTypes.h"
# include <string>

//<<<<<< PUBLIC DEFINES                                                 >>>>>>
//<<<<<< PUBLIC CONSTANTS                                               >>>>>>
//<<<<<< PUBLIC TYPES                                                   >>>>>>

namespace seal { class Storage; }

//<<<<<< PUBLIC VARIABLES                                               >>>>>>
//<<<<<< PUBLIC FUNCTIONS                                               >>>>>>
//<<<<<< CLASS DECLARATIONS                                             >>>>>>

class StorageMaker
{
public:
    virtual ~StorageMaker (void);
    // implicit copy constructor
    // implicit assignment operator

    seal::Storage *open (const std::string &proto,
		    		 const std::string &path,
				 int mode,
				 const std::string &tmpdir);
    bool	check   (const std::string &proto,
		    		 const std::string &path,
				 seal::IOOffset *size = 0) {return check_(proto, path, size);}
private:
    virtual seal::Storage *open_ (const std::string &proto,
		    		 const std::string &path,
				 int mode,
				 const std::string &tmpdir) = 0;
    virtual bool	check_   (const std::string &proto,
		    		 const std::string &path,
				 seal::IOOffset *size = 0);
};

//<<<<<< INLINE PUBLIC FUNCTIONS                                        >>>>>>
//<<<<<< INLINE MEMBER FUNCTIONS                                        >>>>>>

#endif // STORAGE_FACTORY_STORAGE_MAKER_H
