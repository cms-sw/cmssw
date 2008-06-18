#ifndef STORAGE_FACTORY_STORAGE_MAKER_H
# define STORAGE_FACTORY_STORAGE_MAKER_H

# include "Utilities/StorageFactory/interface/IOTypes.h"
# include <string>

class Storage;
class StorageMaker
{
public:
  virtual ~StorageMaker (void);
  // implicit copy constructor
  // implicit assignment operator

  virtual void		stagein (const std::string &proto,
				 const std::string &path);
  virtual Storage *	open (const std::string &proto,
			      const std::string &path,
			      int mode,
			      const std::string &tmpdir) = 0;
  virtual bool		check (const std::string &proto,
			       const std::string &path,
			       IOOffset *size = 0);
};

#endif // STORAGE_FACTORY_STORAGE_MAKER_H
