#ifndef STORAGE_FACTORY_THROW_H
#define STORAGE_FACTORY_THROW_H

#include "FWCore/Utilities/interface/EDMException.h"

void throwStorageError (const char* category,
                        const char *context,
                        const char *call,
                        int error);

void throwStorageError (edm::errors::ErrorCodes category,
                        const char *context,
                        const char *call,
                        int error);

#endif // STORAGE_FACTORY_THROW_H
