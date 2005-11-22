//<<<<<< INCLUDES                                                       >>>>>>

#include "Utilities/StorageFactory/interface/HttpStorageMaker.h"
#include "Utilities/StorageFactory/interface/GsiFTPStorageMaker.h"
#include "Utilities/StorageFactory/interface/LocalStorageMaker.h"
#include "Utilities/StorageFactory/interface/ZipMemberStorageMaker.h"
#include "Utilities/StorageFactory/interface/DCacheStorageMaker.h"
#include "Utilities/StorageFactory/interface/RFIOStorageMaker.h"
#include "Utilities/StorageFactory/interface/RedirectStorageMaker.h"
#include "Utilities/StorageFactory/interface/StorageFactory.h"
#include "PluginManager/ModuleDef.h"

//<<<<<< PRIVATE DEFINES                                                >>>>>>
//<<<<<< PRIVATE CONSTANTS                                              >>>>>>
//<<<<<< PRIVATE TYPES                                                  >>>>>>
//<<<<<< PRIVATE VARIABLE DEFINITIONS                                   >>>>>>
//<<<<<< PUBLIC VARIABLE DEFINITIONS                                    >>>>>>
//<<<<<< CLASS STRUCTURE INITIALIZATION                                 >>>>>>
//<<<<<< PRIVATE FUNCTION DEFINITIONS                                   >>>>>>
//<<<<<< PUBLIC FUNCTION DEFINITIONS                                    >>>>>>
//<<<<<< MEMBER FUNCTION DEFINITIONS                                    >>>>>>

DEFINE_SEAL_MODULE ();
DEFINE_SEAL_PLUGIN (StorageFactory, HttpStorageMaker, "http");
DEFINE_SEAL_PLUGIN (StorageFactory, HttpStorageMaker, "ftp");
DEFINE_SEAL_PLUGIN (StorageFactory, HttpStorageMaker, "web");
DEFINE_SEAL_PLUGIN (StorageFactory, GsiFTPStorageMaker, "gsiftp");
DEFINE_SEAL_PLUGIN (StorageFactory, GsiFTPStorageMaker, "sfn");
DEFINE_SEAL_PLUGIN (StorageFactory, LocalStorageMaker, "file");
DEFINE_SEAL_PLUGIN (StorageFactory, ZipMemberStorageMaker, "zip-member");
DEFINE_SEAL_PLUGIN (StorageFactory, DCacheStorageMaker, "dcache");
DEFINE_SEAL_PLUGIN (StorageFactory, DCacheStorageMaker, "dcap");
DEFINE_SEAL_PLUGIN (StorageFactory, DCacheStorageMaker, "gsidcap");
DEFINE_SEAL_PLUGIN (StorageFactory, RFIOStorageMaker, "rfio");
DEFINE_SEAL_PLUGIN (StorageFactory, RedirectStorageMaker, "redirect");
