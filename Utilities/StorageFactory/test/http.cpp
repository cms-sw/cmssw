//<<<<<< INCLUDES                                                       >>>>>>

#include "Utilities/Configuration/interface/Architecture.h"
#include "Utilities/StorageFactory/interface/StorageFactory.h"
#include "Utilities/StorageFactory/interface/StorageAccount.h"
#include "PluginManager/PluginManager.h"
#include "SealBase/Storage.h"
#include "SealBase/TempFile.h"
#include "SealBase/Filename.h"
#include "SealBase/DebugAids.h"
#include "SealBase/Signal.h"
#include <iostream>

//<<<<<< PRIVATE DEFINES                                                >>>>>>
//<<<<<< PRIVATE CONSTANTS                                              >>>>>>
//<<<<<< PRIVATE TYPES                                                  >>>>>>
//<<<<<< PRIVATE VARIABLE DEFINITIONS                                   >>>>>>
//<<<<<< PUBLIC VARIABLE DEFINITIONS                                    >>>>>>
//<<<<<< CLASS STRUCTURE INITIALIZATION                                 >>>>>>
//<<<<<< PRIVATE FUNCTION DEFINITIONS                                   >>>>>>
//<<<<<< PUBLIC FUNCTION DEFINITIONS                                    >>>>>>
//<<<<<< MEMBER FUNCTION DEFINITIONS                                    >>>>>>

using namespace seal;
int main (int, char **argv)
{
    Signal::handleFatal (argv [0]);
    PluginManager::get ()->initialise ();

    // Make temporary directory for downloads
    Filename tmpdir;
    TempFile::dir (tmpdir);

    Storage	*s = StorageFactory::get ()->open
	("http://cmsdoc.cern.ch/cms.html", IOFlags::OpenRead, tmpdir.name ());
    char	buf [1024];
    IOSize	n;

    ASSERT (s);
    while ((n = s->read (buf, sizeof (buf))))
	std::cout.write (buf, n);

    delete s;

    // Delete temporaries
    Filename::remove (tmpdir, true);

    std::cerr << "stats:\n" << StorageAccount::summaryText ();
    return EXIT_SUCCESS;
}
