//<<<<<< INCLUDES                                                       >>>>>>

#include "Utilities/StorageFactory/plugins/ZipMemberStorageMaker.h"
#include "Utilities/StorageFactory/interface/StorageFactory.h"
#include "SealZip/ZipArchive.h"
#include "SealZip/ZipMember.h"
#include "SealBase/StringOps.h"
#include <boost/thread/tss.hpp>
#include <boost/shared_ptr.hpp>
#include <iostream>
#include <map>

//<<<<<< PRIVATE DEFINES                                                >>>>>>
//<<<<<< PRIVATE CONSTANTS                                              >>>>>>
//<<<<<< PRIVATE TYPES                                                  >>>>>>
//<<<<<< PRIVATE VARIABLE DEFINITIONS                                   >>>>>>
//<<<<<< PUBLIC VARIABLE DEFINITIONS                                    >>>>>>
//<<<<<< CLASS STRUCTURE INITIALIZATION                                 >>>>>>
//<<<<<< PRIVATE FUNCTION DEFINITIONS                                   >>>>>>
//<<<<<< PUBLIC FUNCTION DEFINITIONS                                    >>>>>>
//<<<<<< MEMBER FUNCTION DEFINITIONS                                    >>>>>>

namespace {
    typedef std::map<std::string, boost::shared_ptr<seal::ZipArchive> > ZipArchiveCache;
    typedef ZipArchiveCache::iterator CacheIterator;

    boost::thread_specific_ptr<ZipArchiveCache> s_archiveCaches;
    ZipArchiveCache &archiveCache ()
    {
       if (! s_archiveCaches.get ())
	  s_archiveCaches.reset (new ZipArchiveCache);

       return *s_archiveCaches.get ();
    }
}

seal::Storage *
ZipMemberStorageMaker::doOpen (const std::string & /* proto */,
		             const std::string &path,
			     int mode,
		             const std::string &tmpdir)
{
    using namespace seal;

    // FIXME: On write, create a temporary local file open for write;
    // on close, trigger transfer to destination.  If opening existing
    // file for write, may need to first download.
    //
    // FIXME: Create our own derived type of ZipArchive, and destroy
    // the internal objects on close.
    //
    // FIXME: Support writing by making huge substorage object, which
    // when closes update the zipmember real size, and does next member,
    // which will cause the member's header to be updated in the archive.
    // This allows one (and exactly one), the last member, to be written
    // into the archive directly.
    ASSERT (! (mode & (IOFlags::OpenWrite | IOFlags::OpenCreate)));

    // Make sure fragment identifier exists
    int fragidx = StringOps::rfind (path, '#');
    if (fragidx == -1)
	return 0;
      
    // Get archive and member names
    std::string archiveName (path, 0, fragidx);
    std::string memberName (path, fragidx+1);

    boost::shared_ptr<ZipArchive> &zip = archiveCache () [archiveName];
    if (! zip)
    {
	// Use the factory again to get actual archive
	Storage *zipStore = StorageFactory::get ()->open (archiveName, mode, tmpdir);
	if (! zipStore)
	  return 0;
	
	zip.reset (new ZipArchive (zipStore));
    }

    // Find the requested member.
    ZipMember *member = zip->member (memberName);
    if (! member || member->method () != ZipMember::STORED)
    {
	  // delete zip;
	  return 0;
    }
      
    // FIXME: leaking zip -- who owns that one?
    return zip->openStored (member);
}
