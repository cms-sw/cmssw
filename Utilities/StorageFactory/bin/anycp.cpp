#include "Utilities/StorageFactory/interface/StorageFactory.h"
#include "Utilities/StorageFactory/interface/StorageAccount.h"
#include "PluginManager/PluginManager.h"
#include "SealBase/Storage.h"
#include "SealBase/DebugAids.h"
#include "SealBase/Signal.h"
# include "SealBase/IOError.h"
# include <boost/shared_ptr.hpp>
#include <boost/thread/mutex.hpp>
#include <boost/thread/condition.hpp>
#include <iostream>
#include <boost/thread/thread.hpp>
typedef boost::mutex::scoped_lock ScopedLock;

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


class OutputDropBox {
public:

  OutputDropBox() : os(0), ce(0) {}

  void set(Storage * s, seal::IOBuffer ibuf) {
    ScopedLock gl(lock);
    std::cout << "box set" << std::endl;
    ce=0;
    os = s;
    buffer = ibuf;
    doit.notify_all();
  } 
  
  bool wait () const {
    // first time exit.... 
    if (!os) return true;
    ScopedLock gl(lock);
    std::cout << "box wait" << std::endl;
    done.wait(gl);
    return !ce;
  }
  
  bool write() {
    ScopedLock gl(lock);
    std::cout << "box write" << std::endl;
    doit.wait(gl);
    bool ret=true;
    // os==0 notify thread to exit....
    if (!os) ret=false;
    else
      try {
	os->write(buffer);
      } catch(seal::Error & lce) {
	ce = lce.clone();
      }
    done.notify_all(); 
    return ret;
  }
  
  
  
  seal::IOBuffer buffer;
  Storage * os;
  seal::Error * ce;
  mutable boost::mutex lock;
  mutable boost::condition doit;
  mutable boost::condition done;
};


namespace {

  OutputDropBox dropbox;


  void writeThread() {
    
    std::cout << "start writing thread" << std::endl;

    while(dropbox.write());
    
    std::cout << "end writing thread" << std::endl;
  
  }

}

int main (int argc, char **argv)
{
    Signal::handleFatal (argv [0]);
    PluginManager::get ()->initialise ();

    if (argc <3)
    {
	std::cerr << " please give input and output file names" <<std::endl;
	return EXIT_FAILURE;
    }

    IOOffset    size = -1;
    StorageFactory::get ()->enableAccounting(true);
    bool	exists = StorageFactory::get ()->check(argv [1], &size);
    std::cerr << "exists = " << exists << ", size = " << size << "\n";
    if (! exists) return EXIT_SUCCESS;

     boost::shared_ptr<Storage> is;

    try {
      is.reset(StorageFactory::get ()->open (argv [1]));
    } catch (...) {
      std::cerr << "error in opening input file " << argv[1] << std::endl;
      return EXIT_FAILURE;
    }



    // open output file
    boost::shared_ptr<Storage> os;
    try {
      os.reset(StorageFactory::get ()->open (argv[2],
					     IOFlags::OpenWrite
					     | IOFlags::OpenCreate
					     | IOFlags::OpenTruncate)
	       );
    } catch (...) {
      std::cerr << "error in opening output file " << argv[2] << std::endl;
      return EXIT_FAILURE;
    }
    
    std::vector<char> inbuf(100000);
    std::vector<char> outbuf(100000);
    IOSize	n;
    
    // create thread
    boost::thread_group threads;
    threads.create_thread(&writeThread);
    
    while ((n = is->read (&inbuf[0], inbuf.size()))) {
      // wait thread has finished to write
      if (!dropbox.wait()) break;
      // swap buffers.
      inbuf.swap(outbuf);
      // drop buffer in thread
      dropbox.set(os.get(),seal::IOBuffer(&outbuf[0],n));
    }
    
    std::cout << "main end reading" << std::endl;

    // tell thread to end
    dropbox.wait();
    os.reset();
    dropbox.set(0,seal::IOBuffer());
    
    threads.join_all();
    
    std::cerr << "stats:\n" << StorageAccount::summaryText () << std::endl;
    return EXIT_SUCCESS;
}
