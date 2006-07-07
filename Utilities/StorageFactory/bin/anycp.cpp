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
#include <vector>
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

/*

box not ready: writers wait...
box ready : writers starts, set waits
box ready : first writer end, set waits
box ready : late writers enter -> starts
box ready : first writer re-enter -> wait
set wait, box ready, last writer end -> alldone

*/


struct cond_predicate
{
    cond_predicate(int& var, int val) : _var(var), _val(val) { }

    bool operator()() { return _var == _val; }

    int& _var;
    int _val;
};

class thread_adapter
{
public:
    thread_adapter(void (*func)(void*), void* param) : _func(func), _param(param) { }
    void operator()() const { _func(_param); }
private:
    void (*_func)(void*);
    void* _param;
};


class OutputDropBox {
public:

  OutputDropBox() : outbuf(100000), nout(0),  ce(0), writing(0) {}

  bool set(std::vector<char> & ibuf, IOSize n) {
    //   void set(Storage * s, seal::IOBuffer ibuf) {
    // wait it finishes write to swap
    ScopedLock gl(lock);
    std::cout << "box set " << writing << std::endl;
    done.wait(gl, cond_predicate(writing, 0));
    std::cout << "box setting " << std::endl;
    bool ret = true;
    // if error in previous write return....
    if (ce==0) {
      ce=0;
      outbuf.swap(ibuf);
      nout = n;
    }
    else ret = false;
    undo(); // clean al bits, set writing to its size...
      //    buffer = ibuf;
    doit.notify_all();
    return ret;
  } 
  
  bool wait () const {
    // first time exit.... 
    // if (!nout) return true;
    ScopedLock gl(lock);
    std::cout << "box wait" << std::endl;
    done.wait(gl);
    return !ce;
  }
  
  bool write(Storage * os,int it) {
    ScopedLock gl(lock);
    std::cout << "box write " << nout << std::endl;
    // wait is box empty or this thread already consumed...
    if (m_done[it]) doit.wait(gl);
    std::cout << "box writing " << nout << std::endl;
    bool ret=true;
    // os==0 notify thread to exit....
    if (nout==0) ret=false;
    else
      try {
	std::cout << "box real write " << nout << std::endl;
	os->write(&outbuf[0],nout);
      } catch(seal::Error & lce) {
	ce = lce.clone();
      }
    //    done.notify_all(); 
    {
      ScopedLock wl(wlock);
      m_done[it]=true;
      writing--;
    } 
    done.notify_all(); 
    std::cout << "box write done" << std::endl;
    return ret;
  }
  

  int addWriter(){
    ScopedLock wl(wlock);
    m_done.push_back(true);
    return m_done.size()-1;
  }
 
  void undo() {
    ScopedLock wl(wlock);
    writing= m_done.size();
    std::fill(m_done.begin(),m_done.end(),false);
  }

  std::vector<bool> m_done;
  std::vector<char> outbuf;
  IOSize nout;
  // seal::IOBuffer buffer;
  seal::Error * ce;
  int writing;
  // writing lock
  mutable boost::mutex wlock;
  // swap lock
  mutable boost::mutex lock;
  mutable boost::condition doit;
  mutable boost::condition done;
};


namespace {

  OutputDropBox dropbox;


  void writeThread(void * param) {
    Storage * os = static_cast<Storage * >(param);

    int me = dropbox.addWriter();
    
    std::cout << "start writing thread " << me << std::endl;

    while(dropbox.write(os,me));
    
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


    boost::thread_group threads;

    // open output files
    // create thread
    std::vector<boost::shared_ptr<Storage> > os(argc-2);
    for (int i=0; i<argc-2;i++)
      try {
	os[i].reset(StorageFactory::get ()->open (argv[i+2],
						  IOFlags::OpenWrite
						  | IOFlags::OpenCreate
						  | IOFlags::OpenTruncate)
		    );
	threads.create_thread(thread_adapter(&writeThread,os[i].get())); 
      } catch (...) {
	std::cerr << "error in opening output file " << argv[i+2] << std::endl;
	return EXIT_FAILURE;
      }


    
    std::vector<char> inbuf(100000);
    std::vector<char> outbuf(100000);
    IOSize	n;
    
    
    while ((n = is->read (&inbuf[0], inbuf.size()))) {
      // wait threads have finished to write
      // swap buffers.
      //      inbuf.swap(outbuf);
      // drop buffer in thread
      if (!dropbox.set(inbuf,n)) break;
    }
    
    std::cout << "main end reading" << std::endl;

    // tell thread to end
    // dropbox.wait();
    dropbox.set(inbuf, 0);
    
    threads.join_all();

    
    std::cerr << "stats:\n" << StorageAccount::summaryText () << std::endl;
    return EXIT_SUCCESS;
}
