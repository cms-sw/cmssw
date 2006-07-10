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
#include "SealBase/TimeInfo.h"
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

  OutputDropBox() : outbuf(1000000), nout(0),  ce(0), writing(0) {}

  // called by main 
  bool set(std::vector<char> & ibuf, IOSize n) {
    //   void set(Storage * s, seal::IOBuffer ibuf) {
    // wait that all threads finish write to swap
    ScopedLock gl(lock);
    done.wait(gl, cond_predicate(writing, 0));

    bool ret = true;
    // if error in previous write return....
    if (ce==0) {
      outbuf.swap(ibuf);
      nout = n;
    }
    else ret = false;
    undo(); // clean al bits, set writing to its size...
      //    buffer = ibuf;
    // notify threads buffer is ready
    doit.notify_all();
    return ret;
  } 
  
  // called by thread
  bool write(Storage * os,int it) {
    ScopedLock gl(lock);
    // wait if box empty or this thread already consumed...
    if (m_done[it]) doit.wait(gl);
    bool ret=true;
    // nout==0 notify thread to exit....
    if (nout==0) ret=false;
    else
      try {
	os->write(&outbuf[0],nout);
      } catch(seal::Error & lce) {
	ce = lce.clone();
      } 
    {
      // declare it finishes
      ScopedLock wl(wlock);
      m_done[it]=true;
      writing--;
    } 
    done.notify_all(); 
    return ret;
  }
  
  /* add a writer (called by thread itself)
     return thread index....
   */
  int addWriter(){
    ScopedLock wl(wlock);
    m_done.push_back(true);
    return m_done.size()-1;
  }
 
  // clear bits (declare box ready...)
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


class InputDropBox {
public:

  InputDropBox() : end(false), inbuf(1000000), nin(0),  ce(0) {}

  // called by main 
  IOSize get(std::vector<char> & ibuf) {
    // if thread is over return...
    if (end) return 0;
    // wait that thread finish to read before swapping
    ScopedLock gl(lock);
    if (nin==0) done.wait(gl);

    IOSize ret = 0;
    // if error in previous write return....
    if (ce==0) {
      inbuf.swap(ibuf);
      ret = nin;
    }
    nin=0;
    // notify threads buffer is ready
    doit.notify_all();
    return ret;
  } 
  
  // called by thread
  bool read(Storage * os) {
    ScopedLock gl(lock);
    // wait if box empty or this thread already consumed...
    if (nin!=0) doit.wait(gl);
    bool ret=true;
    // inbuf empty notify thread to exit....
    if (inbuf.empty()) ret=false;
    else
      try {
	nin = os->read(&inbuf[0],inbuf.size());
	if (nin==0) {
	  end=true; 
	  ret=false; // stop thread
	}
      } catch(seal::Error & lce) {
	ce = lce.clone();
      } 
    
    done.notify_all(); 
    return ret;
  }
  
  bool end;
  std::vector<char> inbuf;
  IOSize nin;
  // seal::IOBuffer buffer;
  seal::Error * ce;
  // swap lock
  mutable boost::mutex lock;
  mutable boost::condition doit;
  mutable boost::condition done;
};





namespace {

  InputDropBox  inbox;
  OutputDropBox dropbox;


  void writeThread(void * param) {
    Storage * os = static_cast<Storage * >(param);

    int me = dropbox.addWriter();
    
    std::cout << "start writing thread " << me << std::endl;

    while(dropbox.write(os,me));
    
    std::cout << "end writing thread " << me << std::endl;
  
  }

  void readThread(void * param) {
    Storage * os = static_cast<Storage * >(param);

    std::cout << "start reading thread " << std::endl;

    while(inbox.read(os));
    
    std::cout << "end reading thread " << std::endl;
  
  }


}

int main (int argc, char **argv)
{
  TimeInfo::init ();

  Signal::handleFatal (argv [0]);
  PluginManager::get ()->initialise ();


  // flags to swith therading on/off
  bool readThreadActive = true;
  bool writeThreadActive = true;
  
  if (argc <3)
    {
      std::cerr << " please give input and output file names" <<std::endl;
      return EXIT_FAILURE;
    }
  
  boost::thread_group threads;
  
  IOOffset    size = -1;
  StorageFactory::get ()->enableAccounting(true);
  bool	exists = StorageFactory::get ()->check(argv [1], &size);
  std::cerr << "exists = " << exists << ", size = " << size << "\n";
  if (! exists) return EXIT_SUCCESS;
  
  boost::shared_ptr<Storage> is;
  
  try {
    is.reset(StorageFactory::get ()->open (argv [1]));
    if (readThreadActive) 
      threads.create_thread(thread_adapter(&readThread,is.get())); 
  } catch (...) {
    std::cerr << "error in opening input file " << argv[1] << std::endl;
    return EXIT_FAILURE;
  }
  
  
  
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
      if (writeThreadActive) 
	threads.create_thread(thread_adapter(&writeThread,os[i].get())); 
    } catch (...) {
      std::cerr << "error in opening output file " << argv[i+2] << std::endl;
      return EXIT_FAILURE;
    }
  
  
  
    std::vector<char>  inbuf(1000000);
    std::vector<char> outbuf(1000000);
    IOSize	n;
    
    
    while ((n = readThreadActive ? inbox.get(inbuf) : is->read(&inbuf[0],inbuf.size())))
      {
	//free reading thread
	inbuf.swap(outbuf);
	// wait threads have finished to write
	// drop buffer in thread
	if (writeThreadActive) {
	  if (!dropbox.set(outbuf,n)) break;
	} else
	  for (int i=0; i<os.size();i++)
	    os[i]->write(&outbuf[0],n);
      }
    
    std::cout << "main end reading" << std::endl;
    
    // tell thread to end
    // dropbox.wait();
    inbuf.clear();

    if (readThreadActive) inbox.get(inbuf);
    if (writeThreadActive) dropbox.set(outbuf, 0);
    
    if (writeThreadActive||readThreadActive) threads.join_all();
    
    
    std::cerr << "stats:\n" << StorageAccount::summaryText () << std::endl;
    return EXIT_SUCCESS;
}

