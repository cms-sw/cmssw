#include "Utilities/StorageFactory/interface/StorageFactory.h"
#include "Utilities/StorageFactory/interface/StorageAccount.h"
#include "Utilities/StorageFactory/interface/Storage.h"
#include "FWCore/PluginManager/interface/PluginManager.h"
#include "FWCore/PluginManager/interface/standard.h"
#include "FWCore/Utilities/interface/Exception.h"
#include <boost/shared_ptr.hpp>
#include <boost/thread/mutex.hpp>
#include <boost/thread/condition.hpp>
#include <boost/thread/thread.hpp>
#include <iostream>
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

struct IsDoneWriting
{
  int &flag_;
  IsDoneWriting(int &flag) : flag_(flag) { }
  bool operator()() { return flag_ == 0; }
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

class OutputDropBox
{
public:
  OutputDropBox() : outbuf(1000000), nout(0),  ce(""), writing(0) {}

  // called by main 
  bool set (std::vector<char> &ibuf, IOSize n)
  {
    // wait that all threads finish write to swap
    ScopedLock gl(lock);
    done.wait(gl, IsDoneWriting(writing));

    bool ret = true;
    // if error in previous write return....
    if (ce == "")
    {
      outbuf.swap(ibuf);
      nout = n;
    }
    else
      ret = false;

    undo(); // clean al bits, set writing to its size...

    // notify threads buffer is ready
    doit.notify_all();
    return ret;
  } 

  // called by thread
  bool write (Storage *os, int id)
  {
    ScopedLock gl(lock);
    // wait if box empty or this thread already consumed...
    if (m_done[id])
      doit.wait(gl);

    bool ret=true;
    if (nout==0)
      // notify thread to exit....
      ret=false;
    else
    {
      try { os->write(&outbuf[0], nout); }
      catch (cms::Exception &lce)
      {
	ce = lce.explainSelf();
      } 
    }

    {
      // declare it finishes
      ScopedLock wl(wlock);
      m_done[id]=true;
      writing--;
    } 
    done.notify_all(); 
    return ret;
  }

  // add a writer (called by thread itself), return thread index....
  int addWriter (void)
  {
    ScopedLock wl(wlock);
    m_done.push_back(true);
    return m_done.size()-1;
  }

  // clear bits (declare box ready...)
  void undo()
  {
    ScopedLock wl(wlock);
    writing = m_done.size();
    std::fill(m_done.begin(),m_done.end(),false);
  }

  std::vector<bool> m_done;
  std::vector<char> outbuf;
  IOSize nout;
  std::string ce;
  int writing;
  // writing lock
  mutable boost::mutex wlock;
  // swap lock
  mutable boost::mutex lock;
  mutable boost::condition doit;
  mutable boost::condition done;
};

class InputDropBox
{
public:
  InputDropBox() : end(false), inbuf(1000000), nin(0),  ce("") {}

  // called by main 
  IOSize get(std::vector<char> &ibuf)
  {
    ScopedLock gl(lock);
    if (end)
      // if thread is over return...
      return 0;

    if (nin == 0)
      // wait the thread to finish to read before swapping
      done.wait(gl);

    IOSize ret = 0;
    // if error in previous write return....
    if (ce == "")
    {
      inbuf.swap(ibuf);
      ret = nin;
    }
    nin = 0;
    // notify threads buffer is ready
    doit.notify_all();
    return ret;
  } 

  // called by thread
  bool read (Storage *os)
  {
    ScopedLock gl(lock);
    if (nin != 0)
      // wait if box empty or this thread already consumed...
      doit.wait(gl);

    bool ret=true;

    if (inbuf.empty())
    {
      // inbuf empty notify thread to exit....
      end=true; 
      ret=false;
    }
    else
    {
      try
      {
	nin = os->read(&inbuf[0],inbuf.size());
	if (nin==0)
	{
	  end=true; 
	  ret=false; // stop thread
	}
      }
      catch(cms::Exception &e)
      {
	ce = e.explainSelf();
      } 
    }

    done.notify_all(); 
    return ret;
  }

  bool end;
  std::vector<char> inbuf;
  IOSize nin;
  std::string ce;
  // swap lock
  mutable boost::mutex lock;
  mutable boost::condition doit;
  mutable boost::condition done;
};

static InputDropBox  inbox;
static OutputDropBox dropbox;

static void writeThread (void *param)
{
  Storage *os = static_cast<Storage *> (param);
  int myid = dropbox.addWriter();

  std::cout << "start writing thread " << myid << std::endl;
  while (dropbox.write(os, myid))
    ;
  std::cout << "end writing thread " << myid << std::endl;
}

static void readThread (void *param)
{
  Storage *os = static_cast<Storage *> (param);

  std::cout << "start reading thread" << std::endl;
  while (inbox.read(os))
    ;
  std::cout << "end reading thread" << std::endl;
}

int main (int argc, char **argv)
{
  edmplugin::PluginManager::configure(edmplugin::standard::config());

  if (argc < 3)
  {
    std::cerr << "usage: " << argv[0] << " INFILE OUTFILE...\n";
    return EXIT_FAILURE;
  }

  boost::shared_ptr<Storage>			is;
  std::vector<boost::shared_ptr<Storage> >	os(argc-2);
  boost::thread_group				threads;
  bool						readThreadActive = true;
  bool						writeThreadActive = true;
  IOOffset					size = -1;

  StorageFactory::get ()->enableAccounting(true);
  bool exists = StorageFactory::get ()->check(argv [1], &size);
  std::cerr << "input file exists = " << exists << ", size = " << size << "\n";
  if (! exists) return EXIT_SUCCESS;

  try
  {
    is.reset(StorageFactory::get ()->open (argv [1]));
    if (readThreadActive) 
      threads.create_thread(thread_adapter(&readThread,is.get())); 
  }
  catch (cms::Exception &e)
  {
    std::cerr << "error in opening input file " << argv[1]
	      << ":\n" << e.explainSelf() << std::endl;
    return EXIT_FAILURE;
  }

  // open output files and create threads, one thread per output
  for (int i=0; i < argc-2; i++)
    try
    {
      os[i].reset(StorageFactory::get ()->open (argv[i+2],
						IOFlags::OpenWrite
						| IOFlags::OpenCreate
						| IOFlags::OpenTruncate));
    if (writeThreadActive) 
	threads.create_thread(thread_adapter(&writeThread,os[i].get())); 
  }
  catch (cms::Exception &e)
  {
    std::cerr << "error in opening output file " << argv[i+2]
	      << ":\n" << e.explainSelf() << std::endl;
    return EXIT_FAILURE;
  }

  std::vector<char> inbuf (1048576);
  std::vector<char> outbuf(1048576);
  IOSize	n;

  while ((n = readThreadActive ? inbox.get(inbuf) : is->read(&inbuf[0],inbuf.size())))
  {
    //free reading thread
    inbuf.swap(outbuf);
    // wait threads have finished to write
    // drop buffer in thread
    if (writeThreadActive)
    {
      if (! dropbox.set(outbuf,n))
	break;
    }
    else
      for (size_t i = 0; i < os.size(); i++)
        os[i]->write(&outbuf[0],n);
  }

  std::cout << "main end reading" << std::endl;

  // tell thread to end
  inbuf.clear();

  if (readThreadActive)
    inbox.get(inbuf);
  if (writeThreadActive)
    dropbox.set(outbuf, 0);

  if (writeThreadActive || readThreadActive)
    threads.join_all();

  std::cout << StorageAccount::summaryXML () << std::endl;
  return EXIT_SUCCESS;
}
