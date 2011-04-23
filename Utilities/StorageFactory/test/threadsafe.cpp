#include "Utilities/StorageFactory/test/Test.h"
#include "Utilities/StorageFactory/interface/Storage.h"
#include <boost/thread/thread.hpp>
#include <vector>

static void dump()
{
  std::vector<char> buf(10000,'1');
  Storage *s = StorageFactory::get ()->open
    ("/dev/null", IOFlags::OpenWrite|IOFlags::OpenAppend);

  for (int i = 0; i < 10000; ++i)
    s->write(&buf[0], buf.size());

  s->close();
  delete s;
}

int main (int, char **)
{
  initTest();

  std::cout << "start StorageFactory thread test\n";

  static const int NUMTHREADS = 10;
  boost::thread_group threads;
  for (int i = 0; i < NUMTHREADS; ++i)
    threads.create_thread(&dump);
  threads.join_all();

  std::cout << StorageAccount::summaryXML () << std::endl;
  return EXIT_SUCCESS;
}
