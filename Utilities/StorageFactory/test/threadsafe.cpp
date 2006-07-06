#include "Utilities/StorageFactory/interface/StorageFactory.h"
#include "Utilities/StorageFactory/interface/StorageAccount.h"
#include "PluginManager/PluginManager.h"
#include "SealBase/Storage.h"
#include "SealBase/DebugAids.h"
#include "SealBase/Signal.h"
#include <iostream>
#include <vector>
#include <boost/thread/thread.hpp>

using namespace seal;

namespace {
  
  void dump() {
    
    std::vector<char> buf(10000,'1');
    
    Storage	*s = StorageFactory::get ()->open ("/tmp/innocent/null", 
						   IOFlags::OpenWrite|IOFlags::OpenAppend);



    for (int i=0;i<100;i++)
      s->write(&buf[0],buf.size());
    delete s;
    
  }
  
}


int main (int argc, char **argv)
{
  Signal::handleFatal (argv [0]);
  PluginManager::get ()->initialise ();
  StorageFactory::get ()->enableAccounting(true);
  
  dump();
  
  
  std::cerr << "stats:\n" << StorageAccount::summaryText () << std::endl;
  return EXIT_SUCCESS;
}
