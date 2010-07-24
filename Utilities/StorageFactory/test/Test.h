#include "Utilities/StorageFactory/interface/StorageFactory.h"
#include "Utilities/StorageFactory/interface/StorageAccount.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/MessageLogger/interface/MessageLoggerQ.h"
#include "FWCore/MessageLogger/interface/MessageDrop.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/PluginManager/interface/PresenceFactory.h"
#include "FWCore/PluginManager/interface/PluginManager.h"
#include "FWCore/PluginManager/interface/standard.h"
#include "boost/shared_ptr.hpp"
#include <iostream>

static boost::shared_ptr<edm::Presence> gobbleUpTheGoop;
static void initTest(void)
{
  // Initialise the plug-in manager.
  edmplugin::PluginManager::configure(edmplugin::standard::config());

  // Enable storage accounting
  StorageFactory::get ()->enableAccounting (true);

  // This interface sucks but does the job, which is to silence
  // the message logger chatter about "info" or "debug" messages,
  // and prevent the tests from hanging forever due to lack of a
  // logger.
  try
  {
    gobbleUpTheGoop = boost::shared_ptr<edm::Presence>
      (edm::PresenceFactory::get()->makePresence("MessageServicePresence").release());
  }
  catch (cms::Exception &e)
  {
    std::cerr << e.explainSelf() << std::endl;
  }

  const char *pset =
    "<destinations=-s({63657272})" // cerr
    ";cerr=-P(<noTimeStamps=-B(true);threshold=-S(5741524e494e47)" // WARNING
    ";WARNING=-P(<limit=-I(+0)>);default=-P(<limit=-I(-1)>)>)>";
  edm::MessageLoggerQ::MLqMOD (new std::string);
  edm::MessageLoggerQ::MLqCFG (new edm::ParameterSet (pset));
  edm::LogInfo("AvoidExitCrash")
    << "Message logger, please don't crash at exit if"
    << " nothing else worthwhile was printed. Thanks.";
}
