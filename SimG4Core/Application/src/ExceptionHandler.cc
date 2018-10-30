#include "SimG4Core/Application/interface/ExceptionHandler.h"
#include "SimG4Core/Notification/interface/SimG4Exception.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "globals.hh"
#include <sstream>

ExceptionHandler::ExceptionHandler() 
{}

ExceptionHandler::~ExceptionHandler() 
{}

bool ExceptionHandler::Notify(const char* exceptionOrigin,
			      const char* exceptionCode,
			      G4ExceptionSeverity severity,
			      const char* description)
{
  static const G4String es_banner
    = "\n-------- EEEE ------- G4Exception-START -------- EEEE -------\n";
  static const G4String ee_banner
    = "\n-------- EEEE -------- G4Exception-END --------- EEEE -------\n";
  static const G4String ws_banner
    = "\n-------- WWWW ------- G4Exception-START -------- WWWW -------\n";
  static const G4String we_banner
    = "\n-------- WWWW -------- G4Exception-END --------- WWWW -------\n";

  std::stringstream message;
  message << "*** G4Exception : " << exceptionCode << "\n"
          << "      issued by : " << exceptionOrigin << "\n"
          << description;

  std::stringstream ss;
  switch(severity) {

  case FatalException:
  case FatalErrorInArgument:
  case RunMustBeAborted:
  case EventMustBeAborted:
    ss << es_banner << message.str() << ee_banner;
    throw SimG4Exception(ss.str());
    break;

  case JustWarning:
    edm::LogWarning("SimG4CoreApplication") 
      << ws_banner << message.str() << "*** This is just a warning message. ***"
      << we_banner;
    break;
  }
  return false;
}

