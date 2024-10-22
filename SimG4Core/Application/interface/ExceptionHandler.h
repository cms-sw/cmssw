// ------------------------------------------------------------
//
// Author: V.Ivanchenko - 01.11.2017 - old code re-written
//
// ------------------------------------------------------------
//
// Class description:
//
// Catch Geant4 exception and throw CMS exception allowing
// correctly abort problematic run or notify about a problem
// ------------------------------------------------------------

#ifndef SimG4Core_Application_ExceptionHandler_H
#define SimG4Core_Application_ExceptionHandler_H

#include "G4VExceptionHandler.hh"
#include "G4ExceptionSeverity.hh"

class ExceptionHandler : public G4VExceptionHandler {
public:
  explicit ExceptionHandler(double th, bool tr);
  ~ExceptionHandler() override;

  int operator==(const ExceptionHandler &right) const { return (this == &right); }
  int operator!=(const ExceptionHandler &right) const { return (this != &right); }

  bool Notify(const char *exceptionOrigin,
              const char *exceptionCode,
              G4ExceptionSeverity severity,
              const char *description) override;

  ExceptionHandler(const ExceptionHandler &) = delete;
  ExceptionHandler &operator=(const ExceptionHandler &right) = delete;

private:
  double m_eth;
  int m_number{0};
  bool m_trace;
};

#endif
