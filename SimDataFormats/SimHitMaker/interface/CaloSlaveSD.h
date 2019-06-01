///////////////////////////////////////////////////////////////////////////////
// File: CaloSlaveSD.h
// Date: 10.02
// Description: Interfaces CaloHit to appropriate container for ORCA usage
///////////////////////////////////////////////////////////////////////////////
#ifndef CaloSlaveSD_h
#define CaloSlaveSD_h

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "SimDataFormats/CaloHit/interface/PCaloHit.h"
#include "SimDataFormats/CaloHit/interface/PCaloHitContainer.h"

#include <string>
#include <vector>

class CaloSlaveSD {
public:
  typedef std::vector<PCaloHit> Collection;
  typedef Collection::iterator iterator;
  typedef Collection::const_iterator const_iterator;

  CaloSlaveSD(std::string);
  virtual ~CaloSlaveSD();
  virtual void Initialize();
  std::string name() const { return name_; }
  virtual bool processHits(uint32_t, double, double, double, int, uint16_t depth = 0);
  virtual bool format();
  Collection &hits() { return hits_; }
  std::string type() { return "calo"; }
  virtual const_iterator begin() { return hits_.begin(); }
  virtual const_iterator end() { return hits_.end(); }
  virtual void Clean();
  virtual void ReserveMemory(unsigned int size);

protected:
  Collection hits_;

private:
  std::string name_;
};

#endif  // CaloSlaveSD_h
