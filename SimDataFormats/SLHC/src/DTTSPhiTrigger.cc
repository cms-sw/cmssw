#include <sstream>
#include <iomanip> // for setiosflags(...)
#include "L1Trigger/GlobalMuonTrigger/src/L1MuGMTConfig.h"
#include "CondFormats/L1TObjects/interface/L1MuTriggerScales.h"
#include "SimDataFormats/SLHC/interface/DTTSPhiTrigger.h"
#include "DTUtils.h"


using namespace std;



DTTSPhiTrigger::DTTSPhiTrigger(): 
  DTChambPhSegm(DTChamberId(), 0) 
{}



DTTSPhiTrigger::DTTSPhiTrigger(const DTChambPhSegm& ChPhisegm, 
			       Global3DPoint position,
			       Global3DVector direction): 
  DTChambPhSegm(ChPhisegm),_position(position),_direction(direction)
{
  _wheel      = this->wheel();
  _station    = this->station();
  _sector     = this->sector();
  /// trigger K parameter converted to angle (scaled bit pattern)
  _psi = this->psi();
  /// trigger X parameter converted to angle (scaled bit pattern)
  _psiR = this->psiR();
  /// bending angle (scaled bit pattern)
  _DeltaPsiR = this->DeltaPsiR();
  /// bending angle (unscaled float)
  _phiB = this->phiB();
}



std::string DTTSPhiTrigger::sprint() const
{
  std::ostringstream outString;
  outString << "  wheel "     << this->wheel() 
	    << "  station "   << this->station() 
	    << "  sector "    << this->sector() 
	    << "  traco Nr "  << this->tracoNumber() 
	    << " at step "    << this->step() 
	    << "  code "      << this->code() << endl;
  outString << "  K "         << this->K() 
	    << "  X "         << this->X() 
	    << "  psi "       << this->psi()
	    << "  psiR "      << this->psiR()
	    << "  DeltaPsiR " << this->DeltaPsiR()
	    << "  phiB "      << this->phiB() << endl;
  outString << "  position "  << this->cmsPosition() << endl;
  outString << "  direction"  << this->cmsDirection() << endl;
  return outString.str();
}
 

