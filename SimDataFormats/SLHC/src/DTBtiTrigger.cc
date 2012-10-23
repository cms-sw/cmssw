#include <sstream>
#include <iomanip> // for setiosflags(...)
#include "L1Trigger/GlobalMuonTrigger/src/L1MuGMTConfig.h"
#include "CondFormats/L1TObjects/interface/L1MuTriggerScales.h"
//#include "SLHCUpgradeSimulations/L1Trigger/interface/DTBtiTrigger.h"
#include "SimDataFormats/SLHC/interface/DTBtiTrigger.h"


using namespace std;


DTBtiTrigger::DTBtiTrigger(): DTBtiTrigData() {}



DTBtiTrigger::DTBtiTrigger(const DTBtiTrigData& bti): 
  DTBtiTrigData(bti),_position(Global3DPoint()),_direction(Global3DVector())
{
  _wheel      = this->wheel();
  _station    = this->station();
  _sector     = this->sector();
  _superLayer = this->btiSL();
}




DTBtiTrigger::DTBtiTrigger(const DTBtiTrigData& bti, 
			   Global3DPoint position,
			   Global3DVector direction): 
  DTBtiTrigData(bti),_position(position),_direction(direction)
{
  _wheel      = this->wheel();
  _station    = this->station();
  _sector     = this->sector();
  _superLayer = this->btiSL();
}





std::string DTBtiTrigger::sprint() const
{
  std::ostringstream outString;
  outString << "  wheel "    << this->wheel() 
	    << " station "   << this->station() 
	    << " sector "    << this->sector() 
	    << " SL "        << this->btiSL() 
	    << " Nr "        << this->btiNumber() << endl;
  outString << "  step "     << this->step() 
	    << " code "      << this->code() 
	    << " K "         << this->K() 
	    << " X "         << this->X() << endl;
  outString << "  position " << this->cmsPosition() << endl;
  outString << "  direction" << this->cmsDirection() << endl;
  return outString.str();
}

