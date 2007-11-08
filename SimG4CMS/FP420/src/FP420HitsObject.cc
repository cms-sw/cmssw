#include "SimG4CMS/FP420/interface/FP420HitsObject.h"

FP420HitsObject::FP420HitsObject(std::string n, TrackingSlaveSD::Collection& h): _hits(h), _name(n)
{}
