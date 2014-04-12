#include <SimDataFormats/DigiSimLinks/interface/DTDigiSimLink.h>

using namespace std;
const double DTDigiSimLink::reso =  25./32.; //ns

DTDigiSimLink::DTDigiSimLink(int wireNr, int digiNr, int nTDC, unsigned int trackId, EncodedEventId evId):
  theWire(wireNr),
  theDigiNumber(digiNr),
  theCounts(nTDC),
  theSimTrackId(trackId),
  theEventId(evId)
{}

DTDigiSimLink::DTDigiSimLink(int wireNr, int digiNr, double tdrift, unsigned int trackId, EncodedEventId evId):
  theWire(wireNr),
  theDigiNumber(digiNr),
  theCounts(static_cast<int>(tdrift/reso)),
  theSimTrackId(trackId),
  theEventId(evId)
{}

DTDigiSimLink::DTDigiSimLink():
  theWire(0),
  theDigiNumber(0),
  theCounts(0),
  theSimTrackId(0),
  theEventId(0)
{}

DTDigiSimLink::ChannelType DTDigiSimLink::channel() const {
  ChannelPacking result;
  result.wi = theWire;
  result.num = theDigiNumber;
  DTDigiSimLink::ChannelType* p_result = reinterpret_cast<DTDigiSimLink::ChannelType*>(&result);
  return *p_result;
}

int DTDigiSimLink::wire() const {return theWire;}

int DTDigiSimLink::number() const {return theDigiNumber;}

uint32_t DTDigiSimLink::countsTDC() const { return theCounts; }

double DTDigiSimLink::time() const { return theCounts*reso; }

unsigned int DTDigiSimLink::SimTrackId() const {return theSimTrackId;}

EncodedEventId DTDigiSimLink::eventId() const {return theEventId;}
