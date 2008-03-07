#include <SimDataFormats/DigiSimLinks/interface/DTDigiSimLink.h>

using namespace std;

DTDigiSimLink::DTDigiSimLink(int wireNr, int digiNr, unsigned int trackId, EncodedEventId evId):
  theWire(wireNr),
  theDigiNumber(digiNr),
  theSimTrackId(trackId),
  theEventId(evId)
{}

DTDigiSimLink::DTDigiSimLink():
  theWire(0),
  theDigiNumber(0),
  theSimTrackId(0),
  theEventId(0)
{}

DTDigiSimLink::ChannelType DTDigiSimLink::channel() const {
  ChannelPacking result;
  result.wi = theWire;
  result.num = theDigiNumber;
  return *(reinterpret_cast<DTDigiSimLink::ChannelType*>(&result));
}

int DTDigiSimLink::wire() const {return theWire;}

int DTDigiSimLink::number() const {return theDigiNumber;}

unsigned int DTDigiSimLink::SimTrackId() const {return theSimTrackId;}

EncodedEventId DTDigiSimLink::eventId() const {return theEventId;}
