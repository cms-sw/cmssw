#include <SimDataFormats/DigiSimLinks/interface/DTDigiSimLink.h>

using namespace std;

DTDigiSimLink::DTDigiSimLink(int wireNr, int digiNr, int nTDC, unsigned int trackId, EncodedEventId evId, int base)
    : theWire(wireNr),
      theDigiNumber(digiNr),
      theTDCBase(base),
      theCounts(nTDC),
      theSimTrackId(trackId),
      theEventId(evId) {}

DTDigiSimLink::DTDigiSimLink(int wireNr, int digiNr, double tdrift, unsigned int trackId, EncodedEventId evId, int base)
    : theWire(wireNr),
      theDigiNumber(digiNr),
      theTDCBase(base),
      theCounts(static_cast<int>(tdrift * base / 25.)),
      theSimTrackId(trackId),
      theEventId(evId) {}

DTDigiSimLink::DTDigiSimLink()
    : theWire(0), theDigiNumber(0), theTDCBase(32), theCounts(0), theSimTrackId(0), theEventId(0) {}

DTDigiSimLink::ChannelType DTDigiSimLink::channel() const {
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wuninitialized"
  ChannelPacking result;
  result.wi = theWire;
  result.num = theDigiNumber;
  DTDigiSimLink::ChannelType* p_result = reinterpret_cast<DTDigiSimLink::ChannelType*>(&result);
  return *p_result;
#pragma GCC diagnostic pop
}

int DTDigiSimLink::wire() const { return theWire; }

int DTDigiSimLink::number() const { return theDigiNumber; }

uint32_t DTDigiSimLink::countsTDC() const { return theCounts; }

double DTDigiSimLink::time() const { return theCounts * 25. / theTDCBase; }

unsigned int DTDigiSimLink::SimTrackId() const { return theSimTrackId; }

EncodedEventId DTDigiSimLink::eventId() const { return theEventId; }
