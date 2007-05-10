#ifndef CSCDigitizer_CSCStripAmpResponse_h
#define CSCDigitizer_CSCStripAmpResponse_h


class CSCStripAmpResponse
{
public:
  enum tailShapes {NONE, CONSERVATIVE, RADICAL};

  CSCStripAmpResponse(int shapingTime, int tailShaping);

  float calculateAmpResponse(float t) const;

private:
  int theShapingTime;
  int theTailShaping;
};

#endif

