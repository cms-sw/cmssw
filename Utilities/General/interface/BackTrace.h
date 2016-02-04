#ifndef UTILITIES_GENERAL_BACKTRACE_H
#define UTILITIES_GENERAL_BACKTRACE_H
//
//   V 0.0 
//  V.I.&L.T. 19/07/2001
//   imported from original Lassi code...

#include <iosfwd>

/**
 */
class BackTrace {
public:
  enum {MAX_BACKTRACE_DEPTH=100};
  /// constructor
  BackTrace();

  /// destructor
  ~BackTrace(){}

  void trace() const;
  void trace(std::ostream & out) const;

private:

};

#endif // UTILITIES_GENERAL_BACKTRACE_H
