#ifndef TkDetLayers_DetGroupElementZLess_h
#define TkDetLayers_DetGroupElementZLess_h

#pragma GCC visibility push(hidden)
class DetGroupElementZLess {
public:
  bool operator()(DetGroup a, DetGroup b) {
    return (std::abs(a.front().det()->position().z()) < std::abs(b.front().det()->position().z()));
  }
};

#pragma GCC visibility pop
#endif
