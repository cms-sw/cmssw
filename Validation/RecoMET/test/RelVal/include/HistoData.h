#ifndef HISTO_DATA__H
#define HISTO_DATA__H

#include <string>
#include <string.h>

class HistoData {
public:

  HistoData(std::string Name, int Number);
  virtual ~HistoData();

  // Getters
  std::string GetName() { return name; }
  std::string GetValueX() { return x_value; }
  unsigned short GetType() { return type; }
  int GetNumber() { return number; }
  //int GetNBinsX() { return nbinsx; }
  //int GetNBinsY() { return nbinsy; }
  //float GetMinX() { return xmin; }
  //float GetMaxX() { return xmax; }
  //float GetMinY() { return ymin; }
  //float GetMaxY() { return ymax; }

  // Setters
  void SetType(unsigned short Type) { type = Type; }
  void SetValueX(std::string Value) { x_value = Value; }
  //void SetBinsX(int N, float MinX, float MaxX) { nbinsx = N; xmin = MinX; xmax = MaxX; }
  void SetValueY(std::string Value) { y_value = Value; }
  //void SetBinsY(int N, float MinY, float MaxY) { nbinsy = N; ymin = MinY; ymax = MaxY; }

  // Misc Functions
  void Clear() { memset(this,0,sizeof(struct HistoData)); }
  void Dump();

private:

  std::string name;
  std::string x_value, y_value;
  int number;
  unsigned short type;
  //int nbinsx, nbinsy;
  //float xmin, xmax;
  //float ymin, ymax;

};

#endif // HISTO_DATA__H
