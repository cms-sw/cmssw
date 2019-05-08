#ifndef HISTO_DATA__H
#define HISTO_DATA__H

#include <TFile.h>
#include <TH1.h>
#include <cstring>
#include <string>

class HistoData {
public:
  HistoData(
      std::string Name, int PlotType, int Bin, std::string NewPath, TFile *NewFile, std::string RefPath, TFile *RefFile);
  HistoData(std::string Name, int PlotType, int Bin, TH1 *NewHisto, TH1 *RefHisto);
  virtual ~HistoData() {}

  // Get General Information
  std::string getName() const { return name; }
  // PlotType getType() const { return type; }
  int getType() const { return type; }
  int getBin() const { return bin; }
  TH1 *getNewHisto() const { return newHisto; }
  TH1 *getRefHisto() const { return refHisto; }
  std::string getResultImage() const { return resultImage; }
  std::string getResultTarget() const { return resultTarget; }

  // Projections/Rebinning Getters
  bool getDoDrawErrorBars() const { return doDrawErrorBars; }
  bool getDoAllow1DRebinning() const { return doAllow1DRebinning; }
  bool getDoAllow2DRebinningX() const { return doAllow2DRebinningX; }
  bool getDoAllow2DRebinningY() const { return doAllow2DRebinningY; }
  bool getDoProjectionsX() const { return doProjectionsX; }
  bool getDoProjectionsY() const { return doProjectionsY; }
  int getMaxProjectionsX() const { return maxProjectionsX; }
  int getMaxProjectionsY() const { return maxProjectionsY; }

  // Get Test Results
  bool comparisonSuccess() const { return lowScore != 10 && !isEmpty; }
  float getKSScore() const { return ksScore; }
  float getChi2Score() const { return chi2Score; }
  float getLowScore() const { return lowScore; }
  float getHighScore() const { return highScore; }
  bool getResult() const { return result; }
  bool getIsEmpty() const { return isEmpty; }

  // Get Visual Attributes
  bool getLineUseFillColor() const { return lineUseFillColor; }
  int getSolidLineColor() const { return lineUseFillColor ? solidFillColor : solidLineColor; }
  int getSolidFillColor() const { return solidFillColor; }
  int getSolidFillStyle() const { return solidFillStyle; }
  int getShadedLineColor() const { return lineUseFillColor ? shadedFillColor : shadedLineColor; }
  int getShadedFillColor() const { return shadedFillColor; }
  int getShadedFillStyle() const { return shadedFillStyle; }

  // Set General Information
  void setName(std::string Name) { name = Name; }
  // void setType(PlotType Type) { type = Type; }
  void setType(int PlotType) { type = PlotType; }
  void setBin(int Bin) { bin = Bin; }
  void setResultImage(std::string Image) { resultImage = Image; }
  void setResultTarget(std::string Target) { resultTarget = Target; }

  // Projections/Rebinning Setters
  void setDoDrawErrorBars(bool Toggle) { doDrawErrorBars = Toggle; }
  void setDoAllow1DRebinning(bool Toggle) { doAllow1DRebinning = Toggle; }
  void setDoAllow2DRebinningX(bool Toggle) { doAllow2DRebinningX = Toggle; }
  void setDoAllow2DRebinningY(bool Toggle) { doAllow2DRebinningY = Toggle; }
  void setDoProjectionsX(bool Toggle) { doProjectionsX = Toggle; }
  void setDoProjectionsY(bool Toggle) { doProjectionsY = Toggle; }
  void setMaxProjections(int Num) {
    maxProjectionsX = Num;
    maxProjectionsY = Num;
  }
  void setMaxProjectionsX(bool Num) { maxProjectionsX = Num; }
  void setMaxProjectionsY(bool Num) { maxProjectionsY = Num; }

  // Set Test Results
  void setKSScore(float Score) { ksScore = Score; }
  void setChi2Score(float Score) { chi2Score = Score; }
  void setLowScore(float Score) { lowScore = Score; }
  void setHighScore(float Score) { highScore = Score; }
  void setResult(bool Result);  // also sets display colors
  void setIsEmpty(bool Toggle) { isEmpty = Toggle; }

  // Set Visual Attributes
  void setLineUseFillColor(bool Toggle) { lineUseFillColor = Toggle; }
  void setSolidLineColor(int Color) { solidLineColor = Color; }
  void setSolidFillColor(int Color) { solidFillColor = Color; }
  void setSolidFillStyle(int Style) { solidFillStyle = Style; }
  void setShadedLineColor(int Color) { shadedLineColor = Color; }
  void setShadedFillColor(int Color) { shadedFillColor = Color; }
  void setShadedFillStyle(int Style) { shadedFillStyle = Style; }

  // Misc Functions
  void drawResult(TH1 *Summary, bool Vertical = true, bool SetLabels = false);
  void clear() {
    newHisto->Clear();
    refHisto->Clear();
  };
  inline void dump();

private:
  // Misc. Data
  std::string name;
  // PlotType type;
  int type;
  int bin;
  TH1 *newHisto;
  TH1 *refHisto;
  std::string resultImage;
  std::string resultTarget;
  bool doAllow1DRebinning;
  bool doDrawErrorBars;

  // 2D data members
  bool doAllow2DRebinningX;
  bool doAllow2DRebinningY;
  int maxProjectionsX;
  int maxProjectionsY;
  bool doProjectionsX;
  bool doProjectionsY;

  // Scores/Results
  float ksScore;
  float chi2Score;
  float lowScore;
  float highScore;
  bool result;
  bool isEmpty;

  // Attributes of Results Display
  int passColor, failColor, errorColor;
  int solidLineColor, solidFillColor, solidFillStyle;
  int shadedLineColor, shadedFillColor, shadedFillStyle;
  bool lineUseFillColor;

  // Implementation Function
  void initialize();
};

#endif  // HISTO_DATA__H
