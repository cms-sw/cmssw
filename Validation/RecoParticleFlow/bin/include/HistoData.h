#ifndef HISTO_DATA__H
#define HISTO_DATA__H

#include <string>

class TFile;
class TH1;

class HistoData {
public:

  HistoData(std::string Name, int PlotType, int Bin, std::string NewPath, TFile *NewFile, std::string RefPath, TFile *RefFile);
  HistoData(std::string Name, int PlotType, int Bin, TH1 *NewHisto, TH1 *RefHisto);
  virtual ~HistoData() {}

  // Get General Information
  std::string getName() const { return name; }
  //PlotType getType() const { return type; }
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
  unsigned short getMaxProjectionsX() const { return maxProjectionsX; }
  unsigned short getMaxProjectionsY() const { return maxProjectionsY; }

  // Get Test Results
  bool comparisonSuccess() const { return lowScore != 10 && !isEmpty; }
  float getKSScore() const { return ksScore; }
  float getChi2Score() const { return chi2Score; }
  float getLowScore() const { return lowScore; }
  float getHighScore() const { return highScore; }
  bool getResult() const { return result; }
  bool getIsEmpty() const { return isEmpty; }
  bool getDoDrawScores() const { return doDrawScores; }

  // Get Visual Attributes
  bool getLineUseFillColor() const { return lineUseFillColor; }
  unsigned short getSolidLineColor() const { return lineUseFillColor ? solidFillColor : solidLineColor; }
  unsigned short getSolidFillColor() const { return solidFillColor; }
  unsigned short getSolidFillStyle() const { return solidFillStyle; }
  unsigned short getShadedLineColor() const { return lineUseFillColor ? shadedFillColor : shadedLineColor; }
  unsigned short getShadedFillColor() const { return shadedFillColor; }
  unsigned short getShadedFillStyle() const { return shadedFillStyle; }

  // Set General Information
  void setName(std::string Name) { name = Name; }
  //void setType(PlotType Type) { type = Type; }
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
  void setMaxProjections(unsigned short Num) { maxProjectionsX = Num; maxProjectionsY = Num; }
  void setMaxProjectionsX(unsigned short Num) { maxProjectionsX = Num; }
  void setMaxProjectionsY(unsigned short Num) { maxProjectionsY = Num; }

  // Get Size/Attributes of Generated Plot(s)
  unsigned short getProjectionsHeight() const { return projectionsHeight; }
  unsigned short getProjectionsWidth() const { return projectionsWidth; }
  unsigned short getProjectionsBarsThickness() const { return projectionsBarsThickness; }
  unsigned short getProjectionsTopMargin() const { return projectionsTopMargin; }
  unsigned short getProjectionsLeftMargin() const { return projectionsLeftMargin; }
  unsigned short getProjectionsRightMargin() const { return projectionsRightMargin; }
  unsigned short getProjectionsBottomMargin() const { return projectionsBottomMargin; }
  unsigned short getPlotsHeight() const { return plotsHeight; }
  unsigned short getPlotsWidth() const { return plotsWidth; }
  unsigned short getPlotsTopMargin() const { return plotsTopMargin; }
  unsigned short getPlotsLeftMargin() const { return plotsLeftMargin; }
  unsigned short getPlotsRightMargin() const { return plotsRightMargin; }
  unsigned short getPlotsBottomMargin() const { return plotsBottomMargin; }

  // Set Test Results
  void setKSScore(float Score) { ksScore = Score; }
  void setChi2Score(float Score) { chi2Score = Score; }
  void setLowScore(float Score) { lowScore = Score; }
  void setHighScore(float Score) { highScore = Score; }
  void setResult(bool Result); // also sets display colors
  void setIsEmpty(bool Toggle) { isEmpty = Toggle; }
  void setDoDrawScores(bool Toggle) { doDrawScores = Toggle; }

  // Set Visual Attributes
  void setLineUseFillColor(bool Toggle) { lineUseFillColor = Toggle; }
  void setSolidLineColor(unsigned short Color) { solidLineColor = Color; }
  void setSolidFillColor(unsigned short Color) { solidFillColor = Color; }
  void setSolidFillStyle(unsigned short Style) { solidFillStyle = Style; }
  void setShadedLineColor(unsigned short Color) { shadedLineColor = Color; }
  void setShadedFillColor(unsigned short Color) { shadedFillColor = Color; }
  void setShadedFillStyle(unsigned short Style) { shadedFillStyle = Style; }

  // Set Size/Attributes of Generated Plot(s)
  void setProjectionsWidth(unsigned short Pixels) { projectionsWidth = Pixels; }
  void setProjectionsHeight(unsigned short Pixels) { projectionsHeight = Pixels; }
  void setProjectionsBarsThickness(unsigned short Pixels) { projectionsBarsThickness = Pixels; }
  void setProjectionsTopMargin(unsigned short Pixels) { projectionsTopMargin = Pixels; }
  void setProjectionsLeftMargin(unsigned short Pixels) { projectionsLeftMargin = Pixels; }
  void setProjectionsRightMargin(unsigned short Pixels) { projectionsRightMargin = Pixels; }
  void setProjectionsBottomMargin(unsigned short Pixels) { projectionsBottomMargin = Pixels; }
  void setPlotsHeight(unsigned short Pixels) { plotsHeight = Pixels; }
  void setPlotsWidth(unsigned short Pixels) { plotsWidth = Pixels; }
  void setPlotsTopMargin(unsigned short Pixels) { plotsTopMargin = Pixels; }
  void setPlotsLeftMargin(unsigned short Pixels) { plotsLeftMargin = Pixels; }
  void setPlotsRightMargin(unsigned short Pixels) { plotsRightMargin = Pixels; }
  void setPlotsBottomMargin(unsigned short Pixels) { plotsBottomMargin = Pixels; }

  // Misc Functions
  void drawResult(TH1 *Summary, bool Vertical = true, bool SetLabels = false);
  void clear() { memset(this,0,sizeof(*this)); }

private:

  // Misc. Data
  std::string name;
  //PlotType type;
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
  unsigned short maxProjectionsX;
  unsigned short maxProjectionsY;
  bool doProjectionsX;
  bool doProjectionsY;

  // Scores/Results
  float ksScore;
  float chi2Score;
  float lowScore;
  float highScore;
  bool result;
  bool isEmpty;
  bool doDrawScores;

  // 2d projections summary settings
  unsigned short projectionsWidth; // set by plotter
  unsigned short projectionsHeight; // user defined
  unsigned short projectionsBarsThickness;
  unsigned short projectionsTopMargin;
  unsigned short projectionsLeftMargin;
  unsigned short projectionsRightMargin;
  unsigned short projectionsBottomMargin;

  // 1d distribution plots settings
  unsigned short plotsWidth; // user defined
  unsigned short plotsHeight; // user defined
  unsigned short plotsTopMargin;
  unsigned short plotsLeftMargin;
  unsigned short plotsRightMargin;
  unsigned short plotsBottomMargin;

  // Attributes of Results Display
  unsigned short passColor, failColor, errorColor;
  unsigned short solidLineColor, solidFillColor, solidFillStyle;
  unsigned short shadedLineColor, shadedFillColor, shadedFillStyle;
  bool lineUseFillColor;

  // Implementation Function
  void initialize();

};

#endif // HISTO_DATA__H
