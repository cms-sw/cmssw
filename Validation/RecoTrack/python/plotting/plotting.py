import sys
import math
import array

import ROOT
ROOT.gROOT.SetBatch(True)
ROOT.PyConfig.IgnoreCommandLineOptions = True

# Flag to indicate if it is ok to have missing files or histograms
# Set to true e.g. if there is no reference histograms
missingOk = False

class AlgoOpt:
    """Class to allow algorithm-specific values for e.g. plot bound values"""
    def __init__(self, default, **kwargs):
        """Constructor.

        Arguments:
        default -- default value

        Keyword arguments are treated as a dictionary where the key is a name of an algorithm, and the value is a value for that algorithm
        """
        self._default = default
        self._values = {}
        self._values.update(kwargs)

    def value(self, algo):
        """Get a value for an algorithm."""
        if algo in self._values:
            return self._values[algo]
        return self._default

def _getObject(tdirectory, name):
    obj = tdirectory.Get(name)
    if not obj:
        print "Did not find {obj} from {dir}".format(obj=name, dir=tdirectory.GetPath())
        if missingOk:
            return None
        else:
            sys.exit(1)
    return obj


def _getYmaxWithError(th1):
    return max([th1.GetBinContent(i)+th1.GetBinError(i) for i in xrange(1, th1.GetNbinsX()+1)])

def _findBounds(th1s, xmin=None, xmax=None, ymin=None, ymax=None):
    """Find x-y axis boundaries encompassing a list of TH1s if the bounds are not given in arguments.

    Arguments:
    th1s -- List of TH1s

    Keyword arguments:
    xmin -- Minimum x value; if None, take the minimum of TH1s
    xmax -- Maximum x value; if None, take the maximum of TH1s
    xmin -- Minimum y value; if None, take the minimum of TH1s
    xmax -- Maximum y value; if None, take the maximum of TH1s
    """
    if xmin is None or xmax is None or ymin is None or ymax is None or isinstance(ymax, list):
        xmins = []
        xmaxs = []
        ymins = []
        ymaxs = []
        for th1 in th1s:
            xaxis = th1.GetXaxis()
            xmins.append(xaxis.GetBinLowEdge(xaxis.GetFirst()))
            xmaxs.append(xaxis.GetBinUpEdge(xaxis.GetLast()))
            ymins.append(th1.GetMinimum())
            ymaxs.append(th1.GetMaximum())
#            ymaxs.append(_getYmaxWithError(th1))

        if xmin is None:
            xmin = min(xmins)
        if xmax is None:
            xmax = max(xmaxs)
        if ymin is None:
            ymin = min(ymins)
        if ymax is None:
            ymax = 1.05*max(ymaxs)
        elif isinstance(ymax, list):
            ym = max(ymaxs)
            ymaxs_above = filter(lambda y: y>ym, ymax)
            if len(ymaxs_above) == 0:
                raise Exception("Histogram maximum y %f is above all given ymax values %s" % (ym, str(ymax)))
            ymax = min(ymaxs_above)

    for th1 in th1s:
        th1.GetXaxis().SetRangeUser(xmin, xmax)
        th1.GetYaxis().SetRangeUser(ymin, ymax)

    return (xmin, ymin, xmax, ymax)


class FakeDuplicate:
    """Class to calculate the fake+duplicate rate"""
    def __init__(self, name, assoc, dup, reco, title=""):
        """Constructor.

        Arguments:
        name  -- String for the name of the resulting efficiency histogram
        assoc -- String for the name of the "associated" histogram
        dup   -- String for the name of the "duplicates" histogram
        reco  -- String for the name of the "reco" (denominator) histogram

        Keyword arguments:
        title  -- String for a title of the resulting histogram (default "")

        The result is calculated as 1 - (assoc - dup) / reco
        """
        self._name = name
        self._assoc = assoc
        self._dup = dup
        self._reco = reco
        self._title = title

    def __str__(self):
        """String representation, returns the name"""
        return self._name

    def create(self, tdirectory):
        """Create and return the efficiency histogram from a TDirectory"""
        # Get the numerator/denominator histograms
        hassoc = _getObject(tdirectory, self._assoc)
        hdup = _getObject(tdirectory, self._dup)
        hreco = _getObject(tdirectory, self._reco)

        # Skip if any of them does not exist
        if not hassoc or not hdup or not hreco:
            return None

        hfakedup = hreco.Clone(self._name)
        hfakedup.SetTitle(self._title)

        for i in xrange(1, hassoc.GetNbinsX()+1):
            numerVal = hassoc.GetBinContent(i) - hdup.GetBinContent(i)
            denomVal = hreco.GetBinContent(i)

            fakedupVal = (1 - numerVal / denomVal) if denomVal != 0.0 else 0.0
            errVal = math.sqrt(fakedupVal*(1-fakedupVal)/denomVal) if (denomVal != 0.0 and fakedupVal <= 1) else 0.0

            hfakedup.SetBinContent(i, fakedupVal)
            hfakedup.SetBinError(i, errVal)

        return hfakedup

class AggregateBins:
    def __init__(self, name, histoName, mapping, normalizeTo=None, scale=None):
        self._name = name
        self._histoName = histoName
        self._mapping = mapping
        self._normalizeTo = normalizeTo
        self._scale = scale

    def __str__(self):
        return self._name

    def create(self, tdirectory):
        th1 = _getObject(tdirectory, self._histoName)
        if th1 is None:
            return None

        result = ROOT.TH1F(self._name, self._name, len(self._mapping), 0, len(self._mapping))

        if isinstance(self._mapping, list):
            for i, label in enumerate(self._mapping):
                bin = th1.GetXaxis().FindBin(label)
                if bin > 0:
                    result.SetBinContent(i+1, th1.GetBinContent(bin))
                result.GetXaxis().SetBinLabel(i+1, label)
        else:
            for i, (key, labels) in enumerate(self._mapping.iteritems()):
                sumTime = 0
                for l in labels:
                    bin = th1.GetXaxis().FindBin(l)
                    if bin > 0:
                        sumTime += th1.GetBinContent(bin)
                result.SetBinContent(i+1, sumTime)
                result.GetXaxis().SetBinLabel(i+1, key)

        if self._normalizeTo is not None:
            bin = th1.GetXaxis().FindBin(self._normalizeTo)
            if bin <= 0:
                print "Trying to normalize {name} to {binlabel}, which does not exist".format(name=self._name, binlabel=self._normalizeTo)
                sys.exit(1)
            value = th1.GetBinContent(bin)
            if value != 0:
                result.Scale(1/value)

        if self._scale is not None:
            result.Scale(self._scale)

        return result

class AggregateHistos:
    def __init__(self, name, mapping, normalizeTo=None):
        self._name = name
        self._mapping = mapping
        self._normalizeTo = normalizeTo

    def __str__(self):
        return self._name

    def create(self, tdirectory):
        result = []
        for key, histoName in self._mapping.iteritems():
            th1 = _getObject(tdirectory, histoName)
            if th1 is None:
                continue
            result.append( (key, th1.Integral(0, th1.GetNbinsX()+1)) ) # include under- and overflow bins
        if len(result) == 0:
            return None

        res = ROOT.TH1F(self._name, self._name, len(result), 0, len(result))

        for i, (name, count) in enumerate(result):
            res.SetBinContent(i+1, count)
            res.GetXaxis().SetBinLabel(i+1, name)

        if self._normalizeTo is not None:
            th1 = _getObject(tdirectory, self._normalizeTo)
            if th1 is None:
                return None
            scale = th1.Integral(0, th1.GetNbinsX()+1)
            res.Scale(1/scale)

        return res

# Plot styles
_plotStylesColor = [4, 2, ROOT.kBlack, ROOT.kOrange+7, ROOT.kMagenta-3]
_plotStylesMarker = [21, 20, 22, 34, 33]

def _drawFrame(pad, bounds, xbinlabels=None, xbinlabelsize=None, xbinlabeloption=None, suffix=""):
    if xbinlabels is None:
        frame = pad.DrawFrame(*bounds)
    else:
        # Special form needed if want to set x axis bin labels
        nbins = len(xbinlabels)
        frame = ROOT.TH1F("hframe"+suffix, "", nbins, bounds[0], bounds[2])
        frame.SetBit(ROOT.TH1.kNoStats)
        frame.SetBit(ROOT.kCanDelete)
        frame.SetMinimum(bounds[1])
        frame.SetMaximum(bounds[3])
        frame.GetYaxis().SetLimits(bounds[1], bounds[3])
        frame.Draw("")

        xaxis = frame.GetXaxis()
        for i in xrange(nbins):
            xaxis.SetBinLabel(i+1, xbinlabels[i])
        if xbinlabelsize is not None:
            xaxis.SetLabelSize(xbinlabelsize)
        if xbinlabeloption is not None:
            frame.LabelsOption(xbinlabeloption)

    return frame

class Frame:
    def __init__(self, pad, bounds, nrows, xbinlabels=None, xbinlabelsize=None, xbinlabeloption=None):
        self._pad = pad
        self._frame = _drawFrame(pad, bounds, xbinlabels, xbinlabelsize, xbinlabeloption)

        yoffsetFactor = 1
        xoffsetFactor = 1
        if nrows == 2:
            yoffsetFactor *= 2
            xoffsetFactor *= 2
        elif nrows == 3:
            yoffsetFactor *= 4
            xoffsetFactor *= 3

        self._frame.GetYaxis().SetTitleOffset(self._frame.GetYaxis().GetTitleOffset()*yoffsetFactor)
        self._frame.GetXaxis().SetTitleOffset(self._frame.GetXaxis().GetTitleOffset()*xoffsetFactor)


    def setLogx(self, log):
        self._pad.SetLogx(log)

    def setLogy(self, log):
        self._pad.SetLogy(log)

    def setGridx(self, grid):
        self._pad.SetGridx(grid)

    def setGridy(self, grid):
        self._pad.SetGridy(grid)

    def setTitle(self, title):
        self._frame.SetTitle(title)

    def setXTitle(self, title):
        self._frame.GetXaxis().SetTitle(title)

    def setYTitle(self, title):
        self._frame.GetYaxis().SetTitle(title)

    def setYTitleSize(self, size):
        self._frame.GetYaxis().SetTitleSize(size)

    def setYTitleOffset(self, offset):
        self._frame.GetYaxis().SetTitleSize(offset)

    def redrawAxis(self):
        self._pad.RedrawAxis()

class FrameRatio:
    def __init__(self, pad, bounds, ratioBounds, ratioFactor, nrows, xbinlabels=None, xbinlabelsize=None, xbinlabeloption=None):
        self._parentPad = pad
        self._pad = pad.cd(1)
        if xbinlabels is not None:
            self._frame = _drawFrame(self._pad, bounds, [""]*len(xbinlabels))
        else:
            self._frame = _drawFrame(self._pad, bounds)
        self._padRatio = pad.cd(2)
        self._frameRatio = _drawFrame(self._padRatio, ratioBounds, xbinlabels, xbinlabelsize, xbinlabeloption)

        self._frame.GetXaxis().SetLabelSize(0)
        self._frame.GetXaxis().SetTitleSize(0)

        yoffsetFactor = ratioFactor
        divisionPoint = 1-1/ratioFactor
        xoffsetFactor = 1/divisionPoint #* 0.6

        if nrows == 1:
            xoffsetFactor *= 0.6
        elif nrows == 2:
            yoffsetFactor *= 2
            xoffsetFactor *= 1.5
        elif nrows == 3:
            yoffsetFactor *= 4
            xoffsetFactor *= 2

        self._frame.GetYaxis().SetTitleOffset(self._frameRatio.GetYaxis().GetTitleOffset()*yoffsetFactor)
        self._frameRatio.GetYaxis().SetLabelSize(int(self._frameRatio.GetYaxis().GetLabelSize()*0.8))
        self._frameRatio.GetYaxis().SetTitleOffset(self._frameRatio.GetYaxis().GetTitleOffset()*yoffsetFactor)
        self._frameRatio.GetXaxis().SetTitleOffset(self._frameRatio.GetXaxis().GetTitleOffset()*xoffsetFactor)

        self._frameRatio.GetYaxis().SetNdivisions(4, 5, 0)

        self._frameRatio.GetYaxis().SetTitle("Ratio")

    def setLogx(self, log):
        self._pad.SetLogx(log)
        self._padRatio.SetLogx(log)

    def setLogy(self, log):
        self._pad.SetLogy(log)

    def setGridx(self, grid):
        self._pad.SetGridx(grid)
        self._padRatio.SetGridx(grid)

    def setGridy(self, grid):
        self._pad.SetGridy(grid)
        self._padRatio.SetGridy(grid)

    def setTitle(self, title):
        self._frame.SetTitle(title)

    def setXTitle(self, title):
        self._frameRatio.GetXaxis().SetTitle(title)

    def setYTitle(self, title):
        self._frame.GetYaxis().SetTitle(title)

    def setYTitleRatio(self, title):
        self._frameRatio.GetYaxis().SetTitle(title)

    def setYTitleSize(self, size):
        self._frame.GetYaxis().SetTitleSize(size)
        self._frameRatio.GetYaxis().SetTitleSize(size)

    def setYTitleOffset(self, offset):
        self._frame.GetYaxis().SetTitleSize(offset)
        self._frameRatio.GetYaxis().SetTitleSize(offset)

    def redrawAxis(self):
        self._padRatio.RedrawAxis()
        self._pad.RedrawAxis()

        self._parentPad.cd()

        # pad to hide the lowest y axis label of the main pad
        xmin=0.065
        ymin=0.285
        xmax=0.128
        ymax=0.33
        self._coverPad = ROOT.TPad("coverpad", "coverpad", xmin, ymin, xmax, ymax)
        self._coverPad.SetBorderMode(0)
        self._coverPad.Draw()

        self._pad.cd()
        self._pad.Pop() # Move the first pad on top

def _copyStyle(src, dst):
    properties = []
    if hasattr(src, "GetLineColor") and hasattr(dst, "SetLineColor"):
        properties.extend(["LineColor", "LineStyle", "LineWidth"])
    if hasattr(src, "GetFillColor") and hasattr(dst, "SetFillColor"):
        properties.extend(["FillColor", "FillStyle"])
    if hasattr(src, "GetMarkerColor") and hasattr(dst, "SetMarkerColor"):
        properties.extend(["MarkerColor", "MarkerSize", "MarkerStyle"])

    for prop in properties:
        getattr(dst, "Set"+prop)(getattr(src, "Get"+prop)())

class Plot:
    """Represents one plot, comparing one or more histograms."""
    def __init__(self, name, **kwargs):
        """ Constructor.

        Arguments:
        name -- String for name of the plot, or Efficiency object

        Keyword arguments:
        title        -- String for a title of the plot (default None)
        xtitle       -- String for x axis title (default None)
        ytitle       -- String for y axis title (default None)
        ytitlesize   -- Float for y axis title size (default None)
        ytitleoffset -- Float for y axis title offset (default None)
        xmin         -- Float for x axis minimum (default None, i.e. automatic)
        xmax         -- Float for x axis maximum (default None, i.e. automatic)
        ymin         -- Float for y axis minimum (default 0)
        ymax         -- Float for y axis maximum (default None, i.e. automatic)
        xlog         -- Bool for x axis log status (default False)
        ylog         -- Bool for y axis log status (default False)
        xgrid        -- Bool for x axis grid status (default True)
        ygrid        -- Bool for y axis grid status (default True)
        stat         -- Draw stat box? (default False)
        fit          -- Do gaussian fit? (default False)
        statx        -- Stat box x coordinate (default 0.65)
        staty        -- Stat box y coordinate (default 0.8)
        statyadjust  -- List of floats for stat box y coordinate adjustments (default None)
        normalizeToUnitArea -- Normalize histograms to unit area? (default False)
        profileX     -- Take histograms via ProfileX()? (default False)
        fitSlicesY   -- Take histograms via FitSlicesY() (default False)
        scale        -- Scale histograms by a number (default None)
        xbinlabels   -- List of x axis bin labels (if given, default None)
        xbinlabelsize -- Size of x axis bin labels (default None)
        xbinlabeloption -- Option string for x axis bin labels (default None)
        drawStyle    -- If "hist", draw as line instead of points (default None)
        drawCommand  -- Deliver this to Draw() (default: None for same as drawStyle)
        lineWidth    -- If drawStyle=="hist", the width of line (default 2)
        legendDx     -- Float for moving TLegend in x direction for separate=True (default None)
        legendDy     -- Float for moving TLegend in y direction for separate=True (default None)
        legendDw     -- Float for changing TLegend width for separate=True (default None)
        legendDh     -- Float for changing TLegend height for separate=True (default None)
        ratioYmin    -- Float for y axis minimum in ratio pad (default 0.9)
        ratioYmax    -- Float for y axis maximum in ratio pad (default 1.1)
        ratioUncertainty -- Plot uncertainties on ratio? (default True)
        histogramModifier -- Function to be called in create() to modify the histograms (default None)
        """
        self._name = name

        def _set(attr, default):
            setattr(self, "_"+attr, kwargs.get(attr, default))

        _set("title", None)
        _set("xtitle", None)
        _set("ytitle", None)
        _set("ytitlesize", None)
        _set("ytitleoffset", None)

        _set("xmin", None)
        _set("xmax", None)
        _set("ymin", 0.)
        _set("ymax", None)

        _set("xlog", False)
        _set("ylog", False)
        _set("xgrid", True)
        _set("ygrid", True)

        _set("stat", False)
        _set("fit", False)

        _set("statx", 0.65)
        _set("staty", 0.8)
        _set("statyadjust", None)

        _set("normalizeToUnitArea", False)
        _set("profileX", False)
        _set("fitSlicesY", False)
        _set("scale", None)
        _set("xbinlabels", None)
        _set("xbinlabelsize", None)
        _set("xbinlabeloption", None)

        _set("drawStyle", None)
        _set("drawCommand", None)
        _set("lineWidth", 2)

        _set("legendDx", None)
        _set("legendDy", None)
        _set("legendDw", None)
        _set("legendDh", None)

        _set("ratioYmin", 0.9)
        _set("ratioYmax", 1.1)
        _set("ratioUncertainty", True)

        _set("histogramModifier", None)

        self._histograms = []

    def getNumberOfHistograms(self):
        """Return number of existing histograms."""
        return len(self._histograms)

    def getName(self):
        return str(self._name)

    def _createOne(self, tdir):
        """Create one histogram from a TDirectory."""
        if tdir == None:
            return None

        # If name is Efficiency instead of string, call its create()
        if hasattr(self._name, "create"):
            th1 = self._name.create(tdir)
        else:
            th1 = tdir.Get(self._name)

        # Check the histogram exists
        if th1 == None:
            print "Did not find {histo} from {dir}".format(histo=self._name, dir=tdir.GetPath())
            if missingOk:
                return None
            else:
                sys.exit(1)

        return th1

    def create(self, tdirs):
        """Create histograms from list of TDirectories"""
        self._histograms = [self._createOne(tdir) for tdir in tdirs]

        if self._histogramModifier is not None:
            self._histograms = self._histogramModifier(self._histograms)

        if len(self._histograms) > len(_plotStylesColor):
            raise Exception("More histograms (%d) than there are plot styles (%d) defined. Please define more plot styles in this file" % (len(self._histograms), len(_plotStylesColor)))

        # Modify histograms here in case self._name returns numbers
        # and self._histogramModifier creates the histograms from
        # these numbers
        def _modifyHisto(th1):
            if th1 is None:
                return None

            if self._profileX:
                th1 = th1.ProfileX()

            if self._fitSlicesY:
                ROOT.TH1.AddDirectory(True)
                th1.FitSlicesY()
                th1 = ROOT.gDirectory.Get(th1.GetName()+"_2")
                th1.SetDirectory(None)
                #th1.SetName(th1.GetName()+"_ref")
                ROOT.TH1.AddDirectory(False)

            if self._title is not None:
                th1.SetTitle(self._title)

            if self._scale is not None:
                th1.Scale(self._scale)

            return th1

        self._histograms = map(_modifyHisto, self._histograms)


    def _setStats(self, startingX, startingY):
        """Set stats box."""
        if not self._stat:
            for h in self._histograms:
                if h is not None and hasattr(h, "SetStats"):
                    h.SetStats(0)
            return

        def _doStats(h, col, dy):
            if h is None:
                return
            h.SetStats(True)

            if self._fit:
                h.Fit("gaus", "Q")
                f = h.GetListOfFunctions().FindObject("gaus")
                if f == None:
                    h.SetStats(0)
                    return
                f.SetLineColor(col)
                f.SetLineWidth(1)
            h.Draw()
            ROOT.gPad.Update()
            st = h.GetListOfFunctions().FindObject("stats")
            if self._fit:
                st.SetOptFit(0010)
                st.SetOptStat(1001)
            st.SetX1NDC(startingX)
            st.SetX2NDC(startingX+0.3)
            st.SetY1NDC(startingY+dy)
            st.SetY2NDC(startingY+dy+0.15)
            st.SetTextColor(col)

        dy = 0.0
        for i, h in enumerate(self._histograms):
            if self._statyadjust is not None and i < len(self._statyadjust):
                dy += self._statyadjust[i]

            _doStats(h, _plotStylesColor[i], dy)
            dy -= 0.19

    def _normalize(self):
        """Normalise histograms to unit area"""

        for h in self._histograms:
            if h is None:
                continue
            i = h.Integral()
            if i == 0:
                continue
            h.Scale(1.0/i)

    def draw(self, algo, pad, ratio, ratioFactor, nrows):
        """Draw the histograms using values for a given algorithm."""
#        if len(self._histograms) == 0:
#            print "No histograms for plot {name}".format(name=self._name)
#            return

        if self._normalizeToUnitArea:
            self._normalize()

        def _styleMarker(h, msty, col):
            h.SetMarkerStyle(msty)
            h.SetMarkerColor(col)
            h.SetMarkerSize(0.7)
            h.SetLineColor(1)
            h.SetLineWidth(1)

        def _styleHist(h, msty, col):
            _styleMarker(h, msty, col)
            h.SetLineColor(col)
            h.SetLineWidth(self._lineWidth)

        # Use marker or hist style
        style = _styleMarker
        if self._drawStyle is not None:
            if "hist" in self._drawStyle.lower():
                style = _styleHist
            if len(self._histograms) > 0 and isinstance(self._histograms[0], ROOT.TGraph):
                if "l" in self._drawStyle.lower():
                    style = _styleHist

        # Apply style to histograms, filter out Nones
        histos = []
        for i, h in enumerate(self._histograms):
            if h is None:
                continue
            style(h, _plotStylesMarker[i], _plotStylesColor[i])
            histos.append(h)
        if len(histos) == 0:
            print "No histograms for plot {name}".format(name=self._name)
            return

        # Return value if number, or algo-specific value if AlgoOpt
        def _getVal(val):
            if hasattr(val, "value"):
                return val.value(algo)
            return val

        bounds = _findBounds(histos,
                             xmin=_getVal(self._xmin), xmax=_getVal(self._xmax),
                             ymin=_getVal(self._ymin), ymax=_getVal(self._ymax))

        # Create bounds before stats in order to have the
        # SetRangeUser() calls made before the fit
        #
        # stats is better to be called before frame, otherwise get
        # mess in the plot (that frame creation cleans up)
        if ratio:
            pad.cd(1)
        self._setStats(self._statx, self._staty)

        xbinlabels = self._xbinlabels
        if xbinlabels is None:
            if len(histos[0].GetXaxis().GetBinLabel(1)) > 0:
                xbinlabels = []
                for i in xrange(1, histos[0].GetNbinsX()+1):
                    xbinlabels.append(histos[0].GetXaxis().GetBinLabel(i))

        # Create frame
        if ratio:
            ratioBounds = (bounds[0], self._ratioYmin, bounds[2], self._ratioYmax)
            frame = FrameRatio(pad, bounds, ratioBounds, ratioFactor, nrows, self._xbinlabels, self._xbinlabelsize, self._xbinlabeloption)
        else:
            frame = Frame(pad, bounds, nrows, self._xbinlabels, self._xbinlabelsize, self._xbinlabeloption)

        # Set log and grid
        frame.setLogx(self._xlog)
        frame.setLogy(self._ylog)
        frame.setGridx(self._xgrid)
        frame.setGridy(self._ygrid)

        # Set properties of frame
        frame.setTitle(histos[0].GetTitle())
        if self._xtitle is not None:
            frame.setXTitle(self._xtitle)
        if self._ytitle is not None:
            frame.setYTitle(self._ytitle)
        if self._ytitlesize is not None:
            frame.setYTitleSize(self._ytitlesize)
        if self._ytitleoffset is not None:
            frame.setTitleOffset(self._ytitleoffset)

        if ratio:
            frame._pad.cd()

        # Draw histograms
        opt = "sames" # s for statbox or something?
        ds = ""
        if self._drawStyle is not None:
            ds = self._drawStyle
        if self._drawCommand is not None:
            ds = self._drawCommand
        if len(ds) > 0:
            opt += " "+ds

        if ratio:
            frame._pad.cd()

        for h in histos:
            h.Draw(opt)

        # Draw ratios
        if ratio and len(histos) > 0:
            frame._padRatio.cd()
            self._ratios = self._calculateRatios(histos) # need to keep these in memory too ...
            if self._ratioUncertainty:
                self._ratios[0]._ratio.SetFillStyle(1001)
                self._ratios[0]._ratio.SetFillColor(ROOT.kGray)
                self._ratios[0]._ratio.SetLineColor(ROOT.kGray)
                self._ratios[0]._ratio.SetMarkerColor(ROOT.kGray)
                self._ratios[0]._ratio.SetMarkerSize(0)
                self._ratios[0].draw("E2")
                frame._padRatio.RedrawAxis("G") # redraw grid on top of the uncertainty of denominator
            for r in self._ratios[1:]:
                r.draw()

        frame.redrawAxis()
        self._frame = frame # keep the frame in memory for sure

    def addToLegend(self, legend, legendLabels):
        """Add histograms to a legend.

        Arguments:
        legend       -- TLegend
        legendLabels -- List of strings for the legend labels
        """
        for h, label in zip(self._histograms, legendLabels):
            if h is None:
                continue
            legend.AddEntry(h, label, "LP")

    def _calculateRatios(self, histos):
        def _divideOrZero(numerator, denominator):
            if denominator == 0:
                return 0
            return numerator/denominator
        class WrapTH1:
            def __init__(self, th1, uncertainty):
                self._th1 = th1
                self._uncertainty = uncertainty

                xaxis = th1.GetXaxis()
                xaxis_arr = xaxis.GetXbins()
                if xaxis_arr.GetSize() > 0: # unequal binning
                    lst = [xaxis_arr[i] for i in xrange(0, xaxis_arr.GetSize())]
                    arr = array.array("d", lst)
                    self._ratio = ROOT.TH1F("foo", "foo", xaxis.GetNbins(), arr)
                else:
                    self._ratio = ROOT.TH1F("foo", "foo", xaxis.GetNbins(), xaxis.GetXmin(), xaxis.GetXmax())
                _copyStyle(th1, self._ratio)
                self._ratio.SetStats(0)
                self._ratio.SetLineColor(ROOT.kBlack)
                self._ratio.SetLineWidth(1)
            def draw(self, style=None):
                st = style
                if st is None:
                    if self._uncertainty:
                        st = "EP"
                    else:
                        st = "HIST P"
                self._ratio.Draw("same "+st)
            def begin(self):
                return 1
            def end(self):
                return self._th1.GetNbinsX()+1
            def xvalues(self, bin):
                xval = self._th1.GetBinCenter(bin)
                xlow = xval-self._th1.GetXaxis().GetBinLowEdge(bin)
                xhigh = self._th1.GetXaxis().GetBinUpEdge(bin)-xval
                return (xval, xlow, xhigh)
            def yvalues(self, bin):
                yval = self._th1.GetBinContent(bin)
                yerr = self._th1.GetBinError(bin)
                return (yval, yerr, yerr)
            def y(self, bin):
                return self._th1.GetBinContent(bin)
            def divide(self, bin, scale, xcenter):
                self._ratio.SetBinContent(bin, _divideOrZero(self._th1.GetBinContent(bin), scale))
                self._ratio.SetBinError(bin, _divideOrZero(self._th1.GetBinError(bin), scale))

        class WrapTGraph:
            def __init__(self, gr, uncertainty):
                self._gr = gr
                self._uncertainty
                self._xvalues = []
                self._xerrslow = []
                self._xerrshigh = []
                self._yvalues = []
                self._yerrshigh = []
                self._yerrslow = []
                self._binOffset = 0
            def draw(self, style=None):
                if len(self.xvalues) == 0:
                    return
                st = style
                if st is None:
                    if self._uncertainty:
                        st = "PZ"
                    else:
                        st = "PX"
                self._ratio = ROOT.TGraphAsymmErrors(len(self.xvalues), array.array("d", self.xvalues), array.array("d", self.yvalues),
                                                     array.array("d", self.xerrslow), array.array("d", self.xerrshigh), 
                                                     array.array("d", self.yerrslow), array.array("d", self.yerrshigh))
                self._ratio.Draw("same "+st)
            def begin(self):
                return 0
            def end(self):
                return self._gr.GetN()
            def xvalues(self, bin):
                return (self._gr.GetX()[bin], self._gr.GetErrorXlow(bin), self._gr.GetErrorXhigh(bin))
            def yvalues(self, bin):
                return (self._gr.GetY()[bin], self._gr.GetErrorYlow(bin), self._gr.GetErrorYhigh(bin))
            def y(self, bin):
                return self._gr.GetY()[bin]
            def divide(self, bin, scale, xcenter):
                # Ignore bin if denominator is zero
                if scale == 0:
                    return
                # No more items in the numerator
                if bin >= self._gr.GetN():
                    return
                # denominator is missing an item
                trueBin = bin + self.binOffset
                xval = self._gr.GetX()[trueBin]
                epsilon = 1e-3 * xval # to allow floating-point difference between TGraph and TH1
                if xval+epsilon < xcenter:
                    self.binOffset -= 1
                    return
                # numerator is missing an item
                elif xval-epsilon > xcenter:
                    self.binOffset += 1
                    return

                self.xvalues.append(xval)
                self.xerrslow.append(self._gr.GetErrorXlow(trueBin))
                self.xerrshigh.append(self._gr.GetErrorXhigh(trueBin))
                self.yvalues.append(self._gr.GetY()[trueBin] / scale)
                if self._uncertainty:
                    self.yerrslow.append(self._gr.GetErrorYlow(trueBin) / scale)
                    self.yerrshigh.append(self._gr.GetErrorYhigh(trueBin) / scale)
                else:
                    self.yerrslow.append(0)
                    self.yerrshigh.append(0)

        def wrap(o):
            if isinstance(o, ROOT.TH1):
                return WrapTH1(o, self._ratioUncertainty)
            elif isinstance(o, ROOT.TGrapgh):
                return WrapTGraph(o, self._ratioUncertainty)

        wrappers = [wrap(h) for h in histos]
        ref = wrappers[0]

        for bin in xrange(ref.begin(), ref.end()):
            (scale, ylow, yhigh) = ref.yvalues(bin)
            (xval, xlow, xhigh) = ref.xvalues(bin)
            for w in wrappers:
                w.divide(bin, scale, xval)

        return wrappers

class PlotGroup:
    """Group of plots, results a TCanvas"""
    def __init__(self, name, plots, **kwargs):
        """Constructor.

        Arguments:
        name  -- String for name of the TCanvas, used also as the basename of the picture files
        plots -- List of Plot objects

        Keyword arguments:
        legendDx -- Float for moving TLegend in x direction (default None)
        legendDy -- Float for moving TLegend in y direction (default None)
        legendDw -- Float for changing TLegend width (default None)
        legendDh -- Float for changing TLegend height (default None)
        overrideLegendLabels -- List of strings for legend labels, if given, these are used instead of the ones coming from Plotter (default None)
        """
        self._name = name
        self._plots = plots

        def _set(attr, default):
            setattr(self, "_"+attr, kwargs.get(attr, default))

        _set("legendDx", None)
        _set("legendDy", None)
        _set("legendDw", None)
        _set("legendDh", None)

        _set("overrideLegendLabels", None)

        self._ratioFactor = 1.25

    def create(self, tdirectories):
        """Create histograms from a list of TDirectories."""
        for plot in self._plots:
            plot.create(tdirectories)

    def draw(self, algo, legendLabels, prefix=None, separate=False, saveFormat=".pdf", ratio=False):
        """Draw the histograms using values for a given algorithm.

        Arguments:
        algo          -- string for algorithm
        legendLabels  -- List of strings for legend labels (corresponding to the tdirectories in create())
        prefix        -- Optional string for file name prefix (default None)
        separate      -- Save the plots of a group to separate files instead of a file per group (default False)
        saveFormat   -- String specifying the plot format (default '.pdf')
        ratio        -- Add ratio to the plot (default False)
        """

        if self._overrideLegendLabels is not None:
            legendLabels = self._overrideLegendLabels

        if separate:
            return self._drawSeparate(algo, legendLabels, prefix, saveFormat, ratio)

        cwidth = 1000
        nrows = int((len(self._plots)+1)/2) # this should work also for odd n
        cheight = 500 * nrows

        if ratio:
            cheight = int(cheight*self._ratioFactor)

        canvas = ROOT.TCanvas(self._name, self._name, cwidth, cheight)

        canvas.Divide(2, nrows)
        if ratio:
            for i in xrange(0, len(self._plots)):
                pad = canvas.cd(i+1)
                self._modifyPadForRatio(pad)

        # Draw plots to canvas
        for i, plot in enumerate(self._plots):
            pad = canvas.cd(i+1)
            plot.draw(algo, pad, ratio, self._ratioFactor, nrows)

        # Setup legend
        canvas.cd()
        if len(self._plots) <= 4:
            lx1 = 0.2
            lx2 = 0.9
            ly1 = 0.48
            ly2 = 0.53
        else:
            lx1 = 0.1
            lx2 = 0.9
            ly1 = 0.64
            ly2 = 0.69
        if self._legendDx is not None:
            lx1 += self._legendDx
            lx2 += self._legendDx
        if self._legendDy is not None:
            ly1 += self._legendDy
            ly2 += self._legendDy
        if self._legendDw is not None:
            lx2 += self._legendDw
        if self._legendDh is not None:
            ly1 -= self._legendDh
        plot = max(self._plots, key=lambda p: p.getNumberOfHistograms())
        legend = self._createLegend(plot, legendLabels, lx1, ly1, lx2, ly2)

        return self._save(canvas, saveFormat, prefix=prefix)

    def _drawSeparate(self, algo, legendLabels, prefix, saveFormat, ratio):
        width = 500
        height = 500
        if ratio:
            height = int(height*self._ratioFactor)

        canvas = ROOT.TCanvas(self._name+"Single", self._name, width, height)
        # from TDRStyle
        canvas.SetTopMargin(0.05)
        canvas.SetBottomMargin(0.13)
        canvas.SetLeftMargin(0.16)
        canvas.SetRightMargin(0.05)

        lx1def = 0.6
        lx2def = 0.95
        ly1def = 0.85
        ly2def = 0.95

        ret = []

        for plot in self._plots:
            if ratio:
                canvas.cd()
                self._modifyPadForRatio(canvas)

            # Draw plot to canvas
            canvas.cd()
            plot.draw(algo, canvas, ratio, self._ratioFactor, 1)


            # Setup legend
            lx1 = lx1def
            lx2 = lx2def
            ly1 = ly1def
            ly2 = ly2def

            if plot._legendDx is not None:
                lx1 += plot._legendDx
                lx2 += plot._legendDx
            if plot._legendDy is not None:
                ly1 += plot._legendDy
                ly2 += plot._legendDy
            if plot._legendDw is not None:
                lx2 += plot._legendDw
            if plot._legendDh is not None:
                ly1 -= plot._legendDh

            canvas.cd()
            legend = self._createLegend(plot, legendLabels, lx1, ly1, lx2, ly2, textSize=0.03)

            ret.extend(self._save(canvas, saveFormat, prefix=prefix, postfix="_"+plot.getName(), single=True))
        return ret

    def _modifyPadForRatio(self, pad):
        pad.Divide(1, 2)

        divisionPoint = 1-1/self._ratioFactor

        topMargin = pad.GetTopMargin()
        bottomMargin = pad.GetBottomMargin()
        divisionPoint += (1-divisionPoint)*bottomMargin # correct for (almost-)zeroing bottom margin of pad1
        divisionPointForPad1 = 1-( (1-divisionPoint) / (1-0.02) ) # then correct for the non-zero bottom margin, but for pad1 only

        # Set the lower point of the upper pad to divisionPoint
        pad1 = pad.cd(1)
        yup = 1.0
        ylow = divisionPointForPad1
        xup = 1.0
        xlow = 0.0
        pad1.SetPad(xlow, ylow, xup, yup)
        pad1.SetFillStyle(4000) # transparent
        pad1.SetBottomMargin(0.02) # need some bottom margin here for eps/pdf output (at least in ROOT 5.34)

        # Set the upper point of the lower pad to divisionPoint
        pad2 = pad.cd(2)
        yup = divisionPoint
        ylow = 0.0
        pad2.SetPad(xlow, ylow, xup, yup)
        pad2.SetFillStyle(4000) # transparent
        pad2.SetTopMargin(0.0)
        pad2.SetBottomMargin(bottomMargin/(self._ratioFactor*divisionPoint))

    def _createLegend(self, plot, legendLabels, lx1, ly1, lx2, ly2, textSize=0.016):
        l = ROOT.TLegend(lx1, ly1, lx2, ly2)
        l.SetTextSize(textSize)
        l.SetLineColor(1)
        l.SetLineWidth(1)
        l.SetLineStyle(1)
        l.SetFillColor(0)
        l.SetMargin(0.07)

        plot.addToLegend(l, legendLabels)
        l.Draw()
        return l

    def _save(self, canvas, saveFormat, prefix=None, postfix=None, single=False):
        # Save the canvas to file and clear
        name = self._name
        if prefix is not None:
            name = prefix+name
        if postfix is not None:
            name = name+postfix
        canvas.SaveAs(name+saveFormat)
        if single:
            canvas.Clear()
            canvas.SetLogx(False)
            canvas.SetLogy(False)
        else:
            canvas.Clear("D") # keep subpads

        return [name+saveFormat]


class Plotter:
    """Represent a collection of PlotGroups."""
    def __init__(self, possibleDirs, plotGroups, saveFormat=".pdf"):
        """Constructor.

        Arguments:
        possibleDirs -- List of strings for possible directories of histograms in TFiles
        plotGroups   -- List of PlotGroup objects
        saveFormat   -- String specifying the plot format (default '.pdf')
        """
        self._possibleDirs = possibleDirs
        self._plotGroups = plotGroups
        self._saveFormat = saveFormat

        _absoluteSize = True
        if _absoluteSize:
            font = 43
            titleSize = 22
            labelSize = 22
            statSize = 14
        else:
            font = 42
            titleSize = 0.05
            labelSize = 0.05
            statSize = 0.025

        ROOT.gROOT.SetStyle("Plain")
        ROOT.gStyle.SetPadRightMargin(0.07)
        ROOT.gStyle.SetPadLeftMargin(0.13)
        ROOT.gStyle.SetTitleFont(font, "XYZ")
        ROOT.gStyle.SetTitleSize(titleSize, "XYZ")
        ROOT.gStyle.SetTitleOffset(1.2, "Y")
        #ROOT.gStyle.SetTitleFontSize(0.05)
        ROOT.gStyle.SetLabelFont(font, "XYZ")
        ROOT.gStyle.SetLabelSize(labelSize, "XYZ")
        ROOT.gStyle.SetTextSize(labelSize)
        ROOT.gStyle.SetStatFont(font)
        ROOT.gStyle.SetStatFontSize(statSize)

        ROOT.TH1.AddDirectory(False)

    def setPossibleDirectoryNames(self, possibleDirs):
        self._possibleDirs = possibleDirs

    def getPossibleDirectoryNames(self):
        """Return the list of possible directory names."""
        return self._possibleDirs

    def append(self, plotGroup):
        self._plotGroups.append(plotGroup)

    def set(self, plotGroups):
        self._plotGroups = plotGroups

    def _getDir(self, tfile, subdir):
        """Get TDirectory from TFile."""
        if tfile is None:
            return None
        for pd in self._possibleDirs:
            d = tfile.Get(pd)
            if d:
                if subdir is not None:
                    # Pick associator if given
                    d = d.Get(subdir)
                    if d:
                        return d
                    else:
                        msg = "Did not find subdirectory '%s' from directory '%s' in file %s" % (subdir, pd, tfile.GetName())
                        if missingOk:
                            print msg
                            return None
                        else:
                            raise Exception(msg)
                else:
                    return d
        msg = "Did not find any of directories '%s' from file %s" % (",".join(self._possibleDirs), tfile.GetName())
        if missingOk:
            print msg
            return None
        else:
            raise Exception(msg)

    def create(self, files, labels, subdir=None):
        """Create histograms from a list of TFiles.

        Arguments:
        files  -- List of TFiles
        labels -- List of strings for legend labels corresponding the files
        subdir -- Optional string for subdirectory inside the possibleDirs; if list of strings, then each corresponds to a TFile
        """
        dirs = []
        self._labels = []
        if isinstance(subdir, list):
            for f, l, s in zip(files, labels, subdir):
                d = self._getDir(f, s)
                dirs.append(d)
                self._labels.append(l)
        else:
            for f, l in zip(files, labels):
                d = self._getDir(f, subdir)
                dirs.append(d)
                self._labels.append(l)

        for pg in self._plotGroups:
            pg.create(dirs)

    def draw(self, algo, prefix=None, separate=False, saveFormat=None, ratio=False):
        """Draw and save all plots using settings of a given algorithm.

        Arguments:
        algo     -- String for the algorithm
        prefix   -- Optional string for file name prefix (default None)
        separate -- Save the plots of a group to separate files instead of a file per group (default False)
        saveFormat -- If given, overrides the saveFormat
        ratio    -- Add ratio to the plot (default False)
        """
        ret = []

        sf = self._saveFormat
        if saveFormat is not None:
            sf = saveFormat

        for pg in self._plotGroups:
            ret.extend(pg.draw(algo, self._labels, prefix=prefix, separate=separate, saveFormat=sf, ratio=ratio))
        return ret

