import sys
import math
import array

import ROOT
ROOT.gROOT.SetBatch(True)
ROOT.PyConfig.IgnoreCommandLineOptions = True

def _getObject(tdirectory, name):
    obj = tdirectory.Get(name)
    if not obj:
        print "Did not find {obj} from {dir}".format(obj=name, dir=tdirectory.GetPath())
        return None
    return obj

def _getOrCreateObject(tdirectory, nameOrCreator):
    if hasattr(nameOrCreator, "create"):
        return nameOrCreator.create(tdirectory)
    return _getObject(tdirectory, nameOrCreator)

def _getXmin(obj):
    if isinstance(obj, ROOT.TH1):
        xaxis = obj.GetXaxis()
        return xaxis.GetBinLowEdge(xaxis.GetFirst())
    elif isinstance(obj, ROOT.TGraph) or isinstance(obj, ROOT.TGraph2D):
        return min([obj.GetX()[i] for i in xrange(0, obj.GetN())])*0.9
    raise Exception("Unsupported type %s" % str(obj))

def _getXmax(obj):
    if isinstance(obj, ROOT.TH1):
        xaxis = obj.GetXaxis()
        return xaxis.GetBinUpEdge(xaxis.GetLast())
    elif isinstance(obj, ROOT.TGraph) or isinstance(obj, ROOT.TGraph2D):
        return max([obj.GetX()[i] for i in xrange(0, obj.GetN())])*1.02
    raise Exception("Unsupported type %s" % str(obj))

def _getYmin(obj):
    if isinstance(obj, ROOT.TH1):
        return obj.GetMinimum()
    elif isinstance(obj, ROOT.TGraph) or isinstance(obj, ROOT.TGraph2D):
        return min([obj.GetY()[i] for i in xrange(0, obj.GetN())])
    raise Exception("Unsupported type %s" % str(obj))

def _getYmax(obj):
    if isinstance(obj, ROOT.TH1):
        return obj.GetMaximum()
    elif isinstance(obj, ROOT.TGraph) or isinstance(obj, ROOT.TGraph2D):
        return max([obj.GetY()[i] for i in xrange(0, obj.GetN())])
    raise Exception("Unsupported type %s" % str(obj))

def _getYmaxWithError(th1):
    return max([th1.GetBinContent(i)+th1.GetBinError(i) for i in xrange(1, th1.GetNbinsX()+1)])

def _getYminIgnoreOutlier(th1):
    yvals = filter(lambda n: n>0, [th1.GetBinContent(i) for i in xrange(1, th1.GetNbinsX()+1)])
    yvals.sort()
    if len(yvals) == 0:
        return th1.GetMinimum()
    if len(yvals) == 1:
        return yvals[0]

    # Define outlier as being x10 less than minimum of the 95 % of the non-zero largest values
    ind_min = len(yvals)-1 - int(len(yvals)*0.95)
    min_val = yvals[ind_min]
    for i in xrange(0, ind_min):
        if yvals[i] > 0.1*min_val:
            return yvals[i]

    return min_val

def _findBounds(th1s, ylog, xmin=None, xmax=None, ymin=None, ymax=None):
    """Find x-y axis boundaries encompassing a list of TH1s if the bounds are not given in arguments.

    Arguments:
    th1s -- List of TH1s
    ylog -- Boolean indicating if y axis is in log scale or not (affects the automatic ymax)

    Keyword arguments:
    xmin -- Minimum x value; if None, take the minimum of TH1s
    xmax -- Maximum x value; if None, take the maximum of TH1s
    xmin -- Minimum y value; if None, take the minimum of TH1s
    xmax -- Maximum y value; if None, take the maximum of TH1s
    """

    def y_scale_max(y):
        if ylog:
            return 1.5*y
        return 1.05*y

    def y_scale_min(y):
        # assuming log
        return 0.9*y

    if xmin is None or xmax is None or ymin is None or ymax is None or isinstance(ymin, list) or isinstance(ymax, list):
        xmins = []
        xmaxs = []
        ymins = []
        ymaxs = []
        for th1 in th1s:
            xmins.append(_getXmin(th1))
            xmaxs.append(_getXmax(th1))
            if ylog and isinstance(ymin, list):
                ymins.append(_getYminIgnoreOutlier(th1))
            else:
                ymins.append(_getYmin(th1))
            ymaxs.append(_getYmax(th1))
#            ymaxs.append(_getYmaxWithError(th1))

        if xmin is None:
            xmin = min(xmins)
        if xmax is None:
            xmax = max(xmaxs)
        if ymin is None:
            ymin = min(ymins)
        elif isinstance(ymin, list):
            ym_unscaled = min(ymins)
            ym = y_scale_min(ym_unscaled)
            ymins_below = filter(lambda y: y<=ym, ymin)
            if len(ymins_below) == 0:
                ymin = min(ymin)
                if ym_unscaled < ymin:
                    print "Histogram minimum y %f is below all given ymin values %s, using the smallest one" % (ym, str(ymin))
            else:
                ymin = max(ymins_below)

        if ymax is None:
            ymax = y_scale_max(max(ymaxs))
        elif isinstance(ymax, list):
            ym_unscaled = max(ymaxs)
            ym = y_scale_max(ym_unscaled)
            ymaxs_above = filter(lambda y: y>ym, ymax)
            if len(ymaxs_above) == 0:
                ymax = max(ymax)
                if ym_unscaled > ymax:
                    print "Histogram maximum y %f is above all given ymax values %s, using the maximum one" % (ym_unscaled, str(ymax))
            else:
                ymax = min(ymaxs_above)


    for th1 in th1s:
        th1.GetXaxis().SetRangeUser(xmin, xmax)
        th1.GetYaxis().SetRangeUser(ymin, ymax)

    return (xmin, ymin, xmax, ymax)

class Subtract:
    """Class for subtracting two histograms"""
    def __init__(self, name, nameA, nameB, title=""):
        """Constructor

        Arguments:
        name  -- String for name of the resulting histogram (A-B)
        nameA -- String for A histogram
        nameB -- String for B histogram

        Keyword arguments:
        title -- String for a title of the resulting histogram (default "")

        Uncertainties are calculated with the assumption that B is a
        subset of A, and the histograms contain event counts.
        """
        self._name = name
        self._nameA = nameA
        self._nameB = nameB
        self._title = title

    def __str__(self):
        """String representation, returns the name"""
        return self._name

    def create(self, tdirectory):
        """Create and return the fake+duplicate histogram from a TDirectory"""
        histoA = _getObject(tdirectory, self._nameA)
        histoB = _getObject(tdirectory, self._nameB)

        if not histoA or not histoB:
            return None

        ret = histoA.Clone(self._name)
        ret.SetTitle(self._title)

        for i in xrange(1, histoA.GetNbinsX()+1):
            val = histoA.GetBinContent(i)-histoB.GetBinContent(i)
            ret.SetBinContent(i, val)
            ret.SetBinError(i, math.sqrt(val))

        return ret

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
        """Create and return the fake+duplicate histogram from a TDirectory"""
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
    """Class to create a histogram by aggregating bins of another histogram to a bin of the resulting histogram."""
    def __init__(self, name, histoName, mapping, normalizeTo=None, scale=None, renameBin=None):
        """Constructor.

        Arguments:
        name      -- String for the name of the resulting histogram
        histoName -- String for the name of the source histogram
        mapping   -- Dictionary or list for mapping the bins (see below)

        Keyword arguments:
        normalizeTo -- Optional string of a bin label in the source histogram. If given, all bins of the resulting histogram are divided by the value of this bin.
        scale       -- Optional number for scaling the histogram (passed to ROOT.TH1.Scale())
        renameBin   -- Optional function (string -> string) to rename the bins of the input histogram

        Mapping structure (mapping):

        Dictionary (you probably want to use collections.OrderedDict)
        should be a mapping from the destination bin label to a list
        of source bin labels ("dst -> [src]").

        If the mapping is a list, it should only contain the source
        bin labels. In this case, the resulting histogram contains a
        subset of the bins of the source histogram.
        """
        self._name = name
        self._histoName = histoName
        self._mapping = mapping
        self._normalizeTo = normalizeTo
        self._scale = scale
        self._renameBin = renameBin

    def __str__(self):
        """String representation, returns the name"""
        return self._name

    def create(self, tdirectory):
        """Create and return the histogram from a TDirectory"""
        th1 = _getOrCreateObject(tdirectory, self._histoName)
        if th1 is None:
            return None

        result = ROOT.TH1F(self._name, self._name, len(self._mapping), 0, len(self._mapping))

        # TH1 can't really be used as a map/dict, so convert it here:
        values = {}
        for i in xrange(1, th1.GetNbinsX()+1):
            binLabel = th1.GetXaxis().GetBinLabel(i)
            if self._renameBin is not None:
                binLabel = self._renameBin(binLabel)
            values[binLabel] = (th1.GetBinContent(i), th1.GetBinError(i))

        if isinstance(self._mapping, list):
            for i, label in enumerate(self._mapping):
                try:
                    result.SetBinContent(i+1, values[label][0])
                    result.SetBinError(i+1, values[label][1])
                except KeyError:
                    pass
                result.GetXaxis().SetBinLabel(i+1, label)
        else:
            for i, (key, labels) in enumerate(self._mapping.iteritems()):
                sumTime = 0
                sumErrorSq = 0
                for l in labels:
                    try:
                        sumTime += values[l][0]
                        sumErrorSq += values[l][1]**2
                    except KeyError:
                        pass
                result.SetBinContent(i+1, sumTime)
                result.SetBinError(i+1, math.sqrt(sumErrorSq))
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
    """Class to create a histogram by aggregaging integrals of another histoggrams."""
    def __init__(self, name, mapping, normalizeTo=None):
        """Constructor.

        Arguments:
        name    -- String for the name of the resulting histogram
        mapping -- Dictionary for mapping the bin label to a histogram name

        Keyword arguments:
        normalizeTo -- Optional string for a histogram. If given, all bins of the resulting histograqm are divided by the integral of this histogram.
        """
        self._name = name
        self._mapping = mapping
        self._normalizeTo = normalizeTo

    def __str__(self):
        """String representation, returns the name"""
        return self._name

    def create(self, tdirectory):
        """Create and return the histogram from a TDirectory"""
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

class ROC:
    """Class to construct a ROC curve (e.g. efficiency vs. fake rate) from two histograms"""
    def __init__(self, name, xhistoName, yhistoName, zaxis=False):
        """Constructor.

        Arguments:
        name       -- String for the name of the resulting histogram
        xhistoName -- String for the name of the x-axis histogram (or another "creator" object)
        yhistoName -- String for the name of the y-axis histogram (or another "creator" object)

        Keyword arguments:
        zaxis -- If set to True (default False), create a TGraph2D with z axis showing the cut value (recommended drawStyle 'pcolz')
        """
        self._name = name
        self._xhistoName = xhistoName
        self._yhistoName = yhistoName
        self._zaxis = zaxis

    def __str__(self):
        """String representation, returns the name"""
        return self._name

    def create(self, tdirectory):
        """Create and return the histogram from a TDirectory"""
        xhisto = _getOrCreateObject(tdirectory, self._xhistoName)
        yhisto = _getOrCreateObject(tdirectory, self._yhistoName);
        if xhisto is None or yhisto is None:
            return None

        x = []
        xerrup = []
        xerrdown = []
        y = []
        yerrup = []
        yerrdown = []
        z = []

        for i in xrange(1, xhisto.GetNbinsX()+1):
            x.append(xhisto.GetBinContent(i))
            xerrup.append(xhisto.GetBinError(i))
            xerrdown.append(xhisto.GetBinError(i))

            y.append(yhisto.GetBinContent(i))
            yerrup.append(yhisto.GetBinError(i))
            yerrdown.append(yhisto.GetBinError(i))

            z.append(xhisto.GetXaxis().GetBinUpEdge(i))

        arr = lambda v: array.array("d", v)
        gr = None
        if self._zaxis:
            gr = ROOT.TGraph2D(len(x), arr(x), arr(y), arr(z))
        else:
            gr = ROOT.TGraphAsymmErrors(len(x), arr(x), arr(y), arr(xerrdown), arr(xerrup), arr(yerrdown), arr(yerrup))
        gr.SetTitle("")
        return gr


# Plot styles
_plotStylesColor = [4, 2, ROOT.kBlack, ROOT.kOrange+7, ROOT.kMagenta-3]
_plotStylesMarker = [21, 20, 22, 34, 33]

def _drawFrame(pad, bounds, xbinlabels=None, xbinlabelsize=None, xbinlabeloption=None, suffix=""):
    """Function to draw a frame

    Arguments:
    pad    -- TPad to where the frame is drawn
    bounds -- List or 4-tuple for (xmin, ymin, xmax, ymax)

    Keyword arguments:
    xbinlabels      -- Optional list of strings for x axis bin labels
    xbinlabelsize   -- Optional number for the x axis bin label size
    xbinlabeloption -- Optional string for the x axis bin options (passed to ROOT.TH1.LabelsOption())
    suffix          -- Optional string for a postfix of the frame name
    """
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
    """Class for creating and managing a frame for a simple, one-pad plot"""
    def __init__(self, pad, bounds, nrows, xbinlabels=None, xbinlabelsize=None, xbinlabeloption=None):
        self._pad = pad
        self._frame = _drawFrame(pad, bounds, xbinlabels, xbinlabelsize, xbinlabeloption)

        yoffsetFactor = 1
        xoffsetFactor = 1
        if nrows == 2:
            yoffsetFactor *= 2
            xoffsetFactor *= 2
        elif nrows >= 3:
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

    def adjustMarginLeft(self, adjust):
        self._pad.SetLeftMargin(self._pad.GetLeftMargin()+adjust)
        # Need to redraw frame after adjusting the margin
        self._pad.cd()
        self._frame.Draw("")

    def adjustMarginRight(self, adjust):
        self._pad.SetRightMargin(self._pad.GetRightMargin()+adjust)
        # Need to redraw frame after adjusting the margin
        self._pad.cd()
        self._frame.Draw("")

    def setTitle(self, title):
        self._frame.SetTitle(title)

    def setXTitle(self, title):
        self._frame.GetXaxis().SetTitle(title)

    def setXTitleSize(self, size):
        self._frame.GetXaxis().SetTitleSize(size)

    def setXTitleOffset(self, offset):
        self._frame.GetXaxis().SetTitleOffset(offset)

    def setXLabelSize(self, size):
        self._frame.GetXaxis().SetLabelSize(size)

    def setYTitle(self, title):
        self._frame.GetYaxis().SetTitle(title)

    def setYTitleSize(self, size):
        self._frame.GetYaxis().SetTitleSize(size)

    def setYTitleOffset(self, offset):
        self._frame.GetYaxis().SetTitleOffset(offset)

    def redrawAxis(self):
        self._pad.RedrawAxis()

class FrameRatio:
    """Class for creating and managing a frame for a ratio plot with two subpads"""
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
            xoffsetFactor *= 2.3
        elif nrows >= 4:
            yoffsetFactor *= 5
            xoffsetFactor *= 3

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

    def adjustMarginLeft(self, adjust):
        self._pad.SetLeftMargin(self._pad.GetLeftMargin()+adjust)
        self._padRatio.SetLeftMargin(self._padRatio.GetLeftMargin()+adjust)
        # Need to redraw frame after adjusting the margin
        self._pad.cd()
        self._frame.Draw("")
        self._padRatio.cd()
        self._frameRatio.Draw("")

    def adjustMarginRight(self, adjust):
        self._pad.SetRightMargin(self._pad.GetRightMargin()+adjust)
        self._padRatio.SetRightMargin(self._padRatio.GetRightMargin()+adjust)
        # Need to redraw frames after adjusting the margin
        self._pad.cd()
        self._frame.Draw("")
        self._padRatio.cd()
        self._frameRatio.Draw("")

    def setTitle(self, title):
        self._frame.SetTitle(title)

    def setXTitle(self, title):
        self._frameRatio.GetXaxis().SetTitle(title)

    def setXTitleSize(self, size):
        self._frameRatio.GetXaxis().SetTitleSize(size)

    def setYTitleOffset(self, offset):
        self._frameRatio.GetXaxis().SetTitleOffset(offset)

    def setXLabelSize(self, size):
        self._frameRatio.GetXaxis().SetLabelSize(size)

    def setYTitle(self, title):
        self._frame.GetYaxis().SetTitle(title)

    def setYTitleRatio(self, title):
        self._frameRatio.GetYaxis().SetTitle(title)

    def setYTitleSize(self, size):
        self._frame.GetYaxis().SetTitleSize(size)
        self._frameRatio.GetYaxis().SetTitleSize(size)

    def setYTitleOffset(self, offset):
        self._frame.GetYaxis().SetTitleOffset(offset)
        self._frameRatio.GetYaxis().SetTitleOffset(offset)

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

class FrameTGraph2D:
    """Class for creating and managing a frame for a plot from TGraph2D"""
    def __init__(self, pad, bounds, histos, ratioOrig, ratioFactor):
        self._pad = pad
        if ratioOrig:
            self._pad = pad.cd(1)

            # adjust margins because of not having the ratio, we want
            # the same bottom margin, so some algebra gives this
            (xlow, ylow, width, height) = (self._pad.GetXlowNDC(), self._pad.GetYlowNDC(), self._pad.GetWNDC(), self._pad.GetHNDC())
            xup = xlow+width
            yup = ylow+height

            bottomMargin = self._pad.GetBottomMargin()
            bottomMarginNew = ROOT.gStyle.GetPadBottomMargin()

            ylowNew = yup - (1-bottomMargin)/(1-bottomMarginNew) * (yup-ylow)
            topMarginNew = self._pad.GetTopMargin() * (yup-ylow)/(yup-ylowNew)

            self._pad.SetPad(xlow, ylowNew, xup, yup)
            self._pad.SetTopMargin(topMarginNew)
            self._pad.SetBottomMargin(bottomMarginNew)

        self._view = ROOT.TView.CreateView()
        self._view.SetRange(bounds[0], bounds[1], 0, bounds[2], bounds[3], 20) # 20 is from Harrison-Stetson, may need tuning?
        self._view.Top()
        self._view.ShowAxis()

        self._xtitleoffset = 1.8
        self._ytitleoffset = 2.3

        self._firstHisto = histos[0]

    def setLogx(self, log):
        pass

    def setLogy(self, log):
        pass

    def setGridx(self, grid):
        pass

    def setGridy(self, grid):
        pass

    def adjustMarginLeft(self, adjust):
        self._pad.SetLeftMargin(self._pad.GetLeftMargin()+adjust)
        self._pad.cd()

    def adjustMarginRight(self, adjust):
        self._pad.SetRightMargin(self._pad.GetRightMargin()+adjust)
        self._pad.cd()

    def setTitle(self, title):
        pass

    def setXTitle(self, title):
        self._xtitle = title

    def setXTitleSize(self, size):
        self._xtitlesize = size

    def setXTitleOffset(self, size):
        self._xtitleoffset = size

    def setXLabelSize(self, size):
        self._xlabelsize = size

    def setYTitle(self, title):
        self._ytitle = title

    def setYTitleSize(self, size):
        self._ytitlesize = size

    def setYTitleOffset(self, offset):
        self._ytitleoffset = offset

    def setZTitle(self, title):
        self._firstHisto.GetZaxis().SetTitle(title)

    def setZTitleOffset(self, offset):
        self._firstHisto.GetZaxis().SetTitleOffset(offset)

    def redrawAxis(self):
        # Disabling and enabled the 3D rulers somehow magically moves the axes to their proper places
        ROOT.TAxis3D.ToggleRulers()
        ROOT.TAxis3D.ToggleRulers()
        axis = ROOT.TAxis3D.GetPadAxis()
        axis.SetLabelColor(ROOT.kBlack);
        axis.SetAxisColor(ROOT.kBlack);

        axis.GetXaxis().SetTitleOffset(self._xtitleoffset)
        axis.GetYaxis().SetTitleOffset(self._ytitleoffset)

        if hasattr(self, "_xtitle"):
            axis.GetXaxis().SetTitle(self._xtitle)
        if hasattr(self, "_xtitlesize"):
            axis.GetXaxis().SetTitleSize(self._xtitlesize)
        if hasattr(self, "_xlabelsize"):
            axis.GetXaxis().SetLabelSize(self._labelsize)
        if hasattr(self, "_ytitle"):
            axis.GetYaxis().SetTitle(self._ytitle)
        if hasattr(self, "_ytitlesize"):
            axis.GetYaxis().SetTitleSize(self._ytitlesize)
        if hasattr(self, "_ytitleoffset"):
            axis.GetYaxis().SetTitleOffset(self._ytitleoffset)

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
        xtitlesize   -- Float for x axis title size (default None)
        xtitleoffset -- Float for x axis title offset (default None)
        xlabelsize   -- Float for x axis label size (default None)
        ytitle       -- String for y axis title (default None)
        ytitlesize   -- Float for y axis title size (default None)
        ytitleoffset -- Float for y axis title offset (default None)
        ztitle       -- String for z axis title (default None)
        ztitleoffset -- Float for z axis title offset (default None)
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
        rebinX       -- rebin x axis (default None)
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
        adjustMarginRight  -- Float for adjusting right margin (default None)
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
        _set("xtitlesize", None)
        _set("xtitleoffset", None)
        _set("xlabelsize", None)
        _set("ytitle", None)
        _set("ytitlesize", None)
        _set("ytitleoffset", None)
        _set("ztitle", None)
        _set("ztitleoffset", None)

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
        _set("rebinX", None)

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

        _set("adjustMarginRight", None)

        _set("ratioYmin", 0.9)
        _set("ratioYmax", 1.1)
        _set("ratioUncertainty", True)

        _set("histogramModifier", None)

        self._histograms = []

    def getNumberOfHistograms(self):
        """Return number of existing histograms."""
        return len(filter(lambda h: h is not None, self._histograms))

    def isEmpty(self):
        """Return true if there are no histograms created for the plot"""
        return self.getNumberOfHistograms() == 0

    def isTGraph2D(self):
        for h in self._histograms:
            if isinstance(h, ROOT.TGraph2D):
                return True
        return False

    def getName(self):
        if isinstance(self._name, list):
            return str(self._name[0])
        else:
            return str(self._name)

    def drawRatioUncertainty(self):
        """Return true if the ratio uncertainty should be drawn"""
        return self._ratioUncertainty

    def _createOne(self, index, tdir):
        """Create one histogram from a TDirectory."""
        if tdir == None:
            return None

        # If name is a list, pick the name by the index
        if isinstance(self._name, list):
            name = self._name[index]
        else:
            name = self._name

        return _getOrCreateObject(tdir, name)

    def create(self, tdirs, requireAllHistograms=False):
        """Create histograms from list of TDirectories"""
        self._histograms = [self._createOne(i, tdir) for i, tdir in enumerate(tdirs)]

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
        if requireAllHistograms and None in self._histograms:
            self._histograms = [None]*len(self._histograms)

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
            h.Sumw2()
            h.Scale(1.0/i)

    def draw(self, pad, ratio, ratioFactor, nrows):
        """Draw the histograms using values for a given algorithm."""
#        if len(self._histograms) == 0:
#            print "No histograms for plot {name}".format(name=self._name)
#            return

        isTGraph2D = self.isTGraph2D()
        if isTGraph2D:
            # Ratios for the TGraph2Ds is not that interesting
            ratioOrig = ratio
            ratio = False

        if self._normalizeToUnitArea:
            self._normalize()

        if self._rebinX is not None:
            for h in self._histograms:
                h.Rebin(self._rebinX)

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
            print "No histograms for plot {name}".format(name=self.getName())
            return

        bounds = _findBounds(histos, self._ylog,
                             xmin=self._xmin, xmax=self._xmax,
                             ymin=self._ymin, ymax=self._ymax)

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
        if isTGraph2D:
            frame = FrameTGraph2D(pad, bounds, histos, ratioOrig, ratioFactor)
        else:
            if ratio:
                ratioBounds = (bounds[0], self._ratioYmin, bounds[2], self._ratioYmax)
                frame = FrameRatio(pad, bounds, ratioBounds, ratioFactor, nrows, xbinlabels, self._xbinlabelsize, self._xbinlabeloption)
            else:
                frame = Frame(pad, bounds, nrows, xbinlabels, self._xbinlabelsize, self._xbinlabeloption)

        # Set log and grid
        frame.setLogx(self._xlog)
        frame.setLogy(self._ylog)
        frame.setGridx(self._xgrid)
        frame.setGridy(self._ygrid)

        # Construct draw option string
        opt = "sames" # s for statbox or something?
        ds = ""
        if self._drawStyle is not None:
            ds = self._drawStyle
        if self._drawCommand is not None:
            ds = self._drawCommand
        if len(ds) > 0:
            opt += " "+ds

        # Set properties of frame
        frame.setTitle(histos[0].GetTitle())
        if self._xtitle is not None:
            frame.setXTitle(self._xtitle)
        if self._xtitlesize is not None:
            frame.setXTitleSize(self._xtitlesize)
        if self._xtitleoffset is not None:
            frame.setXTitleOffset(self._xtitleoffset)
        if self._xlabelsize is not None:
            frame.setXLabelSize(self._xlabelsize)
        if self._ytitle is not None:
            frame.setYTitle(self._ytitle)
        if self._ytitlesize is not None:
            frame.setYTitleSize(self._ytitlesize)
        if self._ytitleoffset is not None:
            frame.setYTitleOffset(self._ytitleoffset)
        if self._ztitle is not None:
            frame.setZTitle(self._ztitle)
        if self._ztitleoffset is not None:
            frame.setZTitleOffset(self._ztitleoffset)
        if self._adjustMarginRight is not None:
            frame.adjustMarginRight(self._adjustMarginRight)
        elif "z" in opt:
            frame.adjustMarginLeft(0.03)
            frame.adjustMarginRight(0.08)

        # Draw histograms
        if ratio:
            frame._pad.cd()

        for h in histos:
            h.Draw(opt)

        # Draw ratios
        if ratio and len(histos) > 0:
            frame._padRatio.cd()
            self._ratios = self._calculateRatios(histos) # need to keep these in memory too ...
            if self._ratioUncertainty and self._ratios[0]._ratio is not None:
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

    def addToLegend(self, legend, legendLabels, denomUncertainty):
        """Add histograms to a legend.

        Arguments:
        legend       -- TLegend
        legendLabels -- List of strings for the legend labels
        """
        first = denomUncertainty
        for h, label in zip(self._histograms, legendLabels):
            if h is None:
                first = False
                continue
            if first:
                self._forLegend = h.Clone()
                self._forLegend.SetFillStyle(1001)
                self._forLegend.SetFillColor(ROOT.kGray)
                entry = legend.AddEntry(self._forLegend, label, "lpf")
                first = False
            else:
                legend.AddEntry(h, label, "LP")

    def _calculateRatios(self, histos):
        """Calculate the ratios for a list of histograms"""

        def _divideOrZero(numerator, denominator):
            if denominator == 0:
                return 0
            return numerator/denominator

        # Define wrappers for TH1/TGraph/TGraph2D to have uniform interface
        # TODO: having more global wrappers would make some things simpler also elsewhere in the code
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
            def makeRatio(self):
                pass

        class WrapTGraph:
            def __init__(self, gr, uncertainty):
                self._gr = gr
                self._uncertainty = uncertainty
                self._xvalues = []
                self._xerrslow = []
                self._xerrshigh = []
                self._yvalues = []
                self._yerrshigh = []
                self._yerrslow = []
                self._binOffset = 0
            def draw(self, style=None):
                if self._ratio is None:
                    return
                st = style
                if st is None:
                    if self._uncertainty:
                        st = "PZ"
                    else:
                        st = "PX"
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
                trueBin = bin + self._binOffset
                xvals = self.xvalues(trueBin)
                xval = xvals[0]
                epsilon = 1e-3 * xval # to allow floating-point difference between TGraph and TH1
                if xval+epsilon < xcenter:
                    self._binOffset += 1
                    return
                # numerator is missing an item
                elif xval-epsilon > xcenter:
                    self._binOffset -= 1
                    return

                self._xvalues.append(xval)
                self._xerrslow.append(xvals[1])
                self._xerrshigh.append(xvals[2])
                yvals = self.yvalues(trueBin)
                self._yvalues.append(yvals[0] / scale)
                if self._uncertainty:
                    self._yerrslow.append(yvals[1] / scale)
                    self._yerrshigh.append(yvals[2] / scale)
                else:
                    self._yerrslow.append(0)
                    self._yerrshigh.append(0)
            def makeRatio(self):
                if len(self._xvalues) == 0:
                    self._ratio = None
                    return
                self._ratio = ROOT.TGraphAsymmErrors(len(self._xvalues), array.array("d", self._xvalues), array.array("d", self._yvalues),
                                                     array.array("d", self._xerrslow), array.array("d", self._xerrshigh), 
                                                     array.array("d", self._yerrslow), array.array("d", self._yerrshigh))
                _copyStyle(self._gr, self._ratio)
        class WrapTGraph2D(WrapTGraph):
            def __init__(self, gr, uncertainty):
                WrapTGraph.__init__(self, gr, uncertainty)
            def xvalues(self, bin):
                return (self._gr.GetX()[bin], self._gr.GetErrorX(bin), self._gr.GetErrorX(bin))
            def yvalues(self, bin):
                return (self._gr.GetY()[bin], self._gr.GetErrorY(bin), self._gr.GetErrorY(bin))

        def wrap(o):
            if isinstance(o, ROOT.TH1):
                return WrapTH1(o, self._ratioUncertainty)
            elif isinstance(o, ROOT.TGraph):
                return WrapTGraph(o, self._ratioUncertainty)
            elif isinstance(o, ROOT.TGraph2D):
                return WrapTGraph2D(o, self._ratioUncertainty)

        wrappers = [wrap(h) for h in histos]
        ref = wrappers[0]

        for bin in xrange(ref.begin(), ref.end()):
            (scale, ylow, yhigh) = ref.yvalues(bin)
            (xval, xlow, xhigh) = ref.xvalues(bin)
            for w in wrappers:
                w.divide(bin, scale, xval)

        for w in wrappers:
            w.makeRatio()

        return wrappers

class PlotGroup:
    """Group of plots, results a TCanvas"""
    def __init__(self, name, plots, **kwargs):
        """Constructor.

        Arguments:
        name  -- String for name of the TCanvas, used also as the basename of the picture files
        plots -- List of Plot objects

        Keyword arguments:
        ncols    -- Number of columns (default 2)
        legendDx -- Float for moving TLegend in x direction (default None)
        legendDy -- Float for moving TLegend in y direction (default None)
        legendDw -- Float for changing TLegend width (default None)
        legendDh -- Float for changing TLegend height (default None)
        overrideLegendLabels -- List of strings for legend labels, if given, these are used instead of the ones coming from Plotter (default None)
        onlyForPileup  -- Plots this group only for pileup samples
        """
        self._name = name
        self._plots = plots

        def _set(attr, default):
            setattr(self, "_"+attr, kwargs.get(attr, default))

        _set("ncols", 2)

        _set("legendDx", None)
        _set("legendDy", None)
        _set("legendDw", None)
        _set("legendDh", None)

        _set("overrideLegendLabels", None)

        _set("onlyForPileup", False)

        self._ratioFactor = 1.25

    def onlyForPileup(self):
        """Return True if the PlotGroup is intended only for pileup samples"""
        return self._onlyForPileup

    def create(self, tdirectories, requireAllHistograms=False):
        """Create histograms from a list of TDirectories.

        Arguments:
        tdirectories         -- List of TDirectory objects
        requireAllHistograms -- If True, a plot is produced if histograms from all files are present (default: False)
        """
        for plot in self._plots:
            plot.create(tdirectories, requireAllHistograms)

    def draw(self, legendLabels, prefix=None, separate=False, saveFormat=".pdf", ratio=False):
        """Draw the histograms using values for a given algorithm.

        Arguments:
        legendLabels  -- List of strings for legend labels (corresponding to the tdirectories in create())
        prefix        -- Optional string for file name prefix (default None)
        separate      -- Save the plots of a group to separate files instead of a file per group (default False)
        saveFormat   -- String specifying the plot format (default '.pdf')
        ratio        -- Add ratio to the plot (default False)
        """

        if self._overrideLegendLabels is not None:
            legendLabels = self._overrideLegendLabels

        # Do not draw the group if it would be empty
        onlyEmptyPlots = True
        for plot in self._plots:
            if not plot.isEmpty():
                onlyEmptyPlots = False
                break
        if onlyEmptyPlots:
            return []

        if separate:
            return self._drawSeparate(legendLabels, prefix, saveFormat, ratio)

        cwidth = 500*self._ncols
        nrows = int((len(self._plots)+1)/self._ncols) # this should work also for odd n
        cheight = 500 * nrows

        if ratio:
            cheight = int(cheight*self._ratioFactor)

        canvas = ROOT.TCanvas(self._name, self._name, cwidth, cheight)

        canvas.Divide(self._ncols, nrows)
        if ratio:
            for i in xrange(0, len(self._plots)):
                pad = canvas.cd(i+1)
                self._modifyPadForRatio(pad)

        # Draw plots to canvas
        for i, plot in enumerate(self._plots):
            pad = canvas.cd(i+1)
            if not plot.isEmpty():
                plot.draw(pad, ratio, self._ratioFactor, nrows)

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
            ly2 = 0.67
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
        denomUnc = sum([p.drawRatioUncertainty() for p in self._plots]) > 0
        legend = self._createLegend(plot, legendLabels, lx1, ly1, lx2, ly2,
                                    denomUncertainty=(ratio and denomUnc))

        return self._save(canvas, saveFormat, prefix=prefix)

    def _drawSeparate(self, legendLabels, prefix, saveFormat, ratio):
        """Internal method to do the drawing to separate files per Plot instead of a file per PlotGroup"""
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
            if plot.isEmpty():
                continue

            if ratio:
                canvas.cd()
                self._modifyPadForRatio(canvas)

            # Draw plot to canvas
            canvas.cd()
            plot.draw(canvas, ratio, self._ratioFactor, 1)


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
            legend = self._createLegend(plot, legendLabels, lx1, ly1, lx2, ly2, textSize=0.03,
                                        denomUncertainty=(ratio and plot.drawRatioUncertainty))

            ret.extend(self._save(canvas, saveFormat, prefix=prefix, postfix="_"+plot.getName(), single=True))
        return ret

    def _modifyPadForRatio(self, pad):
        """Internal method to set divide a pad to two for ratio plots"""
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

    def _createLegend(self, plot, legendLabels, lx1, ly1, lx2, ly2, textSize=0.016, denomUncertainty=True):
        l = ROOT.TLegend(lx1, ly1, lx2, ly2)
        l.SetTextSize(textSize)
        l.SetLineColor(1)
        l.SetLineWidth(1)
        l.SetLineStyle(1)
        l.SetFillColor(0)
        l.SetMargin(0.07)

        plot.addToLegend(l, legendLabels, denomUncertainty)
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

class PlotFolder:
    """Represents a collection of PlotGroups, produced from a single folder in a DQM file"""
    def __init__(self, *plotGroups, **kwargs):
        """Constructor.

        Arguments:
        plotGroups     -- List of PlotGroup objects

        Keyword arguments
        loopSubFolders -- Should the subfolders be looped over? (default: True)
        onlyForPileup  -- Plots this folder only for pileup samples
        purpose        -- html.PlotPurpose member class for the purpose of the folder, used for grouping of the plots to the HTML pages
        page           -- Optional string for the page in HTML generatin
        section        -- Optional string for the section within a page in HTML generation
        """
        self._plotGroups = plotGroups
        self._loopSubFolders = kwargs.pop("loopSubFolders", True)
        self._onlyForPileup = kwargs.pop("onlyForPileup", False)
        self._purpose = kwargs.pop("purpose", None)
        self._page = kwargs.pop("page", None)
        self._section = kwargs.pop("section", None)
        if len(kwargs) > 0:
            raise Exception("Got unexpected keyword arguments: "+ ",".join(kwargs.keys()))

    def loopSubFolders(self):
        """Return True if the PlotGroups of this folder should be applied to the all subfolders"""
        return self._loopSubFolders

    def onlyForPileup(self):
        """Return True if the folder is intended only for pileup samples"""
        return self._onlyForPileup

    def getPurpose(self):
        return self._purpose

    def getPage(self):
        return self._page

    def getSection(self):
        return self._section

    def append(self, plotGroup):
        self._plotGroups.append(plotGroup)

    def set(self, plotGroups):
        self._plotGroups = plotGroups

    def create(self, files, labels, possibleDqmFolders, dqmSubFolder=None, isPileupSample=True, requireAllHistograms=False):
        """Create histograms from a list of TFiles.

        Arguments:
        files  -- List of TFiles
        labels -- List of strings for legend labels corresponding the files
        possibleDqmFolders -- List of strings for possible directories of histograms in TFiles
        dqmSubFolder -- Optional string for subdirectory inside the dqmFolder; if list of strings, then each corresponds to a TFile
        isPileupSample -- Is sample pileup (some PlotGroups may limit themselves to pileup)
        requireAllHistograms -- If True, a plot is produced if histograms from all files are present (default: False)
        """

        if len(files) != len(labels):
            raise Exception("len(files) should be len(labels), now they are %d and %d" % (len(files), len(labels)))

        dirs = []
        self._labels = []
        if isinstance(dqmSubFolder, list):
            if len(dqmSubFolder) != len(files):
                raise Exception("When dqmSubFolder is a list, len(dqmSubFolder) should be len(files), now they are %d and %d" % (len(dqmSubFolder), len(files)))
        else:
            dqmSubFolder = [dqmSubFolder]*len(files)

        for fil, sf in zip(files, dqmSubFolder):
            dirs.append(self._getDir(fil, possibleDqmFolders, sf))
        self._labels = labels

        for pg in self._plotGroups:
            if pg.onlyForPileup() and not isPileupSample:
                continue
            pg.create(dirs, requireAllHistograms)

    def draw(self, prefix=None, separate=False, saveFormat=".pdf", ratio=False):
        """Draw and save all plots using settings of a given algorithm.

        Arguments:
        prefix   -- Optional string for file name prefix (default None)
        separate -- Save the plots of a group to separate files instead of a file per group (default False)
        saveFormat   -- String specifying the plot format (default '.pdf')
        ratio    -- Add ratio to the plot (default False)
        """
        ret = []

        for pg in self._plotGroups:
            ret.extend(pg.draw(self._labels, prefix=prefix, separate=separate, saveFormat=saveFormat, ratio=ratio))
        return ret


    def _getDir(self, tfile, possibleDqmFolders, dqmSubFolder):
        """Get TDirectory from TFile."""
        if tfile is None:
            return None
        for pdf in possibleDqmFolders:
            d = tfile.Get(pdf)
            if d:
                if dqmSubFolder is not None:
                    # Pick associator if given
                    d = d.Get(dqmSubFolder)
                    if d:
                        return d
                    else:
                        print "Did not find subdirectory '%s' from directory '%s' in file %s" % (dqmSubFolder, pdf, tfile.GetName())
                        return None
                else:
                    return d
        print "Did not find any of directories '%s' from file %s" % (",".join(possibleDqmFolders), tfile.GetName())
        return None

    # These are to be overridden by derived classes for customisation
    def translateSubFolder(self, dqmSubFolderName):
        """Method called to (possibly) translate a subfolder name to more 'readable' form

        The implementation in this (base) class just returns the
        argument. The idea is that a deriving class might want to do
        something more complex (like trackingPlots.TrackingPlotFolder
        does)
        """
        return dqmSubFolderName

    def getSelectionName(self, plotFolderName, translatedDqmSubFolder):
        """Get selection name (used in output directory name and legend) from the name of PlotterFolder, and a return value of translateSubFolder"""
        ret = ""
        if plotFolderName != "":
            ret += "_"+plotFolderName
        if translatedDqmSubFolder is not None:
            ret += "_"+translatedDqmSubFolder
        return ret

    def limitSubFolder(self, limitOnlyTo, translatedDqmSubFolder):
        """Return True if this subfolder should be processed

        Arguments:
        limitOnlyTo            -- List/set/similar containing the translatedDqmSubFolder 
        translatedDqmSubFolder -- Return value of translateSubFolder
        """
        return translatedDqmSubFolder in limitOnlyTo

class DQMSubFolder:
    """Class to hold the original name and a 'translated' name of a subfolder in the DQM ROOT file"""
    def __init__(self, subfolder, translated):
        self.subfolder = subfolder
        self.translated = translated

    def equalTo(self, other):
        """Equality is defined by the 'translated' name"""
        return self.translated == other.translated

class PlotterFolder:
    """Plotter for one DQM folder.

    This class is supposed to be instantiated by the Plotter class (or
    PlotterItem, to be more specific), and not used directly by the
    user.
    """
    def __init__(self, name, possibleDqmFolders, dqmSubFolders, plotFolder, fallbackNames):
        """
        Constructor

        Arguments:
        name               -- Name of the folder (is used in the output directory naming)
        possibleDqmFolders -- List of strings for possible directories of histograms in TFiles
        dqmSubFolders      -- List of lists of strings for list of subfolders per input file, or None if no subfolders
        plotFolder         -- PlotFolder object
        fallbackNames      -- List of names for backward compatibility (can be empty). These are used only by validation.Validation (class responsible of the release validation workflow) in case the reference file pointed by 'name' does not exist.
        """
        self._name = name
        self._possibleDqmFolders = possibleDqmFolders
        self._plotFolder = plotFolder
        #self._dqmSubFolders = [map(lambda sf: DQMSubFolder(sf, self._plotFolder.translateSubFolder(sf)), lst) for lst in dqmSubFolders]
        if dqmSubFolders is None:
            self._dqmSubFolders = None
        else:
            self._dqmSubFolders = map(lambda sf: DQMSubFolder(sf, self._plotFolder.translateSubFolder(sf)), dqmSubFolders[0])
            self._dqmSubFolders = filter(lambda sf: sf.translated is not None, self._dqmSubFolders)

        self._fallbackNames = fallbackNames

        # TODO: matchmaking of dqmsubfolders in case of differences between files

    def getName(self):
        return self._name

    def getPurpose(self):
        return self._plotFolder.getPurpose()

    def getPage(self):
        return self._plotFolder.getPage()

    def getSection(self):
        return self._plotFolder.getSection()

    def onlyForPileup(self):
        return self._plotFolder.onlyForPileup()

    def getPossibleDQMFolders(self):
        return self._possibleDqmFolders

    def getDQMSubFolders(self, limitOnlyTo=None):
        """Get list of subfolders, possibly limiting to some of them.

        Keyword arguments:
        limitOnlyTo -- Object depending on the PlotFolder type for limiting the set of subfolders to be processed
        """

        if self._dqmSubFolders is None:
            return [None]

        if limitOnlyTo is None:
            return self._dqmSubFolders

        return filter(lambda s: self._plotFolder.limitSubFolder(limitOnlyTo, s.translated), self._dqmSubFolders)

    def getSelectionNameIterator(self, dqmSubFolder):
        """Get a generator for the 'selection name', looping over the name and fallbackNames"""
        for name in [self._name]+self._fallbackNames:
            yield self._plotFolder.getSelectionName(name, dqmSubFolder.translated if dqmSubFolder is not None else None)

    def getSelectionName(self, dqmSubFolder):
        return self.getSelectionNameIterator(dqmSubFolder).next()

    def create(self, files, labels, dqmSubFolder, isPileupSample=True, requireAllHistograms=False):
        """Create histograms from a list of TFiles.
        Arguments:
        files  -- List of TFiles
        labels -- List of strings for legend labels corresponding the files
        dqmSubFolder -- DQMSubFolder object for a subfolder (or None for no subfolder)
        isPileupSample -- Is sample pileup (some PlotGroups may limit themselves to pileup)
        requireAllHistograms -- If True, a plot is produced if histograms from all files are present (default: False)
        """

        # TODO: for cases of differently named subfolders, need to think something here
        self._plotFolder.create(files, labels, self._possibleDqmFolders, dqmSubFolder.subfolder if dqmSubFolder is not None else None, isPileupSample, requireAllHistograms)

    def draw(self, *args, **kwargs):
        """Draw and save all plots using settings of a given algorithm."""
        return self._plotFolder.draw(*args, **kwargs)

class PlotterInstance:
    """Instance of plotter that knows the directory content, holds many folders."""
    def __init__(self, folders):
        self._plotterFolders = filter(lambda f: f is not None, folders)

    def iterFolders(self, limitSubFoldersOnlyTo=None):
        for plotterFolder in self._plotterFolders:
            limitOnlyTo = None
            if limitSubFoldersOnlyTo is not None:
                limitOnlyTo = limitSubFoldersOnlyTo.get(plotterFolder.getName(), None)

            for dqmSubFolder in plotterFolder.getDQMSubFolders(limitOnlyTo=limitOnlyTo):
                yield plotterFolder, dqmSubFolder

# Helper for Plotter
class PlotterItem:
    def __init__(self, name, possibleDirs, plotFolder, fallbackNames=[]):
        """ Constructor

        Arguments:
        name          -- Name of the folder (is used in the output directory naming)
        possibleDirs  -- List of strings for possible directories of histograms in TFiles
        plotFolder    -- PlotFolder object

        Keyword arguments
        fallbackNames -- Optional list of names for backward compatibility. These are used only by validation.Validation (class responsible of the release validation workflow) in case the reference file pointed by 'name' does not exist.
        """
        self._name = name
        self._possibleDirs = possibleDirs
        self._plotFolder = plotFolder
        self._fallbackNames = fallbackNames

    def readDirs(self, files):
        """Read available subfolders from the files

        Arguments:
        files -- List of strings for paths to files, or list of TFiles

        For each file, loop over 'possibleDirs', and read the
        subfolders of first one that exists.

        Returns a PlotterFolder if at least one file for which one of
        'possibleDirs' exists. Otherwise, return None to signal that
        there is nothing available for this PlotFolder.
        """
        subFolders = None
        if self._plotFolder.loopSubFolders():
            subFolders = []
        possibleDirFound = False
        for fname in files:
            isOpenFile = isinstance(fname, ROOT.TFile)
            if isOpenFile:
                tfile = fname
            else:
                tfile = ROOT.TFile.Open(fname)
            for pd in self._possibleDirs:
                d = tfile.Get(pd)
                if d:
                    possibleDirFound = True
                    if subFolders is not None:
                        subf = []
                        for key in d.GetListOfKeys():
                            if isinstance(key.ReadObj(), ROOT.TDirectory):
                                subf.append(key.GetName())
                        subFolders.append(subf)
                    break

            if not isOpenFile:
                tfile.Close()

        if not possibleDirFound:
            return None

        return PlotterFolder(self._name, self._possibleDirs, subFolders, self._plotFolder, self._fallbackNames)

class Plotter:
    """Contains PlotFolders, i.e. the information what plots to do, and creates a helper object to actually produce the plots."""
    def __init__(self):
        self._plots = []

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

    def append(self, *args, **kwargs):
        """Append a plot folder to the plotter.

        All arguments are forwarded to the constructor of PlotterItem.
        """
        self._plots.append(PlotterItem(*args, **kwargs))

    def readDirs(self, *files):
        """Returns PlotterInstance object, which knows how exactly to produce the plots for these files"""
        return PlotterInstance([plotterItem.readDirs(files) for plotterItem in self._plots])
