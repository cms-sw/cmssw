from __future__ import print_function
from __future__ import absolute_import
from builtins import range
import os
import sys
import math
import copy
import array
import difflib
import collections

import six
import ROOT
ROOT.gROOT.SetBatch(True)
ROOT.PyConfig.IgnoreCommandLineOptions = True

from . import html

verbose=False
_ratioYTitle = "Ratio"

def _setStyle():
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

    ROOT.TGaxis.SetMaxDigits(4)

def _getObject(tdirectory, name):
    obj = tdirectory.Get(name)
    if not obj:
        if verbose:
            print("Did not find {obj} from {dir}".format(obj=name, dir=tdirectory.GetPath()))
        return None
    return obj

def _getOrCreateObject(tdirectory, nameOrCreator):
    if hasattr(nameOrCreator, "create"):
        return nameOrCreator.create(tdirectory)
    return _getObject(tdirectory, nameOrCreator)

class GetDirectoryCode:
    class FileNotExist: pass
    class PossibleDirsNotExist: pass
    class SubDirNotExist: pass

    @staticmethod
    def codesToNone(code):
        if code in [GetDirectoryCode.FileNotExist, GetDirectoryCode.PossibleDirsNotExist, GetDirectoryCode.SubDirNotExist]:
            return None
        return code

def _getDirectoryDetailed(tfile, possibleDirs, subDir=None):
    """Get TDirectory from TFile."""
    if tfile is None:
        return GetDirectoryCode.FileNotExist
    for pdf in possibleDirs:
        d = tfile.Get(pdf)
        if d:
            if subDir is not None:
                # Pick associator if given
                d = d.Get(subDir)
                if d:
                    return d
                else:
                    if verbose:
                        print("Did not find subdirectory '%s' from directory '%s' in file %s" % (subDir, pdf, tfile.GetName()))
#                        if "Step" in subDir:
#                            raise Exception("Foo")
                    return GetDirectoryCode.SubDirNotExist
            else:
                return d
    if verbose:
        print("Did not find any of directories '%s' from file %s" % (",".join(possibleDirs), tfile.GetName()))
    return GetDirectoryCode.PossibleDirsNotExist

def _getDirectory(*args, **kwargs):
    return GetDirectoryCode.codesToNone(_getDirectoryDetailed(*args, **kwargs))

def _th1ToOrderedDict(th1, renameBin=None):
    values = collections.OrderedDict()
    for i in range(1, th1.GetNbinsX()+1):
        binLabel = th1.GetXaxis().GetBinLabel(i)
        if renameBin is not None:
            binLabel = renameBin(binLabel)
        values[binLabel] = (th1.GetBinContent(i), th1.GetBinError(i))
    return values

def _createCanvas(name, width, height):
    # silence warning of deleting canvas with the same name
    if not verbose:
        backup = ROOT.gErrorIgnoreLevel
        ROOT.gErrorIgnoreLevel = ROOT.kError
    canvas = ROOT.TCanvas(name, name, width, height)
    if not verbose:
        ROOT.gErrorIgnoreLevel = backup
    return canvas

def _modifyPadForRatio(pad, ratioFactor):
    pad.Divide(1, 2)

    divisionPoint = 1-1/ratioFactor

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
    pad2.SetBottomMargin(bottomMargin/(ratioFactor*divisionPoint))

def _calculateRatios(histos, ratioUncertainty=False):
    """Calculate the ratios for a list of histograms"""

    def _divideOrZero(numerator, denominator):
        if denominator == 0:
            return 0
        return numerator/denominator

    def equal(a, b):
        if a == 0. and b == 0.:
            return True
        return abs(a-b)/max(abs(a),abs(b)) < 1e-3

    def findBins(wrap, bins_xvalues):
        ret = []
        currBin = wrap.begin()
        i = 0
        while i < len(bins_xvalues) and currBin < wrap.end():
            (xcenter, xlow, xhigh) = bins_xvalues[i]
            xlowEdge = xcenter-xlow
            xupEdge = xcenter+xhigh

            (curr_center, curr_low, curr_high) = wrap.xvalues(currBin)
            curr_lowEdge = curr_center-curr_low
            curr_upEdge = curr_center+curr_high

            if equal(xlowEdge, curr_lowEdge) and equal(xupEdge,  curr_upEdge):
                ret.append(currBin)
                currBin += 1
                i += 1
            elif curr_upEdge <= xlowEdge:
                currBin += 1
            elif curr_lowEdge >= xupEdge:
                ret.append(None)
                i += 1
            else:
                ret.append(None)
                currBin += 1
                i += 1
        if len(ret) != len(bins_xvalues):
            ret.extend([None]*( len(bins_xvalues) - len(ret) ))
        return ret

    # Define wrappers for TH1/TGraph/TGraph2D to have uniform interface
    # TODO: having more global wrappers would make some things simpler also elsewhere in the code
    class WrapTH1:
        def __init__(self, th1, uncertainty):
            self._th1 = th1
            self._uncertainty = uncertainty

            xaxis = th1.GetXaxis()
            xaxis_arr = xaxis.GetXbins()
            if xaxis_arr.GetSize() > 0: # unequal binning
                lst = [xaxis_arr[i] for i in range(0, xaxis_arr.GetSize())]
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
        def divide(self, bin, scale):
            self._ratio.SetBinContent(bin, _divideOrZero(self._th1.GetBinContent(bin), scale))
            self._ratio.SetBinError(bin, _divideOrZero(self._th1.GetBinError(bin), scale))
        def makeRatio(self):
            pass
        def getRatio(self):
            return self._ratio

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
        def divide(self, bin, scale):
            # Ignore bin if denominator is zero
            if scale == 0:
                return
            # No more items in the numerator
            if bin >= self._gr.GetN():
                return
            # denominator is missing an item
            xvals = self.xvalues(bin)
            xval = xvals[0]

            self._xvalues.append(xval)
            self._xerrslow.append(xvals[1])
            self._xerrshigh.append(xvals[2])
            yvals = self.yvalues(bin)
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
        def getRatio(self):
            return self._ratio
    class WrapTGraph2D(WrapTGraph):
        def __init__(self, gr, uncertainty):
            WrapTGraph.__init__(self, gr, uncertainty)
        def xvalues(self, bin):
            return (self._gr.GetX()[bin], self._gr.GetErrorX(bin), self._gr.GetErrorX(bin))
        def yvalues(self, bin):
            return (self._gr.GetY()[bin], self._gr.GetErrorY(bin), self._gr.GetErrorY(bin))

    def wrap(o):
        if isinstance(o, ROOT.TH1):
            return WrapTH1(o, ratioUncertainty)
        elif isinstance(o, ROOT.TGraph):
            return WrapTGraph(o, ratioUncertainty)
        elif isinstance(o, ROOT.TGraph2D):
            return WrapTGraph2D(o, ratioUncertainty)

    wrappers = [wrap(h) for h in histos]
    ref = wrappers[0]

    wrappers_bins = []
    ref_bins = [ref.xvalues(b) for b in range(ref.begin(), ref.end())]
    for w in wrappers:
        wrappers_bins.append(findBins(w, ref_bins))

    for i, bin in enumerate(range(ref.begin(), ref.end())):
        (scale, ylow, yhigh) = ref.yvalues(bin)
        for w, bins in zip(wrappers, wrappers_bins):
            if bins[i] is None:
                continue
            w.divide(bins[i], scale)

    for w in wrappers:
        w.makeRatio()

    return wrappers


def _getXmin(obj, limitToNonZeroContent=False):
    if isinstance(obj, ROOT.TH1):
        xaxis = obj.GetXaxis()
        if limitToNonZeroContent:
            for i in range(1, obj.GetNbinsX()+1):
                if obj.GetBinContent(i) != 0:
                    return xaxis.GetBinLowEdge(i)
            # None for all bins being zero
            return None
        else:
            return xaxis.GetBinLowEdge(xaxis.GetFirst())
    elif isinstance(obj, ROOT.TGraph) or isinstance(obj, ROOT.TGraph2D):
        m = min([obj.GetX()[i] for i in range(0, obj.GetN())])
        return m*0.9 if m > 0 else m*1.1
    raise Exception("Unsupported type %s" % str(obj))

def _getXmax(obj, limitToNonZeroContent=False):
    if isinstance(obj, ROOT.TH1):
        xaxis = obj.GetXaxis()
        if limitToNonZeroContent:
            for i in range(obj.GetNbinsX(), 0, -1):
                if obj.GetBinContent(i) != 0:
                    return xaxis.GetBinUpEdge(i)
            # None for all bins being zero
            return None
        else:
            return xaxis.GetBinUpEdge(xaxis.GetLast())
    elif isinstance(obj, ROOT.TGraph) or isinstance(obj, ROOT.TGraph2D):
        m = max([obj.GetX()[i] for i in range(0, obj.GetN())])
        return m*1.1 if m > 0 else m*0.9
    raise Exception("Unsupported type %s" % str(obj))

def _getYmin(obj, limitToNonZeroContent=False):
    if isinstance(obj, ROOT.TH2):
        yaxis = obj.GetYaxis()
        return yaxis.GetBinLowEdge(yaxis.GetFirst())
    elif isinstance(obj, ROOT.TH1):
        if limitToNonZeroContent:
            lst = [obj.GetBinContent(i) for i in range(1, obj.GetNbinsX()+1) if obj.GetBinContent(i) != 0 ]
            return min(lst) if len(lst) != 0 else 0
        else:
            return obj.GetMinimum()
    elif isinstance(obj, ROOT.TGraph) or isinstance(obj, ROOT.TGraph2D):
        m = min([obj.GetY()[i] for i in range(0, obj.GetN())])
        return m*0.9 if m > 0 else m*1.1
    raise Exception("Unsupported type %s" % str(obj))

def _getYmax(obj, limitToNonZeroContent=False):
    if isinstance(obj, ROOT.TH2):
        yaxis = obj.GetYaxis()
        return yaxis.GetBinUpEdge(yaxis.GetLast())
    elif isinstance(obj, ROOT.TH1):
        if limitToNonZeroContent:
            lst = [obj.GetBinContent(i) for i in range(1, obj.GetNbinsX()+1) if obj.GetBinContent(i) != 0 ]
            return max(lst) if len(lst) != 0 else 0
        else:
            return obj.GetMaximum()
    elif isinstance(obj, ROOT.TGraph) or isinstance(obj, ROOT.TGraph2D):
        m = max([obj.GetY()[i] for i in range(0, obj.GetN())])
        return m*1.1 if m > 0 else m*0.9
    raise Exception("Unsupported type %s" % str(obj))

def _getYmaxWithError(th1):
    return max([th1.GetBinContent(i)+th1.GetBinError(i) for i in range(1, th1.GetNbinsX()+1)])

def _getYminIgnoreOutlier(th1):
    yvals = sorted([n for n in [th1.GetBinContent(i) for i in range(1, th1.GetNbinsX()+1)] if n>0])
    if len(yvals) == 0:
        return th1.GetMinimum()
    if len(yvals) == 1:
        return yvals[0]

    # Define outlier as being x10 less than minimum of the 95 % of the non-zero largest values
    ind_min = len(yvals)-1 - int(len(yvals)*0.95)
    min_val = yvals[ind_min]
    for i in range(0, ind_min):
        if yvals[i] > 0.1*min_val:
            return yvals[i]

    return min_val

def _getYminMaxAroundMedian(obj, coverage, coverageRange=None):
    inRange = lambda x: True
    inRange2 = lambda xmin,xmax: True
    if coverageRange:
        inRange = lambda x: coverageRange[0] <= x <= coverageRange[1]
        inRange2 = lambda xmin,xmax: coverageRange[0] <= xmin and xmax <= coverageRange[1]

    if isinstance(obj, ROOT.TH1):
        yvals = [obj.GetBinContent(i) for i in range(1, obj.GetNbinsX()+1) if inRange2(obj.GetXaxis().GetBinLowEdge(i), obj.GetXaxis().GetBinUpEdge(i))]
        yvals = [x for x in yvals if x != 0]
    elif isinstance(obj, ROOT.TGraph) or isinstance(obj, ROOT.TGraph2D):
        yvals = [obj.GetY()[i] for i in range(0, obj.GetN()) if inRange(obj.GetX()[i])]
    else:
        raise Exception("Unsupported type %s" % str(obj))
    if len(yvals) == 0:
        return (0, 0)
    if len(yvals) == 1:
        return (yvals[0], yvals[0])
    if len(yvals) == 2:
        return (yvals[0], yvals[1])

    yvals.sort()
    nvals = int(len(yvals)*coverage)
    if nvals < 2:
        # Take median and +- 1 values
        if len(yvals) % 2 == 0:
            half = len(yvals)/2
            return ( yvals[half-1], yvals[half] )
        else:
            middle = len(yvals)/2
            return ( yvals[middle-1], yvals[middle+1] )
    ind_min = (len(yvals)-nvals)/2
    ind_max = len(yvals)-1 - ind_min

    return (yvals[ind_min], yvals[ind_max])

def _findBounds(th1s, ylog, xmin=None, xmax=None, ymin=None, ymax=None):
    """Find x-y axis boundaries encompassing a list of TH1s if the bounds are not given in arguments.

    Arguments:
    th1s -- List of TH1s
    ylog -- Boolean indicating if y axis is in log scale or not (affects the automatic ymax)

    Keyword arguments:
    xmin -- Minimum x value; if None, take the minimum of TH1s
    xmax -- Maximum x value; if None, take the maximum of TH1s
    ymin -- Minimum y value; if None, take the minimum of TH1s
    ymax -- Maximum y value; if None, take the maximum of TH1s
    """

    (ymin, ymax) = _findBoundsY(th1s, ylog, ymin, ymax)

    if xmin is None or xmax is None or isinstance(xmin, list) or isinstance(max, list):
        xmins = []
        xmaxs = []
        for th1 in th1s:
            xmins.append(_getXmin(th1, limitToNonZeroContent=isinstance(xmin, list)))
            xmaxs.append(_getXmax(th1, limitToNonZeroContent=isinstance(xmax, list)))

        # Filter out cases where histograms have zero content
        xmins = [h for h in xmins if h is not None]
        xmaxs = [h for h in xmaxs if h is not None]

        if xmin is None:
            xmin = min(xmins)
        elif isinstance(xmin, list):
            if len(xmins) == 0: # all histograms zero
                xmin = min(xmin)
                if verbose:
                    print("Histogram is zero, using the smallest given value for xmin from", str(xmin))
            else:
                xm = min(xmins)
                xmins_below = [x for x in xmin if x<=xm]
                if len(xmins_below) == 0:
                    xmin = min(xmin)
                    if xm < xmin:
                        if verbose:
                            print("Histogram minimum x %f is below all given xmin values %s, using the smallest one" % (xm, str(xmin)))
                else:
                    xmin = max(xmins_below)

        if xmax is None:
            xmax = max(xmaxs)
        elif isinstance(xmax, list):
            if len(xmaxs) == 0: # all histograms zero
                xmax = max(xmax)
                if verbose:
                    print("Histogram is zero, using the smallest given value for xmax from", str(xmin))
            else:
                xm = max(xmaxs)
                xmaxs_above = [x for x in xmax if x>xm]
                if len(xmaxs_above) == 0:
                    xmax = max(xmax)
                    if xm > xmax:
                        if verbose:
                            print("Histogram maximum x %f is above all given xmax values %s, using the maximum one" % (xm, str(xmax)))
                else:
                    xmax = min(xmaxs_above)

    for th1 in th1s:
        th1.GetXaxis().SetRangeUser(xmin, xmax)

    return (xmin, ymin, xmax, ymax)

def _findBoundsY(th1s, ylog, ymin=None, ymax=None, coverage=None, coverageRange=None):
    """Find y axis boundaries encompassing a list of TH1s if the bounds are not given in arguments.

    Arguments:
    th1s -- List of TH1s
    ylog -- Boolean indicating if y axis is in log scale or not (affects the automatic ymax)

    Keyword arguments:
    ymin -- Minimum y value; if None, take the minimum of TH1s
    ymax -- Maximum y value; if None, take the maximum of TH1s
    coverage -- If set, use only values within the 'coverage' part around the median are used for min/max (useful for ratio)
    coverageRange -- If coverage and this are set, use only the x axis specified by an (xmin,xmax) pair for the coverage
    """
    if coverage is not None or isinstance(th1s[0], ROOT.TH2):
        # the only use case for coverage for now is ratio, for which
        # the scalings are not needed (actually harmful), so let's
        # just ignore them if 'coverage' is set
        #
        # Also for TH2 do not adjust automatic y bounds
        y_scale_max = lambda y: y
        y_scale_min = lambda y: y
    else:
        if ylog:
            y_scale_max = lambda y: y*1.5
        else:
            y_scale_max = lambda y: y*1.05
        y_scale_min = lambda y: y*0.9 # assuming log

    if ymin is None or ymax is None or isinstance(ymin, list) or isinstance(ymax, list):
        ymins = []
        ymaxs = []
        for th1 in th1s:
            if coverage is not None:
                (_ymin, _ymax) = _getYminMaxAroundMedian(th1, coverage, coverageRange)
            else:
                if ylog and isinstance(ymin, list):
                    _ymin = _getYminIgnoreOutlier(th1)
                else:
                    _ymin = _getYmin(th1, limitToNonZeroContent=isinstance(ymin, list))
                _ymax = _getYmax(th1, limitToNonZeroContent=isinstance(ymax, list))
#                _ymax = _getYmaxWithError(th1)

            ymins.append(_ymin)
            ymaxs.append(_ymax)

        if ymin is None:
            ymin = min(ymins)
        elif isinstance(ymin, list):
            ym_unscaled = min(ymins)
            ym = y_scale_min(ym_unscaled)
            ymins_below = [y for y in ymin if y<=ym]
            if len(ymins_below) == 0:
                ymin = min(ymin)
                if ym_unscaled < ymin:
                    if verbose:
                        print("Histogram minimum y %f is below all given ymin values %s, using the smallest one" % (ym, str(ymin)))
            else:
                ymin = max(ymins_below)

        if ymax is None:
            # in case ymax is automatic, ymin is set by list, and the
            # histograms are zero, ensure here that ymax > ymin
            ymax = y_scale_max(max(ymaxs+[ymin]))
        elif isinstance(ymax, list):
            ym_unscaled = max(ymaxs)
            ym = y_scale_max(ym_unscaled)
            ymaxs_above = [y for y in ymax if y>ym]
            if len(ymaxs_above) == 0:
                ymax = max(ymax)
                if ym_unscaled > ymax:
                    if verbose:
                        print("Histogram maximum y %f is above all given ymax values %s, using the maximum one" % (ym_unscaled, str(ymax)))
            else:
                ymax = min(ymaxs_above)

    for th1 in th1s:
        th1.GetYaxis().SetRangeUser(ymin, ymax)

    return (ymin, ymax)

def _th1RemoveEmptyBins(histos, xbinlabels):
    binsToRemove = set()
    for b in range(1, histos[0].GetNbinsX()+1):
        binEmpty = True
        for h in histos:
            if h.GetBinContent(b) > 0:
                binEmpty = False
                break
        if binEmpty:
            binsToRemove.add(b)

    if len(binsToRemove) > 0:
        # filter xbinlabels
        xbinlab_new = []
        for i in range(len(xbinlabels)):
            if (i+1) not in binsToRemove:
                xbinlab_new.append(xbinlabels[i])
        xbinlabels = xbinlab_new

        # filter histogram bins
        histos_new = []
        for h in histos:
            values = []
            for b in range(1, h.GetNbinsX()+1):
                if b not in binsToRemove:
                    values.append( (h.GetXaxis().GetBinLabel(b), h.GetBinContent(b), h.GetBinError(b)) )

            if len(values) > 0:
                h_new = h.Clone(h.GetName()+"_empty")
                h_new.SetBins(len(values), h.GetBinLowEdge(1), h.GetBinLowEdge(1)+len(values))
                for b, (l, v, e) in enumerate(values):
                    h_new.GetXaxis().SetBinLabel(b+1, l)
                    h_new.SetBinContent(b+1, v)
                    h_new.SetBinError(b+1, e)

                histos_new.append(h_new)
        histos = histos_new

    return (histos, xbinlabels)

def _th2RemoveEmptyBins(histos, xbinlabels, ybinlabels):
    xbinsToRemove = set()
    ybinsToRemove = set()
    for ih, h in enumerate(histos):
        for bx in range(1, h.GetNbinsX()+1):
            binEmpty = True
            for by in range(1, h.GetNbinsY()+1):
                if h.GetBinContent(bx, by) > 0:
                    binEmpty = False
                    break
            if binEmpty:
                xbinsToRemove.add(bx)
            elif ih > 0:
                xbinsToRemove.discard(bx)

        for by in range(1, h.GetNbinsY()+1):
            binEmpty = True
            for bx in range(1, h.GetNbinsX()+1):
                if h.GetBinContent(bx, by) > 0:
                    binEmpty = False
                    break
            if binEmpty:
                ybinsToRemove.add(by)
            elif ih > 0:
                ybinsToRemove.discard(by)

    if len(xbinsToRemove) > 0 or len(ybinsToRemove) > 0:
        xbinlabels_new = []
        xbins = []
        for b in range(1, len(xbinlabels)+1):
            if b not in xbinsToRemove:
                xbinlabels_new.append(histos[0].GetXaxis().GetBinLabel(b))
                xbins.append(b)
        xbinlabels = xbinlabels_new
        ybinlabels_new = []
        ybins = []
        for b in range(1, len(ybinlabels)+1):
            if b not in ybinsToRemove:
                ybinlabels.append(histos[0].GetYaxis().GetBinLabel(b))
                ybins.append(b)
        ybinlabels = xbinlabels_new

        histos_new = []
        if len(xbinlabels) == 0 or len(ybinlabels) == 0:
            return (histos_new, xbinlabels, ybinlabels)
        for h in histos:
            h_new = ROOT.TH2F(h.GetName()+"_empty", h.GetTitle(), len(xbinlabels),0,len(xbinlabels), len(ybinlabels),0,len(ybinlabels))
            for b, l in enumerate(xbinlabels):
                h_new.GetXaxis().SetBinLabel(b+1, l)
            for b, l in enumerate(ybinlabels):
                h_new.GetYaxis().SetBinLabel(b+1, l)

            for ix, bx in enumerate(xbins):
                for iy, by in enumerate(ybins):
                    h_new.SetBinContent(ix+1, iy+1, h.GetBinContent(bx, by))
                    h_new.SetBinError(ix+1, iy+1, h.GetBinError(bx, by))
            histos_new.append(h_new)
        histos = histos_new
    return (histos, xbinlabels, ybinlabels)

def _mergeBinLabelsX(histos):
    return _mergeBinLabels([[h.GetXaxis().GetBinLabel(i) for i in range(1, h.GetNbinsX()+1)] for h in histos])

def _mergeBinLabelsY(histos):
    return _mergeBinLabels([[h.GetYaxis().GetBinLabel(i) for i in range(1, h.GetNbinsY()+1)] for h in histos])

def _mergeBinLabels(labelsAll):
    labels_merged = labelsAll[0]
    for labels in labelsAll[1:]:
        diff = difflib.unified_diff(labels_merged, labels, n=max(len(labels_merged), len(labels)))
        labels_merged = []
        operation = []
        for item in diff: # skip the "header" lines
            if item[:2] == "@@":
                break
        for item in diff:
            operation.append(item[0])
            lab = item[1:]
            if lab in labels_merged:
                # pick the last addition of the bin
                ind = labels_merged.index(lab)
                if operation[ind] == "-" and operation[-1] == "+":
                    labels_merged.remove(lab)
                    del operation[ind] # to keep xbinlabels and operation indices in sync
                elif operation[ind] == "+" and operation[-1] == "-":
                    del operation[-1] # to keep xbinlabels and operation indices in sync
                    continue
                else:
                    raise Exception("This should never happen")
            labels_merged.append(lab)
        # unified_diff returns empty diff if labels_merged and labels are equal
        # so if labels_merged is empty here, it can be just set to labels
        if len(labels_merged) == 0:
            labels_merged = labels

    return labels_merged

def _th1IncludeOnlyBins(histos, xbinlabels):
    histos_new = []
    for h in histos:
        h_new = h.Clone(h.GetName()+"_xbinlabels")
        h_new.SetBins(len(xbinlabels), h.GetBinLowEdge(1), h.GetBinLowEdge(1)+len(xbinlabels))
        for i, label in enumerate(xbinlabels):
            bin = h.GetXaxis().FindFixBin(label)
            if bin >= 0:
                h_new.SetBinContent(i+1, h.GetBinContent(bin))
                h_new.SetBinError(i+1, h.GetBinError(bin))
            else:
                h_new.SetBinContent(i+1, 0)
                h_new.SetBinError(i+1, 0)
        histos_new.append(h_new)
    return histos_new


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

        # Disable canExtend if it is set, otherwise setting the
        # overflow bin will extend instead, possibly causing weird
        # effects downstream
        ret.SetCanExtend(False)

        for i in range(0, histoA.GetNbinsX()+2): # include under- and overflow too
            val = histoA.GetBinContent(i)-histoB.GetBinContent(i)
            ret.SetBinContent(i, val)
            ret.SetBinError(i, math.sqrt(val))

        return ret

class Transform:
    """Class to transform bin contents in an arbitrary way."""
    def __init__(self, name, histo, func, title=""):
        """Constructor.

        Argument:
        name  -- String for name of the resulting histogram
        histo -- String for a source histogram (needs to be cumulative)
        func  -- Function to operate on the bin content
        """
        self._name = name
        self._histo = histo
        self._func = func
        self._title = title

    def __str__(self):
        """String representation, returns the name"""
        return self._name

    def create(self, tdirectory):
        """Create and return the transformed histogram from a TDirectory"""
        histo = _getOrCreateObject(tdirectory, self._histo)
        if not histo:
            return None

        ret = histo.Clone(self._name)
        ret.SetTitle(self._title)

        # Disable canExtend if it is set, otherwise setting the
        # overflow bin will extend instead, possibly causing weird
        # effects downstream
        ret.SetCanExtend(False)

        for i in range(0, histo.GetNbinsX()+2):
            ret.SetBinContent(i, self._func(histo.GetBinContent(i)))
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

        for i in range(1, hassoc.GetNbinsX()+1):
            numerVal = hassoc.GetBinContent(i) - hdup.GetBinContent(i)
            denomVal = hreco.GetBinContent(i)

            fakedupVal = (1 - numerVal / denomVal) if denomVal != 0.0 else 0.0
            errVal = math.sqrt(fakedupVal*(1-fakedupVal)/denomVal) if (denomVal != 0.0 and fakedupVal <= 1) else 0.0

            hfakedup.SetBinContent(i, fakedupVal)
            hfakedup.SetBinError(i, errVal)

        return hfakedup

class CutEfficiency:
    """Class for making a cut efficiency histograms.

          N after cut
    eff = -----------
            N total
    """
    def __init__(self, name, histo, title=""):
        """Constructor

        Arguments:
        name  -- String for name of the resulting histogram
        histo -- String for a source histogram (needs to be cumulative)
        """
        self._name = name
        self._histo = histo
        self._title = title

    def __str__(self):
        """String representation, returns the name"""
        return self._name

    def create(self, tdirectory):
        """Create and return the cut efficiency histogram from a TDirectory"""
        histo = _getOrCreateObject(tdirectory, self._histo)
        if not histo:
            return None

        # infer cumulative direction from the under/overflow bins
        ascending = histo.GetBinContent(0) < histo.GetBinContent(histo.GetNbinsX())
        if ascending:
            n_tot = histo.GetBinContent(histo.GetNbinsX())
        else:
            n_tot = histo.GetBinContent(0)

        if n_tot == 0:
            return histo

        ret = histo.Clone(self._name)
        ret.SetTitle(self._title)

        # calculate efficiency
        for i in range(1, histo.GetNbinsX()+1):
            n = histo.GetBinContent(i)
            val = n/n_tot
            errVal = math.sqrt(val*(1-val)/n_tot)
            ret.SetBinContent(i, val)
            ret.SetBinError(i, errVal)
        return ret

class AggregateBins:
    """Class to create a histogram by aggregating bins of another histogram to a bin of the resulting histogram."""
    def __init__(self, name, histoName, mapping, normalizeTo=None, scale=None, renameBin=None, ignoreMissingBins=False, minExistingBins=None, originalOrder=False, reorder=None):
        """Constructor.

        Arguments:
        name      -- String for the name of the resulting histogram
        histoName -- String for the name of the source histogram
        mapping   -- Dictionary for mapping the bins (see below)

        Keyword arguments:
        normalizeTo -- Optional string of a bin label in the source histogram. If given, all bins of the resulting histogram are divided by the value of this bin.
        scale       -- Optional number for scaling the histogram (passed to ROOT.TH1.Scale())
        renameBin   -- Optional function (string -> string) to rename the bins of the input histogram
        originalOrder -- Boolean for using the order of bins in the histogram (default False)
        reorder     -- Optional function to reorder the bins

        Mapping structure (mapping):

        Dictionary (you probably want to use collections.OrderedDict)
        should be a mapping from the destination bin label to a list
        of source bin labels ("dst -> [src]").
        """
        self._name = name
        self._histoName = histoName
        self._mapping = mapping
        self._normalizeTo = normalizeTo
        self._scale = scale
        self._renameBin = renameBin
        self._ignoreMissingBins = ignoreMissingBins
        self._minExistingBins = minExistingBins
        self._originalOrder = originalOrder
        self._reorder = reorder
        if self._originalOrder and self._reorder is not None:
            raise Exception("reorder is not None and originalOrder is True, please set only one of them")

    def __str__(self):
        """String representation, returns the name"""
        return self._name

    def create(self, tdirectory):
        """Create and return the histogram from a TDirectory"""
        th1 = _getOrCreateObject(tdirectory, self._histoName)
        if th1 is None:
            return None

        binLabels = [""]*len(self._mapping)
        binValues = [None]*len(self._mapping)

        # TH1 can't really be used as a map/dict, so convert it here:
        values = _th1ToOrderedDict(th1, self._renameBin)

        binIndexOrder = [] # for reordering bins if self._originalOrder is True
        for i, (key, labels) in enumerate(six.iteritems(self._mapping)):
            sumTime = 0.
            sumErrorSq = 0.
            nsum = 0
            for l in labels:
                try:
                    sumTime += values[l][0]
                    sumErrorSq += values[l][1]**2
                    nsum += 1
                except KeyError:
                    pass

            if nsum > 0:
                binValues[i] = (sumTime, math.sqrt(sumErrorSq))
            binLabels[i] = key

            ivalue = len(values)+1
            if len(labels) > 0:
                # first label doesn't necessarily exist (especially for
                # the iteration timing plots), so let's test them all
                for lab in labels:
                    if lab in values:
                        ivalue = values.keys().index(lab)
                        break
            binIndexOrder.append( (ivalue, i) )

        if self._originalOrder:
            binIndexOrder.sort(key=lambda t: t[0])
            tmpVal = []
            tmpLab = []
            for i in range(0, len(binValues)):
                fromIndex = binIndexOrder[i][1]
                tmpVal.append(binValues[fromIndex])
                tmpLab.append(binLabels[fromIndex])
            binValues = tmpVal
            binLabels = tmpLab
        if self._reorder is not None:
            order = self._reorder(tdirectory, binLabels)
            binValues = [binValues[i] for i in order]
            binLabels = [binLabels[i] for i in order]

        if self._minExistingBins is not None and (len(binValues)-binValues.count(None)) < self._minExistingBins:
            return None

        if self._ignoreMissingBins:
            for i, val in enumerate(binValues):
                if val is None:
                    binLabels[i] = None
            binValues = [v for v in binValues if v is not None]
            binLabels = [v for v in binLabels if v is not None]
            if len(binValues) == 0:
                return None

        result = ROOT.TH1F(self._name, self._name, len(binValues), 0, len(binValues))
        for i, (value, label) in enumerate(zip(binValues, binLabels)):
            if value is not None:
                result.SetBinContent(i+1, value[0])
                result.SetBinError(i+1, value[1])
            result.GetXaxis().SetBinLabel(i+1, label)

        if self._normalizeTo is not None:
            bin = th1.GetXaxis().FindBin(self._normalizeTo)
            if bin <= 0:
                print("Trying to normalize {name} to {binlabel}, which does not exist".format(name=self._name, binlabel=self._normalizeTo))
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
        for key, histoName in six.iteritems(self._mapping):
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

        for i in range(1, xhisto.GetNbinsX()+1):
            x.append(xhisto.GetBinContent(i))
            xerrup.append(xhisto.GetBinError(i))
            xerrdown.append(xhisto.GetBinError(i))

            y.append(yhisto.GetBinContent(i))
            yerrup.append(yhisto.GetBinError(i))
            yerrdown.append(yhisto.GetBinError(i))

            z.append(xhisto.GetXaxis().GetBinUpEdge(i))

        # If either axis has only zeroes, no graph makes no point
        if x.count(0.0) == len(x) or y.count(0.0) == len(y):
            return None

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

def _drawFrame(pad, bounds, zmax=None, xbinlabels=None, xbinlabelsize=None, xbinlabeloption=None, ybinlabels=None, suffix=""):
    """Function to draw a frame

    Arguments:
    pad    -- TPad to where the frame is drawn
    bounds -- List or 4-tuple for (xmin, ymin, xmax, ymax)

    Keyword arguments:
    zmax            -- Maximum Z, needed for TH2 histograms
    xbinlabels      -- Optional list of strings for x axis bin labels
    xbinlabelsize   -- Optional number for the x axis bin label size
    xbinlabeloption -- Optional string for the x axis bin options (passed to ROOT.TH1.LabelsOption())
    suffix          -- Optional string for a postfix of the frame name
    """
    if xbinlabels is None and ybinlabels is None:
        frame = pad.DrawFrame(*bounds)
    else:
        # Special form needed if want to set x axis bin labels
        nbins = len(xbinlabels)
        if ybinlabels is None:
            frame = ROOT.TH1F("hframe"+suffix, "", nbins, bounds[0], bounds[2])
            frame.SetMinimum(bounds[1])
            frame.SetMaximum(bounds[3])
            frame.GetYaxis().SetLimits(bounds[1], bounds[3])
        else:
            ybins = len(ybinlabels)
            frame = ROOT.TH2F("hframe"+suffix, "", nbins,bounds[0],bounds[2], ybins,bounds[1],bounds[3])
            frame.SetMaximum(zmax)

        frame.SetBit(ROOT.TH1.kNoStats)
        frame.SetBit(ROOT.kCanDelete)
        frame.Draw("")

        xaxis = frame.GetXaxis()
        for i in range(nbins):
            xaxis.SetBinLabel(i+1, xbinlabels[i])
        if xbinlabelsize is not None:
            xaxis.SetLabelSize(xbinlabelsize)
        if xbinlabeloption is not None:
            frame.LabelsOption(xbinlabeloption)

        if ybinlabels is not None:
            yaxis = frame.GetYaxis()
            for i, lab in enumerate(ybinlabels):
                yaxis.SetBinLabel(i+1, lab)
            if xbinlabelsize is not None:
                yaxis.SetLabelSize(xbinlabelsize)
            if xbinlabeloption is not None:
                frame.LabelsOption(xbinlabeloption, "Y")

    return frame

class Frame:
    """Class for creating and managing a frame for a simple, one-pad plot"""
    def __init__(self, pad, bounds, zmax, nrows, xbinlabels=None, xbinlabelsize=None, xbinlabeloption=None, ybinlabels=None):
        self._pad = pad
        self._frame = _drawFrame(pad, bounds, zmax, xbinlabels, xbinlabelsize, xbinlabeloption, ybinlabels)

        yoffsetFactor = 1
        xoffsetFactor = 1
        if nrows == 2:
            yoffsetFactor *= 2
            xoffsetFactor *= 2
        elif nrows >= 5:
            yoffsetFactor *= 1.5
            xoffsetFactor *= 1.5
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
    def __init__(self, pad, bounds, zmax, ratioBounds, ratioFactor, nrows, xbinlabels=None, xbinlabelsize=None, xbinlabeloption=None, ratioYTitle=_ratioYTitle):
        self._parentPad = pad
        self._pad = pad.cd(1)
        if xbinlabels is not None:
            self._frame = _drawFrame(self._pad, bounds, zmax, [""]*len(xbinlabels))
        else:
            self._frame = _drawFrame(self._pad, bounds, zmax)
        self._padRatio = pad.cd(2)
        self._frameRatio = _drawFrame(self._padRatio, ratioBounds, zmax, xbinlabels, xbinlabelsize, xbinlabeloption)

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

        self._frameRatio.GetYaxis().SetTitle(ratioYTitle)

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

    def setXTitleOffset(self, offset):
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
        # set top view
        epsilon = 1e-7
        self._pad.SetPhi(epsilon)
        self._pad.SetTheta(90+epsilon)

        self._firstHisto.GetXaxis().SetTitleOffset(self._xtitleoffset)
        self._firstHisto.GetYaxis().SetTitleOffset(self._ytitleoffset)

        if hasattr(self, "_xtitle"):
            self._firstHisto.GetXaxis().SetTitle(self._xtitle)
        if hasattr(self, "_xtitlesize"):
            self._firstHisto.GetXaxis().SetTitleSize(self._xtitlesize)
        if hasattr(self, "_xlabelsize"):
            self._firstHisto.GetXaxis().SetLabelSize(self._labelsize)
        if hasattr(self, "_ytitle"):
            self._firstHisto.GetYaxis().SetTitle(self._ytitle)
        if hasattr(self, "_ytitlesize"):
            self._firstHisto.GetYaxis().SetTitleSize(self._ytitlesize)
        if hasattr(self, "_ytitleoffset"):
            self._firstHisto.GetYaxis().SetTitleOffset(self._ytitleoffset)

class PlotText:
    """Abstraction on top of TLatex"""
    def __init__(self, x, y, text, size=None, bold=True, align="left", color=ROOT.kBlack, font=None):
        """Constructor.

        Arguments:
        x     -- X coordinate of the text (in NDC)
        y     -- Y coordinate of the text (in NDC)
        text  -- String to draw
        size  -- Size of text (None for the default value, taken from gStyle)
        bold  -- Should the text be bold?
        align -- Alignment of text (left, center, right)
        color -- Color of the text
        font  -- Specify font explicitly
        """
        self._x = x
        self._y = y
        self._text = text

        self._l = ROOT.TLatex()
        self._l.SetNDC()
        if not bold:
            self._l.SetTextFont(self._l.GetTextFont()-20) # bold -> normal
        if font is not None:
            self._l.SetTextFont(font)
        if size is not None:
            self._l.SetTextSize(size)
        if isinstance(align, str):
            if align.lower() == "left":
                self._l.SetTextAlign(11)
            elif align.lower() == "center":
                self._l.SetTextAlign(21)
            elif align.lower() == "right":
                self._l.SetTextAlign(31)
            else:
                raise Exception("Error: Invalid option '%s' for text alignment! Options are: 'left', 'center', 'right'."%align)
        else:
            self._l.SetTextAlign(align)
        self._l.SetTextColor(color)

    def Draw(self, options=None):
        """Draw the text to the current TPad.

        Arguments:
        options -- For interface compatibility, ignored

        Provides interface compatible with ROOT's drawable objects.
        """
        self._l.DrawLatex(self._x, self._y, self._text)


class PlotTextBox:
    """Class for drawing text and a background box."""
    def __init__(self, xmin, ymin, xmax, ymax, lineheight=0.04, fillColor=ROOT.kWhite, transparent=True, **kwargs):
        """Constructor

        Arguments:
        xmin        -- X min coordinate of the box (NDC)
        ymin        -- Y min coordinate of the box (NDC) (if None, deduced automatically)
        xmax        -- X max coordinate of the box (NDC)
        ymax        -- Y max coordinate of the box (NDC)
        lineheight  -- Line height
        fillColor   -- Fill color of the box
        transparent -- Should the box be transparent? (in practive the TPave is not created)

        Keyword arguments are forwarded to constructor of PlotText
        """
        # ROOT.TPave Set/GetX1NDC() etc don't seem to work as expected.
        self._xmin = xmin
        self._xmax = xmax
        self._ymin = ymin
        self._ymax = ymax
        self._lineheight = lineheight
        self._fillColor = fillColor
        self._transparent = transparent
        self._texts = []
        self._textArgs = {}
        self._textArgs.update(kwargs)

        self._currenty = ymax

    def addText(self, text):
        """Add text to current position"""
        self._currenty -= self._lineheight
        self._texts.append(PlotText(self._xmin+0.01, self._currenty, text, **self._textArgs))

    def width(self):
        return self._xmax-self._xmin

    def move(self, dx=0, dy=0, dw=0, dh=0):
        """Move the box and the contained text objects

        Arguments:
        dx -- Movement in x (positive is to right)
        dy -- Movement in y (positive is to up)
        dw -- Increment of width (negative to decrease width)
        dh -- Increment of height (negative to decrease height)

        dx and dy affect to both box and text objects, dw and dh
        affect the box only.
        """
        self._xmin += dx
        self._xmax += dx
        if self._ymin is not None:
            self._ymin += dy
        self._ymax += dy

        self._xmax += dw
        if self._ymin is not None:
            self._ymin -= dh

        for t in self._texts:
            t._x += dx
            t._y += dy

    def Draw(self, options=""):
        """Draw the box and the text to the current TPad.

        Arguments:
        options -- Forwarded to ROOT.TPave.Draw(), and the Draw() of the contained objects
        """
        if not self._transparent:
            ymin = self.ymin
            if ymin is None:
                ymin = self.currenty - 0.01
            self._pave = ROOT.TPave(self.xmin, self.ymin, self.xmax, self.ymax, 0, "NDC")
            self._pave.SetFillColor(self.fillColor)
            self._pave.Draw(options)
        for t in self._texts:
            t.Draw(options)

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

class PlotEmpty:
    """Denotes an empty place in a group."""
    def __init__(self):
        pass

    def getName(self):
        return None

    def drawRatioUncertainty(self):
        return False

    def create(self, *args, **kwargs):
        pass

    def isEmpty(self):
        return True

    def getNumberOfHistograms(self):
        return 0

class Plot:
    """Represents one plot, comparing one or more histograms."""
    def __init__(self, name, **kwargs):
        """ Constructor.

        Arguments:
        name -- String for name of the plot, or Efficiency object

        Keyword arguments:
        fallback     -- Dictionary for specifying fallback (default None)
        outname      -- String for an output name of the plot (default None for the same as 'name')
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
        normalizeToNumberOfEvents -- Normalize histograms to number of events? If yes, the PlotFolder needs 'numberOfEventsHistogram' set to a histogram filled once per event (default False)
        profileX     -- Take histograms via ProfileX()? (default False)
        fitSlicesY   -- Take histograms via FitSlicesY() (default False)
        rebinX       -- rebin x axis (default None)
        scale        -- Scale histograms by a number (default None)
        xbinlabels   -- List of x axis bin labels (if given, default None)
        xbinlabelsize -- Size of x axis bin labels (default None)
        xbinlabeloption -- Option string for x axis bin labels (default None)
        removeEmptyBins -- Bool for removing empty bins, but only if histogram has bin labels (default False)
        printBins    -- Bool for printing bin values, but only if histogram has bin labels (default False)
        drawStyle    -- If "hist", draw as line instead of points (default None)
        drawCommand  -- Deliver this to Draw() (default: None for same as drawStyle)
        lineWidth    -- If drawStyle=="hist", the width of line (default 2)
        legendDx     -- Float for moving TLegend in x direction for separate=True (default None)
        legendDy     -- Float for moving TLegend in y direction for separate=True (default None)
        legendDw     -- Float for changing TLegend width for separate=True (default None)
        legendDh     -- Float for changing TLegend height for separate=True (default None)
        legend       -- Bool to enable/disable legend (default True)
        adjustMarginLeft  -- Float for adjusting left margin (default None)
        adjustMarginRight  -- Float for adjusting right margin (default None)
        ratio        -- Possibility to disable ratio for this particular plot (default None)
        ratioYmin    -- Float for y axis minimum in ratio pad (default: list of values)
        ratioYmax    -- Float for y axis maximum in ratio pad (default: list of values)
        ratioFit     -- Fit straight line in ratio? (default None)
        ratioUncertainty -- Plot uncertainties on ratio? (default True)
        ratioCoverageXrange -- Range of x axis values (xmin,xmax) to limit the automatic ratio y axis range calculation to (default None for disabled)
        histogramModifier -- Function to be called in create() to modify the histograms (default None)
        """
        self._name = name

        def _set(attr, default):
            setattr(self, "_"+attr, kwargs.get(attr, default))

        _set("fallback", None)
        _set("outname", None)

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
        _set("normalizeToNumberOfEvents", False)
        _set("profileX", False)
        _set("fitSlicesY", False)
        _set("rebinX", None)

        _set("scale", None)
        _set("xbinlabels", None)
        _set("xbinlabelsize", None)
        _set("xbinlabeloption", None)
        _set("removeEmptyBins", False)
        _set("printBins", False)

        _set("drawStyle", "EP")
        _set("drawCommand", None)
        _set("lineWidth", 2)

        _set("legendDx", None)
        _set("legendDy", None)
        _set("legendDw", None)
        _set("legendDh", None)
        _set("legend", True)

        _set("adjustMarginLeft", None)
        _set("adjustMarginRight", None)

        _set("ratio", None)
        _set("ratioYmin", [0, 0.2, 0.5, 0.7, 0.8, 0.9, 0.95])
        _set("ratioYmax", [1.05, 1.1, 1.2, 1.3, 1.5, 1.8, 2, 2.5, 3, 4, 5])
        _set("ratioFit", None)
        _set("ratioUncertainty", True)
        _set("ratioCoverageXrange", None)

        _set("histogramModifier", None)

        self._histograms = []

    def setProperties(self, **kwargs):
        for name, value in six.iteritems(kwargs):
            if not hasattr(self, "_"+name):
                raise Exception("No attribute '%s'" % name)
            setattr(self, "_"+name, value)

    def clone(self, **kwargs):
        if not self.isEmpty():
            raise Exception("Plot can be cloned only before histograms have been created")
        cl = copy.copy(self)
        cl.setProperties(**kwargs)
        return cl

    def getNumberOfHistograms(self):
        """Return number of existing histograms."""
        return len([h for h in self._histograms if h is not None])

    def isEmpty(self):
        """Return true if there are no histograms created for the plot"""
        return self.getNumberOfHistograms() == 0

    def isTGraph2D(self):
        for h in self._histograms:
            if isinstance(h, ROOT.TGraph2D):
                return True
        return False

    def isRatio(self, ratio):
        if self._ratio is None:
            return ratio
        return ratio and self._ratio

    def getName(self):
        if self._outname is not None:
            return self._outname
        if isinstance(self._name, list):
            return str(self._name[0])
        else:
            return str(self._name)

    def drawRatioUncertainty(self):
        """Return true if the ratio uncertainty should be drawn"""
        return self._ratioUncertainty

    def _createOne(self, name, index, tdir, nevents):
        """Create one histogram from a TDirectory."""
        if tdir == None:
            return None

        # If name is a list, pick the name by the index
        if isinstance(name, list):
            name = name[index]

        h = _getOrCreateObject(tdir, name)
        if h is not None and self._normalizeToNumberOfEvents and nevents is not None and nevents != 0:
            h.Scale(1.0/nevents)
        return h

    def create(self, tdirNEvents, requireAllHistograms=False):
        """Create histograms from list of TDirectories"""
        self._histograms = [self._createOne(self._name, i, tdirNEvent[0], tdirNEvent[1]) for i, tdirNEvent in enumerate(tdirNEvents)]

        if self._fallback is not None:
            profileX = [self._profileX]*len(self._histograms)
            for i in range(0, len(self._histograms)):
                if self._histograms[i] is None:
                    self._histograms[i] = self._createOne(self._fallback["name"], i, tdirNEvents[i][0], tdirNEvents[i][1])
                    profileX[i] = self._fallback.get("profileX", self._profileX)

        if self._histogramModifier is not None:
            self._histograms = self._histogramModifier(self._histograms)

        if len(self._histograms) > len(_plotStylesColor):
            raise Exception("More histograms (%d) than there are plot styles (%d) defined. Please define more plot styles in this file" % (len(self._histograms), len(_plotStylesColor)))

        # Modify histograms here in case self._name returns numbers
        # and self._histogramModifier creates the histograms from
        # these numbers
        def _modifyHisto(th1, profileX):
            if th1 is None:
                return None

            if profileX:
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

        if self._fallback is not None:
            self._histograms = map(_modifyHisto, self._histograms, profileX)
        else:
            self._histograms = map(lambda h: _modifyHisto(h, self._profileX), self._histograms)
        if requireAllHistograms and None in self._histograms:
            self._histograms = [None]*len(self._histograms)

    def _setStats(self, histos, startingX, startingY):
        """Set stats box."""
        if not self._stat:
            for h in histos:
                if h is not None and hasattr(h, "SetStats"):
                    h.SetStats(0)
            return

        def _doStats(h, col, dy):
            if h is None:
                return
            h.SetStats(True)

            if self._fit and h.GetEntries() > 0.5:
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
                st.SetOptFit(0o010)
                st.SetOptStat(1001)
            st.SetX1NDC(startingX)
            st.SetX2NDC(startingX+0.3)
            st.SetY1NDC(startingY+dy)
            st.SetY2NDC(startingY+dy+0.15)
            st.SetTextColor(col)

        dy = 0.0
        for i, h in enumerate(histos):
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
            if h.GetSumw2().fN <= 0: # to suppress warning
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
            if verbose:
                print("No histograms for plot {name}".format(name=self.getName()))
            return

        # Extract x bin labels, make sure that only bins with same
        # label are compared with each other
        histosHaveBinLabels = len(histos[0].GetXaxis().GetBinLabel(1)) > 0
        xbinlabels = self._xbinlabels
        ybinlabels = None
        if xbinlabels is None:
            if histosHaveBinLabels:
                xbinlabels = _mergeBinLabelsX(histos)
                if isinstance(histos[0], ROOT.TH2):
                    ybinlabels = _mergeBinLabelsY(histos)

                if len(histos) > 1: # don't bother if only one histogram
                    # doing this for TH2 is pending for use case, for now there is only 1 histogram/plot for TH2
                    histos = _th1IncludeOnlyBins(histos, xbinlabels)
                    self._tmp_histos = histos # need to keep these in memory too ...

        # Remove empty bins, but only if histograms have bin labels
        if self._removeEmptyBins and histosHaveBinLabels:
            # at this point, all histograms have been "equalized" by their x binning and labels
            # therefore remove bins which are empty in all histograms
            if isinstance(histos[0], ROOT.TH2):
                (histos, xbinlabels, ybinlabels) = _th2RemoveEmptyBins(histos, xbinlabels, ybinlabels)
            else:
                (histos, xbinlabels) = _th1RemoveEmptyBins(histos, xbinlabels)
            self._tmp_histos = histos # need to keep these in memory too ...
            if len(histos) == 0:
                if verbose:
                    print("No histograms with non-empty bins for plot {name}".format(name=self.getName()))
                return

        if self._printBins and histosHaveBinLabels:
            print("####################")
            print(self._name)
            width = max([len(l) for l in xbinlabels])
            tmp = "%%-%ds " % width
            for b in range(1, histos[0].GetNbinsX()+1):
                s = tmp % xbinlabels[b-1]
                for h in histos:
                    s += "%.3f " % h.GetBinContent(b)
                print(s)
            print()

        bounds = _findBounds(histos, self._ylog,
                             xmin=self._xmin, xmax=self._xmax,
                             ymin=self._ymin, ymax=self._ymax)
        zmax = None
        if isinstance(histos[0], ROOT.TH2):
            zmax = max([h.GetMaximum() for h in histos])

        # need to keep these in memory
        self._mainAdditional = []
        self._ratioAdditional = []

        if ratio:
            self._ratios = _calculateRatios(histos, self._ratioUncertainty) # need to keep these in memory too ...
            ratioHistos = [h for h in [r.getRatio() for r in self._ratios[1:]] if h is not None]

            if len(ratioHistos) > 0:
                ratioBoundsY = _findBoundsY(ratioHistos, ylog=False, ymin=self._ratioYmin, ymax=self._ratioYmax, coverage=0.68, coverageRange=self._ratioCoverageXrange)
            else:
                ratioBoundsY = (0.9, 1,1) # hardcoded default in absence of valid ratio calculations

            if self._ratioFit is not None:
                for i, rh in enumerate(ratioHistos):
                    tf_line = ROOT.TF1("line%d"%i, "[0]+x*[1]")
                    tf_line.SetRange(self._ratioFit["rangemin"], self._ratioFit["rangemax"])
                    fitres = rh.Fit(tf_line, "RINSQ")
                    tf_line.SetLineColor(rh.GetMarkerColor())
                    tf_line.SetLineWidth(2)
                    self._ratioAdditional.append(tf_line)
                    box = PlotTextBox(xmin=self._ratioFit.get("boxXmin", 0.14), ymin=None, # None for automatix
                                      xmax=self._ratioFit.get("boxXmax", 0.35), ymax=self._ratioFit.get("boxYmax", 0.09),
                                      color=rh.GetMarkerColor(), font=43, size=11, lineheight=0.02)
                    box.move(dx=(box.width()+0.01)*i)
                    #box.addText("Const: %.4f" % fitres.Parameter(0))
                    #box.addText("Slope: %.4f" % fitres.Parameter(1))
                    box.addText("Const: %.4f#pm%.4f" % (fitres.Parameter(0), fitres.ParError(0)))
                    box.addText("Slope: %.4f#pm%.4f" % (fitres.Parameter(1), fitres.ParError(1)))
                    self._mainAdditional.append(box)


        # Create bounds before stats in order to have the
        # SetRangeUser() calls made before the fit
        #
        # stats is better to be called before frame, otherwise get
        # mess in the plot (that frame creation cleans up)
        if ratio:
            pad.cd(1)
        self._setStats(histos, self._statx, self._staty)

        # Create frame
        if isTGraph2D:
            frame = FrameTGraph2D(pad, bounds, histos, ratioOrig, ratioFactor)
        else:
            if ratio:
                ratioBounds = (bounds[0], ratioBoundsY[0], bounds[2], ratioBoundsY[1])
                frame = FrameRatio(pad, bounds, zmax, ratioBounds, ratioFactor, nrows, xbinlabels, self._xbinlabelsize, self._xbinlabeloption)
            else:
                frame = Frame(pad, bounds, zmax, nrows, xbinlabels, self._xbinlabelsize, self._xbinlabeloption, ybinlabels=ybinlabels)

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
        if self._xtitle == 'Default':
            frame.setXTitle( histos[0].GetXaxis().GetTitle() )
        elif self._xtitle is not None:
            frame.setXTitle(self._xtitle)
        if self._xtitlesize is not None:
            frame.setXTitleSize(self._xtitlesize)
        if self._xtitleoffset is not None:
            frame.setXTitleOffset(self._xtitleoffset)
        if self._xlabelsize is not None:
            frame.setXLabelSize(self._xlabelsize)
        if self._ytitle == 'Default':
            frame.setYTitle( histos[0].GetYaxis().GetTitle() )
        elif self._ytitle is not None:
            frame.setYTitle(self._ytitle)
        if self._ytitlesize is not None:
            frame.setYTitleSize(self._ytitlesize)
        if self._ytitleoffset is not None:
            frame.setYTitleOffset(self._ytitleoffset)
        if self._ztitle is not None:
            frame.setZTitle(self._ztitle)
        if self._ztitleoffset is not None:
            frame.setZTitleOffset(self._ztitleoffset)
        if self._adjustMarginLeft is not None:
            frame.adjustMarginLeft(self._adjustMarginLeft)
        if self._adjustMarginRight is not None:
            frame.adjustMarginRight(self._adjustMarginRight)
        elif "z" in opt:
            frame.adjustMarginLeft(0.03)
            frame.adjustMarginRight(0.08)

        # Draw histograms
        if ratio:
            frame._pad.cd()

        for i, h in enumerate(histos):
            o = opt
            if isTGraph2D and i == 0:
                o = o.replace("sames", "")
            h.Draw(o)

        for addl in self._mainAdditional:
            addl.Draw("same")

        # Draw ratios
        if ratio and len(histos) > 0:
            frame._padRatio.cd()
            firstRatio = self._ratios[0].getRatio()
            if self._ratioUncertainty and firstRatio is not None:
                firstRatio.SetFillStyle(1001)
                firstRatio.SetFillColor(ROOT.kGray)
                firstRatio.SetLineColor(ROOT.kGray)
                firstRatio.SetMarkerColor(ROOT.kGray)
                firstRatio.SetMarkerSize(0)
                self._ratios[0].draw("E2")
                frame._padRatio.RedrawAxis("G") # redraw grid on top of the uncertainty of denominator
            for r in self._ratios[1:]:
                r.draw()

            for addl in self._ratioAdditional:
                addl.Draw("same")

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

class PlotGroup(object):
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
        legend   -- Bool for disabling legend (default True for legend being enabled)
        overrideLegendLabels -- List of strings for legend labels, if given, these are used instead of the ones coming from Plotter (default None)
        onlyForPileup  -- Plots this group only for pileup samples
        """
        super(PlotGroup, self).__init__()

        self._name = name
        self._plots = plots

        def _set(attr, default):
            setattr(self, "_"+attr, kwargs.get(attr, default))

        _set("ncols", 2)

        _set("legendDx", None)
        _set("legendDy", None)
        _set("legendDw", None)
        _set("legendDh", None)
        _set("legend", True)

        _set("overrideLegendLabels", None)

        _set("onlyForPileup", False)

        self._ratioFactor = 1.25

    def setProperties(self, **kwargs):
        for name, value in six.iteritems(kwargs):
            if not hasattr(self, "_"+name):
                raise Exception("No attribute '%s'" % name)
            setattr(self, "_"+name, value)

    def getName(self):
        return self._name

    def getPlots(self):
        return self._plots

    def remove(self, name):
        for i, plot in enumerate(self._plots):
            if plot.getName() == name:
                del self._plots[i]
                return
        raise Exception("Did not find Plot '%s' from PlotGroup '%s'" % (name, self._name))

    def clear(self):
        self._plots = []

    def append(self, plot):
        self._plots.append(plot)

    def getPlot(self, name):
        for plot in self._plots:
            if plot.getName() == name:
                return plot
        raise Exception("No Plot named '%s'" % name)

    def onlyForPileup(self):
        """Return True if the PlotGroup is intended only for pileup samples"""
        return self._onlyForPileup

    def create(self, tdirectoryNEvents, requireAllHistograms=False):
        """Create histograms from a list of TDirectories.

        Arguments:
        tdirectoryNEvents    -- List of (TDirectory, nevents) pairs
        requireAllHistograms -- If True, a plot is produced if histograms from all files are present (default: False)
        """
        for plot in self._plots:
            plot.create(tdirectoryNEvents, requireAllHistograms)

    def draw(self, legendLabels, prefix=None, separate=False, saveFormat=".pdf", ratio=True, directory=""):
        """Draw the histograms using values for a given algorithm.

        Arguments:
        legendLabels  -- List of strings for legend labels (corresponding to the tdirectories in create())
        prefix        -- Optional string for file name prefix (default None)
        separate      -- Save the plots of a group to separate files instead of a file per group (default False)
        saveFormat   -- String specifying the plot format (default '.pdf')
        ratio        -- Add ratio to the plot (default True)
        directory     -- Directory where to save the file (default "")
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
            return self._drawSeparate(legendLabels, prefix, saveFormat, ratio, directory)

        cwidth = 500*self._ncols
        nrows = int((len(self._plots)+self._ncols-1)/self._ncols) # this should work also for odd n
        cheight = 500 * nrows

        if ratio:
            cheight = int(cheight*self._ratioFactor)

        canvas = _createCanvas(self._name, cwidth, cheight)

        canvas.Divide(self._ncols, nrows)
        if ratio:
            for i, plot in enumerate(self._plots):
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

        return self._save(canvas, saveFormat, prefix=prefix, directory=directory)

    def _drawSeparate(self, legendLabels, prefix, saveFormat, ratio, directory):
        """Internal method to do the drawing to separate files per Plot instead of a file per PlotGroup"""
        width = 500
        height = 500

        canvas = _createCanvas(self._name+"Single", width, height)
        canvasRatio = _createCanvas(self._name+"SingleRatio", width, int(height*self._ratioFactor))

        # from TDRStyle
        for c in [canvas, canvasRatio]:
            c.SetTopMargin(0.05)
            c.SetBottomMargin(0.13)
            c.SetLeftMargin(0.16)
            c.SetRightMargin(0.05)

        lx1def = 0.6
        lx2def = 0.95
        ly1def = 0.85
        ly2def = 0.95

        ret = []

        for plot in self._plots:
            if plot.isEmpty():
                continue

            ratioForThisPlot = plot.isRatio(ratio)
            c = canvas
            if ratioForThisPlot:
                c = canvasRatio
                c.cd()
                self._modifyPadForRatio(c)

            # Draw plot to canvas
            c.cd()
            plot.draw(c, ratioForThisPlot, self._ratioFactor, 1)

            if plot._legend:
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

                c.cd()
                legend = self._createLegend(plot, legendLabels, lx1, ly1, lx2, ly2, textSize=0.03,
                                            denomUncertainty=(ratioForThisPlot and plot.drawRatioUncertainty))

            ret.extend(self._save(c, saveFormat, prefix=prefix, postfix="/"+plot.getName(), single=True, directory=directory))
        return ret

    def _modifyPadForRatio(self, pad):
        """Internal method to set divide a pad to two for ratio plots"""
        _modifyPadForRatio(pad, self._ratioFactor)

    def _createLegend(self, plot, legendLabels, lx1, ly1, lx2, ly2, textSize=0.016, denomUncertainty=True):
        if not self._legend:
            return None

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

    def _save(self, canvas, saveFormat, prefix=None, postfix=None, single=False, directory=""):
        # Save the canvas to file and clear
        name = self._name
        if not os.path.exists(directory+'/'+name):
            os.makedirs(directory+'/'+name)
        if prefix is not None:
            name = prefix+name
        if postfix is not None:
            name = name+postfix
        name = os.path.join(directory, name)

        if not verbose: # silence saved file printout
            backup = ROOT.gErrorIgnoreLevel
            ROOT.gErrorIgnoreLevel = ROOT.kWarning
        canvas.SaveAs(name+saveFormat)
        if not verbose:
            ROOT.gErrorIgnoreLevel = backup

        if single:
            canvas.Clear()
            canvas.SetLogx(False)
            canvas.SetLogy(False)
        else:
            canvas.Clear("D") # keep subpads

        return [name+saveFormat]

class PlotOnSideGroup(PlotGroup):
    """Resembles DQM GUI's "On side" layout.

    Like PlotGroup, but has only a description of a single plot. The
    plot is drawn separately for each file. Useful for 2D histograms."""

    def __init__(self, name, plot, ncols=2, onlyForPileup=False):
        super(PlotOnSideGroup, self).__init__(name, [], ncols=ncols, legend=False, onlyForPileup=onlyForPileup)
        self._plot = plot
        self._plot.setProperties(ratio=False)

    def append(self, *args, **kwargs):
        raise Exception("PlotOnSideGroup.append() is not implemented")

    def create(self, tdirectoryNEvents, requireAllHistograms=False):
        self._plots = []
        for element in tdirectoryNEvents:
            pl = self._plot.clone()
            pl.create([element], requireAllHistograms)
            self._plots.append(pl)

    def draw(self, *args, **kwargs):
        kargs = copy.copy(kwargs)
        kargs["ratio"] = False
        kargs["separate"] = False
        return super(PlotOnSideGroup, self).draw(*args, **kargs)

class PlotFolder:

    """Represents a collection of PlotGroups, produced from a single folder in a DQM file"""
    def __init__(self, *plotGroups, **kwargs):
        """Constructor.

        Arguments:
        plotGroups     -- List of PlotGroup objects

        Keyword arguments
        loopSubFolders -- Should the subfolders be looped over? (default: True)
        onlyForPileup  -- Plots this folder only for pileup samples
        onlyForElectron -- Plots this folder only for electron samples
        onlyForConversion -- Plots this folder only for conversion samples
        onlyForBHadron -- Plots this folder only for B-hadron samples
        purpose        -- html.PlotPurpose member class for the purpose of the folder, used for grouping of the plots to the HTML pages
        page           -- Optional string for the page in HTML generatin
        section        -- Optional string for the section within a page in HTML generation
        numberOfEventsHistogram -- Optional path to histogram filled once per event. Needed if there are any plots normalized by number of events. Path is relative to "possibleDqmFolders".
        """
        self._plotGroups = list(plotGroups)
        self._loopSubFolders = kwargs.pop("loopSubFolders", True)
        self._onlyForPileup = kwargs.pop("onlyForPileup", False)
        self._onlyForElectron = kwargs.pop("onlyForElectron", False)
        self._onlyForConversion = kwargs.pop("onlyForConversion", False)
        self._onlyForBHadron = kwargs.pop("onlyForBHadron", False)
        self._purpose = kwargs.pop("purpose", None)
        self._page = kwargs.pop("page", None)
        self._section = kwargs.pop("section", None)
        self._numberOfEventsHistogram = kwargs.pop("numberOfEventsHistogram", None)
        if len(kwargs) > 0:
            raise Exception("Got unexpected keyword arguments: "+ ",".join(kwargs.keys()))

    def loopSubFolders(self):
        """Return True if the PlotGroups of this folder should be applied to the all subfolders"""
        return self._loopSubFolders

    def onlyForPileup(self):
        """Return True if the folder is intended only for pileup samples"""
        return self._onlyForPileup

    def onlyForElectron(self):
        return self._onlyForElectron

    def onlyForConversion(self):
        return self._onlyForConversion

    def onlyForBHadron(self):
        return self._onlyForBHadron

    def getPurpose(self):
        return self._purpose

    def getPage(self):
        return self._page

    def getSection(self):
        return self._section

    def getNumberOfEventsHistogram(self):
        return self._numberOfEventsHistogram

    def append(self, plotGroup):
        self._plotGroups.append(plotGroup)

    def set(self, plotGroups):
        self._plotGroups = plotGroups

    def getPlotGroups(self):
        return self._plotGroups

    def getPlotGroup(self, name):
        for pg in self._plotGroups:
            if pg.getName() == name:
                return pg
        raise Exception("No PlotGroup named '%s'" % name)

    def create(self, dirsNEvents, labels, isPileupSample=True, requireAllHistograms=False):
        """Create histograms from a list of TFiles.

        Arguments:
        dirsNEvents   -- List of (TDirectory, nevents) pairs
        labels -- List of strings for legend labels corresponding the files
        isPileupSample -- Is sample pileup (some PlotGroups may limit themselves to pileup)
        requireAllHistograms -- If True, a plot is produced if histograms from all files are present (default: False)
        """

        if len(dirsNEvents) != len(labels):
            raise Exception("len(dirsNEvents) should be len(labels), now they are %d and %d" % (len(dirsNEvents), len(labels)))

        self._labels = labels

        for pg in self._plotGroups:
            if pg.onlyForPileup() and not isPileupSample:
                continue
            pg.create(dirsNEvents, requireAllHistograms)

    def draw(self, prefix=None, separate=False, saveFormat=".pdf", ratio=True, directory=""):
        """Draw and save all plots using settings of a given algorithm.

        Arguments:
        prefix   -- Optional string for file name prefix (default None)
        separate -- Save the plots of a group to separate files instead of a file per group (default False)
        saveFormat   -- String specifying the plot format (default '.pdf')
        ratio    -- Add ratio to the plot (default True)
        directory -- Directory where to save the file (default "")
        """
        ret = []

        for pg in self._plotGroups:
            ret.extend(pg.draw(self._labels, prefix=prefix, separate=separate, saveFormat=saveFormat, ratio=ratio, directory=directory))
        return ret


    # These are to be overridden by derived classes for customisation
    def translateSubFolder(self, dqmSubFolderName):
        """Method called to (possibly) translate a subfolder name to more 'readable' form

        The implementation in this (base) class just returns the
        argument. The idea is that a deriving class might want to do
        something more complex (like trackingPlots.TrackingPlotFolder
        does)
        """
        return dqmSubFolderName

    def iterSelectionName(self, plotFolderName, translatedDqmSubFolder):
        """Iterate over possible selections name (used in output directory name and legend) from the name of PlotterFolder, and a return value of translateSubFolder"""
        ret = ""
        if plotFolderName != "":
            ret += "_"+plotFolderName
        if translatedDqmSubFolder is not None:
            ret += "_"+translatedDqmSubFolder
        yield ret

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
    def __init__(self, name, possibleDqmFolders, dqmSubFolders, plotFolder, fallbackNames, fallbackDqmSubFolders, tableCreators):
        """
        Constructor

        Arguments:
        name               -- Name of the folder (is used in the output directory naming)
        possibleDqmFolders -- List of strings for possible directories of histograms in TFiles
        dqmSubFolders      -- List of lists of strings for list of subfolders per input file, or None if no subfolders
        plotFolder         -- PlotFolder object
        fallbackNames      -- List of names for backward compatibility (can be empty). These are used only by validation.Validation (class responsible of the release validation workflow) in case the reference file pointed by 'name' does not exist.
        fallbackDqmSubFolders -- List of dicts of (string->string) for mapping the subfolder names found in the first file to another names. Use case is comparing files that have different iteration naming convention.
        tableCreators      -- List of PlotterTableItem objects for tables to be created from this folder
        """
        self._name = name
        self._possibleDqmFolders = possibleDqmFolders
        self._plotFolder = plotFolder
        #self._dqmSubFolders = [map(lambda sf: DQMSubFolder(sf, self._plotFolder.translateSubFolder(sf)), lst) for lst in dqmSubFolders]
        if dqmSubFolders is None:
            self._dqmSubFolders = None
        else:
            # Match the subfolders between files in case the lists differ
            # equality is by the 'translated' name
            subfolders = {}
            for sf_list in dqmSubFolders:
                for sf in sf_list:
                    sf_translated = self._plotFolder.translateSubFolder(sf)
                    if sf_translated is not None and not sf_translated in subfolders:
                        subfolders[sf_translated] = DQMSubFolder(sf, sf_translated)

            self._dqmSubFolders = subfolders.values()
            self._dqmSubFolders.sort(key=lambda sf: sf.subfolder)

        self._fallbackNames = fallbackNames
        self._fallbackDqmSubFolders = fallbackDqmSubFolders
        self._tableCreators = tableCreators

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

    def onlyForElectron(self):
        return self._plotFolder.onlyForElectron()

    def onlyForConversion(self):
        return self._plotFolder.onlyForConversion()

    def onlyForBHadron(self):
        return self._plotFolder.onlyForBHadron()

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

        return [s for s in self._dqmSubFolders if self._plotFolder.limitSubFolder(limitOnlyTo, s.translated)]

    def getTableCreators(self):
        return self._tableCreators

    def getSelectionNameIterator(self, dqmSubFolder):
        """Get a generator for the 'selection name', looping over the name and fallbackNames"""
        for name in [self._name]+self._fallbackNames:
            for selname in self._plotFolder.iterSelectionName(name, dqmSubFolder.translated if dqmSubFolder is not None else None):
                yield selname

    def getSelectionName(self, dqmSubFolder):
        return next(self.getSelectionNameIterator(dqmSubFolder))

    def create(self, files, labels, dqmSubFolder, isPileupSample=True, requireAllHistograms=False):
        """Create histograms from a list of TFiles.
        Arguments:
        files  -- List of TFiles
        labels -- List of strings for legend labels corresponding the files
        dqmSubFolder -- DQMSubFolder object for a subfolder (or None for no subfolder)
        isPileupSample -- Is sample pileup (some PlotGroups may limit themselves to pileup)
        requireAllHistograms -- If True, a plot is produced if histograms from all files are present (default: False)
        """

        subfolder = dqmSubFolder.subfolder if dqmSubFolder is not None else None
        neventsHisto = self._plotFolder.getNumberOfEventsHistogram()
        dirsNEvents = []

        for tfile in files:
            ret = _getDirectoryDetailed(tfile, self._possibleDqmFolders, subfolder)
            # If file and any of possibleDqmFolders exist but subfolder does not, try the fallbacks
            if ret is GetDirectoryCode.SubDirNotExist:
                for fallbackFunc in self._fallbackDqmSubFolders:
                    fallback = fallbackFunc(subfolder)
                    if fallback is not None:
                        ret = _getDirectoryDetailed(tfile, self._possibleDqmFolders, fallback)
                        if ret is not GetDirectoryCode.SubDirNotExist:
                            break
            d = GetDirectoryCode.codesToNone(ret)
            nev = None
            if neventsHisto is not None and tfile is not None:
                hnev = _getObject(tfile, neventsHisto)
                if hnev is not None:
                    nev = hnev.GetEntries()
            dirsNEvents.append( (d, nev) )

        self._plotFolder.create(dirsNEvents, labels, isPileupSample, requireAllHistograms)

    def draw(self, *args, **kwargs):
        """Draw and save all plots using settings of a given algorithm."""
        return self._plotFolder.draw(*args, **kwargs)


class PlotterInstance:
    """Instance of plotter that knows the directory content, holds many folders."""
    def __init__(self, folders):
        self._plotterFolders = [f for f in folders if f is not None]

    def iterFolders(self, limitSubFoldersOnlyTo=None):
        for plotterFolder in self._plotterFolders:
            limitOnlyTo = None
            if limitSubFoldersOnlyTo is not None:
                limitOnlyTo = limitSubFoldersOnlyTo.get(plotterFolder.getName(), None)

            for dqmSubFolder in plotterFolder.getDQMSubFolders(limitOnlyTo=limitOnlyTo):
                yield plotterFolder, dqmSubFolder

# Helper for Plotter
class PlotterItem:
    def __init__(self, name, possibleDirs, plotFolder, fallbackNames=[], fallbackDqmSubFolders=[]):
        """ Constructor

        Arguments:
        name          -- Name of the folder (is used in the output directory naming)
        possibleDirs  -- List of strings for possible directories of histograms in TFiles
        plotFolder    -- PlotFolder object

        Keyword arguments
        fallbackNames -- Optional list of names for backward compatibility. These are used only by validation.Validation (class responsible of the release validation workflow) in case the reference file pointed by 'name' does not exist.
        fallbackDqmSubFolders -- Optional list of functions for (string->string) mapping the subfolder names found in the first file to another names (function should return None for no mapping). Use case is comparing files that have different iteration naming convention.
        """
        self._name = name
        self._possibleDirs = possibleDirs
        self._plotFolder = plotFolder
        self._fallbackNames = fallbackNames
        self._fallbackDqmSubFolders = fallbackDqmSubFolders
        self._tableCreators = []

    def getName(self):
        return self._name

    def getPlotFolder(self):
        return self._plotFolder

    def appendTableCreator(self, tc):
        self._tableCreators.append(tc)

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
            if fname is None:
                continue

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

        return PlotterFolder(self._name, self._possibleDirs, subFolders, self._plotFolder, self._fallbackNames, self._fallbackDqmSubFolders, self._tableCreators)

class PlotterTableItem:
    def __init__(self, possibleDirs, tableCreator):
        self._possibleDirs = possibleDirs
        self._tableCreator = tableCreator

    def create(self, openFiles, legendLabels, dqmSubFolder):
        if isinstance(dqmSubFolder, list):
            if len(dqmSubFolder) != len(openFiles):
                raise Exception("When dqmSubFolder is a list, len(dqmSubFolder) should be len(openFiles), now they are %d and %d" % (len(dqmSubFolder), len(openFiles)))
        else:
            dqmSubFolder = [dqmSubFolder]*len(openFiles)
        dqmSubFolder = [sf.subfolder if sf is not None else None for sf in dqmSubFolder]

        tbl = []
        for f, sf in zip(openFiles, dqmSubFolder):
            data = None
            tdir = _getDirectory(f, self._possibleDirs, sf)
            if tdir is not None:
                data = self._tableCreator.create(tdir)
            tbl.append(data)

        # Check if we have any content
        allNones = True
        colLen = 0
        for col in tbl:
            if col is not None:
                allNones = False
                colLen = len(col)
                break
        if allNones:
            return None

        # Replace all None columns with lists of column length
        for i in range(len(tbl)):
            if tbl[i] is None:
                tbl[i] = [None]*colLen

        return html.Table(columnHeaders=legendLabels, rowHeaders=self._tableCreator.headers(), table=tbl,
                          purpose=self._tableCreator.getPurpose(), page=self._tableCreator.getPage(), section=self._tableCreator.getSection(dqmSubFolder[0]))

class Plotter:
    """Contains PlotFolders, i.e. the information what plots to do, and creates a helper object to actually produce the plots."""
    def __init__(self):
        self._plots = []
        _setStyle()
        ROOT.TH1.AddDirectory(False)

    def append(self, *args, **kwargs):
        """Append a plot folder to the plotter.

        All arguments are forwarded to the constructor of PlotterItem.
        """
        self._plots.append(PlotterItem(*args, **kwargs))

    def appendTable(self, attachToFolder, *args, **kwargs):
        for plotterItem in self._plots:
            if plotterItem.getName() == attachToFolder:
                plotterItem.appendTableCreator(PlotterTableItem(*args, **kwargs))
                return
        raise Exception("Did not find plot folder '%s' when trying to attach a table creator to it" % attachToFolder)

    def clear(self):
        """Remove all plot folders and tables"""
        self._plots = []

    def getPlotFolderNames(self):
        return [item.getName() for item in self._plots]

    def getPlotFolders(self):
        return [item.getPlotFolder() for item in self._plots]

    def getPlotFolder(self, name):
        for item in self._plots:
            if item.getName() == name:
                return item.getPlotFolder()
        raise Exception("No PlotFolder named '%s'" % name)

    def readDirs(self, *files):
        """Returns PlotterInstance object, which knows how exactly to produce the plots for these files"""
        return PlotterInstance([plotterItem.readDirs(files) for plotterItem in self._plots])
