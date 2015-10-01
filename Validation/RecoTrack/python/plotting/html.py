import os
import collections

def _lowerFirst(s):
    return s[0].lower()+s[1:]

_sampleName = {
    "RelValMinBias": "Min Bias",
    "RelValTTbar": "TTbar",
    "RelValQCD_Pt_600_800": "QCD Pt 600 to 800",
    "RelValQCD_Pt_3000_3500": "QCD Pt 3000 to 3500",
    "RelValQCD_FlatPt_15_3000": "QCD Flat Pt 15 to 3000",
    "RelValZMM": "ZMuMu",
    "RelValWjet_Pt_3000_3500": "Wjet Pt 3000 to 3500",
    "RelValSingleElectronPt35": "Single Electron Pt 35",
    "RelValSingleElectronPt10": "Single Electron Pt 10",
    "RelValSingleMuPt10": "Single Muon Pt 10",
    "RelValSingleMuPt100": "Single Muon Pt 100",
}

_sampleFileName = {
    "RelValMinBias": "minbias",
    "RelValTTbar": "ttbar",
    "RelValQCD_Pt600_800": "qcd600",
    "RelValQCD_Pt3000_3500": "qcd3000",
    "RelValQCD_FlatPt_15_3000": "qcdflat",
    "RelValZMM": "zmm",
    "RelValWjet_Pt_3000_3500": "wjet3000",
    "RelValSingleElectronPt35": "ele35",
    "RelValSingleElectronPt10": "ele10",
    "RelValSingleMuPt10": "mu10",
    "RelValSingleMuPt100": "mu100",
}

_allTPEfficName = "All tracks (all TPs)"
_fromPVName = "Tracks from PV"
_fromPVAllTPName = "Tracks from PV (all TPs)"
_trackQualityNameOrder = collections.OrderedDict([
    ("", "All tracks"),
    ("highPurity", "High purity tracks"),
    ("btvLike", "BTV-like"),
    ("ak4PFJets", "AK4 PF jets"),
    ("allTPEffic_", _allTPEfficName),
    ("allTPEffic_highPurity", _allTPEfficName.replace("All", "High purity")),
    ("fromPV_", _fromPVName),
    ("fromPV_highPurity", "High purity "+_lowerFirst(_fromPVName)),
    ("fromPVAllTP_", _fromPVAllTPName),
    ("fromPVAllTP_highPurity", "High purity "+_lowerFirst(_fromPVAllTPName)),
])

_trackAlgoName = {
    "ootb": "Out of the box"
}

_trackAlgoOrder = [
    'ootb',
    'initialStep',
    'lowPtTripletStep',
    'pixelPairStep',
    'detachedTripletStep',
    'mixedTripletStep',
    'pixelLessStep',
    'tobTecStep',
    'jetCoreRegionalStep',
    'muonSeededStepInOut',
    'muonSeededStepOutIn',
]

_pageNameMap = {
    "summary": "Summary",
    "vertex": "Vertex",
    "miniaod": "MiniAOD"
}

_sectionNameMapOrder = collections.OrderedDict([
    # These are for the summary page
    ("", "All tracks"),
    ("highPurity", "High purity tracks"),
    ("allTPEffic", _allTPEfficName),
    ("allTPEffic_highPurity", _allTPEfficName.replace("All", "High purity")),
    ("fromPV", _fromPVName),
    ("fromPV_highPurity", "High purity "+_lowerFirst(_fromPVName)),
    ("fromPVAllTP", _fromPVAllTPName),
    ("fromPVAllTP_highPurity", "High purity "+_lowerFirst(_fromPVAllTPName)),
    # These are for vertices
    ("offlinePrimaryVertices", "All vertices (offlinePrimaryVertices)"),
    ("selectedOfflinePrimaryVertices", "Selected vertices (selectedOfflinePrimaryVertices)"),
])
_allTPEfficLegend = "All tracks, efficiency denominator contains all TrackingParticles"
_fromPVLegend = "Tracks from reco PV vs. TrackingParticles from gen PV (fake rate includes pileup tracks)"
_fromPVAllTPLegend = "Tracks from reco PV, fake rate numerator contains all TrackingParticles (separates fake tracks from pileup tracks)"
_sectionNameLegend = {
    "btvLike": "BTV-like selected tracks",
    "ak4PFJets": "Tracks from AK4 PF jets (jet corrected pT &gt; 10 GeV)",
    "allTPEffic": _allTPEfficLegend,
    "allTPEffic_": _allTPEfficLegend,
    "allTPEffic_highPurity": _allTPEfficLegend.replace("All", "High purity"),
    "fromPV": _fromPVLegend,
    "fromPV_": _fromPVLegend,
    "fromPV_highPurity": "High purity "+_lowerFirst(_fromPVLegend),
    "fromPVAllTP": _fromPVAllTPLegend,
    "fromPVAllTP_": _fromPVAllTPLegend,
    "fromPVAllTP_highPurity": "High purity "+_lowerFirst(_fromPVAllTPLegend),
}

class PlotPurpose:
    class TrackingIteration: pass
    class TrackingSummary: pass
    class Vertexing: pass
    class MiniAOD: pass

class Page(object):
    def __init__(self, title, base, sampleName):
        self._content = [
            '<html>',
            ' <head>',
            '  <title>%s</title>' % title,
        ]
        if base is not None:
            self._content.append('  <base href="%s"/>' % base)
        self._content.extend([
            ' </head>',
            ' <body>',
            '  '+sampleName,
            '  <br/>',
            '  <br/>',
        ])

        self._plotSets = {}

    def addPlotSet(self, section, plotSet):
        self._plotSets[section] = plotSet

    def write(self, fileName):
        sections = self._orderSets(self._plotSets.keys())

        self._content.extend([
            '  <table>'
            '   <tr>',
        ])

        fileTable = []
        legends = []

        for isec, section in enumerate(sections):
            leg = ""
            if section in _sectionNameLegend:
                legnum = len(legends)+1
                leg = "<sup>%d</sup>" % legnum
                leg2 = "<sup>%d</sup>" % legnum
                legends.append("%s %s" % (leg2, _sectionNameLegend[section]))

            self._content.extend([
                '   <td>%s%s</td>' % (self._mapSectionName(section), leg),
            ])
            files = [(os.path.basename(f), f) for f in self._plotSets[section]]
            for row in fileTable:
                found = False
                for i, (bsf, f) in enumerate(files):
                    if bsf == row[0]:
                        row.append(f)
                        found = True
                        del files[i]
                        break
                if not found:
                    row.append(None)
            for bsf, f in files:
                fileTable.append( [bsf] + [None]*isec + [f] )

        self._content.extend([
            '   </tr>',
        ])

        for row in fileTable:
            self._content.append('   <tr>')
            bs = row[0]
            for elem in row[1:]:
                if elem is not None:
                    self._content.append('    <td><a href="%s">%s</a></td>' % (elem, bs))
                else:
                    self._content.append('    <td></td>')
            self._content.append('   </tr>')


        self._content.extend([
            '  </table>',
        ])
        if len(legends) > 0:
            self._content.extend([
                '  <br/>'
                '  Details:</br>',
            ])
            for leg in legends:
                self._content.append('  %s<br/>' % leg)

        self._content.extend([
            ' </body>',
            '</html>',
        ])

        #print "Writing HTML report page", fileName
        f = open(fileName, "w")
        for line in self._content:
            f.write(line)
            f.write("\n")
        f.close()

    def _mapSectionName(self, section):
        return _sectionNameMapOrder.get(section, section)

    def _orderSets(self, keys):
        ret = []
        for section in _sectionNameMapOrder.keys():
            if section in keys:
                ret.append(section)
                keys.remove(section)
        ret.extend(keys)
        return ret

class PageSet(object):
    def __init__(self, title, base, sampleName, sample, fastVsFull):
        self._title = title
        self._base = base
        self._sampleName = sampleName
        self._pages = collections.OrderedDict()

        self._prefix=""
        if hasattr(sample, "hasPileup"):
            self._prefix = "nopu"
            if sample.hasPileup():
                self._prefix = "pu"+sample.pileupType()
            self._prefix += "_"

        if sample.fastsim():
            self._prefix += "fast_"
            if fastVsFull:
                self._prefix += "full_"

        self._prefix += _sampleFileName.get(sample.label(), sample.label())+"_"

    def addPlotSet(self, plotterFolder, dqmSubFolder, plotFiles):
        pageKey = plotterFolder.getPage()
        if pageKey is None:
            if dqmSubFolder is not None:
                pageKey = dqmSubFolder.translated
            else:
                pageKey = plotterFolder.getName()

        if pageKey not in self._pages:
            page = Page(self._title, self._base, self._sampleName)
            self._pages[pageKey] = page
        else:
            page = self._pages[pageKey]
        sectionName = plotterFolder.getSection()
        if sectionName is None:
            if plotterFolder.getPage() is not None and dqmSubFolder is not None:
                sectionName = dqmSubFolder.translated
            else:
                sectionName = ""

        page.addPlotSet(sectionName, plotFiles)

    def write(self, baseDir):
        #print "TrackingPageSet.write"
        ret = []

        keys = self._orderPages(self._pages.keys())
        for key in keys:
            page = self._pages[key]
            fileName = "%s%s.html" % (self._prefix, key)
            page.write(os.path.join(baseDir, fileName))
            ret.append( (self._mapPagesName(key), fileName) )
        return ret

    def _mapPagesName(self, name):
        return _pageNameMap.get(name, name)

    def _orderPages(self, keys):
        return keys



class TrackingIterPage(Page):
    def __init__(self, *args, **kwargs):
        super(TrackingIterPage, self).__init__(*args, **kwargs)

    def _mapSectionName(self, quality):
        return _trackQualityNameOrder.get(quality, quality)

    def _orderSets(self, qualities):
        ret = []
        for qual in _trackQualityNameOrder.keys():
            if qual in qualities:
                ret.append(qual)
                qualities.remove(qual)
        ret.extend(qualities)
        return ret

class TrackingPageSet(PageSet):
    def __init__(self, *args, **kwargs):
        super(TrackingPageSet, self).__init__(*args, **kwargs)

    def addPlotSet(self, plotterFolder, dqmSubFolder, plotFiles):
        (algo, quality) = dqmSubFolder.translated

        pageName = algo
        sectionName = quality

        # put all non-iterative stuff under OOTB
        if "ootb" not in algo and "Step" not in algo:
            pageName = "ootb"
            sectionName = algo

        folderName = plotterFolder.getName()
        if folderName != "":
            sectionName = folderName+"_"+sectionName

        if pageName not in self._pages:
            page = TrackingIterPage(self._title, self._base, self._sampleName)
            self._pages[pageName] = page
        else:
            page = self._pages[pageName]
        page.addPlotSet(sectionName, plotFiles)

    def _mapPagesName(self, algo): # algo = pageName
        return _trackAlgoName.get(algo, algo)

    def _orderPages(self, algos):
        ret = []
        for algo in _trackAlgoOrder:
            if algo in algos:
                ret.append(algo)
                algos.remove(algo)
        ret.extend(algos)
        return ret



class IndexSection:
    def __init__(self, sample, fastVsFull, title, base):
        self._sample = sample

        self._sampleName = ""
        if sample.fastsim():
            self._sampleName += "FastSim "
            if fastVsFull:
                self._sampleName += "vs FullSim "

        pileup = ""
        if hasattr(sample, "hasPileup"):
            pileup = "with no pileup"
            if sample.hasPileup():
                pileup = "with %s pileup" % sample.pileupType()
        self._sampleName += "%s sample %s" % (_sampleName.get(sample.name(), sample.name()), pileup)

        params = [title, base, self._sampleName, sample, fastVsFull]
        self._summaryPage = PageSet(*params)
        self._iterationPages = TrackingPageSet(*params)
        self._vertexPage = PageSet(*params)
        self._miniaodPage = PageSet(*params)
        self._otherPages = PageSet(*params)

    def addPlots(self, plotterFolder, dqmSubFolder, plotFiles):
        params = [plotterFolder, dqmSubFolder, plotFiles]

        purpose = plotterFolder.getPurpose()
        if purpose is PlotPurpose.TrackingIteration:
            self._iterationPages.addPlotSet(*params)
        elif purpose is PlotPurpose.TrackingSummary:
            self._summaryPage.addPlotSet(*params)
        elif purpose is PlotPurpose.Vertexing:
            self._vertexPage.addPlotSet(*params)
        elif purpose is PlotPurpose.MiniAOD:
            self._miniaodPage.addPlotSet(*params)
        else:
            self._otherPages.addPlotSet(*params)

    def write(self, baseDir):
        ret = [
            "  "+self._sampleName,
            "  <br/>",
            "  <ul>",
            ]

        for pages in [self._summaryPage, self._iterationPages, self._vertexPage, self._miniaodPage, self._otherPages]:
            labelFiles = pages.write(baseDir)
            for label, fname in labelFiles:
                ret.append('   <li><a href="%s">%s</a></li>' % (fname, label))

        ret.extend([
            '  </ul>',
            '  <br/>',
        ])

        return ret

class HtmlReport:
    def __init__(self, validationName, newBaseDir, baseUrl=None):
        self._title = "Tracking validation "+validationName
        self._newBaseDir = newBaseDir
        self._base = baseUrl

        self._index = [
            '<html>',
            ' <head>',
            '  <title>%s</title>' % self._title,
        ]
        if self._base is not None:
            self._index.append('  <base href="%s"/>' % self._base)
        self._index.extend([
            ' </head>',
            ' <body>',
        ])

        self._sections = collections.OrderedDict()

    def beginSample(self, sample, fastVsFull=False):
        key = (sample.digest(), fastVsFull)
        if key in self._sections:
            self._currentSection = self._sections[key]
        else:
            self._currentSection = IndexSection(sample, fastVsFull, self._title, self._base)
            self._sections[key] = self._currentSection

    def addPlots(self, *args, **kwargs):
        self._currentSection.addPlots(*args, **kwargs)

    def write(self):
        # Reorder sections such that Fast vs. Full becomes just after the corresponding Fast
        keys = self._sections.iterkeys()
        newkeys = []
        for key in keys:
            if not key[1]:
                newkeys.append(key)
                continue
            # is fast vs full
            ind_fast = newkeys.index( (key[0], False) )
            newkeys.insert(ind_fast+1, key)

        for key in newkeys:
            section = self._sections[key]
            self._index.extend(section.write(self._newBaseDir))

        self._index.extend([
            " </body>",
            "</html>",
        ])

        f = open(os.path.join(self._newBaseDir, "index.html"), "w")
        for line in self._index:
            f.write(line)
            f.write("\n")
        f.close()

class HtmlReportDummy:
    def __init__(self):
        pass

    def beginSample(self, *args, **kwargs):
        pass

    def addPlots(self, *args, **kwargs):
        pass
