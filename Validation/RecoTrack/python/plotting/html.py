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
    "RelValH125GGgluonfusion": "Higgs to gamma gamma",
    "RelValSingleElectronPt35": "Single Electron Pt 35",
    "RelValSingleElectronPt35Extended": "Single Electron Pt 35 (extended eta)",
    "RelValSingleElectronPt10": "Single Electron Pt 10",
    "RelValSingleMuPt10": "Single Muon Pt 10",
    "RelValSingleMuPt10Extended": "Single Muon Pt 10 (extended eta)",
    "RelValSingleMuPt100": "Single Muon Pt 100",
    "RelValTenMuE_0_200": "Ten muon Pt 0-200",
}

_sampleFileName = {
    "RelValMinBias": "minbias",
    "RelValTTbar": "ttbar",
    "RelValQCD_Pt_600_800": "qcd600",
    "RelValQCD_Pt_3000_3500": "qcd3000",
    "RelValQCD_FlatPt_15_3000": "qcdflat",
    "RelValZMM": "zmm",
    "RelValWjet_Pt_3000_3500": "wjet3000",
    "RelValH125GGgluonfusion": "hgg",
    "RelValSingleElectronPt35": "ele35",
    "RelValSingleElectronPt35Extended": "ele35ext",
    "RelValSingleElectronPt10": "ele10",
    "RelValSingleMuPt10": "mu10",
    "RelValSingleMuPt10Extended": "mu10ext",
    "RelValSingleMuPt100": "mu100",
    "RelValTenMuE_0_200": "tenmu200",
}

_allTPEfficName = "All tracks (all TPs)"
_fromPVName = "Tracks from PV"
_fromPVAllTPName = "Tracks from PV (all TPs)"
_conversionName = "Tracks for conversions"
_gsfName = "Electron GSF tracks"
_trackQualityNameOrder = collections.OrderedDict([
    ("seeding_seeds", "Seeds"),
    ("seeding_seedsa", "Seeds A"),
    ("seeding_seedsb", "Seeds B"),
    ("seeding_seedstripl", "Seeds triplets"),
    ("seeding_seedspair", "Seeds pairs"),
    ("building_", "Built tracks"),
    ("", "All tracks"),
    ("highPurity", "High purity tracks"),
    ("Pt", "Tracks pT &gt; 0.9 GeV"),
    ("highPurityPt", "High purity tracks pT &gt; 0.9 GeV"),
    ("ByOriginalAlgo", "All tracks by originalAlgo"),
    ("highPurityByOriginalAlgo", "High purity tracks by originalAlgo"),
    ("ByAlgoMask", "All tracks by algoMask"),
    ("highPurityByAlgoMask", "High purity tracks by algoMask"),
    ("btvLike", "BTV-like"),
    ("ak4PFJets", "AK4 PF jets"),
    ("allTPEffic_", _allTPEfficName),
    ("allTPEffic_highPurity", _allTPEfficName.replace("All", "High purity")),
    ("fromPV_", _fromPVName),
    ("fromPV_highPurity", "High purity "+_lowerFirst(_fromPVName)),
    ("fromPVAllTP_", _fromPVAllTPName),
    ("fromPVAllTP_highPurity", "High purity "+_lowerFirst(_fromPVAllTPName)),
    ("fromPVAllTP_Pt", _fromPVAllTPName.replace("Tracks", "Tracks pT &gt; 0.9 GeV")),
    ("fromPVAllTP_highPurityPt", "High purity "+_lowerFirst(_fromPVAllTPName).replace("tracks", "tracks pT &gt; 0.9 GeV")),
    ("fromPVAllTP2_", _fromPVAllTPName.replace("PV", "PV v2")),
    ("fromPVAllTP2_highPurity", "High purity "+_lowerFirst(_fromPVAllTPName).replace("PV", "PV v2")),
    ("fromPVAllTP2_Pt", _fromPVAllTPName.replace("Tracks", "Tracks pT &gt; 0.9 GeV").replace("PV", "PV v2")),
    ("fromPVAllTP2_highPurityPt", "High purity "+_lowerFirst(_fromPVAllTPName).replace("tracks", "tracks pT &gt; 0.9 GeV").replace("PV", "PV v2")),
    ("conversion_", _conversionName),
    ("gsf_", _gsfName),
])

_trackAlgoName = {
    "ootb": "Out of the box",
    "iter0" : "Iterative Step 0",
    "iter1" : "Iterative Step 1",
    "iter2" : "Iterative Step 2",
    "iter3" : "Iterative Step 3",
    "iter4" : "Iterative Step 4",
    "iter5" : "Iterative Step 5",
    "iter6" : "Iterative Step 6",
    "iter7" : "Iterative Step 7",
    "iter9" : "Iterative Step 9",
    "iter10": "Iterative Step 10",
}

_trackAlgoOrder = [
    'ootb',
    'initialStepPreSplitting',
    'initialStep',
    'highPtTripletStep',
    'detachedQuadStep',
    'detachedTripletStep',
    'lowPtQuadStep',
    'lowPtTripletStep',
    'pixelPairStep',
    'mixedTripletStep',
    'pixelLessStep',
    'tobTecStep',
    'jetCoreRegionalStep',
    'muonSeededStepInOut',
    'muonSeededStepOutIn',
    'duplicateMerge',
    'convStep',
    'conversionStep',
    'ckfInOutFromConversions',
    'ckfOutInFromConversions',
    'electronGsf',
    'iter0',
    'iter1',
    'iter2',
    'iter3',
    'iter4',
    'iter5',
    'iter6',
    'iter7',
    'iter9',
    'iter10',
]

_pageNameMap = {
    "summary": "Summary",
    "vertex": "Vertex",
    "v0": "V0",
    "miniaod": "MiniAOD",
    "timing": "Timing",
}

_sectionNameMapOrder = collections.OrderedDict([
    # These are for the summary page
    ("seeding_seeds", "Seeds"),
    ("building", "Built tracks"),
    ("", "All tracks"),
    ("highPurity", "High purity tracks"),
    ("btvLike", "BTV-like"),
    ("ak4PFJets", "AK4 PF jets"),
    ("allTPEffic", _allTPEfficName),
    ("allTPEffic_highPurity", _allTPEfficName.replace("All", "High purity")),
    ("fromPV", _fromPVName),
    ("fromPV_highPurity", "High purity "+_lowerFirst(_fromPVName)),
    ("fromPVAllTP", _fromPVAllTPName),
    ("fromPVAllTP_highPurity", "High purity "+_lowerFirst(_fromPVAllTPName)),
    ("conversion", _conversionName),
    ("gsf", _gsfName),
    # These are for vertices
    ("genvertex", "Gen vertices"),
    ("pixelVertices", "Pixel vertices"),
    ("selectedPixelVertices", "Selected pixel vertices"),
    ("firstStepPrimaryVerticesPreSplitting", "firstStepPrimaryVerticesPreSplitting"),
    ("firstStepPrimaryVertices", "firstStepPrimaryVertices"),
    ("offlinePrimaryVertices", "All vertices (offlinePrimaryVertices)"),
    ("selectedOfflinePrimaryVertices", "Selected vertices (selectedOfflinePrimaryVertices)"),
    ("offlinePrimaryVerticesWithBS", "All vertices with BS constraint"),
    ("selectedOfflinePrimaryVerticesWithBS", "Selected vertices with BS constraint"),
    # These are for V0
    ("k0", "K0"),
    ("lambda", "Lambda"),
])
_allTPEfficLegend = "All tracks, efficiency denominator contains all TrackingParticles"
_fromPVLegend = "Tracks from reco PV vs. TrackingParticles from gen PV (fake rate includes pileup tracks)"
_fromPVAllTPLegend = "Tracks from reco PV, fake rate numerator contains all TrackingParticles (separates fake tracks from pileup tracks)"
_fromPVAllTPPtLegend = "Tracks (pT &gt 0.9 GeV) from reco PV, fake rate numerator contains all TrackingParticles (separates fake tracks from pileup tracks)"
_fromPVAllTP2Legend = "Tracks from reco PV (another method), fake rate numerator contains all TrackingParticles (separates fake tracks from pileup tracks)"
_fromPVAllTPPt2Legend = "Tracks (pT &gt 0.9 GeV) from reco PV (another method), fake rate numerator contains all TrackingParticles (separates fake tracks from pileup tracks)"

def _sectionNameLegend():
    return {
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
        "fromPVAllTP_Pt": _fromPVAllTPPtLegend,
        "fromPVAllTP_highPurityPt": "High purity "+_lowerFirst(_fromPVAllTPPtLegend),
        "fromPVAllTP2_": _fromPVAllTP2Legend,
        "fromPVAllTP2_highPurity": "High purity "+_lowerFirst(_fromPVAllTP2Legend),
        "fromPVAllTP2_Pt": _fromPVAllTPPt2Legend,
        "fromPVAllTP2_highPurityPt": "High purity "+_lowerFirst(_fromPVAllTPPt2Legend),
    }

class Table:
    # table [column][row]
    def __init__(self, columnHeaders, rowHeaders, table, purpose, page, section):
        if len(columnHeaders) != len(table):
            raise Exception("Got %d columnHeaders for table with %d columns for page %s, section %s" % (len(columnHeaders), len(table), page, section))
        lenRow = len(table[0])
        for icol, column in enumerate(table):
            if len(column) != lenRow:
                raise Exception("Got non-square table, first column has %d rows, column %d has %d rows" % (lenRow, icol, len(column)))
        if len(rowHeaders) != lenRow:
            raise Exception("Got %d rowHeaders for table with %d rows" % (len(rowHeaders), lenRow))

        self._columnHeaders = columnHeaders
        self._rowHeaders = rowHeaders
        self._table = table

        self._purpose = purpose
        self._page = page
        self._section = section

    def getPurpose(self):
        return self._purpose

    def getPage(self):
        return self._page

    def getSection(self):
        return self._section

    def ncolumns(self):
        return len(self._table)

    def nrows(self):
        return len(self._table[0])

    def columnHeaders(self):
        return self._columnHeaders

    def rowHeaders(self):
        return self._rowHeaders

    def tableAsColumnRow(self):
        return self._table

    def tableAsRowColumn(self):
        return map(list, zip(*self._table))

class PlotPurpose:
    class TrackingIteration: pass
    class TrackingSummary: pass
    class Vertexing: pass
    class MiniAOD: pass
    class Timing: pass

class Page(object):
    def __init__(self, title, sampleName):
        self._content = [
            '<html>',
            ' <head>',
            '  <title>%s</title>' % title,
            ' </head>',
            ' <body>',
            '  '+sampleName,
            '  <br/>',
            '  <br/>',
        ]

        self._plotSets = {}
        self._tables = {}

    def addPlotSet(self, section, plotSet):
        if section in self._plotSets:
            self._plotSets[section].extend(plotSet)
        else:
            self._plotSets[section] = plotSet

    def addTable(self, section, table):
        self._tables[section] = table

    def isEmpty(self):
        for plotSet in self._plotSets.itervalues():
            if len(plotSet) > 0:
                return False

        if len(self._tables) > 0:
            return False

        return True

    def write(self, fileName):
        self._legends = []
        self._sectionLegendIndex = {}
        self._columnHeaders = []
        self._columnHeadersIndex = {}
        self._formatPlotSets()
        self._formatTables()
        self._formatLegend()

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

    def _appendLegend(self, section):
        leg = ""
        legends = _sectionNameLegend()
        if section in legends:
            if section in self._sectionLegendIndex:
                leg = self._sectionLegendIndex[section]
            else:
                legnum = len(self._legends)+1
                leg = "<sup>%d</sup>" % legnum
                leg2 = "<sup>%d)</sup>" % legnum
                self._legends.append("%s %s" % (leg2, legends[section]))
                self._sectionLegendIndex[section] = leg
        return leg

    def _formatPlotSets(self):
        self._content.extend([
            '  <table>'
            '   <tr>',
        ])

        fileTable = []

        sections = self._orderSets(self._plotSets.keys())
        for isec, section in enumerate(sections):
            leg = self._appendLegend(section)

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

    def _appendColumnHeader(self, header):
        leg = ""
        if header in self._columnHeadersIndex:
            leg = self._columnHeadersIndex[header]
        else:
            leg = str(chr(ord('A')+len(self._columnHeaders)))
            self._columnHeaders.append("%s: %s" % (leg, header))
            self._columnHeadersIndex[header] = leg
        return leg

    def _formatTables(self):
        def _allNone(row):
            for item in row:
                if item is not None:
                    return False
            return True

        sections = self._orderSets(self._tables.keys())
        for isec, section in enumerate(sections):
            leg = self._appendLegend(section)

            table = self._tables[section]
            self._content.extend([
                '  <br/>',
                '  %s%s' % (self._mapSectionName(section), leg),
                '  <table border="1">'
            ])

            # table is stored in column-row, need to transpose
            data = table.tableAsRowColumn()

            self._content.extend([
                '   <tr>'
                '   <td></td>'
            ])
            heads = table.columnHeaders()
            if max(map(lambda h: len(h), heads)) > 20:
                heads = [self._appendColumnHeader(h) for h in heads]
            for head in heads:
                self._content.append('    <td>%s</td>' % head)
            self._content.append('   </tr>')

            for irow, row in enumerate(data):
                # Skip row if all values are non-existent
                if _allNone(row):
                    continue

                self._content.extend([
                    '   <tr>'
                    '    <td>%s</td>' % table.rowHeaders()[irow]
                ])
                # align the number columns to right
                for icol, item in enumerate(row):
                    formatted = str(item) if item is not None else ""
                    self._content.append('    <td align="right">%s</td>' % formatted)
                self._content.append('   </tr>')

            self._content.append('  </table>')

            for shortenedColumnHeader in self._columnHeaders:
                self._content.append('  %s<br/>' % shortenedColumnHeader)
            self._columnHeaders = []
            self._columnHeadersIndex = {}

    def _formatLegend(self):
        if len(self._legends) > 0:
            self._content.extend([
                '  <br/>'
                '  Details:</br>',
            ])
            for leg in self._legends:
                self._content.append('  %s<br/>' % leg)


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
    def __init__(self, title, sampleName, sample, fastVsFull):
        self._title = title
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

    def _getPage(self, key, pageClass):
        if key not in self._pages:
            page = pageClass(self._title, self._sampleName)
            self._pages[key] = page
        else:
            page = self._pages[key]
        return page

    def addPlotSet(self, plotterFolder, dqmSubFolder, plotFiles):
        pageKey = plotterFolder.getPage()
        if pageKey is None:
            if dqmSubFolder is not None:
                pageKey = dqmSubFolder.translated
            else:
                pageKey = plotterFolder.getName()

        page = self._getPage(pageKey, Page)
        sectionName = plotterFolder.getSection()
        if sectionName is None:
            if plotterFolder.getPage() is not None and dqmSubFolder is not None:
                sectionName = dqmSubFolder.translated
            else:
                sectionName = ""

        page.addPlotSet(sectionName, plotFiles)

    def addTable(self, table):
        if table is None:
            return

        page = self._getPage(table.getPage(), Page)
        page.addTable(table.getSection(), table)

    def write(self, baseDir):
        #print "TrackingPageSet.write"
        ret = []

        keys = self._orderPages(self._pages.keys())
        for key in keys:
            page = self._pages[key]
            if page.isEmpty():
                continue

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
        #
        # it is bit of a hack to access trackingPlots.TrackingPlotFolder this way,
        # but it was simple and it works
        if algo != "ootb" and not plotterFolder._plotFolder.isAlgoIterative(algo):
            pageName = "ootb"
            sectionName = algo

        folderName = plotterFolder.getName()
        if folderName != "":
            sectionName = folderName+"_"+sectionName

        page = self._getPage(pageName, TrackingIterPage)
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
    def __init__(self, sample, fastVsFull, title):
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
        if hasattr(sample, "customPileupLabel"):
            pileup = sample.customPileupLabel()

        scenario = ""
        if hasattr(sample, "hasScenario") and sample.hasScenario():
            scenario = " (\"%s\")" % sample.scenario()
        self._sampleName += "%s sample%s %s" % (_sampleName.get(sample.name(), sample.name()), scenario, pileup)

        params = [title, self._sampleName, sample, fastVsFull]
        self._summaryPage = PageSet(*params)
        self._iterationPages = TrackingPageSet(*params)
        self._vertexPage = PageSet(*params)
        self._miniaodPage = PageSet(*params)
        self._timingPage = PageSet(*params)
        self._otherPages = PageSet(*params)

        self._purposePageMap = {
            PlotPurpose.TrackingIteration: self._iterationPages,
            PlotPurpose.TrackingSummary: self._summaryPage,
            PlotPurpose.Vertexing: self._vertexPage,
            PlotPurpose.MiniAOD: self._miniaodPage,
            PlotPurpose.Timing: self._timingPage,
        }

    def addPlots(self, plotterFolder, dqmSubFolder, plotFiles):
        page = self._purposePageMap.get(plotterFolder.getPurpose(), self._otherPages)
        page.addPlotSet(plotterFolder, dqmSubFolder, plotFiles)

    def addTable(self, table):
        if table is None:
            return

        page = self._purposePageMap.get(table.getPurpose(), self._otherPages)
        page.addTable(table)
        params = []

    def write(self, baseDir):
        ret = [
            "  "+self._sampleName,
            "  <br/>",
            "  <ul>",
            ]

        for pages in [self._summaryPage, self._iterationPages, self._vertexPage, self._miniaodPage, self._timingPage, self._otherPages]:
            labelFiles = pages.write(baseDir)
            for label, fname in labelFiles:
                ret.append('   <li><a href="%s">%s</a></li>' % (fname, label))

        ret.extend([
            '  </ul>',
            '  <br/>',
        ])

        return ret

class HtmlReport:
    def __init__(self, validationName, newBaseDir):
        self._title = "Tracking validation "+validationName
        self._newBaseDir = newBaseDir

        self._index = [
            '<html>',
            ' <head>',
            '  <title>%s</title>' % self._title,
            ' </head>',
            ' <body>',
        ]

        self._sections = collections.OrderedDict()

    def addNote(self, note):
        self._index.append('  <p>%s</p>'%note)

    def beginSample(self, sample, fastVsFull=False):
        key = (sample.digest(), fastVsFull)
        if key in self._sections:
            self._currentSection = self._sections[key]
        else:
            self._currentSection = IndexSection(sample, fastVsFull, self._title)
            self._sections[key] = self._currentSection

    def addPlots(self, *args, **kwargs):
        self._currentSection.addPlots(*args, **kwargs)

    def addTable(self, *args, **kwargs):
        self._currentSection.addTable(*args, **kwargs)

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

    def addTable(self, *args, **kwargs):
        pass
