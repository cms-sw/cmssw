#!/usr/bin/env python

__author__="Aurelija"
__date__ ="$2010-08-12 10.50.40$"

from optparse import OptionParser
from shutil import rmtree
from os.path import join
import os.path

from Utilities.ReleaseScripts.cmsCodeRules.Formatter import SimpleHTMLFormatter
from Utilities.ReleaseScripts.cmsCodeRules.pickleFileParser import readPicFiles
from Utilities.ReleaseScripts.cmsCodeRules.config import rulesNames, ordering, Configuration, htmlPath

class BuildViewer(object):

    def __init__(self, formatter, pickleDir, logsDir, htmlDir):

        self.formatter = formatter

        self.configuration = Configuration

        self.logsDir = logsDir
        self.pickleDir = pickleDir
        self.htmlDir = htmlDir
        return

    # --------------------------------------------------------------------------------

    def showResults(self):

        rulesResults = readPicFiles(self.pickleDir, True)
        createLogFiles(rulesResults, self.logsDir)

        self.formatter.writeAnchor(ref='top')
        self.formatter.writeH2("CMSSW code rules violation")

        self.formatter.startTable([20,20,20,20,50], 
['Rule','Packages', 'Files','Sum of violations','Description'], cls = 
'descriptionTable')

        for ruleName in rulesNames:
            try:
                ruleRes = rulesResults[ruleName]
		totalViolations = 0
                totalFiles = 0
                for package, packageResult in ruleRes:
                    totalFiles += len(packageResult)
                    for file, lines in packageResult:
                        totalViolations += len(lines)
                self.formatter.writeRow([ruleName,str(len(ruleRes)), 
str(totalFiles), str(totalViolations), 
self.configuration[ruleName]['description']])
            except KeyError:
                self.formatter.writeRow([ruleName,'-', '-', '-',
self.configuration[ruleName]['description']])
        self.formatter.endTable()

        msg = """
<p>
Click on the package links to get list of files
</p>

"""
        self.formatter.write(msg)

        colFmt = [   50    ]
        colLab = ['Package']

        rules = ordering
        for rule in rules:
            colFmt.append(20)
            colLab.append('Rule %s' %rule)

        self.formatter.startTable(colFmt, colLab, cls='mainTable')

        packages = []
        table = []
        tableRow = len(colLab)*tuple('')
        ruleNr = 0
        for ruleName in rules:
            try:
                ruleResult = rulesResults[ruleName]
                for package, packageResult in ruleResult:
                    try:
                        index = packages.index(package)
                        tableRow = table[index] +(str(len(packageResult)),)
                        table[index] = tableRow
                    except ValueError:
                        packages.append(package) 
                        tableRow = ('<a href="logs/'+package+'/log.html"/>'+package,) + tuple('-' for i in range(ruleNr)) + (str(len(packageResult)),) #
                        table.append(tableRow)
                addDash(table, ruleNr)
            except KeyError:
                addDash(table, ruleNr)
            ruleNr += 1

        for row in table:
            self.formatter.writeRow(row)

        self.formatter.endTable()

        return

def addDash(table, ruleNr):
    for index, tableRow in enumerate(table):
        if len(tableRow)-1 != ruleNr + 1:
            table[index] = table[index] + ('-',)

def numberConverter(number):
    number = str(number)
    length = len(number)
    if  length < 3:
       number = (3-length)*str(0) + number
    return number

def createLogFiles(rulesResult, logsDir):
    logDir = join(logsDir,"logs")
    if os.path.exists(logDir):
        rmtree(logDir)
    for ruleName in rulesNames:
        try:
            ruleResult = rulesResult[ruleName]
            for package, packageResult in ruleResult:
                logsDir = join(logDir, package)
                if not os.path.exists(logsDir): os.makedirs(logsDir, 0755)
                file = open(join(logsDir, "log.html"), 'a')
                file.write('Rule %s'%ruleName)
                file.write("<br/>")
                for path, lineNumbers in packageResult:
                    for line in lineNumbers:
                        directory = join(package, path)
                        file.write('<a href="http://cmslxr.fnal.gov/lxr/source/%s#%s">%s:%s</a>\n'%(directory, numberConverter(line), directory, line))
                        file.write("<br/>")
                file.write('\n')
                file.close()
        except KeyError:
            pass

def run(pickleDir, logsDir, htmlDir):
    style = """
    <link rel="stylesheet" type="text/css" href="%s/intbld.css">

    <style type="text/css">
    .info { display: none; }
    </style>

    <script>
    function showHide(obj){
        myname = obj.name;
        $(".detail[name='"+myname+"']").toggle();  // .toggle('slow');
        $(".info[name='"+myname+"']").toggle();
    }
    </script>
    """%htmlPath

    fmtr = SimpleHTMLFormatter(title="CMSSW integration builds", style=style, outFile = open(join(htmlDir,"cmsCRPage.html"), "w"))

    bv = BuildViewer(fmtr, pickleDir, logsDir, htmlDir)
    bv.showResults()

def main():

    parser = OptionParser()
    parser.add_option("-l", "-L", dest="logDir", help = "creates log files to DIRECTORY", metavar = "DIRECTORY", default = os.getcwd())
    parser.add_option("-p", "-P", dest="pickleDir", help = "reads pickle files from DIRECTORY", metavar = "DIRECTORY", default = os.getcwd())
    parser.add_option("-c", "-C", dest="htmlDir", help = "creates cmsCRPage.html file to DIRECTORY", metavar = "DIRECTORY", default = os.getcwd())
    (options, args) = parser.parse_args()

    logsDir = options.logDir
    pickleDir = options.pickleDir
    htmlDir = options.htmlDir

    if not os.path.exists(logsDir):
        print "Error: wrong directory %s"%logsDir
        return

    if not os.path.exists(pickleDir):
        print "Error: wrong directory %s"%pickleDir
        return

    if not os.path.exists(htmlDir):
        print "Error: wrong directory %s"%htmlDir
        return

    run(pickleDir, logsDir, htmlDir)

    return

if __name__ == "__main__":
    main()
