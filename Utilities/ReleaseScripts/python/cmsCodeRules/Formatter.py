#!/usr/bin/env python3

from __future__ import print_function
import os, sys, string

from xml.parsers import expat

# ================================================================================

class SimpleAsciiFormatter :

    def __init__(self, title="CMS SDT pages", style=None, outFile=sys.stdout) :
        self.format = ""
        self.title = title
        return
    
    def showLine(self) :
        print("\n================================================================================\n")
        return
    
    def write(self, arg="") :
        print(arg)
        return

    def writeB(self, arg="") :
        print(arg)
        return

    def writeBr(self) :
        print() 
        return

    def writeH1(self, arg="") :
        print(arg)
        return
    
    def writeH2(self, arg="") :
        print(arg)
        return

    def writeH3(self, arg="") :
        print(arg)
        return
    
    def startTable(self, colSizes, colLabels) :
        self.cols = colSizes
        self.format = ""
        for fmt in self.cols :
            self.format += "%"+str(fmt)+"s "
        print(self.format % tuple(colLabels))

        return

    def writeRow(self, args) :
        print(self.format % tuple(args))

        return

    def endTable(self) :
        self.format = None
        self.cols   = None
        
    
# ================================================================================

class SimpleHTMLFormatter :

    def __init__(self, title="CMS SDT pages ", style=None, outFile=sys.stdout, scriptCode='') :
        import time

        self.headersDone = False
        self.title=title
        self.style = style
        self.scriptCode = scriptCode
        self.outFile = outFile
        self.format = ""

        return

    def addScriptCode(self, scriptCode):
        self.scriptCode += scriptCode

    def __del__(self) :
        if self.headersDone :
            self.trailers()
        else: # in case nothing has been written yet, output an empty page ...
            self.outFile.write( "Content-Type: text/html" + '\n')     # HTML is following
            self.outFile.write( '\n')   # blank line, end of headers
            self.outFile.write( "<html> " + '\n')
            self.outFile.write( "</html> " + '\n')
        
        return

    def showLine(self) :
        self.headers()
        self.outFile.write( "<hr />" + '\n')
        return
    
    def write(self, arg="", bold=False) :
        self.headers()
        if bold:
            self.writeB(arg)
        else:
            self.outFile.write( arg + '\n')
        return

    def writeBr(self) :
        self.headers()
        self.outFile.write( "<br /> <br /> " + '\n')
        return
    
    def writeB(self, arg="") :
        self.headers()
        self.outFile.write( "<b> " + arg + " </b>" + '\n')
        return
    
    def writeH1(self, arg="") :
        self.headers()
        self.outFile.write( "<h1> " + arg + " </h1>" + '\n')
        return
    
    def writeH2(self, arg="") :
        self.headers()
        self.outFile.write( "<h2> " + arg + " </h2>" + '\n')
        return

    def writeH3(self, arg="") :
        self.headers()
        self.outFile.write( "<h3> " + arg + " </h3>" + '\n')
        return
    
    def writeAnchor(self, ref="") :
        self.headers()
        self.outFile.write( '<a name="' + ref + '">&nbsp;</a>')
        return
    
    def startTable(self, colSizes, colLabels, id=None, cls=None, tableAttr=None) :
        # we assume that html headers are done by now !!
        tableString = '<table '
        if tableAttr:
            tableString += tableAttr
        if id:
            tableString += ' id="'+id+'" '
        if cls:
            tableString += ' class="'+cls+'" '
        tableString += '>'
        self.outFile.write( tableString + '\n')

        self.outFile.write( " <thead>\n  <tr>"  + '\n')
        for col in colLabels :
            self.outFile.write( "   <th> <b>" + col + "</b> </th>" + '\n')
        self.outFile.write( "  </tr>\n</thead>" + '\n')
        self.outFile.write( "  <tbody>" + '\n')
        return

    def writeRow(self, args, bold=False, cls=None) :
        # we assume that headers are done by now !!

        if cls:
            self.outFile.write( ' <tr class="'+cls+'"> \n')
        else:
            self.outFile.write( " <tr>" + '\n')
        for arg in args:
            if string.strip(str(arg)) == "" : arg = "&nbsp;"
            if bold: self.outFile.write( '<td class=cellbold> ' )
            else:    self.outFile.write( "  <td> " )
            self.outFile.write( arg )
            
            self.outFile.write( " </td>" + '\n')
        self.outFile.write( " </tr> " + '\n')

        return

    def writeStyledRow(self, args, styles) :
        # we assume that headers are done by now !!
        self.outFile.write( " <tr>" + '\n')
        for arg, cellStyle in zip(args, styles):
            if string.strip(str(arg)) == "" : arg = "&nbsp;"
            cellStyle = cellStyle.strip()
            if cellStyle != '' : self.outFile.write( '<td class='+cellStyle+'> ' )
            else:    self.outFile.write( "  <td> " )
            self.outFile.write( arg )
            self.outFile.write( " </td>" + '\n')
        self.outFile.write( " </tr> " + '\n')

        return

    def endTable(self) :
        # we assume that headers are done by now !!
        self.outFile.write( "</tbody>" + '\n')
        self.outFile.write( "</table>" + '\n')
        
    # --------------------------------------------------------------------------------
    def headers(self) :
        # make sure to write the headers only once ...
        if not self.headersDone :
        
            self.outFile.write( "<html> " + '\n')

            self.outFile.write( "<head> " + '\n')

            if self.style:
                self.outFile.write( self.style + '\n')
            
            self.outFile.write( "<TITLE>" + self.title + "</TITLE>" + '\n')
            if self.scriptCode:
                self.outFile.write( self.scriptCode + '\n' )
            self.outFile.write( "</head> " + '\n')
            self.outFile.write( "<body>" + '\n')
        self.headersDone = True

        return

    # --------------------------------------------------------------------------------
    def trailers(self) :
        # write only if headers have been written ...
        if self.headersDone :
            self.outFile.write( "</body>" + '\n')
            self.outFile.write( "</html> " + '\n')
            pass
        return

    
