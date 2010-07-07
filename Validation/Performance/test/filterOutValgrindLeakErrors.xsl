<?xml version="1.0" encoding="ISO-8859-1"?>
<xsl:stylesheet version="1.0" xmlns:xsl="http://www.w3.org/1999/XSL/Transform">

 <xsl:template match="//error">
  <xsl:choose>
    <xsl:when test="kind='Leak_PossiblyLost'"></xsl:when>
    <xsl:when test="kind='Leak_DefinitelyLost'"></xsl:when>
    <xsl:otherwise>
     <xsl:copy-of select="current()"/>
    </xsl:otherwise>
  </xsl:choose>
 </xsl:template>

<xsl:template match="*">
<xsl:copy-of select="current()"/>
</xsl:template>

<xsl:template match="/valgrindoutput">
<xsl:copy select="current()">
<xsl:apply-templates />
</xsl:copy>
</xsl:template>

</xsl:stylesheet>
