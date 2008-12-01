/*
 * $Id$
 * $__Header$
 *
 * Header-file for the private part of the Unidata units(3) library.
 */

#ifndef	UT_SCAN_H_INCLUDED
#define	UT_SCAN_H_INCLUDED

/*
 *	Macro for declaring functions regardless of the availability of 
 *	function prototypes.  NB: will need double parens in actual use (e.g.
 *	"int func PROTO((int a, char *cp))").
 */
#ifndef	PROTO
#   if defined(__STDC__) || defined(__GNUC__) \
	|| defined(__cplusplus) || defined(c_plusplus)
#	define	PROTO(a)	a
#   else
#	define	PROTO(a)	()
#   endif
#endif

/*
 *	Lexical analyzer:
 */
extern int      utLex		PROTO((void));
extern void     utLexReset	PROTO((void));

#endif	/* UT_SCAN_H_INCLUDED not defined */
