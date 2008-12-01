/*
 * $Id$
 *
 * Copyright (C) 1991,1995 UCAR/Unidata
 *
 * Permission to use, copy, modify, and distribute this software and its
 * documentation for any purpose without fee is hereby granted, provided
 * that the above copyright notice appear in all copies, that both that
 * copyright notice and this permission notice appear in supporting
 * documentation, and that the name of UCAR/Unidata not be used in
 * advertising or publicity pertaining to distribution of the software
 * without specific, written prior permission.  UCAR makes no
 * representations about the suitability of this software for any purpose.
 * It is provided "as is" without express or implied warranty.  It is
 * provided with no support and without obligation on the part of UCAR or
 * Unidata, to assist in its use, correction, modification, or enhancement.
 */

#ifndef	UTPRIVATE_H_INCLUDED
#define	UTPRIVATE_H_INCLUDED

/* #include <udposix.h> */


/*
 * The following #define-s redefine common global symbols so that the
 * udunits(3) syntactical parser and lexical scanner won't interfere
 * with other yacc(1)-derived parsers and scanners.  (This has only been
 * verified for bison(1)-derived parsers and flex(1)-derived scanners).
 */
#define yylval		utlval
#undef	unput
#define	unput		utunput
#define YY_INPUT(buf,result,max_size) \
  { \
      int c = utinput(); \
      result = (c == EOF) ? YY_NULL : (buf[0] = c, 1); \
  }

extern int	UtLineno;	/* input-file line index */
extern int	UnitNotFound;	/* parser didn't find unit */
extern utUnit	*FinalUnit;	/* fully-parsed specification */

/* UD_EXTERN_FUNC(double utencdate,  (int year, int month, int day)); */
/* UD_EXTERN_FUNC(double utencclock, (int hour, int minute, double sec)); */
/* UD_EXTERN_FUNC(int    utinput,    (void)); */
/* UD_EXTERN_FUNC(void   utunput,    (int c)); */
/* UD_EXTERN_FUNC(void   uterror,    (char *msg)); */


#endif	/* UTPRIVATE_H_INCLUDED was not defined above */
