/*
 * $Id$
 *
 * Support functions for the units(3) library.
 */

/*LINTLIBRARY*/

#ifndef	_XOPEN_SOURCE
#   define _XOPEN_SOURCE
#endif
#ifndef	_ANSI_C_SOURCE
#   define _ANSI_C_SOURCE
#endif

/* #include <udposix.h> */
#include <stddef.h>		/* for ptrdiff_t */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <ctype.h>
#include <math.h>
#include <assert.h>
#include <search.h>
#include <math.h>		/* for modf(), floor(), log10(), ceil() */
#include <limits.h>		/* for *PATH_MAX */
#include <float.h>		/* for DBL_DIG */
/* #include "cfortran.h"		/\* for FORTRAN support *\/ */
#include "udunits.h"
#include "utprivate.h"

/* /\* */
/*  * cfortran.h support for unit arguments: */
/*  *\/ */
/* /\* C-returning-to-FORTRAN: *\/ */
/* #define PUNIT_cfFZ(UN,LN)		utUnit* fcallsc(UN,LN)(void */
/* /\* The following might be wrong and require modification for the next */
/*  * version of cfortran.h *\/ */
/* #define PUNIT_cfINT(N,A,B,X,Y,Z)	SIMPLE_cfINT(N,A,B,X,Y,Z) */
/* #define PUNIT_cfUU(PUNIT,A0)		utUnit* A0 */
/* #define PUNIT_cfL			A0 = */
/* #define PUNIT_cfI			return A0; */
/* #define PUNIT_cfK */
/* #define PUNIT_cfSTR(N,T,A,B,C,D,E)	SIMPLE_cfSTR(N,T,A,B,C,D,E) */

/* /\* FORTRAN-calling-C: *\/ */
/* #define PPUNIT_cfV(  T,A,B,F)       SIMPLE_cfV(T,A,B,F) */
/* #define PPUNIT_cfSEP(T,  B)         SIMPLE_cfSEP(T,B) */
/* #define PPUNIT_cfINT(N,A,B,X,Y,Z)   SIMPLE_cfINT(N,A,B,X,Y,Z) */
/* #define PPUNIT_cfSTR(N,T,A,B,C,D,E) SIMPLE_cfSTR(N,T,A,B,C,D,E) */
/* #define PPUNIT_cfCC( T,A,B)         SIMPLE_cfCC(T,A,B) */
/* #define PPUNIT_cfAA( T,A,B)         PPUNIT_cfB(T,A) /\* Argument B not used. *\/ */
/* #define PPUNIT_cfU(  T,A)           PPUNIT_cfN(T,A) */
/* #define PPUNIT_cfN(  T,A)	    utUnit** A */
/* #define PPUNIT_cfT(M,I,A,B,D)	    *A */

/* /\* Output, Fortran character buffer: *\/ */
/* #define CBUF_cfINT(N,A,B,X,Y,Z)		STRING_cfINT(N,A,B,X,Y,Z) */
/* #define CBUF_cfSEP(T,  B)		STRING_cfSEP(T,B) */
/* #define CBUF_cfN(  T,A)			STRING_cfN(T,A) */
/* #define CBUF_cfSTR(N,T,A,B,C,D,E)	STRING_cfSTR(N,T,A,B,C,D,E) */
/* #if defined(vmsFortran) */
/* #   define CBUF_cfT(M,I,A,B,D)		A->dsc$a_pointer, A->dsc$w_length */
/* #elif defined(CRAYFortran) */
/* #   define CBUF_cfT(M,I,A,B,D)		_fcdtocp(A), _fcdlen(A) */
/* #else */
/* #   define CBUF_cfT(M,I,A,B,D)		A, D */
/* #endif */

#ifndef PATH_MAX
#   define PATH_MAX	_POSIX_PATH_MAX
#endif

#undef	DUPSTR
#define	DUPSTR(s)	strcpy((char*)malloc(strlen(s)+1), s)
#undef	ABS
#define	ABS(a)		((a) < 0 ? -(a) : (a))
#undef	MIN
#define	MIN(a,b)	((a) < (b) ? (a) : (b))
#undef	MAX
#define	MAX(a,b)	((a) > (b) ? (a) : (b))

typedef struct {
    char	*name;
    int		nchr;
    int		HasPlural;
    utUnit	unit;
} UnitEntry;

typedef struct {
    char	*name;		/* prefix string (e.g. "milli") */
    UtFactor    factor;		/* corresponding multiplying factor */
    short       nchar;		/* size of prefix string excluding EOS */
} PrefixEntry;

/*
 *  Prefix table in the order required by the prefix-entry comparison function.
 *  The names are Based on ANSI/IEEE Std 260-1978 (a.k.a. ANSI Y10.19-1969) -- 
 *  reaffirmed 1985.  The names must be unique.
 *
 *  NB: The short-prefix symbol corresponding to the prefix "micro" has been 
 *  changed here from the standard one (the Greek letter "mu") to the symbol 
 *  "u".
 */
#define PE(name, factor)    {name, factor, sizeof(name)-1}
static PrefixEntry	PrefixTable[]   = {
    PE("E",     1e18),
    PE("G",     1e9),
    PE("M",     1e6),
    PE("P",     1e15),
    PE("T",     1e12),
    PE("Y",	1e24),
    PE("Z",	1e21),
    PE("a",     1e-18),
    PE("atto",  1e-18),
    PE("c",     1e-2),
    PE("centi", 1e-2),
    PE("d",     1e-1),
    PE("da",    1e1),
    PE("deca",  1e1),	/* Spelling according to "ISO 2955: Information
			 * processing -- Representation of SI and other units
			 * in systems with limited character sets". */
    PE("deci",  1e-1),
    PE("deka",  1e1),	/* Spelling according to "ASTM Designation: E 380 - 85:
			 * Standard for METRIC PRACTICE" and "ANSI/IEEE Std
			 * 260-1978 (Reaffirmed 1985): IEEE Standard Letter 
			 * Symbols for Units of Measurement". */
    PE("exa",   1e18),
    PE("f",     1e-15),
    PE("femto", 1e-15),
    PE("giga",  1e9),
    PE("h",     1e2),
    PE("hecto", 1e2),
    PE("k",     1e3),
    PE("kilo",  1e3),
    PE("m",     1e-3),
    PE("mega",  1e6),
    PE("micro", 1e-6),
    PE("milli", 1e-3),
    PE("n",     1e-9),
    PE("nano",  1e-9),
    PE("p",     1e-12),
    PE("peta",  1e15),
    PE("pico",  1e-12),
    PE("tera",  1e12),
    PE("u",     1e-6),
    PE("y",     1e-24),
    PE("yocto", 1e-24),
    PE("yotta", 1e24),
    PE("z",     1e-21),
    PE("zepto", 1e-21),
    PE("zetta", 1e21),
    NULL
};

static void	*root		= NULL;
static int	initialized	= 0;	/* module initialized = no */
static int	NumberBaseUnits	= 0;	/* number of base units */
static int	HaveStdTimeUnit	= 0;	/* standard time unit set? */
static char	*input_buf;		/* scanner input buffer */
static char	*input_ptr;		/* scanner input position */
static char	*unput_ptr;		/* position in unput() buffer */
static char	unput_buf[512];		/* scanner unput() buffer */
static char	linebuf[512];		/* input units-specification buffer */
static char	BaseName[UT_MAXNUM_BASE_QUANTITIES][UT_NAMELEN];
static FILE	*UtFile;		/* input units-file */
static char	UnitsFilePath[PATH_MAX];/* input units-file path */
static utUnit	StdTimeUnit;		/* standard time unit for determining
					 * if a unit structure refers to a unit
					 * of time
					 */


/*
 * The following two functions convert between Julian day number and
 * Gregorian/Julian dates (Julian dates are used prior to October 15,
 * 1582; Gregorian dates are used after that).  Julian day number 0 is
 * midday, January 1, 4713 BCE.  The Gregorian calendar was adopted
 * midday, October 15, 1582.
 *
 * Author: Robert Iles, March 1994
 *
 * C Porter: Steve Emmerson, October 1995
 *
 * Original: http://www.nag.co.uk:70/nagware/Examples/calendar.f90
 *
 * There is no warranty on this code.
 */


/*
 * Convert a Julian day number to a Gregorian/Julian date.
 */
void
julday_to_gregdate(julday, year, month, day)
    unsigned long	julday;		/* Julian day number to convert */
    int			*year;		/* Gregorian year (out) */
    int			*month;		/* Gregorian month (1-12) (out) */
    int			*day;		/* Gregorian day (1-31) (out) */
{
#if INT_MAX <= 0X7FFF
    long	ja, jb, jd;
#else
    int		ja, jb, jd;
#endif
    int		jc;
    int		je, iday, imonth, iyear;
    double	xc;

    if (julday < 2299161)
	ja = julday;
    else
    {
	int	ia = ((julday - 1867216) - 0.25) / 36524.25;

	ja = julday + 1 + ia - (int)(0.25 * ia);
    }

    jb = ja + 1524;
    xc = ((jb - 2439870) - 122.1) / 365.25;
    jc = 6680.0 + xc;
    jd = 365 * jc + (int)(0.25 * jc);
    je = (int)((jb - jd) / 30.6001);

    iday = (int)(jb - jd - (int)(30.6001 * je));

    imonth = je - 1;
    if (imonth > 12)
	imonth -= 12;

    iyear = jc - 4715;
    if (imonth > 2)
	iyear -= 1;
    if (iyear <= 0)
	iyear -= 1;

    *year = iyear;
    *month = imonth;
    *day = iday;
}


/*
 * Convert a Gregorian/Julian date to a Julian day number.
 *
 * The Gregorian calendar was adopted midday, October 15, 1582.
 */
unsigned long
gregdate_to_julday(year, month, day)
    int		year;	/* Gregorian year */
    int		month;	/* Gregorian month (1-12) */
    int		day;	/* Gregorian day (1-31) */
{
#if INT_MAX <= 0X7FFF
    long		igreg = 15 + 31 * (10 + (12 * 1582));
    long		iy;	/* signed, origin 0 year */
    long		ja;	/* Julian century */
    long		jm;	/* Julian month */
    long		jy;	/* Julian year */
#else
    int			igreg = 15 + 31 * (10 + (12 * 1582));
    int			iy;	/* signed, origin 0 year */
    int			ja;	/* Julian century */
    int			jm;	/* Julian month */
    int			jy;	/* Julian year */
#endif
    unsigned long	julday;	/* returned Julian day number */

    /*
     * Because there is no 0 BC or 0 AD, assume the user wants the start of 
     * the common era if they specify year 0.
     */
    if (year == 0)
	year = 1;

    iy = year;
    if (year < 0)
	iy++;
    if (month > 2)
    {
	jy = iy;
	jm = month + 1;
    }
    else
    {
	jy = iy - 1;
	jm = month + 13;
    }

    /*
     *  Note: SLIGHTLY STRANGE CONSTRUCTIONS REQUIRED TO AVOID PROBLEMS WITH
     *        OPTIMISATION OR GENERAL ERRORS UNDER VMS!
     */
    julday = day + (int)(30.6001 * jm);
    if (jy >= 0)
    {
	julday += 365 * jy;
	julday += 0.25 * jy;
    }
    else
    {
	double		xi = 365.25 * jy;

	if ((int)xi != xi)
	    xi -= 1;
	julday += (int)xi;
    }
    julday += 1720995;

    if (day + (31* (month + (12 * iy))) >= igreg)
    {
	ja = jy/100;
	julday -= ja;
	julday += 2;
	julday += ja/4;
    }

    return julday;
}


/*
 * Encode a date as a double-precision value.
 */
    double
utencdate(year, month, day)
    int		year;
    int		month;
    int		day;
{
    return ((long)gregdate_to_julday(year, month, day) - 
	    (long)gregdate_to_julday(2001, 1, 1)) * 86400.0;
}


/*
 * Encode a time as a double-precision value.
 */
    double
utencclock(hours, minutes, seconds)
    int		hours;
    int		minutes;
    double	seconds;
{
    return (hours*60 + minutes)*60 + seconds;
}


/*
 * Decompose a value into a set of values accounting for uncertainty.
 */
    static void
decomp(value, uncer, nbasis, basis, count)
    double	value;
    double	uncer;		/* >= 0 */
    int		nbasis;
    double	*basis;		/* all values > 0 */
    double	*count;
{
    int		i;

    for (i = 0; i < nbasis; i++)
    {
	double	r = fmod(value, basis[i]);	/* remainder */

	/* Adjust remainder to minimum magnitude. */
	if (ABS(2*r) > basis[i])
	    r += r > 0
		    ? -basis[i]
		    :  basis[i];

	if (ABS(r) <= uncer)
	{
	    /* The value equals a basis multiple within the uncertainty. */
	    double	half = value < 0 ? -basis[i]/2 : basis[i]/2;
	    modf((value+half)/basis[i], count+i);
	    break;
	}

	value = basis[i] * modf(value/basis[i], count+i);
    }

    for (i++; i < nbasis; i++)
	count[i] = 0;
}


/*
 * Decode a time from a double-precision value.
 */
    static void
dectime(value, year, month, day, hour, minute, second)
    double	value;
    int		*year;
    int		*month;
    int		*day;
    int		*hour;
    int		*minute;
    float	*second;
{
    long	days;
    long	hours;
    long	minutes;
    double	seconds;
    double	uncer;		/* uncertainty of input value */
    typedef union
    {
	double	    vec[7];
	struct
	{
	    double	days;
	    double	hours12;
	    double	hours;
	    double	minutes10;
	    double	minutes;
	    double	seconds10;
	    double	seconds;
	}	    ind;
    } Basis;
    Basis	counts;
    static Basis	basis;

    basis.ind.days = 86400;
    basis.ind.hours12 = 43200;
    basis.ind.hours = 3600;
    basis.ind.minutes10 = 600;
    basis.ind.minutes = 60;
    basis.ind.seconds10 = 10;
    basis.ind.seconds = 1;

    uncer = ldexp(value < 0 ? -value : value, -DBL_MANT_DIG);

    days = floor(value/86400.0);
    value -= days * 86400.0;		/* make positive excess */

    decomp(value, uncer, sizeof(basis.vec)/sizeof(basis.vec[0]),
	   basis.vec, counts.vec);

    days += counts.ind.days;
    hours = (int)counts.ind.hours12 * 12 + (int)counts.ind.hours;
    minutes = (int)counts.ind.minutes10 * 10 + (int)counts.ind.minutes;
    seconds = (int)counts.ind.seconds10 * 10 + counts.ind.seconds;

    *second = seconds;
    *minute = minutes;
    *hour = hours;
    julday_to_gregdate(gregdate_to_julday(2001, 1, 1) + days, year, month, day);
}


/*
 * Indicate whether or not the given unit structure refers to a unit of time.
 *
 * This function returns:
 *	1	unit structure is a unit of time;
 *	0	unit structure is not a unit of time.
 */
    int
utIsTime(up)
    const utUnit	*up;
{
    int		status;

    if (!initialized || !HaveStdTimeUnit) {
	status	= 0;
    } else {
	int		iquan;

	for (iquan = 0; iquan < UT_MAXNUM_BASE_QUANTITIES; ++iquan)
	    if (up->power[iquan] != StdTimeUnit.power[iquan])
		break;

	status	= iquan == UT_MAXNUM_BASE_QUANTITIES;
    }

    return status;
}


/* /\* */
/*  * FORTRAN interface to the above function. */
/*  *\/ */
/* FCALLSCFUN1(INT,utIsTime,UTTIME,uttime, */
/*     PPUNIT) */


/*
 * Indicate whether or not the given unit structure has an origin.
 *
 * This function returns:
 *	1	unit structure has an origin;
 *	0	unit structure doesn't have an origin.
 */
    int
utHasOrigin(up)
    const utUnit	*up;
{
    return initialized && up->hasorigin;
}


/* /\* */
/*  * FORTRAN interface to the above function. */
/*  *\/ */
/* FCALLSCFUN1(INT,utHasOrigin,UTORIGIN,utorigin, */
/*     PPUNIT) */


/*
 *  Clear a unit structure by setting it to the dimensionless identity
 *  unit structure.
 *
 *  This function returns a pointer to the unit structure.
 */
    utUnit*
utClear(unit)
    utUnit		*unit;
{
    register int	iquan;

    unit->hasorigin	= 0;
    unit->origin	= 0.0;
    unit->factor	= 1.0;

    for (iquan = 0; iquan < UT_MAXNUM_BASE_QUANTITIES; ++iquan)
	unit->power[iquan] = 0;

    return unit;
}


/* /\* */
/*  * FORTRAN interface to the above function. */
/*  *\/ */
/* FCALLSCSUB1(utClear,UTCLR,utclr, */
/*     PPUNIT) */


/*
 *  Copy a unit structure.
 *
 *  This function returns a pointer to the destination unit structure.
 */
    utUnit*
utCopy(source, dest)
    const utUnit	*source;
    utUnit		*dest;
{
    assert(source != NULL);
    assert(dest != NULL);

    *dest	= *source;

    return dest;
}


/* /\* */
/*  * FORTRAN interface to the above function. */
/*  *\/ */
/* FCALLSCSUB2(utCopy,UTCPY,utcpy, */
/*     PPUNIT,PPUNIT) */


/*
 *  Set the exponent of the given base quantity to unity.
 *
 *  This function returns:
 *	 NULL	error;
 *	!NULL	success (points to the input unit structure).
 */
    utUnit*
utSetPower(unit, position)
    utUnit	*unit;
    int		position;
{
    utUnit	*result	= NULL;

    if (position < 0 || position >= UT_MAXNUM_BASE_QUANTITIES) {
	(void) fprintf(stderr, 
	  "udunits(3): %d is an invalid quantity index.  Valid range is 0-%d\n",
	      position, UT_MAXNUM_BASE_QUANTITIES);
    } else {
	unit->power[position]	= 1;
	result	= unit;
    }

    return result;
}


/* 
 *  Shift the origin of a unit structure.
 *
 *  This function returns a pointer to the destination unit structure.
 */
    utUnit*
utShift(source, amount, result)
    utUnit	*source;
    double	amount;
    utUnit	*result;
{
    assert(source != NULL);
    assert(result != NULL);

    (void) utCopy(source, result);
    result->origin	= source->origin + amount*result->factor;
    result->hasorigin	= 1;

    return result;
}


/* /\* */
/*  * FORTRAN interface to the above function. */
/*  *\/ */
/* FCALLSCSUB3(utShift,UTORIG,utorig, */
/*     PPUNIT,DOUBLE,PPUNIT) */


/* 
 *  Scale a "unit" structure.
 *
 *  This function returns a pointer to the destination unit structure.
 */
    utUnit*
utScale(source, factor, result)
    utUnit     *source;
    double      factor;
    utUnit	*result;
{
    assert(source != NULL);
    assert(result != NULL);

    (void) utCopy(source, result);

    result->factor	*= factor;
    result->origin	*= factor;

    return result;
}


/* /\* */
/*  * FORTRAN interface to the above function. */
/*  *\/ */
/* FCALLSCSUB3(utScale,UTSCAL,utscal, */
/*     PPUNIT,DOUBLE,PPUNIT) */


/* 
 *  Multiply two unit-structures.
 *
 *  This function returns
 *	NULL			failure;
 *	pointer to `result'	success.
 */
    utUnit*
utMultiply(term1, term2, result)
    utUnit     *term1, *term2, *result;
{
    utUnit	*res	= NULL;

    if (term2->hasorigin && term1->hasorigin) {
	(void) fprintf(stderr, 
		   "udunits(3): Can't multiply units with origins\n");
    } else {
	int         iquan;

	result->factor		= term1->factor * term2->factor;
	result->origin		= term1->hasorigin
				    ? term1->origin * term2->factor
				    : term2->origin * term1->factor;
	result->hasorigin	= term1->hasorigin || term2->hasorigin;

	for (iquan = 0; iquan < UT_MAXNUM_BASE_QUANTITIES; ++iquan)
	    result->power[iquan]    = term1->power[iquan] + term2->power[iquan];

	res	= result;
    }

    return res;
}


/* /\* */
/*  * FORTRAN interface to the above function. */
/*  *\/ */
/* FCALLSCSUB3(utMultiply,UTMULT,utmult, */
/*     PPUNIT,PPUNIT,PPUNIT) */


/*
 *  Form the reciprocal of an internal unit specification.
 *
 *  This function returns
 *	NULL			failure;
 *	pointer to `result'	success.
 */
    utUnit*
utInvert(source, result)
    utUnit     *source, *result;
{
    utUnit	*res	= NULL;

    if (source->hasorigin) {
	(void) fprintf(stderr, 
		   "udunits(3): Can't invert a unit with an origin\n");
    } else {
	int         iquan;

	result->factor		= 1./source->factor;
	result->origin		= 0.0;
	result->hasorigin	= 0;

	for (iquan = 0; iquan < UT_MAXNUM_BASE_QUANTITIES; ++iquan)
	    result->power[iquan]  = -source->power[iquan];

	res	= result;
    }

    return res;
}


/* /\* */
/*  * FORTRAN interface to the above function. */
/*  *\/ */
/* FCALLSCSUB2(utInvert,UTINV,utinv, */
/*     PPUNIT,PPUNIT) */


/* 
 *  Divide two unit-structures.
 *
 *  This function returns
 *	NULL			failure;
 *	pointer to `result'	success.
 */
    utUnit*
utDivide(numer, denom, result)
    utUnit     *numer, *denom, *result;
{
    utUnit	*res	= NULL;

    if (denom->hasorigin && numer->hasorigin) {
	(void) fprintf(stderr, 
		   "udunits(3): Can't divide units with origins\n");
    } else {
	int         iquan;

	result->factor		= numer->factor / denom->factor;
	result->origin		= numer->origin;
	result->hasorigin	= numer->hasorigin;

	for (iquan = 0; iquan < UT_MAXNUM_BASE_QUANTITIES; ++iquan)
	    result->power[iquan]    = numer->power[iquan] - denom->power[iquan];

	res	= result;
    }

    return res;
}


/* /\* */
/*  * FORTRAN interface to the above function. */
/*  *\/ */
/* FCALLSCSUB3(utDivide,UTDIV,utdiv, */
/*     PPUNIT,PPUNIT,PPUNIT) */


/* 
 *  Raise a unit-structure to a given power.
 *
 *  This function returns
 *	NULL			failure;
 *	pointer to `result'	success.
 */
    utUnit*
utRaise(source, power, result)
    utUnit	*source;
    int         power;
    utUnit	*result;
{
    utUnit	*res	= NULL;

    if (source->hasorigin) {
	(void) fprintf(stderr, 
	   "udunits(3): Can't exponentiate a unit with an origin\n");
    } else {
	int         iquan;

	result->factor		= pow(source->factor, (double)power);
	result->origin		= 0.0;
	result->hasorigin	= 0;

	for (iquan = 0; iquan < UT_MAXNUM_BASE_QUANTITIES; ++iquan)
	    result->power[iquan]    = source->power[iquan] * power;

	res	= result;
    }

    return res;
}


/* /\* */
/*  * FORTRAN interface to the above function. */
/*  *\/ */
/* FCALLSCSUB3(utRaise,UTEXP,utexp, */
/*     PPUNIT,INT,PPUNIT) */


/*
 *  Open a units-file.
 *
 *  This function returns:
 *	 0		success
 *	UT_ENOFILE	couldn't open file
 */
    static int
OpenUnits(filename)
    char	*filename;
{
    int		status;			/* return status */
    extern int UtLineno;
    if ((UtFile	= fopen(filename, "r")) == NULL) {
	(void) fprintf(stderr, 
		       "udunits(3): Couldn't open units database \"%s\": ", 
		       filename);
	perror("");
	status	= UT_ENOFILE;
    } else {
	UtLineno	= 0;
	linebuf[sizeof(linebuf)-1]	= 0;
	status	= 0;
    }

    return status;
}


/*
 * Close a units file.
 *
 * This function returns void.
 */
    static void
CloseUnits()
{
    if (UtFile != NULL) {
	(void)fclose(UtFile);
	UtFile	= NULL;
    }
}


/*
 *  Decode a unit specification.
 *
 *  This function returns:
 *	 0		success
 *	UT_ESYNTAX	syntax error
 *	UT_EUNKNOWN	unknown specification
 */
    static int
DecodeUnit(spec, unit)
    char	*spec;
    utUnit	*unit;
{
    char	specbuf[512];

    /* Remove leading whitespace */
    while (isspace(*spec))
	++spec;

    (void) strncpy(specbuf, spec, sizeof(specbuf));
    specbuf[sizeof(specbuf)-1]	= 0;

    /* Remove trailing whitespace */
    {
	char	*cp;
	for (cp = specbuf + strlen(specbuf);
	    cp > specbuf && isspace(cp[-1]);
	    --cp)
	    ;	/* EMPTY */
	*cp = 0;
    }

    input_ptr	= input_buf	= specbuf;
    unput_ptr	= unput_buf;

    (void)utClear(FinalUnit = unit);

    UnitNotFound	= 0;
    utrestart((FILE*)0);
    return utparse() == 0 ? 0 : UnitNotFound ? UT_EUNKNOWN : UT_ESYNTAX;
}


/*
 *  Scan (decode) the next entry in the units file.
 *  If the specification is for a base unit, then the returned unit-structure
 *  is cleared except that the value of the next available exponent slot is 
 *  set to one.  A base unit is the unit for a fundamental (base) physical
 *  quantity (e.g. the unit `meter' for the base physical quantity `length').
 *
 *  This function returns:
 *	0		success
 *	UT_EOF		end-of-file encountered
 *	UT_EIO		I/O error
 *	UT_ESYNTAX	syntax error
 *	UT_EUNKNOWN	unknown specification
 *	UT_EALLOC	allocation failure
 */
    static int
ScanUnit(name, sizename, unit, HasPlural)
    char	*name;			/* name buffer */
    size_t	sizename;		/* size of name buffer */
    utUnit	*unit;			/* unit specification */
    int		*HasPlural;		/* specification has a plural form? */
{
    int		status	= 0;		/* return status = success */

    for (;;) {
	char		*cp;
	static char	WhiteSpace[]	= " \t";

	++UtLineno;
	if (fgets(linebuf, (int)sizeof(linebuf), UtFile) == NULL) {
	    if (feof(UtFile)) {
		status	= UT_EOF;
	    } else {
		(void) perror("fgets()");
		status	= UT_EIO;
	    }
	    break;
	}
	
	if (linebuf[strlen(linebuf)-1] != '\n') {
	    (void) fprintf(stderr, 
		       "udunits(3): Input-line longer than %lu-byte buffer\n",
			   (unsigned long)sizeof(linebuf));
	    status	= UT_ESYNTAX;
	    break;
	}

	/* Truncate at comment character */
	if ((cp = strchr(linebuf, '#')) != NULL)
	    *cp	= 0;

	/* Trim trailing whitespace */
	for (cp = linebuf + strlen(linebuf); cp > linebuf; --cp)
	     if (!isspace(cp[-1]))
		break;
	*cp	= 0;

	/* Find first "black" character */
	cp	= linebuf + strspn(linebuf, WhiteSpace);

	if (*cp != 0) {
	    int		n	= strcspn(cp, WhiteSpace);
	    char	buf[512];

	    assert(sizeof(buf) > strlen(linebuf));
	    assert((size_t)sizename > strlen(linebuf));

	    (void)strncpy(name, cp, n); name[n] = 0;

	    cp	+= n;
	    cp	+= strspn(cp, WhiteSpace);
	    n	= strcspn(cp, WhiteSpace);
	    (void)strncpy(buf, cp, n); buf[n]	= 0;

	    if (strcmp(buf, "P") == 0) {
		*HasPlural	= 1;
	    } else if (strcmp(buf, "S") == 0) {
		*HasPlural	= 0;
	    } else {
		(void) fprintf(stderr,
		       "udunits(3): Invalid single/plural indicator \"%s\"\n", 
		       buf);
		status	= UT_ESYNTAX;
		break;
	    }

	    cp	+= n;
	    cp	+= strspn(cp, WhiteSpace);
	    (void)strcpy(buf, cp);

	    if (buf[0] == 0) {
		(void)utClear(unit);
		if (utSetPower(unit, NumberBaseUnits) == NULL) {
		    (void) fprintf(stderr, 
				   "udunits(3): Couldn't set base unit #%d\n", 
				   NumberBaseUnits);
		    status	= UT_EALLOC;
		} else {
		    (void)strncpy(BaseName[NumberBaseUnits], name, 
				  UT_NAMELEN-1);
		    ++NumberBaseUnits;
		}
	    } else {
		if ((status = DecodeUnit(buf, unit)) != 0) {
		    (void) fprintf(stderr, 
		   "udunits(3): Couldn't decode \"%s\" definition \"%s\"\n", 
			   name, buf);
		}
	    }
	    break;
	}					/* if not a layout line */
    }						/* input-line loop */

    if (status != 0 && status != UT_EOF)
	(void) fprintf(stderr, "udunits(3): Error occurred at line %d\n", 
		       UtLineno);

    return status;
}


/*
 * Read and decode the entries in the units-file, adding them to the units-
 * table.
 *
 * This function returns:
 *	0		success
 *	UT_ENOFILE	no units-file
 *	UT_ESYNTAX	syntax error
 *	UT_EUNKNOWN	unknown specification
 *	UT_EALLOC	allocation failure
 *	UT_EIO		I/O error
 */
    static int
ReadUnits(path)
    char	*path;
{
    int		status;				/* return status */

    if ((status = OpenUnits(path)) == 0) {
	for (;;) {
	    int		HasPlural;
	    char	name[512];
	    utUnit	unit;

	    if ((status = ScanUnit(name, sizeof(name), &unit, 
				   &HasPlural)) == UT_EOF) {
		status	= 0;
		break;
	    } else if (status == 0) {
		(status = utAdd(name, HasPlural, &unit));

		if (status) {
		    if (UT_DUP == status) {
			(void) fprintf(stderr,
			       "udunits(3): Replaced unit \"%s\" at line %d\n",
			       name, UtLineno);
		    }
		    else {
			(void) fprintf(stderr,
			       "udunits(3): Couldn't add \"%s\" to "
			       "units-table\n", name);
			break;
		    }
		}
	    } else {
		(void) fprintf(stderr,
			       "udunits(3): Couldn't read units-file \"%s\"\n",
			       path);
		break;
	    }
	}

	CloseUnits();
    }						/* units-file opened */

    return status;
}


/*
 *  Initialize the units(3) package.
 *
 *  This function returns:
 *	0		success
 *	UT_ENOFILE	no units-file
 *	UT_ESYNTAX	syntax error in units-file
 *	UT_EUNKNOWN	unknown specification in units-file
 *	UT_EIO		units-file I/O error
 *	UT_EALLOC	allocation failure
 */
    int
utInit(path)
    const char	*path;
{
    int		status;
    char	pathbuf[PATH_MAX+1];
    char        UT_DEFAULT_PATH[PATH_MAX+1];
    if (path == NULL || path[0] == 0) {
        path    = getenv("UDUNITS_PATH");
        if (path == NULL || path[0] == 0)
	    path = strcpy(pathbuf, UT_DEFAULT_PATH);
    }

    if (initialized && strcmp(path, UnitsFilePath) == 0) {
	(void) fprintf(stderr, 
		"udunits(3): Already initialized from file \"%s\"\n",
		path);
	status	= 0;
    } else {
	utTerm();

	status	= ReadUnits(path);

	if (status == 0) {
	    (void) strncpy(UnitsFilePath, path, sizeof(UnitsFilePath)-1);
	    initialized	= 1;
	    if (utScan("second", &StdTimeUnit) == 0) {
		int	iquan;
		int	seen_one = 0;

		for (iquan = 0; iquan < UT_MAXNUM_BASE_QUANTITIES; ++iquan) {
		    if  (StdTimeUnit.power[iquan] != 0) {
			if (seen_one)
			    break;
			seen_one = 1;
		    }
		}
		HaveStdTimeUnit	= seen_one && 
				  iquan == UT_MAXNUM_BASE_QUANTITIES;
	    }
	} else {
	    utTerm();
	}
    }

    /* (void) fprintf(stderr, "utInit(): returning %d\n", status); */

    return status;
}


/* /\* */
/*  * FORTRAN interface to the above function. */
/*  *\/ */
/* FCALLSCFUN1(INT,utInit,UTOPEN,utopen, */
/*     STRING) */


/*
 *  Indicate if the units(3) package has been initialized.
 *
 *  This function returns:
 *	0		if the units(3) package has not been initialized.
 *     !0		if the units(3) package has been initialized.
 */
    int
utIsInit()
{
    return initialized;
}


/* /\* */
/*  * FORTRAN interface to the above function. */
/*  *\/ */
/* FCALLSCFUN0(LOGICAL,utIsInit,UTISOPEN,utisopen) */


/*
 *  Decode a unit specification.
 *
 *  This function returns:
 *	0		success
 *	UT_ENOINIT	the package hasn't been initialized
 *	UT_EUNKNOWN	unknown specification
 *	UT_ESYNTAX	syntax error
 *	UT_EINVALID	NULL unit argument
 */
    int
utScan(spec, up)
    const char *spec;
    utUnit     *up;
{
    int         status;                    /* return status = success */

    if (spec == NULL) {
	status	= UT_EUNKNOWN;
    } else {
	if (up == NULL) {
	    status	= UT_EINVALID;
	} else {
	    if (!initialized) {
		(void) fprintf(stderr, 
			"udunits(3): Package hasn't been initialized\n");
		status	= UT_ENOINIT;
	    } else {
		utLexReset();
		status	= DecodeUnit(spec, up);
	    }
	}
    }

    return status;
}


/* /\* */
/*  * FORTRAN interface to the above function. */
/*  *\/ */
/* FCALLSCFUN2(INT,utScan,UTDEC,utdec, */
/*     STRING,PPUNIT) */


/*
 *  Encode a unit-structure into a formatted unit-secification.
 *
 *  This function returns:
 *      0               success
 *      UT_ENOINIT      the package hasn't been initialized
 *      UT_EINVALID     the unit-structure is invalid
 *
 *  On error, the string argument is set to the nil-pointer.
 */
    int
utPrint(unit, s)
    register const utUnit	*unit;
    char			**s;
{
    int				status;

    if (!initialized) {
        (void) fprintf(stderr,
                       "udunits(3): Package hasn't been initialized\n");
        *s      = NULL;
        status  = UT_ENOINIT;

    } else {
        if (unit->factor == 0.0) {
            *s          = NULL;
            status      = UT_EINVALID;

        } else {
            register int         iquan;
            register char       *buf    = linebuf;

            *buf        = 0;

            /*  Print the scale factor if it's non-unity. */
            if (unit->factor != 1.0) {
                (void)sprintf(buf, "%.*g ", DBL_DIG, unit->factor);
                buf     += strlen(buf);
            }

            /*  Append the dimensions in terms of base quanitities. */
            for (iquan = 0; iquan < UT_MAXNUM_BASE_QUANTITIES; ++iquan) {
                if (unit->power[iquan] != 0) {
                    if (unit->power[iquan] == 1) {
                        (void)sprintf(buf, "%s ", BaseName[iquan]);
                    } else {
                        (void)sprintf(buf, "%s%d ", BaseName[iquan],
                                      unit->power[iquan]);
                    }
                    buf += strlen(buf);
                }
            }

            /*  Append the origin-shift if it exists. */
            if (unit->hasorigin) {
                if (utIsTime(unit)) {
                    int         year, month, day;
                    int         hours;
                    int         minutes;
                    int         ndigsec;        /* precision of seconds in
                                                 * number of digits */
                    float       seconds;

                    dectime(unit->origin, &year, &month, &day, &hours,
                            &minutes, &seconds);
                    (void)sprintf(buf-1, "s since %d-%02d-%02d %02d:%02d ",
                                  year, month, day,
                                  hours, minutes);
                    buf += strlen(buf);
                    ndigsec     = DBL_DIG -
                        (int) ceil(log10(ABS(unit->origin /
                                                 utencclock(0, 0, 1.0))));
                    ndigsec     = MIN(ndigsec, DBL_DIG);
                    if (ndigsec > 0) {
			int	precision = MAX(0, ndigsec-2);

#if 0
			double	integral_secs;
			double	fractional_secs = modf(seconds, &integral_secs);

			if (precision <= 0)
			    (void) sprintf(buf-1, ":%02d", (int)integral_secs);
			else
			    (void) sprintf(buf-1, ":%02d.%.*f", 
					   (int)integral_secs, precision, 
					   fractional_secs);
#endif

                        (void)sprintf(buf-1, ":%0*.*f ", precision+3,
				      precision, seconds);
                    }
                    (void) strcat(buf, "UTC ");
                } else {
                    (void)sprintf(buf, "@ %.*g ", DBL_DIG, unit->origin);
                }
                buf     += strlen(buf);
            }

            /*  Remove trailing space. */
            if (buf > linebuf)
                buf[-1] = 0;

            *s          = linebuf;
            status      = 0;
        }                                       /* unit-structure is valid */
    }                                           /* package is initialized */

    return status;
}


/* /\* */
/*  * FORTRAN helper function for the above. */
/*  *\/ */
/* static int */
/* utPrint_help(unit, buf, size) */
/*     utUnit	*unit; */
/*     char	*buf; */
/*     size_t	size; */
/* { */
/*     char	*s; */
/*     int		status	= utPrint(unit, &s); */

/*     if (status == 0) */
/*     { */
/* 	size_t	len = strlen(s); */

/* 	if (len <= size) */
/* 	{ */
/* 	    (void) memcpy(buf, s, len); */
/* 	    (void) memset(buf+len, ' ', size-len); */
/* 	} */
/* 	else */
/* 	{ */
/* 	    (void) memcpy(buf, s, size); */
/* 	    status	= UT_ENOROOM; */
/* 	} */
/*     } */

/*     return status; */
/* } */


/* /\* */
/*  * FORTRAN interface to the above functionality. */
/*  *\/ */
/* FCALLSCFUN2(INT,utPrint_help,UTENC,utenc, */
/*     PPUNIT,CBUF) */


/*
 *  Compare two strings.
 *
 *  This routine is used instead of strncmp(3) in order to obtain greater
 *  control over the returned value.
 *
 *  This function returns the index of the first, non-matching character or
 *  the length of the strings if all characters match.
 */
    static ptrdiff_t
mystrcmp(s1, s2)
    register char	*s1, *s2;
{
    register char	*start	= s1;

    for (; *s1 != 0 && *s2 != 0 && *s1 == *s2; ++s1, ++s2)
	continue;

    return s1 - start;
}


/*
 *  Compare two nodes in the units-table by comparing their names.
 *
 *  This function returns:
 *	<0	first argument less than second
 *	=0	first and second arguments are equal
 *	>0	first argument greater than second
 */
    static int
CompareNodes(node1, node2)
    const void	*node1;
    const void	*node2;
{
    /*
     * The following 2 lines might cause lint(1) to emit warnings about
     * "possible pointer alignment".  Thse may safely be ignored.
     */
    register char	*name1	= ((UnitEntry*)node1)->name;
    register char	*name2	= ((UnitEntry*)node2)->name;
    register int	status	= name1[0] - name2[0];	/* quick comparison */

    if (status == 0) {
	/*
	 * Quick comparison failed.  Perform a long comparison.
	 */
	status	= mystrcmp(name1, name2);
	status	= name1[status] - name2[status];
    }

    return status;
}


/*
 *  Compare a target and a node in the units-table by comparing their names.
 *  Allow the target name to be the plural of the node name if appropriate.
 *
 *  This function returns:
 *	<0	first argument less than second
 *	=0	first and second arguments are equal
 *	>0	first argument greater than second
 */
    static int
FindNodes(node1, node2)
    const void	*node1;
    const void	*node2;
{
    /*
     * The following 2 lines might cause lint(1) to emit warnings about
     * "possible pointer alignment".  Thse may safely be ignored.
     */
    register char		*name1	= ((UnitEntry*)node1)->name;
    register char		*name2	= ((UnitEntry*)node2)->name;
    register int		status	= name1[0] - name2[0];
						/* quick comparison */

    if (status == 0) {
	/*
	 * Quick comparison failed.  Perform a long comparison.
	 */
	int	i	= mystrcmp(name1, name2);

	status	= name1[i] - name2[i];

	if (status == 's') {
	    /*
	     * The following 2 lines might cause lint(1) to emit warnings about
	     * "possible pointer alignment".  Thse may safely be ignored.
	     */
	    UnitEntry	*n1	= (UnitEntry*)node1;
	    UnitEntry	*n2	= (UnitEntry*)node2;

	    if (n2->HasPlural && i == n2->nchr && n2->nchr+1 == n1->nchr)
		status	= 0;
	}
    }

    return status;
}


/*
 *  Create a new node.
 *
 *  This function returns:
 *	!=NULL	success
 *	==NULL	failure
 */
    static UnitEntry*
CreateNode(name, HasPlural, unit)
    char	*name;
    int		HasPlural;
    utUnit	*unit;
{
    int		nchr	= strlen(name);
    UnitEntry	*node	= NULL;

    if (nchr+1 > UT_NAMELEN) {
	(void) fprintf(stderr,
		       "udunits(3): The name \"%s\" is too long\n", name);
    } else {
	/*
	 * The following line might cause lint(1) to emit a warning about
	 * "possible pointer alignment".  This may safely be ignored.
	 */
	node	= (UnitEntry*)malloc(sizeof(UnitEntry));

	if (node == NULL) {
	    (void) fprintf(stderr, 
		"udunits(3): Couldn't allocate new entry\n");
	} else {
	    if ((node->name = DUPSTR(name)) == NULL) {
		(void) fprintf(stderr, 
			"udunits(3): Couldn't duplicate name\n");
		(void) free((char*)node);
	    } else {
		node->nchr	= strlen(node->name);
		node->HasPlural	= HasPlural;
		(void) utCopy(unit, &node->unit);
	    }
	}
    }

    return node;
}


/*
 * Copy construct a node (C++ terminology).
 */
    static UnitEntry*
CopyCtorNode(node)
    const UnitEntry	*node;
{
    return CreateNode(node->name, node->HasPlural, &node->unit);
}


/*
 *  Destroy a node.
 *
 *  This function returns void.
 */
    static void
DestroyNode(node)
    UnitEntry	*node;
{
    if (node != NULL) {
	if (node->name != NULL)
	    /*
	     * The following line might cause lint(1) to emit a warning about
	     * "possible pointer alignment".  This may safely be ignored.
	     */
	    (void) free((char*)node->name);
	(void) free((char*)node);
    }
}


/*
 * Assign one node to another.
 */
    static void
AssignNode(to, from)
    UnitEntry		*to;
    const UnitEntry	*from;
{
    DestroyNode(to);
    to->name = DUPSTR(from->name);
    to->nchr = from->nchr;
    to->HasPlural = from->HasPlural;
    to->unit = from->unit;
}
    

/*
 *  Add a unit-structure to the units-table.
 *
 *  This function returns:
 *	0		success
 *	UT_EALLOC	memory allocation failure
 *	UT_DUP		replaced previously-added unit with same name
 */
    int
utAdd(name,  HasPlural, unit)
    char	*name;
    int		HasPlural;
    utUnit	*unit;
{
    int		status	= 0;			/* return status = success */
    UnitEntry	*nodep	= CreateNode(name, HasPlural, unit);

    if (nodep == NULL)
	status = UT_EALLOC;
    else
    {
	UnitEntry	**found	= (UnitEntry**)tsearch((void*)nodep, &root, 
						       CompareNodes);

	if (found == NULL)
	{
	    (void) fprintf(stderr, 
		"udunits(3): Couldn't expand units-table for unit \"%s\"\n",
		name);
	    status	= UT_EALLOC;
	    DestroyNode(nodep);
	}
	else
	if (*found != nodep)
	{
	    DestroyNode(*found);
	    *found = nodep;
	    status = UT_DUP;
	}
    }

    return status;
}


/*
 *  Find the entry in the units-table corresponding to a given name.  
 *  If an entry isn't found, try again using the singular form, if
 *  appropriate.
 *
 *  This function returns:
 *	 NULL	entry not found;
 *	!NULL	entry found.
 */
    static UnitEntry*
FindUnit(name)
    char	*name;
{
    UnitEntry	node;
    UnitEntry	**found;

    node.name	= (char*)name;
    node.nchr	= strlen(name);

    /*
     * The following line might cause lint(1) to emit a warning about
     * "possible pointer alignment".  This may safely be ignored.
     */
    found	= (UnitEntry**)tfind((void*)&node, &root, FindNodes);

    if (found == NULL) {
	/*
	 * Not found.  If appropriate, try again with singular form.
	 */
	if (node.nchr > 1 && node.name[node.nchr-1] == 's') {
	    char	buf[UT_NAMELEN];

	    assert(sizeof(buf) > node.nchr-1);

	    node.name	= strncpy(buf, name, --node.nchr);
	    node.name[node.nchr]	= 0;

	    found	= (UnitEntry**)tfind((void*)&node, &root, 
	    /*
	     * The following line might cause lint(1) to emit a warning about
	     * "possible pointer alignment".  This may safely be ignored.
	     */
					     FindNodes);

	    /*
	     * Ensure that a plural form is allowed.
	     */
	    if (found != NULL && !(*found)->HasPlural)
		found	= NULL;
	}
    }

    return found == NULL ? NULL : *found;
}


/*
 *  Find the entry in a prefix-table corresponding to a possible
 *  prefix.  A linear-search of the prefix-table is performed.
 *  An attempt is made to insure that the longest, possible, matching
 *  entry is returned.
 *
 *  NB: A binary-search of the table is not possible because, for example,
 *  the prefix-entry for the input specification "mzmeters" (where "mz" is 
 *  a made-up prefix) would be after the entry "micro", but the prefix-entry 
 *  corresponding to the specification "mm" (i.e. "m") lies before "micro".
 *  Thus, the binary-search comparison-function can't indicate which direction
 *  to go.
 *
 *  This function returns:
 *	 NULL	not found
 *	!NULL	found
 */
    static PrefixEntry*
FindPrefix(spec)
    char	*spec;
{
    PrefixEntry			*found	= NULL;
    register PrefixEntry	*entry;

    for (entry = PrefixTable; entry->name != NULL; ++entry) {
	register int	status;

	if (entry->name[0] - spec[0] < 0 ||
		(status = strncmp(entry->name, spec, entry->nchar)) < 0)
	    continue;

	if (status > 0)
	    break;

	if (found == NULL || found->nchar < entry->nchar)
	    found	= entry;
    }

    return found;
}


/*
 *  Return the unit-structure corresponding to a unit-specification.
 *
 *  NB:
 *	It is permissible for the specification to consist solely
 *	of a prefix (e.g "milli").
 *
 *	An empty specification returns a dimensionless, unity unit-structure.
 *
 *	On failure, the output unit-structure is unmodified.
 *
 *  This function returns:
 *	0		found (the output unit-structure is set).
 *	UT_ENOINIT	the units-table hasn't been initialized
 *	UT_EUNKNOWN	not found
 */
    int
utFind(spec, up)
    char	*spec;
    utUnit	*up;
{
    int		status	= 0;		/* return status = found */
    UnitEntry	*entry	= NULL;
    double	factor	= 1;

    if (root == NULL) {
	(void) fprintf(stderr, "udunits(3): Units-table is empty\n");
	status	= UT_ENOINIT;
    } else {
	while (*spec != 0) {
	    PrefixEntry	*PrefixEnt;

	    /*
	     *  See if the specification is an isolated unit (e.g. "meter").
	     *  We're done if it is.
	     */
	    if ((entry = FindUnit(spec)) != NULL)
		break;

	    /*
	     *  See if the specification has a multiplying prefix.  If 
	     *  so, then use the prefix's dimensionless value, skip 
	     *  over the prefix characters, and rescan.
	     */

	    if ((PrefixEnt = FindPrefix(spec)) != NULL) {
		factor	*= PrefixEnt->factor;
		spec	+= strlen(PrefixEnt->name);
		continue;
	    }
	    
	    status	= UT_EUNKNOWN;
	    break;
	}					/* while something to decode */
    }						/* units-table is initialized */

    if (status == 0)
	(void)utScale(entry == NULL ? utClear(up) : &entry->unit,
		      factor, up);

    return status;
}


/*
 *  Compute the conversion factor between two unit-structures.
 *
 *  This function returns:
 *	0		success.
 *	UT_ENOINIT	the units-table hasn't been initialized
 *	UT_EINVALID	a structure is invalid
 *	UT_ECONVERT	the structures are not convertable
 */
    int
utConvert(from, to, slope, intercept)
    const utUnit	*from;
    const utUnit	*to;
    double		*slope, *intercept;
{
    int		status	= 0;

    if (!initialized) {
	(void) fprintf(stderr, 
		       "udunits(3): Package hasn't been initialized\n");
	status	= UT_ENOINIT;
    } else {
	if (from->factor == 0.0 || to->factor == 0.0) {
	    status	= UT_EINVALID;
	} else {
	    register int	iquan;

	    for (iquan = 0; iquan < UT_MAXNUM_BASE_QUANTITIES; ++iquan)
		if (from->power[iquan] != to->power[iquan]) {
		    status	= UT_ECONVERT;
		    break;
		}

	    if (status == 0) {
		/*
		 * Allow convertions between units with origins (e.g.
		 * Celsius) and those without (e.g. Kelvin) by using
		 * the fact that those without origins have a zero
		 * origin value.
		 */
		*slope	= from->factor / to->factor;
		*intercept	= (from->origin - to->origin) / to->factor;
	    }
	}
    }						/* package is initialized */

    return status;
}


/*     static int */
/* f_utConvert(from, to, slope, intercept) */
/*     const utUnit	*from; */
/*     const utUnit	*to; */
/*     DOUBLE_PRECISION	*slope; */
/*     DOUBLE_PRECISION	*intercept; */
/* { */
/*     double	tmpSlope; */
/*     double	tmpIntercept; */
/*     int		status; */
/*     status = utConvert(from, to, &tmpSlope, &tmpIntercept); */
/*     *slope = tmpSlope; */
/*     *intercept = tmpIntercept; */
/*     return status; */
/* } */


/* /\* */
/*  * FORTRAN interface to the above functionality. */
/*  *\/ */
/* FCALLSCFUN4(INT,f_utConvert,UTCVT,utcvt, */
/*     PPUNIT,PPUNIT,PDOUBLE,PDOUBLE) */


/*
 * Convert a Gregorian/Julian date and time into a temporal value.
 *
 * Returns:
 *	0		success
 *	UT_EINVALID	not a unit of time
 *	UT_ENOINIT	the units-table hasn't been initialized
 */
    int
utInvCalendar(year, month, day, hour, minute, second, unit, value)
    int		year;
    int		month;
    int		day;
    int		hour;
    int		minute;
    double	second;
    utUnit	*unit;
    double	*value;
{
    int		status;

    if (!utIsTime(unit) || !unit->hasorigin) {
	status	= UT_EINVALID;
    } else {
	*value	= (utencdate(year, month, day) + 
		   utencclock(hour, minute, second) - unit->origin) /
		  unit->factor;
	status	= 0;
    }

    return status;
}


/*     static int */
/* f_utInvCalendar(year, month, day, hour, minute, second, unit, value) */
/*     int			year; */
/*     int			month; */
/*     int			day; */
/*     int			hour; */
/*     int			minute; */
/*     double		second; */
/*     utUnit		*unit; */
/*     DOUBLE_PRECISION	*value; */
/* { */
/*     double	tmpValue; */
/*     int		status = */
/* 	utInvCalendar(year, month, day, hour, minute, second, unit, &tmpValue); */
/*     *value = tmpValue; */
/*     return status; */
/* } */


/* /\* */
/*  * FORTRAN interface to the above function. */
/*  *\/ */
/* FCALLSCFUN8(INT,f_utInvCalendar,UTICALTIME,uticaltime, */
/*     INT,INT,INT,INT, */
/* 	    INT,FLOAT,PPUNIT,PDOUBLE) */


/*
 * Convert a temporal value into UTC Gregorian/Julian date and time.
 *
 * Returns:
 *	0:		success
 *	UT_EINVALID:	not a unit of time
 */
    int
utCalendar(value, unit, year, month, day, hour, minute, second)
    double	value;
    utUnit	*unit;
    int		*year;
    int		*month;
    int		*day;
    int		*hour;
    int		*minute;
    float	*second;
{
    int		status;
    float	sec;

    if (!utIsTime(unit) || !unit->hasorigin) {
	status	= UT_EINVALID;
    } else {
	dectime(unit->origin + value*unit->factor, year, month, day, hour,
		minute, &sec);
	*second = sec;
	status	= 0;
    }

    return status;
}


/* /\* */
/*  * FORTRAN interface to the above function. */
/*  *\/ */
/* FCALLSCFUN8(INT,utCalendar,UTCALTIME,utcaltime, */
/*     DOUBLE,PPUNIT,PINT,PINT,PINT,PINT,PINT,PFLOAT) */


/*
 * Create a unit structure.
 */
    utUnit*
utNew()
{
    return utClear((utUnit*)malloc(sizeof(utUnit)));
}


/* /\* */
/*  * FORTRAN interface to the above functionality. */
/*  *\/ */
/* FCALLSCFUN0(PUNIT,utNew,UTMAKE,utmake) */


/*
 * Destroy a unit structure that was created by utNew().
 */
    void
utDestroy(unit)
    utUnit	*unit;
{
    if (unit != NULL)
	free(unit);
}


/* /\* */
/*  * FORTRAN interface to the above functionality. */
/*  *\/ */
/* FCALLSCSUB1(utDestroy,UTFREE,utfree, */
/*     PPUNIT) */


/*
 *  Free allocated nodes.
 */
    static void
NodeDeleter(node, order, level)
    const void	*node;
    VISIT	order;
    /*ARGSUSED*/
    int		level;
{
    if (order == leaf || order == endorder)
    {
	UnitEntry	*entry = *(UnitEntry**)node;

	(void)tdelete(entry, &root, CompareNodes);
	DestroyNode(entry);
    }
}


/*
 *  Terminate use of this package.
 *
 *  This function returns void.
 */
    void
utTerm()
{
    if (root != NULL) {
	twalk(root, NodeDeleter);
	root		= NULL;
    }
    initialized	= 0;
    NumberBaseUnits	= 0;
    UnitsFilePath[0]	= 0;
    HaveStdTimeUnit	= 0;
}


/* /\* */
/*  * FORTRAN interface to the above functionality. */
/*  *\/ */
/* FCALLSCSUB0(utTerm,UTCLS,utcls) */


/*
 * Return the next character to the scanner.
 */
    int
utinput()
{
    return unput_buf < unput_ptr
		? *--unput_ptr
		: *input_ptr == 0
		    ? EOF
		    : *input_ptr++;
}


/*
 * Save the given character to be returned by the next call to utinput().
 */
    void
utunput(c)
    int		c;
{
    *unput_ptr++	= c;
}


/*
 * LEX "wrap-up" routine.  Indicate no more input.
 */
    int
utwrap()
{
    return 1;
}


/*
 *  YACC error routine:
 */
    void
uterror(s)
    char        	*s;
{
    register int	i;

    (void) fprintf(stderr, "udunits(3): %s:\n", s);
    (void) fputs(input_buf, stderr);
    (void) putc('\n', stderr);
    for (i = 0; i < input_ptr - input_buf; ++i)
	(void) putc(' ', stderr);
    (void) fputs("^\n", stderr);
}
