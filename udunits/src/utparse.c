
# line 2 "utparse.y"
/*
 * $Id$
 *
 * yacc(1)-based parser for decoding formatted units(3) specifications.
 */

/*LINTLIBRARY*/

#include <stdio.h>
#include <string.h>
#include "udunits.h"
#include "utscan.h"
#include "utprivate.h"

/*
 * Allocate storage for the package global variables, which are declared in
 * header file "utprivate.h".
 */
int	UtLineno;		/* input-file line index */
int	UnitNotFound;		/* parser didn't find unit */
utUnit	*FinalUnit;		/* fully-parsed specification */


# line 26 "utparse.y"
typedef union  {
    double	rval;			/* floating-point numerical value */
    long	ival;			/* integer numerical value */
    char	name[UT_NAMELEN];	/* name of a quantity */
    utUnit	unit;			/* "unit" structure */
} YYSTYPE;
# define INT 257
# define ERR 258
# define SHIFT 259
# define SPACE 260
# define MULTIPLY 261
# define DIVIDE 262
# define EXPONENT 263
# define REAL 264
# define NAME 265
# define DATE 266
# define TIME 267
# define ZONE 268
#define yyparse utparse
#define yylex utlex
#define yyerror uterror
#define yylval utlval
#define yyval utval
#define yychar utchar
#define yydebug utdebug
#define yyerrflag uterrflag
#define yynerrs utnerrs
#define yyclearin yychar = -1
#define yyerrok yyerrflag = 0
extern int yychar;
extern int yyerrflag;
#ifndef YYMAXDEPTH
#define YYMAXDEPTH 150
#endif
YYSTYPE yylval, yyval;
typedef int yytabelem;
# define YYERRCODE 256

# line 172 "utparse.y"

static yytabelem yyexca[] ={
-1, 0,
	0, 1,
	-2, 0,
-1, 1,
	0, -1,
	-2, 0,
	};
# define YYNPROD 28
# define YYLAST 240
static yytabelem yyact[]={

     9,    36,    33,    24,    17,    29,    20,    35,    34,     9,
    18,    30,     7,    22,     9,    21,     5,     2,     1,     8,
     6,    14,     4,    25,     0,     0,    23,    19,     0,     0,
     0,     0,    27,    28,     0,     0,     0,    23,    32,     0,
    31,     0,     0,     0,     0,     0,     0,     0,     0,     0,
     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
     0,     0,     0,     0,     0,     0,     0,    10,     0,    13,
    10,    15,    16,     0,    11,    12,    10,    11,     0,    26,
     3,    10,     0,    11,    12,     0,     0,     0,    11,    12 };
static yytabelem yypact[]={

   -26, -1000, -1000, -1000,   -40,  -253, -1000, -1000, -1000,   -31,
 -1000, -1000,  -251,   -37,  -253,   -31,   -31, -1000,  -252,   -30,
 -1000, -1000, -1000, -1000,   -37, -1000,  -265,  -253,  -253, -1000,
 -1000,   -33,   -34,  -267, -1000, -1000, -1000 };
static yytabelem yypgo[]={

     0,    12,    15,    13,    23,    22,    16,    20,    19,    17,
    18 };
static yytabelem yyr1[]={

     0,    10,    10,    10,     9,     9,     9,     5,     5,     5,
     5,     6,     6,     6,     7,     7,     7,     8,     8,     2,
     2,     1,     1,     3,     3,     4,     4,     4 };
static yytabelem yyr2[]={

     0,     1,     3,     3,     3,     7,     7,     3,     5,     7,
     7,     3,     5,     7,     3,     3,     7,     3,     5,     3,
     7,     3,     3,     3,     7,     3,     5,     7 };
static yytabelem yychk[]={

 -1000,   -10,    -9,   256,    -5,    -6,    -7,    -1,    -8,    40,
   257,   264,   265,   259,    -6,   261,   262,   257,   263,    -9,
   257,    -2,    -3,    -1,    40,    -4,   266,    -6,    -6,   257,
    41,    -2,    -3,   267,    41,    41,   268 };
static yytabelem yydef[]={

    -2,    -2,     2,     3,     4,     7,    11,    14,    15,     0,
    21,    22,    17,     0,     8,     0,     0,    12,     0,     0,
    18,     5,     6,    19,     0,    23,    25,     9,    10,    13,
    16,     0,     0,    26,    20,    24,    27 };
typedef struct { char *t_name; int t_val; } yytoktype;
#ifndef YYDEBUG
#	define YYDEBUG	0	/* don't allow debugging */
#endif

#if YYDEBUG

yytoktype yytoks[] =
{
	"INT",	257,
	"ERR",	258,
	"SHIFT",	259,
	"SPACE",	260,
	"MULTIPLY",	261,
	"DIVIDE",	262,
	"EXPONENT",	263,
	"REAL",	264,
	"NAME",	265,
	"DATE",	266,
	"TIME",	267,
	"ZONE",	268,
	"-unknown-",	-1	/* ends search */
};

char * yyreds[] =
{
	"-no such reduction-",
      "unit_spec : /* empty */",
      "unit_spec : origin_exp",
      "unit_spec : error",
      "origin_exp : unit_exp",
      "origin_exp : unit_exp SHIFT value_exp",
      "origin_exp : unit_exp SHIFT timestamp",
      "unit_exp : power_exp",
      "unit_exp : unit_exp power_exp",
      "unit_exp : unit_exp MULTIPLY power_exp",
      "unit_exp : unit_exp DIVIDE power_exp",
      "power_exp : factor_exp",
      "power_exp : power_exp INT",
      "power_exp : power_exp EXPONENT INT",
      "factor_exp : number_exp",
      "factor_exp : quant_exp",
      "factor_exp : '(' origin_exp ')'",
      "quant_exp : NAME",
      "quant_exp : NAME INT",
      "value_exp : number_exp",
      "value_exp : '(' value_exp ')'",
      "number_exp : INT",
      "number_exp : REAL",
      "timestamp : time_exp",
      "timestamp : '(' timestamp ')'",
      "time_exp : DATE",
      "time_exp : DATE TIME",
      "time_exp : DATE TIME ZONE",
};
#endif /* YYDEBUG */
/*
 * (c) Copyright 1990, OPEN SOFTWARE FOUNDATION, INC.
 * ALL RIGHTS RESERVED
 */
/*
 * OSF/1 Release 1.0
 */
/* @(#)yaccpar	1.3  com/cmd/lang/yacc,3.1, 9/7/89 18:46:37 */
/*
** Skeleton parser driver for yacc output
*/

/*
** yacc user known macros and defines
*/
#ifdef YYSPLIT
#   define YYERROR	return(-2)
#else
#   define YYERROR	goto yyerrlab
#endif

#define YYACCEPT	return(0)
#define YYABORT		return(1)
#define YYBACKUP( newtoken, newvalue )\
{\
	if ( yychar >= 0 || ( yyr2[ yytmp ] >> 1 ) != 1 )\
	{\
		yyerror( "syntax error - cannot backup" );\
		goto yyerrlab;\
	}\
	yychar = newtoken;\
	yystate = *yyps;\
	yylval = newvalue;\
	goto yynewstate;\
}
#define YYRECOVERING()	(!!yyerrflag)
#ifndef YYDEBUG
#	define YYDEBUG	1	/* make debugging available */
#endif

/*
** user known globals
*/
int yydebug;			/* set to 1 to get debugging */

/*
** driver internal defines
*/
#define YYFLAG		(-1000)

#ifdef YYSPLIT
#   define YYSCODE { \
			extern int (*yyf[])(); \
			register int yyret; \
			if (yyf[yytmp]) \
			    if ((yyret=(*yyf[yytmp])()) == -2) \
				    goto yyerrlab; \
				else if (yyret>=0) return(yyret); \
		   }
#endif

/*
** global variables used by the parser
*/
YYSTYPE yyv[ YYMAXDEPTH ];	/* value stack */
int yys[ YYMAXDEPTH ];		/* state stack */

YYSTYPE *yypv;			/* top of value stack */
YYSTYPE *yypvt;			/* top of value stack for $vars */
int *yyps;			/* top of state stack */

int yystate;			/* current state */
int yytmp;			/* extra var (lasts between blocks) */

int yynerrs;			/* number of errors */
int yyerrflag;			/* error recovery flag */
int yychar;			/* current input token number */



/*
** yyparse - return 0 if worked, 1 if syntax error not recovered from
*/
int
yyparse()
{
	/*
	** Initialize externals - yyparse may be called more than once
	*/
	yypv = &yyv[-1];
	yyps = &yys[-1];
	yystate = 0;
	yytmp = 0;
	yynerrs = 0;
	yyerrflag = 0;
	yychar = -1;

	goto yystack;
	{
		register YYSTYPE *yy_pv;	/* top of value stack */
		register int *yy_ps;		/* top of state stack */
		register int yy_state;		/* current state */
		register int  yy_n;		/* internal state number info */

		/*
		** get globals into registers.
		** branch to here only if YYBACKUP was called.
		*/
	yynewstate:
		yy_pv = yypv;
		yy_ps = yyps;
		yy_state = yystate;
		goto yy_newstate;

		/*
		** get globals into registers.
		** either we just started, or we just finished a reduction
		*/
	yystack:
		yy_pv = yypv;
		yy_ps = yyps;
		yy_state = yystate;

		/*
		** top of for (;;) loop while no reductions done
		*/
	yy_stack:
		/*
		** put a state and value onto the stacks
		*/
#if YYDEBUG
		/*
		** if debugging, look up token value in list of value vs.
		** name pairs.  0 and negative (-1) are special values.
		** Note: linear search is used since time is not a real
		** consideration while debugging.
		*/
		if ( yydebug )
		{
			register int yy_i;

			printf( "State %d, token ", yy_state );
			if ( yychar == 0 )
				printf( "end-of-file\n" );
			else if ( yychar < 0 )
				printf( "-none-\n" );
			else
			{
				for ( yy_i = 0; yytoks[yy_i].t_val >= 0;
					yy_i++ )
				{
					if ( yytoks[yy_i].t_val == yychar )
						break;
				}
				printf( "%s\n", yytoks[yy_i].t_name );
			}
		}
#endif /* YYDEBUG */
		if ( ++yy_ps >= &yys[ YYMAXDEPTH ] )	/* room on stack? */
		{
			yyerror( "yacc stack overflow" );
			YYABORT;
		}
		*yy_ps = yy_state;
		*++yy_pv = yyval;

		/*
		** we have a new state - find out what to do
		*/
	yy_newstate:
		if ( ( yy_n = yypact[ yy_state ] ) <= YYFLAG )
			goto yydefault;		/* simple state */
#if YYDEBUG
		/*
		** if debugging, need to mark whether new token grabbed
		*/
		yytmp = yychar < 0;
#endif
		if ( ( yychar < 0 ) && ( ( yychar = yylex() ) < 0 ) )
			yychar = 0;		/* reached EOF */
#if YYDEBUG
		if ( yydebug && yytmp )
		{
			register int yy_i;

			printf( "Received token " );
			if ( yychar == 0 )
				printf( "end-of-file\n" );
			else if ( yychar < 0 )
				printf( "-none-\n" );
			else
			{
				for ( yy_i = 0; yytoks[yy_i].t_val >= 0;
					yy_i++ )
				{
					if ( yytoks[yy_i].t_val == yychar )
						break;
				}
				printf( "%s\n", yytoks[yy_i].t_name );
			}
		}
#endif /* YYDEBUG */
		if ( ( ( yy_n += yychar ) < 0 ) || ( yy_n >= YYLAST ) )
			goto yydefault;
		if ( yychk[ yy_n = yyact[ yy_n ] ] == yychar )	/*valid shift*/
		{
			yychar = -1;
			yyval = yylval;
			yy_state = yy_n;
			if ( yyerrflag > 0 )
				yyerrflag--;
			goto yy_stack;
		}

	yydefault:
		if ( ( yy_n = yydef[ yy_state ] ) == -2 )
		{
#if YYDEBUG
			yytmp = yychar < 0;
#endif
			if ( ( yychar < 0 ) && ( ( yychar = yylex() ) < 0 ) )
				yychar = 0;		/* reached EOF */
#if YYDEBUG
			if ( yydebug && yytmp )
			{
				register int yy_i;

				printf( "Received token " );
				if ( yychar == 0 )
					printf( "end-of-file\n" );
				else if ( yychar < 0 )
					printf( "-none-\n" );
				else
				{
					for ( yy_i = 0;
						yytoks[yy_i].t_val >= 0;
						yy_i++ )
					{
						if ( yytoks[yy_i].t_val
							== yychar )
						{
							break;
						}
					}
					printf( "%s\n", yytoks[yy_i].t_name );
				}
			}
#endif /* YYDEBUG */
			/*
			** look through exception table
			*/
			{
				register int *yyxi = yyexca;

				while ( ( *yyxi != -1 ) ||
					( yyxi[1] != yy_state ) )
				{
					yyxi += 2;
				}
				while ( ( *(yyxi += 2) >= 0 ) &&
					( *yyxi != yychar ) )
					;
				if ( ( yy_n = yyxi[1] ) < 0 )
					YYACCEPT;
			}
		}

		/*
		** check for syntax error
		*/
		if ( yy_n == 0 )	/* have an error */
		{
			/* no worry about speed here! */
			switch ( yyerrflag )
			{
			case 0:		/* new error */
				yyerror( "syntax error" );
				goto skip_init;
			yyerrlab:
				/*
				** get globals into registers.
				** we have a user generated syntax type error
				*/
				yy_pv = yypv;
				yy_ps = yyps;
				yy_state = yystate;
				yynerrs++;
			skip_init:
			case 1:
			case 2:		/* incompletely recovered error */
					/* try again... */
				yyerrflag = 3;
				/*
				** find state where "error" is a legal
				** shift action
				*/
				while ( yy_ps >= yys )
				{
					yy_n = yypact[ *yy_ps ] + YYERRCODE;
					if ( yy_n >= 0 && yy_n < YYLAST &&
						yychk[yyact[yy_n]] == YYERRCODE)					{
						/*
						** simulate shift of "error"
						*/
						yy_state = yyact[ yy_n ];
						goto yy_stack;
					}
					/*
					** current state has no shift on
					** "error", pop stack
					*/
#if YYDEBUG
#	define _POP_ "Error recovery pops state %d, uncovers state %d\n"
					if ( yydebug )
						printf( _POP_, *yy_ps,
							yy_ps[-1] );
#	undef _POP_
#endif
					yy_ps--;
					yy_pv--;
				}
				/*
				** there is no state on stack with "error" as
				** a valid shift.  give up.
				*/
				YYABORT;
			case 3:		/* no shift yet; eat a token */
#if YYDEBUG
				/*
				** if debugging, look up token in list of
				** pairs.  0 and negative shouldn't occur,
				** but since timing doesn't matter when
				** debugging, it doesn't hurt to leave the
				** tests here.
				*/
				if ( yydebug )
				{
					register int yy_i;

					printf( "Error recovery discards " );
					if ( yychar == 0 )
						printf( "token end-of-file\n" );
					else if ( yychar < 0 )
						printf( "token -none-\n" );
					else
					{
						for ( yy_i = 0;
							yytoks[yy_i].t_val >= 0;
							yy_i++ )
						{
							if ( yytoks[yy_i].t_val
								== yychar )
							{
								break;
							}
						}
						printf( "token %s\n",
							yytoks[yy_i].t_name );
					}
				}
#endif /* YYDEBUG */
				if ( yychar == 0 )	/* reached EOF. quit */
					YYABORT;
				yychar = -1;
				goto yy_newstate;
			}
		}/* end if ( yy_n == 0 ) */
		/*
		** reduction by production yy_n
		** put stack tops, etc. so things right after switch
		*/
#if YYDEBUG
		/*
		** if debugging, print the string that is the user's
		** specification of the reduction which is just about
		** to be done.
		*/
		if ( yydebug )
			printf( "Reduce by (%d) \"%s\"\n",
				yy_n, yyreds[ yy_n ] );
#endif
		yytmp = yy_n;			/* value to switch over */
		yypvt = yy_pv;			/* $vars top of value stack */
		/*
		** Look in goto table for next state
		** Sorry about using yy_state here as temporary
		** register variable, but why not, if it works...
		** If yyr2[ yy_n ] doesn't have the low order bit
		** set, then there is no action to be done for
		** this reduction.  So, no saving & unsaving of
		** registers done.  The only difference between the
		** code just after the if and the body of the if is
		** the goto yy_stack in the body.  This way the test
		** can be made before the choice of what to do is needed.
		*/
		{
			/* length of production doubled with extra bit */
			register int yy_len = yyr2[ yy_n ];

			if ( !( yy_len & 01 ) )
			{
				yy_len >>= 1;
				yyval = ( yy_pv -= yy_len )[1];	/* $$ = $1 */
				yy_state = yypgo[ yy_n = yyr1[ yy_n ] ] +
					*( yy_ps -= yy_len ) + 1;
				if ( yy_state >= YYLAST ||
					yychk[ yy_state =
					yyact[ yy_state ] ] != -yy_n )
				{
					yy_state = yyact[ yypgo[ yy_n ] ];
				}
				goto yy_stack;
			}
			yy_len >>= 1;
			yyval = ( yy_pv -= yy_len )[1];	/* $$ = $1 */
			yy_state = yypgo[ yy_n = yyr1[ yy_n ] ] +
				*( yy_ps -= yy_len ) + 1;
			if ( yy_state >= YYLAST ||
				yychk[ yy_state = yyact[ yy_state ] ] != -yy_n )
			{
				yy_state = yyact[ yypgo[ yy_n ] ];
			}
		}
					/* save until reenter driver code */
		yystate = yy_state;
		yyps = yy_ps;
		yypv = yy_pv;
	}
	/*
	** code supplied by user is placed in this switch
	*/

		switch(yytmp){

case 1:
# line 59 "utparse.y"
{
			YYACCEPT;
		  } /*NOTREACHED*/ break;
case 2:
# line 62 "utparse.y"
{
			(void)utCopy(&yypvt[-0].unit, FinalUnit);
			YYACCEPT;
		  } /*NOTREACHED*/ break;
case 3:
# line 66 "utparse.y"
{
			yyerrok;
			yyclearin;
			YYABORT;
		  } /*NOTREACHED*/ break;
case 4:
# line 73 "utparse.y"
{
			(void)utCopy(&yypvt[-0].unit, &yyval.unit);
		  } /*NOTREACHED*/ break;
case 5:
# line 76 "utparse.y"
{
			if (utIsTime(&yypvt[-2].unit)) {
			    /*
			     * The shift amount is divided by the unit scale
			     * factor in the following because the shift amount
			     * must be in the units of the first argument (e.g.
			     * 0.555556 kelvins for the fahrenheit unit) and a
			     * timestamp isn't necessarily so proportioned.
			     */
			    (void)utShift(&yypvt[-2].unit, yypvt[-0].rval/yypvt[-2].unit.factor, &yyval.unit);
			} else {
			    (void) utShift(&yypvt[-2].unit, yypvt[-0].rval, &yyval.unit);
			}
		  } /*NOTREACHED*/ break;
case 6:
# line 90 "utparse.y"
{
			if (utIsTime(&yypvt[-2].unit)) {
			    (void)utShift(&yypvt[-2].unit, yypvt[-0].rval/yypvt[-2].unit.factor, &yyval.unit);
			} else {
			    UnitNotFound	= 1;
			    YYERROR;
			}
		  } /*NOTREACHED*/ break;
case 7:
# line 100 "utparse.y"
{
			(void)utCopy(&yypvt[-0].unit, &yyval.unit);
		  } /*NOTREACHED*/ break;
case 8:
# line 103 "utparse.y"
{
			(void)utMultiply(&yypvt[-1].unit, &yypvt[-0].unit, &yyval.unit);
		  } /*NOTREACHED*/ break;
case 9:
# line 106 "utparse.y"
{
			(void)utMultiply(&yypvt[-2].unit, &yypvt[-0].unit, &yyval.unit);
		  } /*NOTREACHED*/ break;
case 10:
# line 109 "utparse.y"
{
			(void)utDivide(&yypvt[-2].unit, &yypvt[-0].unit, &yyval.unit);
		  } /*NOTREACHED*/ break;
case 11:
# line 114 "utparse.y"
{
			(void)utCopy(&yypvt[-0].unit, &yyval.unit);
		  } /*NOTREACHED*/ break;
case 12:
# line 117 "utparse.y"
{
			(void)utRaise(&yypvt[-1].unit, yypvt[-0].ival, &yyval.unit);
		  } /*NOTREACHED*/ break;
case 13:
# line 120 "utparse.y"
{
			(void)utRaise(&yypvt[-2].unit, yypvt[-0].ival, &yyval.unit);
		  } /*NOTREACHED*/ break;
case 14:
# line 125 "utparse.y"
{
			utUnit	unit;
			(void)utScale(utClear(&unit), yypvt[-0].rval, &yyval.unit);
		  } /*NOTREACHED*/ break;
case 15:
# line 129 "utparse.y"
{
			(void)utCopy(&yypvt[-0].unit, &yyval.unit);
		  } /*NOTREACHED*/ break;
case 16:
# line 132 "utparse.y"
{
			(void)utCopy(&yypvt[-1].unit, &yyval.unit);
		  } /*NOTREACHED*/ break;
case 17:
# line 137 "utparse.y"
{
			utUnit     unit;
			if (utFind(yypvt[-0].name, &unit) != 0) {
			    UnitNotFound	= 1;
			    YYERROR;
			}
			(void)utCopy(&unit, &yyval.unit);
		  } /*NOTREACHED*/ break;
case 18:
# line 145 "utparse.y"
{
			utUnit     unit;
			if (utFind(yypvt[-1].name, &unit) != 0) {
			    UnitNotFound	= 1;
			    YYERROR;
			}
			(void)utRaise(&unit, yypvt[-0].ival, &yyval.unit);
		  } /*NOTREACHED*/ break;
case 19:
# line 155 "utparse.y"
{ yyval.rval = yypvt[-0].rval; } /*NOTREACHED*/ break;
case 20:
# line 156 "utparse.y"
{ yyval.rval = yypvt[-1].rval; } /*NOTREACHED*/ break;
case 21:
# line 159 "utparse.y"
{ yyval.rval = yypvt[-0].ival; } /*NOTREACHED*/ break;
case 22:
# line 160 "utparse.y"
{ yyval.rval = yypvt[-0].rval; } /*NOTREACHED*/ break;
case 23:
# line 163 "utparse.y"
{ yyval.rval = yypvt[-0].rval; } /*NOTREACHED*/ break;
case 24:
# line 164 "utparse.y"
{ yyval.rval = yypvt[-1].rval; } /*NOTREACHED*/ break;
case 25:
# line 167 "utparse.y"
{ yyval.rval = yypvt[-0].rval; } /*NOTREACHED*/ break;
case 26:
# line 168 "utparse.y"
{ yyval.rval = yypvt[-1].rval + yypvt[-0].rval; } /*NOTREACHED*/ break;
case 27:
# line 169 "utparse.y"
{ yyval.rval = yypvt[-2].rval + (yypvt[-1].rval - yypvt[-0].rval); } /*NOTREACHED*/ break;
}


	goto yystack;		/* reset registers in driver code */
}
