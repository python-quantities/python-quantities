
typedef union  {
    double	rval;			/* floating-point numerical value */
    long	ival;			/* integer numerical value */
    char	name[UT_NAMELEN];	/* name of a quantity */
    utUnit	unit;			/* "unit" structure */
} YYSTYPE;
extern YYSTYPE yylval;
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
