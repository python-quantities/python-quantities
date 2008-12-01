#include <Python.h>
#include "numpy/arrayobject.h"
#include <udunits.h>

static PyObject *
init(self,args)
     PyObject *self;
     PyObject *args;
{
  char *file;
  if (!PyArg_ParseTuple(args,"s",&file))
    return NULL;
  if ( utInit(file) != 0 )
    {
      PyErr_SetString(PyExc_TypeError, "Error - Must provide a valid file name.");
      return NULL;
    }
  Py_INCREF ((PyObject *)Py_None); return Py_None;
}


static PyObject *
convert(self,args)
     PyObject *self;
     PyObject *args;
{
  char *unit1,*unit2;
  utUnit udunit1,udunit2;
  double Slope,Intercept;
  char err[256];

  if (!PyArg_ParseTuple(args,"ss",&unit1,&unit2))
    return NULL;
  if ( utScan(unit1, &udunit1) !=0 )
    {
      sprintf(err,"UDUNITS Error: invalid udunits: %s",unit1);
      PyErr_SetString(PyExc_TypeError, err);
      return NULL;
    }
  if ( utScan(unit2, &udunit2) !=0 )
    {
      sprintf(err,"UDUNITS Error: invalid udunits: %s",unit2);
      PyErr_SetString(PyExc_TypeError, err);
      return NULL;
    }
  if (utConvert(&udunit1, &udunit2, &Slope, &Intercept) != 0 )
    {
      sprintf(err,"UDUNITS Error: cannot convert from %s to %s",unit1,unit2);
      PyErr_SetString(PyExc_TypeError, "Error Converting.");
      return NULL;
    }
  return Py_BuildValue("dd", Slope, Intercept);
}


static PyObject *
term(self,args)
     PyObject *self;
     PyObject *args;
{
  utTerm();
}
static PyMethodDef MyUdunitsMethods[]= {
  {"init", init , METH_VARARGS},
  {"convert", convert , METH_VARARGS},
  {"term", term , METH_VARARGS},
  {NULL, NULL} /*sentinel */
};

void
initudunits()
{
  (void) Py_InitModule("udunits", MyUdunitsMethods);
  import_array()

}

int main(int argc,char **argv)
{
  Py_SetProgramName(argv[0]);
  Py_Initialize();
  initudunits();}

