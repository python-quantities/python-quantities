# -*- coding: utf-8 -*-

import sys
import pickle
import copy

from .. import units as pq
from ..quantity import Quantity
from ..uncertainquantity import UncertainQuantity
from .. import constants
from .common import TestCase


class TestPersistence(TestCase):

    def test_unitquantity_persistence(self):
        x = pq.m
        y = pickle.loads(pickle.dumps(x))
        self.assertQuantityEqual(x, y)

        x = pq.CompoundUnit("pc/cm**3")
        y = pickle.loads(pickle.dumps(x))
        self.assertQuantityEqual(x, y)

    def test_quantity_persistence(self):
        x = 20*pq.m
        y = pickle.loads(pickle.dumps(x))
        self.assertQuantityEqual(x, y)

    def test_uncertainquantity_persistence(self):
        x = UncertainQuantity(20, 'm', 0.2)
        y = pickle.loads(pickle.dumps(x))
        self.assertQuantityEqual(x, y)

    def test_unitconstant_persistence(self):
        x = constants.m_e
        y = pickle.loads(pickle.dumps(x))
        self.assertQuantityEqual(x, y)

    def test_quantity_object_dtype(self):
        # Regression test for github issue #113
        x = Quantity(1,dtype=object)
        y = pickle.loads(pickle.dumps(x))
        self.assertQuantityEqual(x, y)

    def test_uncertainquantity_object_dtype(self):
        # Regression test for github issue #113
        x = UncertainQuantity(20, 'm', 0.2, dtype=object)
        y = pickle.loads(pickle.dumps(x))
        self.assertQuantityEqual(x, y)

    def test_backward_compat(self):
        """ A few pickles collected before fixing #113 just to make sure we remain backwards compatible. """
        orig = [
            pq.m,
            20*pq.m,
            UncertainQuantity(20, 'm', 0.2),
            constants.m_e,
        ]
        if sys.version_info[0] < 3:
            data = [
                # generated in python 2 with protocol=-1 (i.e. newest == 2)
                b'\x80\x02cquantities.unitquantity\nUnitLength\nq\x00(U\x05meterq\x01NU\x01mq\x02N]q\x03(U\x06metersq\x04U\x05metreq\x05U\x06metresq\x06eNtq\x07Rq\x08K\x01K\x02K\x02\x86q\t\x86q\nb.',
                b'\x80\x02cquantities.quantity\n_reconstruct_quantity\nq\x00(cquantities.quantity\nQuantity\nq\x01cnumpy\nndarray\nq\x02K\x00\x85q\x03U\x01bq\x04tq\x05Rq\x06(K\x01)cnumpy\ndtype\nq\x07U\x02f8q\x08K\x00K\x01\x87q\tRq\n(K\x03U\x01<q\x0bNNNJ\xff\xff\xff\xffJ\xff\xff\xff\xffK\x00tq\x0cb\x89U\x08\x00\x00\x00\x00\x00\x004@q\rcquantities.dimensionality\nDimensionality\nq\x0e)\x81q\x0fcquantities.unitquantity\nUnitLength\nq\x10(U\x05meterq\x11NU\x01mq\x12N]q\x13(U\x06metersq\x14U\x05metreq\x15U\x06metresq\x16eNtq\x17Rq\x18K\x01K\x02K\x02\x86q\x19\x86q\x1abK\x01s}q\x1bbtq\x1cb.',
                b'\x80\x02cquantities.quantity\n_reconstruct_quantity\nq\x00(cquantities.uncertainquantity\nUncertainQuantity\nq\x01cnumpy\nndarray\nq\x02K\x00\x85q\x03U\x01bq\x04tq\x05Rq\x06(K\x01)cnumpy\ndtype\nq\x07U\x02f8q\x08K\x00K\x01\x87q\tRq\n(K\x03U\x01<q\x0bNNNJ\xff\xff\xff\xffJ\xff\xff\xff\xffK\x00tq\x0cb\x89U\x08\x00\x00\x00\x00\x00\x004@q\rcquantities.dimensionality\nDimensionality\nq\x0e)\x81q\x0fcquantities.unitquantity\nUnitLength\nq\x10(U\x05meterq\x11NU\x01mq\x12N]q\x13(U\x06metersq\x14U\x05metreq\x15U\x06metresq\x16eNtq\x17Rq\x18K\x01K\x02K\x02\x86q\x19\x86q\x1abK\x01s}q\x1bbh\x00(cquantities.quantity\nQuantity\nq\x1ch\x02h\x03h\x04tq\x1dRq\x1e(K\x01)h\n\x89U\x08\x9a\x99\x99\x99\x99\x99\xc9?q\x1fh\x0e)\x81q h\x18K\x01s}q!btq"btq#b.',
                b'\x80\x02cquantities.unitquantity\nUnitConstant\nq\x00(U\relectron_massq\x01cquantities.quantity\n_reconstruct_quantity\nq\x02(cquantities.quantity\nQuantity\nq\x03cnumpy\nndarray\nq\x04K\x00\x85q\x05U\x01bq\x06tq\x07Rq\x08(K\x01)cnumpy\ndtype\nq\tU\x02f8q\nK\x00K\x01\x87q\x0bRq\x0c(K\x03U\x01<q\rNNNJ\xff\xff\xff\xffJ\xff\xff\xff\xffK\x00tq\x0eb\x89U\x08N?\xeb\x93\xdcy\xb29q\x0fcquantities.dimensionality\nDimensionality\nq\x10)\x81q\x11cquantities.unitquantity\nUnitMass\nq\x12(U\x08kilogramq\x13NU\x02kgq\x14N]q\x15U\tkilogramsq\x16aNtq\x17Rq\x18K\x01K\x01K\x1b\x86q\x19\x86q\x1abK\x01s}q\x1bbtq\x1cbU\x03m_eq\x1dU\x04m\xe2\x82\x91q\x1e]q\x1fNtq Rq!K\x01K\x00M\x0f\x01\x86q"\x86q#b.',
            ]
        else:
            data = [
                # generated in python 3 with protocol=2
                b'\x80\x02cquantities.unitquantity\nUnitLength\nq\x00(X\x05\x00\x00\x00meterq\x01NX\x01\x00\x00\x00mq\x02N]q\x03(X\x06\x00\x00\x00metersq\x04X\x05\x00\x00\x00metreq\x05X\x06\x00\x00\x00metresq\x06eNtq\x07Rq\x08K\x01K\x02K\x02\x86q\t\x86q\nb.',
                b'\x80\x02cquantities.quantity\n_reconstruct_quantity\nq\x00(cquantities.quantity\nQuantity\nq\x01cnumpy\nndarray\nq\x02K\x00\x85q\x03X\x01\x00\x00\x00bq\x04tq\x05Rq\x06(K\x01)cnumpy\ndtype\nq\x07X\x02\x00\x00\x00f8q\x08K\x00K\x01\x87q\tRq\n(K\x03X\x01\x00\x00\x00<q\x0bNNNJ\xff\xff\xff\xffJ\xff\xff\xff\xffK\x00tq\x0cb\x89c_codecs\nencode\nq\rX\x08\x00\x00\x00\x00\x00\x00\x00\x00\x004@q\x0eX\x06\x00\x00\x00latin1q\x0f\x86q\x10Rq\x11cquantities.dimensionality\nDimensionality\nq\x12)\x81q\x13cquantities.unitquantity\nUnitLength\nq\x14(X\x05\x00\x00\x00meterq\x15NX\x01\x00\x00\x00mq\x16N]q\x17(X\x06\x00\x00\x00metersq\x18X\x05\x00\x00\x00metreq\x19X\x06\x00\x00\x00metresq\x1aeNtq\x1bRq\x1cK\x01K\x02K\x02\x86q\x1d\x86q\x1ebK\x01stq\x1fb.',
                b'\x80\x02cquantities.quantity\n_reconstruct_quantity\nq\x00(cquantities.uncertainquantity\nUncertainQuantity\nq\x01cnumpy\nndarray\nq\x02K\x00\x85q\x03X\x01\x00\x00\x00bq\x04tq\x05Rq\x06(K\x01)cnumpy\ndtype\nq\x07X\x02\x00\x00\x00f8q\x08K\x00K\x01\x87q\tRq\n(K\x03X\x01\x00\x00\x00<q\x0bNNNJ\xff\xff\xff\xffJ\xff\xff\xff\xffK\x00tq\x0cb\x89c_codecs\nencode\nq\rX\x08\x00\x00\x00\x00\x00\x00\x00\x00\x004@q\x0eX\x06\x00\x00\x00latin1q\x0f\x86q\x10Rq\x11cquantities.dimensionality\nDimensionality\nq\x12)\x81q\x13cquantities.unitquantity\nUnitLength\nq\x14(X\x05\x00\x00\x00meterq\x15NX\x01\x00\x00\x00mq\x16N]q\x17(X\x06\x00\x00\x00metersq\x18X\x05\x00\x00\x00metreq\x19X\x06\x00\x00\x00metresq\x1aeNtq\x1bRq\x1cK\x01K\x02K\x02\x86q\x1d\x86q\x1ebK\x01sh\x00(cquantities.quantity\nQuantity\nq\x1fh\x02h\x03h\x04tq Rq!(K\x01)h\n\x89h\rX\x0f\x00\x00\x00\xc2\x9a\xc2\x99\xc2\x99\xc2\x99\xc2\x99\xc2\x99\xc3\x89?q"h\x0f\x86q#Rq$h\x12)\x81q%h\x1cK\x01stq&btq\'b.',
                b'\x80\x02cquantities.unitquantity\nUnitConstant\nq\x00(X\r\x00\x00\x00electron_massq\x01cquantities.quantity\n_reconstruct_quantity\nq\x02(cquantities.quantity\nQuantity\nq\x03cnumpy\nndarray\nq\x04K\x00\x85q\x05X\x01\x00\x00\x00bq\x06tq\x07Rq\x08(K\x01)cnumpy\ndtype\nq\tX\x02\x00\x00\x00f8q\nK\x00K\x01\x87q\x0bRq\x0c(K\x03X\x01\x00\x00\x00<q\rNNNJ\xff\xff\xff\xffJ\xff\xff\xff\xffK\x00tq\x0eb\x89c_codecs\nencode\nq\x0fX\x0c\x00\x00\x00N?\xc3\xab\xc2\x93\xc3\x9cy\xc2\xb29q\x10X\x06\x00\x00\x00latin1q\x11\x86q\x12Rq\x13cquantities.dimensionality\nDimensionality\nq\x14)\x81q\x15cquantities.unitquantity\nUnitMass\nq\x16(X\x08\x00\x00\x00kilogramq\x17NX\x02\x00\x00\x00kgq\x18N]q\x19X\t\x00\x00\x00kilogramsq\x1aaNtq\x1bRq\x1cK\x01K\x01K?\x86q\x1d\x86q\x1ebK\x01stq\x1fbX\x03\x00\x00\x00m_eq X\x04\x00\x00\x00m\xe2\x82\x91q!]q"Ntq#Rq$K\x01K\x00M!\x01\x86q%\x86q&b.',
            ]
        for x,d in zip(orig,data):
            y = pickle.loads(d)
            self.assertQuantityEqual(x, y)

    def test_copy_quantity(self):
        for dtype in [float,object]:
            x = (20*pq.m).astype(dtype)
            y = copy.copy(x)
            self.assertQuantityEqual(x, y)

    def test_copy_uncertainquantity(self):
        for dtype in [float, object]:
            x = UncertainQuantity(20, 'm', 0.2).astype(dtype)
            y = copy.copy(x)
            self.assertQuantityEqual(x, y)