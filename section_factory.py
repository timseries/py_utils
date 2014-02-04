#!/usr/bin/python -tt
from __future__ import generators
from py_utils import *

class SectionFactory(object):
    """
    Factory for creating  objects derived from Section.
    """
    factories = {}
    
    def add_factory(str_id, sec_factory):
        SectionFactory.factories[str_id] = sec_factory
    add_factory = staticmethod(add_factory)
    
    def create_section(ps_parameters,str_section):
        from py_utils import *
        from py_utils.results.results import *
        from py_utils.results import *
        from py_solvers import *
        from py_operators import *

        str_id = ps_parameters.get_section_dict(str_section)['name']
        if not SectionFactory.factories.has_key(str_id):
            SectionFactory.factories[str_id] = \
              eval(str_id + '.Factory()')
        return SectionFactory.factories[str_id]. \
               create(ps_parameters,str_section)
    create_section = staticmethod(create_section)