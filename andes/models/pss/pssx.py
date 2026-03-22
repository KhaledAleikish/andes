"""
Minimal external-command PSS model.

This model provides an explicit interface for applying a stabilizing signal
from an external controller without internal lead-lag or washout dynamics.
"""

from andes.core import NumParam, IdxParam
from andes.models.pss.pssbase import PSSBaseData, PSSBase


class PSSXData(PSSBaseData):
    """Data container for PSSX."""

    def __init__(self):
        super().__init__()
        # Keep base compatibility fields used by PSSBase.
        self.busr = IdxParam(info='Optional remote bus idx', model='Bus', default=None)
        self.busf = IdxParam(info='Optional BusFreq idx', model='BusFreq', default=None)
        self.VCU = NumParam(default=999.0, info='Upper enabling bus voltage')
        self.VCL = NumParam(default=-999.0, info='Lower enabling bus voltage')
        # External stabilizing command (p.u.) to be provided online.
        self.VEXT = NumParam(default=0.0, info='External stabilizing command')


class PSSXModel(PSSBase):
    """PSSX dynamics: vsout follows the external command when enabled."""

    def __init__(self, system, config):
        super().__init__(system, config)
        self.vsout.e_str = 'ue * VEXT - vsout'


class PSSX(PSSXData, PSSXModel):
    """
    External-command PSS model.

    The output relation is:
        vsout = ue * VEXT
    where `ue` is the effective PSS online status from `PSSBase`.
    """

    def __init__(self, system, config):
        PSSXData.__init__(self)
        PSSXModel.__init__(self, system, config)

