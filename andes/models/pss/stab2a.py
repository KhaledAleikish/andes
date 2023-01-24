import logging

from andes.core import NumParam, IdxParam, ExtService, Algeb, Limiter
from andes.core.block import Gain, WashoutOrLag, Lag
from andes.models.pss.pssbase import PSSBaseData, PSSBase


logger = logging.getLogger(__name__)


class STAB2AData(PSSBaseData):
    def __init__(self):
        super().__init__()

        self.T2 = NumParam(default=1, tex_name='T_2', vrange=(0, 10), info='washout time const.')
        self.T3 = NumParam(default=1, tex_name='T_3', vrange=(0, 10), info='2nd stage low-pass filter time const.')
        self.T5 = NumParam(default=1, tex_name='T_5', vrange=(0, 10), info='3rd stage low-pass filter time const.')

        self.K2 = NumParam(default=1, tex_name='K_2', info='Gain before washout')
        self.K3 = NumParam(default=1, tex_name='K_3', info='2nd stage low-pass filter gain')
        self.K4 = NumParam(default=1, tex_name='K_4', info='2nd stage gain')
        self.K5 = NumParam(default=1, tex_name='K_5', info='3rd stage low-pass filter gain')

        self.HLIM_MAX = NumParam(default=0.3, tex_name='HLIM_{MAX}', vrange=(0, 0.3), info='Max. output limit')
        self.HLIM_MIN = NumParam(default=-0.3, tex_name='HLIM_{MIN}', vrange=(-0.3, 0), info='Min. output limit')
        self.HLIM_MIN.vin = -1* self.HLIM_MAX.v

        # Not used:
        self.VCU = NumParam(default=999, tex_name='V_{CU}', vrange=(1, 1.2),
                            unit='p.u.', info='Upper enabling bus voltage')
        self.VCL = NumParam(default=-999, tex_name='V_{CL}', vrange=(0., 1),
                            unit='p.u.', info='Upper enabling bus voltage')
        self.busr = IdxParam(info='Optional remote bus idx', model='Bus', default=None)
        self.busf = IdxParam(info='BusFreq idx for mode 2', model='BusFreq', default=None)

class STAB2AModel(PSSBase):
    """
    STAB2A Stabilizer equation.
    """

    def __init__(self, system, config):
        PSSBase.__init__(self, system, config)

        self.SnSb = ExtService(model='SynGen', src='M', indexer=self.syn, attr='pu_coeff',
                               info='Machine base to sys base factor for power',
                               tex_name='(Sb/Sn)')

        self.sig = Algeb(tex_name='Sig',
                         info='Input signal',
                         )

        self.sig.v_str = 'tm0/SnSb'

        self.sig.e_str = 'te/SnSb - sig'

        self.PK2 = Gain(u=self.sig, K=self.K2)

        self.WO = WashoutOrLag(u=self.PK2_y, T=self.T2, K=self.T2, name='WO', zero_out=False)

        self.V1 = Algeb(tex_name='V_1', info='Washout filter output^3',
                         e_str='WO_y**3 - V1')

        self.PK4 = Gain(u=self.V1, K=self.K4)

        self.V2 = Lag(u=self.V1, T=self.T3, K=self.K3)

        self.V3 = Algeb(tex_name='V_3', info='V2 + K4*V1',
                         e_str='PK4_y + V2_y - V3')

        self.L1 = Lag(u=self.V3, T=self.T5, K=self.K5)

        self.V4 = Algeb(tex_name='V_4', info='Lag output^2',
                         e_str='L1_y**2 - V4')

        self.HLIM = Limiter(u=self.V4, lower=self.HLIM_MIN, upper=self.HLIM_MAX, info='Output limiter')

        self.vsout.e_str = 'HLIM_zi * V4 + HLIM_zu * HLIM_MAX + HLIM_zl * HLIM_MIN - vsout'



class STAB2A(STAB2AData, STAB2AModel):
    """
    STAB2A stabilizer model.

    Input signal: Generator P electrical in Gen MVABase (p.u.)
    """

    def __init__(self, system, config):
        STAB2AData.__init__(self)
        STAB2AModel.__init__(self, system, config)
