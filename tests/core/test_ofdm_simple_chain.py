import unittest
import numpy as np

from src.comnumpy.core import Sequential, Recorder
from src.comnumpy.core.generators import SymbolGenerator
from src.comnumpy.core.mappers import SymbolMapper, SymbolDemapper
from src.comnumpy.core.channels import AWGN, FIRChannel
from src.comnumpy.core.processors import Serial2Parallel, Parallel2Serial
from src.comnumpy.core.utils import get_alphabet
from src.comnumpy.core.metrics import compute_ser
from src.comnumpy.ofdm.processors import (
    CarrierAllocator, FFTProcessor, IFFTProcessor,
    CyclicPrefixer, CyclicPrefixRemover, CarrierExtractor
)
from src.comnumpy.ofdm.compensators import FrequencyDomainEqualizer
from src.comnumpy.ofdm.utils import get_standard_carrier_allocation
from src.comnumpy.ofdm.chains import OFDMTransmitter, OFDMReceiver


class TestOFDMSimpleChain(unittest.TestCase):

    def setUp(self):
        # Basic parameters
        self.M = 16
        self.N = 10000
        self.N_cp = 10
        self.sigma2 = 0

        # Alphabet and mapping
        self.alphabet = get_alphabet("QAM", self.M)

        # Carrier allocation
        self.carrier_type = get_standard_carrier_allocation("802.11ah_128")
        self.N_carrier_data = np.sum(self.carrier_type == 1)
        self.N_carrier_pilots = np.sum(self.carrier_type == 2)

        # Pilots and channel
        self.pilots = 10 * np.ones(self.N_carrier_pilots)
        self.h = np.array([1 + 0.1j, 0.2 - 0.1j, -0.1 + 0.5j])

    def build_manual_chain(self):
        return Sequential([
            SymbolGenerator(self.M),
            Recorder(name="data_tx"),
            SymbolMapper(self.alphabet),
            Serial2Parallel(self.N_carrier_data),
            CarrierAllocator(carrier_type=self.carrier_type, pilots=self.pilots),
            IFFTProcessor(),
            CyclicPrefixer(self.N_cp),
            Parallel2Serial(),
            FIRChannel(self.h),
            AWGN(value=self.sigma2),
            Serial2Parallel(len(self.carrier_type) + self.N_cp),
            CyclicPrefixRemover(self.N_cp),
            FFTProcessor(),
            FrequencyDomainEqualizer(h=self.h),
            CarrierExtractor(self.carrier_type),
            Parallel2Serial(),
            SymbolDemapper(self.alphabet)
        ])

    def build_modular_chain(self):
        return Sequential([
            SymbolGenerator(self.M),
            Recorder(name="data_tx"),
            SymbolMapper(self.alphabet),
            OFDMTransmitter(
                N_carrier_data=self.N_carrier_data,
                N_cp=self.N_cp,
                carrier_type=self.carrier_type,
                pilots=self.pilots
            ),
            FIRChannel(self.h),
            AWGN(value=self.sigma2),
            OFDMReceiver(
                N_carrier_data=self.N_carrier_data,
                N_cp=self.N_cp,
                carrier_type=self.carrier_type,
                h=self.h
            ),
            SymbolDemapper(self.alphabet)
        ])

    def test_manual_chain_ser_zero(self):
        """Test manual OFDM chain with SER=0 under perfect channel knowledge."""
        chain = self.build_manual_chain()
        y = chain(self.N)
        data_tx = chain["data_tx"].get_data()
        ser = compute_ser(data_tx, y)
        np.testing.assert_allclose(ser, 0, atol=1e-8)

    def test_modular_chain_ser_zero(self):
        """Test modular OFDM chain using OFDMTransmitter and OFDMReceiver."""
        chain = self.build_modular_chain()
        y = chain(self.N)
        data_tx = chain["data_tx"].get_data()
        ser = compute_ser(data_tx, y)
        np.testing.assert_allclose(ser, 0, atol=1e-8)


if __name__ == "__main__":
    unittest.main()
