"""Numbered program headers must preserve physical line identities."""
import tempfile
import unittest
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from spindle_power.load_prediction.feed import MachineConfig, parse_nc

class NumberedHeaderTest(unittest.TestCase):
    def test_header_keeps_geometry_and_line_numbers(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'case.nc'
            body = '\nG90 G54 G0 X0 Y0 Z1\nG1 X10 F4000 S10000 M3\nM30\n'
            path.write_text('%1234' + body, encoding='utf-8')
            parsed = parse_nc(path, MachineConfig(), coordinate_mode='workpiece_cutting')
            self.assertEqual(parsed['issues'], [])
            self.assertEqual([(s.line, s.length) for s in parsed['segments']], [(3, 10.)])
            for header, content in [('%1234 G1 X20', body), ('%', body.replace('M30', '%1234\nM30'))]:
                path.write_text(header + content, encoding='utf-8')
                self.assertTrue(parse_nc(path, MachineConfig(), coordinate_mode='workpiece_cutting')['issues'])

if __name__ == '__main__':
    unittest.main()
