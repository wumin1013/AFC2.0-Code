from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from interval_analysis.interval_process_io import ProcessInfoFormatError, read_interval_process


class IntervalProcessIoTests(unittest.TestCase):
    def read_text(self, text, encoding="utf-8"):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "工艺.csv"
            path.write_text(text, encoding=encoding)
            return read_interval_process(path)

    def test_afomilling_preserves_nc_remainder_duplicates_and_source_rows(self):
        frame = self.read_text(
            "序号 N ap(mm) ae(mm) F(mm/min) MRR(mm3/min) s(mm) S(r/min) G\n"
            "1 12 2 3 120 999 0.2 5000 G1 X10 Y2 F120\n\n"
            "2 12 2 3 120 999 0.2 5000 G1 X10 Y2 F120\n"
        )
        self.assertEqual(frame.source_index.tolist(), [0, 1])
        self.assertEqual(frame.line_no_raw.tolist(), [11, 11])
        self.assertEqual(frame.N_str.tolist(), ["N12", "N12"])
        self.assertEqual(frame.source_file_line.tolist(), [2, 4])
        self.assertEqual(frame.gcode_content.tolist(), ["G1 X10 Y2 F120"] * 2)
        self.assertEqual(frame.MRR.tolist(), [12.0, 12.0])
        self.assertEqual(frame.source_mrr.tolist(), [999.0, 999.0])
        self.assertEqual(frame.attrs["s_semantics"], "source_value_unspecified")
        self.assertNotIn("time_s", frame.columns)
        self.assertNotIn("path_cumulative", frame.columns)

    def test_export_csv_quotes_and_state_are_preserved_as_provenance(self):
        frame = self.read_text(
            "N,S(r/min),ap(mm),ae(mm),F(mm/min),s(mm),MRR(mm3/s),G,state_code\n"
            'N3,5000,1,2,300,0.5,100,"G1 X1 (a,b)",2\n'
        )
        self.assertEqual(frame.loc[0, "line_no_raw"], 2)
        self.assertEqual(frame.loc[0, "gcode_content"], "G1 X1 (a,b)")
        self.assertEqual(frame.loc[0, "MRR"], 10.0)
        self.assertEqual(frame.loc[0, "source_state_code"], 2)
        self.assertNotIn("state_code", frame.columns)

    def test_with_without_sequence_and_supported_encodings(self):
        for prefix in ("", "序号,"):
            for encoding in ("utf-8", "utf-8-sig", "utf-16", "gb18030"):
                with self.subTest(prefix=prefix, encoding=encoding):
                    frame = self.read_text(
                        prefix + "N,S(r/min),ap(mm),ae(mm),F(mm/min),s(mm),MRR(mm3/s),G\n"
                        + ("1," if prefix else "") + "N1,0,0,0,0,0,0,M5\n", encoding)
                    self.assertEqual(frame.loc[0, "line_no_raw"], 0)
        frame = self.read_text(
            "N ap(mm) ae(mm) F(mm/min) MRR(mm3/min) s(mm) S(r/min) G\n"
            "4 1 2 60 120 0.1 1000 G1 X1\n")
        self.assertEqual(frame.loc[0, "MRR"], 2.0)

    def test_rejects_invalid_values_missing_fields_and_unknown_layout(self):
        header = "N,S(r/min),ap(mm),ae(mm),F(mm/min),s(mm),MRR(mm3/s),G\n"
        for row in (
            "N0,1000,1,2,60,0.1,2,G1 X1", "N1.5,1000,1,2,60,0.1,2,G1 X1",
            "N1,NaN,1,2,60,0.1,2,G1 X1", "N1,1000,-1,2,60,0.1,2,G1 X1",
            "N1,1000,1,2,60,-0.1,2,G1 X1", "N1,1000,1,2,60,0.1,inf,G1 X1",
            "N1,1000,1,2,60,0.1,G1 X1", "N1,1000,1e308,1e308,60,0.1,2,G1 X1",
        ):
            with self.subTest(row=row), self.assertRaises(ProcessInfoFormatError):
                self.read_text(header + row)
        for text in ("", header, "N,ap,ae,G\n1,1,2,G1", header + "N1,1000,1,2,60,0.1,2,\"bad"):
            with self.subTest(text=text), self.assertRaises(ProcessInfoFormatError):
                self.read_text(text)

    def test_import_does_not_load_gui_or_model_runtime(self):
        code = (
            "import sys;sys.path.insert(0,sys.argv[1]);"
            "import interval_analysis.interval_process_io;"
            "assert not any(x in sys.modules for x in "
            "('tkinter','matplotlib','torch','interval_analysis.shared','spindle_power.load_prediction'))"
        )
        result = subprocess.run([sys.executable, "-c", code, str(SRC_ROOT)],
                                capture_output=True, text=True, check=False)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
