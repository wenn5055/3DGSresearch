"""Check the predeclared size-choice rule, independently of actual results."""

import unittest

from finalize_sphere_study import choose_hdr


class SelectionTests(unittest.TestCase):
    def test_smaller_size_requires_all_three_criteria(self):
        rows = {'HDR-1.0': {'lpips_alex': .1, 'foreground_psnr_db': 18., 'foreground_coverage_fraction': 1.},
                'HDR-0.75': {'lpips_alex': .09, 'foreground_psnr_db': 18.2, 'foreground_coverage_fraction': .999}}
        self.assertEqual(choose_hdr(rows)[0], 'HDR-0.75')
        for key, bad in [('lpips_alex', .11), ('foreground_psnr_db', 17.9), ('foreground_coverage_fraction', .98)]:
            original = rows['HDR-0.75'][key]
            rows['HDR-0.75'][key] = bad
            self.assertEqual(choose_hdr(rows)[0], 'HDR-1.0', key)
            rows['HDR-0.75'][key] = original


if __name__ == '__main__':
    unittest.main()
