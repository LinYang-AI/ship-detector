import unittest
import numpy as np
import tempfile
import shutil
from pathlib import Path
import sys
import os


# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from scripts.prepare_data import rle_decode, create_synthetic_test

class TestDataPipeline(unittest.TestCase):
    
    def setUp(self):
        """Create temporary directory for test outputs."""
        self.test_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up temporary files."""
        shutil.rmtree(self.test_dir)
    
    def test_rle_decode(self):
        """Test RLE decoding with empty mask."""
        mask = rle_decode('', (10, 10))
        self.assertEqual(mask.shape, (100,))
        self.assertEqual(mask.sum(), 0)
        
    def test_rle_decode_simple(self):
        """Test RLE decoding with simple mask."""
        mask = rle_decode('1 5', (3, 3))
        expected = np.array([
            [1, 1, 1],
            [1, 1, 0],
            [0, 0, 0]
        ], dtype=np.uint8).T
        
        np.testing.assert_array_equal(mask, expected)
        
    def test_rle_decode_multiple_runs(self):
        """Test RLE with multiple runs"""
        mask = rle_decode('1 3 7 3', (3, 3))
        self.assertEqual(mask.sum(), 6)
        
    def test_synthetic_data_creation(self):
        """Test synthetic data generation"""
        test_path = create_synthetic_test()
        
        # Check file were created
        test_dir = Path(test_path)
        self.assertTrue((test_dir / 'synthetic.tif').exists())
        self.assertTrue((test_dir / 'synthetic_masks.csv').exists())
        
        # Load and check image
        import cv2
        img = cv2.imread(str(test_dir / 'synthetic.tif'))
        self.assertEqual(img.shape, (512, 512, 3))
        
        # Check CSV
        import pandas as pd
        df = pd.read_csv(test_dir / 'synthetic_masks.csv')
        self.assertEqual(len(df), 1)
        self.assertIn('EncodedPixels', df.columns)
        
        # Clean up
        shutil.rmtree(test_path)
        
    def test_patch_overlap_calculation(self):
        """Test that patches overlap corectly."""
        patch_size = 224
        overlap = 32
        stride = patch_size - overlap
        
        # For a 512x512 image
        img_size = 512
        positions = list(range(0, img_size - patch_size + 1, stride))
        
        # Check we get expected number of patches
        # With stride=192, we should get positions at 0, 192 and possibly 384
        # 384 + 224 = 608 > 512, so only 0, 192, and 288 (512-224)
        self.assertTrue(len(positions) >= 2)
        
        # Check overlap between consecutive patches
        if len(positions) > 1:
            actual_overlap = patch_size - (positions[1] - positions[0])
            self.assertEqual(actual_overlap, overlap)


if __name__ == '__main__':
    unittest.main()